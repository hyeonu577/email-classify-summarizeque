"""메일 본문 이미지를 Pi에서 직접 내려받아 Claude base64 이미지 블록으로 만든다.

SNU 대량메일 이미지 서버(letter2.snu.ac.kr)는 robots.txt로 Claude의 URL 가져오기를 막는다.
그래서 URL을 넘기지 않고 이미지 내용을 직접 싣는다. 실패한 이미지는 건너뛰고 나머지로 요약한다.
주소는 메일 보낸 사람이 정하므로 공개 인터넷 주소만 받는다(사설망·localhost 차단).
"""
import base64
import io
import ipaddress
import logging
import socket
import time
from urllib.parse import urljoin, urlsplit

import requests
from PIL import Image

DOWNLOAD_TIMEOUT = 10  # 연결·읽기 한 번의 대기 상한
DOWNLOAD_DEADLINE = 30  # 한 장을 받는 전체 시간 상한. 조금씩 흘려보내는 서버에 묶이지 않게 한다
IMAGES_DEADLINE = 60  # 이 시간이 지나면 남은 이미지는 받지 않는다
MAX_REDIRECTS = 3
MAX_DOWNLOAD_BYTES = 20 * 1024 * 1024  # 이보다 큰 파일은 끝까지 받지 않는다
MAX_DECODE_PIXELS = 40_000_000  # Pi 메모리(1.8GB)로 디코딩할 수 있는 픽셀 수 상한
MAX_IMAGE_SIDE = 8000  # Claude는 가로·세로 중 하나라도 8000px를 넘으면 요청 전체를 400으로 거부한다
MAX_IMAGE_BASE64_BYTES = 10 * 1024 * 1024  # Claude 이미지 한 장 상한(base64 길이 기준, 넘으면 400)
MAX_REQUEST_IMAGES = 20  # 20장을 넘으면 Claude가 모든 이미지에 2000px 상한을 적용한다
MAX_REQUEST_IMAGE_BYTES = 20 * 1024 * 1024  # 요청 하나에 싣는 base64 합계. Messages API 요청 상한은 32MB
_PASS_THROUGH_TYPES = {'PNG': 'image/png', 'JPEG': 'image/jpeg', 'GIF': 'image/gif', 'WEBP': 'image/webp'}
_JPEG_SOURCES = ('JPEG', 'MPO')  # MPO: 휴대폰 사진(여러 장이 든 JPEG)


def _check_public_host(url):
    parts = urlsplit(url)
    if parts.scheme not in ('http', 'https') or not parts.hostname:
        raise ValueError('http(s) 주소가 아님')
    for *_, sockaddr in socket.getaddrinfo(parts.hostname, parts.port or 443):
        if not ipaddress.ip_address(sockaddr[0]).is_global:
            raise ValueError(f'공개 주소가 아님: {sockaddr[0]}')


def _download(url):
    deadline = time.monotonic() + DOWNLOAD_DEADLINE
    for _ in range(MAX_REDIRECTS + 1):
        _check_public_host(url)
        # 리다이렉트는 직접 따라간다. requests가 따라가면 다음 주소를 검사할 수 없고 리다이렉트 본문을 상한 없이 읽는다
        with requests.get(url, timeout=DOWNLOAD_TIMEOUT, stream=True, allow_redirects=False) as response:
            if response.is_redirect:
                url = urljoin(url, response.headers['location'])
                continue
            response.raise_for_status()
            data = bytearray()
            for chunk in response.iter_content(64 * 1024):
                data += chunk
                if len(data) > MAX_DOWNLOAD_BYTES:
                    raise ValueError(f'{MAX_DOWNLOAD_BYTES} bytes 초과')
                if time.monotonic() > deadline:
                    raise TimeoutError(f'{DOWNLOAD_DEADLINE}초 안에 받지 못함')
            return bytes(data)
    raise ValueError(f'리다이렉트 {MAX_REDIRECTS}번 초과')


def _base64_len(data):
    return (len(data) + 2) // 3 * 4


def _encode(img, fmt):
    out = io.BytesIO()
    if fmt == 'JPEG':
        img.convert('RGB').save(out, format='JPEG', quality=90)
    else:
        img.save(out, format='PNG')
    return out.getvalue()


def _to_claude_image(data):  # (bytes, media_type). Claude가 받는 형식·크기로 맞춘다
    with Image.open(io.BytesIO(data)) as img:
        width, height = img.size
        if width * height > MAX_DECODE_PIXELS:
            raise ValueError(f'{width}x{height}px: 디코딩 상한 초과')
        img.load()  # 깨진 파일은 여기서 거른다. 그대로 보내면 Claude가 요청 전체를 거부한다
        source_format = img.format
        media_type = _PASS_THROUGH_TYPES.get(source_format)
        if media_type and max(width, height) <= MAX_IMAGE_SIDE and _base64_len(data) <= MAX_IMAGE_BASE64_BYTES:
            return data, media_type
        img.thumbnail((MAX_IMAGE_SIDE, MAX_IMAGE_SIDE))
        if source_format not in _JPEG_SOURCES:
            if img.mode not in ('1', 'L', 'LA', 'P', 'RGB', 'RGBA'):
                img = img.convert('RGBA')
            data = _encode(img, 'PNG')
            if _base64_len(data) <= MAX_IMAGE_BASE64_BYTES:
                return data, 'image/png'
        return _encode(img, 'JPEG'), 'image/jpeg'  # 사진·너무 큰 PNG는 JPEG가 훨씬 작다


def image_block(url):
    try:
        data, media_type = _to_claude_image(_download(url))
    except Exception as e:
        logging.warning(f'이미지 건너뜀 ({e}): {url}')
        return None
    if _base64_len(data) > MAX_IMAGE_BASE64_BYTES:
        logging.warning(f'이미지 건너뜀 (base64 {_base64_len(data)} bytes > {MAX_IMAGE_BASE64_BYTES}): {url}')
        return None
    return {'type': 'image',
            'source': {'type': 'base64', 'media_type': media_type, 'data': base64.standard_b64encode(data).decode()}}


def image_blocks(urls, max_images=MAX_REQUEST_IMAGES, max_total_bytes=MAX_REQUEST_IMAGE_BYTES):
    """urls를 순서대로 받아 성공한 것만 max_images장까지 담는다. 합계·시간 상한을 넘으면 나머지는 받지 않는다."""
    blocks = []
    total = 0
    deadline = time.monotonic() + IMAGES_DEADLINE
    for url in dict.fromkeys(urls):  # 같은 이미지를 두 번 받지 않는다
        if len(blocks) >= max_images:
            break
        if time.monotonic() > deadline:
            logging.warning(f'이미지 건너뜀 ({IMAGES_DEADLINE}초 초과, 남은 이미지 생략): {url}')
            break
        block = image_block(url)
        if block is None:
            continue
        total += len(block['source']['data'])
        if total > max_total_bytes:
            logging.warning(f'이미지 건너뜀 (요청당 이미지 합계 {max_total_bytes} bytes 초과, 남은 이미지 생략): {url}')
            break
        blocks.append(block)
    return blocks
