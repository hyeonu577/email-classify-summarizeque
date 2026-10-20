"""메일 본문 이미지를 Pi에서 직접 내려받아 Claude base64 이미지 블록으로 만든다.

SNU 대량메일 이미지 서버(letter2.snu.ac.kr)는 robots.txt로 Claude의 URL 가져오기를 막는다.
그래서 URL을 넘기지 않고 이미지 내용을 직접 싣는다. 실패한 이미지는 건너뛰고 나머지로 요약한다.
"""
import base64
import io
import logging

import requests
from PIL import Image

DOWNLOAD_TIMEOUT = 10
MAX_DOWNLOAD_BYTES = 20 * 1024 * 1024  # 이보다 큰 파일은 끝까지 받지 않는다
MAX_IMAGE_BYTES = 5 * 1024 * 1024  # Claude 이미지 한 장 상한
MAX_IMAGE_SIDE = 8000  # Claude는 가로·세로 중 하나라도 8000px를 넘으면 요청 전체를 400으로 거부한다
MAX_REQUEST_IMAGE_BYTES = 20 * 1024 * 1024  # 요청 하나에 싣는 base64 합계. Messages API 요청 상한은 32MB
_MEDIA_TYPES = {'PNG': 'image/png', 'JPEG': 'image/jpeg', 'GIF': 'image/gif', 'WEBP': 'image/webp'}


def _download(url):
    with requests.get(url, timeout=DOWNLOAD_TIMEOUT, stream=True) as response:
        response.raise_for_status()
        data = bytearray()
        for chunk in response.iter_content(64 * 1024):
            data += chunk
            if len(data) > MAX_DOWNLOAD_BYTES:
                raise ValueError(f'{MAX_DOWNLOAD_BYTES} bytes 초과')
    return bytes(data)


def _to_supported_image(data):  # (bytes, media_type). 지원하지 않는 형식이거나 너무 크면 PNG로 다시 저장한다
    with Image.open(io.BytesIO(data)) as img:
        media_type = _MEDIA_TYPES.get(img.format)
        if media_type is not None and max(img.size) <= MAX_IMAGE_SIDE:
            return data, media_type
        img.thumbnail((MAX_IMAGE_SIDE, MAX_IMAGE_SIDE))
        if img.mode not in ('RGB', 'RGBA', 'L', 'LA', 'P'):
            img = img.convert('RGB')
        out = io.BytesIO()
        img.save(out, format='PNG')
        return out.getvalue(), 'image/png'


def image_block(url):
    try:
        data, media_type = _to_supported_image(_download(url))
    except Exception as e:
        logging.warning(f'이미지 건너뜀 ({e}): {url}')
        return None
    if len(data) > MAX_IMAGE_BYTES:
        logging.warning(f'이미지 건너뜀 ({len(data)} bytes > {MAX_IMAGE_BYTES}): {url}')
        return None
    return {'type': 'image',
            'source': {'type': 'base64', 'media_type': media_type, 'data': base64.standard_b64encode(data).decode()}}


def image_blocks(urls, max_total_bytes=MAX_REQUEST_IMAGE_BYTES):
    blocks = []
    total = 0
    for url in urls:
        block = image_block(url)
        if block is None:
            continue
        size = len(block['source']['data'])
        if total + size > max_total_bytes:
            logging.warning(f'이미지 건너뜀 (요청당 이미지 합계 {max_total_bytes} bytes 초과): {url}')
            continue
        total += size
        blocks.append(block)
    return blocks
