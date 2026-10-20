"""email_images: 메일 이미지를 Pi에서 내려받아 Claude base64 이미지 블록으로 만드는지 확인한다.

네트워크·DNS만 가짜로 바꾸고, 이미지 판별·축소·재인코딩은 실제 Pillow로 한다.
실행: snu_mail 폴더에서 `python -m unittest discover -s tests -t .`
"""
import base64
import contextlib
import io
import ipaddress
import os
import unittest
from unittest import mock

import requests
from PIL import Image

import email_images

PUBLIC_IP = '147.46.10.10'


def make_image(size=(10, 10), fmt='PNG', noise=False):
    img = Image.frombytes('RGB', size, os.urandom(size[0] * size[1] * 3)) if noise else Image.new('RGB', size, 'white')
    out = io.BytesIO()
    if fmt == 'MPO':
        img.save(out, format='MPO', save_all=True, append_images=[img])
    else:
        img.save(out, format=fmt)
    return out.getvalue()


class FakeResponse:
    def __init__(self, content=b'', status=200, location=None):
        self.content = content
        self.status = status
        self.headers = {'location': location} if location else {}
        self.is_redirect = location is not None
        self.read_bytes = 0

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def raise_for_status(self):
        if self.status >= 400:
            raise requests.HTTPError(f'{self.status}')

    def iter_content(self, chunk_size):
        for i in range(0, len(self.content), chunk_size):
            self.read_bytes += chunk_size
            yield self.content[i:i + chunk_size]


def fake_get(pages):
    """pages: url -> bytes, FakeResponse, 또는 예외 인스턴스. 요청한 url은 requested에 쌓인다."""
    requested = []

    def get(url, **kwargs):
        requested.append(url)
        value = pages[url]
        if isinstance(value, BaseException):
            raise value
        return value if isinstance(value, FakeResponse) else FakeResponse(value)

    get.requested = requested
    return get


@contextlib.contextmanager
def fake_network(pages, dns=None):
    """requests.get과 DNS를 바꾼다. dns에 없는 호스트는 공개 주소로 풀린다. 가짜 get을 돌려준다."""
    dns = dns or {}

    def getaddrinfo(host, port, *args, **kwargs):
        try:
            ip = str(ipaddress.ip_address(host))  # IP를 그대로 적은 주소는 그 IP로 풀린다
        except ValueError:
            ip = dns.get(host, PUBLIC_IP)
        return [(None, None, None, '', (ip, port))]

    get = fake_get(pages)
    with mock.patch.object(email_images.requests, 'get', get), \
            mock.patch.object(email_images.socket, 'getaddrinfo', getaddrinfo):
        yield get


def decoded(block):
    return base64.b64decode(block['source']['data'])


def opened(block):
    return Image.open(io.BytesIO(decoded(block)))


class ImageBlockTest(unittest.TestCase):
    def test_downloaded_png_becomes_base64_block(self):
        png = make_image()
        with fake_network({'https://a/x.png': png}):
            block = email_images.image_block('https://a/x.png')
        self.assertEqual(block['type'], 'image')
        self.assertEqual(block['source']['type'], 'base64')
        self.assertEqual(block['source']['media_type'], 'image/png')
        self.assertEqual(decoded(block), png)

    def test_media_type_comes_from_content_not_extension(self):
        jpeg = make_image(fmt='JPEG')
        with fake_network({'https://a/x.png': jpeg}):
            block = email_images.image_block('https://a/x.png')
        self.assertEqual(block['source']['media_type'], 'image/jpeg')
        self.assertEqual(decoded(block), jpeg)

    def test_mpo_photo_is_sent_as_jpeg(self):
        with fake_network({'https://a/photo.jpg': make_image(fmt='MPO')}):
            block = email_images.image_block('https://a/photo.jpg')
        self.assertEqual(block['source']['media_type'], 'image/jpeg')
        self.assertEqual(opened(block).format, 'JPEG')

    def test_tall_png_is_shrunk_to_8000(self):
        with fake_network({'https://a/tall.png': make_image(size=(100, 8100))}):
            block = email_images.image_block('https://a/tall.png')
        self.assertEqual(block['source']['media_type'], 'image/png')
        self.assertEqual(opened(block).size[1], 8000)

    def test_tall_jpeg_is_shrunk_and_stays_jpeg(self):
        with fake_network({'https://a/tall.jpg': make_image(size=(100, 8100), fmt='JPEG')}):
            block = email_images.image_block('https://a/tall.jpg')
        self.assertEqual(block['source']['media_type'], 'image/jpeg')
        self.assertEqual(opened(block).size[1], 8000)

    def test_png_over_size_limit_is_reencoded_as_smaller_jpeg(self):
        png = make_image(size=(200, 200), fmt='PNG', noise=True)
        limit = len(base64.b64encode(png)) - 1
        with fake_network({'https://a/big.png': png}), mock.patch.object(email_images, 'MAX_IMAGE_BASE64_BYTES', limit):
            block = email_images.image_block('https://a/big.png')
        self.assertEqual(block['source']['media_type'], 'image/jpeg')
        self.assertLessEqual(len(block['source']['data']), limit)
        self.assertEqual(opened(block).size, (200, 200))

    def test_image_still_over_size_limit_is_skipped(self):
        with fake_network({'https://a/x.png': make_image()}), mock.patch.object(email_images, 'MAX_IMAGE_BASE64_BYTES', 10):
            self.assertIsNone(email_images.image_block('https://a/x.png'))

    def test_download_failures_return_none(self):
        pages = {'https://a/down.png': requests.ConnectionError('down'),
                 'https://a/404.png': FakeResponse(b'not found', status=404),
                 'https://a/html.png': b'<html>robots</html>',
                 'https://a/truncated.png': make_image(size=(100, 100), noise=True)[:2000]}
        with fake_network(pages):
            for url in pages:
                self.assertIsNone(email_images.image_block(url), url)

    def test_private_addresses_are_never_requested(self):
        urls = ['http://router/cgi?act=x&.png', 'http://127.0.0.1/a.png', 'http://[::1]/a.png', 'file:///etc/a.png']
        with fake_network({}, dns={'router': '192.168.0.1'}) as get:
            for url in urls:
                self.assertIsNone(email_images.image_block(url), url)
        self.assertEqual(get.requested, [])

    def test_redirect_into_private_network_is_not_followed(self):
        pages = {'https://a/x.png': FakeResponse(b'x' * 100_000, status=302, location='http://intranet/x.png')}
        with fake_network(pages, dns={'intranet': '10.0.0.5'}) as get:
            self.assertIsNone(email_images.image_block('https://a/x.png'))
        self.assertEqual(get.requested, ['https://a/x.png'])
        self.assertEqual(pages['https://a/x.png'].read_bytes, 0)

    def test_redirect_to_public_host_is_followed(self):
        png = make_image()
        pages = {'https://a/x.png': FakeResponse(status=302, location='/real.png'), 'https://a/real.png': png}
        with fake_network(pages):
            block = email_images.image_block('https://a/x.png')
        self.assertEqual(decoded(block), png)

    def test_download_stops_reading_past_byte_cap(self):
        response = FakeResponse(b'\0' * 1_000_000)
        with fake_network({'https://a/huge.png': response}), mock.patch.object(email_images, 'MAX_DOWNLOAD_BYTES', 100_000):
            self.assertIsNone(email_images.image_block('https://a/huge.png'))
        self.assertLess(response.read_bytes, 1_000_000)

    def test_download_stops_past_time_limit(self):
        response = FakeResponse(make_image(size=(300, 300), noise=True))
        with fake_network({'https://a/slow.png': response}), mock.patch.object(email_images, 'DOWNLOAD_DEADLINE', -1):
            self.assertIsNone(email_images.image_block('https://a/slow.png'))
        self.assertLess(response.read_bytes, len(response.content))

    def test_image_over_pixel_limit_is_skipped(self):
        with fake_network({'https://a/x.png': make_image(size=(10, 10))}), \
                mock.patch.object(email_images, 'MAX_DECODE_PIXELS', 99):
            self.assertIsNone(email_images.image_block('https://a/x.png'))


class ImageBlocksTest(unittest.TestCase):
    def test_failed_images_are_dropped_and_order_kept(self):
        first, second = make_image(size=(10, 10)), make_image(size=(20, 20))
        pages = {'https://a/1.png': first, 'https://a/bad.png': requests.Timeout('slow'), 'https://a/2.png': second}
        with fake_network(pages):
            blocks = email_images.image_blocks(['https://a/1.png', 'https://a/bad.png', 'https://a/2.png'])
        self.assertEqual([decoded(b) for b in blocks], [first, second])

    def test_duplicate_urls_are_fetched_once(self):
        png = make_image()
        with fake_network({'https://a/1.png': png}) as get:
            blocks = email_images.image_blocks(['https://a/1.png', 'https://a/1.png'])
        self.assertEqual(len(blocks), 1)
        self.assertEqual(get.requested, ['https://a/1.png'])

    def test_stops_at_max_images_counting_only_successes(self):
        png = make_image()
        pages = {'https://a/bad.png': requests.ConnectionError('down'), 'https://a/1.png': png, 'https://a/2.png': png}
        with fake_network(pages) as get:
            blocks = email_images.image_blocks(['https://a/bad.png', 'https://a/1.png', 'https://a/2.png'], max_images=1)
        self.assertEqual(len(blocks), 1)
        self.assertEqual(get.requested, ['https://a/bad.png', 'https://a/1.png'])

    def test_stops_downloading_once_total_budget_is_exceeded(self):
        first, second = make_image(size=(10, 10)), make_image(size=(20, 20))
        pages = {'https://a/1.png': first, 'https://a/2.png': second, 'https://a/3.png': first}
        budget = len(base64.b64encode(first)) + 1
        with fake_network(pages) as get:
            blocks = email_images.image_blocks(['https://a/1.png', 'https://a/2.png', 'https://a/3.png'],
                                               max_total_bytes=budget)
        self.assertEqual([decoded(b) for b in blocks], [first])
        self.assertEqual(get.requested, ['https://a/1.png', 'https://a/2.png'])

    def test_no_new_downloads_after_overall_time_limit(self):
        with fake_network({'https://a/1.png': make_image()}) as get, \
                mock.patch.object(email_images, 'IMAGES_DEADLINE', -1):
            self.assertEqual(email_images.image_blocks(['https://a/1.png']), [])
        self.assertEqual(get.requested, [])


if __name__ == '__main__':
    unittest.main()
