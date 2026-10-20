"""email_images: 메일 이미지를 Pi에서 내려받아 Claude base64 이미지 블록으로 만드는지 확인한다.

네트워크만 가짜로 바꾸고, 이미지 판별·축소는 실제 Pillow로 한다.
실행: snu_mail 폴더에서 `python -m unittest discover -s tests -t .`
"""
import base64
import io
import unittest
from unittest import mock

import requests
from PIL import Image

import email_images


def make_image(size=(10, 10), fmt='PNG'):
    out = io.BytesIO()
    Image.new('RGB', size, 'white').save(out, format=fmt)
    return out.getvalue()


class FakeResponse:
    def __init__(self, content, status=200):
        self.content = content
        self.status = status
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


def decoded(block):
    return base64.b64decode(block['source']['data'])


class ImageBlockTest(unittest.TestCase):
    def test_downloaded_png_becomes_base64_block(self):
        png = make_image()
        with mock.patch.object(email_images.requests, 'get', fake_get({'https://a/x.png': png})):
            block = email_images.image_block('https://a/x.png')
        self.assertEqual(block['type'], 'image')
        self.assertEqual(block['source']['type'], 'base64')
        self.assertEqual(block['source']['media_type'], 'image/png')
        self.assertEqual(decoded(block), png)

    def test_media_type_comes_from_content_not_extension(self):
        jpeg = make_image(fmt='JPEG')
        with mock.patch.object(email_images.requests, 'get', fake_get({'https://a/x.png': jpeg})):
            block = email_images.image_block('https://a/x.png')
        self.assertEqual(block['source']['media_type'], 'image/jpeg')
        self.assertEqual(decoded(block), jpeg)

    def test_side_over_8000px_is_shrunk_to_8000(self):
        tall = make_image(size=(100, 8100))
        with mock.patch.object(email_images.requests, 'get', fake_get({'https://a/tall.png': tall})):
            block = email_images.image_block('https://a/tall.png')
        with Image.open(io.BytesIO(decoded(block))) as img:
            self.assertEqual(img.size[1], 8000)
            self.assertLessEqual(img.size[0], 100)

    def test_download_failures_return_none(self):
        pages = {'https://a/down.png': requests.ConnectionError('down'),
                 'https://a/404.png': FakeResponse(b'not found', status=404),
                 'https://a/html.png': b'<html>robots</html>'}
        with mock.patch.object(email_images.requests, 'get', fake_get(pages)):
            for url in pages:
                self.assertIsNone(email_images.image_block(url), url)

    def test_image_over_size_limit_is_skipped(self):
        png = make_image()
        with mock.patch.object(email_images.requests, 'get', fake_get({'https://a/x.png': png})), \
                mock.patch.object(email_images, 'MAX_IMAGE_BYTES', len(png) - 1):
            self.assertIsNone(email_images.image_block('https://a/x.png'))

    def test_download_stops_reading_past_cap(self):
        response = FakeResponse(b'\0' * 1_000_000)
        with mock.patch.object(email_images.requests, 'get', fake_get({'https://a/huge.png': response})), \
                mock.patch.object(email_images, 'MAX_DOWNLOAD_BYTES', 100_000):
            self.assertIsNone(email_images.image_block('https://a/huge.png'))
        self.assertLess(response.read_bytes, 1_000_000)


class ImageBlocksTest(unittest.TestCase):
    def test_failed_images_are_dropped_and_order_kept(self):
        first, second = make_image(size=(10, 10)), make_image(size=(20, 20))
        pages = {'https://a/1.png': first, 'https://a/bad.png': requests.Timeout('slow'), 'https://a/2.png': second}
        with mock.patch.object(email_images.requests, 'get', fake_get(pages)):
            blocks = email_images.image_blocks(['https://a/1.png', 'https://a/bad.png', 'https://a/2.png'])
        self.assertEqual([decoded(b) for b in blocks], [first, second])

    def test_images_past_total_budget_are_dropped(self):
        first, second = make_image(size=(10, 10)), make_image(size=(20, 20))
        pages = {'https://a/1.png': first, 'https://a/2.png': second}
        budget = len(base64.b64encode(first)) + 1
        with mock.patch.object(email_images.requests, 'get', fake_get(pages)):
            blocks = email_images.image_blocks(['https://a/1.png', 'https://a/2.png'], max_total_bytes=budget)
        self.assertEqual([decoded(b) for b in blocks], [first])


if __name__ == '__main__':
    unittest.main()
