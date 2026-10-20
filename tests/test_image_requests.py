"""요약 요청(실시간 Sonnet, 휴지통 Haiku 배치)이 이미지를 URL이 아니라 base64로 싣는지 확인한다.

letter2.snu.ac.kr은 robots.txt로 Claude의 URL 가져오기를 막는다. Claude·네트워크·파일 쓰기는 가짜로 바꾼다.
main.py는 gitignore된 constants.json 등을 읽으므로 이 Pi(운영 폴더나 링크를 건 worktree)에서만 돈다.
"""
import base64
import unittest
from types import SimpleNamespace as NS
from unittest import mock

import email_images
import main
import trash_mail_summarize
from tests.test_email_images import fake_get, make_image

LETTER = 'https://letter2.snu.ac.kr/upload/massmail/'


class FakeClaude:
    def __init__(self):
        self.calls = []
        self.beta = NS(messages=NS(create=self.create))

    def create(self, **kwargs):
        self.calls.append(kwargs)
        return NS(stop_reason='end_turn', content=[NS(type='text', text='1. 요약입니다.')])


def image_sources(content):
    if isinstance(content, str):
        return []
    return [block['source'] for block in content if block['type'] == 'image']


class RealtimeSummaryTest(unittest.TestCase):
    def test_summarize_email_sends_downloaded_images_as_base64(self):
        first, second = make_image(size=(10, 10)), make_image(size=(20, 20), fmt='JPEG')
        claude = FakeClaude()
        pages = {LETTER + 'a.png': first, LETTER + 'b.jpg': second}
        with mock.patch.object(email_images.requests, 'get', fake_get(pages)), \
                mock.patch.object(main, '_get_anthropic_client', return_value=claude):
            summary = main.summarize_email('제목', '본문', img_urls=[LETTER + 'a.png', LETTER + 'b.jpg'])
        self.assertEqual(summary, '1. 요약입니다.')
        sources = image_sources(claude.calls[0]['messages'][0]['content'])
        self.assertEqual([(s['type'], s['media_type']) for s in sources],
                         [('base64', 'image/png'), ('base64', 'image/jpeg')])
        self.assertEqual([base64.b64decode(s['data']) for s in sources], [first, second])


def trash_mail(hash_, *image_names):
    body = '안내드립니다.\n' + ''.join(f'![]({LETTER}{name})\n' for name in image_names)
    return {'subject': '제목', 'body': body, 'hash': hash_}


class TrashBatchTest(unittest.TestCase):
    def test_batch_request_carries_first_image_as_base64(self):
        png = make_image()
        get = fake_get({LETTER + 'a.png': png, LETTER + 'b.png': png})
        with mock.patch.object(email_images.requests, 'get', get):
            request = trash_mail_summarize.generate_each_line_of_batch_file(trash_mail('h1', 'a.png', 'b.png'))
        sources = image_sources(request['params']['messages'][0]['content'])
        self.assertEqual([(s['type'], s['media_type']) for s in sources], [('base64', 'image/png')])
        self.assertEqual(base64.b64decode(sources[0]['data']), png)
        self.assertEqual(get.requested, [LETTER + 'a.png'])

    def test_batch_request_is_text_only_when_image_download_fails(self):
        get = fake_get({LETTER + 'a.png': ConnectionError('down')})
        with mock.patch.object(email_images.requests, 'get', get):
            request = trash_mail_summarize.generate_each_line_of_batch_file(trash_mail('h1', 'a.png'))
        self.assertIsInstance(request['params']['messages'][0]['content'], str)

    def test_start_batch_stops_attaching_images_past_batch_budget(self):
        png = make_image()
        mails = [trash_mail(f'h{i}', f'{i}.png') for i in range(3)]
        sent = []
        with mock.patch.object(email_images.requests, 'get', fake_get({LETTER + f'{i}.png': png for i in range(3)})), \
                mock.patch.object(trash_mail_summarize, 'BATCH_BYTES_BUDGET', 1), \
                mock.patch.object(trash_mail_summarize, 'get_not_processed_trash_mail_list', return_value=mails), \
                mock.patch.object(trash_mail_summarize, 'start_processing_batch_file',
                                  side_effect=lambda requests: sent.extend(requests) or NS(id='msgbatch_test')), \
                mock.patch.object(trash_mail_summarize, 'update_processing_batch_list'), \
                mock.patch.object(trash_mail_summarize, 'update_trash_email_json_file_with_batch_id'):
            trash_mail_summarize.start_batch()
        self.assertEqual([len(image_sources(r['params']['messages'][0]['content'])) for r in sent], [1, 0, 0])


if __name__ == '__main__':
    unittest.main()
