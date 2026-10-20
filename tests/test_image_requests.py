"""요약 요청(실시간 Sonnet, 휴지통 Haiku 배치)이 이미지를 URL이 아니라 base64로 싣는지 확인한다.

letter2.snu.ac.kr은 robots.txt로 Claude의 URL 가져오기를 막는다. Claude·네트워크·파일 쓰기는 가짜로 바꾼다.
main.py는 gitignore된 constants.json 등을 읽으므로 이 Pi(운영 폴더나 링크를 건 worktree)에서만 돈다.
"""
import base64
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace as NS
from unittest import mock

import main
import trash_mail_summarize
from tests.test_email_images import fake_network, make_image

LETTER = 'https://letter2.snu.ac.kr/upload/massmail/'
REPO = Path(__file__).resolve().parents[1]


class FakeClaude:
    """beta.messages.create 대역. errors에 든 예외를 앞에서부터 하나씩 raise한 뒤 정상 응답한다."""

    def __init__(self, *errors):
        self.errors = list(errors)
        self.calls = []
        self.beta = NS(messages=NS(create=self.create))

    def create(self, **kwargs):
        self.calls.append(kwargs)
        if self.errors:
            raise self.errors.pop(0)
        return NS(stop_reason='end_turn', content=[NS(type='text', text='1. 요약입니다.')])


def image_sources(content):
    if isinstance(content, str):
        return []
    return [block['source'] for block in content if block['type'] == 'image']


def sent_images(call):
    return image_sources(call['messages'][0]['content'])


class RealtimeSummaryTest(unittest.TestCase):
    def test_summarize_email_sends_downloaded_images_as_base64(self):
        first, second = make_image(size=(10, 10)), make_image(size=(20, 20), fmt='JPEG')
        claude = FakeClaude()
        with fake_network({LETTER + 'a.png': first, LETTER + 'b.jpg': second}), \
                mock.patch.object(main, '_get_anthropic_client', return_value=claude):
            summary = main.summarize_email('제목', '본문', img_urls=[LETTER + 'a.png', LETTER + 'b.jpg'])
        self.assertEqual(summary, '1. 요약입니다.')
        sources = sent_images(claude.calls[0])
        self.assertEqual([(s['type'], s['media_type']) for s in sources],
                         [('base64', 'image/png'), ('base64', 'image/jpeg')])
        self.assertEqual([base64.b64decode(s['data']) for s in sources], [first, second])

    def test_rejected_image_request_is_retried_without_images(self):
        claude = FakeClaude(RuntimeError('이미지 거부'))
        with fake_network({LETTER + 'a.png': make_image()}), \
                mock.patch.object(main, '_get_anthropic_client', return_value=claude):
            summary = main.summarize_email('제목', '본문', img_urls=[LETTER + 'a.png'])
        self.assertEqual(summary, '1. 요약입니다.')
        self.assertEqual([len(sent_images(call)) for call in claude.calls], [1, 0])

    def test_failure_is_not_repeated_when_no_image_was_attached(self):
        claude = FakeClaude(RuntimeError('일시 오류'))
        with fake_network({LETTER + 'a.png': ConnectionError('down')}), \
                mock.patch.object(main, '_get_anthropic_client', return_value=claude):
            summary = main.summarize_email('제목', '본문', img_urls=[LETTER + 'a.png'])
        self.assertTrue(summary.startswith('요약 중 에러 발생'))
        self.assertEqual(len(claude.calls), 1)


def trash_mail(hash_, *image_names):
    body = '안내드립니다.\n' + ''.join(f'![]({LETTER}{name})\n' for name in image_names)
    return {'subject': '제목', 'body': body, 'hash': hash_}


def request_images(request):
    return image_sources(request['params']['messages'][0]['content'])


class TrashBatchTest(unittest.TestCase):
    def test_batch_request_carries_first_image_as_base64(self):
        png = make_image()
        with fake_network({LETTER + 'a.png': png, LETTER + 'b.png': png}) as get:
            request = trash_mail_summarize.generate_each_line_of_batch_file(trash_mail('h1', 'a.png', 'b.png'))
        sources = request_images(request)
        self.assertEqual([(s['type'], s['media_type']) for s in sources], [('base64', 'image/png')])
        self.assertEqual(base64.b64decode(sources[0]['data']), png)
        self.assertEqual(get.requested, [LETTER + 'a.png'])

    def test_batch_request_falls_back_to_next_image(self):
        png = make_image()
        with fake_network({LETTER + 'a.png': ConnectionError('down'), LETTER + 'b.png': png}):
            request = trash_mail_summarize.generate_each_line_of_batch_file(trash_mail('h1', 'a.png', 'b.png'))
        self.assertEqual([base64.b64decode(s['data']) for s in request_images(request)], [png])

    def test_batch_request_is_text_only_when_no_image_downloads(self):
        with fake_network({LETTER + 'a.png': ConnectionError('down')}):
            request = trash_mail_summarize.generate_each_line_of_batch_file(trash_mail('h1', 'a.png'))
        self.assertIsInstance(request['params']['messages'][0]['content'], str)

    def test_start_batch_splits_into_batches_under_size_budget(self):
        mails = [trash_mail(f'h{i}', f'{i}.png') for i in range(3)]
        sent = []
        tagged = []

        def create_batch(requests):
            sent.append(requests)
            return NS(id=f'msgbatch_{len(sent)}')

        with fake_network({LETTER + f'{i}.png': make_image() for i in range(3)}), \
                mock.patch.object(trash_mail_summarize, 'BATCH_BYTES_BUDGET', 1), \
                mock.patch.object(trash_mail_summarize, 'get_not_processed_trash_mail_list', return_value=mails), \
                mock.patch.object(trash_mail_summarize, 'start_processing_batch_file', side_effect=create_batch), \
                mock.patch.object(trash_mail_summarize, 'update_processing_batch_list') as recorded, \
                mock.patch.object(trash_mail_summarize, 'update_trash_email_json_file_with_batch_id',
                                  side_effect=lambda mails_, batch_id: tagged.append(([m['hash'] for m in mails_], batch_id))):
            trash_mail_summarize.start_batch()
        self.assertEqual([[r['custom_id'] for r in batch] for batch in sent], [['h0'], ['h1'], ['h2']])
        self.assertEqual([len(request_images(batch[0])) for batch in sent], [1, 1, 1])
        self.assertEqual([c.args[0] for c in recorded.call_args_list], ['msgbatch_1', 'msgbatch_2', 'msgbatch_3'])
        self.assertEqual(tagged, [(['h0'], 'msgbatch_1'), (['h1'], 'msgbatch_2'), (['h2'], 'msgbatch_3')])

    def test_start_batch_keeps_requests_together_within_budget(self):
        mails = [trash_mail(f'h{i}') for i in range(3)]
        sent = []
        with mock.patch.object(trash_mail_summarize, 'get_not_processed_trash_mail_list', return_value=mails), \
                mock.patch.object(trash_mail_summarize, 'start_processing_batch_file',
                                  side_effect=lambda requests: sent.append(requests) or NS(id='msgbatch_1')), \
                mock.patch.object(trash_mail_summarize, 'update_processing_batch_list'), \
                mock.patch.object(trash_mail_summarize, 'update_trash_email_json_file_with_batch_id'):
            trash_mail_summarize.start_batch()
        self.assertEqual([[r['custom_id'] for r in batch] for batch in sent], [['h0', 'h1', 'h2']])


class PackageImportTest(unittest.TestCase):
    def test_trash_module_imports_as_package_like_the_parent_cron(self):
        # 상위 omnia main.py는 `from snu_mail import trash_mail_summarize`로 부른다. 그 폴더에는 true_email/도 있다
        with tempfile.TemporaryDirectory() as parent:
            os.symlink(REPO, f'{parent}/snu_mail')
            os.symlink(REPO / 'true_email', f'{parent}/true_email')
            result = subprocess.run([sys.executable, '-c', 'from snu_mail import trash_mail_summarize'],
                                    cwd=parent, capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)


if __name__ == '__main__':
    unittest.main()
