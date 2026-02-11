import json
import jsonlines
import os
import datetime
import openai
from true_email import true_email
import re
from dotenv import load_dotenv


load_dotenv()


def extract_name_and_email(text):
    # 정규표현식 패턴:
    # 1. ("?([^"<]+)"?) -> 이름 부분: 따옴표(선택) + 내용 추출 + 따옴표(선택)
    # 2. <([^>]+)>    -> 이메일 부분: < > 안의 내용 추출
    pattern = r'"?\s*([^"<]+?)\s*"?\s*<([^>]+)>'
    
    match = re.search(pattern, text)
    if match:
        name = match.group(1).strip()  # 이름 양끝 공백 제거
        email = match.group(2).strip() # 이메일 추출
        return f"{name} ({email})"
    return text


def summarize_email_without_image(email_subject, email_content):
    summarize_api_key = os.getenv('OPENAI_API_KEY_EMAIL')
    system_message = '''You are an expert in summarizing email content in Korean, but you may use terms in other languages if they are technical.

Using the provided email details, summarize the content in three sentences or fewer. Number each sentence for clarity.

# Steps

1. Read and understand the email subject and body.
2. Identify key points and essential information from the email.
3. Construct a concise summary that conveys the main message, ensuring relevance and coherence.
4. Number each sentence in the summary for clarity.

# Output Format

- Three sentences or fewer, each sentence numbered (1, 2, 3). Ensure clarity and focus on the key points from the email content.

# Examples

**Input:** 
- 제목: [Email Subject]
- 본문: [Email Body]

**Output:** 
1. [Summarized sentence one capturing a key point.]
2. [Summarized sentence two covering additional important information.]
3. [Optional summarized sentence three to complete the summary, if necessary.]

(Ensure summaries are realistic in length and use meaningful placeholders.)'''
    message = f'제목: {email_subject}\n본문: {email_content}'
    try:
        client = openai.OpenAI(api_key=summarize_api_key)
        chat_completion = client.chat.completions.create(
            messages=[
                {"role": "system", "content": system_message},
                {"role": "user", "content": message}
            ],
            max_tokens=500,
            temperature=0.3,
            model="gpt-4o-mini",
        )
        return chat_completion.choices[0].message.content
    except Exception as e:
        print(f"요약 중 에러 발생: {e}")
        return f"요약 중 에러 발생: {e}"


def get_openai_client():
    trash_mail_api_key = os.getenv('OPENAI_API_KEY_EMAIL')
    client = openai.OpenAI(api_key=trash_mail_api_key)
    return client


def get_current_path():
    folder_path = '/python/email_classify_summarizeque/'
    folder_exists = os.path.exists(folder_path)
    if folder_exists:
        return folder_path
    else:
        return ''


def get_trash_mail_list():  # 모든 이메일 dictionary가 담겨있는 list 반환
    parsed_data_list = []
    trash_can_path = f'{get_current_path()}trash_can'
    for filename in os.listdir(trash_can_path):
        if filename.endswith('.json'):
            file_path = os.path.join(trash_can_path, filename)
            try:
                with open(file_path, 'r') as f:
                    parsed_data = json.load(f)
                    parsed_data_list.append(parsed_data)
            except json.JSONDecodeError:
                print(f"'{filename}' 파일을 파싱하는 중 오류가 발생했습니다.")
                raise
            except IOError:
                print(f"'{filename}' 파일을 읽는 중 오류가 발생했습니다.")
                raise
    return parsed_data_list


def get_not_processed_trash_mail_list():  # 아직 batch 작업 들어가지 않은 이메일 dictionary가 담겨있는 list 반환
    trash_mail_list = get_trash_mail_list()
    to_be_returned_not_processed_trash_mail_list = []
    for each_mail in trash_mail_list:
        if 'batch_id' not in each_mail.keys():
            to_be_returned_not_processed_trash_mail_list.append(each_mail)
    return to_be_returned_not_processed_trash_mail_list


def extract_image_urls(text):
    pattern = r'!\[.*?\]\((https?://[^\s)]+\.(?:png|jpg|PNG|JPG|gif|GIF|jpeg|JPEG))\)'
    urls = re.findall(pattern, text)
    return urls


def generate_each_line_of_batch_file(email):  # jsonl 파일의 각 줄을 만드는 함수
    system_message = '''이메일 본문을 분석하여 3문장 이하로 요약하세요. 각 문장은 번호를 매기세요. 모든 응답은 한국어로 작성하세요. 단, 전문 용어는 다른 언어를 사용해도 됩니다.

# Steps

1. 주어진 이메일 제목과 본문을 철저히 분석하세요.
2. 본문에서 핵심 정보를 식별하고 추출합니다.
3. 식별된 정보를 바탕으로 요약 문장을 작성하세요.
4. 각 요약 문장을 번호로 구분합니다.

# Output Format

- 3문장 이하로 구성된 문장 목록, 각 문장은 번호가 매겨짐

# Examples

**Input:** 
```
제목: [반도체특성화대학] 반도체 소자 워크샵 참여 모집 공고
본문: 안녕하세요, 반도체특성화대학입니다.

장학생 여러분 중 반도체 소자 워크숍에 참여할 인원을 조사합니다.

관심 있으신 분들의 많은 지원 부탁드리며, 의무사항은 아니니 참고 부탁드립니다.

특강 교재는 무료로 제공됩니다.

행사 일정: 2025년 2월 14일 (금요일) 오후 1시 ~ 4시 (3시간)
특강 내용: 트랜지스터의 기본 원리, NAND Flash의 동작 원리, 실제 소자 (MOS, FeNAND) 측정
행사 장소: 반도체공동연구소 (104동) 도연홀 및 제1 측정교육실

구글 폼 작성은 2025년 1월 20일 오후 1시까지이니, 기한 맞춰 작성 부탁드리겠습니다.

구글 폼 링크: https://forms.gle/231oETLmDik2CHXA6

감사합니다.

반도체특성화대학 드림
```

**Output:**
1. 반도체특성화대학에서 반도체 소자 워크숍에 참여할 장학생을 모집하며, 의무사항은 아닙니다.
2. 워크숍은 2025년 2월 14일 오후 1시부터 4시까지 반도체공동연구소 도연홀 및 제1 측정교육실에서 진행됩니다.
3. 참여 희망자는 2025년 1월 20일 오후 1시까지 구글 폼을 작성해야 합니다.

# Notes

- 전문용어는 다른 언어를 사용할 수 있으니, 적절히 활용하세요.
- 정보를 정확히 요약하여 잘못된 해석이 없도록 주의하세요.'''
    message = f'제목: {email["subject"]}\n본문: {email["body"]}'
    image_list = extract_image_urls(email['body'])
    if image_list:
        image_url = image_list[0]
        final_user_message = [{"type": "text", "text": message},
                              {"type": "image_url", "image_url": {"url": image_url}}]
    else:
        final_user_message = message
    fuit_json = {"custom_id": email['hash'],
                 "method": "POST",
                 "url": "/v1/chat/completions",
                 "body": {
                     "messages": [{"role": "system", "content": system_message},
                                  {"role": "user", "content": final_user_message}],
                     "max_tokens": 500, "temperature": 0.3,
                     "model": "gpt-4.1-nano"}}
    return fuit_json


def generate_batch_file_with_trash_mail_list(trash_mail_list):  # batch 작업 들어가지 않은 이메일 list를 주면 batch 파일 만드는 함수
    jsonl_data = [generate_each_line_of_batch_file(each_trash_mail) for each_trash_mail in trash_mail_list]

    jsonl_file_name = datetime.datetime.now().strftime('%Y%m%d%H%M%S')
    jsonl_file_path = f"{get_current_path()}jsonl_file_folder/{jsonl_file_name}.jsonl"
    with jsonlines.open(jsonl_file_path, mode="w") as writer:
        writer.write_all(jsonl_data)
    print(f'generated batch file: {jsonl_file_path}')
    return jsonl_file_path


def start_processing_batch_file(batch_file_path):  # batch 파일을 주면 작업 시작하는 함수
    batch_input_file = upload_batch_file(batch_file_path)

    client = get_openai_client()

    batch_input_file_id = batch_input_file.id
    batch = client.batches.create(
        input_file_id=batch_input_file_id,
        endpoint="/v1/chat/completions",
        completion_window="24h"
    )
    return batch


def upload_batch_file(batch_file_name):
    client = get_openai_client()
    batch_input_file = client.files.create(
        file=open(batch_file_name, "rb"),
        purpose="batch"
    )
    return batch_input_file


def get_batch_object_with_id(batch_id):
    client = get_openai_client()
    batch = client.batches.retrieve(batch_id)
    return batch


def get_batch_result(batch_id):
    batch = get_batch_object_with_id(batch_id)
    if batch.status == 'completed':
        if batch.request_counts.completed == 0:
            raise Exception('failed')
        output_file_id = batch.output_file_id
        client = get_openai_client()
        try:
            file_response = client.files.content(output_file_id)
            good_result = file_response.text
            print(f'good result: {good_result}')
        except ValueError:
            error_file_id = batch.error_file_id
            file_response = client.files.content(error_file_id)
            bad_result = file_response.text
            print(f'bad result: {bad_result}')
            final_return = bad_result
            return final_return
        if batch.request_counts.failed == 0:
            final_return = good_result
        else:
            error_file_id = batch.error_file_id
            file_response = client.files.content(error_file_id)
            bad_result = file_response.text
            final_return = good_result + bad_result
        return final_return
    elif batch.status in ['validating', 'in_progress', 'finalizing']:
        raise Exception('in progress')
    elif batch.status in ['failed', 'expired', 'cancelling', 'cancelled']:
        raise Exception('failed')
    else:
        raise Exception('unexpected error')


def convert_batch_result_into_readable_form(batch_result):
    batch_result = [json.loads(line) for line in batch_result.splitlines()]
    readable_batch_result = []
    for each_batch_result in batch_result:
        if each_batch_result['response']['status_code'] == 200:
            each_final_answer = each_batch_result["response"]["body"]["choices"][0]["message"]["content"]
        else:
            each_final_answer = '요약 중 오류 발생'
        readable_batch_result.append((each_batch_result['custom_id'], each_final_answer))
    return readable_batch_result


def update_trash_email_json_file_with_batch_id(processed_trash_mail_list, batch_id):
    current_path = get_current_path()
    for each_trash_mail in processed_trash_mail_list:
        each_trash_mail['batch_id'] = batch_id
        with open(f'{current_path}trash_can/{each_trash_mail["hash"]}.json', 'w', encoding='utf-8') as f:
            json.dump(each_trash_mail, f, indent=4, ensure_ascii=False)


def delete_line_from_file(file_path, line_to_delete):
    # 파일의 모든 내용을 읽어옵니다
    with open(file_path, 'r', encoding='UTF-8') as file:
        lines = file.readlines()

    # 삭제하고 싶은 줄을 제외한 내용을 새 리스트에 저장합니다
    new_lines = [line for line in lines if line.strip() != line_to_delete.strip()]

    # 파일을 다시 열어 새로운 내용을 씁니다
    with open(file_path, 'w', encoding='UTF-8') as file:
        file.writelines(new_lines)


def get_processing_batch_list():
    current_path = get_current_path()
    try:
        f = open(f'{current_path}processing batch list.txt', 'r', encoding='UTF-8')
    except FileNotFoundError:
        f = open(f'{current_path}processing batch list.txt', 'w', encoding='UTF-8')
        f.close()
        return list()
    try:
        processing_item_list_ = f.readlines()
    finally:
        f.close()
    return [processing_item.strip() for processing_item in processing_item_list_]


def update_processing_batch_list(text_):
    current_path = get_current_path()
    f = open(f'{current_path}processing batch list.txt', 'a', encoding='UTF-8')
    try:
        f.write(text_)
        f.write('\n')
    finally:
        f.close()


def start_batch():
    procedendum_trash_mail_list = get_not_processed_trash_mail_list()
    if not procedendum_trash_mail_list:
        return
    jsonl_file_path = generate_batch_file_with_trash_mail_list(procedendum_trash_mail_list)
    batch = start_processing_batch_file(jsonl_file_path)
    batch_id = batch.id
    update_processing_batch_list(batch_id)
    update_trash_email_json_file_with_batch_id(procedendum_trash_mail_list, batch_id)


def get_trash_email_list_with_specific_batch_id(batch_id):
    trash_mail_list = get_trash_mail_list()
    to_be_returned_trash_mail_list = []
    for each_mail in trash_mail_list:
        try:
            if each_mail['batch_id'] == batch_id:
                to_be_returned_trash_mail_list.append(each_mail)
        except KeyError:
            continue
    return to_be_returned_trash_mail_list


def check_processing_batch():
    batch_list = get_processing_batch_list()
    final_body = ''
    to_be_deleted_batch_from_batch_list = []
    to_be_deleted_email_from_trash_can = []
    for each_batch in batch_list:
        try:
            batch_result = get_batch_result(each_batch)
        except Exception as e:
            e = str(e)
            if e == 'in progress':
                print(f'{each_batch} in progress')
                continue
            elif e == 'failed':  # 전체 실패한 경우
                failed_email_list = get_trash_email_list_with_specific_batch_id(each_batch)
                for each_failed_email in failed_email_list:
                    email_summary = summarize_email_without_image(each_failed_email['subject'],
                                                                  each_failed_email['body'])
                    sender = extract_name_and_email(each_failed_email['sender'])
                    final_body += f'{each_failed_email["subject"]}\n\n{sender}\n\n{email_summary}\n\n\n\n'
                final_body.strip()
                to_be_deleted_batch_from_batch_list.append(each_batch)
                to_be_deleted_email_from_trash_can.extend(failed_email_list)
                continue
            else:
                raise
        readable_batch_result = convert_batch_result_into_readable_form(batch_result)
        for email_hash_, email_summary in readable_batch_result:
            file_path = f'{get_current_path()}trash_can/{email_hash_}.json'
            with open(file_path, 'r') as f:
                email_dictionary = json.load(f)
            email_subject = email_dictionary['subject']
            if email_summary == '요약 중 오류 발생':
                email_summary = summarize_email_without_image(email_subject, email_dictionary['body'])
            sender = extract_name_and_email(email_dictionary['sender'])
            final_body += f'{email_subject}\n\n{sender}\n\n{email_summary}\n\n\n\n'
        final_body.strip()
        to_be_deleted_batch_from_batch_list.append(each_batch)
        finished_email_list = get_trash_email_list_with_specific_batch_id(each_batch)
        to_be_deleted_email_from_trash_can.extend(finished_email_list)
    if final_body == '':
        return False
    true_email.send_email('SNU 기타 이메일', final_body, receiver='hyeonu@662607015.com')
    for each_batch in to_be_deleted_batch_from_batch_list:
        delete_line_from_file(f'{get_current_path()}processing batch list.txt', each_batch)
    for each_failed_email in to_be_deleted_email_from_trash_can:
        os.remove(f"{get_current_path()}trash_can/{each_failed_email['hash']}.json")
        
    return True


if __name__ == '__main__':
    print('start checking processing batch')
    check_processing_batch()
    print('starting batch')
    start_batch()
