import imaplib
import email
import logging
from email.header import decode_header, make_header
from email.utils import parsedate_to_datetime
import datetime
from typing import List
from zoneinfo import ZoneInfo
import openai
import xxhash
from true_calendar import true_calendar
from true_email import true_email
from true_line import true_line
import os
import html2text
import re
import json
import trash_mail_summarize
from todoist_api_python.api import TodoistAPI
from pydantic import BaseModel, Field
import sqlite3
import os.path
import base64
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from email.message import EmailMessage
import numpy as np
import xgboost as xgb
from sklearn.preprocessing import LabelEncoder
from pathlib import Path
from dotenv import load_dotenv


logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')

# --- Constants ---
_SKIP_SUBJECT = '[서울대학교] 인증코드(Verification Code) 발송'
_SKIP_BODY_FRAGMENT = 'SYSTEM: System failed to mount external device [USB Disk 1] partition'
_TRASH_SUBJECT_KEYWORDS = ['Temperature log']
_TRASH_SENDER_KEYWORDS = ['healthchecks']
_TRASH_BODY_KEYWORDS = ['청강생 승인을 요청드립니다. 서울대학교 New eTL에 승인 여부를 입력해 주세요.']

load_dotenv()

# --- Module-level singletons ---
_openai_client: openai.OpenAI | None = None
_todoist_api: TodoistAPI | None = None
_xgb_models: dict = {}
_label_encoders: dict = {}
_mention_pattern: re.Pattern | None = None

# Todoist priority map keyed by (category, to_me)
_TASK_PRIORITY: dict[tuple, int] = {
    ('연구실', True): 4,
    ('답장', True): 3,
    ('멘션', True): 3,
    ('TA', True): 3,
}


def _get_openai_client() -> openai.OpenAI:
    global _openai_client
    if _openai_client is None:
        _openai_client = openai.OpenAI(api_key=os.getenv('OPENAI_API_KEY_EMAIL'))
    return _openai_client


def _get_todoist_api() -> TodoistAPI:
    global _todoist_api
    if _todoist_api is None:
        _todoist_api = TodoistAPI(os.getenv('TODOIST_API_TOKEN'))
    return _todoist_api


def _get_classifier(classify_type: str) -> xgb.XGBClassifier:
    if classify_type not in _xgb_models:
        m = xgb.XGBClassifier()
        m.load_model(f'{get_current_path()}xgb_model_{classify_type}.json')
        _xgb_models[classify_type] = m
    return _xgb_models[classify_type]


def _get_label_encoder(classify_type: str) -> LabelEncoder:
    if classify_type not in _label_encoders:
        with open(f'{get_current_path()}label_encoder.json', 'r', encoding='utf-8') as f:
            data = json.load(f)
        le = LabelEncoder()
        le.classes_ = np.array(data[classify_type])
        _label_encoders[classify_type] = le
    return _label_encoders[classify_type]


def _get_mention_pattern() -> re.Pattern | None:
    global _mention_pattern
    if _mention_pattern is None:
        my_names_env = os.getenv('MY_NAMES', '')
        if not my_names_env:
            return None
        target_names = [re.escape(n.strip()) for n in my_names_env.split(',')]
        exception_pattern = os.getenv('MY_NAME_EXCEPTIONS', '')
        _mention_pattern = re.compile(
            f'({"|".join(target_names)})(?!(?:{exception_pattern}))',
            re.IGNORECASE
        )
    return _mention_pattern


def check_email():
    def clear_body(dirty_body):
        html_clear = html2text.html2text(dirty_body)
        snu_blahblah_clear = html_clear.split('본 메일은 서울대학교 대량메일시스템을 통해 발송된 메일입니다.')[0]
        return snu_blahblah_clear

    IMAP_SERVER = 'imap.gmail.com'
    IMAP_PORT = 993

    email_address = os.getenv('SNU_GMAIL_EMAIL_ADDRESS')
    password = os.getenv('SNU_GMAIL_PASSWORD')

    kst = ZoneInfo("Asia/Seoul")
    yesterday = datetime.datetime.now(kst).date() - datetime.timedelta(days=1)
    yesterday_str = yesterday.strftime('%d-%b-%Y')

    email_list_ = []
    imap_client = imaplib.IMAP4_SSL(IMAP_SERVER, IMAP_PORT)
    imap_client.login(email_address, password)
    try:
        imap_client.select('Inbox')

        status, response = imap_client.search(None, f'(SINCE "{yesterday_str}")')
        if status == 'OK' and response[0]:
            email_ids = response[0].split()

            for email_id in email_ids:
                res, msg = imap_client.fetch(email_id, "(RFC822 X-GM-THRID)")
                raw_metadata = msg[0][0].decode('utf-8')
                email_message = email.message_from_bytes(msg[0][1])

                message_id = email_message.get('Message-ID')

                thread_id_match = re.search(r'X-GM-THRID\s+(\d+)', raw_metadata)
                if thread_id_match:
                    decimal_thread_id = int(thread_id_match.group(1))
                    gmail_thread_id = format(decimal_thread_id, 'x')
                else:
                    gmail_thread_id = None

                subject = str(make_header(decode_header(email_message.get('Subject'))))
                if _SKIP_SUBJECT in subject:
                    continue

                fr = str(make_header(decode_header(email_message.get('From'))))
                if os.getenv('SNU_GMAIL_EMAIL_ADDRESS') in fr:
                    continue

                to_header = email_message.get('To')
                if to_header:
                    to_addresses = str(make_header(decode_header(to_header)))
                    to_addresses = [addr.strip() for addr in to_addresses.split(',')]
                    to_me = any(os.getenv('SNU_GMAIL_EMAIL_ADDRESS') in each for each in to_addresses)
                else:
                    to_addresses = []
                    to_me = False

                cc_header = email_message.get('Cc')
                if cc_header:
                    cc_addresses = str(make_header(decode_header(cc_header)))
                    cc_addresses = [addr.strip() for addr in cc_addresses.split(',')]
                    to_addresses.extend(cc_addresses)

                received_date = None
                received_datetime = None
                received_date_raw = email_message.get('Date')
                if received_date_raw:
                    dt = parsedate_to_datetime(received_date_raw).astimezone(kst)
                    received_date = dt.strftime('%Y-%m-%d')
                    received_datetime = dt.strftime("%b %d, %Y, at %I:%M %p")

                body_candidate = []
                for part in email_message.walk():
                    if part.get_content_type().startswith('text/'):
                        charset = part.get_content_charset() or 'utf-8'
                        try:
                            body_candidate.append({
                                'type': part.get_content_type(),
                                'body': part.get_payload(decode=True).decode(charset)
                            })
                        except UnicodeDecodeError:
                            body_candidate.append({
                                'type': part.get_content_type(),
                                'body': part.get_payload(decode=True).decode('utf-8', errors='replace')
                            })

                html_dicts = [d for d in body_candidate if d.get('type') == 'text/html']
                plain_dicts = [d for d in body_candidate if d.get('type') == 'text/plain']
                preferred = html_dicts or plain_dicts or body_candidate
                body_content = ''.join(d['body'] for d in preferred)
                body_content = clear_body(body_content)

                if _SKIP_BODY_FRAGMENT in body_content:
                    continue

                email_list_.append({
                    'subject': subject,
                    'sender': fr,
                    'receiver': to_addresses,
                    'date': received_date,
                    'datetime': received_datetime,
                    'body': body_content,
                    'to_me': to_me,
                    'email_id': message_id,
                    'gmail_thread_id': gmail_thread_id,
                })
    finally:
        imap_client.logout()

    return email_list_


def is_reply(email_subject):
    reply_subject = ['Re:', 'RE:', '[RE]']
    for candidate in reply_subject:
        if email_subject.startswith(candidate):
            return True
    return False


def get_embedding(text, model="text-embedding-3-large", _depth=0):
    """텍스트를 OpenAI 임베딩 벡터로 변환"""
    try:
        result = _get_openai_client().embeddings.create(input=[text], model=model)
    except Exception as e:
        if 'Please reduce your prompt' in str(e) and len(text) > 100 and _depth < 5:
            logging.warning('텍스트가 너무 길어 절반으로 줄여 다시 시도합니다...')
            return get_embedding(text[:len(text) // 2], model=model, _depth=_depth + 1)
        raise

    return result.data[0].embedding


def classify_email(email_subject, email_body, classify_type='all'):
    """
    이메일 제목과 본문을 받아서 카테고리를 분류하는 함수

    Args:
        email_subject (str): 이메일 제목
        email_body (str): 이메일 본문
        classify_type (str): 분류 유형 ('all' or 'lab')

    Returns:
        tuple: (예측된 카테고리, 각 카테고리별 확률 딕셔너리)
    """
    full_text = f"<title>{email_subject}</title>\n<body>{email_body}</body>"
    embedding_vector = get_embedding(full_text)

    if classify_type == 'all':
        reward_keywords = ['기프티콘', '상품권', '쿠폰', '보상', '사례금', '사례비', '인건비']
        has_reward = 1 if any(keyword in full_text for keyword in reward_keywords) else 0
        X = np.array(embedding_vector + [has_reward]).reshape(1, -1)
    else:
        X = np.array(embedding_vector).reshape(1, -1)

    model = _get_classifier(classify_type)
    le = _get_label_encoder(classify_type)

    y_pred = model.predict(X)[0]
    y_pred_proba = model.predict_proba(X)[0]

    predicted_category = le.inverse_transform([y_pred])[0]

    category_probs = {
        category: float(prob)
        for category, prob in zip(le.classes_, y_pred_proba)
    }
    logging.info(f"분류 결과: {predicted_category}")
    for cat, prob in sorted(category_probs.items(), key=lambda x: x[1], reverse=True):
        logging.info(f"  {cat}: {prob*100:.2f}%")

    return predicted_category, category_probs


def summarize_email(email_subject, email_content, img_urls=None):
    if img_urls is None:
        img_urls = []
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
    message = [{"type": "text", "text": message}]
    if img_urls:
        for each_img_url in img_urls:
            message.append({"type": "image_url", "image_url": {"url": each_img_url}})
    try:
        chat_completion = _get_openai_client().chat.completions.create(
            messages=[
                {"role": "system", "content": system_message},
                {"role": "user", "content": message}
            ],
            max_tokens=500,
            temperature=0.3,
            model="gpt-4o-mini",
            timeout=60.0,
        )
        return chat_completion.choices[0].message.content
    except Exception as e:
        if img_urls:
            return summarize_email(email_subject, email_content)
        logging.error(f"요약 중 에러 발생: {e}")
        return f"요약 중 에러 발생: {e}"


def _summarize_with_reasoning(email_subject: str, email_content: str, instructions: str) -> str:
    user_input = (
        f"<email_subject>\n{email_subject}\n</email_subject>\n\n"
        f"<email_content>\n{email_content}\n</email_content>"
    )
    try:
        response = _get_openai_client().responses.create(
            model="gpt-5-mini-2025-08-07",
            reasoning={"effort": "medium"},
            instructions=instructions,
            input=user_input,
            timeout=60.0,
        )
        return response.output_text
    except Exception as e:
        logging.error(f"요약 중 에러 발생: {e}")
        return f"요약 중 에러 발생: {e}"


def summarize_reply_email(email_subject, email_content):
    instructions = '''Formatting re-enabled

You are an expert summary assistant.

**Goal:**
Summarize the email content provided in the input into Korean.

**Constraints:**
1. Output a maximum of three numbered sentences (1 to 3 sentences).
2. If the content is brief, use fewer sentences as appropriate.
3. Focus primarily on the most recent message in the thread.
4. Use earlier messages only as context.
5. Technical terms can remain in English/original language.

**Output Format:**
1. [Key point]
2. [Optional: Additional info if needed]
3. [Optional: Conclusion if needed]'''
    return _summarize_with_reasoning(email_subject, email_content, instructions)


def summarize_lab_email(email_subject, email_content):
    instructions = '''Formatting re-enabled

You are an expert summary assistant for laboratory-related emails.

**Goal:**
Summarize the provided email content into Korean.

**Constraints:**
1. Output a maximum of three numbered sentences (1 to 3 sentences).
2. If the content is brief or simple, use fewer sentences as appropriate.
3. Each sentence must contain only one specific key point.
4. Keep sentences concise, clear, and direct.
5. Technical terms may remain in their original language if necessary.

**Output Format:**
1. [Key point 1]
2. [Key point 2 (Optional)]
3. [Key point 3 (Optional)]'''
    return _summarize_with_reasoning(email_subject, email_content, instructions)


def extract_image_urls(text):
    pattern = r'!\[.*?\]\((https?://[^\s)]+\.(?:png|jpg|PNG|JPG|gif|GIF|jpeg|JPEG))\)'
    urls = re.findall(pattern, text)
    return urls


def get_current_path():
    return f'{Path(__file__).resolve().parent}/'


def get_db_path():
    DB_FILENAME = 'checked_items.db'
    return f"{get_current_path()}{DB_FILENAME}"


def init_db():
    db_path = get_db_path()
    with sqlite3.connect(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS checked_items (
                hash_value TEXT PRIMARY KEY,
                title TEXT,
                created_at TEXT
            )
        ''')
        conn.commit()


def update_checked_item_list(hash_value, title):
    current_time = datetime.datetime.now().isoformat()
    db_path = get_db_path()
    with sqlite3.connect(db_path) as conn:
        cursor = conn.cursor()
        try:
            cursor.execute('''
                INSERT OR IGNORE INTO checked_items (hash_value, title, created_at)
                VALUES (?, ?, ?)
            ''', (hash_value, title, current_time))
            conn.commit()
        except sqlite3.Error as e:
            logging.error(f"데이터베이스 에러 발생: {e}")
            raise


def is_checked(hash_value):
    db_path = get_db_path()
    with sqlite3.connect(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute('SELECT 1 FROM checked_items WHERE hash_value = ?', (hash_value,))
        result = cursor.fetchone()
    return result is not None


def get_checked_item_list():
    """
    저장된 모든 아이템 리스트를 반환합니다 (디버깅/확인용).
    """
    db_path = get_db_path()
    with sqlite3.connect(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute('SELECT hash_value, title, created_at FROM checked_items')
        rows = cursor.fetchall()
    return rows


def save_trash(dictionary_):
    current_path = get_current_path()
    trash_can_path = f'{current_path}trash_can'
    if not os.path.exists(trash_can_path):
        os.makedirs(trash_can_path, exist_ok=True)
    with open(f'{trash_can_path}/{dictionary_["hash"]}.json', 'w', encoding='utf-8') as f:
        json.dump(dictionary_, f, indent=4, ensure_ascii=False)


def get_xxh3_128(string):
    """
    문자열을 입력받아 XXH3 128비트 해시값(Hex)을 반환하는 함수
    """
    byte_string = string.encode('utf-8')
    hash_object = xxhash.xxh3_128(byte_string)
    return hash_object.hexdigest()


def am_mentioned(email_):
    pattern = _get_mention_pattern()
    if pattern is None:
        return False
    body = email_.get('body', '')
    return bool(pattern.search(body))


def is_lab_mail(email_):
    all_addresses = list(email_['receiver'])
    all_addresses.append(email_['sender'])
    with open(f'{get_current_path()}lab-email-address', 'r', encoding='utf-8') as file:
        lines = file.readlines()
        lab_addresses = [line.strip() for line in lines if line.strip()]
    return any(lab_add in receive_add for lab_add in lab_addresses for receive_add in all_addresses)


class EventDetails(BaseModel):
    title: str = Field(description="이벤트의 제목")
    start_date_time: datetime.datetime = Field(
        description="이벤트의 시작 일시. Timezone 정보가 반드시 포함된 ISO 8601 형식 (YYYY-MM-DDTHH:MM:SS+HH:MM). 예: 2024-01-01T10:00:00+09:00"
    )
    end_date_time: datetime.datetime = Field(
        description="이벤트의 종료 일시. Timezone 정보가 반드시 포함된 ISO 8601 형식 (YYYY-MM-DDTHH:MM:SS+HH:MM). 예: 2024-01-01T10:00:00+09:00"
    )
    location: str = Field(description="이벤트 장소. 명시되지 않았으면 빈 문자열")
    reason: str = Field(description="판단한 근거에 대한 간략한 설명.")


class EventExtraction(BaseModel):
    events: List[EventDetails] = Field(
        description="이메일에서 추출된 모든 이벤트의 목록. 이벤트가 없으면 빈 리스트를 반환.",
        default_factory=list
    )


class EventDuplicationCheck(BaseModel):
    is_duplicate: bool = Field(
        description="새로운 일정이 기존 일정 목록에 이미 존재하는지(중복인지) 여부. 실질적으로 같은 일정이라면 True."
    )
    reason: str = Field(
        description="중복이라고 판단한 이유 또는 중복이 아니라고 판단한 이유에 대한 간략한 설명"
    )


class ReplyDecision(BaseModel):
    is_reply_needed: bool = Field(
        description="답장이 필요한 이메일인지 여부. (질문이 있거나, 요청이 있는 경우 True)"
    )
    reply_to: str = Field(
        description="답장을 보낼 수신인 이메일 주소. 답장이 필요 없다면 빈 문자열."
    )
    reply_body: str = Field(
        description="제안하는 답장 이메일 본문. 정중하고 명확한 어조로 작성. 답장이 필요 없다면 빈 문자열."
    )
    reason: str = Field(
        description="왜 답장이 필요하거나 필요하지 않다고 판단했는지에 대한 간략한 이유."
    )


def extract_event_info(email_subject: str, email_body: str):
    current_time_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    full_text = f"Subject: {email_subject}\nBody: {email_body}"

    try:
        response = _get_openai_client().responses.parse(
            model="gpt-5-mini-2025-08-07",
            reasoning={"effort": "medium"},
            input=[
                {
                    "role": "system",
                    "content": (
                        f"당신은 이메일을 통해 전달된 일정 정보를 추출하는 비서입니다. "
                        f"현재 시각은 {current_time_str}입니다. "
                        f"이메일 내용을 분석하여 이벤트 제목, 일시, 장소를 추출하세요. "
                        f"규칙: "
                        f"1. 시간대가 명시되지 않은 경우, 한국 표준시(KST, UTC+09:00)로 간주하세요. "
                        f"2. date_time은 반드시 타임존 오프셋(+09:00 등)이 포함된 형식이어야 합니다. "
                        f"3. 종료 일시가 명시되어 있지 않은 경우 시작 일시와 동일하게 작성하세요."
                        f"4. 이벤트 제목은 되도록 간결하게 작성하세요."
                        f"5. 하나의 이메일에 여러 일정이 있다면 모두 리스트에 담아 반환하세요. 단, 하나의 행사를 여러 일정으로 나누는 것을 지양하세요."
                        f"6. 현재 시각을 기준으로 이미 지난 일정은 무시하세요."
                        f"7. 일정을 전달하는 이메일이 아닌 경우 빈 리스트를 반환하세요. 단순한 이전 이메일의 대화 기록은 일정으로 간주하지 마세요."
                    )
                },
                {
                    "role": "user",
                    "content": full_text
                },
            ],
            text_format=EventExtraction,
            timeout=60.0,
        )

        result: EventExtraction = response.output_parsed
        if not result.events:
            logging.info("이메일에서 이벤트 정보를 찾을 수 없습니다")
            return []
        logging.info('일정 추출 완료')
        return result.events

    except Exception as e:
        logging.error(f"시스템 오류 발생: {e}")
        raise


def check_event_duplication(event):
    check_start_date = event.start_date_time - datetime.timedelta(days=1)
    check_end_date = event.end_date_time + datetime.timedelta(days=1)
    existing_events = true_calendar.get_events(start_date=check_start_date, end_date=check_end_date)

    if not existing_events:
        return False

    try:
        new_event_info = f"제목: {event.title}\n시간: {event.start_date_time} ~ {event.end_date_time}"
        existing_events_info = "\n".join([
            f"- 제목: {e['title']} | 시간: {e['start']} ~ {e['end']}"
            for e in existing_events
        ])

        instructions = (
            "당신은 캘린더 일정 관리 비서입니다. "
            "사용자가 추가하려는 '새로운 일정'이 '기존 일정 목록'에 이미 존재하는지 확인하세요. "
            "완전히 동일하지 않더라도, 실질적으로 같은 일정이라면 중복(True)으로 간주하세요."
        )
        user_input = f"--- 새로운 일정 ---\n{new_event_info}\n\n--- 기존 일정 목록 ---\n{existing_events_info}"

        response = _get_openai_client().responses.parse(
            model="gpt-5-mini-2025-08-07",
            reasoning={"effort": "medium"},
            instructions=instructions,
            input=user_input,
            text_format=EventDuplicationCheck,
            timeout=60.0,
        )

        result = response.output_parsed
        logging.info(f"[중복 검사] 중복 여부: {result.is_duplicate}, 이유: {result.reason}")
        return result.is_duplicate

    except Exception as e:
        logging.error(f"분류 중 에러 발생: {e}")
        raise


def _build_email_header(email_) -> str:
    header = f'[{email_["category"]}'
    if email_['category'] == '연구실':
        header += f': {email_["lab category"]}]' if email_["lab category"] != '중요' else ']'
    else:
        header += ']'
    header += '[CC]' if not email_['to_me'] else ''
    header += f' {email_["subject"]}'
    return header


def _send_notifications(email_, header, sender, summary) -> None:
    true_email.self_email(header, f'{sender}\n\n{summary}\n\n\n{email_["body"]}')
    if (email_["category"] in ['답장', '멘션', '연구실', '기프티콘']) and email_['to_me'] and email_.get('lab category', '중요') == '중요':
        true_line.send_text(f'{header}\n\n{sender}\n\n{summary}')


def _handle_calendar_events(email_, header, sender, summary, due_date) -> bool:
    """Extracts and adds calendar events. Returns True if no valid event was added."""
    if email_["category"] not in ['연주회', '연구실']:
        return True

    event_list = extract_event_info(email_['subject'], email_['body'])
    no_valid_event = True
    for each_event in event_list:
        logging.info(f'일정 처리중: {each_event.title}\n{each_event.reason}')
        if check_event_duplication(each_event):
            continue
        true_calendar.add_event(
            title=each_event.title,
            start_time=each_event.start_date_time,
            end_time=each_event.end_date_time,
            location=each_event.location,
            description=f'{sender}\n\n{summary}'
        )
        add_todolist(
            name=f'{header} 일정 검토',
            description=f'{each_event.start_date_time}에 {each_event.title} 일정 생성함\n\n{sender}\n\n{summary}',
            due_date=due_date,
            priority=(3 if email_['category'] == '연구실' and email_['to_me'] else 2)
        )
        no_valid_event = False
    return no_valid_event


def _handle_reply(email_, header, sender, summary, due_date) -> bool:
    """Generates and creates a reply draft if needed. Returns True if a reply was created."""
    if email_["category"] not in ['답장', '연구실', '수업', 'TA', '멘션']:
        return False

    reply_email = generate_email_reply(email_)
    if not reply_email.is_reply_needed:
        return False

    cc = list(email_['receiver'])
    cc.append(email_['sender'])
    cc = [extract_name_and_email(each_address)[2] for each_address in cc
          if reply_email.reply_to not in each_address and os.getenv('SNU_GMAIL_EMAIL_ADDRESS') not in each_address]
    create_reply_draft(
        to_email=reply_email.reply_to,
        cc=cc,
        subject=email_['subject'],
        body=reply_email.reply_body,
        thread_id=email_['gmail_thread_id'],
        original_message_id=email_['email_id'],
        original_content=email_['body'],
        original_datetime=email_['datetime'],
        original_from_header=email_['sender']
    )
    add_todolist(
        name=f'{header} 답장',
        description=f'답장 필요함.\n({reply_email.reason})\n\n{sender}\n\n{summary}',
        due_date=due_date,
        priority=4
    )
    return True


def _add_fallback_task(email_, header, sender, summary, due_date, no_valid_event, reply_needed) -> None:
    """Adds a generic Todoist task when no event or reply was created."""
    if not (no_valid_event and not reply_needed):
        return

    category = email_["category"]
    to_me = email_['to_me']

    if to_me:
        priority = _TASK_PRIORITY.get((category, True), 2)
    elif category == '연구실':
        priority = 2
    else:
        priority = 1

    add_todolist(
        name=header,
        description=f'{sender}\n\n{summary}',
        due_date=due_date,
        priority=priority
    )


def finalize_email(to_be_finalized_email, given_summary):
    logging.info(f'{to_be_finalized_email["subject"]} 마무리 중')

    due_date = 'tomorrow' if datetime.datetime.now(ZoneInfo("Asia/Seoul")).hour >= 19 else 'today'
    sender, _, _ = extract_name_and_email(to_be_finalized_email['sender'])
    header = _build_email_header(to_be_finalized_email)

    _send_notifications(to_be_finalized_email, header, sender, given_summary)
    no_valid_event = _handle_calendar_events(to_be_finalized_email, header, sender, given_summary, due_date)
    reply_needed = _handle_reply(to_be_finalized_email, header, sender, given_summary, due_date)
    _add_fallback_task(to_be_finalized_email, header, sender, given_summary, due_date, no_valid_event, reply_needed)

    update_checked_item_list(to_be_finalized_email['hash'], to_be_finalized_email['subject'])


def add_todolist(name, description, due_date, priority):
    task = _get_todoist_api().add_task(
        content=name,
        description=description,
        due_string=due_date,
        priority=priority,
        labels=['이메일']
    )
    logging.info(f"작업 생성 성공: {task.content} (ID: {task.id})")


def extract_name_and_email(text):
    pattern = r'"?\s*([^"<]+?)\s*"?\s*<([^>]+)>'
    match = re.search(pattern, text)
    if match:
        name = match.group(1).strip()
        addr = match.group(2).strip()
        return f"{name} ({addr})", name, addr
    email_match = re.search(r'[\w.+\-]+@[\w.\-]+\.\w+', text)
    if email_match:
        addr = email_match.group(0)
        return text, None, addr
    return text, None, text


def get_gmail_service():
    """Gmail API 서비스 객체를 생성하고 인증을 처리합니다."""
    SCOPES = ['https://www.googleapis.com/auth/gmail.compose']
    creds = None
    if os.path.exists(f'{get_current_path()}token.json'):
        creds = Credentials.from_authorized_user_file(f'{get_current_path()}token.json', SCOPES)

    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            raise RuntimeError(
                "Gmail OAuth credentials are missing or expired and cannot be refreshed "
                "non-interactively. Run this script once in an interactive session to "
                "generate token.json."
            )
        with open(f'{get_current_path()}token.json', 'w') as token:
            token.write(creds.to_json())

    return build('gmail', 'v1', credentials=creds)


def create_reply_draft(to_email, cc, subject, body, thread_id, original_message_id,
                       original_content, original_datetime, original_from_header):
    service = get_gmail_service()

    try:
        reply_header = f"On {original_datetime}, {original_from_header} wrote:"
        quote_prefix = "> "
        quoted_content = "\n".join([f"{quote_prefix}{line}" for line in original_content.splitlines()])

        to_str = ', '.join(to_email) if isinstance(to_email, list) else to_email
        cc_str = ', '.join(cc) if isinstance(cc, list) else (cc or '')

        recipient_block = f"[수신인] {to_str}"
        if cc_str:
            recipient_block += f"\n[참조] {cc_str}"

        full_body = f"{recipient_block}\n\n{body}\n\n{reply_header}\n{quoted_content}"

        message = EmailMessage()
        message.set_content(full_body)

        if not subject.lower().startswith('re:'):
            message['Subject'] = f"Re: {subject}"
        else:
            message['Subject'] = subject

        message['In-Reply-To'] = original_message_id
        message['References'] = original_message_id

        encoded_message = base64.urlsafe_b64encode(message.as_bytes()).decode()

        create_message = {
            'message': {
                'raw': encoded_message,
                'threadId': thread_id
            }
        }

        draft = service.users().drafts().create(userId='me', body=create_message).execute()
        logging.info(f"임시보관함 저장 성공! Draft ID: {draft['id']}")
        return draft

    except Exception as error:
        logging.error(f"임시보관함 메일 생성 중 에러 발생: {error}")
        raise


def generate_email_reply(given_email):
    sender = given_email['sender']
    recipient = given_email['receiver']
    email_subject = given_email['subject']
    email_body = given_email['body']

    full_context = (
        "<input_email>\n"
        f"From: {sender}\n"
        f"To: {recipient}\n"
        f"Subject: {email_subject}\n"
        f"Body:\n{email_body}\n"
        "</input_email>"
    )
    with open(f'{get_current_path()}reply-style.xml', 'r', encoding='utf-8') as f:
        style_examples = f.read()

    try:
        response = _get_openai_client().responses.parse(
            model="gpt-5-mini-2025-08-07",
            reasoning={"effort": "medium"},
            instructions=(
                f"You are an expert business email assistant. The user's name is {os.getenv('USER_NAME')}. Analyze the email provided in <input_email> tags.\n\n"

                "YOUR GOAL:\n"
                "1. DECIDE: Determine if a reply is strictly necessary. Focus only on the most recent message in the thread. Ignore any unanswered questions or requests from previous emails if they are not reiterated in the latest message. A reply is needed only if the latest message contains explicit questions, new requests, or urgent scheduling needs directed at the user.\n"
                "2. DRAFT: If a reply is needed, draft a polite, concise business email response.\n"
                "   - Match the language of the sender.\n"
                "   - Strictly follow the tone and style shown in the <examples> section below.\n"
                "   - Identify the correct recipient for 'reply_to'.\n\n"

                "OUTPUT RULES:\n"
                "- If no reply is needed, return 'is_reply_needed' as False and empty strings for body/recipient.\n\n"

                f"{style_examples}"
            ),
            input=full_context,
            text_format=ReplyDecision,
            timeout=60.0,
        )

        result: ReplyDecision = response.output_parsed
        logging.info(f"답장 필요: {result.is_reply_needed} / 이유: {result.reason}")
        return result

    except Exception as e:
        logging.error(f"시스템 오류 발생: {e}")
        raise


# Initialise DB once at module load
init_db()


if __name__ == "__main__":
    logging.info('start email classify')
    email_list = check_email()
    logging.info('email read complete')
    important_email_list = []

    for each_email in email_list:
        email_hash = f'{each_email["subject"]}\n{each_email["date"]}\n{each_email["body"]}'
        email_hash = get_xxh3_128(email_hash)
        if is_checked(email_hash):
            continue
        each_email['hash'] = email_hash

        subject = each_email['subject']
        sender = each_email['sender']

        if any(kw in subject for kw in _TRASH_SUBJECT_KEYWORDS):
            classify_result = 'TRASH'
        elif any(kw in sender for kw in _TRASH_SENDER_KEYWORDS):
            classify_result = 'TRASH'
        elif any(kw in each_email['body'] for kw in _TRASH_BODY_KEYWORDS):
            classify_result = 'TRASH'
        else:
            classify_result, _ = classify_email(subject, each_email['body'])

            if classify_result != '연구실' and is_reply(subject):
                classify_result = '답장'

            if classify_result == 'TRASH':
                if is_lab_mail(each_email):
                    classify_result = '연구실'
                elif is_reply(subject):
                    classify_result = '답장'
                elif am_mentioned(each_email):
                    classify_result = '멘션'

        if classify_result == '연구실':
            lab_classify, _ = classify_email(subject, each_email['body'], classify_type='lab')
            each_email['lab category'] = lab_classify

        logging.info(f'[{classify_result}] {subject}')
        if classify_result == 'TRASH':
            save_trash(each_email)
            update_checked_item_list(email_hash, subject)
            continue

        each_email['category'] = classify_result
        important_email_list.append(each_email)

    for each_email in important_email_list:
        if is_reply(each_email['subject']):
            summary = summarize_reply_email(each_email['subject'], each_email['body'])
        elif each_email['category'] == '연구실':
            summary = summarize_lab_email(each_email['subject'], each_email['body'])
        else:
            summary = summarize_email(each_email['subject'], each_email['body'], img_urls=extract_image_urls(each_email['body']))

        finalize_email(each_email, summary)

    logging.info('email summarize complete')
    trash_mail_summarize.start_batch()
