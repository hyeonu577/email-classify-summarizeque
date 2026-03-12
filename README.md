# email-classify-summarizeque

Automated email management system for SNU (Seoul National University) Gmail accounts. Fetches, classifies, summarizes, and takes action on incoming emails.

## Overview

Emails are fetched via IMAP, classified using an XGBoost classifier with OpenAI embeddings, summarized with GPT models, and then processed with follow-up actions depending on the category: notifications, calendar events, Gmail reply drafts, and Todoist tasks.

## Email Categories

**Primary classification (`all`)**
| Category | Description |
|----------|-------------|
| 행정 | Administrative |
| 수업 | Coursework |
| 연주회 | Concerts / performances |
| 기프티콘 | Gift cards / coupons |
| 연구실 | Lab-related |
| TA | Teaching assistant duties |
| TRASH | Unimportant emails |

**Lab sub-classification (`lab`)**
| Category | Description |
|----------|-------------|
| 홍보 | Promotional |
| 콜로퀴움 | Colloquium / seminar |
| 중요 | Important |

## Processing Pipeline

1. **Fetch**: Retrieve recent emails from Gmail inbox via IMAP
2. **Deduplication**: Skip already-processed emails using XXH3-128 hashing (SQLite DB)
3. **Rule-based filtering**: Skip or trash emails matching subject/sender/body keywords
4. **ML classification**: Classify via OpenAI text embeddings + XGBoost
5. **Additional checks**: Detect reply threads, lab emails, and user mentions
6. **Summarization**: Apply category-specific summarization strategies (general / reply / lab)
7. **Follow-up actions**:
   - Forward summary to self via email
   - Send LINE notification for important emails
   - Extract events from concert/lab emails and add to CalDAV calendar
   - Generate Gmail reply drafts when a response is needed
   - Create Todoist tasks with category-based priority
8. **Trash digest**: Batch-summarize trash emails via OpenAI Batch API, then send as a single digest email

## File Structure

```
snu_mail/
├── main.py                    # Main entry point (classify, summarize, act)
├── trash_mail_summarize.py    # Trash email batch summarization via OpenAI Batch API
├── check_db.py                # DB utility for viewing/deleting processing history
├── hash_update.py             # Bulk hash update script
├── constants.json             # Filtering keywords, priority rules, and other config
├── label_encoder.json         # Label encoder classes for XGBoost models
├── xgb_model_all.json         # XGBoost model for primary classification
├── xgb_model_lab.json         # XGBoost model for lab sub-classification
├── reply-style.xml            # Reply style examples for GPT-based reply generation
├── lab-email-address          # Lab-related email address list
├── credentials.json           # Gmail OAuth client credentials
├── token.json                 # Gmail OAuth token
├── checked_items.db           # SQLite DB tracking processed emails
├── trash_can/                 # Temporary storage for trashed email JSON files
├── jsonl_file_folder/         # JSONL files for OpenAI Batch API
├── true_calendar/             # CalDAV calendar module (git submodule)
├── true_email/                # SMTP email sending module (git submodule)
└── true_line/                 # LINE messaging module (git submodule)
```

## Git Submodules

| Module | Repository | Purpose |
|--------|------------|---------|
| true_calendar | hyeonu577/true-calendar | CalDAV calendar event management |
| true_email | hyeonu577/true-email | Email sending via NAS SMTP |
| true_line | hyeonu577/true-line | LINE message notifications |

## Environment Variables

| Variable | Purpose |
|----------|---------|
| `OPENAI_API_KEY_EMAIL` | OpenAI API key (embeddings, summarization, extraction) |
| `SNU_GMAIL_EMAIL_ADDRESS` | SNU Gmail address for IMAP |
| `SNU_GMAIL_PASSWORD` | SNU Gmail app password |
| `TODOIST_API_TOKEN` | Todoist API token |
| `USER_NAME` | User name used in generated replies |
| `MY_NAMES` | Comma-separated names for mention detection |
| `MY_NAME_EXCEPTIONS` | Regex exception pattern for mention detection |

## Key Dependencies

- `openai` - Embeddings, summarization, event extraction, reply generation
- `xgboost`, `scikit-learn`, `numpy` - Email classification
- `xxhash` - Fast hashing for deduplication
- `html2text` - HTML email body conversion
- `todoist-api-python` - Todoist task management
- `google-api-python-client`, `google-auth` - Gmail API (reply drafts)
- `pydantic` - Structured output parsing
- `jsonlines` - Batch API JSONL file generation
- `python-dotenv` - Environment variable loading

## Usage

```bash
# Classify and process emails
python main.py

# Check and start trash email batch summarization
python trash_mail_summarize.py

# View processing history
python check_db.py
```
