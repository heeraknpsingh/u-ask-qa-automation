# 🤖 U-Ask / GovGPT QA Automation Framework

Playwright + Pytest QA automation for the GovGPT chatbot UI, using a **Page Object Model** (`src/u_ask_qa/pages`) with **response** + **security** validators and built-in **run artifacts** (HTML/JSON/CTRF reports + screenshots).

### Features

- **Login automation**: credential-driven login via `.env`
- **UI smoke**: widget readiness, send/receive, error banner checks
- **Locale coverage**: English/Arabic prompt runs (incl. localStorage locale helper)
- **Response validation**: basic refusal/disclaimer/language checks
- **Security checks**: payload reflection detection (XSS/injection style)
- **Artifacts**: per-test screenshots + `reports/report.html`, `reports/test-results.json`, `reports/ctrf-report.json`

### Installation

```bash
uv sync
uv run playwright install chromium
```

### Configuration

Create a `.env` file in the repo root (env var names are case-insensitive):

```env
# Target
BASE_URL=https://govgpt.sandbox.dge.gov.ae/

# Auth
LOGIN_EMAIL=<your-user>
LOGIN_PASSWORD=<your-password>

# Browser / execution
HEADLESS=true
BROWSER_TYPE=chromium   # chromium|firefox|webkit

# Viewport
VIEWPORT_TYPE=desktop   # desktop|mobile|tablet
VIEWPORT_WIDTH=1280     # used for desktop
VIEWPORT_HEIGHT=800     # used for desktop

# Timeouts (ms)
TIMEOUT=30000
ELEMENT_TIMEOUT=10000
RESPONSE_TIMEOUT=60000

# Artifacts
REPORT_DIR=reports
SCREENSHOT_DIR=reports/screenshots

# OpenAI / LLM
OPENAI_API_KEY=<your-openai-api-key>

# Evaluator Selection
EVAL_EVALUATOR_TYPE=deepeval

# Evaluation Thresholds (optional - defaults shown)
EVAL_FAITHFULNESS_THRESHOLD=0.7
EVAL_RELEVANCY_THRESHOLD=0.7
EVAL_HALLUCINATION_THRESHOLD=0.5
EVAL_TOXICITY_THRESHOLD=0.3
EVAL_SIMILARITY_THRESHOLD=0.7
```

### Running tests

Run everything:

```bash
uv run pytest -v
```

Common overrides (CLI):

```bash
uv run pytest --headed
uv run pytest --skip-login
uv run pytest --base-url https://example.com/
uv run pytest --browser-type chromium
uv run pytest --viewport mobile
uv run pytest --language en
```

By marker:

```bash
uv run pytest -m smoke
uv run pytest -m login
uv run pytest -m ui
uv run pytest -m response
uv run pytest -m security
```

### Login flow (what the POM does)

Implemented in `src/u_ask_qa/pages/chatbot_page.py` as `ChatbotPage.login()`:

1. Navigate to `BASE_URL`
2. Click login link: `.text-center span`
3. Fill email: `#email`
4. Fill password: `#password`
5. Click login: `.text-center > button`
6. Wait for user profile: `img[alt="User profile"]`
7. Wait for loading spinner to hide: `.spinner`

### Key selectors (current)

| Element | Selector |
|---------|----------|
| Login screen | `#splash-screen.login` |
| Login link | `.text-center span` |
| Email | `#email` |
| Password | `#password` |
| Login button | `.text-center > button` |
| User profile | `img[alt="User profile"]` |
| Chat widget | `#chat-input` |
| Message input | `#chat-input p` |
| Send button | `#send-message-button` |
| Bot response container | `#response-content-container` |
| Error banner | `.error-message, .error` |

### Project structure (current)

```
u-ask-qa-automation/
├── data/
│   └── test-data.json                 # prompts + security payloads
├── reports/
│   ├── report.html                    # HTML report (generated post-run)
│   ├── test-results.json              # JSON report (generated post-run)
│   ├── ctrf-report.json               # CTRF report (generated post-run)
│   └── screenshots/                   # screenshots for each test (passed/failed)
├── src/u_ask_qa/
│   ├── core/
│   │   └── config.py                  # pydantic-settings (.env)
│   ├── evaluators/
│   │   ├── config.py
│   │   └── llm_evaluator.py
│   ├── pages/
│   │   └── chatbot_page.py            # Chatbot Page Object (navigate/login/send/wait)
│   ├── utils/
│   │   ├── local_storage.py           # locale helper (localStorage)
│   │   └── test_data_loader.py        # loader for data/test-data.json
│   └── validators/
│       ├── response_validator.py      # response quality checks
│       └── security_checker.py        # payload reflection checks
└── tests/
    ├── conftest.py                    # fixtures + CLI opts + report generation
    ├── test_llm_evaluation.py
    ├── test_locale_prompts.py
    ├── test_login.py
    ├── test_response_validation.py
    ├── test_security.py
    └── test_ui_behavior.py
```

### Reports / artifacts

After a run, open:

- **HTML report**: `reports/report.html`
- **JSON report**: `reports/test-results.json`
- **CTRF report**: `reports/ctrf-report.json`
- **Screenshots**: `reports/screenshots/` (saved for each test; filenames include PASSED/FAILED)
