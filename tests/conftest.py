"""
Pytest configuration and fixtures for GovGPT QA Automation.
"""

import logging
from pathlib import Path
from typing import AsyncGenerator, Optional
from datetime import datetime
from collections import defaultdict
from contextvars import ContextVar

import pytest
import pytest_asyncio
from playwright.async_api import (
    Browser,
    BrowserContext,
    Page,
    async_playwright,
    Playwright,
)

from u_ask_qa.core import (
    BrowserType,
    Language,
    Settings,
    ViewportType,
    get_settings,
)
from u_ask_qa.pages import ChatbotPage
from u_ask_qa.utils import TestDataLoader
from u_ask_qa.validators import ResponseValidator, SecurityChecker


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ============================================================================
# Log capture (per test, for reports)
# ============================================================================

_CURRENT_TEST_NODEID: ContextVar[Optional[str]] = ContextVar(
    "_CURRENT_TEST_NODEID", default=None
)
_LOGS_BY_TEST: dict[str, list[str]] = defaultdict(list)


class _PerTestLogCaptureHandler(logging.Handler):
    """Capture log lines per test nodeid for inclusion in reports."""

    def __init__(
        self, max_chars_per_test: int = 50_000, max_lines_per_test: int = 2_000
    ):
        super().__init__()
        self.max_chars_per_test = max_chars_per_test
        self.max_lines_per_test = max_lines_per_test

    def emit(self, record: logging.LogRecord) -> None:
        nodeid = _CURRENT_TEST_NODEID.get()
        if not nodeid:
            return

        try:
            msg = self.format(record)
        except Exception:
            msg = record.getMessage()

        lines = _LOGS_BY_TEST[nodeid]
        lines.append(msg)

        # Keep only last N lines.
        if self.max_lines_per_test and len(lines) > self.max_lines_per_test:
            del lines[: -self.max_lines_per_test]

        # Keep only last ~N chars.
        if self.max_chars_per_test:
            while lines and (sum(len(s) + 1 for s in lines) > self.max_chars_per_test):
                lines.pop(0)


_LOG_CAPTURE_HANDLER = _PerTestLogCaptureHandler()
_LOG_CAPTURE_HANDLER.setLevel(logging.INFO)
_LOG_CAPTURE_HANDLER.setFormatter(
    logging.Formatter(
        "%(asctime)s | %(levelname)s | %(name)s | %(message)s", "%H:%M:%S"
    )
)
logging.getLogger().addHandler(_LOG_CAPTURE_HANDLER)


# ============================================================================
# Command Line Options
# ============================================================================


def pytest_addoption(parser):
    parser.addoption("--language", default="en", choices=["en", "ar", "both"])
    parser.addoption(
        "--viewport", default="desktop", choices=["desktop", "mobile", "tablet"]
    )
    parser.addoption(
        "--browser-type", default="chromium", choices=["chromium", "firefox", "webkit"]
    )
    parser.addoption("--headed", action="store_true", default=False)
    parser.addoption("--slow-mo", type=int, default=0)
    parser.addoption("--base-url", default=None)
    parser.addoption(
        "--skip-login", action="store_true", default=False, help="Skip login step"
    )


def pytest_configure(config):
    for marker in [
        "ui: UI behavior tests",
        "response: GPT response validation tests",
        "security: Security and injection tests",
        "smoke: Quick smoke tests",
        "arabic: Arabic language tests",
        "english: English language tests",
        "mobile: Mobile viewport tests",
        "desktop: Desktop viewport tests",
        "login: Login tests",
    ]:
        config.addinivalue_line("markers", marker)


# ============================================================================
# Settings (sync fixtures)
# ============================================================================


@pytest.fixture(scope="session")
def test_settings(request) -> Settings:
    logger.info("=" * 60)
    logger.info("INITIALIZING TEST SETTINGS")
    logger.info("=" * 60)

    settings = get_settings()
    settings.test_language = Language(request.config.getoption("--language"))
    settings.viewport_type = ViewportType(request.config.getoption("--viewport"))
    settings.browser_type = BrowserType(request.config.getoption("--browser-type"))
    settings.headless = not request.config.getoption("--headed")
    settings.slow_mo = request.config.getoption("--slow-mo")

    if request.config.getoption("--base-url"):
        settings.base_url = request.config.getoption("--base-url")

    settings.ensure_directories()

    logger.info(f"  Base URL: {settings.base_url}")
    logger.info(f"  Language: {settings.test_language.value}")
    logger.info(f"  Browser: {settings.browser_type.value}")
    logger.info(f"  Headless: {settings.headless}")
    vp = settings.get_viewport()
    logger.info(
        f"  Viewport: {settings.viewport_type.value} ({vp.get('width')}x{vp.get('height')})"
    )
    logger.info(f"  Slow Mo: {settings.slow_mo}ms")
    logger.info("=" * 60)

    return settings


@pytest.fixture(scope="session")
def viewport_config(test_settings) -> dict:
    config = test_settings.get_viewport()
    logger.info(f"Viewport config: {config}")
    return config


@pytest.fixture(scope="session")
def skip_login(request) -> bool:
    skip = request.config.getoption("--skip-login")
    logger.info(f"Skip login: {skip}")
    return skip


# ============================================================================
# Browser (async fixtures)
# ============================================================================


@pytest_asyncio.fixture(scope="function")
async def playwright_instance() -> AsyncGenerator[Playwright, None]:
    logger.info("Starting Playwright...")
    async with async_playwright() as p:
        logger.info("Playwright started")
        yield p
    logger.info("Playwright closed")


@pytest_asyncio.fixture(scope="function")
async def browser(
    playwright_instance: Playwright, test_settings
) -> AsyncGenerator[Browser, None]:
    logger.info(f"Launching {test_settings.browser_type.value} browser...")
    logger.info(f"  Headless: {test_settings.headless}")
    logger.info(f"  Slow Mo: {test_settings.slow_mo}ms")

    launchers = {
        BrowserType.CHROMIUM: playwright_instance.chromium,
        BrowserType.FIREFOX: playwright_instance.firefox,
        BrowserType.WEBKIT: playwright_instance.webkit,
    }
    browser = await launchers[test_settings.browser_type].launch(
        headless=test_settings.headless,
        slow_mo=test_settings.slow_mo,
    )
    logger.info("Browser launched")

    yield browser

    logger.info("Closing browser...")
    await browser.close()
    logger.info("Browser closed")


@pytest_asyncio.fixture(scope="function")
async def context(
    browser: Browser, viewport_config, test_settings
) -> AsyncGenerator[BrowserContext, None]:
    locale = "ar-AE" if test_settings.test_language == Language.ARABIC else "en-US"

    logger.info("Creating browser context...")
    logger.info(f"  Viewport: {viewport_config}")
    logger.info(f"  Locale: {locale}")

    context = await browser.new_context(
        viewport=viewport_config,
        locale=locale,
        timezone_id="Asia/Dubai",
        ignore_https_errors=True,
    )
    context.set_default_timeout(test_settings.timeout)
    logger.info(f"Context created | Timeout: {test_settings.timeout}ms")

    yield context

    logger.info("Closing context...")
    await context.close()
    logger.info("Context closed")


@pytest_asyncio.fixture(scope="function")
async def page(
    context: BrowserContext, test_settings, request
) -> AsyncGenerator[Page, None]:
    logger.info("Creating new page...")
    page = await context.new_page()
    page.set_default_timeout(test_settings.timeout)
    logger.info("Page created")
    yield page
    try:
        settings = get_settings()
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        test_name = (
            request.node.name.replace("[", "_").replace("]", "_").replace("/", "_")
        )
        # Check if test passed or failed
        if hasattr(request.node, "rep_call"):
            status = (
                "PASSED"
                if request.node.rep_call.passed
                else "FAILED"
                if request.node.rep_call.failed
                else "SKIPPED"
            )
        else:
            status = "COMPLETED"
        screenshot_path = (
            settings.screenshot_dir / f"{status}_{test_name}_{timestamp}.png"
        )
        await page.screenshot(path=str(screenshot_path))
        logger.info(f"Screenshot: {screenshot_path}")
        # Persist on the test item so report generation can link it.
        setattr(request.node, "screenshot_path", str(screenshot_path))
    except Exception as e:
        logger.warning(f"Screenshot failed: {e}")
    logger.info("Closing page...")
    await page.close()
    logger.info("Page closed")


# ============================================================================
# Page Objects
# ============================================================================


@pytest_asyncio.fixture(scope="function")
async def chatbot_page(page: Page, test_settings, skip_login) -> ChatbotPage:
    """Create chatbot page object with login."""
    logger.info("=" * 60)
    logger.info("SETTING UP CHATBOT PAGE (with login)")
    logger.info("=" * 60)

    chatbot = ChatbotPage(page, test_settings.test_language)
    await chatbot.navigate()

    if not skip_login:
        logged_in = await chatbot.login()
        assert logged_in, "Login failed - check credentials in .env"

    logger.info("Chatbot page ready")
    return chatbot


@pytest_asyncio.fixture(scope="function")
async def chatbot_page_no_login(page: Page, test_settings) -> ChatbotPage:
    """Create chatbot page object WITHOUT login (for login tests)."""
    logger.info("=" * 60)
    logger.info("SETTING UP CHATBOT PAGE (no login)")
    logger.info("=" * 60)
    chatbot = ChatbotPage(page, test_settings.test_language)
    await chatbot.navigate()
    logger.info("Chatbot page ready (no login)")
    return chatbot


@pytest_asyncio.fixture(scope="function")
async def chatbot_page_en(page: Page, skip_login) -> ChatbotPage:
    """Create English chatbot page object with login."""
    chatbot = ChatbotPage(page, Language.ENGLISH)
    await chatbot.navigate()
    if not skip_login:
        await chatbot.login()
    return chatbot


@pytest_asyncio.fixture(scope="function")
async def chatbot_page_ar(page: Page, skip_login) -> ChatbotPage:
    """Create Arabic chatbot page object with login."""
    chatbot = ChatbotPage(page, Language.ARABIC)
    await chatbot.navigate()

    if not skip_login:
        await chatbot.login()

    return chatbot


# ============================================================================
# Helpers (sync fixtures)
# ============================================================================


@pytest.fixture(scope="session")
def test_data() -> TestDataLoader:
    return TestDataLoader(Path("data"))


@pytest.fixture(scope="session")
def response_validator() -> ResponseValidator:
    return ResponseValidator()


@pytest.fixture(scope="session")
def security_checker() -> SecurityChecker:
    return SecurityChecker()


# ============================================================================
# Test Data (sync fixtures)
# ============================================================================


@pytest.fixture
def smoke_prompts(test_data):
    return test_data.get_smoke_prompts()


@pytest.fixture
def english_prompts(test_data):
    return test_data.get_english_prompts()


@pytest.fixture
def arabic_prompts(test_data):
    return test_data.get_arabic_prompts()


@pytest.fixture
def prompt_pairs(test_data):
    return test_data.get_prompt_pairs()


@pytest.fixture
def edge_cases(test_data):
    return test_data.get_edge_cases()


@pytest.fixture
def xss_payloads(test_data):
    return test_data.get_xss_payloads()


@pytest.fixture
def prompt_injections(test_data):
    return test_data.get_prompt_injections()


@pytest.fixture
def sql_injections(test_data):
    return test_data.get_sql_injections()


@pytest.fixture
def all_security_payloads(test_data):
    return test_data.get_security()


# ============================================================================
# Test Result Hook (for tracking pass/fail)
# ============================================================================


@pytest.hookimpl(tryfirst=True, hookwrapper=True)
def pytest_runtest_makereport(item, call):
    """Store test result for screenshot naming."""
    outcome = yield
    rep = outcome.get_result()
    setattr(item, f"rep_{rep.when}", rep)

    if rep.when == "call":
        logger.info(f"Test completed: {item.name} | Status: {rep.outcome.upper()}")


@pytest.hookimpl(tryfirst=True)
def pytest_runtest_setup(item):
    _CURRENT_TEST_NODEID.set(item.nodeid)


@pytest.hookimpl(tryfirst=True)
def pytest_runtest_call(item):
    _CURRENT_TEST_NODEID.set(item.nodeid)


@pytest.hookimpl(tryfirst=True)
def pytest_runtest_teardown(item):
    _CURRENT_TEST_NODEID.set(item.nodeid)


# ============================================================================
# Report Generation
# ============================================================================


def pytest_sessionfinish(session, exitstatus):
    """Generate reports after test session."""
    import json
    import html as html_lib

    logger.info("=" * 60)
    logger.info("GENERATING REPORTS")
    logger.info("=" * 60)

    settings = get_settings()
    settings.ensure_directories()

    # Collect results
    results = {
        "timestamp": datetime.now().isoformat(),
        "total": session.testscollected,
        "passed": 0,
        "failed": 0,
        "skipped": 0,
        "tests": [],
    }

    for item in session.items:

        def _get_rep(phase: str):
            return getattr(item, f"rep_{phase}", None)

        def _failure_details(rep):
            if not rep or not getattr(rep, "failed", False):
                return None
            longreprtext = getattr(rep, "longreprtext", None)
            if longreprtext:
                return longreprtext
            longrepr = getattr(rep, "longrepr", None)
            return (
                str(longrepr)
                if longrepr is not None
                else "Test failed (no further details)."
            )

        rep_setup = _get_rep("setup")
        rep_call = _get_rep("call")
        rep_teardown = _get_rep("teardown")

        # Prefer the phase that actually failed (setup/call/teardown).
        failure_phase = None
        failure_text = None
        for phase_name, rep in (
            ("setup", rep_setup),
            ("call", rep_call),
            ("teardown", rep_teardown),
        ):
            ft = _failure_details(rep)
            if ft:
                failure_phase = phase_name
                failure_text = ft
                break

        test_result = {
            "name": item.name,
            "nodeid": item.nodeid,
            "outcome": "unknown",
            "duration": 0,
            "failure": None,
            "logs": None,
            "screenshot": None,
        }

        # Outcome/duration are primarily based on call phase, but setup/teardown failures are possible.
        primary_rep = rep_call or rep_setup or rep_teardown
        if primary_rep:
            test_result["outcome"] = primary_rep.outcome
            test_result["duration"] = getattr(primary_rep, "duration", 0) or 0

            if getattr(primary_rep, "passed", False):
                results["passed"] += 1
            elif getattr(primary_rep, "failed", False):
                results["failed"] += 1
            elif getattr(primary_rep, "skipped", False):
                results["skipped"] += 1

        if failure_text:
            test_result["failure"] = {
                "phase": failure_phase,
                "message": failure_text.splitlines()[-1]
                if failure_text
                else "Test failed",
                "trace": failure_text,
            }

        logs = _LOGS_BY_TEST.get(item.nodeid)
        if logs:
            test_result["logs"] = "\n".join(logs)

        screenshot_path = getattr(item, "screenshot_path", None)
        if screenshot_path:
            test_result["screenshot"] = screenshot_path

        results["tests"].append(test_result)

    # Save JSON report
    json_path = settings.report_dir / "test-results.json"
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"📊 JSON report: {json_path}")

    # Save CTRF report
    ctrf_report = {
        "results": {
            "tool": {"name": "pytest"},
            "summary": {
                "tests": results["total"],
                "passed": results["passed"],
                "failed": results["failed"],
                "skipped": results["skipped"],
                "pending": 0,
                "other": 0,
                "start": results["timestamp"],
                "stop": datetime.now().isoformat(),
            },
            "tests": [
                {
                    "name": t["name"],
                    "status": t["outcome"],
                    "duration": int(t["duration"] * 1000),
                    # Non-standard but useful fields for debugging in CTRF consumers.
                    "message": (t["failure"]["message"] if t.get("failure") else None),
                    "trace": (t["failure"]["trace"] if t.get("failure") else None),
                    "logs": t.get("logs"),
                    "screenshot": t.get("screenshot"),
                }
                for t in results["tests"]
            ],
        }
    }

    ctrf_path = settings.report_dir / "ctrf-report.json"
    with open(ctrf_path, "w") as f:
        json.dump(ctrf_report, f, indent=2)
    logger.info(f"📊 CTRF report: {ctrf_path}")

    # Generate HTML report
    test_rows = ""
    for t in results["tests"]:
        failure_html = ""
        if t.get("failure"):
            escaped_trace = html_lib.escape(t["failure"]["trace"] or "")
            escaped_phase = html_lib.escape(t["failure"]["phase"] or "")
            failure_html = f"""
            <details class="failure-details">
                <summary>Failure ({escaped_phase})</summary>
                <pre>{escaped_trace}</pre>
            </details>
            """

        logs_html = ""
        if t.get("logs"):
            escaped_logs = html_lib.escape(t["logs"] or "")
            logs_html = f"""
            <details class="logs-details">
                <summary>Logs</summary>
                <pre>{escaped_logs}</pre>
            </details>
            """

        screenshot_html = ""
        if t.get("screenshot"):
            # Link relative to the report directory so it works when opening report.html from disk.
            from pathlib import Path

            screenshot_path = Path(t["screenshot"])
            try:
                rel_path = screenshot_path.relative_to(settings.report_dir)
            except Exception:
                rel_path = screenshot_path

            escaped_path = html_lib.escape(rel_path.as_posix())
            screenshot_html = f"""
            <details class="screenshot-details">
                <summary>Screenshot</summary>
                <div><a href="{escaped_path}" target="_blank" rel="noopener">Open screenshot</a></div>
            </details>
            """

        test_rows += f"""
        <tr class="row-{t["outcome"]}">
            <td>{t["nodeid"]}</td>
            <td class="status-{t["outcome"]}">{t["outcome"].upper()}</td>
            <td>{t["duration"]:.2f}s</td>
            <td>{failure_html}{screenshot_html}{logs_html}</td>
        </tr>
        """

    html_content = f"""<!DOCTYPE html>
<html>
<head>
    <title>Test Report - {results["timestamp"]}</title>
    <style>
        body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; margin: 40px; background: #f9fafb; }}
        .container {{ max-width: 1200px; margin: 0 auto; background: white; padding: 30px; border-radius: 12px; box-shadow: 0 2px 8px rgba(0,0,0,0.1); }}
        h1 {{ color: #1f2937; margin-bottom: 10px; }}
        .timestamp {{ color: #6b7280; margin-bottom: 20px; }}
        .summary {{ display: flex; gap: 15px; margin: 25px 0; flex-wrap: wrap; }}
        .stat {{ padding: 20px 30px; border-radius: 10px; color: white; text-align: center; min-width: 120px; }}
        .stat-label {{ font-size: 14px; opacity: 0.9; }}
        .stat-value {{ font-size: 32px; font-weight: bold; }}
        .passed {{ background: linear-gradient(135deg, #22c55e, #16a34a); }}
        .failed {{ background: linear-gradient(135deg, #ef4444, #dc2626); }}
        .skipped {{ background: linear-gradient(135deg, #f59e0b, #d97706); }}
        .total {{ background: linear-gradient(135deg, #3b82f6, #2563eb); }}
        table {{ border-collapse: collapse; width: 100%; margin-top: 25px; }}
        th {{ background: #f3f4f6; padding: 14px; text-align: left; font-weight: 600; border-bottom: 2px solid #e5e7eb; }}
        td {{ padding: 12px 14px; border-bottom: 1px solid #e5e7eb; }}
        .row-passed {{ background: #f0fdf4; }}
        .row-failed {{ background: #fef2f2; }}
        .row-skipped {{ background: #fffbeb; }}
        .status-passed {{ color: #16a34a; font-weight: bold; }}
        .status-failed {{ color: #dc2626; font-weight: bold; }}
        .status-skipped {{ color: #d97706; font-weight: bold; }}
        details.failure-details summary {{ cursor: pointer; font-weight: 600; }}
        details.failure-details pre {{ white-space: pre-wrap; background: #111827; color: #f9fafb; padding: 12px; border-radius: 8px; overflow-x: auto; }}
        details.logs-details summary {{ cursor: pointer; font-weight: 600; margin-top: 6px; }}
        details.logs-details pre {{ white-space: pre-wrap; background: #0b1220; color: #e5e7eb; padding: 12px; border-radius: 8px; overflow-x: auto; border: 1px solid #111827; }}
        details.screenshot-details summary {{ cursor: pointer; font-weight: 600; margin-top: 6px; }}
    </style>
</head>
<body>
    <div class="container">
        <h1>🧪 GovGPT Test Report</h1>
        <p class="timestamp">Generated: {results["timestamp"]}</p>
        
        <div class="summary">
            <div class="stat total">
                <div class="stat-value">{results["total"]}</div>
                <div class="stat-label">Total</div>
            </div>
            <div class="stat passed">
                <div class="stat-value">{results["passed"]}</div>
                <div class="stat-label">Passed</div>
            </div>
            <div class="stat failed">
                <div class="stat-value">{results["failed"]}</div>
                <div class="stat-label">Failed</div>
            </div>
            <div class="stat skipped">
                <div class="stat-value">{results["skipped"]}</div>
                <div class="stat-label">Skipped</div>
            </div>
        </div>
        
        <table>
            <tr>
                <th>Test</th>
                <th>Status</th>
                <th>Duration</th>
                <th>Details</th>
            </tr>
            {test_rows}
        </table>
    </div>
</body>
</html>
"""

    html_path = settings.report_dir / "report.html"
    with open(html_path, "w") as f:
        f.write(html_content)
    logger.info(f"📊 HTML report: {html_path}")

    logger.info("=" * 60)
    logger.info(
        f"SUMMARY: {results['passed']} passed, {results['failed']} failed, {results['skipped']} skipped"
    )
    logger.info("=" * 60)
