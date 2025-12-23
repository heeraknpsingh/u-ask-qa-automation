"""Test reporting utilities for generating HTML and JSON reports."""

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field
from .config import get_settings


class TestResult(BaseModel):
    name: str
    status: str  # passed, failed, skipped
    duration_ms: float = 0
    error_message: Optional[str] = None
    screenshot_path: Optional[str] = None
    language: str = "en"


class TestSuiteResult(BaseModel):
    name: str
    tests: List[TestResult] = Field(default_factory=list)
    start_time: datetime = Field(default_factory=datetime.now)


class TestReporter:
    """Generates human-readable and machine-readable test summaries."""

    def __init__(self, report_dir: Optional[Path] = None):
        self.settings = get_settings()
        self.report_dir = report_dir or self.settings.report_dir
        self.report_dir.mkdir(parents=True, exist_ok=True)
        self.suites: List[TestSuiteResult] = []

    def add_suite(self, suite: TestSuiteResult):
        self.suites.append(suite)

    def _get_summary(self) -> Dict[str, Any]:
        all_tests = [t for s in self.suites for t in s.tests]
        total = len(all_tests)
        passed = sum(1 for t in all_tests if t.status == "passed")

        return {
            "total": total,
            "passed": passed,
            "failed": total - passed,
            "pass_rate": round((passed / total * 100), 2) if total > 0 else 0,
        }

    def save_json_report(self, filename: str = "results.json"):
        """Saves detailed suite data to JSON."""
        data = {
            "summary": self._get_summary(),
            "timestamp": datetime.now().isoformat(),
            "suites": [s.model_dump() for s in self.suites],
        }
        path = self.report_dir / filename
        with open(path, "w") as f:
            json.dump(data, f, indent=2, default=str)
        return path

    def generate_html_report(self, filename: str = "report.html"):
        """Generates a styled HTML report."""
        summary = self._get_summary()
        rows = []
        for s in self.suites:
            for t in s.tests:
                status_color = "#22c55e" if t.status == "passed" else "#ef4444"
                rows.append(f"""
                    <tr>
                        <td>{t.name}</td>
                        <td>{s.name}</td>
                        <td style="color: {status_color}; font-weight: bold;">{t.status.upper()}</td>
                        <td>{round(t.duration_ms, 2)}ms</td>
                    </tr>
                """)

        html_content = f"""
        <html>
            <body style="font-family: sans-serif; padding: 20px;">
                <h1>Test Report - {summary["pass_rate"]}% Pass</h1>
                <p>Total: {summary["total"]} | Passed: {summary["passed"]} | Failed: {summary["failed"]}</p>
                <table border="1" style="width: 100%; border-collapse: collapse;">
                    <tr style="background: #eee;"><th>Test</th><th>Suite</th><th>Status</th><th>Duration</th></tr>
                    {"".join(rows)}
                </table>
            </body>
        </html>
        """
        path = self.report_dir / filename
        path.write_text(html_content)
        return path
