"""Core module for U-Ask QA Automation Framework."""

from .browser import BrowserManager, get_page
from .config import BrowserType, Language, Settings, ViewportType, get_settings
from .reporter import TestReporter, TestResult, TestSuiteResult

__all__ = [
    "BrowserManager",
    "get_page",
    "BrowserType",
    "Language",
    "Settings",
    "ViewportType",
    "get_settings",
    "TestReporter",
    "TestResult",
    "TestSuiteResult",
]
