"""
Configuration management for GovGPT QA Automation Framework.
"""

from enum import Enum
from pathlib import Path

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Language(str, Enum):
    ENGLISH = "en"
    ARABIC = "ar"
    BOTH = "both"


class ViewportType(str, Enum):
    DESKTOP = "desktop"
    MOBILE = "mobile"
    TABLET = "tablet"


class BrowserType(str, Enum):
    CHROMIUM = "chromium"
    FIREFOX = "firefox"
    WEBKIT = "webkit"


class ViewportConfig:
    """Default viewport sizes (used if not set in .env)."""

    DESKTOP = {"width": 1280, "height": 800}
    TABLET = {"width": 768, "height": 1024}
    MOBILE = {"width": 375, "height": 812}


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # Target URL
    base_url: str = Field(default="https://govgpt.sandbox.dge.gov.ae/")

    # Login Credentials
    login_email: str = Field(default="")
    login_password: str = Field(default="")

    # Test Configuration
    test_language: Language = Field(default=Language.ENGLISH)
    viewport_type: ViewportType = Field(default=ViewportType.DESKTOP)
    browser_type: BrowserType = Field(default=BrowserType.CHROMIUM)

    # Custom Viewport Size (from .env)
    viewport_width: int = Field(default=1280)
    viewport_height: int = Field(default=800)

    # Browser Settings
    headless: bool = Field(default=True)
    slow_mo: int = Field(default=0)
    timeout: int = Field(default=30000)
    element_timeout: int = Field(default=10000)

    # Response Validation
    response_timeout: int = Field(default=60000)
    similarity_threshold: float = Field(default=0.7)
    max_response_length: int = Field(default=5000)

    # Reporting
    screenshot_on_failure: bool = Field(default=True)
    generate_html_report: bool = Field(default=True)
    report_dir: Path = Field(default=Path("reports"))
    # Store screenshots under reports by default to keep artifacts together.
    screenshot_dir: Path = Field(default=Path("reports/screenshots"))

    @field_validator("report_dir", "screenshot_dir", mode="before")
    @classmethod
    def ensure_path(cls, v):
        if isinstance(v, str):
            return Path(v)
        return v

    @field_validator("screenshot_dir", mode="after")
    @classmethod
    def normalize_screenshot_dir(cls, v: Path, info):
        """
        Back-compat / convenience:
        - If screenshot_dir is set to bare "screenshots", place it under report_dir.
        This keeps all run artifacts in one place (reports/).
        """
        report_dir = info.data.get("report_dir") if hasattr(info, "data") else None
        if report_dir and v == Path("screenshots"):
            return Path(report_dir) / "screenshots"
        return v

    def get_viewport(self) -> dict:
        """Get viewport based on type, using custom size for desktop."""
        if self.viewport_type == ViewportType.DESKTOP:
            return {"width": self.viewport_width, "height": self.viewport_height}
        elif self.viewport_type == ViewportType.TABLET:
            return ViewportConfig.TABLET
        elif self.viewport_type == ViewportType.MOBILE:
            return ViewportConfig.MOBILE
        return {"width": self.viewport_width, "height": self.viewport_height}

    def ensure_directories(self):
        self.report_dir.mkdir(parents=True, exist_ok=True)
        self.screenshot_dir.mkdir(parents=True, exist_ok=True)


settings = Settings()


def get_settings() -> Settings:
    return settings
