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

class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )
    # Target URL
    base_url: str = Field(default="")
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
    timeout: int = Field(default=30000)
    element_timeout: int = Field(default=10000)
    # Response Validation
    response_timeout: int = Field(default=60000)
    # Reporting
    report_dir: Path = Field(default=Path("reports"))
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
        """If screenshot_dir is bare 'screenshots', place under report_dir."""
        report_dir = info.data.get("report_dir") if hasattr(info, "data") else None
        if report_dir and v == Path("screenshots"):
            return Path(report_dir) / "screenshots"
        return v

    def get_viewport(self) -> dict:
        """Get viewport dimensions based on viewport_type."""
        viewports = {
            ViewportType.DESKTOP: {"width": self.viewport_width, "height": self.viewport_height},
            ViewportType.TABLET: {"width": 768, "height": 1024},
            ViewportType.MOBILE: {"width": 375, "height": 812},
        }
        return viewports.get(self.viewport_type, viewports[ViewportType.DESKTOP])

    def ensure_directories(self):
        self.report_dir.mkdir(parents=True, exist_ok=True)
        self.screenshot_dir.mkdir(parents=True, exist_ok=True)


_settings: Settings | None = None


def get_settings() -> Settings:
    global _settings
    if _settings is None:
        _settings = Settings()
    return _settings