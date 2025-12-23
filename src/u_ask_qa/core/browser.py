"""Browser management for Playwright-based testing."""

from contextlib import asynccontextmanager
from typing import AsyncGenerator, Optional
from playwright.async_api import Browser, Page, Playwright, async_playwright
from .config import BrowserType, Settings, get_settings


class BrowserManager:
    """Manages Playwright browser and context lifecycle."""

    def __init__(self, settings: Optional[Settings] = None):
        self.settings = settings or get_settings()
        self._playwright: Optional[Playwright] = None
        self._browser: Optional[Browser] = None

    async def start(self):
        """Initialize Playwright and launch the configured browser."""
        self._playwright = await async_playwright().start()

        launchers = {
            BrowserType.CHROMIUM: self._playwright.chromium,
            BrowserType.FIREFOX: self._playwright.firefox,
            BrowserType.WEBKIT: self._playwright.webkit,
        }

        launcher = launchers.get(self.settings.browser_type, self._playwright.chromium)
        self._browser = await launcher.launch(
            headless=self.settings.headless, slow_mo=self.settings.slow_mo
        )

    async def stop(self):
        """Gracefully close browser and playwright instances."""
        if self._browser:
            await self._browser.close()
        if self._playwright:
            await self._playwright.stop()

    @asynccontextmanager
    async def page_context(self) -> AsyncGenerator[Page, None]:
        """Context manager for a single page with its own context."""
        if not self._browser:
            await self.start()

        context = await self._browser.new_context(
            viewport=self.settings.get_viewport(), ignore_https_errors=True
        )
        context.set_default_timeout(self.settings.timeout)

        page = await context.new_page()
        try:
            yield page
        finally:
            await page.close()
            await context.close()


@asynccontextmanager
async def get_page() -> AsyncGenerator[Page, None]:
    """Simplified entry point to get a page for testing."""
    manager = BrowserManager()
    try:
        async with manager.page_context() as page:
            yield page
    finally:
        await manager.stop()
