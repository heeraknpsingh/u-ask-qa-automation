"""
Helpers for interacting with browser localStorage via Playwright.
"""

from __future__ import annotations
import json
from typing import Literal
from playwright.async_api import Page

LocaleValue = Literal["en", "ar"]

async def set_locale_in_local_storage(
    page: Page,
    locale: LocaleValue,
    *,
    key: str = "locale",
    apply_to_current_page: bool = True,
    persist_for_future_navigations: bool = True,
    reload_page: bool = False,
) -> None:
    """
    Set a locale value ("en" / "ar") in `window.localStorage`.
    - If `apply_to_current_page` is True, we try to write immediately via `page.evaluate`.
    - If `persist_for_future_navigations` is True, we register an init script so the value
      is set before app code runs on subsequent navigations/reloads.
    - If `reload_page` is True, we reload after setting (useful if the app reads locale at boot).
    """
    if locale not in ("en", "ar"):
        raise ValueError(f"locale must be 'en' or 'ar', got: {locale!r}")
    payload = {"key": key, "value": locale}
    if persist_for_future_navigations:
        # `add_init_script` doesn't support passing args in the Python API, so embed safely.
        await page.add_init_script(
            f"""
            () => {{
              const payload = {json.dumps(payload)};
              try {{
                window.localStorage.setItem(payload.key, payload.value);
              }} catch (e) {{}}
            }}
            """
        )
    if apply_to_current_page:
        # Works when we're on an origin that allows localStorage access.
        await page.evaluate(
            """
            (payload) => {
              try {
                window.localStorage.setItem(payload.key, payload.value);
              } catch (e) {}
            }
            """,
            payload,
        )
    if reload_page:
        await page.reload(wait_until="domcontentloaded")
