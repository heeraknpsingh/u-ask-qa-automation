"""
Page Object Model for GovGPT Chatbot.
"""
import asyncio
import html as _html
import logging
import re
import time
from typing import Dict, List, Optional, Tuple
from playwright.async_api import Page, TimeoutError as PlaywrightTimeout
from ..core.config import Language, get_settings
from ..utils.local_storage import LocaleValue, set_locale_in_local_storage

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


class ChatbotPage:
    """Page object for GovGPT Chatbot UI."""

    SELECTORS = {
        # Login
        "login_screen": "#splash-screen.login",
        "log_in_link": ".text-center span",
        "email_input": "#email",
        "password_input": "#password",
        "log_in_button": ".text-center > button",
        # Logged in
        "user_profile": 'img[alt="User profile"]',
        "loading_screen": ".spinner",
        # Chat
        "chat_widget": "#chat-input",
        "message_input": "#chat-input p",
        "send_button": "#send-message-button",
        "bot_message": "#response-content-container",
        "loading_indicator": "button svg path[d*='M12.002 14.4']",
        "error_message": ".error-message, .error",
    }

    # Browser-like headers for fetching sources
    FETCH_HEADERS = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,image/apng,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.9,ar;q=0.8",
        "Accept-Encoding": "gzip, deflate, br",
        "Connection": "keep-alive",
        "Upgrade-Insecure-Requests": "1",
    }

    def __init__(self, page: Page, language: Language = Language.ENGLISH):
        self.page = page
        self.language = language
        self.settings = get_settings()
        logger.info("ChatbotPage initialized")

    async def navigate(self) -> bool:
        """Navigate to the chatbot URL."""
        url = self.settings.base_url
        logger.info(f"Navigating to: {url}")
        try:
            response = await self.page.goto(
                url,
                wait_until="domcontentloaded",
                timeout=self.settings.element_timeout,
            )
            logger.info(
                f"Page loaded | Status: {response.status if response else 'N/A'}"
            )
            await self.page.wait_for_selector(
                self.SELECTORS["log_in_link"],
                state="visible",
                timeout=self.settings.element_timeout,
            )
            return response.ok if response else False
        except Exception as e:
            logger.error(f"Navigation failed: {e}")
            return False

    async def login(
        self, email: Optional[str] = None, password: Optional[str] = None
    ) -> bool:
        """Perform login flow."""
        email = email or self.settings.login_email
        password = password or self.settings.login_password
        if not email or not password:
            logger.error("Login credentials not configured")
            raise ValueError("Set LOGIN_EMAIL and LOGIN_PASSWORD in .env")
        logger.info("=" * 50)
        logger.info("STARTING LOGIN FLOW")
        logger.info("=" * 50)
        try:
            # Step 1: Wait for login link
            logger.info(
                f"Step 1: Waiting for login link | Selector: {self.SELECTORS['log_in_link']}"
            )
            await self.page.wait_for_selector(
                self.SELECTORS["log_in_link"],
                state="visible",
                timeout=self.settings.element_timeout,
            )
            logger.info("Login link visible")

            # Step 2: Click login link
            logger.info("Step 2: Clicking login link...")
            await self.page.click(self.SELECTORS["log_in_link"])
            logger.info("Login link clicked")

            # Step 3: Wait for email input
            logger.info(
                f"Step 3: Waiting for email input | Selector: {self.SELECTORS['email_input']}"
            )
            await self.page.wait_for_selector(
                self.SELECTORS["email_input"],
                state="visible",
                timeout=self.settings.element_timeout,
            )
            logger.info("Email input visible")

            # Step 4: Enter email
            logger.info(f"Step 4: Entering email: {email[:3]}***")
            await self.page.fill(self.SELECTORS["email_input"], email)
            logger.info("Email entered")

            # Step 5: Enter password
            logger.info("Step 5: Entering password: ***")
            await self.page.fill(self.SELECTORS["password_input"], password)
            logger.info("Password entered")

            # Step 6: Click login button
            logger.info(
                f"Step 6: Clicking login button | Selector: {self.SELECTORS['log_in_button']}"
            )
            await self.page.click(self.SELECTORS["log_in_button"])
            logger.info("Login button clicked")

            # Step 7: Wait for user profile
            logger.info(
                f"Step 7: Waiting for user profile | Selector: {self.SELECTORS['user_profile']}"
            )
            await self.page.wait_for_selector(
                self.SELECTORS["user_profile"],
                state="visible",
                timeout=self.settings.element_timeout,
            )
            logger.info(
                f"Step 8: Waiting for loading screen to go off | Selector: {self.SELECTORS['loading_screen']}"
            )
            await self.page.wait_for_selector(
                self.SELECTORS["loading_screen"],
                state="hidden",
                timeout=self.settings.element_timeout,
            )
            logger.info("=" * 50)
            logger.info("LOGIN SUCCESSFUL")
            logger.info("=" * 50)
            return True
        except PlaywrightTimeout as e:
            logger.error(f"TIMEOUT: {e}")
            await self._log_page_state()
            return False
        except Exception as e:
            logger.error(f"LOGIN ERROR: {e}")
            await self._log_page_state()
            return False

    async def _log_page_state(self):
        """Log current page state for debugging."""
        try:
            logger.info("-" * 50)
            logger.info("DEBUG: Current page state")
            logger.info(f"  URL: {self.page.url}")
            for name, selector in self.SELECTORS.items():
                try:
                    is_visible = await self.page.is_visible(selector)
                    logger.info(
                        f"  {name}: {'visible' if is_visible else 'not visible'}"
                    )
                except Exception:
                    logger.info(f"  {name}: ? error checking")
            logger.info("-" * 50)
        except Exception as e:
            logger.error(f"Error logging page state: {e}")

    async def is_logged_in(self) -> bool:
        """Check if user is logged in."""
        logger.info("Checking login status...")
        try:
            result = await self.page.is_visible(self.SELECTORS["user_profile"])
            logger.info(f"Login status: {'logged in' if result else 'not logged in'}")
            return result
        except Exception as e:
            logger.error(f"Error checking login: {e}")
            return False

    async def is_widget_loaded(self) -> bool:
        """Check if chat widget is loaded."""
        logger.info(f"Checking widget | Selector: {self.SELECTORS['chat_widget']}")
        try:
            await self.page.wait_for_selector(
                self.SELECTORS["chat_widget"],
                state="visible",
                timeout=self.settings.element_timeout,
            )
            logger.info("Widget loaded")
            return True
        except Exception as e:
            logger.error(f"Widget not loaded: {e}")
            return False

    async def is_input_ready(self) -> bool:
        """Check if message input is ready."""
        logger.info(
            f"Checking input ready | Selector: {self.SELECTORS['message_input']}"
        )
        try:
            await self.page.wait_for_selector(
                self.SELECTORS["message_input"],
                state="visible",
                timeout=self.settings.element_timeout,
            )
            logger.info("Input ready")
            return True
        except Exception as e:
            logger.error(f"Input not ready: {e}")
            return False

    async def is_ready(self) -> bool:
        """Check if the chat UI is ready for interaction."""
        logger.info("Checking if chat widget is ready...")
        try:
            try:
                await self.page.wait_for_selector(
                    self.SELECTORS["loading_screen"],
                    state="hidden",
                    timeout=self.settings.element_timeout,
                )
            except Exception:
                pass

            widget_ok = await self.is_widget_loaded()
            input_ok = await self.is_input_ready()

            if not widget_ok or not input_ok:
                await self._log_page_state()
                return False

            try:
                if await self.page.is_visible(self.SELECTORS["error_message"]):
                    logger.error("Error message is visible; widget not ready")
                    return False
            except Exception:
                pass

            logger.info("Chat widget ready")
            return True
        except Exception as e:
            logger.error(f"Error checking widget readiness: {e}")
            return False

    async def send_message(self, message: str) -> bool:
        """Send a message in the chat."""
        logger.info(
            f"Sending message: {message[:100]}{'...' if len(message) > 100 else ''}"
        )
        try:
            logger.info("Clicking input...")
            input_el = await self.page.wait_for_selector(
                self.SELECTORS["message_input"], state="visible"
            )
            await input_el.click()
            logger.info("Input focused")

            logger.info("Typing message...")
            await self.page.keyboard.type(message)
            logger.info("Message typed")

            send_btn = await self.page.query_selector(self.SELECTORS["send_button"])
            if send_btn and await send_btn.is_visible():
                logger.info("Clicking send button...")
                await send_btn.click()
                logger.info("Send button clicked")
            else:
                logger.info("Pressing Enter...")
                await self.page.keyboard.press("Enter")
                logger.info("Enter pressed")
            return True
        except Exception as e:
            logger.error(f"Send message error: {e}")
            return False

    async def wait_for_response(self, timeout: Optional[int] = None) -> bool:
        """Wait for the chatbot to finish generating a response."""
        target_timeout = timeout or self.settings.response_timeout
        logger.info(f"Waiting for AI response completion | Timeout: {target_timeout}ms")
        start_time = time.time()

        async def _get_last_bot_text() -> Optional[str]:
            try:
                elements = await self.page.query_selector_all(
                    self.SELECTORS["bot_message"]
                )
                if not elements:
                    return None
                text = (await elements[-1].inner_text()) or ""
                return text.strip() or None
            except Exception:
                return None

        async def _get_bot_count() -> int:
            try:
                elements = await self.page.query_selector_all(
                    self.SELECTORS["bot_message"]
                )
                return len(elements)
            except Exception:
                return 0

        baseline_text = await _get_last_bot_text()
        baseline_count = await _get_bot_count()
        try:
            stop_selector = self.SELECTORS["loading_indicator"]
            send_button = self.SELECTORS["send_button"]
            try:
                await self.page.wait_for_selector(
                    stop_selector, state="visible", timeout=2000
                )
                logger.info("Processing detected: AI is currently streaming response...")
            except Exception:
                logger.info("Initial processing icon not detected; checking if already finished.")

            deadline = start_time + (target_timeout / 1000)
            next_heartbeat = start_time
            placeholder_texts = {
                "working on it...",
                "working on it",
                "thinking...",
                "processing...",
            }

            while time.time() < deadline:
                state = await self.page.evaluate(
                    """
                    ({ stopSelector, sendSelector }) => {
                      const stopIcon = document.querySelector(stopSelector);
                      const sendButton = document.querySelector(sendSelector);
                      return {
                        stopVisible: !!stopIcon,
                        sendPresent: sendButton !== null,
                      };
                    }
                    """,
                    {"stopSelector": stop_selector, "sendSelector": send_button},
                )

                done_generating = (not state.get("stopVisible")) and state.get("sendPresent")

                last_text = await _get_last_bot_text()
                bot_count = await _get_bot_count()
                last_text_norm = (last_text or "").strip().lower()
                looks_like_placeholder = (not last_text) or (last_text_norm in placeholder_texts)

                response_ready = False
                if not looks_like_placeholder:
                    if bot_count > baseline_count:
                        response_ready = True
                    elif baseline_text is None:
                        response_ready = True
                    elif last_text != baseline_text:
                        response_ready = True

                if done_generating and response_ready:
                    await asyncio.sleep(0.8)
                    break

                now = time.time()
                if now >= next_heartbeat:
                    elapsed = round(now - start_time, 1)
                    remaining = max(0.0, round(deadline - now, 1))
                    logger.info(
                        "Still waiting for response... elapsed=%ss remaining=%ss "
                        "(stopVisible=%s sendPresent=%s done=%s responseReady=%s botCount=%s)",
                        elapsed, remaining, state.get("stopVisible"), state.get("sendPresent"),
                        done_generating, response_ready, bot_count,
                    )
                    next_heartbeat = now + 5.0

                await asyncio.sleep(0.5)

            duration = round(time.time() - start_time, 2)
            logger.info(f"Response completed in {duration}s")
            return True
        except Exception as e:
            logger.error(f"Timed out or error while waiting for response: {e}")
            await self.take_screenshot("wait_for_response_error")
            response = await self.get_last_response()
            return response is not None

    async def get_last_response(self) -> Optional[str]:
        """Get the last bot response text."""
        logger.info("Getting last response...")
        try:
            elements = await self.page.query_selector_all(self.SELECTORS["bot_message"])
            if elements:
                last = elements[-1]
                text = (await last.inner_text()).strip()
                logger.info(f"Response found: {text[:100]}{'...' if len(text) > 100 else ''}")
                return text
            logger.info("No response found")
        except Exception as e:
            logger.error(f"Error getting response: {e}")
        return None

    async def get_all_responses(self) -> List[str]:
        """Get all bot responses."""
        responses = []
        try:
            elements = await self.page.query_selector_all(self.SELECTORS["bot_message"])
            for el in elements:
                text = await el.inner_text()
                if text:
                    responses.append(text.strip())
            logger.info(f"Found {len(responses)} responses")
        except Exception as e:
            logger.error(f"Error getting responses: {e}")
        return responses

    @staticmethod
    def _parse_reference_titles_from_text(text: str) -> List[str]:
        """Parse 'References' section from a rendered response text."""
        if not text:
            return []
        t = re.sub(r"\r\n?", "\n", text).strip()
        if not t:
            return []
        idx = t.lower().rfind("references")
        if idx == -1:
            return []
        after = t[idx:]
        lines = [ln.strip() for ln in after.split("\n") if ln.strip()]
        titles: List[str] = []
        for ln in lines[1:]:
            m = re.match(r"^\s*(\d{1,2})\s*[:.\-]\s*(.+?)\s*$", ln)
            if m:
                titles.append(m.group(2).strip())
            elif titles and ln.lower() in {"sources", "source"}:
                break
        return titles

    async def open_sources_panel(self) -> bool:
        """Click the 'Sources' button when present."""
        try:
            btn_span = self.page.locator("button > span.text-sm", has_text=re.compile(r"^sources$", re.I))
            if await btn_span.count() > 0:
                count = await btn_span.count()
                logger.info("Sources button found via selector `button > span.text-sm` (count=%d)", count)
                await btn_span.click()
                logger.info("Clicked Sources button (span.text-sm)")
                return True

            btn = self.page.get_by_role("button", name=re.compile(r"^sources$", re.I))
            if await btn.count() == 0:
                logger.info("Sources button not found")
                return False
            logger.info("Sources button found via role lookup (count=%d)", await btn.count())
            await btn.first.click()
            logger.info("Clicked Sources button (role lookup)")
            return True
        except Exception:
            return False

    async def get_source_links(self) -> List[Dict[str, str]]:
        """Extract the reference/source links shown in the UI after opening Sources."""
        refs: List[Dict[str, str]] = []
        try:
            await self.page.wait_for_timeout(200)
            start = time.time()
            last_count = -1
            stable_hits = 0
            while (time.time() - start) < 5.0:
                try:
                    c = await self.page.locator(".min-w-0 a, .min-w-0 > a").count()
                except Exception:
                    c = 0
                if c > 0 and c == last_count:
                    stable_hits += 1
                    if stable_hits >= 2:
                        break
                else:
                    stable_hits = 0
                last_count = c
                await self.page.wait_for_timeout(250)

            container = None
            try:
                heading = self.page.get_by_text(re.compile(r"^references$", re.I))
                if await heading.count() > 0:
                    container = heading.first.locator(
                        "xpath=ancestor-or-self::*[self::section or self::div][1]"
                    )
            except Exception:
                container = None

            def _loc(sel: str):
                return (container.locator(sel) if container is not None else self.page.locator(sel))

            candidate_selectors = [".min-w-0 > a", ".min-w-0 a", "a[href^='http']"]

            links = None
            for sel in candidate_selectors:
                loc = _loc(sel)
                c = await loc.count()
                logger.info("Source link candidates via `%s`: %d", sel, c)
                if c > 0:
                    links = loc
                    break

            if links is None:
                return []

            count = await links.count()
            for i in range(count):
                a = links.nth(i)
                title = ((await a.inner_text()) or "").strip()
                url = ((await a.get_attribute("href")) or "").strip()
                if url or title:
                    refs.append({"title": title, "url": url})

            logger.info("Extracted %d source links", len(refs))
            for r in refs:
                logger.info("Source link | title=%s | url=%s", r.get("title") or "", r.get("url") or "")
        except Exception as e:
            logger.warning("Failed to extract source links: %s", e)
        return refs

    @staticmethod
    def _html_to_text(raw_html: str) -> str:
        """Lightweight HTML -> text conversion."""
        if not raw_html:
            return ""
        h = raw_html
        h = re.sub(r"(?is)<script.*?>.*?</script>", " ", h)
        h = re.sub(r"(?is)<style.*?>.*?</style>", " ", h)
        h = re.sub(r"(?is)<noscript.*?>.*?</noscript>", " ", h)
        h = re.sub(r"(?i)</(p|div|li|section|article|main|header|footer|br|h[1-6])\s*>", "\n", h)
        h = re.sub(r"(?s)<[^>]+>", " ", h)
        h = _html.unescape(h)
        h = re.sub(r"[ \t\r\f\v]+", " ", h)
        h = re.sub(r"\n\s*\n\s*", "\n\n", h)
        return h.strip()

    @classmethod
    def _extract_main_text_from_html(cls, raw_html: str) -> str:
        """Heuristic main-content extraction."""
        if not raw_html:
            return ""
        m = re.search(r"(?is)<main\b[^>]*>(.*?)</main>", raw_html)
        if m:
            return cls._html_to_text(m.group(1))
        a = re.search(r"(?is)<article\b[^>]*>(.*?)</article>", raw_html)
        if a:
            return cls._html_to_text(a.group(1))
        return cls._html_to_text(raw_html)

    async def _extract_main_text_from_dom(self, page: Page) -> str:
        """Extract main content from a rendered page DOM."""
        try:
            for sel in ("main", "article", "body"):
                loc = page.locator(sel)
                if await loc.count() > 0:
                    txt = (await loc.first.inner_text()) or ""
                    txt = re.sub(r"\s+", " ", txt).strip()
                    if txt:
                        return txt
        except Exception:
            return ""
        return ""

    async def _fetch_source_api(self, url: str, timeout: int) -> Tuple[Optional[int], str]:
        """
        Fetch source content via API request (fast, no browser navigation).
        Returns (status_code, extracted_text).
        """
        try:
            resp = await self.page.request.get(
                url,
                timeout=timeout,
                headers=self.FETCH_HEADERS,
            )
            status = resp.status
            raw_html = await resp.text()
            extracted = self._extract_main_text_from_html(raw_html)
            extracted = re.sub(r"\s+", " ", extracted).strip()
            return status, extracted
        except Exception as e:
            logger.debug(f"API fetch exception: {url} | {e}")
            raise

    async def _fetch_source_browser(self, url: str, timeout: int) -> str:
        """
        Fetch source content via browser navigation (slower but more reliable).
        Opens a new tab, navigates to URL, extracts text, closes tab.
        """
        new_page = await self.page.context.new_page()
        try:
            # Set browser-like headers
            await new_page.set_extra_http_headers(self.FETCH_HEADERS)
            
            # Navigate with network idle to ensure content loads
            await new_page.goto(url, wait_until="domcontentloaded", timeout=timeout)
            
            # Wait a bit for dynamic content
            await new_page.wait_for_timeout(1000)
            
            # Try to wait for main content
            try:
                await new_page.wait_for_selector("main, article, body", state="visible", timeout=3000)
            except Exception:
                pass
            
            # Extract text
            extracted = await self._extract_main_text_from_dom(new_page)
            
            # If DOM extraction failed, try JavaScript evaluation
            if not extracted or len(extracted) < 50:
                try:
                    extracted = await new_page.evaluate("""
                        () => {
                            // Remove script, style, nav, footer elements
                            const remove = ['script', 'style', 'nav', 'footer', 'header', 'aside'];
                            remove.forEach(tag => {
                                document.querySelectorAll(tag).forEach(el => el.remove());
                            });
                            
                            // Try main content areas
                            const main = document.querySelector('main') || 
                                         document.querySelector('article') || 
                                         document.querySelector('[role="main"]') ||
                                         document.querySelector('.content') ||
                                         document.querySelector('#content') ||
                                         document.body;
                            
                            return main ? main.innerText : '';
                        }
                    """)
                    extracted = re.sub(r"\s+", " ", extracted or "").strip()
                except Exception:
                    pass
            
            return extracted
        finally:
            await new_page.close()

    async def _fetch_single_source(
        self,
        url: str,
        candidate_idx: int,
        total_candidates: int,
        timeout: int,
        max_chars: int,
    ) -> Tuple[str, str, Optional[str]]:
        """
        Fetch a single source with multiple fallback methods.
        
        Returns: (url, extracted_content, error_message or None)
        """
        # Method 1: API request (fast)
        logger.info(
            "Fetching source (API) | candidate=%d/%d | url=%s",
            candidate_idx, total_candidates, url
        )
        
        try:
            status, extracted = await self._fetch_source_api(url, timeout)
            if extracted and len(extracted) >= 50:
                logger.info(
                    "API fetch success | candidate=%d | status=%d | chars=%d",
                    candidate_idx, status, len(extracted)
                )
                return url, extracted[:max_chars], None
            else:
                logger.info(
                    "API fetch returned insufficient content | candidate=%d | status=%d | chars=%d",
                    candidate_idx, status, len(extracted) if extracted else 0
                )
        except Exception as api_err:
            logger.warning(
                "API fetch failed | candidate=%d | url=%s | err=%s",
                candidate_idx, url, str(api_err)[:200]
            )

        # Method 2: Browser navigation (slower but handles JavaScript, cookies, etc.)
        logger.info(
            "Fetching source (Browser fallback) | candidate=%d/%d | url=%s",
            candidate_idx, total_candidates, url
        )
        
        try:
            extracted = await self._fetch_source_browser(url, timeout)
            if extracted and len(extracted) >= 50:
                logger.info(
                    "Browser fetch success | candidate=%d | chars=%d",
                    candidate_idx, len(extracted)
                )
                return url, extracted[:max_chars], None
            else:
                logger.warning(
                    "Browser fetch returned insufficient content | candidate=%d | chars=%d",
                    candidate_idx, len(extracted) if extracted else 0
                )
        except Exception as browser_err:
            logger.warning(
                "Browser fetch failed | candidate=%d | url=%s | err=%s",
                candidate_idx, url, str(browser_err)[:200]
            )

        # All methods failed
        error_msg = f"All fetch methods failed for {url}"
        logger.warning(error_msg)
        return url, "", error_msg

    async def get_visible_references(self) -> List[Dict[str, str]]:
        """Extract visible references (title + optional url) from the UI."""
        refs: List[Dict[str, str]] = []
        refs = await self.get_source_links()
        if refs:
            return refs
        try:
            heading = self.page.get_by_text(re.compile(r"^references$", re.I))
            if await heading.count() > 0:
                container = heading.first.locator("xpath=ancestor-or-self::*[self::section or self::div][1]")
                links = container.locator("a")
                link_count = await links.count()
                if link_count > 0:
                    for i in range(link_count):
                        a = links.nth(i)
                        title = ((await a.inner_text()) or "").strip()
                        url = ((await a.get_attribute("href")) or "").strip()
                        if title:
                            refs.append({"title": title, "url": url})
                    if refs:
                        return refs
        except Exception:
            pass

        try:
            last = await self.get_last_response()
            titles = self._parse_reference_titles_from_text(last or "")
            for t in titles:
                refs.append({"title": t, "url": ""})
        except Exception:
            pass
        return refs

    async def build_combined_context_from_source_links(
        self,
        refs: List[Dict[str, str]],
        *,
        max_sources: int = 10,
        max_chars_per_source: int = 4000,
        navigation_timeout_ms: Optional[int] = None,
    ) -> Tuple[str, List[Dict[str, str]]]:
        """
        Fetch and concatenate page visible text from reference URLs.
        
        Returns:
            Tuple of (combined_context_string, fetch_results)
            where fetch_results is a list of dicts with url, success, chars, error
        """
        chunks: List[str] = []
        fetch_results: List[Dict[str, str]] = []
        timeout = navigation_timeout_ms or self.settings.element_timeout
        
        usable = [r for r in refs if (r.get("url") or "").strip().startswith("http")]
        total_candidates = min(len(usable), max_sources)
        
        logger.info(
            "Building combined context from %d usable source URLs (max=%d)",
            len(usable), max_sources
        )
        
        success_idx = 0
        for candidate_idx, r in enumerate(usable[:max_sources], start=1):
            url = r["url"].strip()
            title = (r.get("title") or "").strip()
            
            # Fetch with fallbacks
            _, extracted, error = await self._fetch_single_source(
                url=url,
                candidate_idx=candidate_idx,
                total_candidates=total_candidates,
                timeout=timeout,
                max_chars=max_chars_per_source,
            )
            
            # Track result
            fetch_result = {
                "url": url,
                "title": title,
                "success": bool(extracted),
                "chars": len(extracted) if extracted else 0,
                "error": error,
            }
            fetch_results.append(fetch_result)
            
            if not extracted:
                logger.warning(
                    "Skipping source candidate %d (no content extracted) | url=%s",
                    candidate_idx, url
                )
                continue
            
            success_idx += 1
            header = f"SOURCE {success_idx}: {title or '(no title)'} | {url}"
            chunk_full = f"{header}\n{extracted}"
            chunks.append(chunk_full)
            
            logger.info(
                "Fetched source %d (candidate=%d/%d) | url=%s | chars=%d",
                success_idx, candidate_idx, total_candidates, url, len(extracted)
            )
            logger.info("Scraped content for source %d:\n%s", success_idx, chunk_full)

        combined = "\n\n---\n\n".join(chunks)
        
        # Log summary
        successful = sum(1 for r in fetch_results if r["success"])
        failed = len(fetch_results) - successful
        logger.info(
            "Combined context built | successful=%d | failed=%d | total_chars=%d",
            successful, failed, len(combined)
        )
        
        if failed > 0:
            failed_urls = [r["url"] for r in fetch_results if not r["success"]]
            logger.warning("Failed to fetch sources: %s", failed_urls)
        
        if combined:
            logger.info("Combined context:\n%s", combined)
        
        return combined, fetch_results

    async def get_response_with_context(self) -> Dict:
        """
        Get the last response along with context from source links.
        
        Returns dict with:
            - response: The bot's response text
            - sources: List of source link dicts
            - context: Combined context string from sources
            - context_chunks: List of individual context chunks
            - fetch_results: Details about each fetch attempt
            - source_stats: Summary of fetch success/failure
        """
        response = await self.get_last_response()
        
        # Try to open sources panel and get links
        await self.open_sources_panel()
        sources = await self.get_source_links()
        
        # Build context from sources
        context = ""
        context_list = []
        fetch_results = []
        
        if sources:
            context, fetch_results = await self.build_combined_context_from_source_links(sources)
            if context:
                context_list = [context]
        
        # Calculate stats
        total_sources = len(sources)
        fetched_sources = sum(1 for r in fetch_results if r.get("success"))
        failed_sources = total_sources - fetched_sources
        
        return {
            "response": response,
            "sources": sources,
            "context": context_list,
            "fetch_results": fetch_results,
            "source_stats": {
                "total": total_sources,
                "fetched": fetched_sources,
                "failed": failed_sources,
                "coverage": fetched_sources / total_sources if total_sources > 0 else 0,
            },
        }

    async def has_error(self) -> bool:
        """Check if error message is displayed."""
        try:
            return await self.page.is_visible(self.SELECTORS["error_message"])
        except Exception:
            return False

    async def take_screenshot(self, name: str) -> str:
        """Take screenshot and return path."""
        path = self.settings.screenshot_dir / f"{name}.png"
        await self.page.screenshot(path=str(path))
        logger.info(f"Screenshot saved: {path}")
        return str(path)

    async def set_locale_in_local_storage(
        self,
        locale: LocaleValue,
        *,
        key: str = "locale",
        reload_page: bool = True,
    ) -> None:
        """Set locale ("en"/"ar") in browser localStorage."""
        await set_locale_in_local_storage(
            self.page,
            locale,
            key=key,
            apply_to_current_page=True,
            persist_for_future_navigations=True,
            reload_page=reload_page,
        )
