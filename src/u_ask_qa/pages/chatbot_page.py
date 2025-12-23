"""
Page Object Model for GovGPT Chatbot.
"""

import asyncio
import logging
import time
from typing import List, Optional

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
        "all_messages": ".user-message [dir=ltr],#response-content-container",
        "loading_indicator": "button svg path[d*='M12.002 14.4']",
        "error_message": ".error-message, .error",
        # Suggestion chips (best-effort; UI can vary)
        "suggestion_chips": (
            "button.suggestion-chip, button[class*='suggest'], button[class*='chip'], "
            "[role='button'][class*='suggest'], [role='button'][class*='chip']"
        ),
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
            # Check what elements are visible
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
        """
        Check if the chat UI is ready for interaction.

        Used by UI smoke tests to ensure the widget + input are visible and any
        initial loading spinner has disappeared.
        """
        logger.info("Checking if chat widget is ready...")
        try:
            # Wait for any global loading spinner to disappear if present.
            try:
                await self.page.wait_for_selector(
                    self.SELECTORS["loading_screen"],
                    state="hidden",
                    timeout=self.settings.element_timeout,
                )
            except Exception:
                # Spinner might not exist in all states; ignore.
                pass

            widget_ok = await self.is_widget_loaded()
            input_ok = await self.is_input_ready()

            if not widget_ok or not input_ok:
                await self._log_page_state()
                return False

            # If an error banner is already visible, treat as not ready.
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
            # Focus and type
            logger.info("Clicking input...")
            input_el = await self.page.wait_for_selector(
                self.SELECTORS["message_input"], state="visible"
            )
            await input_el.click()
            logger.info("Input focused")

            logger.info("Typing message...")
            await self.page.keyboard.type(message)
            logger.info("Message typed")

            # Send
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
        """
        Waits for the chatbot to finish generating a response by polling
        the browser's DOM for the transition from 'Stop' icon to 'Send' button.
        """
        target_timeout = timeout or self.settings.response_timeout
        logger.info(f"Waiting for AI response completion | Timeout: {target_timeout}ms")
        start_time = time.time()
        try:
            # 1. Brief wait to see if the processing state (Stop button) appears.
            # We don't want to fail if the response is instant, so we use a short timeout.
            stop_selector = self.SELECTORS["loading_indicator"]
            send_button = self.SELECTORS["send_button"]
            try:
                await self.page.wait_for_selector(
                    stop_selector, state="visible", timeout=2000
                )
                logger.info(
                    "Processing detected: AI is currently streaming response..."
                )
            except Exception:
                logger.info(
                    "ℹInitial processing icon not detected; checking if already finished."
                )

            # 2. High-performance polling inside the browser.
            # This function returns True only when the Stop icon is gone AND the Send button is back.
            await self.page.wait_for_function(
                f"""
                () => {{
                    const stopIcon = document.querySelector("{stop_selector}");
                    const sendButton = document.querySelector("{send_button}");
                    // Condition: Stop icon must be absent AND Send button must be present
                    return !stopIcon && sendButton !== null;
                }}
                """,
                timeout=target_timeout,
            )
            await asyncio.sleep(0.8)
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
                logger.info(
                    f"Response found: {text[:100]}{'...' if len(text) > 100 else ''}"
                )
                return text
            logger.info("ℹ No response found")
        except Exception as e:
            logger.error(f"✗ Error getting response: {e}")
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
        """
        Set locale ("en"/"ar") in browser localStorage.

        By default we reload the page so the app can pick up the new locale.
        """
        await set_locale_in_local_storage(
            self.page,
            locale,
            key=key,
            apply_to_current_page=True,
            persist_for_future_navigations=True,
            reload_page=reload_page,
        )
