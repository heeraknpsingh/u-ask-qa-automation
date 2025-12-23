"""
Login Tests for GovGPT Chatbot.
"""

# pylint: disable=import-error,too-few-public-methods

import pytest
from playwright.async_api import TimeoutError as PlaywrightTimeoutError
from u_ask_qa.pages import ChatbotPage
from u_ask_qa.core.config import get_settings


class TestLogin:
    @pytest.mark.login
    @pytest.mark.smoke
    async def test_login_page_loads(self, chatbot_page_no_login: ChatbotPage):
        page = chatbot_page_no_login.page
        login_screen = await page.wait_for_selector(
            ChatbotPage.SELECTORS["login_screen"],
            state="visible",
            timeout=get_settings().element_timeout,
        )
        assert login_screen is not None, "Login screen should be visible"

    @pytest.mark.login
    @pytest.mark.smoke
    async def test_login_success(self, chatbot_page_no_login: ChatbotPage):
        logged_in = await chatbot_page_no_login.login()
        assert logged_in, "Login should succeed with valid credentials"
        is_logged = await chatbot_page_no_login.is_logged_in()
        assert is_logged, "User profile should be visible after login"

    @pytest.mark.login
    async def test_user_profile_visible_after_login(self, chatbot_page: ChatbotPage):
        is_logged = await chatbot_page.is_logged_in()
        assert is_logged, "User profile should be visible"

    @pytest.mark.login
    async def test_chat_input_available_after_login(self, chatbot_page: ChatbotPage):
        is_ready = await chatbot_page.is_input_ready()
        assert is_ready, "Chat input should be available after login"

    @pytest.mark.login
    async def test_can_send_message_after_login(self, chatbot_page: ChatbotPage):
        await chatbot_page.is_input_ready()
        success = await chatbot_page.send_message("Hello")
        assert success, "Should be able to send message after login"
        received = await chatbot_page.wait_for_response()
        assert received, "Should receive response"
        response = await chatbot_page.get_last_response()
        assert response, "Response should not be empty"


class TestLoginFields:
    """Tests for login form fields."""

    @pytest.mark.login
    async def test_login_fields_visible(self, chatbot_page_no_login: ChatbotPage):
        """Verify login form fields are visible after opening the login form."""
        page = chatbot_page_no_login.page
        # Click login link
        await page.click(ChatbotPage.SELECTORS["log_in_link"])
        # Check email field
        try:
            await page.wait_for_selector(
                ChatbotPage.SELECTORS["email_input"],
                state="visible",
                timeout=get_settings().element_timeout,
            )
        except PlaywrightTimeoutError as e:
            pytest.fail(f"Email input should be visible (timed out): {e}")
        # Check password field
        try:
            await page.wait_for_selector(
                ChatbotPage.SELECTORS["password_input"],
                state="visible",
                timeout=get_settings().element_timeout,
            )
        except PlaywrightTimeoutError as e:
            pytest.fail(f"Password input should be visible (timed out): {e}")
        # Check login button
        try:
            await page.wait_for_selector(
                ChatbotPage.SELECTORS["log_in_button"],
                state="visible",
                timeout=get_settings().element_timeout,
            )
        except PlaywrightTimeoutError as e:
            pytest.fail(f"Login button should be visible (timed out): {e}")
