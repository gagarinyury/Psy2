"""
DeepSeek API client for reasoning and generation tasks.

Provides HTTP client with retries, timeouts and structured prompting.
"""

import logging
from typing import Any, Dict, List

import httpx
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential_jitter,
)

from app.core.settings import settings

logger = logging.getLogger(__name__)


class DeepSeekClient(httpx.AsyncClient):
    """
    Async HTTP client for DeepSeek API with built-in retries and timeouts.

    Extends httpx.AsyncClient with DeepSeek-specific configuration:
    - Authorization headers
    - Retry logic for timeouts and server errors
    - Chat completion methods for reasoning and generation
    """

    def __init__(self, **kwargs):
        """
        Initialize DeepSeek client with default configuration.

        Args:
            **kwargs: Additional arguments passed to httpx.AsyncClient
        """
        # Set default timeout from settings
        if "timeout" not in kwargs:
            kwargs["timeout"] = settings.DEEPSEEK_TIMEOUT_S

        # Set base URL
        if "base_url" not in kwargs:
            kwargs["base_url"] = settings.DEEPSEEK_BASE_URL

        # Set default headers
        headers = kwargs.get("headers", {})
        if settings.DEEPSEEK_API_KEY:
            headers["Authorization"] = (
                f"Bearer {settings.DEEPSEEK_API_KEY.get_secret_value()}"
            )
        headers["Content-Type"] = "application/json"
        kwargs["headers"] = headers

        super().__init__(**kwargs)

    def _should_retry(exception):
        """Определяет нужно ли ретраить исключение."""
        if isinstance(exception, httpx.ReadTimeout):
            return True
        if isinstance(exception, httpx.HTTPStatusError):
            # Ретраим только 429 и 5xx ошибки
            return (
                exception.response.status_code == 429
                or exception.response.status_code >= 500
            )
        return False

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential_jitter(initial=1, max=10, jitter=2),
        retry=_should_retry,
        before_sleep=lambda retry_state: logger.warning(
            f"DeepSeek API retry {retry_state.attempt_number}: {retry_state.outcome.exception()}"
        ),
    )
    async def _make_chat_request(
        self, model: str, messages: List[Dict[str, str]], **kwargs
    ) -> Dict[str, Any]:
        """
        Make a chat completion request with retry logic.

        Args:
            model: DeepSeek model name
            messages: Chat messages in OpenAI format
            **kwargs: Additional parameters for the API call

        Returns:
            API response as dict

        Raises:
            httpx.HTTPStatusError: On 4xx/5xx responses after retries
            httpx.ReadTimeout: On timeout after retries
        """
        payload = {"model": model, "messages": messages, **kwargs}

        logger.debug(f"DeepSeek API request to {model}", extra={"payload": payload})

        try:
            response = await self.post("/chat/completions", json=payload)
            response.raise_for_status()

            result = response.json()
            logger.debug("DeepSeek API response", extra={"response": result})

            return result

        except httpx.HTTPStatusError as e:
            # Don't retry on 4xx client errors (except 429)
            if e.response.status_code < 500 and e.response.status_code != 429:
                logger.error(
                    f"DeepSeek API client error {e.response.status_code}: {e.response.text}"
                )
                raise
            else:
                # Retry on 429 (rate limit) and 5xx (server errors)
                logger.warning(
                    f"DeepSeek API error {e.response.status_code}, retrying..."
                )
                raise

    async def reasoning(
        self, messages: List[Dict[str, str]], **kwargs
    ) -> Dict[str, Any]:
        """
        Perform reasoning task using DeepSeek reasoning model.

        Args:
            messages: Chat messages with system prompt and user input
            **kwargs: Additional parameters (temperature, max_tokens, etc.)

        Returns:
            API response containing reasoning result
        """
        return await self._make_chat_request(
            model=settings.DEEPSEEK_REASONING_MODEL, messages=messages, **kwargs
        )

    async def generate(
        self, messages: List[Dict[str, str]], **kwargs
    ) -> Dict[str, Any]:
        """
        Perform text generation using DeepSeek base model.

        Args:
            messages: Chat messages with system prompt and user input
            **kwargs: Additional parameters (temperature, max_tokens, etc.)

        Returns:
            API response containing generated text
        """
        return await self._make_chat_request(
            model=settings.DEEPSEEK_BASE_MODEL, messages=messages, **kwargs
        )

    async def __aenter__(self):
        """Async context manager entry."""
        await super().__aenter__()
        return self

    async def __aexit__(self, *args):
        """Async context manager exit."""
        await super().__aexit__(*args)
