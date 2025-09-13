import json
import pytest
import httpx
from unittest.mock import patch, MagicMock

from app.llm.deepseek_client import DeepSeekClient
from app.core.settings import settings


@pytest.mark.anyio
async def test_deepseek_client_initialization():
    """
    Тест инициализации DeepSeekClient с правильными настройками.
    """
    # Mock API key для тестирования
    with patch.object(settings, "DEEPSEEK_API_KEY") as mock_key:
        mock_key.get_secret_value.return_value = "test-api-key"

        async with DeepSeekClient() as client:
            # Проверяем base_url (с учетом trailing slash)
            assert str(client.base_url).rstrip(
                "/"
            ) == settings.DEEPSEEK_BASE_URL.rstrip("/")

            # Проверяем заголовки
            headers = client.headers
            assert headers["Authorization"] == "Bearer test-api-key"
            assert headers["Content-Type"] == "application/json"

            # Проверяем timeout
            assert client.timeout.read == settings.DEEPSEEK_TIMEOUT_S


@pytest.mark.anyio
async def test_reasoning_method():
    """
    Тест метода reasoning с mock HTTP запросом.
    """
    # Mock ответ от API
    mock_response = {
        "choices": [
            {
                "message": {
                    "content": json.dumps(
                        {
                            "content_plan": ["test reasoning result"],
                            "style_directives": {"tempo": "medium", "length": "short"},
                            "state_updates": {
                                "trust_delta": 0.1,
                                "fatigue_delta": 0.05,
                            },
                            "telemetry": {"chosen_ids": ["test-id"]},
                        }
                    )
                }
            }
        ]
    }

    # Тестовые сообщения
    test_messages = [
        {"role": "system", "content": "Test system prompt"},
        {"role": "user", "content": "Test user input"},
    ]

    with patch.object(settings, "DEEPSEEK_API_KEY") as mock_key:
        mock_key.get_secret_value.return_value = "test-api-key"

        client = DeepSeekClient()

        # Mock успешный HTTP post запрос
        with patch.object(client, "post") as mock_post:
            mock_response_obj = MagicMock()
            mock_response_obj.raise_for_status = MagicMock()
            mock_response_obj.json = MagicMock(return_value=mock_response)
            mock_post.return_value = mock_response_obj

            # Вызываем reasoning
            result = await client.reasoning(test_messages, temperature=0.7)

            # Проверяем что post был вызван с правильными параметрами
            mock_post.assert_called_once_with(
                "/chat/completions",
                json={
                    "model": settings.DEEPSEEK_REASONING_MODEL,
                    "messages": test_messages,
                    "temperature": 0.7,
                },
            )

            # Проверяем результат
            assert result == mock_response

        await client.aclose()


@pytest.mark.anyio
async def test_generate_method():
    """
    Тест метода generate с mock HTTP запросом.
    """
    # Mock ответ от API
    mock_response = {
        "choices": [{"message": {"content": "Generated patient response here"}}]
    }

    # Тестовые сообщения
    test_messages = [
        {"role": "system", "content": "Generation prompt"},
        {"role": "user", "content": "Content plan data"},
    ]

    with patch.object(settings, "DEEPSEEK_API_KEY") as mock_key:
        mock_key.get_secret_value.return_value = "test-api-key"

        client = DeepSeekClient()

        # Mock успешный HTTP post запрос
        with patch.object(client, "post") as mock_post:
            mock_response_obj = MagicMock()
            mock_response_obj.raise_for_status = MagicMock()
            mock_response_obj.json = MagicMock(return_value=mock_response)
            mock_post.return_value = mock_response_obj

            # Вызываем generate
            result = await client.generate(test_messages, max_tokens=150)

            # Проверяем что post был вызван с правильными параметрами
            mock_post.assert_called_once_with(
                "/chat/completions",
                json={
                    "model": settings.DEEPSEEK_BASE_MODEL,
                    "messages": test_messages,
                    "max_tokens": 150,
                },
            )

            # Проверяем результат
            assert result == mock_response

        await client.aclose()


@pytest.mark.anyio
async def test_retry_logic_with_mocked_make_request():
    """
    Тест логики ретраев путем мокинга _make_chat_request.
    """
    test_messages = [{"role": "user", "content": "test"}]
    success_response = {"choices": [{"message": {"content": "success"}}]}

    with patch.object(settings, "DEEPSEEK_API_KEY") as mock_key:
        mock_key.get_secret_value.return_value = "test-api-key"

        client = DeepSeekClient()

        # Mock _make_chat_request для успешного выполнения после ретраев
        with patch.object(client, "_make_chat_request") as mock_make_request:
            mock_make_request.return_value = success_response

            result = await client.reasoning(test_messages)

            # Проверяем что метод был вызван
            mock_make_request.assert_called_once_with(
                model=settings.DEEPSEEK_REASONING_MODEL, messages=test_messages
            )

            assert result == success_response

        await client.aclose()


@pytest.mark.anyio
async def test_http_status_error_handling():
    """
    Тест обработки HTTP ошибок.
    """
    test_messages = [{"role": "user", "content": "test"}]

    with patch.object(settings, "DEEPSEEK_API_KEY") as mock_key:
        mock_key.get_secret_value.return_value = "test-api-key"

        client = DeepSeekClient()

        # Mock post для возвращения 400 ошибки
        with patch.object(client, "post") as mock_post:
            error_response = MagicMock()
            error_response.status_code = 400
            error_response.text = "Bad Request"

            http_error = httpx.HTTPStatusError(
                "Bad Request", request=MagicMock(), response=error_response
            )
            mock_post.side_effect = http_error

            # Должна подняться исключение
            with pytest.raises(httpx.HTTPStatusError):
                await client.reasoning(test_messages)

        await client.aclose()


@pytest.mark.anyio
async def test_timeout_error_handling():
    """
    Тест обработки таймаутов.
    """
    test_messages = [{"role": "user", "content": "test"}]

    with patch.object(settings, "DEEPSEEK_API_KEY") as mock_key:
        mock_key.get_secret_value.return_value = "test-api-key"

        client = DeepSeekClient()

        # Mock post для возвращения timeout
        with patch.object(client, "post") as mock_post:
            mock_post.side_effect = httpx.ReadTimeout("Request timeout")

            # Должна подняться исключение после всех ретраев
            with pytest.raises(httpx.ReadTimeout):
                await client.reasoning(test_messages)

        await client.aclose()


@pytest.mark.anyio
async def test_client_without_api_key():
    """
    Тест что клиент работает без API ключа (для тестирования).
    """
    with patch.object(settings, "DEEPSEEK_API_KEY", None):
        async with DeepSeekClient() as client:
            # Проверяем что Authorization заголовок не установлен
            assert "Authorization" not in client.headers
            assert client.headers["Content-Type"] == "application/json"


@pytest.mark.anyio
async def test_successful_request_structure():
    """
    Тест структуры успешного запроса без ретраев.
    """
    test_messages = [{"role": "user", "content": "test input"}]
    expected_response = {
        "choices": [{"message": {"content": "model response"}}],
        "model": settings.DEEPSEEK_REASONING_MODEL,
        "usage": {"total_tokens": 50},
    }

    with patch.object(settings, "DEEPSEEK_API_KEY") as mock_key:
        mock_key.get_secret_value.return_value = "test-key"

        client = DeepSeekClient()

        # Mock успешный post запрос
        with patch.object(client, "post") as mock_post:
            mock_response = MagicMock()
            mock_response.raise_for_status = MagicMock()
            mock_response.json = MagicMock(return_value=expected_response)
            mock_post.return_value = mock_response

            result = await client.reasoning(test_messages, max_tokens=100)

            # Проверяем структуру запроса
            mock_post.assert_called_once_with(
                "/chat/completions",
                json={
                    "model": settings.DEEPSEEK_REASONING_MODEL,
                    "messages": test_messages,
                    "max_tokens": 100,
                },
            )

            # Проверяем ответ
            assert result == expected_response
            assert result["choices"][0]["message"]["content"] == "model response"

        await client.aclose()
