import pytest
from unittest.mock import patch, MagicMock, AsyncMock

from app.orchestrator.nodes.generate_llm import generate_llm, _create_fallback_response


@pytest.mark.anyio
async def test_generate_llm_success():
    """
    Тест успешной генерации через DeepSeek API.
    """
    # Mock данные для входа
    content_plan = ["I've been feeling quite anxious lately", "Sleep hasn't been great"]

    style_directives = {"tempo": "calm", "length": "medium"}

    patient_context = "Patient with generalized anxiety disorder"

    # Mock успешный ответ от DeepSeek
    mock_response = {
        "choices": [
            {
                "message": {
                    "content": "I've been really struggling with anxiety recently, and it's been affecting my sleep too."
                }
            }
        ]
    }

    # Mock DeepSeekClient
    with patch(
        "app.orchestrator.nodes.generate_llm.DeepSeekClient"
    ) as mock_client_class:
        mock_client = MagicMock()
        mock_client.generate = AsyncMock(return_value=mock_response)
        mock_client_class.return_value.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client_class.return_value.__aexit__ = AsyncMock(return_value=None)

        result = await generate_llm(content_plan, style_directives, patient_context)

        # Проверяем что получили строку
        assert isinstance(result, str)
        assert len(result) > 0
        assert (
            result
            == "I've been really struggling with anxiety recently, and it's been affecting my sleep too."
        )

        # Проверяем что API был вызван правильно
        mock_client.generate.assert_called_once()
        call_args = mock_client.generate.call_args[0][0]  # messages argument
        assert len(call_args) == 2
        assert call_args[0]["role"] == "system"
        assert call_args[1]["role"] == "user"

        # Проверяем параметры вызова
        call_kwargs = mock_client.generate.call_args[1]
        assert call_kwargs["temperature"] == 0.7
        assert call_kwargs["max_tokens"] == 200


@pytest.mark.anyio
async def test_generate_llm_empty_response_fallback():
    """
    Тест fallback при пустом ответе от DeepSeek.
    """
    content_plan = ["I feel confused"]
    style_directives = {"tempo": "medium", "length": "short"}

    # Mock пустой ответ
    mock_response = {"choices": []}

    with patch(
        "app.orchestrator.nodes.generate_llm.DeepSeekClient"
    ) as mock_client_class:
        mock_client = MagicMock()
        mock_client.generate = AsyncMock(return_value=mock_response)
        mock_client_class.return_value.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client_class.return_value.__aexit__ = AsyncMock(return_value=None)

        result = await generate_llm(content_plan, style_directives)

        # Проверяем fallback response
        assert result == "I feel confused"


@pytest.mark.anyio
async def test_generate_llm_empty_content_fallback():
    """
    Тест fallback при пустом контенте от DeepSeek.
    """
    content_plan = ["I'm feeling lost", "Don't know what to say"]
    style_directives = {"tempo": "calm", "length": "medium"}

    # Mock ответ с пустым содержимым
    mock_response = {
        "choices": [
            {
                "message": {
                    "content": "   "  # Только пробелы
                }
            }
        ]
    }

    with patch(
        "app.orchestrator.nodes.generate_llm.DeepSeekClient"
    ) as mock_client_class:
        mock_client = MagicMock()
        mock_client.generate = AsyncMock(return_value=mock_response)
        mock_client_class.return_value.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client_class.return_value.__aexit__ = AsyncMock(return_value=None)

        result = await generate_llm(content_plan, style_directives)

        # Проверяем fallback response
        assert result == "I'm feeling lost Don't know what to say"


@pytest.mark.anyio
async def test_generate_llm_api_exception_fallback():
    """
    Тест fallback при исключении в API.
    """
    content_plan = ["Something went wrong"]
    style_directives = {"tempo": "medium", "length": "short"}

    # Mock исключение при вызове API
    with patch(
        "app.orchestrator.nodes.generate_llm.DeepSeekClient"
    ) as mock_client_class:
        mock_client = MagicMock()
        mock_client.generate = AsyncMock(side_effect=Exception("API connection failed"))
        mock_client_class.return_value.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client_class.return_value.__aexit__ = AsyncMock(return_value=None)

        result = await generate_llm(content_plan, style_directives)

        # Проверяем fallback response
        assert result == "Something went wrong"


@pytest.mark.anyio
async def test_generate_llm_quoted_response():
    """
    Тест удаления кавычек из ответа.
    """
    content_plan = ["I need help"]
    style_directives = {"tempo": "medium", "length": "short"}

    # Mock ответ с кавычками
    mock_response = {
        "choices": [{"message": {"content": '"I really need some help with this."'}}]
    }

    with patch(
        "app.orchestrator.nodes.generate_llm.DeepSeekClient"
    ) as mock_client_class:
        mock_client = MagicMock()
        mock_client.generate = AsyncMock(return_value=mock_response)
        mock_client_class.return_value.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client_class.return_value.__aexit__ = AsyncMock(return_value=None)

        result = await generate_llm(content_plan, style_directives)

        # Проверяем что кавычки удалены
        assert result == "I really need some help with this."
        assert not result.startswith('"')
        assert not result.endswith('"')


@pytest.mark.anyio
async def test_generate_llm_length_short():
    """
    Тест ограничения длины для короткого ответа.
    """
    content_plan = ["Multiple points"]
    style_directives = {"tempo": "medium", "length": "short"}

    # Mock ответ с несколькими предложениями
    mock_response = {
        "choices": [
            {
                "message": {
                    "content": "I have many things to say about this. There are multiple issues. It's quite complex."
                }
            }
        ]
    }

    with patch(
        "app.orchestrator.nodes.generate_llm.DeepSeekClient"
    ) as mock_client_class:
        mock_client = MagicMock()
        mock_client.generate = AsyncMock(return_value=mock_response)
        mock_client_class.return_value.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client_class.return_value.__aexit__ = AsyncMock(return_value=None)

        result = await generate_llm(content_plan, style_directives)

        # Проверяем что оставлено только первое предложение
        assert result == "I have many things to say about this."
        assert result.count(".") == 1


@pytest.mark.anyio
async def test_generate_llm_length_long():
    """
    Тест ограничения длины для длинного ответа.
    """
    content_plan = ["Very detailed explanation needed"]
    style_directives = {"tempo": "medium", "length": "long"}

    # Mock ответ с множеством предложений
    mock_response = {
        "choices": [
            {
                "message": {
                    "content": "First sentence. Second sentence. Third sentence. Fourth sentence. Fifth sentence."
                }
            }
        ]
    }

    with patch(
        "app.orchestrator.nodes.generate_llm.DeepSeekClient"
    ) as mock_client_class:
        mock_client = MagicMock()
        mock_client.generate = AsyncMock(return_value=mock_response)
        mock_client_class.return_value.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client_class.return_value.__aexit__ = AsyncMock(return_value=None)

        result = await generate_llm(content_plan, style_directives)

        # Проверяем что оставлено максимум 3 предложения
        sentences = result.split(".")
        valid_sentences = [s.strip() for s in sentences if s.strip()]
        assert len(valid_sentences) <= 3


@pytest.mark.anyio
async def test_generate_llm_no_patient_context():
    """
    Тест работы без patient_context.
    """
    content_plan = ["Default context test"]
    style_directives = {"tempo": "medium", "length": "medium"}

    # Mock ответ
    mock_response = {
        "choices": [
            {
                "message": {
                    "content": "This is a test response without specific context."
                }
            }
        ]
    }

    with patch(
        "app.orchestrator.nodes.generate_llm.DeepSeekClient"
    ) as mock_client_class:
        mock_client = MagicMock()
        mock_client.generate = AsyncMock(return_value=mock_response)
        mock_client_class.return_value.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client_class.return_value.__aexit__ = AsyncMock(return_value=None)

        # Вызываем без patient_context
        result = await generate_llm(content_plan, style_directives)

        assert isinstance(result, str)
        assert len(result) > 0

        # Проверяем что в запросе был использован дефолтный контекст
        call_args = mock_client.generate.call_args[0][0]  # messages
        user_message = call_args[1]["content"]
        assert "General therapy patient" in user_message


def test_create_fallback_response():
    """
    Тест создания fallback ответа.
    """
    # Тест с пустым content_plan
    result = _create_fallback_response([])
    assert result == "I'm not sure how to respond right now."

    # Тест с одним элементом
    result = _create_fallback_response(["Single point"])
    assert result == "Single point"

    # Тест с множественными элементами - должны взяться первые 2
    result = _create_fallback_response(["First", "Second", "Third", "Fourth"])
    assert result == "First Second"

    # Тест с двумя элементами
    result = _create_fallback_response(["Point A", "Point B"])
    assert result == "Point A Point B"
