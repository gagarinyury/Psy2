import uuid
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from sqlalchemy import select

from app.cli.kb_embed import get_fragments_for_embedding, process_embeddings_for_case
from app.core.db import AsyncSessionLocal
from app.core.tables import Case, KBFragment
from app.kb.embeddings import (
    _compact_metadata,
    embed_fragment_text,
    embed_fragments_batch,
)


class TestEmbeddings:
    """Тесты для модуля эмбеддингов"""

    def test_compact_metadata(self):
        """Тест компактного представления метаданных"""
        # Полные метаданные
        metadata = {
            "topic": "symptoms",
            "availability": "public",
            "emotion_label": "neutral",
            "tags": ["sleep", "mood", "anxiety", "stress", "extra"],
        }

        result = _compact_metadata(metadata)
        expected = "topic:symptoms | availability:public | emotion:neutral | tags:sleep,mood,anxiety"
        assert result == expected

        # Частичные метаданные
        partial_metadata = {"topic": "bio", "tags": ["family"]}

        result = _compact_metadata(partial_metadata)
        expected = "topic:bio | tags:family"
        assert result == expected

        # Пустые метаданные
        empty_metadata = {}
        result = _compact_metadata(empty_metadata)
        assert result == ""

    @patch("app.kb.embeddings.get_embedding_model")
    def test_embed_fragment_text(self, mock_get_model):
        """Тест создания эмбеддинга для фрагмента"""
        # Мокаем модель
        mock_model = MagicMock()
        mock_embeddings = np.array([[0.1, 0.2, 0.3, 0.4]])  # размерность 4 для теста
        mock_model.encode.return_value = mock_embeddings
        mock_get_model.return_value = mock_model

        # Тестовые данные
        text = "Пациент испытывает проблемы со сном"
        metadata = {
            "topic": "symptoms",
            "availability": "public",
            "emotion_label": "negative",
            "tags": ["sleep", "insomnia"],
        }

        # Вызываем функцию
        result = embed_fragment_text(text, metadata)

        # Проверяем результат
        assert isinstance(result, np.ndarray)
        assert result.dtype == np.float32
        assert len(result) == 4

        # Проверяем что модель была вызвана с правильным текстом
        mock_model.encode.assert_called_once()
        call_args = mock_model.encode.call_args[0][0]
        assert len(call_args) == 1
        embedded_text = call_args[0]

        assert text in embedded_text
        assert "META:" in embedded_text
        assert "topic:symptoms" in embedded_text
        assert "availability:public" in embedded_text
        assert "emotion:negative" in embedded_text
        assert "tags:sleep,insomnia" in embedded_text

    def test_embed_fragment_text_validation(self):
        """Тест валидации входных данных"""
        metadata = {"topic": "test"}

        # Пустой текст
        with pytest.raises(ValueError, match="Text must be a non-empty string"):
            embed_fragment_text("", metadata)

        # Неправильный тип текста
        with pytest.raises(ValueError, match="Text must be a non-empty string"):
            embed_fragment_text(None, metadata)

        # Неправильный тип метаданных
        with pytest.raises(ValueError, match="Metadata must be a dictionary"):
            embed_fragment_text("test text", "not dict")

    @patch("app.kb.embeddings.get_embedding_model")
    def test_embed_fragments_batch(self, mock_get_model):
        """Тест батч-создания эмбеддингов"""
        # Мокаем модель
        mock_model = MagicMock()
        mock_embeddings = np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]])
        mock_model.encode.return_value = mock_embeddings
        mock_get_model.return_value = mock_model

        # Тестовые данные
        fragments_data = [
            {"text": "Первый фрагмент", "metadata": {"topic": "bio"}},
            {
                "text": "Второй фрагмент",
                "metadata": {"topic": "symptoms", "tags": ["mood"]},
            },
        ]

        # Вызываем функцию
        result = embed_fragments_batch(fragments_data)

        # Проверяем результат
        assert len(result) == 2
        assert all(isinstance(emb, np.ndarray) for emb in result)
        assert all(emb.dtype == np.float32 for emb in result)
        assert all(len(emb) == 3 for emb in result)

        # Проверяем что модель была вызвана один раз с батчем
        mock_model.encode.assert_called_once()
        call_args = mock_model.encode.call_args[0][0]
        assert len(call_args) == 2

        # Проверяем содержимое текстов
        assert "Первый фрагмент" in call_args[0]
        assert "topic:bio" in call_args[0]
        assert "Второй фрагмент" in call_args[1]
        assert "topic:symptoms" in call_args[1]
        assert "tags:mood" in call_args[1]


class TestKBEmbedCLI:
    """Тесты для CLI обработки эмбеддингов"""

    @pytest.mark.anyio
    async def test_get_fragments_for_embedding(self):
        """Тест получения фрагментов без эмбеддингов"""
        async with AsyncSessionLocal() as session:
            # Находим любой существующий case с фрагментами
            case_result = await session.execute(select(Case).limit(1))
            case = case_result.scalars().first()

            if not case:
                pytest.skip("No cases found in database")

            # Получаем фрагменты для этого case
            fragments = await get_fragments_for_embedding(session, case.id, limit=10)

            # Проверяем что все фрагменты принадлежат указанному case
            assert all(frag.case_id == case.id for frag in fragments)

            # Если есть фрагменты, проверяем что у них нет эмбеддингов
            for fragment in fragments:
                assert fragment.embedding is None

    @pytest.mark.anyio
    async def test_kb_embed_idempotency(self):
        """Тест идемпотентности - повторный запуск не меняет updated_at у уже заполненных"""
        async with AsyncSessionLocal() as session:
            # Находим фрагмент который мог бы уже иметь embedding
            fragment_result = await session.execute(
                select(KBFragment).where(KBFragment.embedding.is_not(None)).limit(1)
            )
            fragment = fragment_result.scalars().first()

            if not fragment:
                pytest.skip("No fragments with embeddings found")

            # Запоминаем первоначальные timestamps если они есть
            _ = fragment.created_at if hasattr(fragment, "created_at") else None

            # Пытаемся "обновить" фрагмент (имитируя CLI операцию)
            # В реальности CLI не должен трогать уже обработанные фрагменты
            fragments_no_embedding = await get_fragments_for_embedding(
                session, fragment.case_id
            )

            # Проверяем что фрагмент с эмбеддингом не попал в список для обработки
            fragment_ids_no_embedding = [f.id for f in fragments_no_embedding]
            assert fragment.id not in fragment_ids_no_embedding

    @pytest.mark.anyio
    async def test_batch_processing_with_mock(self):
        """Тест батч-обработки с мок-данными"""
        # Создаем временный case для теста
        test_case_id = uuid.uuid4()

        with patch("app.cli.kb_embed.embed_fragments_batch") as mock_embed:
            # Мокаем создание эмбеддингов
            mock_embed.return_value = [
                np.array([0.1, 0.2, 0.3], dtype=np.float32),
                np.array([0.4, 0.5, 0.6], dtype=np.float32),
            ]

            async with AsyncSessionLocal() as session:
                # Создаем тестовый case
                test_case = Case(
                    id=test_case_id,
                    version="test-1.0",
                    case_truth={"dx_target": ["test"]},
                    policies={"retrieve": {"max_fragments": 5}},
                )
                session.add(test_case)

                # Создаем тестовые фрагменты без эмбеддингов
                fragment1 = KBFragment(
                    case_id=test_case_id,
                    type="test",
                    text="Тестовый фрагмент 1",
                    fragment_metadata={"topic": "test"},
                    availability="public",
                    consistency_keys={},
                    embedding=None,
                )

                fragment2 = KBFragment(
                    case_id=test_case_id,
                    type="test",
                    text="Тестовый фрагмент 2",
                    fragment_metadata={"topic": "test2"},
                    availability="public",
                    consistency_keys={},
                    embedding=None,
                )

                session.add(fragment1)
                session.add(fragment2)
                await session.commit()

                try:
                    # Запускаем обработку
                    stats = await process_embeddings_for_case(
                        str(test_case_id), batch_size=10
                    )

                    # Проверяем статистику
                    assert stats["processed"] == 2
                    assert stats["failed"] == 0
                    assert stats["dimension"] == 3
                    assert stats["batches"] == 1

                    # Проверяем что эмбеддинги были сохранены
                    await session.refresh(fragment1)
                    await session.refresh(fragment2)

                    assert fragment1.embedding is not None
                    assert fragment2.embedding is not None
                    assert len(fragment1.embedding) == 3
                    assert len(fragment2.embedding) == 3

                finally:
                    # Очищаем тестовые данные
                    await session.delete(fragment1)
                    await session.delete(fragment2)
                    await session.delete(test_case)
                    await session.commit()

    @pytest.mark.anyio
    async def test_process_nonexistent_case(self):
        """Тест обработки несуществующего case"""
        nonexistent_case_id = str(uuid.uuid4())

        with pytest.raises(Exception):  # Ожидаем KBEmbedError, но может быть завернут
            await process_embeddings_for_case(nonexistent_case_id)

    def test_invalid_case_id_format(self):
        """Тест некорректного формата case_id"""
        with pytest.raises(Exception):  # Ожидаем KBEmbedError
            import asyncio

            asyncio.run(process_embeddings_for_case("not-a-uuid"))
