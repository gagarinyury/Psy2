"""
Провайдер эмбеддингов для knowledge base фрагментов.

Используется для создания векторных представлений KB фрагментов
с учетом их метаданных.
"""

import uuid
import numpy as np
from typing import Dict, Any, List, Tuple
from functools import lru_cache

from sentence_transformers import SentenceTransformer
from sqlalchemy import select, update, func
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.db import AsyncSessionLocal
from app.core.tables import KBFragment
from app.infra.logging import get_logger

logger = get_logger()


@lru_cache(maxsize=1)
def get_embedding_model() -> SentenceTransformer:
    """
    Получает модель для создания эмбеддингов (кешируется).

    Returns:
        SentenceTransformer: Предобученная модель bge-m3
    """
    model_name = "BAAI/bge-m3"
    logger.info("Loading embedding model", model=model_name)

    try:
        model = SentenceTransformer(model_name)
        logger.info(
            "Embedding model loaded successfully",
            model=model_name,
            max_seq_length=model.max_seq_length,
        )
        return model
    except Exception as e:
        logger.error("Failed to load embedding model", model=model_name, error=str(e))
        raise


def _compact_metadata(metadata: Dict[str, Any]) -> str:
    """
    Создает компактное представление метаданных для включения в эмбеддинг.

    Args:
        metadata: Словарь метаданных фрагмента

    Returns:
        str: Компактная строка метаданных
    """
    parts = []

    # Topic
    if topic := metadata.get("topic"):
        parts.append(f"topic:{topic}")

    # Availability
    if availability := metadata.get("availability"):
        parts.append(f"availability:{availability}")

    # Emotion label
    if emotion_label := metadata.get("emotion_label"):
        parts.append(f"emotion:{emotion_label}")

    # Tags (первые 3)
    if tags := metadata.get("tags"):
        if isinstance(tags, list) and tags:
            limited_tags = tags[:3]
            tags_str = ",".join(limited_tags)
            parts.append(f"tags:{tags_str}")

    return " | ".join(parts) if parts else ""


def embed_fragment_text(text: str, metadata: Dict[str, Any]) -> np.ndarray:
    """
    Создает эмбеддинг для текста фрагмента KB.

    Args:
        text: Основной текст фрагмента
        metadata: Метаданные фрагмента (topic, availability, emotion_label, tags)

    Returns:
        np.ndarray: Вектор эмбеддинга

    Raises:
        ValueError: При некорректных входных данных
        RuntimeError: При ошибках модели
    """
    if not text or not isinstance(text, str):
        raise ValueError("Text must be a non-empty string")

    if not isinstance(metadata, dict):
        raise ValueError("Metadata must be a dictionary")

    try:
        # Формируем текст для эмбеддинга: text + META информация
        compact_meta = _compact_metadata(metadata)
        embedding_text = text

        if compact_meta:
            embedding_text += f"\nMETA: {compact_meta}"

        # Получаем модель и создаем эмбеддинг
        model = get_embedding_model()

        # Создаем эмбеддинг
        embeddings = model.encode([embedding_text])
        embedding_vector = embeddings[0]

        logger.debug(
            "Embedding created successfully",
            text_length=len(text),
            meta_length=len(compact_meta),
            embedding_dimension=len(embedding_vector),
        )

        return embedding_vector.astype(np.float32)

    except Exception as e:
        logger.error(
            "Failed to create embedding",
            text_preview=text[:100] + "..." if len(text) > 100 else text,
            error=str(e),
        )
        raise RuntimeError(f"Embedding creation failed: {e}")


def embed_fragments_batch(fragments_data: List[Dict[str, Any]]) -> List[np.ndarray]:
    """
    Создает эмбеддинги для батча фрагментов (более эффективно).

    Args:
        fragments_data: Список словарей с ключами 'text' и 'metadata'

    Returns:
        List[np.ndarray]: Список векторов эмбеддингов

    Raises:
        ValueError: При некорректных входных данных
        RuntimeError: При ошибках модели
    """
    if not fragments_data:
        return []

    try:
        # Подготавливаем тексты для батч-обработки
        embedding_texts = []
        for fragment in fragments_data:
            if "text" not in fragment or "metadata" not in fragment:
                raise ValueError("Each fragment must have 'text' and 'metadata' keys")

            text = fragment["text"]
            metadata = fragment["metadata"]

            if not text or not isinstance(text, str):
                raise ValueError("Fragment text must be a non-empty string")

            compact_meta = _compact_metadata(metadata)
            embedding_text = text

            if compact_meta:
                embedding_text += f"\nMETA: {compact_meta}"

            embedding_texts.append(embedding_text)

        # Батч-кодирование
        model = get_embedding_model()
        embeddings = model.encode(embedding_texts)

        # Конвертируем в список numpy массивов
        result = [embedding.astype(np.float32) for embedding in embeddings]

        logger.debug(
            "Batch embeddings created successfully",
            batch_size=len(fragments_data),
            embedding_dimension=len(result[0]) if result else 0,
        )

        return result

    except Exception as e:
        logger.error(
            "Failed to create batch embeddings",
            batch_size=len(fragments_data),
            error=str(e),
        )
        raise RuntimeError(f"Batch embedding creation failed: {e}")


async def _get_fragments_for_embedding(
    session: AsyncSession, case_id: uuid.UUID, limit: int = 128
) -> List[KBFragment]:
    """Получает фрагменты для case_id, у которых embedding IS NULL."""
    query = (
        select(KBFragment)
        .where(KBFragment.case_id == case_id, KBFragment.embedding.is_(None))
        .order_by(KBFragment.id)
        .limit(limit)
    )

    result = await session.execute(query)
    fragments = result.scalars().all()
    return list(fragments)


async def _update_fragments_embeddings(
    session: AsyncSession,
    fragments_with_embeddings: List[Tuple[KBFragment, np.ndarray]],
) -> int:
    """Обновляет эмбеддинги фрагментов в базе данных."""
    if not fragments_with_embeddings:
        return 0

    updated_count = 0
    for fragment, embedding in fragments_with_embeddings:
        await session.execute(
            update(KBFragment)
            .where(KBFragment.id == fragment.id)
            .values(embedding=embedding.tolist())
        )
        updated_count += 1

    await session.commit()
    return updated_count


async def run_embed(case_id: uuid.UUID) -> Dict[str, int]:
    """
    Вычисляет эмбеддинги для NULL-записей; возвращает {processed:int, dim:int}.

    Args:
        case_id: UUID случая для обработки эмбеддингов

    Returns:
        dict: Статистика обработки с ключами 'processed' и 'dim'
    """
    logger.info("Starting embedding processing", case_id=str(case_id))

    processed = 0
    dimension = 0

    async with AsyncSessionLocal() as session:
        # Проверяем наличие фрагментов для case
        case_check = await session.execute(
            select(func.count()).select_from(
                select(KBFragment).where(KBFragment.case_id == case_id).subquery()
            )
        )
        total_fragments = case_check.scalar()

        if total_fragments == 0:
            logger.warning("No KB fragments found for case", case_id=str(case_id))
            return {"processed": 0, "dim": 0}

        # Обрабатываем батчами пока есть фрагменты без эмбеддингов
        while True:
            fragments = await _get_fragments_for_embedding(session, case_id)

            if not fragments:
                logger.info("No more fragments to process")
                break

            try:
                # Подготавливаем данные для батч-эмбеддинга
                fragments_data = []
                for fragment in fragments:
                    fragments_data.append(
                        {
                            "text": fragment.text,
                            "metadata": fragment.fragment_metadata or {},
                        }
                    )

                # Создаем эмбеддинги батчем
                embeddings = embed_fragments_batch(fragments_data)

                if embeddings and dimension == 0:
                    dimension = len(embeddings[0])

                # Обновляем БД
                fragments_with_embeddings = list(zip(fragments, embeddings))
                updated_count = await _update_fragments_embeddings(
                    session, fragments_with_embeddings
                )

                processed += updated_count

                logger.info(
                    "Batch processed successfully",
                    batch_size=len(fragments),
                    updated_count=updated_count,
                    total_processed=processed,
                )

            except Exception as e:
                logger.error("Failed to process batch", error=str(e))
                continue

    logger.info(
        "Embedding processing completed",
        case_id=str(case_id),
        processed=processed,
        dimension=dimension,
    )

    return {"processed": processed, "dim": dimension}
