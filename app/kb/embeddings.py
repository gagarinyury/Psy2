"""
Провайдер эмбеддингов для knowledge base фрагментов.

Используется для создания векторных представлений KB фрагментов
с учетом их метаданных.
"""

import numpy as np
from typing import Dict, Any, List
from functools import lru_cache

from sentence_transformers import SentenceTransformer
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
