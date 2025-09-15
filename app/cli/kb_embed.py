#!/usr/bin/env python3
"""
CLI для batch обработки эмбеддингов knowledge base фрагментов.

Использование:
    python -m app.cli.kb_embed run --case-id <uuid> --batch 128
"""

import asyncio
import sys
import uuid
from typing import List, Optional, Tuple

import click
import numpy as np
from sqlalchemy import func, select, update
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.db import AsyncSessionLocal
from app.core.tables import KBFragment
from app.infra.logging import get_logger, setup_logging
from app.kb.embeddings import embed_fragments_batch

# Setup logging
setup_logging()
logger = get_logger()


class KBEmbedError(Exception):
    """Ошибки обработки эмбеддингов KB"""

    pass


async def get_fragments_for_embedding(
    session: AsyncSession, case_id: uuid.UUID, limit: Optional[int] = None
) -> List[KBFragment]:
    """
    Получает фрагменты для case_id, у которых embedding IS NULL.

    Args:
        session: AsyncSession для работы с БД
        case_id: UUID case'а
        limit: Максимальное количество фрагментов для получения

    Returns:
        List[KBFragment]: Список фрагментов без эмбеддингов
    """
    query = (
        select(KBFragment)
        .where(KBFragment.case_id == case_id, KBFragment.embedding.is_(None))
        .order_by(KBFragment.id)
    )

    if limit:
        query = query.limit(limit)

    result = await session.execute(query)
    fragments = result.scalars().all()

    logger.debug(
        "Retrieved fragments for embedding", case_id=str(case_id), count=len(fragments)
    )

    return list(fragments)


async def update_fragments_embeddings(
    session: AsyncSession,
    fragments_with_embeddings: List[Tuple[KBFragment, np.ndarray]],
) -> int:
    """
    Обновляет эмбеддинги фрагментов в базе данных.

    Args:
        session: AsyncSession для работы с БД
        fragments_with_embeddings: Список кортежей (фрагмент, эмбеддинг)

    Returns:
        int: Количество обновленных записей
    """
    if not fragments_with_embeddings:
        return 0

    try:
        updated_count = 0

        for fragment, embedding in fragments_with_embeddings:
            # Обновляем embedding, но НЕ трогаем updated_at (идемпотентность)
            await session.execute(
                update(KBFragment)
                .where(KBFragment.id == fragment.id)
                .values(embedding=embedding.tolist())
            )
            updated_count += 1

        await session.commit()

        logger.debug("Updated fragment embeddings", updated_count=updated_count)

        return updated_count

    except Exception as e:
        await session.rollback()
        logger.error("Failed to update fragment embeddings", error=str(e))
        raise KBEmbedError(f"Database update failed: {e}")


async def process_embeddings_for_case(case_id: str, batch_size: int = 128) -> dict:
    """
    Обрабатывает эмбеддинги для всех фрагментов case'а.

    Args:
        case_id: Строковое представление UUID case'а
        batch_size: Размер батча для обработки

    Returns:
        dict: Статистика обработки
    """
    try:
        case_uuid = uuid.UUID(case_id)
    except ValueError:
        raise KBEmbedError(f"Invalid case_id format: {case_id}")

    logger.info("Starting embedding processing", case_id=case_id, batch_size=batch_size)

    stats = {"processed": 0, "failed": 0, "skipped": 0, "dimension": 0, "batches": 0}

    async with AsyncSessionLocal() as session:
        # Проверяем существование case
        case_check = await session.execute(
            select(func.count()).select_from(
                select(KBFragment).where(KBFragment.case_id == case_uuid).subquery()
            )
        )
        total_fragments = case_check.scalar()

        if total_fragments == 0:
            raise KBEmbedError(f"No KB fragments found for case_id: {case_id}")

        logger.info(
            "Found KB fragments for case",
            case_id=case_id,
            total_fragments=total_fragments,
        )

        # Обрабатываем батчами
        while True:
            # Получаем следующий батч фрагментов без эмбеддингов
            fragments = await get_fragments_for_embedding(
                session, case_uuid, batch_size
            )

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

                if len(embeddings) != len(fragments):
                    raise KBEmbedError(
                        f"Embedding count mismatch: got {len(embeddings)}, expected {len(fragments)}"
                    )

                # Запоминаем размерность эмбеддингов
                if embeddings and stats["dimension"] == 0:
                    stats["dimension"] = len(embeddings[0])

                # Подготавливаем данные для обновления БД
                fragments_with_embeddings = list(zip(fragments, embeddings))

                # Обновляем БД
                updated_count = await update_fragments_embeddings(
                    session, fragments_with_embeddings
                )

                stats["processed"] += updated_count
                stats["batches"] += 1

                logger.info(
                    "Batch processed successfully",
                    batch_number=stats["batches"],
                    batch_size=len(fragments),
                    updated_count=updated_count,
                    total_processed=stats["processed"],
                )

            except Exception as e:
                logger.error(
                    "Failed to process batch",
                    batch_number=stats["batches"] + 1,
                    error=str(e),
                )
                stats["failed"] += len(fragments)
                # Продолжаем обработку следующего батча
                continue

    logger.info(
        "Embedding processing completed",
        case_id=case_id,
        processed=stats["processed"],
        failed=stats["failed"],
        batches=stats["batches"],
        dimension=stats["dimension"],
    )

    return stats


@click.group()
def cli():
    """KB Embedding CLI - утилита для batch обработки эмбеддингов"""
    pass


@cli.command()
@click.option("--case-id", required=True, help="UUID случая для обработки эмбеддингов")
@click.option(
    "--batch", default=128, help="Размер батча для обработки (по умолчанию 128)"
)
def run(case_id: str, batch: int):
    """
    Создает эмбеддинги для всех KB фрагментов указанного случая.

    Обрабатывает только фрагменты где embedding IS NULL.
    Идемпотентная операция - повторный запуск безопасен.
    """
    try:
        # Валидация параметров
        if batch <= 0:
            raise KBEmbedError("Batch size must be positive")

        if batch > 1000:
            logger.warning("Large batch size may cause memory issues", batch_size=batch)

        # Запускаем обработку
        stats = asyncio.run(process_embeddings_for_case(case_id, batch))

        # Выводим результаты
        click.echo(f"✓ Embedding processing completed for case {case_id}")
        click.echo(f"  Processed: {stats['processed']} fragments")
        click.echo(f"  Failed: {stats['failed']} fragments")
        click.echo(f"  Batches: {stats['batches']}")
        click.echo(f"  Embedding dimension: {stats['dimension']}")

        if stats["failed"] > 0:
            click.echo(
                f"⚠ Warning: {stats['failed']} fragments failed to process", err=True
            )
            sys.exit(1)

    except KBEmbedError as e:
        logger.error("KB embedding processing failed", error=str(e))
        click.echo(f"✗ Error: {e}", err=True)
        sys.exit(1)
    except Exception as e:
        logger.error("Unexpected error", error=str(e))
        click.echo(f"✗ Unexpected error: {e}", err=True)
        sys.exit(1)


if __name__ == "__main__":
    cli()
