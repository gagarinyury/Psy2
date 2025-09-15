#!/usr/bin/env python3
"""
CLI для загрузки случаев из JSON файлов в базу данных.

Использование:
    python -m app.cli.case_loader load app/examples/demo_case.json
"""

import asyncio
import json
import sys
import uuid
from pathlib import Path

import click
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.db import AsyncSessionLocal
from app.core.policies import Policies
from app.core.tables import Case, KBFragment
from app.infra.logging import get_logger, setup_logging

# Setup logging
setup_logging()
logger = get_logger()


class CaseLoaderError(Exception):
    """Ошибки загрузки случая"""

    pass


async def load_case(session: AsyncSession, case_data: dict, kb_data: list) -> str:
    """
    Загружает случай в базу данных (ядро без создания session).

    Args:
        session: AsyncSession для работы с БД
        case_data: Данные случая из JSON
        kb_data: Список KB фрагментов из JSON

    Returns:
        str: ID созданного случая

    Raises:
        CaseLoaderError: При ошибках загрузки
    """
    try:
        # Создаем новую запись случая
        new_case = Case(
            version=case_data["version"],
            case_truth=case_data["case_truth"],
            policies=case_data["policies"],
        )

        session.add(new_case)
        await session.flush()  # Получаем ID для использования в kb_fragments

        case_id = new_case.id
        logger.info("Case record created", case_id=str(case_id))

        # Загружаем kb_fragments с upsert по id в рамках case_id
        fragments_processed = 0
        fragments_inserted = 0
        fragments_updated = 0

        for kb_item in kb_data:
            if "id" not in kb_item:
                logger.warning("KB item missing 'id' field, skipping", item=kb_item)
                continue

            # Проверяем обязательные поля
            required_kb_fields = ["type", "text", "metadata"]
            missing_fields = [f for f in required_kb_fields if f not in kb_item]
            if missing_fields:
                logger.warning(
                    "KB item missing required fields, skipping",
                    kb_id=kb_item["id"],
                    missing_fields=missing_fields,
                )
                continue

            metadata = kb_item["metadata"]

            # Генерируем UUID на основе case_id + строкового id из JSON
            # Это гарантирует уникальность в рамках конкретного кейса
            kb_id_str = f"{case_id}:{kb_item['id']}"
            kb_uuid = uuid.uuid5(uuid.NAMESPACE_OID, kb_id_str)

            # Проверяем, есть ли уже такой фрагмент в рамках данного case_id
            existing_fragment_query = select(KBFragment).where(
                KBFragment.id == kb_uuid, KBFragment.case_id == case_id
            )
            existing_result = await session.execute(existing_fragment_query)
            existing_fragment = existing_result.scalars().first()

            if existing_fragment:
                # Обновляем существующий фрагмент в рамках того же case_id
                existing_fragment.type = kb_item["type"]
                existing_fragment.text = kb_item["text"]
                existing_fragment.fragment_metadata = metadata
                existing_fragment.availability = metadata.get("availability", "public")
                existing_fragment.consistency_keys = metadata.get("consistency_keys", [])
                existing_fragment.embedding = None
                fragments_updated += 1
            else:
                # Создаем новый фрагмент
                new_fragment = KBFragment(
                    id=kb_uuid,
                    case_id=case_id,
                    type=kb_item["type"],
                    text=kb_item["text"],
                    fragment_metadata=metadata,
                    availability=metadata.get("availability", "public"),
                    consistency_keys=metadata.get("consistency_keys", []),
                    embedding=None,
                )
                session.add(new_fragment)
                fragments_inserted += 1

            fragments_processed += 1

            logger.debug(
                "KB fragment processed",
                kb_id=kb_item["id"],
                type=kb_item["type"],
                availability=metadata.get("availability"),
            )

        await session.commit()

        # Подсчитываем реальное количество фрагментов для этого case
        count_result = await session.execute(
            select(KBFragment).where(KBFragment.case_id == case_id)
        )
        final_count = len(count_result.scalars().all())

        logger.info(
            "Case loading completed successfully",
            case_id=str(case_id),
            version=case_data["version"],
            kb_fragments_processed=fragments_processed,
            kb_fragments_inserted=fragments_inserted,
            kb_fragments_updated=fragments_updated,
            kb_fragments_in_db=final_count,
        )

        return str(case_id)

    except Exception as e:
        await session.rollback()
        logger.error("Database error during case loading", error=str(e))
        raise CaseLoaderError(f"Database error: {e}")


async def load_case_from_file(file_path: str) -> str:
    """
    Загружает случай из JSON файла в базу данных (CLI wrapper).

    Args:
        file_path: Путь к JSON файлу со случаем

    Returns:
        str: ID созданного случая

    Raises:
        CaseLoaderError: При ошибках загрузки
    """
    logger.info("Starting case load", file_path=file_path)

    # Проверяем существование файла
    path = Path(file_path)
    if not path.exists():
        raise CaseLoaderError(f"File not found: {file_path}")

    # Загружаем JSON
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except (json.JSONDecodeError, IOError) as e:
        raise CaseLoaderError(f"Failed to load JSON file: {e}")

    # Валидируем структуру
    if "case" not in data or "kb" not in data:
        raise CaseLoaderError("Invalid JSON structure: missing 'case' or 'kb'")

    case_data = data["case"]
    kb_data = data["kb"]

    if not isinstance(kb_data, list):
        raise CaseLoaderError("'kb' must be an array")

    # Проверяем обязательные поля case
    required_case_fields = ["version", "case_truth", "policies"]
    for field in required_case_fields:
        if field not in case_data:
            raise CaseLoaderError(f"Missing required case field: {field}")

    # Валидируем policies через Pydantic модель
    try:
        Policies.model_validate(case_data["policies"])
        logger.debug("Policies validation passed")
    except Exception as e:
        logger.error("Policies validation failed", error=str(e))
        raise CaseLoaderError(f"Invalid policies structure: {e}")

    logger.info(
        "JSON file loaded and validated",
        version=case_data.get("version"),
        kb_fragments_count=len(kb_data),
    )

    # Загружаем в БД с собственной session
    async with AsyncSessionLocal() as session:
        case_id = await load_case(session, case_data, kb_data)
        return case_id


@click.group()
def cli():
    """Case Loader CLI - утилита для загрузки случаев в БД"""
    pass


@cli.command()
@click.argument("file_path", type=click.Path(exists=True))
def load(file_path: str):
    """
    Загружает случай из JSON файла в базу данных.

    FILE_PATH: Путь к JSON файлу со случаем
    """
    try:
        case_id = asyncio.run(load_case_from_file(file_path))
        click.echo(f"✓ Case loaded successfully from {file_path}")
        click.echo(f"Case ID: {case_id}")
    except CaseLoaderError as e:
        logger.error("Case loading failed", error=str(e))
        click.echo(f"✗ Error: {e}", err=True)
        sys.exit(1)
    except Exception as e:
        logger.error("Unexpected error", error=str(e))
        click.echo(f"✗ Unexpected error: {e}", err=True)
        sys.exit(1)


if __name__ == "__main__":
    cli()
