#!/usr/bin/env python3
"""
Smoke tests CLI для быстрой проверки всей системы.

Использование:
    python -m app.cli.smoke run
"""

import asyncio
import json
import logging
import sys
from typing import Any, Dict

import click
import httpx
from sqlalchemy import select, func

from app.core.db import AsyncSessionLocal
from app.core.tables import Case, KBFragment, Session, TelemetryTurn
from app.cli.case_loader import load_case_from_file

# Setup minimal logging that goes to stderr, not stdout
logging.basicConfig(level=logging.ERROR, stream=sys.stderr)
logger = logging.getLogger(__name__)

# Base URL for API calls
API_BASE_URL = "http://localhost:8000"


class SmokeTestError(Exception):
    """Ошибки smoke тестов"""

    pass


async def ensure_demo_case_exists() -> str:
    """
    Убеждается что демо-кейс загружен и возвращает case_id.

    Returns:
        str: case_id демо-кейса
    """
    try:
        # Загружаем демо-кейс из examples/demo_case.json
        demo_file_path = "app/examples/demo_case.json"
        case_id = await load_case_from_file(demo_file_path)
        return case_id
    except Exception as e:
        logger.error(f"Failed to ensure demo case: {e}")
        raise SmokeTestError(f"Failed to load demo case: {e}")


async def create_session(case_id: str) -> str:
    """
    Создает новую сессию для case.

    Args:
        case_id: UUID case

    Returns:
        str: session_id
    """
    try:
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{API_BASE_URL}/session", json={"case_id": case_id}, timeout=10.0
            )

            if response.status_code != 200:
                raise SmokeTestError(
                    f"Session creation failed: {response.status_code} {response.text}"
                )

            session_data = response.json()
            session_id = session_data["session_id"]
            return session_id

    except httpx.RequestError as e:
        raise SmokeTestError(f"Network error creating session: {e}")
    except Exception as e:
        raise SmokeTestError(f"Failed to create session: {e}")


async def perform_turn(case_id: str, session_id: str, utterance: str) -> Dict[str, Any]:
    """
    Выполняет один turn через API.

    Args:
        case_id: UUID case
        session_id: UUID session
        utterance: Фраза терапевта

    Returns:
        Dict с результатом turn'а
    """
    try:
        turn_data = {
            "therapist_utterance": utterance,
            "session_state": {
                "affect": "neutral",
                "trust": 0.5,
                "fatigue": 0.1,
                "access_level": 1,
                "risk_status": "none",
                "last_turn_summary": "",
            },
            "case_id": case_id,
            "session_id": session_id,
            "options": {},
        }

        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{API_BASE_URL}/turn", json=turn_data, timeout=15.0
            )

            if response.status_code != 200:
                raise SmokeTestError(
                    f"Turn failed: {response.status_code} {response.text}"
                )

            turn_response = response.json()
            return turn_response

    except httpx.RequestError as e:
        raise SmokeTestError(f"Network error in turn: {e}")
    except Exception as e:
        raise SmokeTestError(f"Failed to perform turn: {e}")


async def get_db_counts() -> Dict[str, int]:
    """
    Получает количество записей в основных таблицах.

    Returns:
        Dict с count'ами для каждой таблицы
    """
    try:
        async with AsyncSessionLocal() as db:
            counts = {}

            # Count cases
            cases_result = await db.execute(select(func.count(Case.id)))
            counts["cases"] = cases_result.scalar()

            # Count kb_fragments
            kb_result = await db.execute(select(func.count(KBFragment.id)))
            counts["kb_fragments"] = kb_result.scalar()

            # Count sessions
            sessions_result = await db.execute(select(func.count(Session.id)))
            counts["sessions"] = sessions_result.scalar()

            # Count telemetry_turns
            telemetry_result = await db.execute(select(func.count(TelemetryTurn.id)))
            counts["telemetry_turns"] = telemetry_result.scalar()

            return counts

    except Exception as e:
        logger.error(f"Failed to get DB counts: {e}")
        raise SmokeTestError(f"Failed to get DB counts: {e}")


async def run_smoke_test() -> Dict[str, Any]:
    """
    Запускает полный smoke test.

    Returns:
        Dict с отчетом о smoke test
    """
    try:
        # 1. Ensure demo case exists
        case_id = await ensure_demo_case_exists()

        # 2. Create session
        session_id = await create_session(case_id)

        # 3. Perform test turns
        turns = []

        # Turn 1: Sleep question
        turn1 = await perform_turn(
            case_id, session_id, "Как вы спите последние недели?"
        )
        turns.append(
            {
                "utterance": "Как вы спите последние недели?",
                "intent": turn1.get("eval_markers", {}).get("intent"),
                "risk": turn1.get("risk_status"),
                "used_fragments": turn1.get("used_fragments", []),
            }
        )

        # Turn 2: Suicide risk question
        turn2 = await perform_turn(case_id, session_id, "Бывают ли мысли о суициде?")
        turns.append(
            {
                "utterance": "Бывают ли мысли о суициде?",
                "intent": turn2.get("eval_markers", {}).get("intent"),
                "risk": turn2.get("risk_status"),
                "used_fragments": turn2.get("used_fragments", []),
            }
        )

        # 4. Get DB counts
        db_counts = await get_db_counts()

        # 5. Build report
        report = {
            "case_id": case_id,
            "session_id": session_id,
            "turns": turns,
            "db_counts": db_counts,
            "status": "success",
        }

        return report

    except Exception as e:
        logger.error(f"Smoke test failed: {e}")
        return {
            "status": "failed",
            "error": str(e),
            "case_id": None,
            "session_id": None,
            "turns": [],
            "db_counts": {},
        }


@click.group()
def cli():
    """Smoke Test CLI - быстрые проверки работоспособности системы"""
    pass


@cli.command()
def run():
    """
    Запускает полный smoke test и выводит JSON отчет.
    """
    try:
        report = asyncio.run(run_smoke_test())

        # Print JSON report
        print(json.dumps(report, indent=2, ensure_ascii=False))

        # Exit with appropriate code
        if report.get("status") == "success":
            sys.exit(0)
        else:
            sys.exit(1)

    except Exception as e:
        error_report = {
            "status": "failed",
            "error": str(e),
            "case_id": None,
            "session_id": None,
            "turns": [],
            "db_counts": {},
        }
        print(json.dumps(error_report, indent=2, ensure_ascii=False))
        sys.exit(1)


if __name__ == "__main__":
    cli()
