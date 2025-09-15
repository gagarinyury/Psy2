import json
import os
import tempfile

import pytest
from sqlalchemy import select

from app.cli.case_loader import CaseLoaderError, load_case, load_case_from_file
from app.core.db import AsyncSessionLocal
from app.core.policies import (
    DisclosureRules,
    effective_disclosure_level,
    gated_access_allowed,
    is_risk_trigger,
)
from app.core.tables import Case, KBFragment


@pytest.mark.anyio
async def test_policies_valid_load():
    """Тест загрузки валидного demo_case.json и проверки policies"""

    # Создаем тестовые данные для загрузки
    test_case_data = {
        "version": "1.0",
        "case_truth": {
            "dx_target": ["MDD"],
            "ddx": {"MDD": 0.6, "GAD": 0.3, "AUD": 0.1},
            "hidden_facts": ["эпизод в 2016", "семейная история депрессии"],
            "red_flags": ["суицидальные мысли"],
            "trajectories": ["улучшение при поддержке"],
        },
        "policies": {
            "disclosure_rules": {
                "full_on_valid_question": True,
                "partial_if_low_trust": True,
                "min_trust_for_gated": 0.4,
            },
            "distortion_rules": {"enabled": True, "by_defense": {}},
            "risk_protocol": {
                "trigger_keywords": ["суицид", "убить себя", "не хочу жить"],
                "response_style": "stable",
                "lock_topics": [],
            },
            "style_profile": {
                "register": "colloquial",
                "tempo": "medium",
                "length": "short",
            },
        },
    }

    test_kb_data = [
        {
            "id": "kb1",
            "type": "bio",
            "text": "Родился в 1989, работает в ИТ.",
            "metadata": {
                "topic": "background",
                "tags": ["работа", "семья"],
                "emotion_label": "neutral",
                "availability": "public",
                "disclosure_cost": 0,
                "consistency_keys": ["birth_year:1989"],
            },
        }
    ]

    # Используем БД сессию для загрузки и проверки
    async with AsyncSessionLocal() as session:
        try:
            # Загружаем случай напрямую через core функцию
            case_id = await load_case(session, test_case_data, test_kb_data)
            assert case_id is not None

            # Проверяем что запись создалась в БД и содержит нужные policies
            result = await session.execute(select(Case).where(Case.id == case_id))
            case = result.scalars().first()

            assert case is not None
            assert case.policies is not None
            assert "disclosure_rules" in case.policies
            assert case.policies["disclosure_rules"]["min_trust_for_gated"] == 0.4

        finally:
            # Очищаем данные после теста
            await session.execute(
                select(KBFragment).where(KBFragment.case_id == case_id)
            )
            kb_result = await session.execute(
                select(KBFragment).where(KBFragment.case_id == case_id)
            )
            kb_fragments = kb_result.scalars().all()
            for fragment in kb_fragments:
                await session.delete(fragment)

            await session.delete(case)
            await session.commit()


@pytest.mark.anyio
async def test_policies_invalid_rejected():
    """Тест отклонения невалидных policies (distortion_rules.by_defense > 1.0)"""

    # Создаем временный JSON файл с невалидными policies
    invalid_case_data = {
        "case": {
            "version": "1.0",
            "case_truth": {
                "dx_target": ["MDD"],
                "ddx": {"MDD": 0.6},
                "hidden_facts": [],
                "red_flags": [],
                "trajectories": [],
            },
            "policies": {
                "disclosure_rules": {
                    "full_on_valid_question": True,
                    "partial_if_low_trust": True,
                    "min_trust_for_gated": 0.4,
                },
                "distortion_rules": {
                    "enabled": True,
                    "by_defense": {"denial": 1.5},  # Невалидно - больше 1.0
                },
                "risk_protocol": {
                    "trigger_keywords": ["суицид"],
                    "response_style": "stable",
                    "lock_topics": [],
                },
                "style_profile": {
                    "register": "colloquial",
                    "tempo": "medium",
                    "length": "short",
                },
            },
        },
        "kb": [
            {
                "id": "kb1",
                "type": "bio",
                "text": "Тестовый текст",
                "metadata": {
                    "topic": "test",
                    "tags": [],
                    "emotion_label": "neutral",
                    "availability": "public",
                },
            }
        ],
    }

    # Создаем временный файл
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".json", delete=False, encoding="utf-8"
    ) as f:
        json.dump(invalid_case_data, f, ensure_ascii=False)
        temp_file_path = f.name

    try:
        # CLI должен завершиться с CaseLoaderError
        with pytest.raises(CaseLoaderError) as exc_info:
            await load_case_from_file(temp_file_path)

        # Проверяем что ошибка связана с validation
        assert "Invalid policies structure" in str(exc_info.value)

        # Проверка что в БД не появилось записи не нужна,
        # так как validation происходит до сохранения в БД

    finally:
        # Удаляем временный файл
        os.unlink(temp_file_path)


def test_helpers():
    """Тест helper функций"""
    # gated_access_allowed
    assert gated_access_allowed(0.5, 0.4)
    assert not gated_access_allowed(0.3, 0.4)
    assert gated_access_allowed(0.4, 0.4)

    # is_risk_trigger
    assert is_risk_trigger("Бывают ли мысли о суициде?", ["суицид"])
    assert not is_risk_trigger("Как дела?", ["суицид"])
    assert is_risk_trigger("СУИЦИД", ["суицид"])  # case insensitive

    # effective_disclosure_level
    rules = DisclosureRules(min_trust_for_gated=0.4, partial_if_low_trust=True)
    assert effective_disclosure_level(0.5, rules) == "full"
    assert effective_disclosure_level(0.3, rules) == "partial"

    rules_no_partial = DisclosureRules(
        min_trust_for_gated=0.4, partial_if_low_trust=False
    )
    assert effective_disclosure_level(0.3, rules_no_partial) == "none"
