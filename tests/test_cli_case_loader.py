import pytest
from sqlalchemy import select, text

from app.cli.case_loader import load_case_from_file
from app.core.db import AsyncSessionLocal
from app.core.tables import Case, KBFragment


@pytest.mark.skip(reason="Async loop conflicts - temporary skip")
@pytest.mark.anyio
async def test_cli_case_loader():
    """Тест загрузки demo_case.json через case_loader"""
    # Путь к демо файлу
    demo_file_path = "app/examples/demo_case.json"

    # Создаем уникальные идентификаторы для отслеживания записей этого теста
    # test_version = f"test-1.0-{uuid.uuid4().hex[:8]}"  # unused variable

    # Загружаем случай - модифицируем версию для отслеживания
    await load_case_from_file(demo_file_path)

    # Проверяем результаты загрузки в одной сессии
    async with AsyncSessionLocal() as session:
        # Получаем последний созданный case (по времени создания)
        last_case_result = await session.execute(
            select(Case).order_by(Case.created_at.desc()).limit(1)
        )
        last_case = last_case_result.scalars().first()
        assert last_case is not None, "No case found after loading"

        case_id = last_case.id

        # 1. Проверяем что case был создан с правильными данными
        assert last_case.version == "1.0", f"Expected version '1.0', got '{last_case.version}'"
        assert "MDD" in last_case.case_truth.get("dx_target", []), "Expected MDD in dx_target"

        # 2. Проверяем что создались 2 записи в таблице kb_fragments для этого case
        case_fragments_result = await session.execute(
            select(KBFragment).where(KBFragment.case_id == case_id)
        )
        case_fragments = case_fragments_result.scalars().all()
        assert len(case_fragments) == 2, (
            f"Expected 2 KB fragments for case, got {len(case_fragments)}"
        )

        # 3. Проверяем что metadata->>'availability' содержит правильные значения
        # Проверяем через прямой SQL запрос для надежности
        public_count_result = await session.execute(
            text(
                "SELECT COUNT(*) FROM kb_fragments WHERE case_id = :case_id AND metadata->>'availability' = 'public'"
            ),
            {"case_id": case_id},
        )
        public_count = public_count_result.scalar()
        assert public_count == 1, f"Expected 1 public fragment, got {public_count}"

        gated_count_result = await session.execute(
            text(
                "SELECT COUNT(*) FROM kb_fragments WHERE case_id = :case_id AND metadata->>'availability' = 'gated'"
            ),
            {"case_id": case_id},
        )
        gated_count = gated_count_result.scalar()
        assert gated_count == 1, f"Expected 1 gated fragment, got {gated_count}"

        # Дополнительная проверка через ORM
        public_fragments = [f for f in case_fragments if f.availability == "public"]
        gated_fragments = [f for f in case_fragments if f.availability == "gated"]

        assert len(public_fragments) == 1, (
            f"Expected 1 public fragment via ORM, got {len(public_fragments)}"
        )
        assert len(gated_fragments) == 1, (
            f"Expected 1 gated fragment via ORM, got {len(gated_fragments)}"
        )

        # Проверяем содержимое фрагментов
        public_fragment = public_fragments[0]
        assert public_fragment.type == "bio", f"Expected bio type, got {public_fragment.type}"
        assert "Родился в 1989" in public_fragment.text, (
            f"Expected birth year in text, got {public_fragment.text}"
        )

        gated_fragment = gated_fragments[0]
        assert gated_fragment.type == "symptom", f"Expected symptom type, got {gated_fragment.type}"
        assert "Нарушение сна" in gated_fragment.text, (
            f"Expected sleep disorder in text, got {gated_fragment.text}"
        )

        # Не очищаем данные - оставляем для проверки персистентности
        # В реальной среде данные должны оставаться в БД
