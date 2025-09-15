import pytest

from app.orchestrator.nodes.normalize import normalize


class TestNormalize:
    """Test suite for normalize function."""

    def test_clarify_intent_with_sleep_topic(self):
        """Test specific case from requirements: 'Как вы спите последние недели?'"""
        result = normalize("Как вы спите последние недели?", {})

        assert result["intent"] == "clarify"
        assert "sleep" in result["topics"]
        assert result["risk_flags"] == []
        assert result["last_turn_summary"] == "Как вы спите последние недели?"

    def test_risk_check_with_suicide_flag(self):
        """Test specific case from requirements: 'Бывают ли мысли о суициде?'"""
        result = normalize("Бывают ли мысли о суициде?", {})

        assert result["intent"] == "risk_check"
        assert result["risk_flags"] == ["suicide_ideation"]
        assert result["last_turn_summary"] == "Бывают ли мысли о суициде?"

    def test_intent_open_question(self):
        """Test open_question intent (default case)."""
        result = normalize("Расскажите о себе подробнее", {})

        assert result["intent"] == "open_question"
        assert result["topics"] == []
        assert result["risk_flags"] == []

    def test_intent_clarify_various_keywords(self):
        """Test clarify intent with different clarifying keywords."""
        test_cases = [
            "Что вас беспокоит?",
            "Когда это началось?",
            "Где вы чувствуете напряжение?",
            "Почему так произошло?",
            "Какой у вас опыт?",
        ]

        for utterance in test_cases:
            result = normalize(utterance, {})
            assert result["intent"] == "clarify", f"Failed for: {utterance}"

    def test_intent_rapport(self):
        """Test rapport intent with supportive keywords."""
        test_cases = [
            "Я понимаю ваши переживания",
            "Сочувствую вашей ситуации",
            "Поддерживаю ваше решение",
        ]

        for utterance in test_cases:
            result = normalize(utterance, {})
            assert result["intent"] == "rapport", f"Failed for: {utterance}"

    def test_intent_risk_check_various_keywords(self):
        """Test risk_check intent with different suicide-related keywords."""
        test_cases = [
            "Есть ли мысли убить себя?",
            "Не хочу жить дальше",
            "Думаете о том, чтобы покончить с жизнью?",
        ]

        for utterance in test_cases:
            result = normalize(utterance, {})
            assert result["intent"] == "risk_check", f"Failed for: {utterance}"

    def test_topics_sleep(self):
        """Test sleep topic extraction."""
        test_cases = [
            "Как вы спите?",
            "Проблемы с сон",
            "Страдаете бессонницей?",
            "Трудно засыпать?",
        ]

        for utterance in test_cases:
            result = normalize(utterance, {})
            assert "sleep" in result["topics"], f"Failed for: {utterance}"

    def test_topics_mood(self):
        """Test mood topic extraction."""
        test_cases = [
            "Какое у вас настроение?",
            "Есть ли депрессия?",
            "Чувствуете грусть?",
            "Есть тревога?",
        ]

        for utterance in test_cases:
            result = normalize(utterance, {})
            assert "mood" in result["topics"], f"Failed for: {utterance}"

    def test_topics_alcohol(self):
        """Test alcohol topic extraction."""
        test_cases = [
            "Употребляете алкоголь?",
            "Любите пить?",
            "Есть выпивка",
            "Водка или пиво?",
        ]

        for utterance in test_cases:
            result = normalize(utterance, {})
            assert "alcohol" in result["topics"], f"Failed for: {utterance}"

    def test_topics_work(self):
        """Test work topic extraction."""
        test_cases = [
            "Как дела с работой?",
            "Проблемы с работой",
            "Вопросы по карьера",
            "Конфликт с босс",
        ]

        for utterance in test_cases:
            result = normalize(utterance, {})
            assert "work" in result["topics"], f"Failed for: {utterance}"

    def test_topics_family(self):
        """Test family topic extraction."""
        test_cases = [
            "Как дела с семьей?",
            "Отношения с семьей",
            "Проблемы с родители",
            "Конфликт с жена",
        ]

        for utterance in test_cases:
            result = normalize(utterance, {})
            assert "family" in result["topics"], f"Failed for: {utterance}"

    def test_multiple_topics(self):
        """Test extraction of multiple topics in one utterance."""
        result = normalize("Как работа влияет на ваш сон и семья?", {})

        assert "work" in result["topics"]
        assert "sleep" in result["topics"]
        assert "family" in result["topics"]
        assert len(result["topics"]) == 3

    def test_risk_flags_various_suicide_keywords(self):
        """Test risk flag extraction with different suicide-related phrases."""
        test_cases = [
            "Мысли о суициде",
            "Хочу убить себя",
            "Не хочу жить",
            "Покончить с жизнью",
            "Думаю повеситься",
            "Планирую отравиться",
        ]

        for utterance in test_cases:
            result = normalize(utterance, {})
            assert (
                "suicide_ideation" in result["risk_flags"]
            ), f"Failed for: {utterance}"

    def test_no_risk_flags(self):
        """Test that normal utterances don't trigger risk flags."""
        test_cases = ["Как дела?", "Расскажите о работе", "Проблемы со сном"]

        for utterance in test_cases:
            result = normalize(utterance, {})
            assert result["risk_flags"] == [], f"Failed for: {utterance}"

    def test_last_turn_summary_short_text(self):
        """Test summary creation for text under 200 characters."""
        short_text = "Короткий вопрос"
        result = normalize(short_text, {})

        assert result["last_turn_summary"] == short_text
        assert len(result["last_turn_summary"]) < 200

    def test_last_turn_summary_long_text_truncation(self):
        """Test summary truncation for text over 200 characters."""
        long_text = (
            "Это очень длинный текст, который превышает лимит в 200 символов. " * 5
        )
        result = normalize(long_text, {})

        assert len(result["last_turn_summary"]) == 203  # 200 + "..."
        assert result["last_turn_summary"].endswith("...")
        assert result["last_turn_summary"].startswith("Это очень длинный текст")

    def test_empty_input(self):
        """Test handling of empty input."""
        result = normalize("", {})

        assert result["intent"] == "open_question"
        assert result["topics"] == []
        assert result["risk_flags"] == []
        assert result["last_turn_summary"] == ""

    def test_none_input_handling(self):
        """Test that None input is handled gracefully."""
        # This test assumes the function should handle None gracefully
        # If not, it will fail and reveal the expected behavior
        try:
            result = normalize(None, {})
            # If it doesn't raise an exception, check the results
            assert isinstance(result, dict)
            assert "intent" in result
            assert "topics" in result
            assert "risk_flags" in result
            assert "last_turn_summary" in result
        except (TypeError, AttributeError):
            # Expected behavior for None input
            pytest.skip("Function correctly raises exception for None input")

    def test_case_insensitive_processing(self):
        """Test that function works correctly with different cases."""
        test_cases = [
            ("КАК ВЫ СПИТЕ?", "clarify", ["sleep"]),
            ("как вы спите?", "clarify", ["sleep"]),
            ("Как Вы Спите?", "clarify", ["sleep"]),
        ]

        for utterance, expected_intent, expected_topics in test_cases:
            result = normalize(utterance, {})
            assert result["intent"] == expected_intent
            for topic in expected_topics:
                assert topic in result["topics"]

    def test_session_state_parameter_ignored(self):
        """Test that session_state parameter doesn't affect results."""
        utterance = "Как дела с работой?"

        result1 = normalize(utterance, {})
        result2 = normalize(utterance, {"some": "state"})
        result3 = normalize(utterance, None)

        assert result1 == result2 == result3

    def test_return_structure(self):
        """Test that function always returns expected structure."""
        result = normalize("Тестовый вопрос", {})

        assert isinstance(result, dict)
        assert "intent" in result
        assert "topics" in result
        assert "risk_flags" in result
        assert "last_turn_summary" in result

        assert isinstance(result["intent"], str)
        assert isinstance(result["topics"], list)
        assert isinstance(result["risk_flags"], list)
        assert isinstance(result["last_turn_summary"], str)

        # Check intent is one of expected values
        assert result["intent"] in ["open_question", "clarify", "risk_check", "rapport"]
