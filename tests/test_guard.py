from app.orchestrator.nodes.guard import guard


class TestGuard:
    """Test suite for guard function."""

    def test_guard_no_risk(self):
        """Test guard with empty risk_flags - should return unchanged content."""
        reason_output = {
            "content_plan": ["Patient shows signs of anxiety", "Consider CBT approach"],
            "distortions_plan": [],
            "style_directives": {"tempo": "normal", "length": "medium"},
            "state_updates": {"trust_delta": 0.02, "fatigue_delta": 0.0},
            "telemetry": {"chosen_ids": [1, 2], "count": 2},
        }
        policies = {"risk_protocol": {"escalation_threshold": 0.8}}
        risk_flags = []

        result = guard(reason_output, policies, risk_flags)

        # Check structure
        assert isinstance(result, dict)
        assert "safe_output" in result
        assert "risk_status" in result

        # Risk status should be none
        assert result["risk_status"] == "none"

        # Content should be unchanged
        assert result["safe_output"] == reason_output
        assert result["safe_output"]["content_plan"] == [
            "Patient shows signs of anxiety",
            "Consider CBT approach",
        ]
        assert result["safe_output"]["style_directives"]["tempo"] == "normal"

    def test_guard_with_risk(self):
        """Test guard with risk_flags - should filter content and set acute status."""
        reason_output = {
            "content_plan": ["Patient shows signs of anxiety", "Consider CBT approach"],
            "distortions_plan": [],
            "style_directives": {"tempo": "normal", "length": "medium"},
            "state_updates": {"trust_delta": 0.02, "fatigue_delta": 0.0},
            "telemetry": {"chosen_ids": [1, 2], "count": 2},
        }
        policies = {"risk_protocol": {"escalation_threshold": 0.8}}
        risk_flags = ["suicide_ideation"]

        result = guard(reason_output, policies, risk_flags)

        # Check structure
        assert isinstance(result, dict)
        assert "safe_output" in result
        assert "risk_status" in result

        # Risk status should be acute
        assert result["risk_status"] == "acute"

        # Content plan should be replaced with risk protocol message
        assert result["safe_output"]["content_plan"] == [
            "[Риск-триггер: обращение к протоколу]"
        ]

        # Tempo should be overridden to calm
        assert result["safe_output"]["style_directives"]["tempo"] == "calm"

    def test_guard_preserves_other_fields(self):
        """Test that guard preserves all other fields when filtering risk content."""
        reason_output = {
            "content_plan": ["Original content", "More content"],
            "distortions_plan": [{"type": "cognitive", "target": "catastrophizing"}],
            "style_directives": {
                "tempo": "fast",
                "length": "long",
                "register": "informal",
            },
            "state_updates": {"trust_delta": 0.05, "fatigue_delta": -0.01},
            "telemetry": {
                "chosen_ids": ["frag-1", "frag-2", "frag-3"],
                "candidates_count": 5,
                "chosen_count": 3,
                "content_plan_size": 2,
            },
        }
        policies = {}
        risk_flags = ["suicide_ideation", "self_harm"]

        result = guard(reason_output, policies, risk_flags)

        safe_output = result["safe_output"]

        # Content plan should be replaced
        assert safe_output["content_plan"] == ["[Риск-триггер: обращение к протоколу]"]

        # Tempo should be overridden
        assert safe_output["style_directives"]["tempo"] == "calm"

        # All other fields should be preserved exactly
        assert safe_output["distortions_plan"] == [
            {"type": "cognitive", "target": "catastrophizing"}
        ]
        assert safe_output["style_directives"]["length"] == "long"
        assert safe_output["style_directives"]["register"] == "informal"
        assert safe_output["state_updates"] == {
            "trust_delta": 0.05,
            "fatigue_delta": -0.01,
        }
        assert safe_output["telemetry"] == {
            "chosen_ids": ["frag-1", "frag-2", "frag-3"],
            "candidates_count": 5,
            "chosen_count": 3,
            "content_plan_size": 2,
        }

    def test_guard_style_override(self):
        """Test that tempo is correctly overridden to 'calm' when risk is detected."""
        # Test with existing style_directives
        reason_output = {
            "content_plan": ["Some content"],
            "style_directives": {"tempo": "fast", "length": "short"},
            "distortions_plan": [],
            "state_updates": {"trust_delta": 0.0, "fatigue_delta": 0.0},
            "telemetry": {},
        }
        policies = {}
        risk_flags = ["suicide_ideation"]

        result = guard(reason_output, policies, risk_flags)

        assert result["safe_output"]["style_directives"]["tempo"] == "calm"
        assert (
            result["safe_output"]["style_directives"]["length"] == "short"
        )  # preserved

        # Test with missing style_directives
        reason_output_no_style = {
            "content_plan": ["Some content"],
            "distortions_plan": [],
            "state_updates": {"trust_delta": 0.0, "fatigue_delta": 0.0},
            "telemetry": {},
        }

        result = guard(reason_output_no_style, policies, risk_flags)

        assert "style_directives" in result["safe_output"]
        assert result["safe_output"]["style_directives"]["tempo"] == "calm"

    def test_guard_multiple_risk_flags(self):
        """Test guard behavior with multiple risk flags."""
        reason_output = {
            "content_plan": ["Original content"],
            "style_directives": {"tempo": "normal", "length": "medium"},
            "distortions_plan": [],
            "state_updates": {"trust_delta": 0.02, "fatigue_delta": 0.0},
            "telemetry": {"chosen_ids": [1]},
        }
        policies = {}
        risk_flags = ["suicide_ideation", "self_harm", "substance_abuse"]

        result = guard(reason_output, policies, risk_flags)

        # Should still be acute status regardless of number of flags
        assert result["risk_status"] == "acute"

        # Content should be filtered same way
        assert result["safe_output"]["content_plan"] == [
            "[Риск-триггер: обращение к протоколу]"
        ]
        assert result["safe_output"]["style_directives"]["tempo"] == "calm"

    def test_guard_empty_reason_output(self):
        """Test guard handling of empty or None reason_output."""
        policies = {}
        risk_flags = []

        # Test with None
        result = guard(None, policies, risk_flags)
        assert result["risk_status"] == "none"
        assert result["safe_output"] == {}

        # Test with empty dict
        result = guard({}, policies, risk_flags)
        assert result["risk_status"] == "none"
        assert result["safe_output"] == {}

        # Test with None and risk flags
        risk_flags = ["suicide_ideation"]
        result = guard(None, policies, risk_flags)
        assert result["risk_status"] == "acute"
        assert result["safe_output"]["content_plan"] == [
            "[Риск-триггер: обращение к протоколу]"
        ]
        assert result["safe_output"]["style_directives"]["tempo"] == "calm"

    def test_guard_empty_content_plan(self):
        """Test guard with reason_output that has empty content_plan."""
        reason_output = {
            "content_plan": [],
            "style_directives": {"tempo": "normal", "length": "short"},
            "distortions_plan": [],
            "state_updates": {"trust_delta": -0.01, "fatigue_delta": 0.0},
            "telemetry": {"chosen_ids": [], "count": 0},
        }
        policies = {}
        risk_flags = ["suicide_ideation"]

        result = guard(reason_output, policies, risk_flags)

        assert result["risk_status"] == "acute"
        # Should still replace empty content_plan with risk message
        assert result["safe_output"]["content_plan"] == [
            "[Риск-триггер: обращение к протоколу]"
        ]
        assert result["safe_output"]["style_directives"]["tempo"] == "calm"

    def test_guard_deep_copy_behavior(self):
        """Test that guard does not modify original reason_output object."""
        original_reason_output = {
            "content_plan": ["Original content"],
            "style_directives": {"tempo": "fast", "length": "medium"},
            "distortions_plan": [],
            "state_updates": {"trust_delta": 0.02, "fatigue_delta": 0.0},
            "telemetry": {},
        }
        policies = {}
        risk_flags = ["suicide_ideation"]

        # Make a copy to compare later
        original_copy = original_reason_output.copy()

        result = guard(original_reason_output, policies, risk_flags)

        # Original should be unchanged
        assert original_reason_output == original_copy
        assert original_reason_output["content_plan"] == ["Original content"]
        assert original_reason_output["style_directives"]["tempo"] == "fast"

        # Result should be modified
        assert result["safe_output"]["content_plan"] == [
            "[Риск-триггер: обращение к протоколу]"
        ]
        assert result["safe_output"]["style_directives"]["tempo"] == "calm"

    def test_guard_return_structure(self):
        """Test that guard always returns expected structure."""
        reason_output = {
            "content_plan": ["Test"],
            "style_directives": {"tempo": "normal"},
            "distortions_plan": [],
            "state_updates": {},
            "telemetry": {},
        }
        policies = {}

        # Test with no risk
        result = guard(reason_output, policies, [])
        assert isinstance(result, dict)
        assert len(result) == 2
        assert "safe_output" in result
        assert "risk_status" in result
        assert result["risk_status"] in ["none", "acute"]

        # Test with risk
        result = guard(reason_output, policies, ["suicide_ideation"])
        assert isinstance(result, dict)
        assert len(result) == 2
        assert "safe_output" in result
        assert "risk_status" in result
        assert result["risk_status"] in ["none", "acute"]

    def test_guard_risk_status_values(self):
        """Test that risk_status only returns expected values."""
        reason_output = {"content_plan": [], "style_directives": {}}
        policies = {}

        # No risk should return "none"
        result = guard(reason_output, policies, [])
        assert result["risk_status"] == "none"

        # Single risk flag should return "acute"
        result = guard(reason_output, policies, ["suicide_ideation"])
        assert result["risk_status"] == "acute"

        # Multiple risk flags should return "acute"
        result = guard(reason_output, policies, ["suicide_ideation", "self_harm"])
        assert result["risk_status"] == "acute"
