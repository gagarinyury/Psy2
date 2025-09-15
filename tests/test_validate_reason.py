"""
Tests for reason payload validation and auto-repair.

Tests pure functions without async, covering all normalization scenarios.
"""

from app.llm.validate import validate_reason_payload


class TestValidateReasonPayload:
    def test_valid_payload_unchanged(self):
        """Test that valid payload passes through unchanged."""
        payload = {
            "content_plan": ["Hello", "How are you?"],
            "style_directives": {"tempo": "medium", "length": "short"},
            "state_updates": {"trust_delta": 0.1, "fatigue_delta": 0.05},
            "telemetry": {"chosen_ids": ["frag1", "frag2"]},
        }

        candidates = [
            {"id": "frag1", "text": "Fragment 1"},
            {"id": "frag2", "text": "Fragment 2"},
        ]

        result, warnings = validate_reason_payload(payload, candidates)

        assert result["content_plan"] == ["Hello", "How are you?"]
        assert result["style_directives"]["tempo"] == "medium"
        assert result["style_directives"]["length"] == "short"
        assert result["state_updates"]["trust_delta"] == 0.1
        assert result["state_updates"]["fatigue_delta"] == 0.05
        assert result["telemetry"]["chosen_ids"] == ["frag1", "frag2"]
        assert len(warnings) == 0

    def test_trust_delta_clamping(self):
        """Test trust_delta is clamped to [-0.2, 0.2]."""
        payload = {
            "content_plan": ["test"],
            "state_updates": {"trust_delta": 0.5, "fatigue_delta": 0.1},
        }

        result, warnings = validate_reason_payload(payload, [])

        assert result["state_updates"]["trust_delta"] == 0.2
        assert "trust_delta 0.5 clamped to 0.2" in warnings

        # Test negative clamping
        payload["state_updates"]["trust_delta"] = -0.8
        result, warnings = validate_reason_payload(payload, [])

        assert result["state_updates"]["trust_delta"] == -0.2
        assert "trust_delta -0.8 clamped to -0.2" in warnings

    def test_fatigue_delta_clamping(self):
        """Test fatigue_delta is clamped to [0.0, 0.2]."""
        payload = {
            "content_plan": ["test"],
            "state_updates": {"trust_delta": 0.0, "fatigue_delta": 0.5},
        }

        result, warnings = validate_reason_payload(payload, [])

        assert result["state_updates"]["fatigue_delta"] == 0.2
        assert "fatigue_delta 0.5 clamped to 0.2" in warnings

        # Test negative clamping to 0.0
        payload["state_updates"]["fatigue_delta"] = -0.1
        result, warnings = validate_reason_payload(payload, [])

        assert result["state_updates"]["fatigue_delta"] == 0.0
        assert "fatigue_delta -0.1 clamped to 0.0" in warnings

    def test_nan_values_handled(self):
        """Test NaN values are converted to 0.0."""
        payload = {
            "content_plan": ["test"],
            "state_updates": {
                "trust_delta": float("nan"),
                "fatigue_delta": float("inf"),
            },
        }

        result, warnings = validate_reason_payload(payload, [])

        assert result["state_updates"]["trust_delta"] == 0.0
        assert result["state_updates"]["fatigue_delta"] == 0.0
        assert "trust_delta was NaN/inf, set to 0.0" in warnings
        assert "fatigue_delta was NaN/inf, set to 0.0" in warnings

    def test_invalid_tempo_fixed(self):
        """Test invalid tempo values are set to 'medium'."""
        payload = {
            "content_plan": ["test"],
            "style_directives": {"tempo": "super_fast", "length": "short"},
        }

        result, warnings = validate_reason_payload(payload, [])

        assert result["style_directives"]["tempo"] == "medium"
        assert "tempo 'super_fast' invalid, set to 'medium'" in warnings

    def test_invalid_length_fixed(self):
        """Test invalid length values are set to 'short'."""
        payload = {
            "content_plan": ["test"],
            "style_directives": {"tempo": "medium", "length": "tiny"},
        }

        result, warnings = validate_reason_payload(payload, [])

        assert result["style_directives"]["length"] == "short"
        assert "length 'tiny' invalid, set to 'short'" in warnings

    def test_chosen_ids_filtered_for_valid_candidates(self):
        """Test chosen_ids keeps only IDs from valid candidates."""
        payload = {
            "content_plan": ["test"],
            "telemetry": {"chosen_ids": ["frag1", "invalid_id", "frag2", "another_invalid"]},
        }

        candidates = [
            {"id": "frag1", "text": "Fragment 1"},
            {"id": "frag2", "text": "Fragment 2"},
        ]

        result, warnings = validate_reason_payload(payload, candidates)

        assert set(result["telemetry"]["chosen_ids"]) == {"frag1", "frag2"}
        assert "chosen_id 'invalid_id' not in valid candidates, removed" in warnings
        assert "chosen_id 'another_invalid' not in valid candidates, removed" in warnings

    def test_chosen_ids_deduplication(self):
        """Test chosen_ids are deduplicated."""
        payload = {
            "content_plan": ["test"],
            "telemetry": {"chosen_ids": ["frag1", "frag2", "frag1", "frag2"]},
        }

        candidates = [
            {"id": "frag1", "text": "Fragment 1"},
            {"id": "frag2", "text": "Fragment 2"},
        ]

        result, warnings = validate_reason_payload(payload, candidates)

        assert result["telemetry"]["chosen_ids"] == ["frag1", "frag2"]

    def test_empty_chosen_ids_substituted(self):
        """Test empty chosen_ids gets all candidate IDs when candidates exist."""
        payload = {
            "content_plan": ["test"],
            "telemetry": {"chosen_ids": []},
        }

        candidates = [
            {"id": "frag1", "text": "Fragment 1"},
            {"id": "frag2", "text": "Fragment 2"},
        ]

        result, warnings = validate_reason_payload(payload, candidates)

        assert set(result["telemetry"]["chosen_ids"]) == {"frag1", "frag2"}
        assert "chosen_ids was empty, substituted candidate IDs" in warnings

    def test_empty_content_plan_auto_repair(self):
        """Test empty content_plan is auto-repaired from candidates."""
        payload = {
            "content_plan": [],
            "telemetry": {},
        }

        candidates = [
            {
                "id": "frag1",
                "text": "This is a longer text that should be truncated at 200 characters. " * 5,
            },
            {"id": "frag2", "text": "Short fragment"},
        ]

        result, warnings = validate_reason_payload(payload, candidates)

        assert len(result["content_plan"]) <= 2  # max 2 elements
        assert len(result["content_plan"]) > 0  # should be repaired
        assert result["telemetry"]["llm_empty_plan"] is True
        assert "content_plan was empty, generated from candidates" in warnings

        # Check that text is truncated at 200 chars
        for plan_item in result["content_plan"]:
            assert len(plan_item) <= 200

    def test_content_plan_trimmed_and_limited(self):
        """Test content_plan is trimmed and limited to max 2 elements."""
        payload = {
            "content_plan": ["  first  ", "", "  second  ", "third", "fourth"],
        }

        result, warnings = validate_reason_payload(payload, [])

        assert result["content_plan"] == [
            "first",
            "second",
        ]  # max 2, trimmed, empty removed

    def test_malformed_data_structures_handled(self):
        """Test malformed data structures are handled gracefully."""
        payload = {
            "content_plan": "not a list",
            "style_directives": "not a dict",
            "state_updates": [],
            "telemetry": "not a dict",
        }

        result, warnings = validate_reason_payload(payload, [])

        assert isinstance(result["content_plan"], list)
        assert isinstance(result["style_directives"], dict)
        assert isinstance(result["state_updates"], dict)
        assert isinstance(result["telemetry"], dict)

        assert result["style_directives"]["tempo"] == "medium"
        assert result["style_directives"]["length"] == "short"
        assert result["state_updates"]["trust_delta"] == 0.0
        assert result["state_updates"]["fatigue_delta"] == 0.0
        assert result["telemetry"]["chosen_ids"] == []

        assert len(warnings) > 0  # Should have warnings for malformed structures

    def test_validation_warnings_added_to_telemetry(self):
        """Test that validation warnings are added to telemetry."""
        payload = {
            "content_plan": ["test"],
            "style_directives": {"tempo": "invalid", "length": "invalid"},
            "state_updates": {"trust_delta": 1.0, "fatigue_delta": -0.5},
        }

        result, warnings = validate_reason_payload(payload, [])

        assert "validation_warnings" in result["telemetry"]
        assert len(result["telemetry"]["validation_warnings"]) > 0
        assert result["telemetry"]["validation_warnings"] == warnings

    def test_empty_payload_handled(self):
        """Test completely empty payload is handled."""
        result, warnings = validate_reason_payload({}, [])

        assert "content_plan" in result
        assert "style_directives" in result
        assert "state_updates" in result
        assert "telemetry" in result

        assert result["content_plan"] == []
        assert result["style_directives"]["tempo"] == "medium"
        assert result["style_directives"]["length"] == "short"
        assert result["state_updates"]["trust_delta"] == 0.0
        assert result["state_updates"]["fatigue_delta"] == 0.0
        assert result["telemetry"]["chosen_ids"] == []

    def test_non_string_content_plan_items_removed(self):
        """Test non-string items in content_plan are removed with warnings."""
        payload = {
            "content_plan": [
                "valid text",
                123,
                None,
                {"invalid": "dict"},
                "another valid",
            ],
        }

        result, warnings = validate_reason_payload(payload, [])

        assert result["content_plan"] == ["valid text", "another valid"]
        warning_types = [w for w in warnings if "content_plan item was not string" in w]
        assert len(warning_types) >= 2  # Should warn about non-string items
