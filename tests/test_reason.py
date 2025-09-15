from app.orchestrator.nodes.reason import reason


class TestReason:
    """Test suite for reason function."""

    def test_reason_with_candidates(self):
        """Test reasoning with non-empty candidates - should have positive trust delta."""
        case_truth = {"case_id": "test-case"}
        session_state = {"trust": 0.5, "fatigue": 0.2}
        candidates = [
            {
                "id": "fragment-1",
                "type": "insight",
                "text": "Patient shows signs of anxiety",
                "metadata": {"topic": "mood"},
            },
            {
                "id": "fragment-2",
                "type": "background",
                "text": "Family history of depression",
                "metadata": {"topic": "family"},
            },
            {
                "id": "fragment-3",
                "type": "insight",
                "text": "Sleep patterns disrupted",
                "metadata": {"topic": "sleep"},
            },
        ]
        policies = {"style_profile": {"tempo": "slow", "length": "medium"}}

        result = reason(case_truth, session_state, candidates, policies)

        # Check structure
        assert isinstance(result, dict)
        assert "content_plan" in result
        assert "distortions_plan" in result
        assert "style_directives" in result
        assert "state_updates" in result
        assert "telemetry" in result

        # Check content plan contains first 1-2 texts
        assert len(result["content_plan"]) <= 2
        assert len(result["content_plan"]) > 0
        assert result["content_plan"][0] == "Patient shows signs of anxiety"
        assert result["content_plan"][1] == "Family history of depression"

        # Check trust delta is positive when candidates exist
        assert result["state_updates"]["trust_delta"] == 0.02

        # Check telemetry includes chosen IDs
        assert result["telemetry"]["chosen_ids"] == ["fragment-1", "fragment-2"]
        assert result["telemetry"]["candidates_count"] == 3
        assert result["telemetry"]["chosen_count"] == 2

        # Check distortions plan is empty as per spec
        assert result["distortions_plan"] == []

    def test_reason_empty_candidates(self):
        """Test reasoning with empty candidates - should have negative trust delta."""
        case_truth = {"case_id": "test-case"}
        session_state = {"trust": 0.5, "fatigue": 0.2}
        candidates = []
        policies = {"style_profile": {"tempo": "fast", "length": "short"}}

        result = reason(case_truth, session_state, candidates, policies)

        # Check structure
        assert isinstance(result, dict)

        # Check content plan is empty
        assert result["content_plan"] == []

        # Check trust delta is negative when no candidates
        assert result["state_updates"]["trust_delta"] == -0.01

        # Check telemetry reflects empty state
        assert result["telemetry"]["chosen_ids"] == []
        assert result["telemetry"]["candidates_count"] == 0
        assert result["telemetry"]["chosen_count"] == 0
        assert result["telemetry"]["content_plan_size"] == 0

    def test_content_plan_selection(self):
        """Test that content plan correctly selects first 1-2 texts from candidates."""
        case_truth = {}
        session_state = {}

        # Test with single candidate
        single_candidate = [{"id": "single", "text": "Only text available", "type": "insight"}]

        result = reason(case_truth, session_state, single_candidate, {})
        assert len(result["content_plan"]) == 1
        assert result["content_plan"][0] == "Only text available"
        assert result["telemetry"]["chosen_ids"] == ["single"]

        # Test with multiple candidates - should take first 2
        multiple_candidates = [
            {"id": "first", "text": "First text", "type": "insight"},
            {"id": "second", "text": "Second text", "type": "background"},
            {"id": "third", "text": "Third text", "type": "insight"},
            {"id": "fourth", "text": "Fourth text", "type": "background"},
        ]

        result = reason(case_truth, session_state, multiple_candidates, {})
        assert len(result["content_plan"]) == 2
        assert result["content_plan"][0] == "First text"
        assert result["content_plan"][1] == "Second text"
        assert result["telemetry"]["chosen_ids"] == ["first", "second"]

    def test_style_directives_from_policies(self):
        """Test that style directives are correctly extracted from policies."""
        case_truth = {}
        session_state = {}
        candidates = []

        # Test with complete style profile
        policies_with_style = {
            "style_profile": {
                "tempo": "slow",
                "length": "long",
                "register": "formal",  # Should be ignored as not in output spec
            }
        }

        result = reason(case_truth, session_state, candidates, policies_with_style)
        assert result["style_directives"]["tempo"] == "slow"
        assert result["style_directives"]["length"] == "long"
        # Should only contain tempo and length
        assert len(result["style_directives"]) == 2

        # Test with missing style profile - should use defaults
        policies_no_style = {}
        result = reason(case_truth, session_state, candidates, policies_no_style)
        assert result["style_directives"]["tempo"] == "medium"
        assert result["style_directives"]["length"] == "short"

        # Test with None policies
        result = reason(case_truth, session_state, candidates, None)
        assert result["style_directives"]["tempo"] == "medium"
        assert result["style_directives"]["length"] == "short"

        # Test with partial style profile
        policies_partial = {
            "style_profile": {
                "tempo": "fast"
                # length missing - should use default
            }
        }
        result = reason(case_truth, session_state, candidates, policies_partial)
        assert result["style_directives"]["tempo"] == "fast"
        assert result["style_directives"]["length"] == "short"  # default

    def test_candidates_without_text_field(self):
        """Test handling of candidates that don't have text field."""
        case_truth = {}
        session_state = {}
        candidates = [
            {
                "id": "no-text-1",
                "type": "insight",
                # Missing 'text' field
            },
            {"id": "has-text", "type": "background", "text": "This has text"},
        ]
        policies = {}

        result = reason(case_truth, session_state, candidates, policies)

        # Should only include candidates with text field
        assert len(result["content_plan"]) == 1
        assert result["content_plan"][0] == "This has text"
        assert result["telemetry"]["chosen_ids"] == ["has-text"]

    def test_candidates_without_id_field(self):
        """Test handling of candidates without id field."""
        case_truth = {}
        session_state = {}
        candidates = [
            {
                "type": "insight",
                "text": "Text without ID",
                # Missing 'id' field
            }
        ]
        policies = {}

        result = reason(case_truth, session_state, candidates, policies)

        # Should still process but use "unknown" as id
        assert len(result["content_plan"]) == 1
        assert result["content_plan"][0] == "Text without ID"
        assert result["telemetry"]["chosen_ids"] == ["unknown"]

    def test_state_updates_structure(self):
        """Test that state updates have correct structure."""
        result = reason({}, {}, [], {})

        assert "state_updates" in result
        state_updates = result["state_updates"]

        assert "trust_delta" in state_updates
        assert "fatigue_delta" in state_updates

        assert isinstance(state_updates["trust_delta"], (int, float))
        assert isinstance(state_updates["fatigue_delta"], (int, float))

        # Fatigue delta should be 0 in current implementation
        assert state_updates["fatigue_delta"] == 0.0

    def test_telemetry_structure(self):
        """Test that telemetry has correct structure and data types."""
        candidates = [{"id": "test-1", "text": "Test text", "type": "insight"}]

        result = reason({}, {}, candidates, {})

        assert "telemetry" in result
        telemetry = result["telemetry"]

        # Check required fields exist
        assert "candidates_count" in telemetry
        assert "chosen_count" in telemetry
        assert "chosen_ids" in telemetry
        assert "content_plan_size" in telemetry

        # Check data types
        assert isinstance(telemetry["candidates_count"], int)
        assert isinstance(telemetry["chosen_count"], int)
        assert isinstance(telemetry["chosen_ids"], list)
        assert isinstance(telemetry["content_plan_size"], int)

        # Check values match expectations
        assert telemetry["candidates_count"] == 1
        assert telemetry["chosen_count"] == 1
        assert telemetry["content_plan_size"] == 1

    def test_empty_parameters(self):
        """Test handling of empty/None parameters."""
        # All parameters empty
        result = reason({}, {}, [], {})

        # Should return valid structure even with empty inputs
        assert isinstance(result, dict)
        assert result["content_plan"] == []
        assert result["distortions_plan"] == []
        assert result["state_updates"]["trust_delta"] == -0.01  # negative for empty candidates
        assert result["telemetry"]["candidates_count"] == 0

    def test_return_structure_completeness(self):
        """Test that function always returns complete expected structure."""
        result = reason({}, {}, [], {})

        # Check all required keys present
        required_keys = [
            "content_plan",
            "distortions_plan",
            "style_directives",
            "state_updates",
            "telemetry",
        ]
        for key in required_keys:
            assert key in result, f"Missing required key: {key}"

        # Check data types
        assert isinstance(result["content_plan"], list)
        assert isinstance(result["distortions_plan"], list)
        assert isinstance(result["style_directives"], dict)
        assert isinstance(result["state_updates"], dict)
        assert isinstance(result["telemetry"], dict)
