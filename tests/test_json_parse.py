"""
Tests for JSON parsing functions.

Tests pure functions without async, following specification scenarios.
"""

import pytest
from app.llm.json_parse import (
    extract_json_blocks,
    parse_llm_json,
    normalize_reason_payload,
)


class TestExtractJsonBlocks:
    def test_json_code_block_extracted(self):
        text = """Here is the result:
```json
{"content_plan": ["test"], "style_directives": {"tempo": "fast"}}
```
Some more text."""

        blocks = extract_json_blocks(text)
        assert len(blocks) >= 1
        assert '"content_plan"' in blocks[0]
        assert '"test"' in blocks[0]

    def test_general_code_block_extracted(self):
        text = """The response:
```
{"content_plan": ["general test"], "telemetry": {"chosen_ids": []}}
```
End."""

        blocks = extract_json_blocks(text)
        assert len(blocks) >= 1
        assert '"content_plan"' in blocks[0]

    def test_raw_braces_extracted(self):
        text = """Here is the plan: {"content_plan": ["raw test"], "state_updates": {"trust_delta": 0.1}} - that's it."""

        blocks = extract_json_blocks(text)
        assert len(blocks) >= 1
        assert '"content_plan"' in blocks[0]
        assert '"raw test"' in blocks[0]

    def test_nested_braces_handled(self):
        text = """Complex: {"outer": {"inner": {"deep": "value"}}, "list": [1, 2, 3]} done."""

        blocks = extract_json_blocks(text)
        assert len(blocks) >= 1
        assert '"outer"' in blocks[0]
        assert '"inner"' in blocks[0]

    def test_no_json_returns_empty(self):
        text = "No JSON here at all, just plain text."

        blocks = extract_json_blocks(text)
        assert blocks == []


class TestParseJsonLlm:
    def test_valid_json_block_parses(self):
        text = """```json
{
    "content_plan": ["test message"],
    "style_directives": {"tempo": "medium", "length": "short"},
    "state_updates": {"trust_delta": 0.1, "fatigue_delta": -0.05},
    "telemetry": {"chosen_ids": ["id1", "id2"]}
}
```"""

        result = parse_llm_json(text)
        assert isinstance(result, dict)
        assert result["content_plan"] == ["test message"]
        assert result["style_directives"]["tempo"] == "medium"
        assert result["state_updates"]["trust_delta"] == 0.1
        assert result["telemetry"]["chosen_ids"] == ["id1", "id2"]

    def test_code_block_without_json_marker_parses(self):
        text = """```
{
    "content_plan": ["no json marker"],
    "style_directives": {"tempo": "slow"}
}
```"""

        result = parse_llm_json(text)
        assert isinstance(result, dict)
        assert result["content_plan"] == ["no json marker"]
        assert result["style_directives"]["tempo"] == "slow"

    def test_raw_text_with_braces_parses(self):
        text = """Here is the response: {"content_plan": ["raw response"], "telemetry": {"chosen_ids": []}} - end."""

        result = parse_llm_json(text)
        assert isinstance(result, dict)
        assert result["content_plan"] == ["raw response"]

    def test_json5_with_comments_and_trailing_commas_parses(self):
        text = """```json
{
    // This is a comment
    "content_plan": ["json5 test"], // Another comment
    "style_directives": {
        "tempo": "fast",
        "length": "long", // Trailing comma next
    },
    "telemetry": {"chosen_ids": []}, // Final trailing comma
}
```"""

        result = parse_llm_json(text)
        assert isinstance(result, dict)
        assert result["content_plan"] == ["json5 test"]
        assert result["style_directives"]["tempo"] == "fast"

    def test_malformed_json_raises_error(self):
        text = "No JSON content here, just garbage text without braces."

        with pytest.raises(ValueError, match="No JSON candidates found"):
            parse_llm_json(text)

    def test_invalid_json_raises_error(self):
        text = """```json
{
    "content_plan": [unclosed quote and missing bracket
}
```"""

        with pytest.raises(ValueError, match="Failed to parse JSON"):
            parse_llm_json(text)


class TestNormalizeReasonPayload:
    def test_complete_payload_normalized(self):
        data = {
            "content_plan": ["message 1", "message 2"],
            "style_directives": {"tempo": "fast", "length": "long"},
            "state_updates": {"trust_delta": 0.2, "fatigue_delta": -0.1},
            "telemetry": {"chosen_ids": ["frag1", "frag2"]},
            "extra_field": "ignored",
        }

        result = normalize_reason_payload(data)

        assert result["content_plan"] == ["message 1", "message 2"]
        assert result["style_directives"] == {"tempo": "fast", "length": "long"}
        assert result["state_updates"] == {"trust_delta": 0.2, "fatigue_delta": -0.1}
        assert result["telemetry"] == {"chosen_ids": ["frag1", "frag2"]}
        assert "extra_field" not in result

    def test_missing_fields_get_defaults(self):
        data = {"content_plan": ["test"]}

        result = normalize_reason_payload(data)

        assert result["content_plan"] == ["test"]
        assert result["style_directives"] == {"tempo": "medium", "length": "short"}
        assert result["state_updates"] == {"trust_delta": 0.0, "fatigue_delta": 0.0}
        assert result["telemetry"] == {"chosen_ids": []}

    def test_wrong_types_corrected(self):
        data = {
            "content_plan": "single string instead of list",
            "style_directives": "not a dict",
            "state_updates": {"trust_delta": "not a number", "fatigue_delta": None},
            "telemetry": {"chosen_ids": "not a list"},
        }

        result = normalize_reason_payload(data)

        assert result["content_plan"] == ["single string instead of list"]
        assert result["style_directives"] == {"tempo": "medium", "length": "short"}
        assert result["state_updates"] == {"trust_delta": 0.0, "fatigue_delta": 0.0}
        assert result["telemetry"] == {"chosen_ids": []}

    def test_partial_nested_fields_completed(self):
        data = {
            "content_plan": ["test"],
            "style_directives": {"tempo": "custom"},  # missing length
            "state_updates": {"trust_delta": 0.5},  # missing fatigue_delta
            "telemetry": {},  # missing chosen_ids
        }

        result = normalize_reason_payload(data)

        assert result["style_directives"]["tempo"] == "custom"
        assert result["style_directives"]["length"] == "short"
        assert result["state_updates"]["trust_delta"] == 0.5
        assert result["state_updates"]["fatigue_delta"] == 0.0
        assert result["telemetry"]["chosen_ids"] == []

    def test_empty_dict_raises_error(self):
        with pytest.raises(ValueError, match="Empty dictionary provided"):
            normalize_reason_payload({})

    def test_meaningless_payload_raises_error(self):
        data = {
            "content_plan": [],
            "state_updates": {"trust_delta": 0.0, "fatigue_delta": 0.0},
        }

        with pytest.raises(ValueError, match="no meaningful data"):
            normalize_reason_payload(data)
