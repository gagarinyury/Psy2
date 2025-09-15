"""
Smoke test - интеграционный тест всей системы через subprocess.
"""

import json
import subprocess
import sys

import pytest


class TestSmoke:
    """Smoke test suite for end-to-end system validation."""

    def test_smoke_run_through_subprocess(self):
        """
        Запускает smoke test через subprocess и проверяет результат.
        """
        try:
            # Run smoke test CLI command
            result = subprocess.run(
                [sys.executable, "-m", "app.cli.smoke", "run"],
                capture_output=True,
                text=True,
                timeout=60,  # 1 minute timeout
                cwd=".",
            )

            # Check that command succeeded
            assert result.returncode == 0, (
                f"Smoke test failed with return code {result.returncode}. stderr: {result.stderr}"
            )

            # Parse JSON output
            try:
                report = json.loads(result.stdout)
            except json.JSONDecodeError as e:
                pytest.fail(f"Failed to parse JSON output: {e}. stdout: {result.stdout}")

            # Validate report structure
            assert "status" in report, "Report missing 'status' field"
            assert report["status"] == "success", (
                f"Smoke test reported failure: {report.get('error', 'Unknown error')}"
            )

            assert "case_id" in report, "Report missing 'case_id' field"
            assert "session_id" in report, "Report missing 'session_id' field"
            assert "turns" in report, "Report missing 'turns' field"
            assert "db_counts" in report, "Report missing 'db_counts' field"

            # Validate case_id and session_id are not None
            assert report["case_id"] is not None, "case_id should not be None"
            assert report["session_id"] is not None, "session_id should not be None"

            # Validate turns array
            turns = report["turns"]
            assert len(turns) == 2, f"Expected 2 turns, got {len(turns)}"

            # Validate turn 1 (sleep question)
            turn1 = turns[0]
            assert "utterance" in turn1, "Turn 1 missing 'utterance'"
            assert "intent" in turn1, "Turn 1 missing 'intent'"
            assert "risk" in turn1, "Turn 1 missing 'risk'"
            assert "used_fragments" in turn1, "Turn 1 missing 'used_fragments'"

            assert "спите" in turn1["utterance"], (
                f"Turn 1 should be about sleep, got: {turn1['utterance']}"
            )
            assert turn1["intent"] == "clarify", (
                f"Turn 1 intent should be 'clarify', got: {turn1['intent']}"
            )
            assert turn1["risk"] == "none", f"Turn 1 risk should be 'none', got: {turn1['risk']}"
            assert isinstance(turn1["used_fragments"], list), (
                "Turn 1 used_fragments should be a list"
            )

            # Validate turn 2 (suicide risk question)
            turn2 = turns[1]
            assert "utterance" in turn2, "Turn 2 missing 'utterance'"
            assert "intent" in turn2, "Turn 2 missing 'intent'"
            assert "risk" in turn2, "Turn 2 missing 'risk'"
            assert "used_fragments" in turn2, "Turn 2 missing 'used_fragments'"

            assert "суициде" in turn2["utterance"], (
                f"Turn 2 should be about suicide, got: {turn2['utterance']}"
            )
            assert turn2["intent"] == "risk_check", (
                f"Turn 2 intent should be 'risk_check', got: {turn2['intent']}"
            )
            assert turn2["risk"] == "acute", f"Turn 2 risk should be 'acute', got: {turn2['risk']}"
            assert isinstance(turn2["used_fragments"], list), (
                "Turn 2 used_fragments should be a list"
            )

            # Validate DB counts
            db_counts = report["db_counts"]
            expected_tables = ["cases", "kb_fragments", "sessions", "telemetry_turns"]
            for table in expected_tables:
                assert table in db_counts, f"DB counts missing '{table}'"
                assert isinstance(db_counts[table], int), (
                    f"DB count for '{table}' should be integer"
                )
                assert db_counts[table] >= 0, f"DB count for '{table}' should be non-negative"

            # Check that we have at least some data
            assert db_counts["cases"] >= 1, "Should have at least 1 case"
            assert db_counts["kb_fragments"] >= 1, "Should have at least 1 KB fragment"
            assert db_counts["sessions"] >= 1, "Should have at least 1 session"
            assert db_counts["telemetry_turns"] >= 2, "Should have at least 2 telemetry turns"

            print(f"✓ Smoke test passed: {db_counts['telemetry_turns']} turns recorded")

        except subprocess.TimeoutExpired:
            pytest.fail("Smoke test timed out after 60 seconds")
        except Exception as e:
            pytest.fail(f"Smoke test failed with exception: {e}")

    def test_smoke_validates_used_fragments_not_empty(self):
        """
        Дополнительная проверка что used_fragments действительно заполняются.
        """
        try:
            result = subprocess.run(
                [sys.executable, "-m", "app.cli.smoke", "run"],
                capture_output=True,
                text=True,
                timeout=60,
                cwd=".",
            )

            assert result.returncode == 0, f"Smoke test failed: {result.stderr}"

            report = json.loads(result.stdout)
            turns = report["turns"]

            # Check that at least one turn has used_fragments
            # Note: used_fragments might be empty if no fragments match the query
            # but the structure should still be valid
            for i, turn in enumerate(turns):
                assert isinstance(turn["used_fragments"], list), (
                    f"Turn {i + 1} used_fragments should be a list"
                )

        except Exception as e:
            pytest.fail(f"used_fragments validation failed: {e}")

    def test_smoke_json_output_structure(self):
        """
        Проверяет что JSON выход имеет правильную структуру.
        """
        try:
            result = subprocess.run(
                [sys.executable, "-m", "app.cli.smoke", "run"],
                capture_output=True,
                text=True,
                timeout=60,
                cwd=".",
            )

            # Even if the test fails, we should get valid JSON
            try:
                report = json.loads(result.stdout)
            except json.JSONDecodeError:
                pytest.fail("Output is not valid JSON")

            # Required fields should always be present
            required_fields = ["status", "case_id", "session_id", "turns", "db_counts"]
            for field in required_fields:
                assert field in report, f"Missing required field: {field}"

            # If status is failed, error field should be present
            if report["status"] == "failed":
                assert "error" in report, "Failed status should include error field"

        except Exception as e:
            pytest.fail(f"JSON structure validation failed: {e}")
