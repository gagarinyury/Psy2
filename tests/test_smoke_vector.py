import json
import subprocess

import pytest


@pytest.mark.anyio
async def test_vector_smoke():
    """
    Тест векторного режима smoke test через subprocess.
    """
    try:
        # Run smoke test CLI command with vector mode
        result = subprocess.run(
            [
                "poetry",
                "run",
                "python",
                "-m",
                "app.cli.smoke",
                "run",
                "--vector",
                "--trust-a",
                "0.5",
                "--trust-b",
                "0.5",
                "--json-only",
            ],
            capture_output=True,
            text=True,
            timeout=120,  # 2 minute timeout
            cwd=".",
        )

        # Check that command succeeded
        assert (
            result.returncode == 0
        ), f"Vector smoke test failed with return code {result.returncode}. stderr: {result.stderr}"

        # Parse JSON output - find the last valid JSON line
        # (there may be structured logging mixed in)
        lines = result.stdout.strip().split("\n")
        report = None

        # Try parsing lines from the end to find the smoke test report JSON
        for line in reversed(lines):
            if not line.strip():
                continue
            try:
                potential_report = json.loads(line)
                # Check if this looks like a smoke test report
                if (
                    isinstance(potential_report, dict)
                    and "status" in potential_report
                    and "mode" in potential_report
                ):
                    report = potential_report
                    break
            except json.JSONDecodeError:
                continue

        if report is None:
            pytest.fail(
                f"Could not find valid smoke test report in output. Lines: {len(lines)}. Last few lines: {lines[-3:] if len(lines) >= 3 else lines}"
            )

        # Validate report structure
        assert "status" in report, "Report missing 'status' field"
        assert (
            report["status"] == "success"
        ), f"Vector smoke test reported failure: {report.get('error', 'Unknown error')}"

        # Check mode
        assert "mode" in report, "Report missing 'mode' field"
        assert (
            report["mode"] == "vector"
        ), f"Expected mode 'vector', got {report['mode']}"

        # Validate basic structure
        assert "case_id" in report, "Report missing 'case_id' field"
        assert "session_id" in report, "Report missing 'session_id' field"
        assert "turns" in report, "Report missing 'turns' field"
        assert "db_counts" in report, "Report missing 'db_counts' field"
        assert "embed_stats" in report, "Report missing 'embed_stats' field"

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

        assert (
            "спите" in turn1["utterance"]
        ), f"Turn 1 should be about sleep, got: {turn1['utterance']}"
        assert (
            turn1["risk"] == "none"
        ), f"Turn 1 risk should be 'none', got: {turn1['risk']}"
        assert isinstance(
            turn1["used_fragments"], list
        ), "Turn 1 used_fragments should be a list"
        assert (
            len(turn1["used_fragments"]) > 0
        ), "Turn 1 used_fragments should not be empty in vector mode"

        # Validate turn 2 (suicide risk question)
        turn2 = turns[1]
        assert "utterance" in turn2, "Turn 2 missing 'utterance'"
        assert "intent" in turn2, "Turn 2 missing 'intent'"
        assert "risk" in turn2, "Turn 2 missing 'risk'"
        assert "used_fragments" in turn2, "Turn 2 missing 'used_fragments'"

        assert (
            "суициде" in turn2["utterance"]
        ), f"Turn 2 should be about suicide, got: {turn2['utterance']}"
        assert (
            turn2["risk"] == "acute"
        ), f"Turn 2 risk should be 'acute', got: {turn2['risk']}"
        assert isinstance(
            turn2["used_fragments"], list
        ), "Turn 2 used_fragments should be a list"

        # Validate DB counts
        db_counts = report["db_counts"]
        expected_tables = ["cases", "kb_fragments", "sessions", "telemetry_turns"]
        for table in expected_tables:
            assert table in db_counts, f"DB counts missing '{table}'"
            assert isinstance(
                db_counts[table], int
            ), f"DB count for '{table}' should be integer"
            assert (
                db_counts[table] >= 0
            ), f"DB count for '{table}' should be non-negative"

        # Check that we have at least some data
        assert db_counts["cases"] >= 1, "Should have at least 1 case"
        assert db_counts["kb_fragments"] >= 1, "Should have at least 1 KB fragment"
        assert db_counts["sessions"] >= 1, "Should have at least 1 session"
        assert (
            db_counts["telemetry_turns"] >= 2
        ), "Should have at least 2 telemetry turns"

        # Validate embed_stats
        embed_stats = report["embed_stats"]
        assert "processed" in embed_stats, "embed_stats missing 'processed'"
        assert "dim" in embed_stats, "embed_stats missing 'dim'"
        assert isinstance(
            embed_stats["processed"], int
        ), "embed_stats['processed'] should be int"
        assert isinstance(embed_stats["dim"], int), "embed_stats['dim'] should be int"

        print(
            f"✓ Vector smoke test passed: {db_counts['telemetry_turns']} turns recorded"
        )
        print(
            f"✓ Embeddings: {embed_stats['processed']} processed, {embed_stats['dim']} dimensions"
        )

    except subprocess.TimeoutExpired:
        pytest.fail("Vector smoke test timed out after 120 seconds")
    except Exception as e:
        pytest.fail(f"Vector smoke test failed with exception: {e}")


@pytest.mark.anyio
async def test_vector_smoke_json_structure():
    """
    Проверяет структуру JSON output smoke test в векторном режиме.
    """
    try:
        result = subprocess.run(
            [
                "poetry",
                "run",
                "python",
                "-m",
                "app.cli.smoke",
                "run",
                "--vector",
                "--json-only",
            ],
            capture_output=True,
            text=True,
            timeout=120,
            cwd=".",
        )

        assert result.returncode == 0, f"Command failed: {result.stderr}"

        # Find the smoke test report JSON (ignore logging output)
        lines = result.stdout.strip().split("\n")
        report = None

        for line in reversed(lines):
            if not line.strip():
                continue
            try:
                potential_report = json.loads(line)
                if (
                    isinstance(potential_report, dict)
                    and "status" in potential_report
                    and "mode" in potential_report
                ):
                    report = potential_report
                    break
            except json.JSONDecodeError:
                continue

        assert (
            report is not None
        ), f"Could not find smoke test report in {len(lines)} lines"
        assert "status" in report, "JSON output should contain status field"
        assert "mode" in report, "JSON output should contain mode field"
        assert report["mode"] == "vector", "Mode should be 'vector'"

    except Exception as e:
        pytest.fail(f"JSON structure test failed: {e}")
