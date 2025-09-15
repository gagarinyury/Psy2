#!/usr/bin/env python3
"""
Generate API.md documentation from OpenAPI schema.

Usage:
    python scripts/gen_api_md.py --base-url http://localhost:8000
"""

import argparse
import sys
from typing import Any, Dict
from urllib.parse import urljoin

import requests


def get_openapi_schema(base_url: str) -> Dict[str, Any]:
    """Fetch OpenAPI schema from the API server."""
    health_url = urljoin(base_url, "/health")
    openapi_url = urljoin(base_url, "/openapi.json")

    try:
        # Check if server is running
        health_response = requests.get(health_url, timeout=5)
        health_response.raise_for_status()
        print(f"✅ Server is running at {base_url}")

        # Fetch OpenAPI schema
        openapi_response = requests.get(openapi_url, timeout=10)
        openapi_response.raise_for_status()

        return openapi_response.json()

    except requests.exceptions.RequestException as e:
        print(f"❌ Error connecting to {base_url}: {e}")
        print("Make sure the server is running: uvicorn app.main:app --port 8000")
        sys.exit(1)


def format_request_body(schema: Dict[str, Any], operation: Dict[str, Any]) -> str:
    """Format request body schema for documentation."""
    if "requestBody" not in operation:
        return "None"

    request_body = operation["requestBody"]
    if "content" not in request_body:
        return "None"

    content = request_body["content"]
    if "application/json" not in content:
        return "JSON"

    json_content = content["application/json"]
    if "schema" not in json_content:
        return "JSON"

    schema_ref = json_content["schema"]

    # Handle schema references
    if "$ref" in schema_ref:
        ref_path = schema_ref["$ref"]
        if ref_path.startswith("#/components/schemas/"):
            schema_name = ref_path.split("/")[-1]
            return f"`{schema_name}`"

    # Handle inline schemas
    if "type" in schema_ref and schema_ref["type"] == "object":
        return "JSON Object"

    return "JSON"


def format_response(operation: Dict[str, Any]) -> str:
    """Format response description for documentation."""
    responses = operation.get("responses", {})

    if "200" in responses:
        response_200 = responses["200"]
        description = response_200.get("description", "Success")

        # Try to get response schema
        content = response_200.get("content", {})
        if "application/json" in content:
            json_content = content["application/json"]
            if "schema" in json_content and "$ref" in json_content["schema"]:
                ref_path = json_content["schema"]["$ref"]
                if ref_path.startswith("#/components/schemas/"):
                    schema_name = ref_path.split("/")[-1]
                    return f"{description} (`{schema_name}`)"

        return description

    return "Success"


def generate_api_documentation(schema: Dict[str, Any]) -> str:
    """Generate markdown documentation from OpenAPI schema."""

    # Extract basic info
    info = schema.get("info", {})
    title = info.get("title", "API Documentation")
    description = info.get("description", "")
    version = info.get("version", "1.0.0")

    # Start building markdown
    md_lines = [
        f"# {title}",
        "",
        f"**Version:** {version}",
        "",
        description,
        "",
        "## API Endpoints",
        "",
        "| Method | Path | Description | Request Body | Response 200 |",
        "|--------|------|-------------|--------------|--------------|"
    ]

    # Process paths
    paths = schema.get("paths", {})

    # Sort paths for consistent output
    sorted_paths = sorted(paths.items())

    for path, path_operations in sorted_paths:
        # Sort methods for consistent output
        sorted_operations = sorted(path_operations.items())

        for method, operation in sorted_operations:
            if method.upper() in ["GET", "POST", "PUT", "DELETE", "PATCH"]:
                summary = operation.get("summary", operation.get("description", ""))
                request_body = format_request_body(schema, operation)
                response = format_response(operation)

                # Clean up summary - remove newlines and limit length
                summary = summary.replace("\n", " ").strip()
                if len(summary) > 80:
                    summary = summary[:77] + "..."

                md_lines.append(
                    f"| {method.upper()} | `{path}` | {summary} | {request_body} | {response} |"
                )

    # Add examples section
    md_lines.extend([
        "",
        "## Example Usage",
        "",
        "### Health Check",
        "```bash",
        "curl http://localhost:8000/health",
        "```",
        "",
        "### Create Case",
        "```bash",
        "curl -X POST http://localhost:8000/case \\",
        "  -H 'Content-Type: application/json' \\",
        "  -d '{",
        '    "case_truth": {',
        '      "dx_target": ["MDD"],',
        '      "ddx": {"MDD": 0.6},',
        '      "hidden_facts": ["family history"],',
        '      "red_flags": ["suicidal ideation"]',
        "    },",
        '    "policies": {',
        '      "disclosure_rules": {"min_trust_for_gated": 0.4},',
        '      "risk_protocol": {"trigger_keywords": ["suicide"]},',
        '      "style_profile": {"register": "colloquial"}',
        "    }",
        "  }'",
        "```",
        "",
        "### Process Turn",
        "```bash",
        "curl -X POST http://localhost:8000/turn \\",
        "  -H 'Content-Type: application/json' \\",
        "  -H 'X-Session-ID: <session_id>' \\",
        "  -d '{",
        '    "therapist_utterance": "How are you sleeping?",',
        '    "session_state": {',
        '      "affect": "neutral",',
        '      "trust": 0.5,',
        '      "fatigue": 0.1,',
        '      "access_level": 1,',
        '      "risk_status": "none",',
        '      "last_turn_summary": ""',
        "    },",
        '    "case_id": "<case_id>",',
        '    "session_id": "<session_id>"',
        "  }'",
        "```",
        "",
        "### Configure Rate Limiting",
        "```bash",
        "curl -X POST http://localhost:8000/admin/rate_limit \\",
        "  -H 'Content-Type: application/json' \\",
        "  -d '{",
        '    "enabled": true,',
        '    "session_per_min": 20,',
        '    "ip_per_min": 120,',
        '    "fail_open": false',
        "  }'",
        "```",
        "",
        "---",
        "",
        f"*Generated from OpenAPI schema on {info.get('version', 'unknown version')}*"
    ])

    return "\n".join(md_lines)


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Generate API documentation from OpenAPI schema")
    parser.add_argument(
        "--base-url",
        default="http://localhost:8000",
        help="Base URL of the API server (default: http://localhost:8000)"
    )
    parser.add_argument(
        "--output",
        default="docs/API.md",
        help="Output file path (default: docs/API.md)"
    )

    args = parser.parse_args()

    print(f"🔗 Connecting to {args.base_url}")
    schema = get_openapi_schema(args.base_url)

    print("📝 Generating API documentation...")
    documentation = generate_api_documentation(schema)

    # Ensure docs directory exists
    import os
    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    # Write documentation
    with open(args.output, "w", encoding="utf-8") as f:
        f.write(documentation)

    print(f"✅ API documentation generated: {args.output}")
    print(f"📄 Found {len(schema.get('paths', {}))} API paths")


if __name__ == "__main__":
    main()