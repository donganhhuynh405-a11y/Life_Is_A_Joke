#!/usr/bin/env python3
"""
generate_api_docs.py - Generate API documentation from FastAPI routes.

Generates:
  - OpenAPI JSON (docs/api/openapi.json)
  - Markdown API docs (docs/API_generated.md)

Usage:
    python scripts/generate_api_docs.py [--output-dir docs/api]
"""

import argparse
import json
import logging
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT_DIR))

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("generate_api_docs")


def get_fastapi_schema() -> dict:
    """Try to load the FastAPI app and extract OpenAPI schema."""
    try:
        from src.api.v1.routes import create_app

        app = create_app()
        return app.openapi()
    except ImportError:
        logger.warning("FastAPI app not importable - using static OpenAPI spec")
        return _load_static_spec()
    except Exception as e:
        logger.warning("Failed to load FastAPI app: %s - using static spec", e)
        return _load_static_spec()


def _load_static_spec() -> dict:
    """Load the existing static OpenAPI spec."""
    static_spec = ROOT_DIR / "docs" / "api" / "openapi.yaml"
    if static_spec.exists():
        try:
            import yaml  # type: ignore
            with open(static_spec) as f:
                return yaml.safe_load(f)
        except ImportError:
            pass

    # Return a minimal spec
    return {
        "openapi": "3.0.0",
        "info": {
            "title": "Trading Bot API",
            "version": "1.3.0",
            "description": "Advanced crypto trading bot REST API",
        },
        "paths": {},
    }


def schema_to_markdown(schema: dict) -> str:
    """Convert OpenAPI schema to Markdown documentation."""
    lines = []
    info = schema.get("info", {})

    lines.append(f"# {info.get('title', 'API Documentation')}")
    lines.append("")
    lines.append(f"**Version:** {info.get('version', 'N/A')}")
    lines.append("")
    if desc := info.get("description"):
        lines.append(desc)
        lines.append("")

    lines.append("## Table of Contents")
    lines.append("")

    paths = schema.get("paths", {})
    for path, methods in sorted(paths.items()):
        for method in methods:
            if method in ("get", "post", "put", "delete", "patch"):
                op = methods[method]
                tag = (op.get("tags") or ["General"])[0]
                summary = op.get("summary", path)
                anchor = f"#{method}-{path.replace('/', '-').strip('-')}"
                lines.append(f"- [{method.upper()} {path} - {summary}]({anchor})")

    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("## Endpoints")
    lines.append("")

    for path, methods in sorted(paths.items()):
        for method, op in methods.items():
            if method not in ("get", "post", "put", "delete", "patch"):
                continue

            lines.append(f"### `{method.upper()} {path}`")
            lines.append("")

            if summary := op.get("summary"):
                lines.append(f"**{summary}**")
                lines.append("")

            if desc := op.get("description"):
                lines.append(desc)
                lines.append("")

            # Parameters
            params = op.get("parameters", [])
            if params:
                lines.append("**Parameters:**")
                lines.append("")
                lines.append("| Name | In | Type | Required | Description |")
                lines.append("|------|----|------|----------|-------------|")
                for p in params:
                    name = p.get("name", "")
                    location = p.get("in", "")
                    ptype = p.get("schema", {}).get("type", "any")
                    required = "✓" if p.get("required") else ""
                    pdesc = p.get("description", "")
                    lines.append(f"| `{name}` | {location} | {ptype} | {required} | {pdesc} |")
                lines.append("")

            # Request body
            if body := op.get("requestBody"):
                lines.append("**Request Body:**")
                lines.append("")
                content = body.get("content", {})
                for ct, ct_schema in content.items():
                    lines.append(f"Content-Type: `{ct}`")
                    if schema_ref := ct_schema.get("schema"):
                        lines.append(f"```json\n{json.dumps(schema_ref, indent=2)}\n```")
                lines.append("")

            # Responses
            responses = op.get("responses", {})
            if responses:
                lines.append("**Responses:**")
                lines.append("")
                lines.append("| Code | Description |")
                lines.append("|------|-------------|")
                for code, resp in responses.items():
                    rdesc = resp.get("description", "")
                    lines.append(f"| {code} | {rdesc} |")
                lines.append("")

            lines.append("---")
            lines.append("")

    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate API documentation")
    parser.add_argument(
        "--output-dir", default="docs/api", help="Output directory for docs"
    )
    parser.add_argument(
        "--format", choices=["json", "markdown", "both"], default="both",
        help="Output format"
    )
    args = parser.parse_args()

    output_dir = ROOT_DIR / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Generating API documentation...")
    schema = get_fastapi_schema()

    if args.format in ("json", "both"):
        json_path = output_dir / "openapi.json"
        with open(json_path, "w") as f:
            json.dump(schema, f, indent=2)
        logger.info("✓ OpenAPI JSON: %s", json_path)

    if args.format in ("markdown", "both"):
        md_content = schema_to_markdown(schema)
        md_path = ROOT_DIR / "docs" / "API_generated.md"
        with open(md_path, "w") as f:
            f.write(md_content)
        logger.info("✓ Markdown docs: %s", md_path)

    logger.info("Documentation generation complete.")


if __name__ == "__main__":
    main()
