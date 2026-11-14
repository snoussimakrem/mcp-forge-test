"""Free OpenRouter AI analysis with offline-safe fallbacks."""

import json
import re

from openai import APIError, OpenAI

from app.config import settings


def openrouter_client() -> OpenAI:
    return OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=settings.openrouter_api_key,
    )


def summarize(
    objective: str,
    repository: str,
    context: list[dict],
) -> str:
    if not settings.openrouter_api_key:
        return (
            f"Offline analysis prepared for {repository}. "
            "Add OPENROUTER_API_KEY as a Colab Secret "
            "to enable free AI analysis."
        )

    try:
        completion = openrouter_client().chat.completions.create(
            model=settings.openrouter_model,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a senior software engineer reviewing a repository. "
                        "You operate in read-only mode. "
                        "Never claim to have changed files, created commits, "
                        "or opened pull requests. Give concise safe findings."
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        f"Objective:\n{objective}\n\n"
                        f"Repository:\n{repository}\n\n"
                        f"Retrieved context:\n{context}"
                    ),
                },
            ],
        )

        return completion.choices[0].message.content or (
            "The free model returned no text."
        )

    except APIError as error:
        return (
            f"Free AI analysis is temporarily unavailable: {error}. "
            "Repository inspection still completed safely."
        )


def normalize_changes(value: object) -> list[dict]:
    """Keep only valid, reviewable AI change objects."""

    if not isinstance(value, list):
        return []

    changes = []

    for item in value:
        if not isinstance(item, dict):
            continue

        path = item.get("path")
        rationale = item.get("rationale")
        unified_diff = item.get("unified_diff")

        if all(
            isinstance(field, str)
            for field in (path, rationale, unified_diff)
        ):
            changes.append(
                {
                    "path": path[:250],
                    "rationale": rationale[:3000],
                    "unified_diff": unified_diff[:12000],
                }
            )

    return changes[:5]


def normalize_strings(value: object, limit: int = 10) -> list[str]:
    if not isinstance(value, list):
        return []

    return [
        item[:3000]
        for item in value
        if isinstance(item, str)
    ][:limit]


def propose_structured(
    objective: str,
    repository: str,
    context: list[dict],
) -> dict:
    """Create a reviewable proposal. It never edits files or GitHub."""

    if not settings.openrouter_api_key:
        return {
            "summary": (
                "Structured proposal unavailable: add OPENROUTER_API_KEY "
                "as a Colab Secret."
            ),
            "changes": [],
            "tests": [],
            "risks": ["No AI provider is configured."],
            "provider": "offline",
        }

    compact_context = [
        {
            "file": item.get("title"),
            "content": item.get("text", "")[:4000],
        }
        for item in context
    ]

    try:
        completion = openrouter_client().chat.completions.create(
            model=settings.openrouter_model,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a cautious senior software engineer. "
                        "Return valid JSON only, with exactly this shape: "
                        '{"summary":"string","changes":[{"path":"string",'
                        '"rationale":"string","unified_diff":"string"}],'
                        '"tests":["string"],"risks":["string"]}. '
                        "Propose reviewable unified diffs only. "
                        "Never claim files were changed, commits were created, "
                        "or a pull request was opened."
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        f"Repository: {repository}\n"
                        f"Objective: {objective}\n"
                        f"Relevant source context: {compact_context}"
                    ),
                },
            ],
        )

        raw = completion.choices[0].message.content or "{}"
        raw = raw.strip()

        if raw.startswith("```json"):
            raw = raw[len("```json"):]

        if raw.startswith("```"):
            raw = raw[len("```"):]

        if raw.endswith("```"):
            raw = raw[:-3]

        parsed = json.loads(raw.strip())

        return {
            "summary": str(parsed.get("summary", "No summary returned."))[:5000],
            "changes": normalize_changes(parsed.get("changes")),
            "tests": normalize_strings(parsed.get("tests")),
            "risks": normalize_strings(parsed.get("risks")),
            "provider": "openrouter",
        }

    except (APIError, ValueError, json.JSONDecodeError) as error:
        return {
            "summary": (
                "The AI provider did not return valid structured JSON. "
                "No proposal was applied."
            ),
            "changes": [],
            "tests": [],
            "risks": [str(error)],
            "provider": "openrouter",
        }


def redact_ci_log(log: str, limit: int = 12000) -> str:
    """Redact common credential formats before returning or sending logs."""

    redacted = re.sub(
        r"(?i)\b(OPENAI_API_KEY|OPENROUTER_API_KEY|GITHUB_TOKEN|"
        r"AWS_SECRET_ACCESS_KEY|AWS_ACCESS_KEY_ID|PASSWORD|TOKEN)"
        r"\s*=\s*[^\s]+",
        r"\1=[REDACTED]",
        log,
    )

    redacted = re.sub(
        r"\b(?:ghp|github_pat|sk-or-v1|sk)-[A-Za-z0-9_-]{8,}\b",
        "[REDACTED_TOKEN]",
        redacted,
    )

    return redacted[:limit]


def diagnose_ci_failure(
    repository: str,
    run_id: int,
    failed_logs: list[dict],
) -> str:
    """Return a proposal for a failed CI run. Never applies it."""

    safe_logs = [
        {
            "job": item["job"],
            "log": redact_ci_log(item["log"]),
        }
        for item in failed_logs
    ]

    if not settings.openrouter_api_key:
        return (
            "AI diagnosis is unavailable because OPENROUTER_API_KEY "
            "is not configured. The failed job names and redacted logs "
            "were retrieved safely."
        )

    try:
        completion = openrouter_client().chat.completions.create(
            model=settings.openrouter_model,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a cautious CI engineer. Analyze only the "
                        "provided redacted GitHub Actions logs. Return exactly: "
                        "1. Root cause, 2. Evidence, 3. Proposed fix, "
                        "4. Files affected, 5. Unified diff proposal, "
                        "6. Validation steps, 7. Risks. "
                        "Never claim files were changed or a pull request was created."
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        f"Repository: {repository}\n"
                        f"Workflow run: {run_id}\n"
                        f"Failed-job logs: {safe_logs}"
                    ),
                },
            ],
        )

        return completion.choices[0].message.content or (
            "The AI provider returned an empty diagnosis."
        )

    except APIError as error:
        return f"AI diagnosis is temporarily unavailable: {error}"
