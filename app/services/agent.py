"""Free OpenRouter AI analysis with an offline fallback."""

import re

from openai import OpenAI, APIError

from app.config import settings


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
        client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=settings.openrouter_api_key,
        )

        completion = client.chat.completions.create(
            model=settings.openrouter_model,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a senior software engineer reviewing a repository. "
                        "You operate in read-only mode. "
                        "Never claim to have changed files, created commits, "
                        "or opened pull requests. "
                        "Give concise and safe engineering findings."
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


def propose_patch(
    objective: str,
    repository: str,
    context: list[dict],
) -> str:
    """Generate a proposal only. It never edits files or GitHub."""

    if not settings.openrouter_api_key:
        return (
            "Patch proposal unavailable: add OPENROUTER_API_KEY "
            "as a Colab Secret."
        )

    compact_context = [
        {
            "file": item.get("title"),
            "content": item.get("text", "")[:4000],
        }
        for item in context
    ]

    try:
        client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=settings.openrouter_api_key,
        )

        completion = client.chat.completions.create(
            model=settings.openrouter_model,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a cautious senior software engineer. "
                        "Return a patch proposal only. "
                        "Do not claim the proposal was applied. "
                        "Use this exact structure: "
                        "1. Problem, 2. Proposed change, "
                        "3. Files affected, 4. Unified diff, "
                        "5. Tests, 6. Risks."
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

        return completion.choices[0].message.content or (
            "The AI provider returned an empty proposal."
        )

    except APIError as error:
        return f"Patch proposal unavailable: {error}"


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
        client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=settings.openrouter_api_key,
        )

        completion = client.chat.completions.create(
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
