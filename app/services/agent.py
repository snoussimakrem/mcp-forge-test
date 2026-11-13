"""Free OpenRouter AI analysis with an offline fallback."""

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
