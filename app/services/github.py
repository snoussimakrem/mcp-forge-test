"""Read-only GitHub client."""

from dataclasses import dataclass
from pathlib import PurePosixPath
import base64
import httpx


class GitHubReadError(RuntimeError):
    pass


@dataclass
class RepositorySnapshot:
    full_name: str
    default_branch: str
    files: list[dict]
    readme: str | None


class GitHubReadClient:
    text_extensions = {
        ".py", ".js", ".ts", ".tsx", ".jsx",
        ".java", ".go", ".rs", ".cs",
        ".json", ".yaml", ".yml", ".toml",
        ".md", ".txt", ".html", ".css",
        ".sql", ".sh", ".dockerfile",
    }

    def __init__(self, token: str | None = None) -> None:
        headers = {"Accept": "application/vnd.github+json"}

        if token:
            headers["Authorization"] = f"Bearer {token}"

        self.client = httpx.Client(
            base_url="https://api.github.com",
            headers=headers,
            timeout=25.0,
        )

    def _get(self, path: str) -> dict | list:
        response = self.client.get(path)

        if response.status_code >= 400:
            raise GitHubReadError(
                f"GitHub returned {response.status_code}: {response.text}"
            )

        return response.json()

    def snapshot(
        self,
        repository: str,
        ref: str,
    ) -> RepositorySnapshot:
        metadata = self._get(f"/repos/{repository}")
        branch = ref or metadata["default_branch"]

        try:
            files = self._get(
                f"/repos/{repository}/contents?ref={branch}"
            )
        except GitHubReadError as error:
            if "GitHub returned 404" in str(error):
                files = []
            else:
                raise

        readme = None

        try:
            readme_data = self._get(
                f"/repos/{repository}/readme?ref={branch}"
            )

            readme = base64.b64decode(
                readme_data["content"]
            ).decode("utf-8", errors="replace")

        except GitHubReadError:
            pass

        return RepositorySnapshot(
            full_name=repository,
            default_branch=branch,
            files=files,
            readme=readme,
        )

    def repository_documents(
        self,
        repository: str,
        ref: str,
        max_files: int = 20,
        max_characters: int = 12000,
    ) -> list[dict]:
        """Read a safe, limited set of text files recursively."""

        try:
            tree = self._get(
                f"/repos/{repository}/git/trees/{ref}?recursive=1"
            )
        except GitHubReadError:
            return []

        documents = []

        for entry in tree.get("tree", []):
            if len(documents) >= max_files:
                break

            if entry["type"] != "blob":
                continue

            path = entry["path"]
            suffix = PurePosixPath(path).suffix.lower()

            if suffix not in self.text_extensions:
                continue

            try:
                file_data = self._get(
                    f"/repos/{repository}/contents/{path}?ref={ref}"
                )

                content = base64.b64decode(
                    file_data["content"]
                ).decode("utf-8", errors="replace")

                documents.append(
                    {
                        "title": path,
                        "kind": "source_code",
                        "path": path,
                        "text": content[:max_characters],
                    }
                )

            except (GitHubReadError, KeyError):
                continue

        return documents
