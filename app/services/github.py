"""Read-only GitHub client."""

from dataclasses import dataclass
from io import BytesIO
from zipfile import BadZipFile, ZipFile
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

    def _get_text(self, path: str) -> str:
        """Read an API endpoint that returns plain text or a ZIP log archive."""

        response = self.client.get(path, follow_redirects=True)

        if response.status_code >= 400:
            raise GitHubReadError(
                f"GitHub returned {response.status_code}: {response.text}"
            )

        content = response.content

        if content.startswith(b"PK"):
            try:
                with ZipFile(BytesIO(content)) as archive:
                    parts = []
                    total_size = 0

                    for entry in archive.infolist()[:20]:
                        if entry.is_dir() or entry.file_size > 1_000_000:
                            continue

                        total_size += entry.file_size

                        if total_size > 2_000_000:
                            break

                        parts.append(
                            archive.read(entry).decode(
                                "utf-8",
                                errors="replace",
                            )
                        )

                    return "\n".join(parts)

            except BadZipFile as error:
                raise GitHubReadError(
                    "GitHub returned an unreadable workflow log archive."
                ) from error

        return content.decode("utf-8", errors="replace")

    def workflow_job_log(
        self,
        repository: str,
        job_id: int,
    ) -> str:
        """Read one GitHub Actions job log without modifying GitHub."""

        return self._get_text(
            f"/repos/{repository}/actions/jobs/{job_id}/logs"
        )

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

    def workflow_run_details(
        self,
        repository: str,
        run_id: int,
    ) -> tuple[dict, list[dict]]:
        """Return one workflow run and its jobs, read-only."""

        run = self._get(
            f"/repos/{repository}/actions/runs/{run_id}"
        )

        jobs_payload = self._get(
            f"/repos/{repository}/actions/runs/{run_id}/jobs?per_page=100"
        )

        return run, jobs_payload.get("jobs", [])
