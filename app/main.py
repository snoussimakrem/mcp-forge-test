from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles

from app.config import settings
from app.models import AnalyzeRequest, AnalyzeResponse, Finding
from app.services.agent import summarize, propose_patch
from app.services.github import GitHubReadClient, GitHubReadError
from app.services.memory import MemoryStore


app = FastAPI(
    title="MCP Forge API",
    version="0.2.0",
)

memory = MemoryStore(
    collection=settings.qdrant_collection,
    url=settings.qdrant_url,
)


@app.get("/health")
def health() -> dict:
    return {
        "status": "ok",
        "mode": "read-only",
        "github_writes": settings.allow_github_writes,
    }


@app.post("/api/analyze", response_model=AnalyzeResponse)
def analyze(request: AnalyzeRequest) -> AnalyzeResponse:
    github = GitHubReadClient(settings.github_token)

    try:
        snapshot = github.snapshot(
            repository=request.repository,
            ref=request.ref,
        )

        documents = github.repository_documents(
            repository=request.repository,
            ref=snapshot.default_branch,
        )

    except GitHubReadError as error:
        raise HTTPException(
            status_code=422,
            detail=str(error),
        ) from error

    if snapshot.readme:
        documents.append(
            {
                "title": "README.md",
                "kind": "documentation",
                "path": "README.md",
                "text": snapshot.readme,
            }
        )

    indexed = memory.index(documents)

    related = (
        memory.search(request.objective)
        if indexed
        else []
    )

    findings = []

    if not snapshot.readme:
        findings.append(
            Finding(
                severity="medium",
                title="No README detected",
                detail="Add project setup, architecture, and test instructions.",
            )
        )

    manifests = {
        "pyproject.toml",
        "package.json",
        "pom.xml",
        "build.gradle",
    }

    if not any(
        entry["name"].lower() in manifests
        for entry in snapshot.files
    ):
        findings.append(
            Finding(
                severity="low",
                title="Build manifest not detected",
                detail=(
                    "Add reproducible build and test configuration "
                    "before enabling automated workflows."
                ),
            )
        )

    if indexed:
        findings.append(
            Finding(
                severity="info",
                title="Repository context indexed",
                detail=(
                    f"{indexed} text files were embedded in Qdrant "
                    "for semantic retrieval."
                ),
            )
        )

    return AnalyzeResponse(
        repository=snapshot.full_name,
        ref=snapshot.default_branch,
        files_indexed=indexed,
        summary=summarize(
            objective=request.objective,
            repository=snapshot.full_name,
            context=related,
        ),
        findings=findings,
        write_actions_enabled=settings.allow_github_writes,
    )



@app.get("/api/ci/{owner}/{repository_name}")
def recent_ci_runs(owner: str, repository_name: str) -> dict:
    """Return recent GitHub Actions workflow runs."""

    repository = f"{owner}/{repository_name}"

    try:
        payload = GitHubReadClient(
            settings.github_token
        )._get(
            f"/repos/{repository}/actions/runs?per_page=5"
        )

    except GitHubReadError as error:
        raise HTTPException(
            status_code=422,
            detail=str(error),
        ) from error

    return {
        "repository": repository,
        "runs": [
            {
                "name": run["name"],
                "status": run["status"],
                "conclusion": run["conclusion"],
                "branch": run["head_branch"],
                "url": run["html_url"],
            }
            for run in payload.get("workflow_runs", [])
        ],
    }



@app.post("/api/propose")
def propose(request: AnalyzeRequest) -> dict:
    """Generate a reviewable patch proposal without changing GitHub."""

    github = GitHubReadClient(settings.github_token)

    try:
        snapshot = github.snapshot(
            repository=request.repository,
            ref=request.ref,
        )

        documents = github.repository_documents(
            repository=request.repository,
            ref=snapshot.default_branch,
        )

    except GitHubReadError as error:
        raise HTTPException(
            status_code=422,
            detail=str(error),
        ) from error

    if snapshot.readme:
        documents.append(
            {
                "title": "README.md",
                "kind": "documentation",
                "path": "README.md",
                "text": snapshot.readme,
            }
        )

    if documents:
        memory.index(documents)

    context = memory.search(request.objective) if documents else []

    return {
        "repository": snapshot.full_name,
        "ref": snapshot.default_branch,
        "proposal": propose_patch(
            objective=request.objective,
            repository=snapshot.full_name,
            context=context,
        ),
        "safety": (
            "Proposal only: no file, branch, commit, pull request, "
            "or deployment was created."
        ),
    }


frontend_directory = Path(__file__).resolve().parents[1] / "frontend"

app.mount(
    "/",
    StaticFiles(
        directory=frontend_directory,
        html=True,
    ),
    name="frontend",
)
