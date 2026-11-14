import logging
import uuid
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

from app.config import settings
from app.models import (
    AnalyzeRequest,
    AnalyzeResponse,
    DraftPullRequestRequest,
    ErrorResponse,
    Finding,
    ProposalResponse,
)
from app.services.agent import (
    summarize,
    propose_structured,
    diagnose_ci_failure,
    redact_ci_log,
)
from app.services.github import GitHubReadClient, GitHubReadError
from app.services.memory import MemoryStore
from app.logging_config import configure_logging


logger = configure_logging(settings.log_level)

app = FastAPI(
    title="MCP Forge API",
    version="0.2.0",
)

memory = MemoryStore(
    collection=settings.qdrant_collection,
    url=settings.qdrant_url,
)



@app.middleware("http")
async def attach_request_id(request: Request, call_next):
    request_id = request.headers.get("X-Request-ID", str(uuid.uuid4()))
    request.state.request_id = request_id

    response = await call_next(request)

    response.headers["X-Request-ID"] = request_id

    logger.info(
        "%s %s -> %s request_id=%s",
        request.method,
        request.url.path,
        response.status_code,
        request_id,
    )

    return response


@app.exception_handler(RequestValidationError)
async def validation_error_handler(
    request: Request,
    error: RequestValidationError,
):
    request_id = getattr(request.state, "request_id", "unknown")

    logger.warning(
        "validation_error request_id=%s errors=%s",
        request_id,
        error.errors(),
    )

    return JSONResponse(
        status_code=422,
        content=ErrorResponse(
            error_id=request_id,
            message="Validation error",
        ).model_dump(),
    )


@app.exception_handler(HTTPException)
async def http_error_handler(
    request: Request,
    error: HTTPException,
):
    request_id = getattr(request.state, "request_id", "unknown")

    logger.warning(
        "http_error request_id=%s status=%s",
        request_id,
        error.status_code,
    )

    return JSONResponse(
        status_code=error.status_code,
        content=ErrorResponse(
            error_id=request_id,
            message=str(error.detail),
        ).model_dump(),
    )


@app.exception_handler(Exception)
async def unexpected_error_handler(
    request: Request,
    error: Exception,
):
    request_id = getattr(request.state, "request_id", "unknown")

    logger.exception(
        "unexpected_error request_id=%s",
        request_id,
    )

    return JSONResponse(
        status_code=500,
        content=ErrorResponse(
            error_id=request_id,
            message="Internal server error",
        ).model_dump(),
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




@app.get("/api/ci/{owner}/{repository_name}/{run_id}")
def ci_run_details(
    owner: str,
    repository_name: str,
    run_id: int,
) -> dict:
    """Inspect one GitHub Actions run without changing GitHub."""

    repository = f"{owner}/{repository_name}"

    try:
        run, jobs = GitHubReadClient(
            settings.github_token
        ).workflow_run_details(repository, run_id)

    except GitHubReadError as error:
        raise HTTPException(
            status_code=422,
            detail=str(error),
        ) from error

    conclusion = run.get("conclusion")

    if run.get("status") != "completed":
        diagnosis = "This workflow is still running."
    elif conclusion == "success":
        diagnosis = (
            "CI completed successfully. No fix is proposed because "
            "the workflow has no reported failure."
        )
    elif conclusion in {"failure", "timed_out", "cancelled"}:
        diagnosis = (
            "CI did not complete successfully. Inspect the failed job "
            "and its failed step before generating a fix proposal."
        )
    else:
        diagnosis = (
            "CI completed with a non-success conclusion. "
            "Review the jobs and steps below."
        )

    return {
        "repository": repository,
        "run_id": run_id,
        "name": run.get("name"),
        "status": run.get("status"),
        "conclusion": conclusion,
        "branch": run.get("head_branch"),
        "event": run.get("event"),
        "url": run.get("html_url"),
        "created_at": run.get("created_at"),
        "updated_at": run.get("updated_at"),
        "diagnosis": diagnosis,
        "jobs": [
            {
                "name": job.get("name"),
                "status": job.get("status"),
                "conclusion": job.get("conclusion"),
                "started_at": job.get("started_at"),
                "completed_at": job.get("completed_at"),
                "steps": [
                    {
                        "name": step.get("name"),
                        "status": step.get("status"),
                        "conclusion": step.get("conclusion"),
                        "number": step.get("number"),
                    }
                    for step in job.get("steps", [])
                ],
            }
            for job in jobs
        ],
        "safety": "Read-only inspection: no workflow, file, commit, or pull request was changed.",
    }



@app.post("/api/ci/{owner}/{repository_name}/{run_id}/diagnose")
def diagnose_ci_run(
    owner: str,
    repository_name: str,
    run_id: int,
) -> dict:
    """Diagnose failed CI jobs and return a proposal only."""

    repository = f"{owner}/{repository_name}"
    github = GitHubReadClient(settings.github_token)

    try:
        run, jobs = github.workflow_run_details(repository, run_id)

    except GitHubReadError as error:
        raise HTTPException(
            status_code=422,
            detail=str(error),
        ) from error

    failed_jobs = [
        job for job in jobs
        if job.get("conclusion") in {"failure", "timed_out", "cancelled"}
    ]

    if not failed_jobs:
        return {
            "repository": repository,
            "run_id": run_id,
            "failure_detected": False,
            "diagnosis": (
                "No failed job was found in this workflow run. "
                "No logs were retrieved and no fix was proposed."
            ),
            "safety": (
                "Read-only: no workflow, file, branch, commit, "
                "pull request, or deployment was changed."
            ),
        }

    failed_logs = []

    for job in failed_jobs[:2]:
        try:
            failed_logs.append(
                {
                    "job": job.get("name", "Unnamed job"),
                    "log": github.workflow_job_log(
                        repository,
                        job["id"],
                    ),
                }
            )
        except GitHubReadError as error:
            failed_logs.append(
                {
                    "job": job.get("name", "Unnamed job"),
                    "log": f"Log retrieval failed: {error}",
                }
            )

    return {
        "repository": repository,
        "run_id": run_id,
        "failure_detected": True,
        "failed_jobs": [
            {
                "name": job.get("name"),
                "conclusion": job.get("conclusion"),
                "steps": [
                    {
                        "name": step.get("name"),
                        "conclusion": step.get("conclusion"),
                    }
                    for step in job.get("steps", [])
                    if step.get("conclusion") != "success"
                ],
            }
            for job in failed_jobs
        ],
        "redacted_log_previews": [
            {
                "job": item["job"],
                "log": redact_ci_log(item["log"], limit=2000),
            }
            for item in failed_logs
        ],
        "proposal": diagnose_ci_failure(
            repository=repository,
            run_id=run_id,
            failed_logs=failed_logs,
        ),
        "safety": (
            "Proposal only: no file, branch, commit, pull request, "
            "workflow rerun, or deployment was created."
        ),
    }



@app.post("/api/pull-requests/draft")
def create_draft_pull_request(
    request: DraftPullRequestRequest,
) -> dict:
    """Create a draft PR only after deliberate opt-in."""

    if not settings.allow_github_writes:
        raise HTTPException(
            status_code=403,
            detail=(
                "GitHub writes are disabled. Set ALLOW_GITHUB_WRITES=true "
                "only when you are ready to create a draft pull request."
            ),
        )

    if request.branch == request.base:
        raise HTTPException(
            status_code=422,
            detail="The proposal branch must be different from the base branch.",
        )

    github = GitHubReadClient(settings.github_token)

    try:
        github.create_branch(
            repository=request.repository,
            base=request.base,
            branch=request.branch,
        )

        for file_change in request.files:
            github.create_or_update_file(
                repository=request.repository,
                branch=request.branch,
                path=file_change.path,
                content=file_change.content,
                message=(
                    "chore: apply approved MCP Forge proposal "
                    f"to {file_change.path}"
                ),
            )

        pull_request = github.create_draft_pull_request(
            repository=request.repository,
            base=request.base,
            branch=request.branch,
            title=request.title,
            body=request.body,
        )

    except GitHubReadError as error:
        raise HTTPException(
            status_code=422,
            detail=str(error),
        ) from error

    return {
        "repository": request.repository,
        "branch": request.branch,
        "pull_request_number": pull_request["number"],
        "pull_request_url": pull_request["html_url"],
        "draft": pull_request["draft"],
        "safety": (
            "A draft pull request was created on a separate branch. "
            "Nothing was merged into the base branch."
        ),
    }


@app.post("/api/propose", response_model=ProposalResponse)
def propose(request: AnalyzeRequest) -> ProposalResponse:
    """Generate a structured proposal without changing GitHub."""

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

    proposal = propose_structured(
        objective=request.objective,
        repository=snapshot.full_name,
        context=context,
    )

    return ProposalResponse(
        repository=snapshot.full_name,
        ref=snapshot.default_branch,
        summary=proposal["summary"],
        changes=proposal["changes"],
        tests=proposal["tests"],
        risks=proposal["risks"],
        provider=proposal["provider"],
        safety=(
            "Proposal only: no file, branch, commit, pull request, "
            "or deployment was created."
        ),
    )


frontend_directory = Path(__file__).resolve().parents[1] / "frontend"

app.mount(
    "/",
    StaticFiles(
        directory=frontend_directory,
        html=True,
    ),
    name="frontend",
)
