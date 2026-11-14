from typing import Literal

from pydantic import BaseModel, Field, field_validator


class AnalyzeRequest(BaseModel):
    repository: str = Field(
        pattern=r"^[\w.-]+/[\w.-]+$",
        examples=["snoussimakrem/mcp-forge-test"],
    )
    ref: str = "main"
    objective: str = (
        "Analyze the repository and identify safe engineering improvements."
    )


class Finding(BaseModel):
    severity: str
    title: str
    detail: str


class AnalyzeResponse(BaseModel):
    repository: str
    ref: str
    files_indexed: int
    summary: str
    findings: list[Finding]
    write_actions_enabled: bool


class ErrorResponse(BaseModel):
    error_id: str
    message: str


class FileChange(BaseModel):
    path: str = Field(
        pattern=r"^[A-Za-z0-9_./-]+$",
        max_length=250,
    )
    content: str = Field(max_length=50_000)

    @field_validator("path")
    @classmethod
    def reject_unsafe_paths(cls, value: str) -> str:
        if value.startswith("/") or ".." in value.split("/"):
            raise ValueError(
                "File paths must be relative and cannot contain '..'."
            )
        return value


class DraftPullRequestRequest(BaseModel):
    repository: str = Field(pattern=r"^[\w.-]+/[\w.-]+$")
    base: str = Field(default="main", pattern=r"^[\w./-]+$")
    branch: str = Field(pattern=r"^mcp-forge/[\w./-]+$")
    title: str = Field(min_length=5, max_length=120)
    body: str = Field(min_length=20, max_length=10_000)
    files: list[FileChange] = Field(min_length=1, max_length=5)
    confirmation: Literal["CREATE_DRAFT_PR"]


class ProposedChange(BaseModel):
    path: str
    rationale: str
    unified_diff: str


class ProposalResponse(BaseModel):
    repository: str
    ref: str
    summary: str
    changes: list[ProposedChange]
    tests: list[str]
    risks: list[str]
    provider: str
    safety: str
