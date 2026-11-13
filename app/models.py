from pydantic import BaseModel, Field


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
