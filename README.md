# MCP Forge

MCP Forge is an approval-based AI engineering agent for GitHub repositories.

It begins in **read-only mode**:

1. Reads a GitHub repository.
2. Stores repository context in Qdrant semantic memory.
3. Uses a free AI provider through OpenRouter.
4. Produces engineering findings and safe improvement proposals.
5. Does not create commits, branches, or pull requests automatically.

## Current MVP

- FastAPI backend
- GitHub read-only repository inspection
- Qdrant in-memory semantic search
- Free OpenRouter AI analysis
- Empty-repository detection
- Safety-first API health check

## API endpoints

### Health check

```text
GET /health
