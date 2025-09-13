# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Alpha Development Guidelines

**Local-only deployment** - each user runs their own instance.

### Core Principles

- **No backwards compatibility** - remove deprecated code immediately
- **Detailed errors over graceful failures** - we want to identify and fix issues fast
- **Break things to improve them** - alpha is for rapid iteration

### Error Handling

**Core Principle**: In alpha, we need to intelligently decide when to fail hard and fast to quickly address issues, and when to allow processes to complete in critical services despite failures. Read below carefully and make intelligent decisions on a case-by-case basis.

#### When to Fail Fast and Loud (Let it Crash!)

These errors should stop execution and bubble up immediately:

- **Service startup failures** - If credentials, database, or any service can't initialize, the system should crash with a clear error
- **Missing configuration** - Missing environment variables or invalid settings should stop the system
- **Database connection failures** - Don't hide connection issues, expose them
- **Authentication/authorization failures** - Security errors must be visible and halt the operation
- **Data corruption or validation errors** - Never silently accept bad data, Pydantic should raise
- **Critical dependencies unavailable** - If a required service is down, fail immediately
- **Invalid data that would corrupt state** - Never store zero embeddings, null foreign keys, or malformed JSON

#### When to Complete but Log Detailed Errors

These operations should continue but track and report failures clearly:

- **Batch processing** - When crawling websites or processing documents, complete what you can and report detailed failures for each item
- **Background tasks** - Embedding generation, async jobs should finish the queue but log failures
- **WebSocket events** - Don't crash on a single event failure, log it and continue serving other clients
- **Optional features** - If projects/tasks are disabled, log and skip rather than crash
- **External API calls** - Retry with exponential backoff, then fail with a clear message about what service failed and why

#### Critical Nuance: Never Accept Corrupted Data

When a process should continue despite failures, it must **skip the failed item entirely** rather than storing corrupted data:

**❌ WRONG - Silent Corruption:**

```python
try:
    embedding = create_embedding(text)
except Exception as e:
    embedding = [0.0] * 1536  # NEVER DO THIS - corrupts database
    store_document(doc, embedding)
```

**✅ CORRECT - Skip Failed Items:**

```python
try:
    embedding = create_embedding(text)
    store_document(doc, embedding)  # Only store on success
except Exception as e:
    failed_items.append({'doc': doc, 'error': str(e)})
    logger.error(f"Skipping document {doc.id}: {e}")
    # Continue with next document, don't store anything
```

**✅ CORRECT - Batch Processing with Failure Tracking:**

```python
def process_batch(items):
    results = {'succeeded': [], 'failed': []}

    for item in items:
        try:
            result = process_item(item)
            results['succeeded'].append(result)
        except Exception as e:
            results['failed'].append({
                'item': item,
                'error': str(e),
                'traceback': traceback.format_exc()
            })
            logger.error(f"Failed to process {item.id}: {e}")

    # Always return both successes and failures
    return results
```

#### Error Message Guidelines

- Include context about what was being attempted when the error occurred
- Preserve full stack traces with `exc_info=True` in Python logging
- Use specific exception types, not generic Exception catching
- Include relevant IDs, URLs, or data that helps debug the issue
- Never return None/null to indicate failure - raise an exception with details
- For batch operations, always report both success count and detailed failure list

### Code Quality

- Remove dead code immediately rather than maintaining it - no backward compatibility or legacy functions
- Prioritize functionality over production-ready patterns
- Focus on user experience and feature completeness
- When updating code, don't reference what is changing (avoid keywords like LEGACY, CHANGED, REMOVED), instead focus on comments that document just the functionality of the code

## Architecture Overview

Archon V2 Alpha is a microservices-based knowledge management system with MCP (Model Context Protocol) integration:

- **Frontend (port 3737)**: React + TypeScript + Vite + TailwindCSS
- **Main Server (port 8181)**: FastAPI + Socket.IO for real-time updates
- **MCP Server (port 8051)**: Lightweight HTTP-based MCP protocol server
- **Agents Service (port 8052)**: PydanticAI agents for AI/ML operations
- **Database**: Supabase (PostgreSQL + pgvector for embeddings)

## Development Commands

### Frontend (archon-ui-main/)

```bash
npm run dev              # Start development server on port 3737
npm run build            # Build for production
npm run lint             # Run ESLint
npm run test             # Run Vitest tests
npm run test:coverage    # Run tests with coverage report
```

### Backend (python/)

```bash
# Using uv package manager
uv sync                  # Install/update dependencies
uv run pytest            # Run tests
uv run python -m src.server.main  # Run server locally

# With Docker
docker-compose up --build -d       # Start all services
docker-compose logs -f             # View logs
docker-compose restart              # Restart services
```

### Docker Development Workflow

**CRITICAL: Python import caching requires container rebuilds after code changes.**

```bash
# After Python code changes (new functions, imports, modifications)
docker compose down && docker compose up --build -d

# After minor changes or config updates
docker compose restart archon-server

# Debug: Clear cache if rebuild fails
docker exec Archon-Server find /app -name "__pycache__" -type d -exec rm -rf {} +

# Debug: Verify code changes applied
docker exec Archon-Server python -c "
from your.module import your_function; print('✅ Import successful')
"
```

### Testing

```bash
# Frontend tests (from archon-ui-main/)
npm run test:coverage:stream       # Run with streaming output
npm run test:ui                    # Run with Vitest UI

# Backend tests (from python/)
uv run pytest tests/test_api_essentials.py -v
uv run pytest tests/test_service_integration.py -v
```

## Key API Endpoints

### Knowledge Base

- `POST /api/knowledge/crawl` - Crawl a website
- `POST /api/knowledge/upload` - Upload documents (PDF, DOCX, MD)
- `GET /api/knowledge/items` - List knowledge items
- `POST /api/knowledge/search` - RAG search

### MCP Integration

- `GET /api/mcp/health` - MCP server status
- `POST /api/mcp/tools/{tool_name}` - Execute MCP tool
- `GET /api/mcp/tools` - List available tools

### Projects & Tasks (when enabled)

- `GET /api/projects` - List projects
- `POST /api/projects` - Create project
- `GET /api/projects/{id}/tasks` - Get project tasks
- `POST /api/projects/{id}/tasks` - Create task

### OpenAI Free Provider

- `POST /api/openai-free/config` - Configure fallback provider for OpenAI Free
- `GET /api/openai-free/config` - Get current OpenAI Free configuration  
- `GET /api/openai-free/usage` - View current token usage statistics
- `DELETE /api/openai-free/usage/cleanup` - Clean up old usage records

## Socket.IO Events

Real-time updates via Socket.IO on port 8181:

- `crawl_progress` - Website crawling progress
- `project_creation_progress` - Project setup progress
- `task_update` - Task status changes
- `knowledge_update` - Knowledge base changes

## Environment Variables

Required in `.env`:

```bash
SUPABASE_URL=https://your-project.supabase.co
SUPABASE_SERVICE_KEY=your-service-key-here
```

Optional:

```bash
OPENAI_API_KEY=your-openai-key        # Can be set via UI
LOGFIRE_TOKEN=your-logfire-token      # For observability
LOG_LEVEL=INFO                         # DEBUG, INFO, WARNING, ERROR
```

## OpenAI Free Provider Configuration

The OpenAI Free provider offers access to OpenAI models with daily token limits and automatic fallback functionality.

### Configuration Requirements

**Base URL**: The system automatically configures the base URL as `https://api.openai.com/v1` when OpenAI Free is selected as the LLM provider. If the base URL field appears empty in the UI, it should be set to this value.

**Required Settings**:
- **LLM Provider**: `openai_free`
- **LLM Base URL**: `https://api.openai.com/v1`
- **Model Choice**: Any OpenAI Free compatible model
- **Fallback Provider**: Configure an alternate provider for when limits are exceeded

### Token Limits

**Premium Models (250,000 tokens/day)**:
- gpt-5, gpt-5-chat-latest, gpt-4.1, gpt-4o, o1, o3

**Mini Models (2,500,000 tokens/day)**:
- gpt-5-mini, gpt-5-nano, gpt-4.1-mini, gpt-4.1-nano, gpt-4o-mini, o1-mini, o3-mini, o4-mini, codex-mini-latest

### Database Schema

Token usage is tracked in the `archon_token_usage` table:
```sql
CREATE TABLE archon_token_usage (
    provider_name VARCHAR(50) CHECK (provider_name IN ('openai_free')),
    model_name VARCHAR(100) NOT NULL,
    usage_date DATE NOT NULL DEFAULT CURRENT_DATE,
    tokens_used INTEGER NOT NULL DEFAULT 0,
    token_limit INTEGER NOT NULL,
    UNIQUE(provider_name, model_name, usage_date)
);
```

### Troubleshooting

**Empty Base URL**: If the LLM Base URL field is empty, manually set it to `https://api.openai.com/v1` via the Settings UI or API:
```bash
curl -X PUT http://localhost:8181/api/credentials/LLM_BASE_URL \
  -H "Content-Type: application/json" \
  -d '{"value": "https://api.openai.com/v1", "is_encrypted": false, "category": "rag_strategy"}'
```

## RAG System Configuration

The RAG (Retrieval-Augmented Generation) system integrates multiple AI services for knowledge-based responses.

### RAG Agent Architecture

**Services Integration**:
- **Agents Service (port 8052)**: PydanticAI-based RAG agent
- **Main Server (port 8181)**: OpenAI Free wrapper and credential management
- **Ollama Service**: Local embedding generation with Qwen models
- **Supabase**: Vector storage with pgvector extension

### Key Configuration Points

**RAG Agent Model Configuration**:
```bash
# Set via credentials API or environment
RAG_AGENT_MODEL=openai:gpt-5-mini  # PydanticAI format: provider:model
```

**Critical Environment Variables** (docker-compose.yml):
```yaml
archon-agents:
  environment:
    - ARCHON_SERVER_PORT=8181  # Required for credential fetching
```

### Streaming vs Non-Streaming Mode

**Organization Verification Issue**: OpenAI requires organization verification for streaming access to premium models (gpt-5-mini, gpt-4o, etc.).

**Workaround**: Use non-streaming mode for RAG queries to avoid verification requirement while maintaining full functionality.

**API Endpoints**:
```bash
# Non-streaming RAG query (recommended)
POST http://localhost:8052/agents/run
{
  "agent_type": "rag",
  "prompt": "Your question here",
  "context": {
    "source_filter": null,
    "match_count": 3
  }
}

# Streaming RAG query (requires organization verification)
POST http://localhost:8052/agents/rag/stream
```

### OpenAI Free Wrapper Integration

**Architecture**: PydanticAI agents must integrate with OpenAI Free wrapper for token tracking and fallback functionality.

**Integration Pattern**:
```python
# In RAG agent initialization
provider_config = await credential_service.get_active_provider("llm")
if provider_config.get("provider") == "openai_free":
    # Use OpenAI Free wrapper with token tracking
    client = get_openai_free_client()
    # Integrate with PydanticAI Agent
```

**Verification Commands**:
```bash
# Check RAG agent logs for correct integration
docker compose logs archon-agents | grep -E "(openai_free|wrapper|fallback)"

# Test RAG query end-to-end
curl -X POST http://localhost:8052/agents/run \
  -H "Content-Type: application/json" \
  -d '{"agent_type": "rag", "prompt": "test query", "context": {}}'
```

### Common RAG Issues

1. **Direct OpenAI API Calls**: Agent bypassing wrapper, calling `https://api.openai.com/v1/chat/completions` directly
   - **Fix**: Ensure proper OpenAI Free wrapper integration in agent initialization

2. **Missing Credentials**: Agents service cannot fetch credentials from main server
   - **Fix**: Add `ARCHON_SERVER_PORT=8181` to agents service environment

3. **Browser RAG Search Limited**: Knowledge Base UI only searches titles, not content
   - **Workaround**: Use direct API endpoints for full content search

## File Organization

### Frontend Structure

- `src/components/` - Reusable UI components
- `src/pages/` - Main application pages
- `src/services/` - API communication and business logic
- `src/hooks/` - Custom React hooks
- `src/contexts/` - React context providers

### Backend Structure

- `src/server/` - Main FastAPI application
- `src/server/api_routes/` - API route handlers
- `src/server/services/` - Business logic services
- `src/mcp/` - MCP server implementation
- `src/agents/` - PydanticAI agent implementations

## Database Schema

Key tables in Supabase:

- `sources` - Crawled websites and uploaded documents
- `documents` - Processed document chunks with embeddings
- `projects` - Project management (optional feature)
- `tasks` - Task tracking linked to projects
- `code_examples` - Extracted code snippets

## Common Development Tasks

### Add a new API endpoint

1. Create route handler in `python/src/server/api_routes/`
2. Add service logic in `python/src/server/services/`
3. Include router in `python/src/server/main.py`
4. Update frontend service in `archon-ui-main/src/services/`

### Add a new UI component

1. Create component in `archon-ui-main/src/components/`
2. Add to page in `archon-ui-main/src/pages/`
3. Include any new API calls in services
4. Add tests in `archon-ui-main/test/`

### Debug MCP connection issues

1. Check MCP health: `curl http://localhost:8051/health`
2. View MCP logs: `docker-compose logs archon-mcp`
3. Test tool execution via UI MCP page
4. Verify Supabase connection and credentials

## Code Quality Standards

We enforce code quality through automated linting and type checking:

- **Python 3.12** with 120 character line length
- **Ruff** for linting - checks for errors, warnings, unused imports, and code style
- **Mypy** for type checking - ensures type safety across the codebase
- **Auto-formatting** on save in IDEs to maintain consistent style
- Run `uv run ruff check` and `uv run mypy src/` locally before committing

## MCP Tools Available

When connected to Cursor/Windsurf:

- `archon:perform_rag_query` - Search knowledge base
- `archon:search_code_examples` - Find code snippets
- `archon:manage_project` - Project operations
- `archon:manage_task` - Task management
- `archon:get_available_sources` - List knowledge sources

## Important Notes

- Projects feature is optional - toggle in Settings UI
- All services communicate via HTTP, not gRPC
- Socket.IO handles all real-time updates
- Frontend uses Vite proxy for API calls in development
- Python backend uses `uv` for dependency management
- Docker Compose handles service orchestration

ADDITIONAL CONTEXT FOR SPECIFICALLY HOW TO USE ARCHON ITSELF:
@CLAUDE-ARCHON.md
- when making any changes to the core code, after applying all the changes please run "docker compose down" and "docker compose up --build -d" rebuild, then use playwright mcp to look at http://localhost:3737/settings to see if everything is working. Please use this workflow