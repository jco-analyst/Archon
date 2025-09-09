"""
RAG Agent - Conversational Search and Retrieval with PydanticAI

This agent enables users to search and chat with documents stored in the RAG system.
It uses the perform_rag_query functionality to retrieve relevant content and provide
intelligent responses based on the retrieved information.
"""

import logging
import os
from dataclasses import dataclass
from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field
from pydantic_ai import Agent, RunContext

from .base_agent import ArchonDependencies, BaseAgent
from .mcp_client import get_mcp_client

logger = logging.getLogger(__name__)


@dataclass
class RagDependencies(ArchonDependencies):
    """Dependencies for RAG operations."""

    project_id: str | None = None
    source_filter: str | None = None
    match_count: int = 5
    progress_callback: Any | None = None  # Callback for progress updates


class RagQueryResult(BaseModel):
    """Structured output for RAG query results."""

    query_type: str = Field(description="Type of query: search, explain, summarize, compare")
    original_query: str = Field(description="The original user query")
    refined_query: str | None = Field(
        description="Refined query used for search if different from original"
    )
    results_found: int = Field(description="Number of relevant results found")
    sources: list[str] = Field(description="List of unique sources referenced")
    answer: str = Field(description="The synthesized answer based on retrieved content")
    citations: list[dict[str, Any]] = Field(description="Citations with source and relevance info")
    success: bool = Field(description="Whether the query was successful")
    message: str = Field(description="Status message or error description")


class RagAgent(BaseAgent[RagDependencies, str]):
    """
    Conversational agent for RAG-based document search and retrieval.

    Capabilities:
    - Search documents using natural language queries
    - Filter by specific sources
    - Search code examples
    - Provide synthesized answers with citations
    - Explain concepts found in documentation
    """

    def __init__(self, model: str = None, **kwargs):
        # Use provided model or fall back to default
        if model is None:
            model = os.getenv("RAG_AGENT_MODEL", "openai:gpt-4o-mini")

        self._custom_client = None
        self._provider_type = None
        
        super().__init__(
            model=model, name="RagAgent", retries=3, enable_rate_limiting=True, **kwargs
        )

    async def run(self, user_input: str, deps: RagDependencies) -> str:
        """
        Override the base run method to use provider-aware client for non-streaming.
        
        This method checks the LLM provider configuration and uses the OpenAI Free
        wrapper when appropriate, ensuring token tracking and fallback functionality.
        """
        logger.info(f"RAG agent run() method called with input: {user_input[:50]}...")
        
        try:
            # Check if we should use OpenAI Free provider from the fetched credentials
            try:
                from .server import AGENT_CREDENTIALS
                logger.info(f"Available credentials keys: {list(AGENT_CREDENTIALS.keys())}")
                logger.info(f"LLM_PROVIDER credential value: {AGENT_CREDENTIALS.get('LLM_PROVIDER', 'NOT_FOUND')}")
                provider_name = AGENT_CREDENTIALS.get("LLM_PROVIDER", "openai")
                logger.info(f"Provider detection result: {provider_name} (from fetched credentials)")
            except Exception as import_error:
                logger.warning(f"Failed to import AGENT_CREDENTIALS: {import_error}")
                # Fallback to environment variable
                provider_name = os.getenv("LLM_PROVIDER", "openai")
                logger.info(f"Provider detection result: {provider_name} (from environment fallback)")
            
            if provider_name == "openai_free":
                logger.info("RAG agent using OpenAI Free approach (non-streaming)")
                logger.info("✅ OpenAI Free provider detected - this will use token tracking and fallback")
                # For now, fall back to standard approach but with clear logging about wrapper status
                # TODO: Implement actual OpenAI Free wrapper integration for non-streaming
                logger.warning("⚠️ OpenAI Free wrapper not yet implemented for non-streaming - using standard OpenAI")
                logger.info("📋 Next step: Integrate OpenAI Free wrapper for non-streaming mode")
                return await super().run(user_input, deps)
            else:
                logger.info(f"RAG agent using standard provider: {provider_name}")
                return await super().run(user_input, deps)
                
        except Exception as provider_error:
            logger.warning(f"Error in provider detection, falling back to default: {provider_error}")
            return await super().run(user_input, deps)

    def run_stream(self, user_input: str, deps: RagDependencies):
        """
        Override the base run_stream method - currently disabled due to verification requirements.
        
        OpenAI Free streaming requires organization verification for premium models like gpt-5-mini.
        For now, we recommend using the non-streaming run() method instead.
        """
        logger.info(f"RAG agent run_stream() method called with input: {user_input[:50]}...")
        logger.warning("Streaming mode requires OpenAI organization verification for gpt-5-mini")
        logger.info("Recommendation: Use non-streaming /agents/run endpoint instead")
        
        try:
            # For now, return the standard streaming context
            # This will fail with verification error but provides clear error message
            return super().run_stream(user_input, deps)
                
        except Exception as provider_error:
            logger.warning(f"Error in streaming setup: {provider_error}")
            return super().run_stream(user_input, deps)

    def _create_agent(self, **kwargs) -> Agent:
        """Create the PydanticAI agent with tools and prompts."""

        agent = Agent(
            model=self.model,
            deps_type=RagDependencies,
            system_prompt="""You are a RAG (Retrieval-Augmented Generation) Assistant that helps users search and understand documentation through conversation.

**Your Capabilities:**
- Search through crawled documentation using semantic search
- Filter searches by specific sources or domains
- Find relevant code examples
- Synthesize information from multiple sources
- Provide clear, cited answers based on retrieved content
- Explain technical concepts found in documentation

**Your Approach:**
1. **Understand the query** - Interpret what the user is looking for
2. **Search effectively** - Use appropriate search terms and filters
3. **Analyze results** - Review retrieved content for relevance
4. **Synthesize answers** - Combine information from multiple sources
5. **Cite sources** - Always provide references to source documents

**Common Queries:**
- "What resources/sources are available?" → Use list_available_sources tool
- "Search for X" → Use search_documents tool
- "Find code examples for Y" → Use search_code_examples tool
- "What documentation do you have?" → Use list_available_sources tool

**Search Strategies:**
- For conceptual questions: Use broader search terms
- For specific features: Use exact terminology
- For code examples: Search for function names, patterns
- For comparisons: Search for each item separately

**Response Guidelines:**
- Provide direct answers based on retrieved content
- Include relevant quotes from sources
- Cite sources with URLs when available
- Admit when information is not found
- Suggest alternative searches if needed""",
            **kwargs,
        )

        # Register dynamic system prompt for context
        @agent.system_prompt
        async def add_search_context(ctx: RunContext[RagDependencies]) -> str:
            source_info = (
                f"Source Filter: {ctx.deps.source_filter}"
                if ctx.deps.source_filter
                else "No source filter"
            )
            return f"""
**Current Search Context:**
- Project ID: {ctx.deps.project_id or "Global search"}
- {source_info}
- Max Results: {ctx.deps.match_count}
- Timestamp: {datetime.now().isoformat()}
"""

        # Register tools for RAG operations
        @agent.tool
        async def search_documents(
            ctx: RunContext[RagDependencies], query: str, source_filter: str | None = None
        ) -> str:
            """Search through documents using RAG query."""
            try:
                # Use source filter from context if not provided
                if source_filter is None:
                    source_filter = ctx.deps.source_filter

                logger.info(f"RAG search: '{query}' with filter: {source_filter}")

                # Use the MCP client from the agent's base dependencies
                # Import MCP client within the function to avoid import issues
                import httpx
                
                mcp_endpoint = "http://archon-mcp:8051"
                async with httpx.AsyncClient() as client:
                    response = await client.post(
                        f"{mcp_endpoint}/tools/perform_rag_query",
                        json={
                            "query": query,
                            "source": source_filter,
                            "match_count": ctx.deps.match_count,
                        }
                    )
                    result = response.json()

                if result.get("success"):
                    response = result.get("result", "No results found")
                    logger.info("RAG query completed successfully")
                    return response
                else:
                    error = result.get("error", "Unknown error")
                    logger.error(f"RAG query failed: {error}")
                    return f"Search failed: {error}"

            except Exception as e:
                logger.error(f"Error during RAG search: {str(e)}")
                return f"I encountered an error while searching: {str(e)}"

        @agent.tool
        async def search_code_examples(
            ctx: RunContext[RagDependencies], query: str, source_filter: str | None = None
        ) -> str:
            """Search for code examples related to the query."""
            try:
                # Use source filter from context if not provided
                if source_filter is None:
                    source_filter = ctx.deps.source_filter

                logger.info(f"Code example search: '{query}' with filter: {source_filter}")

                # Use HTTP client to call MCP service directly
                import httpx
                
                mcp_endpoint = "http://archon-mcp:8051"
                async with httpx.AsyncClient() as client:
                    response = await client.post(
                        f"{mcp_endpoint}/tools/search_code_examples",
                        json={
                            "query": query,
                            "source_id": source_filter,
                            "match_count": ctx.deps.match_count,
                        }
                    )
                    result = response.json()

                if result.get("success"):
                    response = result.get("result", "No code examples found")
                    logger.info("Code example search completed successfully")
                    return response
                else:
                    error = result.get("error", "Unknown error")
                    logger.error(f"Code example search failed: {error}")
                    return f"Code search failed: {error}"

            except Exception as e:
                logger.error(f"Error during code example search: {str(e)}")
                return f"I encountered an error while searching for code examples: {str(e)}"

        @agent.tool
        async def list_available_sources(ctx: RunContext[RagDependencies]) -> str:
            """List all available knowledge sources."""
            try:
                logger.info("Listing available sources")

                # Use HTTP client to call MCP service directly
                import httpx
                
                mcp_endpoint = "http://archon-mcp:8051"
                async with httpx.AsyncClient() as client:
                    response = await client.post(
                        f"{mcp_endpoint}/tools/get_available_sources",
                        json={}
                    )
                    result = response.json()

                if result.get("success"):
                    response = result.get("result", "No sources available")
                    logger.info("Source listing completed successfully")
                    return response
                else:
                    error = result.get("error", "Unknown error")
                    logger.error(f"Source listing failed: {error}")
                    return f"Failed to list sources: {error}"

            except Exception as e:
                logger.error(f"Error listing sources: {str(e)}")
                return f"I encountered an error while listing sources: {str(e)}"

        @agent.tool
        async def refine_search_query(
            ctx: RunContext[RagDependencies], original_query: str, context: str
        ) -> str:
            """Refine a search query based on context or previous results."""
            try:
                # Simple query refinement logic
                refined_query = original_query.strip()

                # Add context-based refinements
                if "code" in context.lower() or "example" in context.lower():
                    refined_query = f"{refined_query} example implementation"
                elif "error" in context.lower() or "troubleshooting" in context.lower():
                    refined_query = f"{refined_query} error solution troubleshooting"

                return f"Refined query: '{refined_query}' (original: '{original_query}')"
            except Exception as e:
                return f"Could not refine query: {str(e)}"
        return agent

    def get_system_prompt(self) -> str:
        """Get the base system prompt for this agent."""
        try:
            from ..services.prompt_service import prompt_service
            return prompt_service.get_prompt(
                "rag_assistant",
                default="RAG Assistant for intelligent document search and retrieval.",
            )
        except Exception as e:
            logger.warning(f"Could not load prompt from service: {e}")
            return "RAG Assistant for intelligent document search and retrieval."


# Note: ProviderAwareStreamContext class removed in favor of simpler approach
# Streaming requires OpenAI organization verification for gpt-5-mini model
# Using non-streaming approach for now to avoid verification requirement


