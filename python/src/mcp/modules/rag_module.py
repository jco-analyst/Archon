"""
RAG Module for Archon MCP Server (HTTP-based version)

This module provides tools for:
- RAG query and search
- Source management
- Code example extraction and search

This version uses HTTP calls to the server service instead of importing
service modules directly, enabling true microservices architecture.
"""

import json
import logging
import os
from urllib.parse import urljoin

import httpx

from mcp.server.fastmcp import Context, FastMCP

# Import service discovery for HTTP communication
from src.server.config.service_discovery import get_api_url

logger = logging.getLogger(__name__)


def get_setting(key: str, default: str = "false") -> str:
    """Get a setting from environment variable."""
    return os.getenv(key, default)


def get_bool_setting(key: str, default: bool = False) -> bool:
    """Get a boolean setting from environment variable."""
    value = get_setting(key, "false" if not default else "true")
    return value.lower() in ("true", "1", "yes", "on")


def register_rag_tools(mcp: FastMCP):
    """Register all RAG tools with the MCP server."""

    @mcp.tool()
    async def get_available_sources(ctx: Context) -> str:
        """
        Get list of available sources in the knowledge base.

        This tool uses HTTP call to the API service.

        Returns:
            JSON string with list of sources
        """
        try:
            api_url = get_api_url()
            timeout = httpx.Timeout(30.0, connect=5.0)

            async with httpx.AsyncClient(timeout=timeout) as client:
                response = await client.get(urljoin(api_url, "/api/rag/sources"))

                if response.status_code == 200:
                    result = response.json()
                    sources = result.get("sources", [])

                    return json.dumps(
                        {"success": True, "sources": sources, "count": len(sources)}, indent=2
                    )
                else:
                    error_detail = response.text
                    return json.dumps(
                        {"success": False, "error": f"HTTP {response.status_code}: {error_detail}"},
                        indent=2,
                    )

        except Exception as e:
            logger.error(f"Error getting sources: {e}")
            return json.dumps({"success": False, "error": str(e)}, indent=2)

    @mcp.tool()
    async def perform_rag_query(
        ctx: Context, query: str, source: str = None, match_count: int = 5
    ) -> str:
        """
        Perform a RAG (Retrieval Augmented Generation) query on stored content.

        This tool searches the vector database for content relevant to the query and returns
        the matching documents. Optionally filter by source domain.
        Get the source by using the get_available_sources tool before calling this search!

        Args:
            query: The search query
            source: Optional source domain to filter results (e.g., 'example.com')
            match_count: Maximum number of results to return (default: 5)

        Returns:
            JSON string with search results
        """
        try:
            api_url = get_api_url()
            timeout = httpx.Timeout(30.0, connect=5.0)

            async with httpx.AsyncClient(timeout=timeout) as client:
                request_data = {"query": query, "match_count": match_count}
                if source:
                    request_data["source"] = source

                response = await client.post(urljoin(api_url, "/api/rag/query"), json=request_data)

                if response.status_code == 200:
                    result = response.json()
                    return json.dumps(
                        {
                            "success": True,
                            "results": result.get("results", []),
                            "reranked": result.get("reranked", False),
                            "error": None,
                        },
                        indent=2,
                    )
                else:
                    error_detail = response.text
                    return json.dumps(
                        {
                            "success": False,
                            "results": [],
                            "error": f"HTTP {response.status_code}: {error_detail}",
                        },
                        indent=2,
                    )

        except Exception as e:
            logger.error(f"Error performing RAG query: {e}")
            return json.dumps({"success": False, "results": [], "error": str(e)}, indent=2)

    @mcp.tool()
    async def search_code_examples(
        ctx: Context, query: str, source_id: str = None, match_count: int = 5
    ) -> str:
        """
        Search for code examples relevant to the query.

        This tool searches the vector database for code examples relevant to the query and returns
        the matching examples with their summaries. Optionally filter by source_id.
        Get the source_id by using the get_available_sources tool before calling this search!

        Use the get_available_sources tool first to see what sources are available for filtering.

        Args:
            query: The search query
            source_id: Optional source ID to filter results (e.g., 'example.com')
            match_count: Maximum number of results to return (default: 5)

        Returns:
            JSON string with search results
        """
        try:
            api_url = get_api_url()
            timeout = httpx.Timeout(30.0, connect=5.0)

            async with httpx.AsyncClient(timeout=timeout) as client:
                request_data = {"query": query, "match_count": match_count}
                if source_id:
                    request_data["source"] = source_id

                # Call the dedicated code examples endpoint
                response = await client.post(
                    urljoin(api_url, "/api/rag/code-examples"), json=request_data
                )

                if response.status_code == 200:
                    result = response.json()
                    return json.dumps(
                        {
                            "success": True,
                            "results": result.get("results", []),
                            "reranked": result.get("reranked", False),
                            "error": None,
                        },
                        indent=2,
                    )
                else:
                    error_detail = response.text
                    return json.dumps(
                        {
                            "success": False,
                            "results": [],
                            "error": f"HTTP {response.status_code}: {error_detail}",
                        },
                        indent=2,
                    )

        except Exception as e:
            logger.error(f"Error searching code examples: {e}")
            return json.dumps({"success": False, "results": [], "error": str(e)}, indent=2)

    @mcp.tool()
    async def add_markdown_document(
        ctx: Context,
        content: str,
        filename: str,
        source_id: str = None,
        tags: list[str] = None,
        knowledge_type: str = "documentation"
    ) -> str:
        """
        Add a markdown document directly to the RAG system via HTTP API.

        This tool uses Archon's document upload API to process markdown content.
        The document will be chunked, embedded, and stored in the vector database.

        IMPORTANT FOR CONTEXT DETECTION:
        - If source_id is not provided, try to detect the current project context
        - For git repositories, use: git remote get-url origin to get repo name
        - For local projects, use the current directory name
        - Avoid generic source IDs - use meaningful project names like "archon", "my-project"
        - This enables better RAG filtering when searching within specific projects

        Args:
            content: The markdown content to add to the RAG system
            filename: Name for the document (used for source identification)
            source_id: Optional custom source ID - detect current project if not provided
            tags: Optional list of tags to associate with the document
            knowledge_type: Type of knowledge (default: "documentation")

        Returns:
            JSON string with processing results including chunks stored and word count

        Example:
            # Auto-detect project context
            add_markdown_document(
                content="# API Documentation\\n\\nThis explains the REST API...",
                filename="api_docs.md",
                source_id="archon",  # Detected from git repo or directory
                tags=["api", "documentation"]
            )
        """
        try:
            import io
            import time
            
            logger.info(f"Adding markdown document via HTTP API: {filename} ({len(content)} chars)")
            
            # Prepare the content as a file-like object
            content_bytes = content.encode('utf-8')
            
            # Create multipart form data
            files = {
                'file': (filename, io.BytesIO(content_bytes), 'text/markdown')
            }
            
            # Prepare form data
            form_data = {
                'knowledge_type': knowledge_type,
                'tags': json.dumps(tags or [])
            }
            
            # Add project context if source_id provided
            if source_id and source_id.strip():
                form_data['project_context'] = source_id.strip()
            
            # Get API URL
            api_url = get_api_url()
            timeout = httpx.Timeout(120.0, connect=10.0)  # Longer timeout for processing
            
            async with httpx.AsyncClient(timeout=timeout) as client:
                logger.info(f"Calling upload API: {urljoin(api_url, '/api/documents/upload')}")
                
                response = await client.post(
                    urljoin(api_url, "/api/documents/upload"),
                    files=files,
                    data=form_data
                )
                
                logger.info(f"Upload API response: {response.status_code}")
                
                if response.status_code == 200:
                    result = response.json()
                    progress_id = result.get("progressId")
                    
                    if progress_id:
                        # Poll for completion
                        logger.info(f"Document upload started, progress_id: {progress_id}")
                        
                        # Wait a moment for processing to begin
                        import asyncio
                        await asyncio.sleep(2.0)
                        
                        # Poll for completion with exponential backoff
                        max_attempts = 30  # Maximum 5 minutes
                        attempt = 0
                        
                        while attempt < max_attempts:
                            try:
                                # Use the knowledge API's task status check
                                status_url = urljoin(api_url, f"/api/knowledge-items/task/{progress_id}")
                                status_response = await client.get(status_url)
                                
                                if status_response.status_code == 200:
                                    status_data = status_response.json()
                                    status = status_data.get("status", "unknown")
                                    
                                    logger.info(f"Processing status: {status}")
                                    
                                    if status == "complete":
                                        # Success!
                                        result_data = status_data.get("result", {})
                                        chunks_stored = result_data.get("chunks_stored", 0)
                                        word_count = result_data.get("total_word_count", 0)
                                        final_source_id = result_data.get("source_id", source_id)
                                        
                                        response_data = {
                                            "success": True,
                                            "message": f"Successfully added {filename} to RAG system",
                                            "chunks_stored": chunks_stored,
                                            "total_word_count": word_count,
                                            "source_id": final_source_id,
                                            "filename": filename,
                                            "knowledge_type": knowledge_type,
                                            "tags": tags or [],
                                            "progress_id": progress_id
                                        }
                                        logger.info(f"Document processed successfully: {response_data}")
                                        return json.dumps(response_data, indent=2)
                                        
                                    elif status == "error":
                                        error_msg = status_data.get("error", "Unknown processing error")
                                        logger.error(f"Document processing failed: {error_msg}")
                                        return json.dumps({
                                            "success": False,
                                            "error": f"Processing failed: {error_msg}",
                                            "filename": filename,
                                            "progress_id": progress_id
                                        }, indent=2)
                                        
                                    elif status in ["running", "processing"]:
                                        # Still processing, wait and retry
                                        wait_time = min(5.0 + (attempt * 0.5), 15.0)  # Exponential backoff, max 15s
                                        await asyncio.sleep(wait_time)
                                        attempt += 1
                                        continue
                                        
                                    else:
                                        # Unknown status, wait and retry
                                        await asyncio.sleep(3.0)
                                        attempt += 1
                                        continue
                                        
                                elif status_response.status_code == 404:
                                    # Task not found - might have completed and been cleaned up
                                    logger.warning(f"Task {progress_id} not found - assuming completion")
                                    return json.dumps({
                                        "success": True,
                                        "message": f"Document {filename} appears to have been processed successfully",
                                        "note": "Task completed and was cleaned up before we could get final status",
                                        "filename": filename,
                                        "progress_id": progress_id
                                    }, indent=2)
                                    
                                else:
                                    logger.warning(f"Status check failed: {status_response.status_code}")
                                    await asyncio.sleep(5.0)
                                    attempt += 1
                                    
                            except Exception as e:
                                logger.warning(f"Error checking status (attempt {attempt}): {e}")
                                await asyncio.sleep(5.0)
                                attempt += 1
                        
                        # Timeout reached
                        return json.dumps({
                            "success": False,
                            "error": "Document processing timed out after 5 minutes",
                            "note": "The document may still be processing in the background",
                            "filename": filename,
                            "progress_id": progress_id
                        }, indent=2)
                        
                    else:
                        # No progress ID returned - immediate result
                        return json.dumps({
                            "success": True,
                            "message": f"Document upload initiated: {filename}",
                            "result": result,
                            "filename": filename
                        }, indent=2)
                        
                else:
                    error_detail = response.text
                    error_msg = f"Upload failed: HTTP {response.status_code}: {error_detail}"
                    logger.error(error_msg)
                    return json.dumps({
                        "success": False,
                        "error": error_msg,
                        "filename": filename
                    }, indent=2)
                    
        except Exception as e:
            error_msg = f"Failed to add document: {str(e)}"
            logger.error(f"Unexpected error in add_markdown_document: {error_msg}", exc_info=True)
            return json.dumps({
                "success": False,
                "error": error_msg,
                "filename": filename if 'filename' in locals() else "unknown"
            }, indent=2)






    # Log successful registration
    logger.info("✓ RAG tools registered (HTTP-based version + Direct service layer)")

