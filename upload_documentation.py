#!/usr/bin/env python3
"""
Upload the MCP implementation documentation to Archon's RAG system
"""
import asyncio
import httpx
import json

async def upload_documentation():
    """Upload the MCP implementation documentation via MCP tool"""
    
    # Read the documentation file
    with open('/media/jonathanco/Backup/archon/MCP_add_markdown_document_implementation.md', 'r') as f:
        content = f.read()
    
    # MCP tool call payload
    payload = {
        "jsonrpc": "2.0",
        "id": "upload-doc-1",
        "method": "tools/call",
        "params": {
            "name": "add_markdown_document",
            "arguments": {
                "content": content,
                "filename": "MCP_add_markdown_document_implementation.md",
                "tags": ["mcp", "implementation", "documentation", "tools", "archon", "rag"],
                "knowledge_type": "documentation"
            }
        }
    }
    
    try:
        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json, text/event-stream",
        }
        
        async with httpx.AsyncClient() as client:
            # Initialize session
            init_payload = {
                "jsonrpc": "2.0",
                "id": "init-1",
                "method": "initialize",
                "params": {
                    "protocolVersion": "2024-11-05",
                    "capabilities": {"tools": {}},
                    "clientInfo": {"name": "upload-client", "version": "1.0.0"}
                }
            }
            
            init_response = await client.post(
                "http://localhost:8051/mcp",
                json=init_payload,
                headers=headers,
                timeout=30.0
            )
            
            if init_response.status_code != 200:
                print(f"Failed to initialize session: {init_response.text}")
                return
                
            session_id = init_response.headers.get('mcp-session-id')
            if session_id:
                headers['mcp-session-id'] = session_id
                print(f"Session initialized: {session_id}")
            
            # Send initialized notification
            await client.post(
                "http://localhost:8051/mcp",
                json={"jsonrpc": "2.0", "method": "notifications/initialized"},
                headers=headers,
                timeout=30.0
            )
            
            print("Uploading documentation...")
            
            # Upload the documentation
            response = await client.post(
                "http://localhost:8051/mcp",
                json=payload,
                headers=headers,
                timeout=120.0
            )
            
            print(f"Status Code: {response.status_code}")
            
            if response.status_code == 200:
                response_text = response.text
                print("Raw Response:")
                print(response_text)
                
                # Parse SSE format if needed
                if "data:" in response_text:
                    lines = response_text.strip().split('\n')
                    for line in lines:
                        if line.startswith('data:'):
                            json_data = line[5:].strip()
                            if json_data and json_data != '[DONE]':
                                result = json.loads(json_data)
                                print("✅ Documentation uploaded successfully!")
                                print(json.dumps(result, indent=2))
                                return
                else:
                    result = json.loads(response_text)
                    print("✅ Documentation uploaded successfully!")
                    print(json.dumps(result, indent=2))
            else:
                print("❌ Upload failed!")
                print(f"Response: {response.text}")
                
    except Exception as e:
        print(f"❌ Upload failed with error: {str(e)}")

if __name__ == "__main__":
    asyncio.run(upload_documentation())