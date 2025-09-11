# RAG OpenAI Free Wrapper - Base URL Override Implementation

**Date**: 2025-09-10  
**Status**: Ready for implementation  
**Approach**: Environment Variable Override at Startup  
**Project**: Archon RAG Debug (d7243341-474a-42ea-916e-4dda894dae95)

## 🎯 SOLUTION SUMMARY

**Problem**: RAG agent makes direct OpenAI API calls instead of using OpenAI Free wrapper  
**Root Cause**: PydanticAI creates and caches OpenAI clients before runtime configuration can be applied  
**Solution**: Configure OpenAI environment variables during agents service startup

## 📋 IMPLEMENTATION DETAILS

### **Modified File**: `python/src/agents/server.py`
### **Function**: `fetch_credentials_from_server()`
### **Location**: Line 63, Column 10

**Code Added**:
```python
# Special handling for OpenAI Free wrapper
llm_provider = credentials.get("LLM_PROVIDER")
if llm_provider == "openai_free":
    wrapper_base_url = f"http://archon-server:{server_port}/api/openai-free"
    os.environ["OPENAI_BASE_URL"] = wrapper_base_url
    os.environ["OPENAI_API_KEY"] = "wrapper-bypass-token"
    logger.info(f"🎯 OpenAI Free wrapper configured: {wrapper_base_url}")
    logger.info("✅ PydanticAI will use OpenAI Free wrapper for all agents")
```

**Integration Point**: Added after existing credential environment variable setting loop, before storing global credentials.

### **Why This Works**:
1. **Startup Timing**: Environment variables set before PydanticAI agents are initialized
2. **Framework Native**: PydanticAI's OpenAI client constructor respects `OPENAI_BASE_URL`
3. **Zero Overhead**: No per-request processing or complex client management
4. **System Integration**: Uses existing credential fetching infrastructure

## 🧪 VERIFICATION PLAN

### **Testing Commands**:

**1. Restart Service**:
```bash
docker compose restart archon-agents
```

**2. Check Startup Logs**:
```bash
docker compose logs archon-agents | grep -E "(OpenAI Free wrapper configured|PydanticAI will use)"
```

**Expected Startup Logs**:
```
INFO:src.agents.server:🎯 OpenAI Free wrapper configured: http://archon-server:8181/api/openai-free
INFO:src.agents.server:✅ PydanticAI will use OpenAI Free wrapper for all agents
```

**3. Test RAG Agent**:
```bash
curl -X POST http://localhost:8052/agents/run \
  -H "Content-Type: application/json" \
  -d '{"agent_type": "rag", "prompt": "test wrapper", "context": {}}'
```

**4. Monitor RAG Execution**:
```bash
docker compose logs archon-agents --tail=0 --follow | grep -E "HTTP Request: POST"
```

### **Success Indicators**:

**✅ BEFORE (Problem)**:
```
INFO:httpx:HTTP Request: POST https://api.openai.com/v1/chat/completions "HTTP/1.1 200 OK"
```

**✅ AFTER (Fixed)**:
```
INFO:httpx:HTTP Request: POST http://archon-server:8181/api/openai-free/chat/completions "HTTP/1.1 200 OK"
```

## 📊 CURRENT STATUS

### **✅ Completed**:
- [x] Modified `fetch_credentials_from_server()` function
- [x] Added OpenAI Free wrapper environment configuration  
- [x] Enhanced logging for verification
- [x] Updated comprehensive documentation

### **🔄 Ready for Testing**:
- [ ] Restart agents service to apply changes
- [ ] Verify startup configuration logs
- [ ] Test RAG agent wrapper integration
- [ ] Confirm token tracking activation
- [ ] Test fallback provider functionality

## 🎯 EXPECTED OUTCOMES

After successful implementation:

1. **Automatic Wrapper Usage**: All PydanticAI agents automatically route to OpenAI Free wrapper when `LLM_PROVIDER=openai_free`
2. **Token Tracking Active**: Daily usage limits enforced with database tracking
3. **Fallback Functionality**: Automatic fallback to configured provider when limits exceeded
4. **Zero Performance Impact**: Configuration happens once at startup, no runtime overhead
5. **Framework Compatibility**: Works with PydanticAI's natural configuration mechanism

## 🔗 RELATED FILES

**Core Implementation**:
- `python/src/agents/server.py` - Main implementation
- `python/src/server/services/openai_free_wrapper.py` - Wrapper service (already working)

**Documentation**:
- `RAG_SYSTEM_INVESTIGATION.md` - Complete investigation history
- `RAG_DEBUGGING_WORKFLOW.md` - Testing workflow and commands
- `RAG_BASE_URL_OVERRIDE_IMPLEMENTATION.md` - This implementation guide

**Testing**:
- Direct API testing via `curl` commands
- Log monitoring via `docker compose logs`
- Wrapper endpoint verification via main server API

## 🚀 DEPLOYMENT STEPS

1. **Apply Changes**: Code already modified in `server.py`
2. **Restart Service**: `docker compose restart archon-agents`
3. **Verify Startup**: Check for wrapper configuration logs
4. **Test Integration**: Run RAG agent test queries
5. **Validate Behavior**: Confirm wrapper calls in logs
6. **Update Tasks**: Mark investigation tasks as completed

This implementation leverages the framework's natural configuration mechanism rather than fighting against it, resulting in a clean, efficient, and maintainable solution.