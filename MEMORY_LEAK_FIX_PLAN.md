# Memory Leak Fix Implementation Plan

## Problem Confirmed
- **Issue**: Archon using crawl4ai==0.6.2 has known memory leak bug
- **Symptoms**: Multiprocessing workers spawn Chrome processes consuming 15GB+ RAM
- **Root Cause**: Chrome processes not cleaned up properly after crawling tasks

## Solution Found
- **Fix Available**: Upgrade to crawl4ai==0.7.4 
- **Fixes Include**:
  - Memory management refactor
  - Critical stability improvements  
  - 3x performance boost
  - Enhanced concurrency fixes
  - Better Docker stability

## Research Results
- **Issue #943**: Confirmed exact match to our problem on GitHub
- **Issue #1256**: Memory leak on repeated requests confirmed
- **v0.7.4 Release**: Specifically addresses these issues with complete memory management rewrite

## Implementation Steps
1. **Update pyproject.toml**: Change `crawl4ai==0.6.2` to `crawl4ai==0.7.4`
2. **Test upgrade**: Check for API compatibility issues
3. **Rebuild containers**: `docker compose down && docker compose up --build -d`
4. **Monitor memory**: Verify the fix resolves runaway worker processes
5. **Load test**: Ensure no regression in crawling functionality

## Files to Check After Upgrade
- `python/pyproject.toml` - Version dependency
- `python/src/server/services/crawler_manager.py` - Initialization patterns
- `python/src/server/services/crawling/strategies/batch.py` - MemoryAdaptiveDispatcher usage
- `python/src/server/services/crawling/strategies/recursive.py` - MemoryAdaptiveDispatcher usage

## Backup Plan
If 0.7.4 has breaking changes:
- Keep 0.6.2 but set `dispatcher=None` in batch/recursive strategies
- Add manual Chrome process cleanup in crawler_manager.py
- Implement container restart strategy

## Success Criteria
- No more 15GB+ memory consumption by workers
- Chrome processes properly cleaned up after tasks
- System remains stable during extended crawling operations
- Memory usage stays within reasonable bounds (< 2GB per container)

---
**Created**: 2025-09-03  
**Status**: Ready for implementation  
**Priority**: Critical - System stability issue