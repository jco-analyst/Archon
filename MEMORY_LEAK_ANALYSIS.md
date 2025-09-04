# Archon Memory Leak Analysis - September 3, 2025

## Issue Summary
Archon server experiencing massive memory leak in multiprocessing workers causing system instability.

## Problem Details

### System Impact
- **Memory Usage**: 15GB consumed by single Python process
- **System Performance**: Load average 16-17 (should be <4)
- **Swap Usage**: 83% (16GB of 19GB swap used)
- **I/O Wait**: 27-49% (system spending most time waiting for disk)

### Process Details
- **Main Process**: PID 2313857 - `python -m uvicorn src.server.main:socket_app --host 0.0.0.0 --port 8181 --reload`
- **Runaway Worker**: PID 2314234 - `/usr/local/bin/python -c from multiprocessing.spawn import spawn_main; spawn_main(tracker_fd=5, pipe_handle=7) --multiprocessing-fork`
- **Container**: Archon-Server (d44b2b98601c)
- **Port**: 8181

### Docker Container Stack
```
d44b2b98601c   archon-archon-server    "python -m uvicorn s…"    Up 45 minutes   0.0.0.0:8181->8181/tcp   Archon-Server
99d7096a0612   archon-archon-mcp       "python -m src.mcp.m…"    Up 45 minutes   0.0.0.0:8051->8051/tcp   Archon-MCP  
4d065efdf35b   archon-frontend         "docker-entrypoint.s…"    Up 45 minutes   0.0.0.0:3737->5173/tcp   Archon-UI
be6c998de6cc   archon-archon-agents    "/bin/sh -c 'sh -c \"…"   Up 45 minutes   0.0.0.0:8052->8052/tcp   Archon-Agents
```

### Process Tree Analysis
```
python(2313857) [uvicorn server]
├── python(2314233) [worker]
└── python(2314234) [runaway worker - 15GB memory]
    └── node(2315399)
        └── chrome(2315444)
            └── chrome(2315459)
                └── chrome(2315479)
```

## Root Cause Analysis

### Likely Causes
1. **Multiprocessing Worker Stuck**: Worker process in infinite loop or deadlock
2. **Memory Leak in Browser Automation**: Chrome processes spawned by worker suggest web scraping/automation
3. **Resource Cleanup Failure**: Worker not properly cleaning up after tasks
4. **Reload Mode Issue**: `--reload` flag may be causing worker spawning issues

### Contributing Factors
- **Multiprocessing Spawn Method**: Using spawn instead of fork may cause issues
- **Browser Automation**: Chrome processes indicate headless browser usage
- **Container Resource Limits**: No memory limits set on container

## Investigation Areas

### Code Locations to Check
1. **`src.server.main:socket_app`** - Main server entry point
2. **Multiprocessing configuration** - Check worker spawn settings
3. **Browser automation code** - Look for Chrome/Selenium usage
4. **WebSocket handling** - socket_app suggests WebSocket connections
5. **Task queues** - Background job processing

### System Monitoring
```bash
# Monitor memory usage
watch -n 1 'ps aux --sort=-%mem | head -10'

# Monitor specific process
watch -n 1 'procs --sortd memory | head -10'

# Check container resources
docker stats Archon-Server
```

## Immediate Fixes Applied

### Kill Runaway Process
```bash
sudo kill -9 2314234
```

### Container Restart
```bash
docker restart d44b2b98601c
```

## Prevention Strategies

### 1. Add Container Memory Limits
```yaml
# docker-compose.yml
services:
  archon-server:
    mem_limit: 2g
    memswap_limit: 2g
```

### 2. Process Monitoring
```bash
# Add to crontab for monitoring
*/5 * * * * ps aux | awk '$4 > 10 { print strftime("%Y-%m-%d %H:%M:%S"), $0 }' >> /var/log/memory-hogs.log
```

### 3. Code Review Areas
- [ ] Check multiprocessing worker cleanup
- [ ] Review browser automation resource management
- [ ] Add timeout mechanisms for long-running tasks
- [ ] Implement worker health checks
- [ ] Add memory usage monitoring

### 4. Alternative Configurations
- Consider using `--workers 1` instead of multiprocessing
- Switch from `--reload` to production mode
- Use process pools with explicit limits

## Diagnostic Commands

### Memory Analysis
```bash
# Check process memory maps
cat /proc/2314234/smaps | grep -E "^(VmSize|VmRSS|VmSwap)"

# Monitor real-time memory
top -p 2314234

# Check for memory leaks
valgrind --leak-check=full python your_script.py
```

### Container Debugging
```bash
# Enter container
docker exec -it Archon-Server /bin/bash

# Check container logs
docker logs Archon-Server --tail=100

# Monitor container resources
docker stats --no-stream Archon-Server
```

## Next Steps
1. **Code Review**: Focus on multiprocessing and browser automation code
2. **Add Monitoring**: Implement memory usage alerts
3. **Set Resource Limits**: Add memory constraints to containers
4. **Test Load**: Reproduce issue under controlled conditions
5. **Consider Architecture**: Evaluate if multiprocessing is necessary

---

**Date**: September 3, 2025  
**System**: PopOS 22.04, Docker containers  
**Archon Version**: Running in development mode with --reload  
**Impact**: Critical - System near unusable due to memory exhaustion