"""
Memory Monitoring Service for Archon - Phase 4 Memory Leak Prevention

This service provides comprehensive memory monitoring capabilities to detect
and alert on potential memory leaks before they impact system performance.

Features:
- Real-time memory usage tracking
- Configurable threshold alerts
- Memory leak detection heuristics
- Cleanup operation scheduling
- Integration with existing logging systems
"""

import asyncio
import gc
import logging
import os
import psutil
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

from ..config.logfire_config import get_logger

logger = get_logger(__name__)


@dataclass
class MemoryMetrics:
    """Memory usage metrics snapshot."""
    timestamp: datetime
    process_memory_mb: float
    system_memory_percent: float
    virtual_memory_mb: float
    rss_memory_mb: float
    heap_objects: int
    gc_collections: Dict[int, int]
    custom_metrics: Dict[str, Any] = field(default_factory=dict)


@dataclass
class MemoryThreshold:
    """Memory monitoring threshold configuration."""
    name: str
    metric_key: str  # Key to check in MemoryMetrics
    threshold_value: float
    comparison: str = "greater_than"  # "greater_than", "less_than", "percent_change"
    alert_cooldown_minutes: int = 5
    last_alert: Optional[datetime] = None
    enabled: bool = True


class MemoryMonitor:
    """
    Comprehensive memory monitoring service with leak detection and alerting.
    
    This service continuously monitors memory usage patterns and can detect
    potential memory leaks before they become critical system issues.
    """
    
    def __init__(self):
        self.process = psutil.Process()
        self.monitoring_active = False
        self.metrics_history: List[MemoryMetrics] = []
        self.max_history_size = 1440  # 24 hours at 1-minute intervals
        
        # Default monitoring thresholds
        self.thresholds: List[MemoryThreshold] = [
            MemoryThreshold(
                name="High Process Memory",
                metric_key="process_memory_mb", 
                threshold_value=1024.0,  # 1GB
                alert_cooldown_minutes=10
            ),
            MemoryThreshold(
                name="System Memory Usage",
                metric_key="system_memory_percent",
                threshold_value=85.0,  # 85%
                alert_cooldown_minutes=5
            ),
            MemoryThreshold(
                name="Memory Growth Rate",
                metric_key="process_memory_mb",
                threshold_value=20.0,  # 20% growth in 10 minutes
                comparison="percent_change",
                alert_cooldown_minutes=15
            )
        ]
        
        self._monitor_task: Optional[asyncio.Task] = None
        self._cleanup_task: Optional[asyncio.Task] = None
        
        logger.info("🧠 [MEMORY MONITOR] Memory monitoring service initialized")
    
    async def start_monitoring(self, interval_seconds: int = 60) -> None:
        """Start continuous memory monitoring."""
        if self.monitoring_active:
            logger.warning("🧠 [MEMORY MONITOR] Monitoring already active")
            return
        
        self.monitoring_active = True
        self._monitor_task = asyncio.create_task(self._monitoring_loop(interval_seconds))
        self._cleanup_task = asyncio.create_task(self._cleanup_loop())
        
        logger.info(f"🧠 [MEMORY MONITOR] Started monitoring with {interval_seconds}s interval")
    
    async def stop_monitoring(self) -> None:
        """Stop memory monitoring."""
        self.monitoring_active = False
        
        if self._monitor_task and not self._monitor_task.done():
            self._monitor_task.cancel()
            try:
                await self._monitor_task
            except asyncio.CancelledError:
                pass
        
        if self._cleanup_task and not self._cleanup_task.done():
            self._cleanup_task.cancel()
            try:
                await self._cleanup_task
            except asyncio.CancelledError:
                pass
        
        logger.info("🧠 [MEMORY MONITOR] Monitoring stopped")
    
    def get_current_metrics(self) -> MemoryMetrics:
        """Get current memory usage metrics."""
        try:
            # Process-specific memory info
            memory_info = self.process.memory_info()
            memory_percent = self.process.memory_percent()
            
            # System memory info
            system_memory = psutil.virtual_memory()
            
            # Python garbage collector info
            gc_stats = {}
            for generation in range(3):
                gc_stats[generation] = gc.get_count()[generation]
            
            # Count Python objects
            gc.collect()  # Force collection for accurate count
            heap_objects = len(gc.get_objects())
            
            # Custom metrics for Archon-specific tracking
            custom_metrics = self._get_archon_specific_metrics()
            
            return MemoryMetrics(
                timestamp=datetime.now(),
                process_memory_mb=memory_info.rss / (1024 * 1024),  # Convert to MB
                system_memory_percent=system_memory.percent,
                virtual_memory_mb=memory_info.vms / (1024 * 1024),
                rss_memory_mb=memory_info.rss / (1024 * 1024),
                heap_objects=heap_objects,
                gc_collections=gc_stats,
                custom_metrics=custom_metrics
            )
            
        except Exception as e:
            logger.error(f"🧠 [MEMORY MONITOR] Error collecting metrics: {e}")
            # Return minimal metrics on error
            return MemoryMetrics(
                timestamp=datetime.now(),
                process_memory_mb=0.0,
                system_memory_percent=0.0,
                virtual_memory_mb=0.0,
                rss_memory_mb=0.0,
                heap_objects=0,
                gc_collections={}
            )
    
    def _get_archon_specific_metrics(self) -> Dict[str, Any]:
        """Collect Archon-specific memory metrics."""
        metrics = {}
        
        try:
            # Import here to avoid circular dependencies
            from ..api_routes.socketio_handlers import (
                _last_broadcast_times, document_states, document_locks
            )
            from ..api_routes.agent_chat_api import sessions
            
            metrics.update({
                "broadcast_times_count": len(_last_broadcast_times),
                "document_states_count": len(document_states),
                "document_locks_count": len(document_locks),
                "chat_sessions_count": len(sessions),
                "total_tracked_objects": (
                    len(_last_broadcast_times) + len(document_states) + 
                    len(document_locks) + len(sessions)
                )
            })
            
        except ImportError:
            # Services may not be available during startup
            logger.debug("🧠 [MEMORY MONITOR] Some services not available for metrics collection")
        except Exception as e:
            logger.warning(f"🧠 [MEMORY MONITOR] Error collecting custom metrics: {e}")
        
        return metrics
    
    async def _monitoring_loop(self, interval_seconds: int) -> None:
        """Main monitoring loop."""
        logger.info("🧠 [MEMORY MONITOR] Monitoring loop started")
        
        while self.monitoring_active:
            try:
                # Collect current metrics
                metrics = self.get_current_metrics()
                self.metrics_history.append(metrics)
                
                # Trim history to prevent memory leaks in the monitor itself
                if len(self.metrics_history) > self.max_history_size:
                    self.metrics_history = self.metrics_history[-self.max_history_size:]
                
                # Check thresholds and send alerts
                await self._check_thresholds(metrics)
                
                # Log metrics periodically (every 10 minutes)
                if len(self.metrics_history) % 10 == 0:
                    await self._log_memory_status(metrics)
                
                await asyncio.sleep(interval_seconds)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"🧠 [MEMORY MONITOR] Error in monitoring loop: {e}")
                await asyncio.sleep(interval_seconds)  # Continue monitoring despite errors
        
        logger.info("🧠 [MEMORY MONITOR] Monitoring loop stopped")
    
    async def _cleanup_loop(self) -> None:
        """Periodic cleanup operations to prevent memory leaks."""
        logger.info("🧠 [MEMORY MONITOR] Cleanup loop started")
        
        while self.monitoring_active:
            try:
                await asyncio.sleep(600)  # Run cleanup every 10 minutes
                
                if not self.monitoring_active:
                    break
                
                await self._run_periodic_cleanup()
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"🧠 [MEMORY MONITOR] Error in cleanup loop: {e}")
        
        logger.info("🧠 [MEMORY MONITOR] Cleanup loop stopped")
    
    async def _run_periodic_cleanup(self) -> None:
        """Run periodic cleanup operations."""
        logger.info("🧠 [MEMORY MONITOR] Running periodic cleanup")
        cleanup_summary = []
        
        try:
            # Clean up chat sessions
            from ..api_routes.agent_chat_api import cleanup_chat_sessions
            sessions_cleaned = cleanup_chat_sessions(max_age_hours=3)  # More aggressive than disconnect
            if sessions_cleaned > 0:
                cleanup_summary.append(f"{sessions_cleaned} chat sessions")
            
        except Exception as e:
            logger.warning(f"🧠 [MEMORY MONITOR] Failed to clean chat sessions: {e}")
        
        try:
            # Force garbage collection
            collected_objects = gc.collect()
            if collected_objects > 0:
                cleanup_summary.append(f"{collected_objects} Python objects")
            
        except Exception as e:
            logger.warning(f"🧠 [MEMORY MONITOR] Failed to run garbage collection: {e}")
        
        if cleanup_summary:
            logger.info(f"🧠 [MEMORY MONITOR] Periodic cleanup completed: {', '.join(cleanup_summary)}")
        else:
            logger.debug("🧠 [MEMORY MONITOR] Periodic cleanup completed - no items to clean")
    
    async def _check_thresholds(self, metrics: MemoryMetrics) -> None:
        """Check memory thresholds and send alerts if exceeded."""
        current_time = datetime.now()
        
        for threshold in self.thresholds:
            if not threshold.enabled:
                continue
            
            # Check cooldown period
            if (threshold.last_alert and 
                current_time - threshold.last_alert < timedelta(minutes=threshold.alert_cooldown_minutes)):
                continue
            
            # Get metric value
            metric_value = self._get_metric_value(metrics, threshold.metric_key)
            if metric_value is None:
                continue
            
            # Check threshold
            alert_triggered = False
            
            if threshold.comparison == "greater_than":
                alert_triggered = metric_value > threshold.threshold_value
            elif threshold.comparison == "less_than":
                alert_triggered = metric_value < threshold.threshold_value
            elif threshold.comparison == "percent_change":
                alert_triggered = self._check_percent_change(threshold.metric_key, threshold.threshold_value)
            
            if alert_triggered:
                await self._send_alert(threshold, metric_value, metrics)
                threshold.last_alert = current_time
    
    def _get_metric_value(self, metrics: MemoryMetrics, metric_key: str) -> Optional[float]:
        """Extract metric value from MemoryMetrics object."""
        if hasattr(metrics, metric_key):
            return getattr(metrics, metric_key)
        elif metric_key in metrics.custom_metrics:
            return metrics.custom_metrics[metric_key]
        else:
            logger.warning(f"🧠 [MEMORY MONITOR] Unknown metric key: {metric_key}")
            return None
    
    def _check_percent_change(self, metric_key: str, threshold_percent: float) -> bool:
        """Check if metric has grown by threshold_percent in the last 10 minutes."""
        if len(self.metrics_history) < 10:
            return False
        
        current_value = self._get_metric_value(self.metrics_history[-1], metric_key)
        past_value = self._get_metric_value(self.metrics_history[-10], metric_key)
        
        if current_value is None or past_value is None or past_value == 0:
            return False
        
        percent_change = ((current_value - past_value) / past_value) * 100
        return percent_change > threshold_percent
    
    async def _send_alert(self, threshold: MemoryThreshold, metric_value: float, metrics: MemoryMetrics) -> None:
        """Send memory threshold alert."""
        alert_msg = (
            f"🚨 [MEMORY ALERT] {threshold.name}: "
            f"{threshold.metric_key}={metric_value:.2f} exceeds threshold {threshold.threshold_value:.2f}"
        )
        
        logger.warning(alert_msg)
        
        # Also log current system state for debugging
        logger.warning(
            f"🚨 [MEMORY ALERT] System state - "
            f"Process: {metrics.process_memory_mb:.1f}MB, "
            f"System: {metrics.system_memory_percent:.1f}%, "
            f"Objects: {metrics.heap_objects:,}, "
            f"Tracked: {metrics.custom_metrics.get('total_tracked_objects', 'N/A')}"
        )
    
    async def _log_memory_status(self, metrics: MemoryMetrics) -> None:
        """Log current memory status."""
        logger.info(
            f"🧠 [MEMORY STATUS] "
            f"Process: {metrics.process_memory_mb:.1f}MB, "
            f"System: {metrics.system_memory_percent:.1f}%, "
            f"Objects: {metrics.heap_objects:,}, "
            f"Broadcast times: {metrics.custom_metrics.get('broadcast_times_count', 0)}, "
            f"Doc states: {metrics.custom_metrics.get('document_states_count', 0)}, "
            f"Chat sessions: {metrics.custom_metrics.get('chat_sessions_count', 0)}"
        )
    
    def get_memory_report(self) -> Dict[str, Any]:
        """Generate comprehensive memory usage report."""
        if not self.metrics_history:
            return {"error": "No metrics available"}
        
        current_metrics = self.metrics_history[-1]
        
        # Calculate trends
        memory_trend = "stable"
        if len(self.metrics_history) >= 10:
            recent_avg = sum(m.process_memory_mb for m in self.metrics_history[-5:]) / 5
            older_avg = sum(m.process_memory_mb for m in self.metrics_history[-10:-5]) / 5
            
            if recent_avg > older_avg * 1.1:
                memory_trend = "increasing"
            elif recent_avg < older_avg * 0.9:
                memory_trend = "decreasing"
        
        return {
            "timestamp": current_metrics.timestamp.isoformat(),
            "current_memory_mb": current_metrics.process_memory_mb,
            "system_memory_percent": current_metrics.system_memory_percent,
            "heap_objects": current_metrics.heap_objects,
            "memory_trend": memory_trend,
            "custom_metrics": current_metrics.custom_metrics,
            "thresholds_status": [
                {
                    "name": t.name,
                    "enabled": t.enabled,
                    "last_alert": t.last_alert.isoformat() if t.last_alert else None
                }
                for t in self.thresholds
            ],
            "metrics_history_count": len(self.metrics_history)
        }


# Global memory monitor instance
_memory_monitor: Optional[MemoryMonitor] = None


def get_memory_monitor() -> MemoryMonitor:
    """Get the global memory monitor instance."""
    global _memory_monitor
    if _memory_monitor is None:
        _memory_monitor = MemoryMonitor()
    return _memory_monitor


async def start_memory_monitoring() -> None:
    """Start the global memory monitoring service."""
    monitor = get_memory_monitor()
    await monitor.start_monitoring()


async def stop_memory_monitoring() -> None:
    """Stop the global memory monitoring service."""
    monitor = get_memory_monitor()
    await monitor.stop_monitoring()