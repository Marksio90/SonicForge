"""Memory Management Daemon for 24/7 Long-Running Processes.

On an 8 GB laptop, MusicGen-small occupies 4–6 GB, leaving tight headroom.
This daemon handles memory pressure through several complementary strategies
(based on research findings, 2026-02):

1. **Periodic GC** — `gc.collect()` every 5 minutes to catch slow leaks
2. **Bounded data structures** — enforces maxsize on deques and caches
3. **psutil monitoring** — alerts when RSS exceeds threshold
4. **Scheduled restart** — process restart every 12 hours as safety valve
5. **Swap advisory** — logs warning if system swap is insufficient

Recommended system configuration:
  - 8 GB RAM: set 8 GB swap (`sudo swapoff -a && sudo dd if=/dev/zero of=/swapfile
    bs=1G count=8 && sudo mkswap /swapfile && sudo swapon /swapfile`)
  - 16 GB RAM: comfortable without swap pressure
"""

import asyncio
import gc
import logging
import os
import sys
import time
from collections import deque
from typing import Any

import structlog

logger = structlog.get_logger(__name__)

# Memory thresholds
MEMORY_WARNING_MB = int(os.environ.get("MEMORY_WARNING_MB", "6000"))  # 6 GB
MEMORY_CRITICAL_MB = int(os.environ.get("MEMORY_CRITICAL_MB", "7500"))  # 7.5 GB
GC_INTERVAL_SECONDS = int(os.environ.get("GC_INTERVAL_SECONDS", "300"))  # 5 min
PROCESS_RESTART_INTERVAL_HOURS = int(os.environ.get("PROCESS_RESTART_HOURS", "12"))
SWAP_WARNING_GB = float(os.environ.get("SWAP_WARNING_GB", "4.0"))


class MemoryManager:
    """Daemon that monitors and manages memory for long-running SonicForge processes.

    Attach to the FastAPI application lifecycle:

        from app.services.memory_manager import MemoryManager
        memory_mgr = MemoryManager()

        @app.on_event("startup")
        async def startup():
            await memory_mgr.start()

        @app.on_event("shutdown")
        async def shutdown():
            await memory_mgr.stop()
    """

    def __init__(self):
        self._gc_task: asyncio.Task | None = None
        self._monitor_task: asyncio.Task | None = None
        self._running = False
        self._start_time = time.monotonic()
        self._gc_count = 0
        self._peak_rss_mb = 0.0
        # Rolling 1-hour window of memory samples (one per minute)
        self._memory_history: deque[float] = deque(maxlen=60)

    async def start(self) -> None:
        """Start background memory management tasks."""
        if self._running:
            return
        self._running = True
        self._start_time = time.monotonic()

        self._gc_task = asyncio.create_task(self._gc_loop(), name="memory_gc_loop")
        self._monitor_task = asyncio.create_task(
            self._monitor_loop(), name="memory_monitor_loop"
        )
        self._check_swap()
        logger.info(
            "memory_manager_started",
            gc_interval_s=GC_INTERVAL_SECONDS,
            warning_mb=MEMORY_WARNING_MB,
            critical_mb=MEMORY_CRITICAL_MB,
            restart_hours=PROCESS_RESTART_INTERVAL_HOURS,
        )

    async def stop(self) -> None:
        """Cancel background tasks gracefully."""
        self._running = False
        for task in (self._gc_task, self._monitor_task):
            if task and not task.done():
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass
        logger.info("memory_manager_stopped")

    async def _gc_loop(self) -> None:
        """Run garbage collection on a fixed interval."""
        while self._running:
            await asyncio.sleep(GC_INTERVAL_SECONDS)
            before = self._get_rss_mb()
            collected = gc.collect()
            after = self._get_rss_mb()
            freed = max(0.0, before - after)
            self._gc_count += 1

            logger.info(
                "gc_cycle",
                cycle=self._gc_count,
                objects_collected=collected,
                rss_before_mb=round(before, 1),
                rss_after_mb=round(after, 1),
                freed_mb=round(freed, 1),
            )

    async def _monitor_loop(self) -> None:
        """Monitor memory every minute and trigger actions at thresholds."""
        while self._running:
            await asyncio.sleep(60)
            rss_mb = self._get_rss_mb()
            self._memory_history.append(rss_mb)
            self._peak_rss_mb = max(self._peak_rss_mb, rss_mb)

            uptime_hours = (time.monotonic() - self._start_time) / 3600

            if rss_mb >= MEMORY_CRITICAL_MB:
                logger.error(
                    "memory_critical",
                    rss_mb=round(rss_mb),
                    threshold_mb=MEMORY_CRITICAL_MB,
                    uptime_hours=round(uptime_hours, 1),
                )
                # Force aggressive GC
                gc.collect(generation=2)

            elif rss_mb >= MEMORY_WARNING_MB:
                logger.warning(
                    "memory_high",
                    rss_mb=round(rss_mb),
                    threshold_mb=MEMORY_WARNING_MB,
                    uptime_hours=round(uptime_hours, 1),
                )
                gc.collect()

            # Scheduled restart every N hours as safety valve
            if (
                PROCESS_RESTART_INTERVAL_HOURS > 0
                and uptime_hours >= PROCESS_RESTART_INTERVAL_HOURS
            ):
                logger.info(
                    "scheduled_restart",
                    uptime_hours=round(uptime_hours, 1),
                    rss_mb=round(rss_mb),
                    peak_rss_mb=round(self._peak_rss_mb),
                )
                # Give supervisord/Docker restart: always a chance to restart cleanly
                sys.exit(0)

    def _get_rss_mb(self) -> float:
        """Get current process RSS in megabytes."""
        try:
            import psutil
            process = psutil.Process(os.getpid())
            return process.memory_info().rss / (1024 * 1024)
        except ImportError:
            # Fallback: read /proc/self/status
            try:
                with open("/proc/self/status") as f:
                    for line in f:
                        if line.startswith("VmRSS:"):
                            return float(line.split()[1]) / 1024  # kB → MB
            except OSError:
                pass
            return 0.0

    def _check_swap(self) -> None:
        """Warn if system swap is insufficient for model loading."""
        try:
            import psutil
            swap = psutil.swap_memory()
            swap_gb = swap.total / (1024 ** 3)
            if swap_gb < SWAP_WARNING_GB:
                logger.warning(
                    "insufficient_swap",
                    swap_gb=round(swap_gb, 1),
                    recommended_gb=SWAP_WARNING_GB,
                    hint=(
                        "Low swap may cause OOM kills during model loading. "
                        "Recommended: sudo fallocate -l 8G /swapfile && "
                        "sudo chmod 600 /swapfile && sudo mkswap /swapfile && "
                        "sudo swapon /swapfile"
                    ),
                )
        except ImportError:
            pass

    def get_stats(self) -> dict:
        """Return current memory statistics for health checks."""
        rss_mb = self._get_rss_mb()
        uptime_hours = (time.monotonic() - self._start_time) / 3600
        avg_mb = (
            sum(self._memory_history) / len(self._memory_history)
            if self._memory_history
            else 0.0
        )
        return {
            "rss_mb": round(rss_mb, 1),
            "peak_rss_mb": round(self._peak_rss_mb, 1),
            "avg_rss_1h_mb": round(avg_mb, 1),
            "gc_cycles": self._gc_count,
            "uptime_hours": round(uptime_hours, 2),
            "warning_threshold_mb": MEMORY_WARNING_MB,
            "critical_threshold_mb": MEMORY_CRITICAL_MB,
            "next_restart_in_hours": max(
                0,
                PROCESS_RESTART_INTERVAL_HOURS - uptime_hours,
            ) if PROCESS_RESTART_INTERVAL_HOURS > 0 else None,
        }


# Global singleton — attach to FastAPI lifespan
memory_manager = MemoryManager()
