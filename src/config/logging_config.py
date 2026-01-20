"""Centralized logging configuration with flow tracing support."""

from __future__ import annotations

import functools
import logging
import sys
import threading
import time
import uuid
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional, TypeVar

F = TypeVar('F', bound=Callable[..., Any])


class GraphRAGFormatter(logging.Formatter):
    """Custom formatter with color support and structured output."""

    COLORS = {
        'DEBUG': '\033[36m',
        'INFO': '\033[32m',
        'WARNING': '\033[33m',
        'ERROR': '\033[31m',
        'CRITICAL': '\033[35m',
        'RESET': '\033[0m',
        'DIM': '\033[2m',
    }

    STEP_ICONS = {
        'start': '▶',
        'end': '✓',
        'error': '✗',
        'info': '•',
        'substep': '  ↳',
    }

    def __init__(self, fmt: Optional[str] = None, datefmt: Optional[str] = None, use_colors: bool = True):
        super().__init__(fmt, datefmt)
        self.use_colors = use_colors and sys.stdout.isatty()

    def format(self, record: logging.LogRecord) -> str:
        trace_id = getattr(record, 'trace_id', None)
        step_type = getattr(record, 'step_type', None)
        duration = getattr(record, 'duration', None)

        prefix_parts = []
        if trace_id:
            prefix_parts.append(f"[{trace_id[:8]}]")
        if step_type and step_type in self.STEP_ICONS:
            prefix_parts.append(self.STEP_ICONS[step_type])
        prefix = " ".join(prefix_parts) + " " if prefix_parts else ""

        suffix = f" ({duration:.3f}s)" if duration is not None else ""

        if self.use_colors:
            level_color = self.COLORS.get(record.levelname, '')
            reset = self.COLORS['RESET']
            dim = self.COLORS['DIM']
            timestamp = datetime.fromtimestamp(record.created).strftime('%H:%M:%S.%f')[:-3]
            return (
                f"{dim}{timestamp}{reset} | "
                f"{level_color}{record.levelname:8}{reset} | "
                f"{dim}{record.name:30}{reset} | "
                f"{prefix}{record.getMessage()}{suffix}"
            )
        return f"{prefix}{super().format(record)}{suffix}"


@dataclass
class TraceContext:
    """Context for tracking execution flow."""

    trace_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    steps: List[Dict[str, Any]] = field(default_factory=list)
    start_time: float = field(default_factory=time.time)
    current_step: int = 0

    def add_step(self, name: str, status: str = "completed", duration: Optional[float] = None,
                 details: Optional[Dict[str, Any]] = None) -> None:
        self.current_step += 1
        self.steps.append({
            "step": self.current_step,
            "name": name,
            "status": status,
            "duration": duration,
            "details": details or {},
            "timestamp": time.time(),
        })

    def get_summary(self) -> Dict[str, Any]:
        return {
            "trace_id": self.trace_id,
            "total_duration": time.time() - self.start_time,
            "step_count": len(self.steps),
            "steps": self.steps,
        }


_trace_context = threading.local()


def get_current_trace() -> Optional[TraceContext]:
    return getattr(_trace_context, 'current', None)


def set_current_trace(trace: Optional[TraceContext]) -> None:
    _trace_context.current = trace


class FlowLogger:
    """Logger wrapper with flow tracing capabilities."""

    def __init__(self, name: str):
        self.logger = logging.getLogger(name)
        self.name = name

    def _log_with_context(self, level: int, msg: str, step_type: Optional[str] = None,
                          duration: Optional[float] = None, **kwargs) -> None:
        trace = get_current_trace()
        extra = kwargs.pop('extra', {})
        extra['step_type'] = step_type
        extra['duration'] = duration
        extra['trace_id'] = trace.trace_id if trace else None
        self.logger.log(level, msg, extra=extra, **kwargs)

    def step_start(self, step_name: str, details: str = "") -> float:
        msg = f"Starting: {step_name}" + (f" - {details}" if details else "")
        self._log_with_context(logging.INFO, msg, step_type='start')
        return time.time()

    def step_end(self, step_name: str, start_time: float, details: str = "") -> None:
        duration = time.time() - start_time
        msg = f"Completed: {step_name}" + (f" - {details}" if details else "")
        self._log_with_context(logging.INFO, msg, step_type='end', duration=duration)
        trace = get_current_trace()
        if trace:
            trace.add_step(step_name, "completed", duration)

    def step_error(self, step_name: str, error: Exception, start_time: Optional[float] = None) -> None:
        duration = time.time() - start_time if start_time else None
        msg = f"Failed: {step_name} - {type(error).__name__}: {error}"
        self._log_with_context(logging.ERROR, msg, step_type='error', duration=duration)
        trace = get_current_trace()
        if trace:
            trace.add_step(step_name, "failed", duration, {"error": str(error)})

    def substep(self, msg: str) -> None:
        self._log_with_context(logging.DEBUG, msg, step_type='substep')

    def info(self, msg: str, **kwargs) -> None:
        self._log_with_context(logging.INFO, msg, step_type='info', **kwargs)

    def debug(self, msg: str, **kwargs) -> None:
        self._log_with_context(logging.DEBUG, msg, **kwargs)

    def warning(self, msg: str, **kwargs) -> None:
        self._log_with_context(logging.WARNING, msg, **kwargs)

    def error(self, msg: str, **kwargs) -> None:
        self._log_with_context(logging.ERROR, msg, **kwargs)


def get_flow_logger(name: str) -> FlowLogger:
    return FlowLogger(name)


def get_logger(name: str) -> FlowLogger:
    return FlowLogger(name)


def trace_step(step_name: Optional[str] = None):
    """Decorator to trace a function as a step."""
    def decorator(func: F) -> F:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            name = step_name or func.__name__
            logger = get_flow_logger(func.__module__)
            start = logger.step_start(name)
            try:
                result = func(*args, **kwargs)
                logger.step_end(name, start)
                return result
            except Exception as e:
                logger.step_error(name, e, start)
                raise
        return wrapper  # type: ignore
    return decorator


def trace_flow(flow_name: str):
    """Decorator to trace an entire flow with a new trace context."""
    def decorator(func: F) -> F:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            logger = get_flow_logger(func.__module__)
            trace = TraceContext()
            set_current_trace(trace)
            logger.info(f"{'='*60}")
            logger.info(f"FLOW START: {flow_name} [Trace: {trace.trace_id[:8]}]")
            logger.info(f"{'='*60}")
            start = time.time()
            try:
                result = func(*args, **kwargs)
                duration = time.time() - start
                logger.info(f"{'='*60}")
                logger.info(f"FLOW COMPLETE: {flow_name} ({duration:.3f}s)")
                logger.info(f"Steps completed: {len(trace.steps)}")
                logger.info(f"{'='*60}")
                return result
            except Exception as e:
                duration = time.time() - start
                logger.error(f"{'='*60}")
                logger.error(f"FLOW FAILED: {flow_name} ({duration:.3f}s)")
                logger.error(f"Error: {type(e).__name__}: {e}")
                logger.error(f"{'='*60}")
                raise
            finally:
                set_current_trace(None)
        return wrapper  # type: ignore
    return decorator


@contextmanager
def trace_context(flow_name: str):
    """Context manager for tracing a flow."""
    logger = get_flow_logger(__name__)
    trace = TraceContext()
    set_current_trace(trace)
    logger.info(f"{'='*60}")
    logger.info(f"FLOW START: {flow_name} [Trace: {trace.trace_id[:8]}]")
    logger.info(f"{'='*60}")
    start = time.time()
    try:
        yield trace
        duration = time.time() - start
        logger.info(f"{'='*60}")
        logger.info(f"FLOW COMPLETE: {flow_name} ({duration:.3f}s)")
        logger.info(f"{'='*60}")
    except Exception as e:
        duration = time.time() - start
        logger.error(f"{'='*60}")
        logger.error(f"FLOW FAILED: {flow_name} ({duration:.3f}s)")
        logger.error(f"Error: {type(e).__name__}: {e}")
        logger.error(f"{'='*60}")
        raise
    finally:
        set_current_trace(None)


@contextmanager
def log_step(logger: FlowLogger, step_name: str, details: str = ""):
    """Context manager for logging a step."""
    start = logger.step_start(step_name, details)
    try:
        yield
        logger.step_end(step_name, start)
    except Exception as e:
        logger.step_error(step_name, e, start)
        raise


def configure_logging(level: int = logging.INFO, use_colors: bool = True,
                      log_file: Optional[str] = None, detailed: bool = False) -> None:
    """Configure logging for the application."""
    root = logging.getLogger()
    root.handlers.clear()
    root.setLevel(level)

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_handler.setFormatter(GraphRAGFormatter(use_colors=use_colors))
    root.addHandler(console_handler)

    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(level)
        file_handler.setFormatter(logging.Formatter(
            "%(asctime)s | %(levelname)-8s | %(name)-30s | %(message)s"
        ))
        root.addHandler(file_handler)

    for logger_name in ["httpx", "httpcore", "neo4j", "urllib3"]:
        logging.getLogger(logger_name).setLevel(logging.WARNING)

    if not detailed:
        for logger_name in ["langchain", "langchain_community"]:
            logging.getLogger(logger_name).setLevel(logging.WARNING)
