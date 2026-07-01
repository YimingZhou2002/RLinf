"""Sidecar timeline tracing utilities for RLinf."""

from .writer import append_event, append_timeline_event, is_enabled

__all__ = ["append_event", "append_timeline_event", "is_enabled"]
