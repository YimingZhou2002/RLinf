"""Backward-compatible entry point for existing cfg-based timeline call sites."""

from rlinf_timeline.writer import append_timeline_event

__all__ = ["append_timeline_event"]
