"""Optional RLinf sidecar bootstrap.

Python imports this module automatically when /mnt/public/zengwen/rlinf_utils is
on PYTHONPATH. It is inert unless one of the sidecar env flags is enabled.
"""

try:
    from rlinf_timeline.autopatch import install_from_env

    install_from_env()
except Exception as exc:  # pragma: no cover
    import os
    import sys

    if os.environ.get("RLINF_TIMELINE_DEBUG", "").lower() in {"1", "true", "yes"}:
        print(f"[rlinf_timeline] bootstrap failed: {exc}", file=sys.stderr)
