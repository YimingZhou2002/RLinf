"""Batch driver for `run_embodiment.sh`.

Reads a YAML file describing a list of runs (see `batch_runs.yaml`), and launches
each one serially by invoking `run_embodiment.sh` with Hydra dotted-key overrides
passed through the `EXTRA_OVERRIDES` env var. On failure of a single run it records
the error and continues to the next; a final summary is written to the batch dir.

Usage:
    python toolkits/batch_run/batch_run.py [path/to/batch_runs.yaml]
"""

import json
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

from omegaconf import OmegaConf

# Resolve repo root from this file's location (toolkits/batch_run/batch_run.py).
REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_DIR = Path(__file__).resolve().parent
RUN_EMBODIMENT_SH = REPO_ROOT / "examples" / "embodiment" / "run_embodiment.sh"
DEFAULT_CONFIG = SCRIPT_DIR / "batch_runs.yaml"

# Profiler sidecar plotters. These live outside this repo (see README in the
# profiler dir). They consume the timeline/ and nvitop/ JSONL dirs produced
# during a run and render interactive HTML.
PROFILER_DIR = Path(
    "/mnt/public2/zhouyiming/humanize/RLinf/toolkits/embodied_tuner/profiler"
)
PLOT_TIMELINE = PROFILER_DIR / "plot_timeline.py"
PLOT_NVITOP = PROFILER_DIR / "plot_nvitop.py"


def _fmt_override_value(v):
    """Render a scalar for a Hydra CLI override."""
    if isinstance(v, bool):
        return "true" if v else "false"
    return str(v)


def build_overrides(run, log_path: str):
    """Flatten one run entry into a list of Hydra dotted-key overrides.

    Only fields actually present are emitted so omitted fields fall back to the
    config yaml defaults. `runner.logger.log_path` is always appended last so it
    wins over the value set by run_embodiment.sh.
    """
    overrides = []

    cp = run.get("component_placement") or {}
    for sub in ("actor", "env", "rollout"):
        if sub in cp:
            overrides.append(
                f"cluster.component_placement.{sub}={_fmt_override_value(cp[sub])}"
            )

    rollout = run.get("rollout") or {}
    if "enable_offload" in rollout:
        overrides.append(
            f"rollout.enable_offload={_fmt_override_value(rollout['enable_offload'])}"
        )

    env_train = run.get("env_train") or {}
    if "total_num_envs" in env_train:
        overrides.append(
            f"env.train.total_num_envs={_fmt_override_value(env_train['total_num_envs'])}"
        )
    if "rollout_epoch" in env_train:
        overrides.append(
            f"env.train.rollout_epoch={_fmt_override_value(env_train['rollout_epoch'])}"
        )
    if "enable_offload" in env_train:
        overrides.append(
            f"env.train.enable_offload={_fmt_override_value(env_train['enable_offload'])}"
        )

    actor = run.get("actor") or {}
    if "micro_batch_size" in actor:
        overrides.append(
            f"actor.micro_batch_size={_fmt_override_value(actor['micro_batch_size'])}"
        )
    if "enable_offload" in actor:
        overrides.append(
            f"actor.enable_offload={_fmt_override_value(actor['enable_offload'])}"
        )

    for raw in run.get("extra") or []:
        overrides.append(str(raw))

    # Append last so it overrides run_embodiment.sh's own runner.logger.log_path.
    overrides.append(f"runner.logger.log_path={log_path}")
    return overrides


def snapshot_dict(run, overrides, log_path):
    """Human-readable record of the key params for this run."""
    cp = run.get("component_placement") or {}
    rollout = run.get("rollout") or {}
    env_train = run.get("env_train") or {}
    actor = run.get("actor") or {}
    return {
        "name": run.get("name"),
        "key_params": {
            "cluster.component_placement.actor": cp.get("actor"),
            "cluster.component_placement.env": cp.get("env"),
            "cluster.component_placement.rollout": cp.get("rollout"),
            "rollout.enable_offload": rollout.get("enable_offload"),
            "env.train.enable_offload": env_train.get("enable_offload"),
            "env.train.total_num_envs": env_train.get("total_num_envs"),
            "env.train.rollout_epoch": env_train.get("rollout_epoch"),
            "actor.enable_offload": actor.get("enable_offload"),
            "actor.micro_batch_size": actor.get("micro_batch_size"),
        },
        "extra": list(run.get("extra") or []),
        "overrides": overrides,
        "runner.logger.log_path": log_path,
    }


def write_snapshot(snapshot, run_dir: Path):
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "config_snapshot.json").write_text(
        json.dumps(snapshot, indent=2, ensure_ascii=False)
    )
    # Markdown human-readable version.
    lines = [f"# Run: {snapshot['name']}", ""]
    lines.append("## Key parameters")
    lines.append("| Key | Value |")
    lines.append("|---|---|")
    for k, v in snapshot["key_params"].items():
        lines.append(f"| `{k}` | `{v}` |")
    lines.append("")
    lines.append("## Overrides emitted")
    for o in snapshot["overrides"]:
        lines.append(f"- `{o}`")
    lines.append("")
    lines.append(f"rl logger log_path: `{snapshot['runner.logger.log_path']}`")
    (run_dir / "config_snapshot.md").write_text("\n".join(lines))


def stream_process(cmd, env, log_file: Path):
    """Run cmd, streaming merged stdout/stderr to both console and log_file.

    Wraps the command with `stdbuf -oL -eL` and sets PYTHONUNBUFFERED=1 so the
    nested `run_embodiment.sh | tee` pipeline flushes line-by-line; otherwise
    block-buffered Python/tee output leaves the log_file empty until the process
    exits or fills its buffer.
    """
    log_file.parent.mkdir(parents=True, exist_ok=True)
    env = env.copy()
    env["PYTHONUNBUFFERED"] = "1"
    # stdbuf forces line buffering on stdout/stderr of the whole pipeline, so
    # we receive each line as it is produced instead of in 4KB/8KB chunks.
    wrapped = ["stdbuf", "-oL", "-eL"] + list(cmd)
    with log_file.open("w") as f:
        proc = subprocess.Popen(
            wrapped,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            cwd=str(REPO_ROOT),
            env=env,
            text=True,
            bufsize=1,
        )
        assert proc.stdout is not None
        for line in proc.stdout:
            sys.stdout.write(line)
            f.write(line)
        return proc.wait()


def _plot_one(plot_script: Path, data_dir: Path, run_dir: Path):
    """Run one profiler plotter against ``data_dir``.

    Returns the path to the generated HTML, or None when there is nothing to plot
    (missing/empty dir, plotter unavailable, or the plotter errored). Errors are
    logged but never raised so a plotting failure cannot abort the batch.
    """
    if not plot_script.exists():
        print(f"[batch] plotter not found, skip: {plot_script}", file=sys.stderr)
        return None
    if not data_dir.exists() or not any(data_dir.iterdir()):
        print(f"[batch] no profile data, skip {plot_script.name}: {data_dir}")
        return None
    try:
        # Plotters default to HTML written into their data dir; keep that so the
        # output lands next to its JSONL source.
        proc = subprocess.run(
            ["python", str(plot_script), str(data_dir)],
            cwd=str(REPO_ROOT),
            capture_output=True,
            text=True,
            timeout=600,
        )
        if proc.returncode != 0:
            print(
                f"[batch] {plot_script.name} failed (rc={proc.returncode}): "
                f"{proc.stderr.strip()[:500]}",
                file=sys.stderr,
            )
            return None
        # The plotter prints the output path it wrote.
        out = proc.stdout.strip().splitlines()[-1].strip() if proc.stdout.strip() else ""
        if out:
            return out
        # Fallback: infer the default output name the plotter uses.
        default = {
            "plot_timeline.py": "timeline_gantt.html",
            "plot_nvitop.py": "nvitop_resources.html",
        }.get(plot_script.name)
        if default and (data_dir / default).exists():
            return str(data_dir / default)
        return None
    except subprocess.TimeoutExpired:
        print(f"[batch] {plot_script.name} timed out, skip", file=sys.stderr)
        return None
    except Exception as e:  # noqa: BLE001 - plotting must not abort the batch
        print(f"[batch] {plot_script.name} error: {e}", file=sys.stderr)
        return None


def plot_run_profiles(run_dir: Path):
    """Render timeline + nvitop HTML for one completed run.

    Called after each run finishes (success or failure). Returns a dict of the
    produced HTML paths keyed by profile name.
    """
    out = {}
    tl = _plot_one(PLOT_TIMELINE, run_dir / "timeline", run_dir)
    if tl:
        out["timeline_html"] = tl
    nv = _plot_one(PLOT_NVITOP, run_dir / "nvitop", run_dir)
    if nv:
        out["nvitop_html"] = nv
    if out:
        print(f"[batch] profile html: {out}")
    return out


def main() -> int:
    config_path = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_CONFIG
    if not config_path.exists():
        print(f"[batch] config not found: {config_path}", file=sys.stderr)
        return 2

    cfg = OmegaConf.load(config_path)
    config_name = cfg.get("config_name", "maniskill_ppo_openvla")
    robot_platform = cfg.get("robot_platform", "LIBERO")
    runs = list(cfg.get("runs") or [])
    if not runs:
        print("[batch] no runs defined; nothing to do.")
        return 0

    # Per-run names must be unique within the batch.
    seen_names = {}
    for r in runs:
        n = r.get("name") or "run"
        seen_names[n] = seen_names.get(n, 0) + 1
        if seen_names[n] > 1:
            r["name"] = f"{n}_{seen_names[n]}"

    batch_ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    batch_dir = REPO_ROOT / "logs" / f"batch_{batch_ts}"
    batch_dir.mkdir(parents=True, exist_ok=True)
    print(f"[batch] batch dir: {batch_dir}")

    results = []
    any_failed = False
    for idx, run in enumerate(runs, 1):
        name = run.get("name")
        run_dir = batch_dir / name
        log_path = str(run_dir)
        overrides = build_overrides(run, log_path)
        snapshot = snapshot_dict(run, overrides, log_path)
        write_snapshot(snapshot, run_dir)
        console_log = run_dir / f"{name}.log"

        extra = " ".join(overrides)
        env = os.environ.copy()
        env["EXTRA_OVERRIDES"] = extra
        # Tell run_embodiment.sh to put run_embodiment.log (stdout) inside this
        # run's dir, so stdout, snapshot and rl-logger outputs all colocate.
        env["RUN_LOG_DIR"] = str(run_dir)

        cmd = ["bash", str(RUN_EMBODIMENT_SH), str(config_name), str(robot_platform)]
        print(
            f"\n[batch] ({idx}/{len(runs)}) run='{name}' "
            f"overrides='{extra}'"
        )
        t0 = time.time()
        try:
            rc = stream_process(cmd, env, console_log)
            ok = rc == 0
        except Exception as e:  # noqa: BLE001 - keep going
            rc = -1
            ok = False
            with console_log.open("a") as f:
                f.write(f"\n[batch] exception launching run: {e}\n")
        elapsed = time.time() - t0
        if not ok:
            any_failed = True
        status = "ok" if ok else "FAIL"
        print(
            f"[batch] run='{name}' status={status} rc={rc} "
            f"elapsed={elapsed:.1f}s log={console_log}"
        )
        # Render profiler HTML (timeline + nvitop) for this run now that it has
        # finished. Never aborts the batch on plotting errors.
        profile_html = plot_run_profiles(run_dir)
        results.append(
            {
                "name": name,
                "status": status,
                "return_code": rc,
                "elapsed_sec": round(elapsed, 1),
                "log_path": str(console_log),
                "config_snapshot": str(run_dir / "config_snapshot.json"),
                "key_params": snapshot["key_params"],
                "profile_html": profile_html,
            }
        )

    # Summary
    summary_json = batch_dir / "summary.json"
    summary_json.write_text(json.dumps(results, indent=2, ensure_ascii=False))
    summary_md = batch_dir / "summary.md"
    lines = [f"# Batch summary — {batch_ts}", ""]
    lines.append("| # | name | status | elapsed(s) | actor | env | rollout | rollout_ol | env_ol | actor_ol | tne | re | mbs | timeline | nvitop | log |")
    lines.append("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for i, r in enumerate(results, 1):
        kp = r["key_params"]
        ph = r.get("profile_html") or {}
        tl = ph.get("timeline_html")
        nv = ph.get("nvitop_html")
        lines.append(
            f"| {i} | {r['name']} | {r['status']} | {r['elapsed_sec']} | "
            f"{kp.get('cluster.component_placement.actor')} | "
            f"{kp.get('cluster.component_placement.env')} | "
            f"{kp.get('cluster.component_placement.rollout')} | "
            f"{kp.get('rollout.enable_offload')} | "
            f"{kp.get('env.train.enable_offload')} | "
            f"{kp.get('actor.enable_offload')} | "
            f"{kp.get('env.train.total_num_envs')} | "
            f"{kp.get('env.train.rollout_epoch')} | "
            f"{kp.get('actor.micro_batch_size')} | "
            f"{('[timeline](' + tl + ')') if tl else '-'} | "
            f"{('[nvitop](' + nv + ')') if nv else '-'} | "
            f"{r['log_path']} |"
        )
    summary_md.write_text("\n".join(lines))

    print(f"\n[batch] summary: {summary_md}")
    print(f"[batch] summary (json): {summary_json}")
    n_ok = sum(1 for r in results if r["status"] == "ok")
    print(f"[batch] done: {n_ok}/{len(results)} ok")
    return 1 if any_failed else 0


if __name__ == "__main__":
    sys.exit(main())
