# batch_run

Serially launch a list of embodiment training runs, each an invocation of
`examples/embodiment/run_embodiment.sh` with per-run Hydra overrides. Per-run
stdout, a config snapshot, the RL-logger outputs, and (if profiler data is
present) timeline/nvitop HTML all land together under `logs/batch_<ts>/<name>/`,
and a combined `summary.md` / `summary.json` is written to the batch dir.

## Usage

```bash
# default batch (toolkits/batch_run/batch_runs.yaml)
bash toolkits/batch_run/run_batch.sh

# a specific batch config
bash toolkits/batch_run/run_batch.sh toolkits/batch_run/batch_runs_ofat.yaml
```

## Files

- `batch_run.py` — the driver.
- `run_batch.sh` — thin wrapper that sets `PYTHONPATH` and calls the driver.
- `batch_runs.yaml` — default batch; documents the recognized per-run fields.
- `batch_runs_ofat.yaml` — one-factor-at-a-time sweep.
- `batch_runs_verify_baseline.yaml`, `batch_runs_verify_offload.yaml` — small
  verification batches.

## How it plugs in

The driver passes overrides to `run_embodiment.sh` via the `EXTRA_OVERRIDES` env
var and the per-run log dir via `RUN_LOG_DIR` (both consumed by the two lines
added to `examples/embodiment/run_embodiment.sh`).

Profiler plotting is optional: `PROFILER_DIR` in `batch_run.py` points at the
external `plot_timeline.py` / `plot_nvitop.py` scripts; if they or the profile
data are absent, plotting is skipped without aborting the batch.
