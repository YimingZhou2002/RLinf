#!/usr/bin/env bash
#
# Ask Codex - One-shot consultation with Codex
#
# Sends a question or task to codex exec and returns the response.
# This is an active, one-shot skill (unlike the passive RLCR loop).
#
# Usage:
#   ask-codex.sh [--codex-model MODEL:EFFORT] [--codex-timeout SECONDS] [--codex-session KEY] [question...]
#
# Output:
#   stdout: Codex's response (for Claude to read)
#   stderr: Status/debug info (model, effort, log paths)
#
# Storage:
#   Project-local: .humanize/skill/<unique-id>/{input,output,metadata}.md
#   Cache: ~/.cache/humanize/<sanitized-path>/skill-<unique-id>/codex-run.{cmd,out,log}
#

set -euo pipefail

# ========================================
# Source Shared Libraries
# ========================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)"

# This is a vendored copy of humanize's ask-codex.sh, kept next to the
# embodied_tuner critic so we can add a --stdin flag (avoids Linux's
# 128 KiB per-argv MAX_ARG_STRLEN cap when the prompt is large).
# The shared libraries still live in the humanize plugin install; point
# at them via HUMANIZE_SCRIPTS_DIR (default: the current plugin cache).
HUMANIZE_SCRIPTS_DIR="${HUMANIZE_SCRIPTS_DIR:-/root/.claude/plugins/cache/PolyArch/humanize/1.17.0/scripts}"

# Source portable timeout wrapper
source "$HUMANIZE_SCRIPTS_DIR/portable-timeout.sh"

# Source shared loop library for DEFAULT_CODEX_MODEL and DEFAULT_CODEX_EFFORT
HOOKS_LIB_DIR="$(cd "$HUMANIZE_SCRIPTS_DIR/../hooks/lib" && pwd)"
source "$HOOKS_LIB_DIR/loop-common.sh"

# ========================================
# Default Configuration
# ========================================

DEFAULT_ASK_CODEX_TIMEOUT=3600

CODEX_MODEL="$DEFAULT_CODEX_MODEL"
CODEX_EFFORT="$DEFAULT_CODEX_EFFORT"
CODEX_TIMEOUT="$DEFAULT_ASK_CODEX_TIMEOUT"
USE_STDIN=false
# When set, all invocations sharing this key run inside ONE Codex conversation:
# the first records the session id printed by `codex exec`, later ones resume
# it. Empty = current behaviour (a fresh one-shot session per call).
CODEX_SESSION_KEY=""

# ========================================
# Help
# ========================================

show_help() {
    cat << 'HELP_EOF'
ask-codex - One-shot consultation with Codex

USAGE:
  /humanize:ask-codex [OPTIONS] <question or task>

OPTIONS:
  --codex-model <MODEL:EFFORT>
                       Codex model and reasoning effort (default from config, fallback gpt-5.5:high)
  --codex-timeout <SECONDS>
                       Timeout for the Codex query in seconds (default: 3600)
  --stdin              Read the question from stdin instead of argv (avoids
                       the 128 KiB per-argv cap for large prompts)
  --codex-session <KEY>
                       Bind this call to a persistent Codex conversation named
                       KEY. The first call for a KEY starts a session and records
                       its id; subsequent calls resume it (`codex exec resume`),
                       so a whole tuner campaign shares one context. Omit for the
                       default one-shot-per-call behaviour.
  -h, --help           Show this help message

DESCRIPTION:
  Sends a one-shot question or task to Codex and returns the response.
  Unlike the RLCR loop, this is a single consultation without iteration.

  The response is saved to .humanize/skill/<unique-id>/output.md for reference.

EXAMPLES:
  /humanize:ask-codex How should I structure the authentication module?
  /humanize:ask-codex --codex-model gpt-5.5:high What are the performance bottlenecks?
  /humanize:ask-codex --codex-timeout 300 Review the error handling in src/api/

ENVIRONMENT:
  HUMANIZE_CODEX_BYPASS_SANDBOX
    Set to "true" or "1" to bypass Codex sandbox protections.
    WARNING: This is dangerous. See README for details.
HELP_EOF
    exit 0
}

# ========================================
# Parse Arguments
# ========================================

QUESTION_PARTS=()
OPTIONS_DONE=false

while [[ $# -gt 0 ]]; do
    if [[ "$OPTIONS_DONE" == "true" ]]; then
        # After first positional token or --, all remaining args are question text
        QUESTION_PARTS+=("$1")
        shift
        continue
    fi
    case $1 in
        -h|--help)
            show_help
            ;;
        --)
            # Explicit end-of-options marker
            OPTIONS_DONE=true
            shift
            ;;
        --codex-model)
            if [[ -z "${2:-}" ]]; then
                echo "Error: --codex-model requires a MODEL:EFFORT argument" >&2
                exit 1
            fi
            # Parse MODEL:EFFORT format (same pattern as setup-rlcr-loop.sh)
            if [[ "$2" == *:* ]]; then
                CODEX_MODEL="${2%%:*}"
                CODEX_EFFORT="${2#*:}"
            else
                CODEX_MODEL="$2"
                CODEX_EFFORT="$DEFAULT_CODEX_EFFORT"
            fi
            shift 2
            ;;
        --codex-timeout)
            if [[ -z "${2:-}" ]]; then
                echo "Error: --codex-timeout requires a number argument (seconds)" >&2
                exit 1
            fi
            if ! [[ "$2" =~ ^[0-9]+$ ]]; then
                echo "Error: --codex-timeout must be a positive integer (seconds), got: $2" >&2
                exit 1
            fi
            CODEX_TIMEOUT="$2"
            shift 2
            ;;
        --stdin)
            USE_STDIN=true
            shift
            ;;
        --codex-session)
            if [[ -z "${2:-}" ]]; then
                echo "Error: --codex-session requires a KEY argument" >&2
                exit 1
            fi
            CODEX_SESSION_KEY="$2"
            shift 2
            ;;
        -*)
            echo "Error: Unknown option: $1" >&2
            echo "Use --help for usage information" >&2
            exit 1
            ;;
        *)
            # First positional token: stop parsing options, rest is question
            QUESTION_PARTS+=("$1")
            OPTIONS_DONE=true
            shift
            ;;
    esac
done

# Join question parts into a single string (use ${arr[*]+...} to avoid set -u crash on bash 3.2)
QUESTION="${QUESTION_PARTS[*]+"${QUESTION_PARTS[*]}"}"

# When --stdin is set, read the entire prompt from stdin. This bypasses the
# Linux 128 KiB per-argv MAX_ARG_STRLEN cap (execve E2BIG) that fires when
# the caller tries to pass a large prompt as a single positional argument.
if [[ "$USE_STDIN" == "true" ]]; then
    if [[ -n "$QUESTION" ]]; then
        echo "Error: --stdin cannot be combined with a positional question" >&2
        exit 1
    fi
    QUESTION="$(cat)"
fi

# ========================================
# Validate Prerequisites
# ========================================

# Check codex is available
if ! command -v codex &>/dev/null; then
    echo "Error: 'codex' command is not installed or not in PATH" >&2
    echo "" >&2
    echo "Please install Codex CLI: https://github.com/openai/codex" >&2
    echo "Then retry: /humanize:ask-codex <your question>" >&2
    exit 1
fi

# Check question is not empty
if [[ -z "$QUESTION" ]]; then
    echo "Error: No question or task provided" >&2
    echo "" >&2
    echo "Usage: /humanize:ask-codex [OPTIONS] <question or task>" >&2
    echo "" >&2
    echo "For help: /humanize:ask-codex --help" >&2
    exit 1
fi

# Validate codex model for safety (alphanumeric, hyphen, underscore, dot)
if [[ ! "$CODEX_MODEL" =~ ^[a-zA-Z0-9._-]+$ ]]; then
    echo "Error: Codex model contains invalid characters" >&2
    echo "  Model: $CODEX_MODEL" >&2
    echo "  Only alphanumeric, hyphen, underscore, dot allowed" >&2
    exit 1
fi

# Validate codex effort for safety (alphanumeric, hyphen, underscore)
if [[ ! "$CODEX_EFFORT" =~ ^[a-zA-Z0-9_-]+$ ]]; then
    echo "Error: Codex effort contains invalid characters" >&2
    echo "  Effort: $CODEX_EFFORT" >&2
    echo "  Only alphanumeric, hyphen, underscore allowed" >&2
    exit 1
fi

# Validate codex session key: it becomes a filename, so reject path separators
# and dot-prefixes (same guard as cancel-rlcr-session.sh).
if [[ -n "$CODEX_SESSION_KEY" ]]; then
    if [[ "$CODEX_SESSION_KEY" == *"/"* || "$CODEX_SESSION_KEY" == *"\\"* ]]; then
        echo "Error: --codex-session must not contain a path separator: $CODEX_SESSION_KEY" >&2
        exit 1
    fi
    if [[ "$CODEX_SESSION_KEY" == .* ]]; then
        echo "Error: --codex-session must not start with a dot: $CODEX_SESSION_KEY" >&2
        exit 1
    fi
    if [[ ! "$CODEX_SESSION_KEY" =~ ^[A-Za-z0-9._-]+$ ]]; then
        echo "Error: --codex-session allows only alphanumerics, dot, underscore, dash: $CODEX_SESSION_KEY" >&2
        exit 1
    fi
fi

# ========================================
# Detect Project Root
# ========================================

PROJECT_ROOT="$(resolve_project_root)" || {
    echo "Error: Cannot determine project root." >&2
    echo "  Set CLAUDE_PROJECT_DIR or run inside a git repository." >&2
    exit 1
}

# ========================================
# Create Storage Directories
# ========================================

TIMESTAMP=$(date +%Y-%m-%d_%H-%M-%S)
UNIQUE_ID="${TIMESTAMP}-$$-$(head -c 4 /dev/urandom | od -An -tx1 | tr -d ' \n')"

# Project-local storage: .humanize/skill/<unique-id>/
SKILL_DIR="$PROJECT_ROOT/.humanize/skill/$UNIQUE_ID"
mkdir -p "$SKILL_DIR"

# Cache storage: ~/.cache/humanize/<sanitized-path>/skill-<unique-id>/
# Falls back to project-local .humanize/cache/ if home cache is not writable
SANITIZED_PROJECT_PATH=$(echo "$PROJECT_ROOT" | sed 's/[^a-zA-Z0-9._-]/-/g' | sed 's/--*/-/g')
CACHE_BASE="${XDG_CACHE_HOME:-$HOME/.cache}"
CACHE_DIR="$CACHE_BASE/humanize/$SANITIZED_PROJECT_PATH/skill-$UNIQUE_ID"
if ! mkdir -p "$CACHE_DIR" 2>/dev/null; then
    CACHE_DIR="$SKILL_DIR/cache"
    mkdir -p "$CACHE_DIR"
    echo "ask-codex: warning: home cache not writable, using $CACHE_DIR" >&2
fi

# ========================================
# Resolve Codex Session (shared-conversation mode)
# ========================================

# When --codex-session KEY is set, persist the Codex conversation id under a
# stable per-(project, key) path so every call with the same KEY resumes the
# same conversation instead of cold-starting. The first call has no recorded
# id and runs a fresh `codex exec`; it then captures the "session id: <uuid>"
# line Codex prints and writes it here for the next call to resume.
CODEX_SESSION_ID=""
CODEX_SESSION_STATE_FILE=""
CODEX_SESSION_RESUME=false
if [[ -n "$CODEX_SESSION_KEY" ]]; then
    SESSION_STATE_DIR="$CACHE_BASE/humanize/$SANITIZED_PROJECT_PATH/codex-session"
    if ! mkdir -p "$SESSION_STATE_DIR" 2>/dev/null; then
        SESSION_STATE_DIR="$SKILL_DIR/codex-session"
        mkdir -p "$SESSION_STATE_DIR"
    fi
    CODEX_SESSION_STATE_FILE="$SESSION_STATE_DIR/$CODEX_SESSION_KEY.sid"
    if [[ -s "$CODEX_SESSION_STATE_FILE" ]]; then
        CODEX_SESSION_ID="$(head -n1 "$CODEX_SESSION_STATE_FILE" | tr -d '[:space:]')"
        if [[ -n "$CODEX_SESSION_ID" ]]; then
            CODEX_SESSION_RESUME=true
        fi
    fi
fi

# ========================================
# Save Input
# ========================================

cat > "$SKILL_DIR/input.md" << EOF
# Ask Codex Input

## Question

$QUESTION

## Configuration

- Model: $CODEX_MODEL
- Effort: $CODEX_EFFORT
- Timeout: ${CODEX_TIMEOUT}s
- Timestamp: $TIMESTAMP
- Tool: codex
EOF

# ========================================
# Build Codex Command
# ========================================

# Probe supported hook feature names before disabling them. Codex has used
# different names across releases, and older CLIs reject unknown feature names.
CODEX_DISABLE_HOOKS_ARGS=()
_CODEX_DISABLE_HOOKS_CACHE="$SKILL_DIR/.codex-disable-hooks-features"
if [[ -f "$_CODEX_DISABLE_HOOKS_CACHE" ]]; then
    while IFS= read -r feature_name; do
        case "$feature_name" in
            hooks|plugin_hooks|codex_hooks)
                CODEX_DISABLE_HOOKS_ARGS+=("--disable" "$feature_name")
                ;;
        esac
    done < "$_CODEX_DISABLE_HOOKS_CACHE"
else
    CODEX_HELP_OUTPUT="$(codex --help </dev/null 2>&1 || true)"
    if grep -q -- '--disable' <<< "$CODEX_HELP_OUTPUT"; then
        _CODEX_DISABLE_HOOK_FEATURES=()
        for feature_name in hooks plugin_hooks codex_hooks; do
            if codex --disable "$feature_name" --help </dev/null >/dev/null 2>&1; then
                CODEX_DISABLE_HOOKS_ARGS+=("--disable" "$feature_name")
                _CODEX_DISABLE_HOOK_FEATURES+=("$feature_name")
            fi
        done
        printf '%s\n' ${_CODEX_DISABLE_HOOK_FEATURES[@]+"${_CODEX_DISABLE_HOOK_FEATURES[@]}"} > "$_CODEX_DISABLE_HOOKS_CACHE" 2>/dev/null || true
    else
        : > "$_CODEX_DISABLE_HOOKS_CACHE" 2>/dev/null || true
    fi
fi

# Build codex exec arguments (same pattern as loop-codex-stop-hook.sh)
# Use ${arr[@]+"${arr[@]}"} to safely expand possibly-empty arrays under set -u (bash 3.2 compat)
CODEX_EXEC_ARGS=(${CODEX_DISABLE_HOOKS_ARGS[@]+"${CODEX_DISABLE_HOOKS_ARGS[@]}"} "-m" "$CODEX_MODEL")
if [[ -n "$CODEX_EFFORT" ]]; then
    CODEX_EXEC_ARGS+=("-c" "model_reasoning_effort=${CODEX_EFFORT}")
fi

# Determine automation flag based on environment variable
CODEX_AUTO_FLAG="--full-auto"
if [[ "${HUMANIZE_CODEX_BYPASS_SANDBOX:-}" == "true" ]] || [[ "${HUMANIZE_CODEX_BYPASS_SANDBOX:-}" == "1" ]]; then
    CODEX_AUTO_FLAG="--dangerously-bypass-approvals-and-sandbox"
fi

CODEX_EXEC_ARGS+=("$CODEX_AUTO_FLAG" "-C" "$PROJECT_ROOT")

# Resume-mode argv. `codex exec resume` does NOT accept --full-auto or -C, so
# we drop them and cd into the project root at run time instead. Sandbox parity:
# bypass when requested, otherwise Codex's default non-interactive policy (a
# read-only consult needs no write access). --skip-git-repo-check keeps resume
# working when the campaign cwd is not the git root.
CODEX_RESUME_ARGS=(${CODEX_DISABLE_HOOKS_ARGS[@]+"${CODEX_DISABLE_HOOKS_ARGS[@]}"} "-m" "$CODEX_MODEL")
if [[ -n "$CODEX_EFFORT" ]]; then
    CODEX_RESUME_ARGS+=("-c" "model_reasoning_effort=${CODEX_EFFORT}")
fi
if [[ "${HUMANIZE_CODEX_BYPASS_SANDBOX:-}" == "true" ]] || [[ "${HUMANIZE_CODEX_BYPASS_SANDBOX:-}" == "1" ]]; then
    CODEX_RESUME_ARGS+=("--dangerously-bypass-approvals-and-sandbox")
fi
CODEX_RESUME_ARGS+=("--skip-git-repo-check")

# ========================================
# Save Debug Command
# ========================================

CODEX_CMD_FILE="$CACHE_DIR/codex-run.cmd"
CODEX_STDOUT_FILE="$CACHE_DIR/codex-run.out"
CODEX_STDERR_FILE="$CACHE_DIR/codex-run.log"

{
    echo "# Codex ask-codex invocation debug info"
    echo "# Timestamp: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
    echo "# Working directory: $PROJECT_ROOT"
    echo "# Timeout: $CODEX_TIMEOUT seconds"
    echo ""
    echo "codex exec ${CODEX_EXEC_ARGS[*]} \"<prompt>\""
    echo ""
    echo "# Prompt content:"
    echo "$QUESTION"
} > "$CODEX_CMD_FILE"

# ========================================
# Run Codex
# ========================================

echo "ask-codex: model=$CODEX_MODEL effort=$CODEX_EFFORT timeout=${CODEX_TIMEOUT}s" >&2
echo "ask-codex: cache=$CACHE_DIR" >&2
echo "ask-codex: running codex exec..." >&2

# Portable epoch-to-ISO8601 formatter (GNU date -d vs BSD date -r)
epoch_to_iso() {
    local epoch="$1"
    date -u -d "@$epoch" +%Y-%m-%dT%H:%M:%SZ 2>/dev/null ||
    date -u -r "$epoch" +%Y-%m-%dT%H:%M:%SZ 2>/dev/null ||
    echo "unknown"
}

START_TIME=$(date +%s)

CODEX_EXIT_CODE=0
if [[ "$CODEX_SESSION_RESUME" == "true" ]]; then
    echo "ask-codex: resuming codex session $CODEX_SESSION_ID (key=$CODEX_SESSION_KEY)" >&2
    printf '%s' "$QUESTION" | ( cd "$PROJECT_ROOT" && run_with_timeout "$CODEX_TIMEOUT" \
        codex exec resume "${CODEX_RESUME_ARGS[@]}" "$CODEX_SESSION_ID" - ) \
        > "$CODEX_STDOUT_FILE" 2> "$CODEX_STDERR_FILE" || CODEX_EXIT_CODE=$?
else
    printf '%s' "$QUESTION" | run_with_timeout "$CODEX_TIMEOUT" codex exec "${CODEX_EXEC_ARGS[@]}" - \
        > "$CODEX_STDOUT_FILE" 2> "$CODEX_STDERR_FILE" || CODEX_EXIT_CODE=$?
fi

END_TIME=$(date +%s)
DURATION=$((END_TIME - START_TIME))

echo "ask-codex: exit_code=$CODEX_EXIT_CODE duration=${DURATION}s" >&2

# ========================================
# Handle Results
# ========================================

# Check for timeout
if [[ $CODEX_EXIT_CODE -eq 124 ]]; then
    echo "Error: Codex timed out after ${CODEX_TIMEOUT} seconds" >&2
    echo "" >&2
    echo "Try increasing the timeout:" >&2
    echo "  /humanize:ask-codex --codex-timeout $((CODEX_TIMEOUT * 2)) <your question>" >&2
    echo "" >&2
    echo "Debug logs: $CACHE_DIR" >&2

    # Save metadata even on timeout
    cat > "$SKILL_DIR/metadata.md" << EOF
---
tool: codex
model: $CODEX_MODEL
effort: $CODEX_EFFORT
timeout: $CODEX_TIMEOUT
exit_code: 124
duration: ${DURATION}s
status: timeout
started_at: $(epoch_to_iso "$START_TIME")
---
EOF
    exit 124
fi

# Check for non-zero exit
if [[ $CODEX_EXIT_CODE -ne 0 ]]; then
    echo "Error: Codex exited with code $CODEX_EXIT_CODE" >&2
    if [[ -s "$CODEX_STDERR_FILE" ]]; then
        echo "" >&2
        echo "Codex stderr (last 20 lines):" >&2
        tail -20 "$CODEX_STDERR_FILE" >&2
    fi
    echo "" >&2
    echo "Debug logs: $CACHE_DIR" >&2

    # Save metadata
    cat > "$SKILL_DIR/metadata.md" << EOF
---
tool: codex
model: $CODEX_MODEL
effort: $CODEX_EFFORT
timeout: $CODEX_TIMEOUT
exit_code: $CODEX_EXIT_CODE
duration: ${DURATION}s
status: error
started_at: $(epoch_to_iso "$START_TIME")
---
EOF
    exit "$CODEX_EXIT_CODE"
fi

# Check for empty stdout
if [[ ! -s "$CODEX_STDOUT_FILE" ]]; then
    echo "Error: Codex returned empty response" >&2
    if [[ -s "$CODEX_STDERR_FILE" ]]; then
        echo "" >&2
        echo "Codex stderr (last 20 lines):" >&2
        tail -20 "$CODEX_STDERR_FILE" >&2
    fi
    echo "" >&2
    echo "Debug logs: $CACHE_DIR" >&2

    cat > "$SKILL_DIR/metadata.md" << EOF
---
tool: codex
model: $CODEX_MODEL
effort: $CODEX_EFFORT
timeout: $CODEX_TIMEOUT
exit_code: 0
duration: ${DURATION}s
status: empty_response
started_at: $(epoch_to_iso "$START_TIME")
---
EOF
    exit 1
fi

# ========================================
# Record Session Id (first round of a shared session)
# ========================================

# On the first successful call for a --codex-session KEY, capture the session
# id Codex printed ("session id: <uuid>") so the next call resumes this exact
# conversation. Skipped when resuming (the id is already known).
if [[ -n "$CODEX_SESSION_KEY" && "$CODEX_SESSION_RESUME" != "true" && -n "$CODEX_SESSION_STATE_FILE" ]]; then
    CAPTURED_SID="$(grep -oiE 'session id:[[:space:]]*[0-9a-fA-F-]{36}' "$CODEX_STDERR_FILE" 2>/dev/null \
        | head -n1 | grep -oiE '[0-9a-fA-F-]{36}' | head -n1)"
    if [[ -n "$CAPTURED_SID" ]]; then
        if printf '%s\n' "$CAPTURED_SID" > "$CODEX_SESSION_STATE_FILE" 2>/dev/null; then
            echo "ask-codex: recorded codex session $CAPTURED_SID (key=$CODEX_SESSION_KEY)" >&2
        else
            echo "ask-codex: warning: could not persist session id to $CODEX_SESSION_STATE_FILE" >&2
        fi
    else
        echo "ask-codex: warning: no session id found in codex output; next call for key=$CODEX_SESSION_KEY starts fresh" >&2
    fi
fi

# ========================================
# Save Output and Metadata
# ========================================

# Save Codex response to project-local storage
cp "$CODEX_STDOUT_FILE" "$SKILL_DIR/output.md"

# Save metadata
cat > "$SKILL_DIR/metadata.md" << EOF
---
tool: codex
model: $CODEX_MODEL
effort: $CODEX_EFFORT
timeout: $CODEX_TIMEOUT
exit_code: 0
duration: ${DURATION}s
status: success
started_at: $(epoch_to_iso "$START_TIME")
session_key: ${CODEX_SESSION_KEY:-none}
session_resumed: $CODEX_SESSION_RESUME
---
EOF

echo "ask-codex: response saved to $SKILL_DIR/output.md" >&2

# ========================================
# Output Response
# ========================================

# Output Codex's response to stdout (clean output for Claude to read)
cat "$CODEX_STDOUT_FILE"
