#!/usr/bin/env bash
#
# Ask Claude - One-shot consultation with the Claude Code CLI.
#
# Interface parity with ask-codex.sh so the embodied_tuner critic can swap
# backends by pointing --ask-codex-path (or the CLI's --critic-backend flag)
# at this script. The prompt is read from stdin when --stdin is passed, or
# from concatenated positional arguments otherwise.
#
# Usage:
#   ask-claude.sh [--claude-model MODEL[:EFFORT]] [--claude-timeout SECONDS]
#                 [--stdin] [question...]
#
# Output:
#   stdout: Claude's response (fed back to the critic)
#   stderr: Status/debug info (model, effort, log paths)
#
# Storage: mirrors ask-codex.sh
#   Project-local: .humanize/skill/<unique-id>/{input,output,metadata}.md
#   Cache: ~/.cache/humanize/<sanitized-path>/skill-<unique-id>/claude-run.{cmd,out,log}
#

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)"

# Shared libraries live in the humanize plugin install (same as ask-codex.sh
# in this directory). Override with HUMANIZE_SCRIPTS_DIR when relocating.
HUMANIZE_SCRIPTS_DIR="${HUMANIZE_SCRIPTS_DIR:-/root/.claude/plugins/cache/PolyArch/humanize/1.17.0/scripts}"
source "$HUMANIZE_SCRIPTS_DIR/portable-timeout.sh"
HOOKS_LIB_DIR="$(cd "$HUMANIZE_SCRIPTS_DIR/../hooks/lib" && pwd)"
source "$HOOKS_LIB_DIR/loop-common.sh"

# Reuse humanize's Codex defaults for effort semantics; ask-claude accepts the
# same MODEL:EFFORT parsing but picks its own default model.
DEFAULT_ASK_CLAUDE_TIMEOUT=3600
DEFAULT_CLAUDE_MODEL="sonnet"
CLAUDE_MODEL="$DEFAULT_CLAUDE_MODEL"
CLAUDE_EFFORT="${DEFAULT_CODEX_EFFORT:-high}"
CLAUDE_TIMEOUT="$DEFAULT_ASK_CLAUDE_TIMEOUT"
USE_STDIN=false

show_help() {
    cat << 'HELP_EOF'
ask-claude - One-shot consultation with the Claude Code CLI

USAGE:
  ask-claude.sh [OPTIONS] <question or task>
  ask-claude.sh [OPTIONS] --stdin < prompt.txt

OPTIONS:
  --claude-model <MODEL[:EFFORT]>
                       Claude model alias/id and optional effort tier
                       (default: sonnet; effort defaults from humanize config)
  --claude-timeout <SECONDS>
                       Timeout for the Claude query in seconds (default: 3600)
  --stdin              Read the question from stdin instead of argv (avoids
                       the 128 KiB per-argv cap for large prompts)
  -h, --help           Show this help message

NOTES:
  - Runs `claude -p --model <M> --effort <E>` non-interactively.
  - Disables all tools so the response is pure text (critic returns JSON).
  - When invoked as root, --dangerously-skip-permissions is not passed
    because the Claude CLI refuses that combination.
HELP_EOF
    exit 0
}

QUESTION_PARTS=()
OPTIONS_DONE=false
while [[ $# -gt 0 ]]; do
    if [[ "$OPTIONS_DONE" == "true" ]]; then
        QUESTION_PARTS+=("$1"); shift; continue
    fi
    case $1 in
        -h|--help) show_help ;;
        --) OPTIONS_DONE=true; shift ;;
        --claude-model)
            if [[ -z "${2:-}" ]]; then
                echo "Error: --claude-model requires a MODEL[:EFFORT] argument" >&2
                exit 1
            fi
            if [[ "$2" == *:* ]]; then
                CLAUDE_MODEL="${2%%:*}"
                CLAUDE_EFFORT="${2#*:}"
            else
                CLAUDE_MODEL="$2"
            fi
            shift 2
            ;;
        --claude-timeout)
            if [[ -z "${2:-}" ]]; then
                echo "Error: --claude-timeout requires a number argument (seconds)" >&2
                exit 1
            fi
            if ! [[ "$2" =~ ^[0-9]+$ ]]; then
                echo "Error: --claude-timeout must be a positive integer (seconds), got: $2" >&2
                exit 1
            fi
            CLAUDE_TIMEOUT="$2"
            shift 2
            ;;
        --stdin) USE_STDIN=true; shift ;;
        -*) echo "Error: Unknown option: $1" >&2; echo "Use --help for usage information" >&2; exit 1 ;;
        *) QUESTION_PARTS+=("$1"); OPTIONS_DONE=true; shift ;;
    esac
done

QUESTION="${QUESTION_PARTS[*]+"${QUESTION_PARTS[*]}"}"

if [[ "$USE_STDIN" == "true" ]]; then
    if [[ -n "$QUESTION" ]]; then
        echo "Error: --stdin cannot be combined with a positional question" >&2
        exit 1
    fi
    QUESTION="$(cat)"
fi

if ! command -v claude &>/dev/null; then
    echo "Error: 'claude' command is not installed or not in PATH" >&2
    exit 1
fi

if [[ -z "$QUESTION" ]]; then
    echo "Error: No question or task provided" >&2
    echo "Usage: ask-claude.sh [OPTIONS] <question or task>" >&2
    exit 1
fi

# Safety checks (same regexes as ask-codex.sh)
if [[ ! "$CLAUDE_MODEL" =~ ^[a-zA-Z0-9._-]+$ ]]; then
    echo "Error: Claude model contains invalid characters: $CLAUDE_MODEL" >&2
    exit 1
fi
if [[ ! "$CLAUDE_EFFORT" =~ ^[a-zA-Z0-9_-]+$ ]]; then
    echo "Error: Claude effort contains invalid characters: $CLAUDE_EFFORT" >&2
    exit 1
fi

PROJECT_ROOT="$(resolve_project_root)" || {
    echo "Error: Cannot determine project root." >&2
    echo "  Set CLAUDE_PROJECT_DIR or run inside a git repository." >&2
    exit 1
}

TIMESTAMP=$(date +%Y-%m-%d_%H-%M-%S)
UNIQUE_ID="${TIMESTAMP}-$$-$(head -c 4 /dev/urandom | od -An -tx1 | tr -d ' \n')"
SKILL_DIR="$PROJECT_ROOT/.humanize/skill/$UNIQUE_ID"
mkdir -p "$SKILL_DIR"

SANITIZED_PROJECT_PATH=$(echo "$PROJECT_ROOT" | sed 's/[^a-zA-Z0-9._-]/-/g' | sed 's/--*/-/g')
CACHE_BASE="${XDG_CACHE_HOME:-$HOME/.cache}"
CACHE_DIR="$CACHE_BASE/humanize/$SANITIZED_PROJECT_PATH/skill-$UNIQUE_ID"
if ! mkdir -p "$CACHE_DIR" 2>/dev/null; then
    CACHE_DIR="$SKILL_DIR/cache"
    mkdir -p "$CACHE_DIR"
    echo "ask-claude: warning: home cache not writable, using $CACHE_DIR" >&2
fi

cat > "$SKILL_DIR/input.md" << EOF
# Ask Claude Input

## Question

$QUESTION

## Configuration

- Model: $CLAUDE_MODEL
- Effort: $CLAUDE_EFFORT
- Timeout: ${CLAUDE_TIMEOUT}s
- Timestamp: $TIMESTAMP
- Tool: claude
EOF

# Build claude arguments. We disable tools (no bash/edit/etc.) so the response
# is pure text — the critic contract is JSON out, no side effects.
CLAUDE_ARGS=("-p" "--model" "$CLAUDE_MODEL" "--effort" "$CLAUDE_EFFORT")
CLAUDE_ARGS+=("--disallowedTools" "*")
CLAUDE_ARGS+=("--add-dir" "$PROJECT_ROOT")
# --dangerously-skip-permissions is rejected under root; only add it otherwise
# and only when the caller opts in via HUMANIZE_CLAUDE_BYPASS_PERMISSIONS=1.
if [[ "$(id -u)" != "0" ]] && \
   { [[ "${HUMANIZE_CLAUDE_BYPASS_PERMISSIONS:-}" == "true" ]] || \
     [[ "${HUMANIZE_CLAUDE_BYPASS_PERMISSIONS:-}" == "1" ]]; }; then
    CLAUDE_ARGS+=("--dangerously-skip-permissions")
fi

CLAUDE_CMD_FILE="$CACHE_DIR/claude-run.cmd"
CLAUDE_STDOUT_FILE="$CACHE_DIR/claude-run.out"
CLAUDE_STDERR_FILE="$CACHE_DIR/claude-run.log"

{
    echo "# ask-claude invocation debug info"
    echo "# Timestamp: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
    echo "# Working directory: $PROJECT_ROOT"
    echo "# Timeout: $CLAUDE_TIMEOUT seconds"
    echo ""
    echo "claude ${CLAUDE_ARGS[*]} < <prompt>"
    echo ""
    echo "# Prompt content:"
    echo "$QUESTION"
} > "$CLAUDE_CMD_FILE"

echo "ask-claude: model=$CLAUDE_MODEL effort=$CLAUDE_EFFORT timeout=${CLAUDE_TIMEOUT}s" >&2
echo "ask-claude: cache=$CACHE_DIR" >&2
echo "ask-claude: running claude -p..." >&2

epoch_to_iso() {
    local epoch="$1"
    date -u -d "@$epoch" +%Y-%m-%dT%H:%M:%SZ 2>/dev/null ||
    date -u -r "$epoch" +%Y-%m-%dT%H:%M:%SZ 2>/dev/null ||
    echo "unknown"
}

START_TIME=$(date +%s)
CLAUDE_EXIT_CODE=0
printf '%s' "$QUESTION" | ( cd "$PROJECT_ROOT" && \
    run_with_timeout "$CLAUDE_TIMEOUT" claude "${CLAUDE_ARGS[@]}" ) \
    > "$CLAUDE_STDOUT_FILE" 2> "$CLAUDE_STDERR_FILE" || CLAUDE_EXIT_CODE=$?
END_TIME=$(date +%s)
DURATION=$((END_TIME - START_TIME))

echo "ask-claude: exit_code=$CLAUDE_EXIT_CODE duration=${DURATION}s" >&2

if [[ $CLAUDE_EXIT_CODE -eq 124 ]]; then
    echo "Error: Claude timed out after ${CLAUDE_TIMEOUT} seconds" >&2
    echo "Debug logs: $CACHE_DIR" >&2
    cat > "$SKILL_DIR/metadata.md" << EOF
---
tool: claude
model: $CLAUDE_MODEL
effort: $CLAUDE_EFFORT
timeout: $CLAUDE_TIMEOUT
exit_code: 124
duration: ${DURATION}s
status: timeout
started_at: $(epoch_to_iso "$START_TIME")
---
EOF
    exit 124
fi

if [[ $CLAUDE_EXIT_CODE -ne 0 ]]; then
    echo "Error: Claude exited with code $CLAUDE_EXIT_CODE" >&2
    if [[ -s "$CLAUDE_STDERR_FILE" ]]; then
        echo "" >&2
        echo "Claude stderr (last 20 lines):" >&2
        tail -20 "$CLAUDE_STDERR_FILE" >&2
    fi
    echo "Debug logs: $CACHE_DIR" >&2
    cat > "$SKILL_DIR/metadata.md" << EOF
---
tool: claude
model: $CLAUDE_MODEL
effort: $CLAUDE_EFFORT
timeout: $CLAUDE_TIMEOUT
exit_code: $CLAUDE_EXIT_CODE
duration: ${DURATION}s
status: error
started_at: $(epoch_to_iso "$START_TIME")
---
EOF
    exit "$CLAUDE_EXIT_CODE"
fi

if [[ ! -s "$CLAUDE_STDOUT_FILE" ]]; then
    echo "Error: Claude returned empty response" >&2
    if [[ -s "$CLAUDE_STDERR_FILE" ]]; then
        echo "" >&2
        echo "Claude stderr (last 20 lines):" >&2
        tail -20 "$CLAUDE_STDERR_FILE" >&2
    fi
    echo "Debug logs: $CACHE_DIR" >&2
    cat > "$SKILL_DIR/metadata.md" << EOF
---
tool: claude
model: $CLAUDE_MODEL
effort: $CLAUDE_EFFORT
timeout: $CLAUDE_TIMEOUT
exit_code: 0
duration: ${DURATION}s
status: empty_response
started_at: $(epoch_to_iso "$START_TIME")
---
EOF
    exit 1
fi

cp "$CLAUDE_STDOUT_FILE" "$SKILL_DIR/output.md"
cat > "$SKILL_DIR/metadata.md" << EOF
---
tool: claude
model: $CLAUDE_MODEL
effort: $CLAUDE_EFFORT
timeout: $CLAUDE_TIMEOUT
exit_code: 0
duration: ${DURATION}s
status: success
started_at: $(epoch_to_iso "$START_TIME")
---
EOF

echo "ask-claude: response saved to $SKILL_DIR/output.md" >&2
cat "$CLAUDE_STDOUT_FILE"
