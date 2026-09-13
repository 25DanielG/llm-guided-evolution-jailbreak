#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

: "${JB_BEHAVIORS_PATH:?Set JB_BEHAVIORS_PATH to an approved, checksum-verified benchmark CSV}"
if [[ ! -f "$JB_BEHAVIORS_PATH" ]]; then
    echo "Behavior CSV does not exist: $JB_BEHAVIORS_PATH" >&2
    exit 2
fi
if [[ "$(basename "$JB_BEHAVIORS_PATH")" == "curated.csv" ]]; then
    echo "curated.csv is a benign test fixture; select an approved benchmark CSV." >&2
    exit 2
fi

export JB_VLLM_API_KEY="${JB_VLLM_API_KEY:-$(python3 -c 'import secrets; print(secrets.token_urlsafe(32))')}"
export JB_JUDGE_MODE="${JB_JUDGE_MODE:-llm}"
export JB_TARGET_SERVED_NAME="${JB_TARGET_SERVED_NAME:-target}"
export JB_JUDGE_SERVED_NAME="${JB_JUDGE_SERVED_NAME:-judge}"
export JB_N_BEHAVIORS_PER_EVAL="${JB_N_BEHAVIORS_PER_EVAL:-4}"
export JB_MAX_CONCURRENCY="${JB_MAX_CONCURRENCY:-2}"
export JB_CACHE_DIR="${JB_CACHE_DIR:-${HOME}/scratch/jb_cache}"
export JB_REQUEST_TIMEOUT="${JB_REQUEST_TIMEOUT:-120}"
export JB_SERVER_READY_TIMEOUT="${JB_SERVER_READY_TIMEOUT:-1200}"

mkdir -p run_job_outputs/{server,evaluation,evolution,islands,jailbreak-vllm}
mkdir -p sota/Jailbreak/results sota/Jailbreak/models/llmge_models
rm -f hostname.log sota/Jailbreak/target_host.log sota/Jailbreak/judge_host.log

VLLM_JOB=""
MUTATOR_JOB=""
RUN_JOB=""
cleanup_on_error() {
    status=$?
    if [[ "$status" -ne 0 && -z "$RUN_JOB" ]]; then
        [[ -n "$VLLM_JOB" ]] && scancel "$VLLM_JOB" 2>/dev/null || true
        [[ -n "$MUTATOR_JOB" ]] && scancel "$MUTATOR_JOB" 2>/dev/null || true
    fi
    exit "$status"
}
trap cleanup_on_error EXIT

VLLM_JOB="$(sbatch --parsable --export=ALL jailbreak_vllm.sbatch)"
MUTATOR_JOB="$(sbatch --parsable --export=ALL,SUBMIT_ISLAND_CONTROLLER=0 server.sh)"
echo "Submitted target/judge service: $VLLM_JOB"
echo "Submitted mutation service: $MUTATOR_JOB"

wait_for_file() {
    path=$1
    label=$2
    for _ in $(seq 1 120); do
        [[ -s "$path" ]] && return 0
        sleep 10
    done
    echo "Timed out waiting for $label host file: $path" >&2
    return 1
}

wait_for_url() {
    url=$1
    label=$2
    auth=${3:-0}
    for _ in $(seq 1 120); do
        if [[ "$auth" == "1" ]]; then
            curl -fsS --connect-timeout 5 --max-time 10 \
                -H "Authorization: Bearer $JB_VLLM_API_KEY" "$url" >/dev/null && return 0
        else
            curl -fsS --connect-timeout 5 --max-time 10 "$url" >/dev/null && return 0
        fi
        sleep 10
    done
    echo "Timed out waiting for $label endpoint: $url" >&2
    return 1
}

wait_for_file hostname.log mutation
wait_for_file sota/Jailbreak/target_host.log target
wait_for_file sota/Jailbreak/judge_host.log judge

wait_for_url "http://$(cat hostname.log):8137/" mutation
wait_for_url "http://$(cat sota/Jailbreak/target_host.log)/v1/models" target 1
wait_for_url "http://$(cat sota/Jailbreak/judge_host.log)/v1/models" judge 1

uv run python sota/Jailbreak/preflight.py --mode pace
uv run python sota/Jailbreak/eval.py --model prompt --variant_dir sota/Jailbreak

RUN_JOB="$(sbatch --parsable --export=ALL run.sh)"
CLEANUP_JOB="$(sbatch --parsable --job-name=Jailbreak_cleanup \
    --dependency="afterany:$RUN_JOB" \
    --wrap="scancel $VLLM_JOB $MUTATOR_JOB")"

mkdir -p .run_state
STATE_FILE=".run_state/jailbreak-${RUN_JOB}.env"
{
    printf 'RUN_JOB=%q\n' "$RUN_JOB"
    printf 'VLLM_JOB=%q\n' "$VLLM_JOB"
    printf 'MUTATOR_JOB=%q\n' "$MUTATOR_JOB"
    printf 'CLEANUP_JOB=%q\n' "$CLEANUP_JOB"
} > "$STATE_FILE"

trap - EXIT
echo "Evolution job: $RUN_JOB"
echo "Cleanup job: $CLEANUP_JOB"
echo "Recovery state: $STATE_FILE"
