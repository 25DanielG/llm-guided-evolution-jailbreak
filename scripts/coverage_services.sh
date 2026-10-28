#!/bin/bash
# Process-group lifecycle and readiness retries for coverage model services.
# Source this file from the launcher; keep stdout reserved for the service PID.
start_service() {
    local label="$1" logfile="$2"; shift 2
    local cache_dir="$KERNEL_CACHE_BASE/${label// /_}"
    mkdir -p "$cache_dir"
    setsid env VLLM_CACHE_ROOT="$cache_dir/vllm" \
        TORCHINDUCTOR_CACHE_DIR="$cache_dir/torchinductor" \
        TRITON_CACHE_DIR="$cache_dir/triton" \
        FLASHINFER_WORKSPACE_BASE="$cache_dir/flashinfer" \
        "$@" > "$logfile" 2>&1 &
    echo "$!"
}

wait_or_die() {
    local url="$1" pid="$2" label="$3" logfile="$4"; shift 4
    local attempt deadline ready
    for attempt in 1 2 3; do
        deadline=$((SECONDS + ${COVERAGE_STARTUP_TIMEOUT:-1200}))
        ready=false
        while kill -0 "$pid" 2>/dev/null; do
            # Orphaned children can remain zombies briefly after exiting.
            if [[ "$(ps -o stat= -p "$pid" 2>/dev/null)" == Z* ]]; then break; fi
            if curl --connect-timeout 5 --max-time 10 -sf "$url" >/dev/null 2>&1; then
                ready=true
                break
            fi
            if (( SECONDS >= deadline )); then break; fi
            sleep "${COVERAGE_READINESS_POLL:-5}"
        done
        if "$ready"; then
            echo "$pid"
            return 0
        fi
        echo "$label failed readiness on attempt $attempt. Recent log:" >&2
        tail -n 35 "$logfile" >&2
        kill -TERM -- "-$pid" 2>/dev/null || true
        sleep "${COVERAGE_RETRY_DELAY:-10}"
        # Reap all workers in the failed process group before reusing its GPUs.
        kill -KILL -- "-$pid" 2>/dev/null || true
        if (( attempt == 3 )); then
            echo "ERROR: $label failed readiness after 3 attempts" >&2
            return 1
        fi
        cp "$logfile" "$logfile.attempt$attempt"
        pid=$(start_service "$label" "$logfile" "$@")
    done
}
