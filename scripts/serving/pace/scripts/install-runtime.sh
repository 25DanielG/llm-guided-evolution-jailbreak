#!/usr/bin/env bash
set -euo pipefail

: "${PACE_INFERENCE_STATE:?Set an external state directory}"
repo=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)
checkout=$(cd -- "$repo/../../.." && pwd -P)
state=$(realpath -m -- "$PACE_INFERENCE_STATE")
case "$state/" in "$checkout/"*) echo 'State must be outside the checkout' >&2; exit 2;; esac
case "$checkout/" in "$state/"*) echo 'State must not contain the checkout' >&2; exit 2;; esac
case "${1:-}" in
  inkling) lane=inkling; backend=cu132 ;;
  lightning|muse|super-vl) lane=stable; backend=cu130 ;;
  *) echo 'Usage: bash scripts/install-runtime.sh MODEL' >&2; exit 2 ;;
esac
lock="$repo/locks/$lane.lock"
[[ -s "$lock" ]] || { echo "Missing resolved lock: $lock" >&2; exit 1; }
umask 077
mkdir -p -- "$state"
[[ -O "$state" && $(stat -c '%a' -- "$state") == 700 ]] || {
  echo 'State must be an owner-only directory with permissions 0700' >&2; exit 1;
}
mkdir -p -- "$state/runtimes"
export UV_CACHE_DIR="$state/uv-cache"
runtime="$state/runtimes/$lane"
exec 9>"$state/runtimes/$lane.install.lock"
flock 9
lock_hash=$(sha256sum "$lock" | cut -d' ' -f1)
configure_policy() {
  "$runtime/bin/python" -I -B "$repo/backend_policy.py" configure-private-install
}
if [[ -e "$runtime" ]]; then
  if [[ -f "$runtime/.pace-lock-sha256" && $(cat "$runtime/.pace-lock-sha256") == "$lock_hash" ]]; then
    uv --no-config pip check --python "$runtime/bin/python"
    [[ "$lane" == inkling ]] || configure_policy
    exit
  fi
  if [[ ! -f "$runtime/.pace-install-in-progress" || $(cat "$runtime/.pace-install-in-progress") != "$lock_hash" ]]; then
    echo "Existing runtime at $runtime. Use a fresh PACE_INFERENCE_STATE directory." >&2
    exit 1
  fi
else
  python=$(uv --no-config python find --no-python-downloads "${PACE_PYTHON:-3.12.15}")
  "$python" -EB -c 'import sys; sys.exit(sys.version_info[:3] != (3, 12, 15))' || {
    echo 'The resolved locks require Python 3.12.15' >&2; exit 1;
  }
  # Allow room for both copied wheels and the uv cache on a first install.
  cd -- "$repo"
  "$python" -EB - <<'PY'
from pace_inference import cache_root, disk_headroom, GIB

fs, free, inodes = disk_headroom(cache_root())
print(f"{fs}: {free / GIB:.2f} GiB and {inodes} inodes free before runtime install")
if free < 20 * GIB or inodes < 140_000:
    raise SystemExit("First runtime install needs 20 GiB and 140,000 inodes of headroom")
PY
  uv --no-config venv --python "$python" --no-python-downloads "$runtime"
  printf '%s\n' "$lock_hash" > "$runtime/.pace-install-in-progress"
fi

# Copy mode keeps restrictive file permissions local to this environment.
uv --no-config pip install --require-hashes --only-binary :all: --link-mode copy \
  --python "$runtime/bin/python" --torch-backend="$backend" -r "$lock"
uv --no-config pip check --python "$runtime/bin/python"
uv --no-config pip freeze --python "$runtime/bin/python" > "$runtime/installed.txt"
configure_policy
mv -- "$runtime/.pace-install-in-progress" "$runtime/.pace-lock-sha256"
