"""Opt-in per-request telemetry shared by concurrent CPU clients."""
import fcntl
import json
import os
from pathlib import Path
import time


def record(model, started, *, usage=None, error=None):
    filename = os.getenv('DT_REQUEST_LOG')
    if not filename:
        return
    path = Path(filename)
    path.parent.mkdir(parents=True, exist_ok=True)
    row = dict(model=model, phase=os.getenv('DT_PHASE', 'search'),
               generation=os.getenv('JB_GEN', '0'), seconds=time.monotonic() - started,
               usage=usage or {}, error=error, timestamp=time.time())
    with path.open('a') as stream:
        fcntl.flock(stream, fcntl.LOCK_EX)
        stream.write(json.dumps(row) + '\n')
        stream.flush()
        fcntl.flock(stream, fcntl.LOCK_UN)
