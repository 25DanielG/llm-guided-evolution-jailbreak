"""Opt-in subprocess seeding for the paired experiment."""
import hashlib
import json
import os
from pathlib import Path


def request_seed(payload):
    encoded = json.dumps(payload, sort_keys=True, ensure_ascii=True)
    return int.from_bytes(hashlib.sha256((os.environ['DT_SEED'] + encoded).encode()).digest()[:4], 'big')


def seed_process(argv):
    if 'DT_SEED' not in os.environ:
        return
    import random
    import numpy as np
    seed = request_seed([Path(arg).name for arg in argv])
    random.seed(seed)
    np.random.seed(seed)
