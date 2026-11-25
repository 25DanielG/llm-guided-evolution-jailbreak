"""Install the backend guard before importing vLLM in every spawned process."""

import os
import sys

try:
    from backend_policy import install_process_guard

    install_process_guard()
except BaseException as error:
    # Python ordinarily ignores exceptions from sitecustomize. Stop instead.
    sys.stderr.write(f'PACE_BACKEND guard failed: {error}\n')
    sys.stderr.flush()
    os._exit(78)
