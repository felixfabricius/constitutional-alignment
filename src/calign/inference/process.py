"""Process exit for CLIs that load vLLM.

With LoRA enabled, a vLLM 0.29 engine-core child process kept the interpreter alive after all outputs were written
(chunk 5: a suite run hung 70 min after its manifest), and a plain `os._exit` orphaned that child together with its
~75 GB of GPU memory (the next job's engine failed to start). `run_and_exit(main)` runs the CLI, then terminates the
children, and exits hard with the CLI's exit code. Use it only under `if __name__ == "__main__":` (never in tests).
"""

from __future__ import annotations

import logging
import os
import sys
from collections.abc import Callable

LOGGER = logging.getLogger(__name__)


def terminate_children(timeout: float = 30.0) -> None:
    """Terminate (then kill) this process's children, i.e. vLLM's engine-core process, so its GPU memory is freed."""
    try:
        import psutil  # a vLLM dependency; absent on machines without vLLM
    except ImportError:
        return
    kids = psutil.Process().children(recursive=True)
    for k in kids:
        k.terminate()
    _, alive = psutil.wait_procs(kids, timeout=timeout)
    for k in alive:
        k.kill()


def exit_code(e: SystemExit) -> int:
    if e.code is None:
        return 0
    if isinstance(e.code, int):
        return e.code
    print(e.code, file=sys.stderr)
    return 1


def run_and_exit(main: Callable[[], object]) -> None:
    code = 0
    try:
        main()
    except SystemExit as e:
        code = exit_code(e)
    except BaseException:
        LOGGER.exception("failed")
        code = 1
    sys.stdout.flush()
    sys.stderr.flush()
    terminate_children()
    logging.shutdown()
    os._exit(code)
