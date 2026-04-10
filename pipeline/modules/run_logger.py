import atexit
import json
import os
import sys
from datetime import datetime
from typing import Any, Dict, Optional


class _TeeStream:
    def __init__(self, stream, log_file):
        self._stream = stream
        self._log_file = log_file

    def write(self, data):
        self._stream.write(data)
        self._log_file.write(data)
        return len(data)

    def flush(self):
        self._stream.flush()
        self._log_file.flush()

    def isatty(self):
        return self._stream.isatty()



def _json_default(obj):
    return str(obj)



def init_run_logging(script_subdir: str, hyperparams: Optional[Dict[str, Any]] = None):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_dir = os.path.join("logs", script_subdir)
    os.makedirs(base_dir, exist_ok=True)

    log_path = os.path.join(base_dir, f"{timestamp}.log")
    log_file = open(log_path, "a", encoding="utf-8", buffering=1)

    original_stdout = sys.stdout
    original_stderr = sys.stderr

    sys.stdout = _TeeStream(original_stdout, log_file)
    sys.stderr = _TeeStream(original_stderr, log_file)

    state = {
        "script_subdir": script_subdir,
        "log_path": log_path,
        "log_file": log_file,
        "original_stdout": original_stdout,
        "original_stderr": original_stderr,
        "closed": False,
    }

    print(f"[RUN_START] {datetime.now().isoformat()}")
    print(f"[RUN_LOG_PATH] {log_path}")
    if hyperparams is not None:
        print("[HYPERPARAMS]", json.dumps(hyperparams, indent=2, default=_json_default))

    def _cleanup():
        close_run_logging(state, status="exit")

    atexit.register(_cleanup)
    return state



def log_run_results(run_log_state, results: Dict[str, Any]):
    print("[RUN_RESULTS]", json.dumps(results, indent=2, default=_json_default))



def close_run_logging(run_log_state, status: str = "success"):
    if not run_log_state or run_log_state.get("closed"):
        return

    print(f"[RUN_END] {datetime.now().isoformat()} | status={status}")

    sys.stdout = run_log_state["original_stdout"]
    sys.stderr = run_log_state["original_stderr"]

    run_log_state["log_file"].flush()
    run_log_state["log_file"].close()
    run_log_state["closed"] = True
