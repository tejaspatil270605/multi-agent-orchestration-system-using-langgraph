"""Structured audit log: one JSON line per event (attempt, error category, provenance,
validation outcome) in logs/run_<id>.jsonl, so a run can be reviewed afterwards."""
import contextvars
import json
import os
import threading
from datetime import datetime

RUN_ID = datetime.now().strftime("%Y%m%d-%H%M%S")
_lock = threading.Lock()
_ctx = contextvars.ContextVar("audit_ctx", default={})


def log_path() -> str:
    return os.path.join(os.getenv("AUDIT_LOG_DIR", "logs"), f"run_{RUN_ID}.jsonl")


def set_context(**fields):
    """Fields (agent, task, attempt ...) added to every following event of this call chain."""
    _ctx.set({**_ctx.get(), **fields})


def log_event(event: str, **fields) -> dict:
    record = {"ts": datetime.now().isoformat(timespec="seconds"), "run": RUN_ID,
              "event": event, **_ctx.get(), **fields}
    if os.getenv("AUDIT_DISABLE") != "1":
        try:
            path = log_path()
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with _lock, open(path, "a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
        except OSError:
            pass                                   # logging must never break a run
    return record
