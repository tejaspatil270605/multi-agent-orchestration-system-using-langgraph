from typing import TypedDict, Optional, Annotated
from schemas import ExecutionPlan, Task, AgentResult
from typing import List, Dict


def merge_task_status(old, new):
    return {**(old or {}), **(new or {})}


def merge_result(old, new):
    """Merge dicts coming from parallel agents.
    A value of None means "delete this key" (used by task_router to clear
    stale agent results after they have been handled)."""
    merged = {**(old or {}), **(new or {})}
    return {k: v for k, v in merged.items() if v is not None}


class State(TypedDict):
    user_input: str
    additional_input: Optional[str]
    plan: Optional[ExecutionPlan]
    task: Optional[Task]
    task_index: Optional[int]
    result: Annotated[Dict[int, str], merge_result]
    completed_tasks: Optional[List[int]]
    resume_task: Optional[int]
    task_status: Annotated[Dict[int, str], merge_task_status]
    retry_count: Annotated[Dict[int, int], merge_task_status]
    agent_result: Annotated[Dict[int, AgentResult], merge_result]
    # --- execution vs quality, failure type, input origin, retry cache ---
    validation: Annotated[Dict[int, dict], merge_result]     # {task: {status, checks, notes}}  (quality)
    failures: Annotated[Dict[int, dict], merge_result]       # {task: {category, transient, max_retries, message}}
    provenance: Annotated[Dict[str, dict], merge_result]     # {"input_for_task_1": {source, sha256_12, ...}}
    cache: Annotated[Dict[int, dict], merge_result]          # {task: {...}} reused by retries (no repeated searches)
