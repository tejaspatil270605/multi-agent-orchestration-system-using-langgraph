import hashlib
from datetime import datetime
from langgraph.types import interrupt
from state import State


def human_input_node(state: State):
    """Pause the graph and ask the user for the information a task is missing.
    The answer is stored together with its ORIGIN (typed by the user, a demo constant, a file ...)."""
    waiting_task = sorted(
        index
        for index, status in (state.get('task_status') or {}).items()
        if status == 'waiting_for_input'
    )

    if not waiting_task:
        return {}

    task_index = waiting_task[0]
    task = state['plan'].tasks[task_index]
    agent_result = (state.get('agent_result') or {}).get(task_index)
    reason = agent_result.reason if agent_result else "More information is required."
    needed = agent_result.output if agent_result else ""

    # NOTE: on resume LangGraph re-runs this node from the top, so everything
    # above this line must stay side-effect free.
    answer = interrupt({
        "type": "input required",
        "task_index": task_index,
        "task": task.description,
        "message": (
            f"Task {task_index} needs additional information.\n"
            f"Reason: {reason}"
        ),
        "details": needed,
    })

    # the answer is either plain text or {"text": ..., "source": ...}
    if isinstance(answer, dict):
        text, source = str(answer.get("text", "")), str(answer.get("source", "unspecified"))
    else:
        text, source = str(answer), "typed by the user"

    provenance = {
        f"input_for_task_{task_index}": {
            "source": source,
            "task": task_index,
            "task_description": task.description,
            "received_at": datetime.now().isoformat(timespec="seconds"),
            "sha256_12": hashlib.sha256(text.strip().encode("utf-8")).hexdigest()[:12],
            "chars": len(text),
            "preview": " ".join(text.split())[:80],
        }
    }
    return {
        'additional_input': text,
        'resume_task': task_index,
        'provenance': provenance,
    }
