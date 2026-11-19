from state import State
import audit

MAX_RETRIES = 2


def task_router(state: State):
    """Read the fresh agent results and update the status of every task.

    statuses: pending -> running -> completed
              running -> waiting_for_input -> (user answers) -> running
              running -> failed -> retrying (up to MAX_RETRIES) -> failed
              blocked = a dependency failed for good
    """
    plan = state['plan']
    completed_tasks = set(state.get('completed_tasks') or [])
    current_status = state.get('task_status') or {}
    retry_count = state.get('retry_count') or {}
    agent_results = state.get('agent_result') or {}
    resume_task = state.get('resume_task')

    updated_status = {}
    updated_retry_count = {}
    handled_results = {}          # {index: None} -> removes the stale result from state
    handled_failures = {}         # {index: None} -> removes the failure record once a retry is scheduled
    failures = state.get('failures') or {}

    # 1. absorb the results the agents just produced
    for index, agent_result in agent_results.items():
        if agent_result.status == 'completed':
            completed_tasks.add(index)
            updated_status[index] = "completed"
        elif agent_result.status == 'waiting_for_input':
            updated_status[index] = "waiting_for_input"
        elif agent_result.status == 'failed':
            updated_status[index] = "failed"

    # 2. walk the plan in order (dependencies always have a smaller index)
    for index, task in enumerate(plan.tasks):
        if index in completed_tasks:
            updated_status[index] = "completed"
            continue

        status = updated_status.get(index, current_status.get(index, "pending"))

        if status == "waiting_for_input":
            if resume_task == index:               # the user just answered
                updated_status[index] = "running"
                handled_results[index] = None
            else:
                updated_status[index] = "waiting_for_input"
            continue

        if status == "failed":
            retries = retry_count.get(index, 0)
            failure = failures.get(index) or {}
            allowed = min(failure.get('max_retries', MAX_RETRIES), MAX_RETRIES)   # permanent errors: 0
            if retries < allowed:
                updated_retry_count[index] = retries + 1
                updated_status[index] = "retrying"
                handled_results[index] = None      # forget the old failure
                handled_failures[index] = None
                print(f"Task {index}: transient failure [{failure.get('category', 'unknown')}] - retry "
                      f"{retries + 1}/{allowed}")
            else:
                updated_status[index] = "failed"
                if retries == 0 and allowed == 0:
                    print(f"Task {index}: permanent failure [{failure.get('category', 'unknown')}] - not retried")
            continue

        dependency_failed = any(
            updated_status.get(dependency) in ("failed", "blocked")
            for dependency in task.dependencies
        )
        if dependency_failed:
            updated_status[index] = "blocked"
            continue

        dependency_ready = all(
            dependency in completed_tasks
            for dependency in task.dependencies
        )
        if dependency_ready and status == "pending":
            updated_status[index] = "running"
        else:
            updated_status[index] = status

    print("\nUpdated Task Status:", updated_status)
    if updated_retry_count:
        print("Retry Count:", {**retry_count, **updated_retry_count})

    audit.log_event("router", task_status=updated_status, retry_count={**retry_count, **updated_retry_count})
    return {
        "failures": handled_failures,
        "task_status": updated_status,
        "retry_count": updated_retry_count,
        "completed_tasks": sorted(completed_tasks),
        "agent_result": handled_results,
        "resume_task": None,
    }
