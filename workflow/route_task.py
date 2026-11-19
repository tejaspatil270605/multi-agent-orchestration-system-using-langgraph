from state import State
from langgraph.types import Send


def route_task(state: State):
    """Decide what happens next:
       - a list of Send(...)  -> run every ready task in parallel
       - "human_input"        -> nothing can run, but a task waits for the user
       - "finish"             -> everything is done (or nothing more can run)
    """
    plan = state['plan']
    completed_task = set(state.get('completed_tasks') or [])
    task_status = state.get('task_status') or {}
    all_results = state.get('result') or {}

    if len(completed_task) == len(plan.tasks):
        return "finish"

    send = []
    for index, task in enumerate(plan.tasks):
        # skip already completed task
        if index in completed_task:
            continue

        status = task_status.get(index, 'pending')

        # only tasks the router marked as running / retrying are dispatched
        if status not in ('running', 'retrying'):
            continue
        if not all(dep in completed_task for dep in task.dependencies):
            continue

        if status == 'retrying':
            print(f"\nRetrying Task {index}: {task.description}")

        dependency_result = {
            dependency: all_results[dependency]
            for dependency in task.dependencies
            if dependency in all_results
        }

        payload = {
            "user_input": state["user_input"],
            "additional_input": state.get('additional_input'),
            "task": task,
            "task_index": index,
            "result": dependency_result,
            "task_status": task_status,
            "retry_count": state.get('retry_count', {}),
            "provenance": state.get('provenance', {}),
            "cache": state.get('cache', {}),
            "failures": state.get('failures', {}),
        }
        send.append(Send(task.assigned_agent, payload))

    # run everything that is ready first - independent tasks never wait for the user
    if send:
        return send

    waiting_tasks = [
        index for index, status in task_status.items()
        if status == "waiting_for_input"
    ]
    if waiting_tasks:
        print(f"\nWorkflow is waiting for user input for tasks: {waiting_tasks}")
        return "human_input"

    return "finish"
