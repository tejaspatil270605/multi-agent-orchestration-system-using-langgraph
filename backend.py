import os
import time
import audit
import agents
from workflow.graph import graph
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.types import Command
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer

serde = JsonPlusSerializer(allowed_msgpack_modules=[
    ("schemas", "ExecutionPlan"),
    ("schemas", "Task"),
    ("schemas", "AgentResult")])

USER_REQUEST = ("Research LangGraph and analyse Python data independently,"
                "then write a beginner-friendly article using both results.")

# DEMO_MODE=1 (default): a built-in demo dataset answers the "please provide data" question automatically
# and is LABELLED as a demo constant. DEMO_MODE=0: you are asked to type or paste the data yourself.
DEMO_MODE = os.getenv("DEMO_MODE", "1") == "1"
DEMO_ANSWER = """ Python usage dataset:
                  Month: Jan, Feb, Mar, Apr, May
                  PyPI downloads: 1000, 1200, 1500, 1800, 2100
              """
DEMO_SOURCE = "built-in demo dataset (not supplied by the user)"


def ask_user(question: dict) -> dict:
    """The answer for a paused task, together with where it came from."""
    if DEMO_MODE:
        return {"text": DEMO_ANSWER, "source": DEMO_SOURCE}
    print("Paste your data, then press Enter on an empty line:")
    lines = []
    while True:
        line = input()
        if not line.strip():
            break
        lines.append(line)
    return {"text": "\n".join(lines), "source": "typed or pasted by the user at the prompt"}


def show(result):
    status = result.get("task_status") or {}
    validation = result.get("validation") or {}
    failures = result.get("failures") or {}
    retries = result.get("retry_count") or {}
    plan = result.get("plan")

    print("\n========== AUDIT SUMMARY ==========")
    print(f"{'task':<5}{'agent':<16}{'execution':<12}{'validation':<17}{'attempts':<9}notes")
    needs_review = []
    for index in sorted(status):
        v = validation.get(index) or {}
        v_status = v.get("status", "n/a")
        agent = plan.tasks[index].assigned_agent if plan else "?"
        f = failures.get(index) or {}
        note = (f"failure: {f.get('category')}" if f else "") or "; ".join(v.get("notes", [])[:1])
        print(f"{index:<5}{agent:<16}{status[index]:<12}{v_status:<17}{retries.get(index, 0) + 1:<9}{note[:70]}")
        if v_status not in ("passed", "n/a"):
            needs_review.append(index)
    for key, info in sorted((result.get("provenance") or {}).items()):
        print(f"input provenance [{key}]: " + "; ".join(f"{k}={v}" for k, v in info.items() if k in ("source", "origin", "sha256_12")))
    st = agents.STATS
    print(f"LLM calls: {st['llm_calls']} (errors: {st['llm_errors']}), tokens used: {st['tokens']}, "
          f"waited for rate limits: {st['gate_wait_s']:.0f}s + backoff {st['backoff_wait_s']:.0f}s, "
          f"total run time: {time.time() - START:.0f}s")
    print(f"audit log: {audit.log_path()}")
    if needs_review:
        print("\n" + "!" * 60)
        print(f"REVIEW REQUIRED for task(s) {needs_review}: do not publish before checking these results.")
        for index in needs_review:
            v = validation[index]
            print(f"  task {index}: checks {v.get('checks')}")
            for note in v.get("notes", []):
                print(f"    - {note[:220]}")
        print("!" * 60)

    for index, text in sorted((result.get("result") or {}).items()):
        flag = "  [REVIEW REQUIRED]" if index in needs_review else ""
        print(f"\n--- Task {index} output{flag} ---\n{text}")


START = time.time()
with SqliteSaver.from_conn_string("orchestration.db") as memory:
    memory.serde = serde
    graph = graph.compile(checkpointer=memory)
    config = {'configurable': {'thread_id': 'user_1'}}
    result = graph.invoke({
        "user_input": USER_REQUEST,
        "additional_input": None,
        "plan": None,
        "task": None,
        "task_index": None,
        "result": {},
        "resume_task": None,
        "completed_tasks": [],
        "task_status": {},
        "retry_count": {},
        "agent_result": {},
        "validation": {},
        "failures": {},
        "provenance": {},
        "cache": {},
    }, config=config)

    # human-in-the-loop: keep answering until the graph stops asking
    while result.get("__interrupt__"):
        question = result["__interrupt__"][0].value
        print("\n>>>", question["message"])
        result = graph.invoke(Command(resume=ask_user(question)), config=config)

    show(result)
