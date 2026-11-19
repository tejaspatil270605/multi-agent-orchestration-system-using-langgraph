"""Retry policy and status tracking of the task router: python -m pytest tests/test_router.py"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from schemas import ExecutionPlan, Task, AgentResult
from workflow.task_router import task_router


def plan():
    mk = lambda a, d: Task(description="t", assigned_agent=a, required_input=[], expected_output="", complexity="low", dependencies=d)
    return ExecutionPlan(tasks=[mk("research_agent", []), mk("data_agent", []), mk("writing_agent", [0, 1])])


def run(agent_results, failures=None, retry_count=None, status=None, completed=None):
    state = {"plan": plan(), "completed_tasks": completed or [], "task_status": status or {0: "running", 1: "running", 2: "pending"},
             "retry_count": retry_count or {}, "agent_result": agent_results, "failures": failures or {}, "resume_task": None}
    return task_router(state)


FAIL = AgentResult(status="failed", output="", reason="x")
DONE = AgentResult(status="completed", output="o", reason="r")


def test_transient_failure_is_retried_and_failure_record_cleared():
    out = run({0: FAIL, 1: DONE}, failures={0: {"category": "rate_limit", "transient": True, "max_retries": 2}})
    assert out["task_status"][0] == "retrying" and out["retry_count"] == {0: 1}
    assert out["failures"] == {0: None} and out["agent_result"] == {0: None}


def test_permanent_failure_is_not_retried_and_blocks_dependents():
    out = run({0: FAIL, 1: DONE}, failures={0: {"category": "missing_key", "transient": False, "max_retries": 0}})
    assert out["task_status"] == {0: "failed", 1: "completed", 2: "blocked"} and out["retry_count"] == {}


def test_retry_budget_is_bounded():
    f = {0: {"category": "rate_limit", "transient": True, "max_retries": 2}}
    assert run({0: FAIL}, failures=f, retry_count={0: 1})["task_status"][0] == "retrying"
    assert run({0: FAIL}, failures=f, retry_count={0: 2})["task_status"][0] == "failed"
    unknown = {0: {"category": "unknown", "transient": True, "max_retries": 1}}
    assert run({0: FAIL}, failures=unknown, retry_count={0: 1})["task_status"][0] == "failed"
    assert run({0: FAIL}, failures={}, retry_count={0: 0})["task_status"][0] == "retrying"   # no record: default policy
