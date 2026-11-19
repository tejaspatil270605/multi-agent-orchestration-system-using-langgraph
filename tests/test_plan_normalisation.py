"""Supervisor plan sanitising: python -m pytest tests/test_plan_normalisation.py"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from schemas import ExecutionPlan, Task
from supervisor.supervisor import normalise_plan


def plan(*specs):
    return ExecutionPlan(tasks=[Task(description=d, assigned_agent=a, required_input=[],
                                     expected_output="", complexity="low", dependencies=deps)
                                for d, a, deps in specs])


def test_forward_reference_is_reordered_and_remapped():
    p = normalise_plan(plan(("write", "writing_agent", [1, 2]),
                            ("research", "research_agent", []),
                            ("data", "data_agent", [])))
    assert [t.description for t in p.tasks] == ["research", "data", "write"]
    assert p.tasks[2].dependencies == [0, 1]


def test_invalid_self_and_cycle_dependencies_are_removed():
    p = normalise_plan(plan(("a", "research_agent", [0, 9, -1]),
                            ("b", "data_agent", [2]), ("c", "coding_agent", [1])))
    for i, t in enumerate(p.tasks):
        assert all(0 <= d < i for d in t.dependencies)
    assert p.tasks[0].dependencies == []


def test_good_plan_unchanged():
    p = normalise_plan(plan(("r", "research_agent", []), ("w", "writing_agent", [0])))
    assert [t.description for t in p.tasks] == ["r", "w"] and p.tasks[1].dependencies == [0]
