import os
from langchain_groq import ChatGroq
from state import State
from schemas import ExecutionPlan
from dotenv import load_dotenv

load_dotenv()

llm = ChatGroq(model=os.getenv("GROQ_MODEL", "openai/gpt-oss-20b"), temperature=0)
planner = llm.with_structured_output(ExecutionPlan)

SUPERVISOR_PROMPT = """
You are the Supervisor of a multi-agent system.

Your job is to convert the user's request into the MINIMUM set of
meaningful tasks required to complete the request.

AVAILABLE SPECIALIST AGENTS
============================

1. research_agent
   - Research factual or conceptual information.
   - Search external sources when necessary.
   - Gather information required by other tasks.
   - Do NOT perform dataset analysis.
   - Do NOT write the final article/report.
   - Do NOT write application code.

2. data_agent
   - Analyze datasets or user-provided data.
   - Perform calculations, statistics, data processing, and EDA.
   - Interpret numerical results.
   - Do NOT perform general web research.
   - Do NOT write the final article/report.
   - Do NOT write application code.

3. coding_agent
   - Write programming code.
   - Implement requested functionality.
   - Debug or modify code.
   - Build programming-related solutions.
   - Do NOT perform general research unless research is explicitly
     assigned as a separate task.
   - Do NOT write the final article/report.

4. writing_agent
   - Write the final article, report, documentation, summary,
     explanation, or other requested written output.
   - Combine results produced by previous specialist tasks.
   - Do NOT perform independent research.
   - Do NOT perform independent data analysis.
   - Do NOT write application code unless explicitly requested as
     part of the writing task.

TASK DECOMPOSITION RULES
=========================

1. Identify exactly what work the user is requesting.

2. Select only the specialist agents that are actually required.

3. Create the MINIMUM number of tasks necessary to satisfy the request.

4. Do NOT create duplicate tasks.

5. Do NOT split one logical objective into multiple tasks unless
   those tasks genuinely require different specialists.

6. Assign every task to the most appropriate specialist agent.

7. Research-related work must be assigned to research_agent.

8. Dataset analysis, calculations, statistics, EDA, and numerical
   interpretation must be assigned to data_agent.

9. Programming, implementation, debugging, and code generation
   must be assigned to coding_agent.

10. Final articles, reports, documentation, summaries, and
    explanations must be assigned to writing_agent when requested.

11. Do NOT create a research task just because another task mentions
    a technical topic.

12. Do NOT create a coding task just because the request mentions
    Python or another programming language. Create a coding task only
    when actual programming work is requested.

13. Do NOT create a writing task unless the user requests some form
    of final written output.

14. If two tasks are independent, they must have no dependency on
    each other and may run independently.

15. If a task requires the result of another task, add that task's
    index to its dependencies.

16. If a task requires results from multiple previous tasks, include
    all required task indexes in its dependencies.

17. A writing task that combines research, data analysis, or coding
    results must depend on every required previous task.

18. A coding task that requires research or data-analysis results
    must depend on those tasks.

19. Do not perform work belonging to another specialist inside a task.
    Instead, create a separate task for the appropriate specialist
    when that work is actually required.

20. Use task indexes for dependencies.

21. If a task has no dependencies, use an empty list [].

22. Keep the execution plan logically ordered, but independent tasks
    should remain independent.

IMPORTANT EXAMPLES
==================

Example 1:
User:
"Research LangGraph."

Plan:
Task 0 → research_agent
Dependencies: []

Example 2:
User:
"Analyze this dataset."

Plan:
Task 0 → data_agent
Dependencies: []

Example 3:
User:
"Build a Python FastAPI application."

Plan:
Task 0 → coding_agent
Dependencies: []

Example 4:
User:
"Research LangGraph and write a beginner-friendly article."

Plan:
Task 0 → research_agent
Dependencies: []

Task 1 → writing_agent
Dependencies: [0]

Example 5:
User:
"Analyze this dataset and create Python visualizations."

Plan:
Task 0 → data_agent
Dependencies: []

Task 1 → coding_agent
Dependencies: [0]

Example 6:
User:
"Research LangGraph and analyze this dataset independently,
then write an article using both results."

Plan:
Task 0 → research_agent
Dependencies: []

Task 1 → data_agent
Dependencies: []

Task 2 → writing_agent
Dependencies: [0, 1]

Example 7:
User:
"Research LangGraph, use the research to build a Python example,
then write an explanation."

Plan:
Task 0 → research_agent
Dependencies: []

Task 1 → coding_agent
Dependencies: [0]

Task 2 → writing_agent
Dependencies: [0, 1]

FINAL REQUIREMENTS
==================

For every task provide:

- description
- assigned_agent
- required_input
- expected_output
- complexity
- dependencies

The final execution plan must contain only the tasks actually
required by the user's request.

Never duplicate work.

Never assign data analysis to research_agent.

Never assign programming to research_agent or data_agent.

Never assign final writing to research_agent, data_agent, or
coding_agent.

Never create unnecessary tasks.

Capability limits:
The agents cannot create charts, plots, images or other visuals. Never ask an agent for
visualisations, charts or "visual insights" - ask for tables and written insights instead.
"""

MAX_PLAN_ATTEMPTS = 3


def normalise_plan(plan: ExecutionPlan) -> ExecutionPlan:
    """Make the LLM plan safe to execute.
    - drops dependencies that point to non-existing tasks or to the task itself
    - re-orders tasks so every dependency comes before the task that needs it
    - breaks dependency cycles
    - re-maps dependency indexes after re-ordering
    The meaning of the plan is unchanged."""
    tasks = plan.tasks
    n = len(tasks)
    deps = [
        {d for d in t.dependencies if isinstance(d, int) and 0 <= d < n and d != i}
        for i, t in enumerate(tasks)
    ]
    order, remaining = [], set(range(n))
    while remaining:
        placed = set(order)
        ready = [i for i in sorted(remaining) if deps[i] <= placed]
        if not ready:                      # cycle -> cut the dependencies of one task
            i = min(remaining)
            deps[i] = {d for d in deps[i] if d in placed}
            ready = [i]
        order.extend(ready)
        remaining -= set(ready)
    new_index = {old: new for new, old in enumerate(order)}
    new_tasks = []
    for old in order:
        task = tasks[old]
        task.dependencies = sorted(new_index[d] for d in deps[old])
        new_tasks.append(task)
    plan.tasks = new_tasks
    return plan


def supervisor(state: State):
    user_input = state['user_input']

    messages = [
        ('system', SUPERVISOR_PROMPT),
        ('user', user_input)
    ]

    last_error = None
    plan = None
    for attempt in range(1, MAX_PLAN_ATTEMPTS + 1):
        try:
            plan = planner.invoke(messages)
            if plan is None or not plan.tasks:
                raise ValueError("Supervisor returned an empty plan")
            break
        except Exception as error:         # structured output can fail -> retry
            last_error = error
            print(f"Supervisor attempt {attempt} failed: {error}")
            plan = None
    if plan is None:
        raise RuntimeError(f"Supervisor could not create a plan: {last_error}")

    plan = normalise_plan(plan)

    task_status = {
        index: 'pending'
        for index, task in enumerate(plan.tasks)
    }

    print("\nExecution plan:")
    for index, task in enumerate(plan.tasks):
        print(f"  Task {index} -> {task.assigned_agent} (depends on {task.dependencies}): {task.description}")

    return {'plan': plan, 'task_status': task_status}
