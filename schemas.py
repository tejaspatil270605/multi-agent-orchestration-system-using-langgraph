from pydantic import BaseModel, Field
from typing import List, Literal, Optional

AgentName = Literal["research_agent", "data_agent", "coding_agent", "writing_agent"]


class Task(BaseModel):
    description: str = Field(description="what needs to be done")
    assigned_agent: AgentName = Field(
        description="Which specialist should handle this task. "
                    "One of: research_agent, data_agent, coding_agent, writing_agent"
    )
    required_input: List[str] = Field(description="Information required to complete the task")
    expected_output: str = Field(description="What output this task should produce")
    complexity: Literal["low", "medium", "high"] = Field(
        description="Task complexity: low, medium, or high"
    )
    dependencies: List[int] = Field(
        default_factory=list,
        description="Indexes of tasks that must finish before this task"
    )


class ExecutionPlan(BaseModel):
    tasks: List[Task]


class AgentResult(BaseModel):
    status: Literal["completed",
                    "waiting_for_input",
                    "failed"
                    ] = Field(description=("Task execution status. "
                                           "Must be one of: completed, waiting_for_input, failed"
                                           ))
    output: str = Field(description="The actual output produced by the agent")
    reason: str = Field(description="Why the task has this status")


# Research Agent Schemas
class EvidenceItem(BaseModel):
    title: str
    url: str
    published_at: Optional[str] = None
    snippets: Optional[str] = None
    sources: Optional[str] = None


class SearchQueries(BaseModel):
    queries: List[str] = Field(
        min_length=5,
        max_length=7,
        description="5 to 7 focused and non-redundant search queries for the research task"
    )


class EvidencePack(BaseModel):
    evidence: List[EvidenceItem] = Field(default_factory=list)

