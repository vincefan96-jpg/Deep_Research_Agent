from pydantic import BaseModel, Field
from typing import Optional
from enum import Enum


class StepType(str, Enum):
    THOUGHT = "thought"
    ACTION = "action"
    OBSERVATION = "observation"


class ResearchRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=2000)


class ToolCall(BaseModel):
    tool_name: str
    params: dict


class Step(BaseModel):
    round: int
    type: StepType
    content: str
    tool_name: Optional[str] = None
    tool_params: Optional[dict] = None


class PlanEvent(BaseModel):
    sub_questions: list[str]


class StepEvent(BaseModel):
    type: str  # "thought" | "action" | "observation"
    round: int
    content: str
    tool_name: Optional[str] = None
    tool_params: Optional[dict] = None


class CrossCheckEvent(BaseModel):
    consistency: str  # "high" | "medium" | "low"
    conflicts: list[str]
    verified_facts: list[str]


class ReportEvent(BaseModel):
    markdown: str
    sources: list[str]


class ResearchSession(BaseModel):
    id: str
    query: str
    status: str
    created_at: str
    report: Optional[str] = None
    steps: list[Step] = []
