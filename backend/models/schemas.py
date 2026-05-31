from pydantic import BaseModel, Field
from typing import Optional
from enum import Enum


class StepType(str, Enum):
    THOUGHT = "thought"
    ACTION = "action"
    OBSERVATION = "observation"

#前端发送的研究请求
class ResearchRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=2000)

#工具调用记录
class ToolCall(BaseModel):
    tool_name: str
    params: dict

#内存和数据库中的步骤记录
class Step(BaseModel):
    round: int
    type: StepType
    content: str
    tool_name: Optional[str] = None
    tool_params: Optional[dict] = None

#SSE plan 事件
class PlanEvent(BaseModel):
    sub_questions: list[str]

#SSE step 事件（每次 Thought/Action/Observation）
class StepEvent(BaseModel):
    type: str  # "thought" | "action" | "observation"
    round: int
    content: str
    tool_name: Optional[str] = None
    tool_params: Optional[dict] = None

#SSE report 事件
class ReportEvent(BaseModel):
    markdown: str
    sources: list[str]

#数据库中的完整调研会话
class ResearchSession(BaseModel):
    id: str
    query: str
    status: str
    created_at: str
    report: Optional[str] = None
    steps: list[Step] = []
