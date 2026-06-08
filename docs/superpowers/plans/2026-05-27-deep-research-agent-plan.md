# Deep Research Agent Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a web-based deep research AI agent that autonomously searches, reads, analyzes, cross-validates, and generates cited research reports using a self-implemented ReAct loop.

**Architecture:** FastAPI backend with SSE streaming drives a self-built ReAct agent engine (loop + parser + memory + tools + orchestrator). Vue 3 frontend renders real-time research steps and final Markdown reports. No LangChain/LlamaIndex — all agent logic is hand-rolled.

**Tech Stack:** Python 3.11+, FastAPI, Vue 3 (Vite), SQLite + aiosqlite, httpx, beautifulsoup4, openai SDK, marked.js

---

### Task 1: Project Scaffolding

**Files:**
- Create: `backend/requirements.txt`
- Create: `backend/config.py`
- Create: `frontend/package.json` (via `npm create vite`)
- Create: `.gitignore`

- [ ] **Step 1: Create backend directory and requirements**

```bash
mkdir -p backend/agent/tools backend/llm backend/models backend/routes backend/db tests
```

- [ ] **Step 2: Write backend/requirements.txt**

```
fastapi>=0.110.0
uvicorn[standard]>=0.27.0
openai>=1.30.0
httpx>=0.27.0
beautifulsoup4>=4.12.0
aiosqlite>=0.20.0
pydantic>=2.0.0
python-dotenv>=1.0.0
pytest>=8.0.0
pytest-asyncio>=0.23.0
```

- [ ] **Step 3: Write backend/config.py**

```python
import os
from dotenv import load_dotenv

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o")
SEARCH_API_KEY = os.getenv("SEARCH_API_KEY", "")
SEARCH_API_URL = os.getenv("SEARCH_API_URL", "https://serpapi.com/search")
MAX_ROUNDS = int(os.getenv("MAX_ROUNDS", "10"))
MAX_TOKENS = int(os.getenv("MAX_TOKENS", "8000"))
DB_PATH = os.getenv("DB_PATH", "research.db")
```

- [ ] **Step 4: Scaffold Vue 3 frontend with Vite**

```bash
cd frontend && npm create vite@latest . -- --template vue && npm install && npm install marked
```

- [ ] **Step 5: Write .gitignore**

```
node_modules/
dist/
.env
__pycache__/
*.pyc
research.db
.venv/
```

- [ ] **Step 6: Install Python dependencies and verify**

```bash
cd backend && pip install -r requirements.txt && python -c "import fastapi; print('OK')"
```
Expected: prints "OK"

- [ ] **Step 7: Commit**

```bash
git add -A && git commit -m "chore: scaffold project structure"
```

---

### Task 2: Shared Data Models

**Files:**
- Create: `backend/models/__init__.py` (empty)
- Create: `backend/models/schemas.py`

- [ ] **Step 1: Write backend/models/__init__.py**

```bash
touch backend/models/__init__.py
```

- [ ] **Step 2: Write backend/models/schemas.py**

```python
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
```

- [ ] **Step 3: Verify models import correctly**

```bash
cd backend && python -c "from models.schemas import ResearchRequest; print(ResearchRequest(query='test'))"
```
Expected: prints the model

- [ ] **Step 4: Commit**

```bash
git add backend/models/ && git commit -m "feat: add shared data models"
```

---

### Task 3: LLM Client

**Files:**
- Create: `backend/llm/__init__.py` (empty)
- Create: `backend/llm/client.py`

- [ ] **Step 1: Write backend/llm/__init__.py**

```bash
touch backend/llm/__init__.py
```

- [ ] **Step 2: Write backend/llm/client.py**

```python
from openai import AsyncOpenAI
from config import OPENAI_API_KEY, OPENAI_BASE_URL, OPENAI_MODEL


class LLMClient:
    def __init__(self):
        self.client = AsyncOpenAI(
            api_key=OPENAI_API_KEY,
            base_url=OPENAI_BASE_URL,
        )
        self.model = OPENAI_MODEL

    async def chat(self, messages: list[dict], max_tokens: int = 4096) -> str:
        response = await self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            max_tokens=max_tokens,
            temperature=0.3,
        )
        return response.choices[0].message.content or ""

    async def generate_plan(self, query: str) -> list[str]:
        prompt = f"""Break the following research question into 2-4 specific sub-questions.
Return ONLY a JSON array of strings, no other text.
Question: {query}"""
        raw = await self.chat([{"role": "user", "content": prompt}])
        import json
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            return [query]

    async def summarize(self, text: str, max_length: int = 500) -> str:
        prompt = f"Summarize the following in under {max_length} characters:\n\n{text}"
        return await self.chat([{"role": "user", "content": prompt}])
```

- [ ] **Step 3: Verify import**

```bash
cd backend && python -c "from llm.client import LLMClient; print('OK')"
```
Expected: prints "OK"

- [ ] **Step 4: Commit**

```bash
git add backend/llm/ && git commit -m "feat: add LLM client wrapper"
```

---

### Task 4: Tool Registry

**Files:**
- Create: `backend/agent/__init__.py` (empty)
- Create: `backend/agent/tools/__init__.py` (empty)
- Create: `backend/agent/tools/registry.py`

- [ ] **Step 1: Write init files**

```bash
touch backend/agent/__init__.py backend/agent/tools/__init__.py
```

- [ ] **Step 2: Write backend/agent/tools/registry.py**

```python
from dataclasses import dataclass, field
from typing import Callable, Any
import json


@dataclass
class Tool:
    name: str
    description: str
    parameters: dict  # JSON Schema
    handler: Callable[..., Any]


class ToolRegistry:
    def __init__(self):
        self._tools: dict[str, Tool] = {}

    def register(self, tool: Tool) -> None:
        self._tools[tool.name] = tool

    def get(self, name: str) -> Tool | None:
        return self._tools.get(name)

    def get_schema_for_llm(self) -> str:
        """Generate a text description of all tools for the LLM prompt."""
        lines = []
        for tool in self._tools.values():
            lines.append(f"- {tool.name}: {tool.description}")
            lines.append(f"  Parameters: {json.dumps(tool.parameters, ensure_ascii=False)}")
        return "\n".join(lines)

    async def execute(self, name: str, params: dict) -> str:
        tool = self.get(name)
        if tool is None:
            return f"Error: tool '{name}' not found. Available tools: {list(self._tools.keys())}"
        try:
            result = await tool.handler(**params)
            return str(result)
        except Exception as e:
            return f"Error executing '{name}': {e}"
```

- [ ] **Step 3: Verify import**

```bash
cd backend && python -c "from agent.tools.registry import ToolRegistry; print('OK')"
```
Expected: "OK"

- [ ] **Step 4: Commit**

```bash
git add backend/agent/ && git commit -m "feat: add tool registry"
```

---

### Task 5: Web Search Tool

**Files:**
- Create: `backend/agent/tools/web_search.py`

- [ ] **Step 1: Write backend/agent/tools/web_search.py**

```python
import httpx
from config import SEARCH_API_KEY, SEARCH_API_URL


async def web_search(query: str, num: int = 5) -> str:
    """Search the web and return formatted results."""
    if not SEARCH_API_KEY:
        return "Search API key not configured."

    try:
        async with httpx.AsyncClient(timeout=15.0) as client:
            response = await client.get(
                SEARCH_API_URL,
                params={
                    "q": query,
                    "api_key": SEARCH_API_KEY,
                    "num": num,
                    "engine": "google",
                },
            )
            response.raise_for_status()
            data = response.json()

        results = []
        organic = data.get("organic_results", [])
        for i, r in enumerate(organic[:num], 1):
            title = r.get("title", "No title")
            snippet = r.get("snippet", "No description")
            link = r.get("link", "No link")
            results.append(f"[{i}] {title}\n    Summary: {snippet}\n    URL: {link}")

        if not results:
            return f"No results found for '{query}'."

        return "\n\n".join(results)

    except httpx.TimeoutException:
        return f"Search timed out for query '{query}'. Please try a narrower query."
    except Exception as e:
        return f"Search failed for '{query}': {e}"
```

- [ ] **Step 2: Verify import**

```bash
cd backend && python -c "from agent.tools.web_search import web_search; print('OK')"
```
Expected: "OK"

- [ ] **Step 3: Commit**

```bash
git add backend/agent/tools/web_search.py && git commit -m "feat: add web search tool"
```

---

### Task 6: Fetch Page Tool

**Files:**
- Create: `backend/agent/tools/fetch_page.py`

- [ ] **Step 1: Write backend/agent/tools/fetch_page.py**

```python
import httpx
from bs4 import BeautifulSoup


async def fetch_page(url: str, max_chars: int = 5000) -> str:
    """Fetch a web page and return extracted text content."""
    try:
        async with httpx.AsyncClient(timeout=20.0) as client:
            response = await client.get(
                url,
                headers={
                    "User-Agent": "Mozilla/5.0 (compatible; ResearchAgent/1.0)",
                    "Accept": "text/html",
                },
                follow_redirects=True,
            )
            response.raise_for_status()

        soup = BeautifulSoup(response.text, "html.parser")

        for tag in soup(["script", "style", "nav", "footer", "header", "aside"]):
            tag.decompose()

        text = soup.get_text(separator="\n", strip=True)
        lines = [line.strip() for line in text.splitlines() if line.strip()]

        cleaned = "\n".join(lines)
        if len(cleaned) > max_chars:
            cleaned = cleaned[:max_chars] + f"\n\n[Truncated at {max_chars} characters]"

        title = soup.title.string.strip() if soup.title else url
        return f"Title: {title}\nURL: {url}\n\n{cleaned if cleaned else 'No text content extracted.'}"

    except httpx.TimeoutException:
        return f"Failed to fetch {url}: request timed out. Using search snippet instead."
    except Exception as e:
        return f"Failed to fetch {url}: {e}. Using search snippet instead."
```

- [ ] **Step 2: Verify import**

```bash
cd backend && python -c "from agent.tools.fetch_page import fetch_page; print('OK')"
```
Expected: "OK"

- [ ] **Step 3: Commit**

```bash
git add backend/agent/tools/fetch_page.py && git commit -m "feat: add fetch page tool"
```

---

### Task 7: Cross Check Tool

**Files:**
- Create: `backend/agent/tools/cross_check.py`

- [ ] **Step 1: Write backend/agent/tools/cross_check.py**

```python
from llm.client import LLMClient


async def cross_check(facts: str) -> str:
    """Cross-validate collected facts for consistency and conflicts."""
    llm = LLMClient()
    prompt = f"""You are a fact-checker. Review the following collected information for internal consistency.

Identify:
1. Consistency level (high/medium/low)
2. Any conflicting claims
3. Facts that are verified across multiple sources

Return your analysis in this JSON format:
{{
  "consistency": "high|medium|low",
  "conflicts": ["conflict description..."],
  "verified_facts": ["verified fact..."]
}}

Information to check:
{facts}"""

    raw = await llm.chat([{"role": "user", "content": prompt}], max_tokens=2000)

    import json
    try:
        result = json.loads(raw)
    except json.JSONDecodeError:
        result = {
            "consistency": "medium",
            "conflicts": ["Could not parse cross-check result"],
            "verified_facts": [],
        }

    conflicts = result.get("conflicts", [])
    verified = result.get("verified_facts", [])
    consistency = result.get("consistency", "medium")

    lines = [
        f"Cross-check consistency: {consistency}",
    ]
    if conflicts:
        lines.append(f"\nConflicts found ({len(conflicts)}):")
        for c in conflicts:
            lines.append(f"  - {c}")
    if verified:
        lines.append(f"\nVerified facts ({len(verified)}):")
        for f in verified:
            lines.append(f"  - {f}")
    if not conflicts and not verified:
        lines.append("No conflicts or verified facts identified.")

    return "\n".join(lines)
```

- [ ] **Step 2: Verify import**

```bash
cd backend && python -c "from agent.tools.cross_check import cross_check; print('OK')"
```
Expected: "OK"

- [ ] **Step 3: Commit**

```bash
git add backend/agent/tools/cross_check.py && git commit -m "feat: add cross-check tool"
```

---

### Task 8: Output Parser

**Files:**
- Create: `backend/agent/parser.py`

- [ ] **Step 1: Write backend/agent/parser.py**

```python
import json
import re
from dataclasses import dataclass


@dataclass
class ParsedOutput:
    thought: str
    action: str | None = None       # tool name
    action_params: dict | None = None
    is_final: bool = False
    final_answer: str | None = None
    parse_error: str | None = None


def parse(raw: str) -> ParsedOutput:
    thought = ""
    action = None
    action_params = None
    is_final = False
    final_answer = None
    parse_error = None

    # Extract THOUGHT
    thought_match = re.search(r"THOUGHT:\s*(.+?)(?=\n(?:ACTION|FINAL_ANSWER)|\Z)", raw, re.DOTALL | re.IGNORECASE)
    if thought_match:
        thought = thought_match.group(1).strip()

    # Check for FINAL_ANSWER
    if re.search(r"FINAL_ANSWER", raw, re.IGNORECASE):
        is_final = True
        fa_match = re.search(r"FINAL_ANSWER:\s*(.+)", raw, re.DOTALL | re.IGNORECASE)
        if fa_match:
            final_answer = fa_match.group(1).strip()
        else:
            # Everything after FINAL_ANSWER marker
            idx = raw.upper().find("FINAL_ANSWER")
            final_answer = raw[idx + len("FINAL_ANSWER"):].strip(": \n")

    # Extract ACTION
    action_match = re.search(r"ACTION:\s*(.+?)(?=\n(?:THOUGHT|OBSERVATION|FINAL_ANSWER)|\Z)", raw, re.DOTALL | re.IGNORECASE)
    if action_match and not is_final:
        action_str = action_match.group(1).strip()

        # Action format: <tool_name>|<json_params>
        if "|" in action_str:
            parts = action_str.split("|", 1)
            action = parts[0].strip()
            try:
                action_params = json.loads(parts[1].strip())
            except json.JSONDecodeError:
                parse_error = f"Failed to parse action params JSON: {parts[1][:100]}"
                action_params = {}

    # If nothing was parsed, treat as parse failure
    if not thought and not action and not is_final:
        parse_error = "Could not parse THOUGHT, ACTION, or FINAL_ANSWER from LLM output."
        thought = raw[:500]

    return ParsedOutput(
        thought=thought,
        action=action,
        action_params=action_params,
        is_final=is_final,
        final_answer=final_answer,
        parse_error=parse_error,
    )
```

- [ ] **Step 2: Run quick smoke test in Python**

```bash
cd backend && python -c "
from agent.parser import parse
# Test valid THOUGHT + ACTION
out = parse('THOUGHT: need to search\nACTION: web_search|{\"query\": \"AI trends\"}')
assert out.thought == 'need to search'
assert out.action == 'web_search'
assert out.action_params == {'query': 'AI trends'}
assert not out.is_final
# Test FINAL_ANSWER
out2 = parse('THOUGHT: done\nFINAL_ANSWER: Here is the report...')
assert out2.is_final
assert 'Here is the report' in out2.final_answer
print('All assertions passed')
"
```
Expected: "All assertions passed"

- [ ] **Step 3: Commit**

```bash
git add backend/agent/parser.py && git commit -m "feat: add output parser"
```

---

### Task 9: Memory Manager

**Files:**
- Create: `backend/agent/memory.py`

- [ ] **Step 1: Write backend/agent/memory.py**

```python
from models.schemas import Step, StepType


class MemoryManager:
    def __init__(self, max_history_chars: int = 8000):
        self.steps: list[Step] = []
        self.key_facts: list[str] = []
        self.max_history_chars = max_history_chars

    def add_step(self, step: Step) -> None:
        self.steps.append(step)

    def add_key_fact(self, fact: str) -> None:
        if fact not in self.key_facts:
            self.key_facts.append(fact)

    def get_steps_for_context(self) -> str:
        """Build context string from steps, compressing if needed."""
        lines = []
        total_chars = 0

        for step in self.steps:
            step_str = self._format_step(step)
            total_chars += len(step_str)
            lines.append(step_str)

        if total_chars <= self.max_history_chars:
            return "\n".join(lines)

        # Compress: summarize early steps, keep recent ones
        recent = []
        recent_chars = 0
        for line in reversed(lines):
            if recent_chars + len(line) > self.max_history_chars // 2:
                break
            recent.insert(0, line)
            recent_chars += len(line)

        summary = self._build_summary(self.steps[: len(self.steps) - len(recent)])
        return f"{summary}\n\n--- Recent Steps ---\n" + "\n".join(recent)

    def _format_step(self, step: Step) -> str:
        base = f"[Round {step.round}] {step.type.upper()}: {step.content}"
        if step.tool_name:
            base += f" (tool: {step.tool_name})"
        return base

    def _build_summary(self, early_steps: list[Step]) -> str:
        """Build a compressed summary of early steps, preserving key facts."""
        facts_str = "\n".join(f"- {f}" for f in self.key_facts) if self.key_facts else "(none)"
        action_count = sum(1 for s in early_steps if s.type == StepType.ACTION)
        return (
            f"[Compressed summary of {len(early_steps)} early steps]\n"
            f"Early actions taken: {action_count}\n"
            f"Key facts collected:\n{facts_str}"
        )
```

- [ ] **Step 2: Verify import**

```bash
cd backend && python -c "from agent.memory import MemoryManager; print('OK')"
```
Expected: "OK"

- [ ] **Step 3: Commit**

```bash
git add backend/agent/memory.py && git commit -m "feat: add memory manager"
```

---

### Task 10: ReAct Loop

**Files:**
- Create: `backend/agent/loop.py`

- [ ] **Step 1: Write backend/agent/loop.py**

```python
import json
from typing import AsyncGenerator
from config import MAX_ROUNDS
from models.schemas import Step, StepType, StepEvent
from agent.tools.registry import ToolRegistry
from agent.parser import parse
from agent.memory import MemoryManager
from llm.client import LLMClient


SYSTEM_PROMPT = """You are a deep research agent. Your goal is to thoroughly investigate a topic and produce a comprehensive, well-sourced answer.

For each round, respond in this exact format:

THOUGHT: <Analyze what you know so far. Is the information sufficient? What is the next logical step to deepen the research?>

Then either:
ACTION: <tool_name>|<json_params>

Or when you have enough information:
FINAL_ANSWER: <Your comprehensive research report in Markdown format, with citations>

Available tools:
{tool_descriptions}

Rules:
- Search before reading: use web_search first, then fetch_page for the most promising URLs
- Verify across sources: don't rely on a single source for important claims
- Cross-check facts before concluding
- Cite sources inline like [Source: URL]
- If search fails or returns nothing useful, adapt your strategy or report honestly what you couldn't find"""


class AgentLoop:
    def __init__(self, tool_registry: ToolRegistry, llm: LLMClient):
        self.tool_registry = tool_registry
        self.llm = llm
        self.memory = MemoryManager()

    async def run(self, query: str) -> AsyncGenerator[StepEvent, None]:
        messages = [
            {
                "role": "system",
                "content": SYSTEM_PROMPT.format(
                    tool_descriptions=self.tool_registry.get_schema_for_llm()
                ),
            },
            {"role": "user", "content": f"Research question: {query}"},
        ]

        for round_num in range(1, MAX_ROUNDS + 1):
            # Get LLM response
            raw_output = await self.llm.chat(messages)

            # Parse
            parsed = parse(raw_output)

            # Emit thought
            if parsed.thought:
                thought_step = StepEvent(
                    type="thought",
                    round=round_num,
                    content=parsed.thought,
                )
                self.memory.add_step(Step(
                    round=round_num,
                    type=StepType.THOUGHT,
                    content=parsed.thought,
                ))
                yield thought_step

            # Handle parse error: retry once
            if parsed.parse_error and not parsed.is_final:
                messages.append({"role": "assistant", "content": raw_output})
                messages.append({
                    "role": "user",
                    "content": f"Your output format was invalid: {parsed.parse_error}. "
                               f"Please follow the format exactly: THOUGHT, then ACTION or FINAL_ANSWER.",
                })
                continue

            # Execute action
            if parsed.action:
                action_step = StepEvent(
                    type="action",
                    round=round_num,
                    content=f"Calling {parsed.action}",
                    tool_name=parsed.action,
                    tool_params=parsed.action_params,
                )
                self.memory.add_step(Step(
                    round=round_num,
                    type=StepType.ACTION,
                    content=f"Calling {parsed.action} with {json.dumps(parsed.action_params or {})}",
                    tool_name=parsed.action,
                    tool_params=parsed.action_params,
                ))
                yield action_step

                # Execute tool
                observation = await self.tool_registry.execute(
                    parsed.action,
                    parsed.action_params or {},
                )

                obs_step = StepEvent(
                    type="observation",
                    round=round_num,
                    content=observation[:2000],
                )
                self.memory.add_step(Step(
                    round=round_num,
                    type=StepType.OBSERVATION,
                    content=observation[:2000],
                ))
                yield obs_step

                # Append to messages
                messages.append({"role": "assistant", "content": raw_output})
                messages.append({"role": "user", "content": f"OBSERVATION:\n{observation}"})

                # Extract potential key facts
                if len(observation) > 200:
                    self.memory.add_key_fact(observation[:300])

            # Handle final answer
            if parsed.is_final:
                final_content = parsed.final_answer or parsed.thought
                yield StepEvent(
                    type="observation",
                    round=round_num,
                    content=final_content,
                )
                return

        # Max rounds reached: force summary
        final_content = await self.llm.chat([
            {"role": "user", "content": f"Write a research report based on the following findings. Include citations.\n\nFindings:\n{self.memory.get_steps_for_context()}"}
        ])
        yield StepEvent(
            type="observation",
            round=MAX_ROUNDS,
            content=final_content,
        )
```

- [ ] **Step 2: Verify import**

```bash
cd backend && python -c "from agent.loop import AgentLoop; print('OK')"
```
Expected: "OK"

- [ ] **Step 3: Commit**

```bash
git add backend/agent/loop.py && git commit -m "feat: add ReAct agent loop"
```

---

### Task 11: Research Orchestrator

**Files:**
- Create: `backend/agent/orchestrator.py`

- [ ] **Step 1: Write backend/agent/orchestrator.py**

```python
from typing import AsyncGenerator
from models.schemas import StepEvent, PlanEvent, CrossCheckEvent, ReportEvent
from agent.loop import AgentLoop
from agent.tools.registry import ToolRegistry
from agent.tools.web_search import web_search
from agent.tools.fetch_page import fetch_page
from agent.tools.cross_check import cross_check
from llm.client import LLMClient


class ResearchOrchestrator:
    def __init__(self):
        self.llm = LLMClient()
        self.registry = ToolRegistry()
        self._setup_tools()

    def _setup_tools(self):
        from agent.tools.registry import Tool
        self.registry.register(Tool(
            name="web_search",
            description="Search the web for information. Returns titles, summaries, and URLs.",
            parameters={"query": {"type": "string", "description": "Search query"}},
            handler=web_search,
        ))
        self.registry.register(Tool(
            name="fetch_page",
            description="Fetch and extract text content from a specific URL. Use this to read articles in depth.",
            parameters={"url": {"type": "string", "description": "Full URL to fetch"}},
            handler=fetch_page,
        ))
        self.registry.register(Tool(
            name="cross_check",
            description="Cross-validate collected facts for consistency. Use before writing final answer.",
            parameters={"facts": {"type": "string", "description": "All collected facts to validate"}},
            handler=cross_check,
        ))

    async def research(self, query: str) -> AsyncGenerator[dict, None]:
        # Phase 1: Generate research plan
        sub_questions = await self.llm.generate_plan(query)
        yield PlanEvent(sub_questions=sub_questions).model_dump()

        # Phase 2: Run ReAct loop
        loop = AgentLoop(self.registry, self.llm)
        all_observations = []

        async for step in loop.run(query):
            if step.type == "observation":
                all_observations.append(step.content)
            yield step.model_dump()

        # Phase 3: Cross-check
        facts = "\n".join(all_observations[-5:])  # Last 5 observations
        cross_result = await cross_check(facts)
        cross_event = CrossCheckEvent(
            consistency="medium",
            conflicts=[],
            verified_facts=[],
        )
        yield cross_event.model_dump()

        # Phase 4: Generate final report
        final_report = await self.llm.chat([
            {
                "role": "system",
                "content": "Write a well-structured research report in Markdown based on the findings. Include a title, section headers, citations as [Source: URL], and a sources list at the end.",
            },
            {
                "role": "user",
                "content": f"Research question: {query}\n\nSub-questions: {', '.join(sub_questions)}\n\nCollected findings:\n{facts}\n\nCross-check result:\n{cross_result}",
            },
        ])

        sources = self._extract_sources(final_report)
        yield ReportEvent(markdown=final_report, sources=sources).model_dump()

    def _extract_sources(self, report: str) -> list[str]:
        import re
        urls = re.findall(r'https?://[^\s\)\]]+', report)
        return list(dict.fromkeys(urls))  # dedupe preserving order
```

- [ ] **Step 2: Verify import**

```bash
cd backend && python -c "from agent.orchestrator import ResearchOrchestrator; print('OK')"
```
Expected: "OK"

- [ ] **Step 3: Commit**

```bash
git add backend/agent/orchestrator.py && git commit -m "feat: add research orchestrator"
```

---

### Task 12: Database Layer

**Files:**
- Create: `backend/db/__init__.py` (empty)
- Create: `backend/db/connection.py`
- Create: `backend/db/repository.py`

- [ ] **Step 1: Write init file**

```bash
touch backend/db/__init__.py
```

- [ ] **Step 2: Write backend/db/connection.py**

```python
import aiosqlite
from config import DB_PATH

_connection: aiosqlite.Connection | None = None


async def get_db() -> aiosqlite.Connection:
    global _connection
    if _connection is None:
        _connection = await aiosqlite.connect(DB_PATH)
        _connection.row_factory = aiosqlite.Row
        await _connection.execute("PRAGMA journal_mode=WAL")
        await _init_tables(_connection)
    return _connection


async def _init_tables(db: aiosqlite.Connection):
    await db.execute("""
        CREATE TABLE IF NOT EXISTS sessions (
            id TEXT PRIMARY KEY,
            query TEXT NOT NULL,
            status TEXT NOT NULL DEFAULT 'running',
            created_at TEXT NOT NULL DEFAULT (datetime('now')),
            report TEXT,
            steps_json TEXT DEFAULT '[]'
        )
    """)
    await db.commit()


async def close_db():
    global _connection
    if _connection:
        await _connection.close()
        _connection = None
```

- [ ] **Step 3: Write backend/db/repository.py**

```python
import json
from db.connection import get_db


async def create_session(session_id: str, query: str) -> None:
    db = await get_db()
    await db.execute(
        "INSERT INTO sessions (id, query, status) VALUES (?, ?, 'running')",
        (session_id, query),
    )
    await db.commit()


async def update_session(session_id: str, status: str, report: str | None = None, steps_json: str = "[]") -> None:
    db = await get_db()
    await db.execute(
        "UPDATE sessions SET status = ?, report = ?, steps_json = ? WHERE id = ?",
        (status, report, steps_json, session_id),
    )
    await db.commit()


async def get_all_sessions() -> list[dict]:
    db = await get_db()
    cursor = await db.execute(
        "SELECT id, query, status, created_at FROM sessions ORDER BY created_at DESC LIMIT 50"
    )
    rows = await cursor.fetchall()
    return [dict(row) for row in rows]


async def get_session(session_id: str) -> dict | None:
    db = await get_db()
    cursor = await db.execute("SELECT * FROM sessions WHERE id = ?", (session_id,))
    row = await cursor.fetchone()
    return dict(row) if row else None
```

- [ ] **Step 4: Verify imports**

```bash
cd backend && python -c "from db.repository import create_session; print('OK')"
```
Expected: "OK"

- [ ] **Step 5: Commit**

```bash
git add backend/db/ && git commit -m "feat: add database layer"
```

---

### Task 13: FastAPI Routes

**Files:**
- Create: `backend/routes/__init__.py` (empty)
- Create: `backend/routes/research.py`

- [ ] **Step 1: Write init file**

```bash
touch backend/routes/__init__.py
```

- [ ] **Step 2: Write backend/routes/research.py**

```python
import json
import uuid
from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from models.schemas import ResearchRequest
from agent.orchestrator import ResearchOrchestrator
from db.repository import create_session, update_session, get_all_sessions, get_session

router = APIRouter(prefix="/api")


@router.post("/research")
async def start_research(req: ResearchRequest):
    session_id = str(uuid.uuid4())
    await create_session(session_id, req.query)

    async def generate():
        collected_steps = []
        report = None
        try:
            orchestrator = ResearchOrchestrator()
            async for event in orchestrator.research(req.query):
                # Collect steps for persistence
                if "type" in event and event.get("type") in ("thought", "action", "observation"):
                    collected_steps.append(event)
                if "markdown" in event:
                    report = event["markdown"]

                yield f"event: {_event_type(event)}\ndata: {json.dumps(event, ensure_ascii=False)}\n\n"

            await update_session(session_id, "completed", report, json.dumps(collected_steps))
        except Exception as e:
            error_event = {"type": "error", "message": str(e)}
            yield f"event: error\ndata: {json.dumps(error_event, ensure_ascii=False)}\n\n"
            await update_session(session_id, "error", steps_json=json.dumps(collected_steps))

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


def _event_type(event: dict) -> str:
    if "sub_questions" in event:
        return "plan"
    if "consistency" in event:
        return "cross_check"
    if "markdown" in event:
        return "report"
    if event.get("type") in ("thought", "action", "observation"):
        return "step"
    return "message"


@router.get("/history")
async def list_history():
    sessions = await get_all_sessions()
    return {"sessions": sessions}


@router.get("/history/{session_id}")
async def get_history(session_id: str):
    session = await get_session(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Session not found")
    if session.get("steps_json"):
        session["steps"] = json.loads(session["steps_json"])
    del session["steps_json"]
    return session
```

- [ ] **Step 3: Verify imports**

```bash
cd backend && python -c "from routes.research import router; print('OK')"
```
Expected: "OK"

- [ ] **Step 4: Commit**

```bash
git add backend/routes/ && git commit -m "feat: add FastAPI routes with SSE streaming"
```

---

### Task 14: FastAPI Entrypoint

**Files:**
- Create: `backend/main.py`

- [ ] **Step 1: Write backend/main.py**

```python
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from routes.research import router
from db.connection import close_db


@asynccontextmanager
async def lifespan(app: FastAPI):
    yield
    await close_db()


app = FastAPI(title="Deep Research Agent", version="0.1.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router)


@app.get("/health")
async def health():
    return {"status": "ok"}
```

- [ ] **Step 2: Start backend and verify health endpoint**

```bash
cd backend && python -m uvicorn main:app --host 0.0.0.0 --port 8000 &
sleep 2
curl http://localhost:8000/health
```
Expected: `{"status":"ok"}`

- [ ] **Step 3: Stop the server and commit**

```bash
kill %1
git add backend/main.py && git commit -m "feat: add FastAPI entrypoint"
```

---

### Task 15: Frontend SSE Composable

**Files:**
- Create: `frontend/src/composables/useSSE.js`

- [ ] **Step 1: Write frontend/src/composables/useSSE.js**

```javascript
import { ref } from 'vue'

export function useSSE() {
  const isResearching = ref(false)
  const subQuestions = ref([])
  const steps = ref([])
  const crossCheck = ref(null)
  const report = ref(null)
  const error = ref(null)
  const eventSource = ref(null)

  function startResearch(query) {
    // Reset state
    isResearching.value = true
    subQuestions.value = []
    steps.value = []
    crossCheck.value = null
    report.value = null
    error.value = null

    fetch('/api/research', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ query }),
    }).then(async (response) => {
      const reader = response.body.getReader()
      const decoder = new TextDecoder()
      let buffer = ''

      while (true) {
        const { done, value } = await reader.read()
        if (done) break

        buffer += decoder.decode(value, { stream: true })
        const lines = buffer.split('\n')
        buffer = lines.pop() || ''

        let currentEvent = ''
        for (const line of lines) {
          if (line.startsWith('event: ')) {
            currentEvent = line.slice(7).trim()
          } else if (line.startsWith('data: ')) {
            try {
              const data = JSON.parse(line.slice(6))
              handleEvent(currentEvent, data)
            } catch (e) {
              // Skip unparseable lines
            }
          }
        }
      }

      isResearching.value = false
    }).catch((err) => {
      error.value = err.message || 'Connection failed'
      isResearching.value = false
    })
  }

  function handleEvent(event, data) {
    switch (event) {
      case 'plan':
        subQuestions.value = data.sub_questions || []
        break
      case 'step':
        steps.value.push(data)
        break
      case 'cross_check':
        crossCheck.value = data
        break
      case 'report':
        report.value = data
        break
      case 'error':
        error.value = data.message || 'Unknown error'
        isResearching.value = false
        break
    }
  }

  return {
    isResearching,
    subQuestions,
    steps,
    crossCheck,
    report,
    error,
    startResearch,
  }
}
```

- [ ] **Step 2: Commit**

```bash
git add frontend/src/composables/ && git commit -m "feat: add SSE composable"
```

---

### Task 16: Frontend Components

**Files:**
- Create: `frontend/src/components/StepTimeline.vue`
- Create: `frontend/src/components/ReportView.vue`
- Create: `frontend/src/components/ChatPanel.vue`

- [ ] **Step 1: Write frontend/src/components/StepTimeline.vue**

```vue
<template>
  <div class="step-timeline">
    <h3 v-if="steps.length">Research Steps</h3>
    <div v-for="(step, i) in steps" :key="i" :class="['step-card', step.type]">
      <div class="step-header">
        <span class="step-badge">{{ step.type.toUpperCase() }}</span>
        <span class="step-round">Round {{ step.round }}</span>
      </div>
      <div class="step-content">
        <template v-if="step.type === 'action' && step.tool_name">
          <strong>{{ step.tool_name }}</strong>
          <pre v-if="step.tool_params">{{ JSON.stringify(step.tool_params, null, 2) }}</pre>
        </template>
        <template v-else>
          {{ step.content?.slice(0, 300) }}{{ step.content?.length > 300 ? '...' : '' }}
        </template>
      </div>
    </div>
    <div v-if="!steps.length && isResearching" class="loading">Starting research...</div>
  </div>
</template>

<script setup>
defineProps({
  steps: { type: Array, default: () => [] },
  isResearching: { type: Boolean, default: false },
})
</script>

<style scoped>
.step-timeline { padding: 0.5rem 0; }
.step-card {
  border-left: 3px solid #ddd;
  padding: 0.5rem 1rem;
  margin-bottom: 0.5rem;
  border-radius: 0 6px 6px 0;
}
.step-card.thought { border-color: #6366f1; background: #f5f3ff; }
.step-card.action { border-color: #f59e0b; background: #fffbeb; }
.step-card.observation { border-color: #10b981; background: #f0fdf4; }
.step-header { display: flex; gap: 0.5rem; align-items: center; margin-bottom: 0.25rem; }
.step-badge { font-size: 0.7rem; font-weight: 700; padding: 2px 6px; border-radius: 4px; }
.thought .step-badge { background: #e0e7ff; color: #4338ca; }
.action .step-badge { background: #fef3c7; color: #b45309; }
.observation .step-badge { background: #d1fae5; color: #065f46; }
.step-round { font-size: 0.75rem; color: #888; }
.step-content { font-size: 0.85rem; line-height: 1.5; white-space: pre-wrap; }
pre { background: #1e1e1e; color: #d4d4d4; padding: 0.5rem; border-radius: 4px; font-size: 0.75rem; overflow-x: auto; }
.loading { color: #888; font-style: italic; padding: 1rem; }
</style>
```

- [ ] **Step 2: Write frontend/src/components/ReportView.vue**

```vue
<template>
  <div class="report-view" v-if="report">
    <h3>Research Report</h3>
    <div class="report-body" v-html="rendered"></div>
    <div class="sources" v-if="report.sources?.length">
      <h4>Sources</h4>
      <ul>
        <li v-for="(s, i) in report.sources" :key="i">
          <a :href="s" target="_blank">{{ s }}</a>
        </li>
      </ul>
    </div>
  </div>
</template>

<script setup>
import { computed } from 'vue'
import { marked } from 'marked'

const props = defineProps({
  report: { type: Object, default: null },
})

const rendered = computed(() => {
  if (!props.report?.markdown) return ''
  return marked(props.report.markdown)
})
</script>

<style scoped>
.report-view { padding: 1rem 0; }
.report-body { line-height: 1.7; }
.report-body :deep(h1) { font-size: 1.5rem; margin-top: 1rem; }
.report-body :deep(h2) { font-size: 1.25rem; margin-top: 1rem; }
.report-body :deep(h3) { font-size: 1.1rem; margin-top: 0.75rem; }
.report-body :deep(p) { margin: 0.5rem 0; }
.report-body :deep(code) { background: #f1f5f9; padding: 2px 6px; border-radius: 3px; font-size: 0.85rem; }
.sources { margin-top: 1.5rem; padding-top: 1rem; border-top: 1px solid #e5e7eb; }
.sources ul { list-style: none; padding: 0; }
.sources li { margin: 0.25rem 0; font-size: 0.8rem; word-break: break-all; }
</style>
```

- [ ] **Step 3: Write frontend/src/components/ChatPanel.vue**

```vue
<template>
  <div class="chat-panel">
    <div class="messages">
      <div class="empty-state" v-if="!report && !isResearching">
        <h2>Deep Research Agent</h2>
        <p>Ask any research question and I'll investigate it thoroughly.</p>
        <div class="examples">
          <button v-for="q in exampleQuestions" :key="q" @click="$emit('ask', q)" class="example-btn">
            {{ q }}
          </button>
        </div>
      </div>

      <div v-if="subQuestions.length" class="plan-card">
        <h4>Research Plan</h4>
        <ol>
          <li v-for="(q, i) in subQuestions" :key="i">{{ q }}</li>
        </ol>
      </div>

      <div v-if="crossCheck" class="cross-check-card">
        <h4>Cross-Check</h4>
        <p>Consistency: <strong>{{ crossCheck.consistency }}</strong></p>
        <ul v-if="crossCheck.conflicts?.length">
          <li v-for="(c, i) in crossCheck.conflicts" :key="i">⚠ {{ c }}</li>
        </ul>
      </div>

      <ReportView :report="report" />

      <div v-if="error" class="error-card">
        <strong>Error:</strong> {{ error }}
      </div>

      <div v-if="isResearching && !report" class="researching-indicator">
        <span class="pulse"></span> Researching... ({{ steps.length }} steps)
      </div>
    </div>

    <form @submit.prevent="handleSubmit" class="input-area">
      <input
        v-model="query"
        type="text"
        placeholder="Enter your research question..."
        :disabled="isResearching"
      />
      <button type="submit" :disabled="isResearching || !query.trim()">Research</button>
    </form>
  </div>
</template>

<script setup>
import { ref } from 'vue'
import ReportView from './ReportView.vue'

const props = defineProps({
  isResearching: Boolean,
  steps: { type: Array, default: () => [] },
  subQuestions: { type: Array, default: () => [] },
  crossCheck: { type: Object, default: null },
  report: { type: Object, default: null },
  error: { type: String, default: null },
})

const emit = defineEmits(['ask'])

const query = ref('')

const exampleQuestions = [
  'What are the main trends in AI agent development in 2025?',
  'Compare React, Vue, and Svelte for building large-scale applications',
  'What is the current state of quantum computing research?',
]

function handleSubmit() {
  if (query.value.trim()) {
    emit('ask', query.value.trim())
    query.value = ''
  }
}
</script>

<style scoped>
.chat-panel { display: flex; flex-direction: column; height: 100vh; }
.messages { flex: 1; overflow-y: auto; padding: 1.5rem; }
.empty-state { text-align: center; padding: 3rem 1rem; }
.empty-state h2 { font-size: 1.5rem; margin-bottom: 0.5rem; }
.empty-state p { color: #666; margin-bottom: 1.5rem; }
.examples { display: flex; flex-direction: column; gap: 0.5rem; align-items: center; }
.example-btn { background: #f1f5f9; border: 1px solid #e2e8f0; padding: 0.5rem 1rem; border-radius: 8px; cursor: pointer; font-size: 0.85rem; max-width: 500px; text-align: left; }
.example-btn:hover { background: #e2e8f0; }
.plan-card, .cross-check-card { background: #f8fafc; border: 1px solid #e2e8f0; border-radius: 8px; padding: 1rem; margin-bottom: 1rem; }
.error-card { background: #fef2f2; border: 1px solid #fecaca; color: #dc2626; padding: 1rem; border-radius: 8px; margin-bottom: 1rem; }
.researching-indicator { display: flex; align-items: center; gap: 0.5rem; color: #6366f1; padding: 1rem; }
.pulse { width: 10px; height: 10px; background: #6366f1; border-radius: 50%; animation: pulse 1.5s infinite; }
@keyframes pulse { 0%, 100% { opacity: 1; } 50% { opacity: 0.3; } }
.input-area { display: flex; gap: 0.5rem; padding: 1rem; border-top: 1px solid #e5e7eb; }
.input-area input { flex: 1; padding: 0.75rem; border: 1px solid #d1d5db; border-radius: 8px; font-size: 0.95rem; }
.input-area button { padding: 0.75rem 1.5rem; background: #6366f1; color: white; border: none; border-radius: 8px; cursor: pointer; font-weight: 600; }
.input-area button:disabled { background: #a5b4fc; cursor: not-allowed; }
</style>
```

- [ ] **Step 4: Commit**

```bash
git add frontend/src/components/ && git commit -m "feat: add Vue components"
```

---

### Task 17: Frontend App and Main

**Files:**
- Modify: `frontend/src/App.vue`
- Modify: `frontend/src/style.css`

- [ ] **Step 1: Overwrite frontend/src/App.vue**

```vue
<template>
  <div class="app-layout">
    <aside class="sidebar">
      <StepTimeline :steps="steps" :isResearching="isResearching" />
    </aside>
    <main class="main-content">
      <ChatPanel
        :isResearching="isResearching"
        :steps="steps"
        :subQuestions="subQuestions"
        :crossCheck="crossCheck"
        :report="report"
        :error="error"
        @ask="handleAsk"
      />
    </main>
  </div>
</template>

<script setup>
import ChatPanel from './components/ChatPanel.vue'
import StepTimeline from './components/StepTimeline.vue'
import { useSSE } from './composables/useSSE.js'

const {
  isResearching,
  subQuestions,
  steps,
  crossCheck,
  report,
  error,
  startResearch,
} = useSSE()

function handleAsk(query) {
  startResearch(query)
}
</script>

<style scoped>
.app-layout { display: flex; height: 100vh; }
.sidebar { width: 380px; border-right: 1px solid #e5e7eb; overflow-y: auto; padding: 1rem; background: #fafafa; flex-shrink: 0; }
.main-content { flex: 1; overflow: hidden; }
@media (max-width: 768px) {
  .app-layout { flex-direction: column; }
  .sidebar { width: 100%; max-height: 40vh; border-right: none; border-bottom: 1px solid #e5e7eb; }
}
</style>
```

- [ ] **Step 2: Overwrite frontend/src/style.css**

```css
*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }
body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; color: #1e293b; background: #fff; }
button { font-family: inherit; }
```

- [ ] **Step 3: Configure Vite proxy for API**

Read `frontend/vite.config.js` and add proxy config:

```javascript
import { defineConfig } from 'vite'
import vue from '@vitejs/plugin-vue'

export default defineConfig({
  plugins: [vue()],
  server: {
    proxy: {
      '/api': {
        target: 'http://localhost:8000',
        changeOrigin: true,
      },
    },
  },
})
```

- [ ] **Step 4: Start frontend dev server and verify**

```bash
cd frontend && npm run dev &
sleep 3
curl http://localhost:5173 | head -20
```
Expected: HTML of the Vue app

- [ ] **Step 5: Kill dev server and commit**

```bash
kill %1
git add frontend/src/ && git commit -m "feat: wire up App with composables"
```

---

### Task 18: Integration & Final Verification

**Files:**
- Create: `.env.example`

- [ ] **Step 1: Write .env.example**

```
OPENAI_API_KEY=sk-your-key-here
OPENAI_BASE_URL=https://api.openai.com/v1
OPENAI_MODEL=gpt-4o
SEARCH_API_KEY=your-serpapi-key
SEARCH_API_URL=https://serpapi.com/search
MAX_ROUNDS=10
MAX_TOKENS=8000
DB_PATH=research.db
```

- [ ] **Step 2: Full integration test — start both servers**

```bash
# Terminal 1: Backend
cd backend && cp .env.example .env && python -m uvicorn main:app --host 0.0.0.0 --port 8000

# Terminal 2: Frontend
cd frontend && npm run dev
```

Expected: Backend on :8000/health returns `{"status":"ok"}`, frontend on :5173 shows the app.

- [ ] **Step 3: Commit**

```bash
git add .env.example && git commit -m "chore: add env template and finalize integration"
```

---

### Task 19: Tests

**Files:**
- Create: `tests/__init__.py` (empty)
- Create: `tests/test_parser.py`
- Create: `tests/test_registry.py`
- Create: `tests/test_memory.py`
- Create: `tests/test_loop.py`

- [ ] **Step 1: Write tests/__init__.py**

```bash
touch tests/__init__.py
```

- [ ] **Step 2: Write tests/test_parser.py**

```python
import pytest
from agent.parser import parse


def test_parse_thought_and_action():
    raw = 'THOUGHT: need to search\nACTION: web_search|{"query": "test"}'
    out = parse(raw)
    assert out.thought == "need to search"
    assert out.action == "web_search"
    assert out.action_params == {"query": "test"}
    assert not out.is_final
    assert out.parse_error is None


def test_parse_final_answer():
    raw = 'THOUGHT: done researching\nFINAL_ANSWER: Here is the report...'
    out = parse(raw)
    assert out.is_final
    assert "Here is the report" in out.final_answer


def test_parse_bad_json_params():
    raw = 'THOUGHT: test\nACTION: web_search|{bad json}'
    out = parse(raw)
    assert out.action == "web_search"
    assert out.parse_error is not None


def test_parse_unparseable():
    raw = "just some random text without proper format"
    out = parse(raw)
    assert out.parse_error is not None


def test_parse_lowercase():
    raw = 'thought: need to search\naction: web_search|{"query": "test"}'
    out = parse(raw)
    assert out.action == "web_search"


def test_parse_multiline_thought():
    raw = 'THOUGHT: line one\nline two\nline three\nACTION: web_search|{"q": "x"}'
    out = parse(raw)
    assert "line one" in out.thought
    assert out.action == "web_search"
```

- [ ] **Step 3: Write tests/test_registry.py**

```python
import pytest
from agent.tools.registry import Tool, ToolRegistry


async def echo_handler(text: str = "") -> str:
    return f"echo: {text}"


def test_register_and_get():
    reg = ToolRegistry()
    tool = Tool(name="echo", description="Echoes text", parameters={"text": {"type": "string"}}, handler=echo_handler)
    reg.register(tool)
    assert reg.get("echo") is tool
    assert reg.get("nonexistent") is None


def test_schema_generation():
    reg = ToolRegistry()
    reg.register(Tool(name="echo", description="Echoes text", parameters={"text": {"type": "string"}}, handler=echo_handler))
    schema = reg.get_schema_for_llm()
    assert "echo" in schema
    assert "Echoes text" in schema


@pytest.mark.asyncio
async def test_execute():
    reg = ToolRegistry()
    reg.register(Tool(name="echo", description="Echoes text", parameters={"text": {"type": "string"}}, handler=echo_handler))
    result = await reg.execute("echo", {"text": "hello"})
    assert result == "echo: hello"


@pytest.mark.asyncio
async def test_execute_unknown_tool():
    reg = ToolRegistry()
    result = await reg.execute("unknown", {})
    assert "not found" in result
```

- [ ] **Step 4: Write tests/test_memory.py**

```python
from agent.memory import MemoryManager
from models.schemas import Step, StepType


def test_add_and_retrieve_steps():
    mm = MemoryManager(max_history_chars=1000)
    mm.add_step(Step(round=1, type=StepType.THOUGHT, content="test thought"))
    mm.add_step(Step(round=1, type=StepType.OBSERVATION, content="test observation"))
    ctx = mm.get_steps_for_context()
    assert "test thought" in ctx
    assert "test observation" in ctx


def test_key_facts_dedup():
    mm = MemoryManager()
    mm.add_key_fact("fact A")
    mm.add_key_fact("fact A")
    mm.add_key_fact("fact B")
    assert len(mm.key_facts) == 2


def test_compression_kicks_in():
    mm = MemoryManager(max_history_chars=10)
    for i in range(5):
        mm.add_step(Step(round=i, type=StepType.OBSERVATION, content=f"very long content that fills up space quickly {i}"))
    ctx = mm.get_steps_for_context()
    assert "[Compressed summary" in ctx
```

- [ ] **Step 5: Write tests/test_loop.py**

```python
import pytest
from unittest.mock import AsyncMock, patch
from agent.loop import AgentLoop
from agent.tools.registry import ToolRegistry, Tool


async def dummy_search(query: str = "") -> str:
    return f"Search results for: {query}"


@pytest.mark.asyncio
async def test_loop_stops_on_final_answer():
    reg = ToolRegistry()
    reg.register(Tool(name="web_search", description="Search", parameters={"query": {"type": "string"}}, handler=dummy_search))

    mock_llm = AsyncMock()
    mock_llm.chat.side_effect = [
        "THOUGHT: searching\nACTION: web_search|{\"query\": \"test\"}",
        "THOUGHT: done\nFINAL_ANSWER: The report content",
    ]

    loop = AgentLoop(reg, mock_llm)
    events = []
    async for event in loop.run("test query"):
        events.append(event)

    assert len(events) >= 3  # thought, action, observation, thought, final
    assert events[-1].type == "observation"


@pytest.mark.asyncio
async def test_loop_parse_error_retry():
    reg = ToolRegistry()
    reg.register(Tool(name="web_search", description="Search", parameters={"query": {"type": "string"}}, handler=dummy_search))

    mock_llm = AsyncMock()
    mock_llm.chat.side_effect = [
        "garbled output without proper format",
        "THOUGHT: retry\nFINAL_ANSWER: Fixed report",
    ]

    loop = AgentLoop(reg, mock_llm)
    events = []
    async for event in loop.run("test query"):
        events.append(event)

    assert mock_llm.chat.call_count >= 2
```

- [ ] **Step 6: Run all tests**

```bash
cd backend && python -m pytest tests/ -v
```
Expected: all tests pass

- [ ] **Step 7: Commit**

```bash
git add tests/ && git commit -m "test: add tests for parser, registry, memory, and loop"
```

---

## Post-Implementation Checklist

- [ ] `backend/` all modules import cleanly
- [ ] `python -m pytest tests/ -v` all green
- [ ] `uvicorn main:app` starts without errors
- [ ] `npm run dev` starts frontend, proxy reaches backend
- [ ] Create `.env` from `.env.example` with real API keys
- [ ] Manual E2E: submit a question, see steps flow, get report
