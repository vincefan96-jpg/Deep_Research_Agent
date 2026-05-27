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
