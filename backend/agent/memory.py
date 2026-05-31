from models.schemas import Step, StepType


class MemoryManager:
    def __init__(self, max_history_chars: int = 8000):
        self.steps: list[Step] = [] # 存储所有推理步骤的有序列表
        self.key_facts: list[str] = [] # 存储从推理过程中提取的关键事实
        self.max_history_chars = max_history_chars # 上下文压缩阈值（默认 8000 字符），当步骤总长度超过此值时触发压缩

    def add_step(self, step: Step) -> None:
        self.steps.append(step)

    def add_key_fact(self, fact: str) -> None:
        if fact not in self.key_facts: # 自动去重
            self.key_facts.append(fact)

    def get_steps_for_context(self) -> str:
        """Build context string from steps, compressing if needed."""
        lines = []
        total_chars = 0
        # 遍历所有 steps，用 _format_step 格式化每个步骤
        # 计算总字符数
        # 如果未超过阈值 → 直接返回完整上下文
        for step in self.steps:
            step_str = self._format_step(step)
            total_chars += len(step_str)
            lines.append(step_str)

        if total_chars <= self.max_history_chars:
            return "\n".join(lines)

        # Compress: summarize early steps, keep recent ones
        recent = []
        recent_chars = 0
        for line in reversed(lines): # 从末尾开始逆序遍历，将最近的步骤保留到 recent 列表中
            if recent_chars + len(line) > self.max_history_chars // 2: # 保留上限为 max_history_chars // 2（一半的配额给最近步骤）
                break
            recent.insert(0, line)
            recent_chars += len(line)
        
        # 剩余的早期步骤用 _build_summary 生成压缩摘要
        summary = self._build_summary(self.steps[: len(self.steps) - len(recent)])
        return f"{summary}\n\n--- Recent Steps ---\n" + "\n".join(recent)# 返回格式：[摘要] \n\n--- Recent Steps ---\n [最近的详细步骤]
    
    # 格式：[Round {round}] {type}: {content} (tool: {tool_name})
    def _format_step(self, step: Step) -> str:
        base = f"[Round {step.round}] {step.type.upper()}: {step.content}"
        if step.tool_name:
            base += f" (tool: {step.tool_name})"
        return base

    def _build_summary(self, early_steps: list[Step]) -> str:
        """Build a compressed summary of early steps, preserving key facts."""
        #统计早期步骤的总数和 ACTION 类型的数量
        #列出所有已收集的关键事实（如果没有则为 (none)）
        facts_str = "\n".join(f"- {f}" for f in self.key_facts) if self.key_facts else "(none)"
        action_count = sum(1 for s in early_steps if s.type == StepType.ACTION)
        return (
            f"[Compressed summary of {len(early_steps)} early steps]\n"
            f"Early actions taken: {action_count}\n"
            f"Key facts collected:\n{facts_str}"
        )
