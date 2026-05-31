import json
import re
from dataclasses import dataclass

#LLM 输出解析器，负责将大语言模型返回的原始文本解析成结构化的 ParsedOutput 对象
@dataclass
class ParsedOutput:
    thought: str  ## 模型的思考过程
    action: str | None = None       # 要调用的工具名
    action_params: dict | None = None # 工具的参数（JSON）
    is_final: bool = False # 是否为最终答案
    final_answer: str | None = None # 最终答案文本
    parse_error: str | None = None # 解析失败时的错误信息


def parse(raw: str) -> ParsedOutput:
    thought = ""
    action = None
    action_params = None
    is_final = False
    final_answer = None
    parse_error = None

    # 提取的是 THOUGHT: 后面直到 \nACTION、\nFINAL_ANSWER 或字符串末尾之间的文本
    thought_match = re.search(r"THOUGHT:\s*(.+?)(?=\n(?:ACTION|FINAL_ANSWER)|\Z)", raw, re.DOTALL | re.IGNORECASE)
    if thought_match:
        thought = thought_match.group(1).strip()

    # 检查 FINAL_ANSWER
    if re.search(r"FINAL_ANSWER", raw, re.IGNORECASE):
        is_final = True
        fa_match = re.search(r"FINAL_ANSWER:\s*(.+)", raw, re.DOTALL | re.IGNORECASE)
        if fa_match:
            final_answer = fa_match.group(1).strip()
        else:
            # 容错：没有冒号的情况
            idx = raw.upper().find("FINAL_ANSWER")
            final_answer = raw[idx + len("FINAL_ANSWER"):].strip(": \n")

    # 提取 ACTION
    action_match = re.search(r"ACTION:\s*(.+?)(?=\n(?:THOUGHT|OBSERVATION|FINAL_ANSWER)|\Z)", raw, re.DOTALL | re.IGNORECASE)
    # 关键逻辑：如果已经判定为 final，则跳过 action 提取
    if action_match and not is_final:
        action_str = action_match.group(1).strip()

        # Action 格式：<工具名>|<JSON参数>
        if "|" in action_str:
            parts = action_str.split("|", 1)
            action = parts[0].strip()
            try: # 参数部分用 json.loads() 解析，解析失败则设置 parse_error
                action_params = json.loads(parts[1].strip())
            except json.JSONDecodeError:
                parse_error = f"无法解析 Action 参数 JSON：{parts[1][:100]}"
                action_params = {}

    # 如果什么都没解析到，记录错误并将原始输出的前 500 字符作为 thought 保存——方便调试和恢复。
    if not thought and not action and not is_final:
        parse_error = "无法从 LLM 输出中解析 THOUGHT、ACTION 或 FINAL_ANSWER。"
        thought = raw[:500]

    return ParsedOutput(
        thought=thought,
        action=action,
        action_params=action_params,
        is_final=is_final,
        final_answer=final_answer,
        parse_error=parse_error,
    )
