from llm.client import LLMClient


async def cross_check(facts: str) -> str:
    """对收集的事实进行交叉验证，检查一致性和矛盾。"""
    llm = LLMClient()
    prompt = f"""你是一名事实核查员。请审查以下收集到的信息，检查其内部一致性。

请识别：
1. 一致性水平（高/中/低）
2. 任何相互矛盾的结论
3. 被多个来源交叉验证的事实

请以以下 JSON 格式返回分析结果：
{{
  "consistency": "高|中|低",
  "conflicts": ["矛盾描述..."],
  "verified_facts": ["已验证的事实..."]
}}

待审查信息：
{facts}"""

    raw = await llm.chat([{"role": "user", "content": prompt}], max_tokens=2000)

    import json
    try:
        result = json.loads(raw)
    except json.JSONDecodeError:
        result = {
            "consistency": "中",
            "conflicts": ["无法解析交叉验证结果"],
            "verified_facts": [],
        }

    conflicts = result.get("conflicts", [])
    verified = result.get("verified_facts", [])
    consistency = result.get("consistency", "中")

    lines = [
        f"交叉验证一致性：{consistency}",
    ]
    if conflicts:
        lines.append(f"\n发现矛盾（{len(conflicts)} 处）：")
        for c in conflicts:
            lines.append(f"  - {c}")
    if verified:
        lines.append(f"\n已验证的事实（{len(verified)} 条）：")
        for f in verified:
            lines.append(f"  - {f}")
    if not conflicts and not verified:
        lines.append("未发现矛盾或已验证的事实。")

    return "\n".join(lines)
