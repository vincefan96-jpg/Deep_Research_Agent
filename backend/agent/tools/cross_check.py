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
