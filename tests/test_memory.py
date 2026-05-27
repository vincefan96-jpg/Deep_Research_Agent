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
