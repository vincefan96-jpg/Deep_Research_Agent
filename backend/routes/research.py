import json
import uuid
from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from models.schemas import ResearchRequest
from agent.orchestrator import ResearchOrchestrator
from db.repository import create_session, update_session, get_all_sessions, get_session, delete_session

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


@router.delete("/history/{session_id}")
async def remove_history(session_id: str):
    deleted = await delete_session(session_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="Session not found")
    return {"ok": True}
