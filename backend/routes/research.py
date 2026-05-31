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
    # 每次请求生成一个 UUID 作为会话 ID，立即写入数据库（状态为 "running"），用于后续历史记录检索
    session_id = str(uuid.uuid4())
    await create_session(session_id, req.query)

    async def generate():
        collected_steps = [] # 收集中间步骤，最终持久化
        report = None # 最终报告
        try:
            orchestrator = ResearchOrchestrator()
            async for event in orchestrator.research(req.query):
                # Collect steps for persistence
                if "type" in event and event.get("type") in ("thought", "action", "observation"):
                    collected_steps.append(event)
                if "markdown" in event:
                    report = event["markdown"]
                # SSE 协议格式：event: <类型>\ndata: <JSON>\n\n
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
            "Cache-Control": "no-cache", # 禁止浏览器/代理缓存
            "Connection": "keep-alive",  # 保持连接
            "X-Accel-Buffering": "no",   # 禁用 Nginx 缓冲（SSE 必须实时推送）
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

# 返回最近 50 条会话摘要（id、query、status、created_at）
@router.get("/history")
async def list_history():
    sessions = await get_all_sessions()
    return {"sessions": sessions}

# 返回单个会话的完整信息，含解析后的 steps 列表
@router.get("/history/{session_id}")
async def get_history(session_id: str):
    session = await get_session(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Session not found")
    
    # 数据库中 steps_json 是 JSON 字符串，这里将其反序列化为 steps 列表再返回，并删除原始字符串字段，方便前端直接使用
    if session.get("steps_json"):
        session["steps"] = json.loads(session["steps_json"])
    del session["steps_json"]
    return session

# 删除指定会话
@router.delete("/history/{session_id}")
async def remove_history(session_id: str):
    deleted = await delete_session(session_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="Session not found")
    return {"ok": True}
