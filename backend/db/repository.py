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


async def delete_session(session_id: str) -> bool:
    db = await get_db()
    cursor = await db.execute("DELETE FROM sessions WHERE id = ?", (session_id,))
    await db.commit()
    return cursor.rowcount > 0
