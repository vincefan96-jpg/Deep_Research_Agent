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
