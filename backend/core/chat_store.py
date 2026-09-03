"""Persistent chat history -- stdlib sqlite3 only, no new dependency.

Per CLAUDE.md's own note on this feature: "real persistence (SQLite, no new
dependency -- stdlib sqlite3 is enough) storing conversations and messages,
plus a sidebar UI." This module is that storage layer; app.py wires the UI.

No auth system exists (see CLAUDE.md), so conversations are scoped to an
anonymous `session_id` the caller supplies -- app.py generates one per
browser via gr.BrowserState and persists it in that browser's local storage,
so one visitor never sees another visitor's conversations, without needing
real user accounts.
"""

import sqlite3
import time
import uuid
import logging
from pathlib import Path
from threading import Lock
from typing import List, Dict, Optional

logger = logging.getLogger(__name__)

DB_PATH = Path(__file__).resolve().parents[2] / "data" / "chat_history.sqlite3"

# One process-wide lock. SQLite handles concurrent readers fine but this app's
# write volume is low (one row per completed chat turn), so a single lock
# avoids "database is locked" errors under concurrent Gradio sessions without
# needing a connection pool.
_lock = Lock()

_SCHEMA = """
CREATE TABLE IF NOT EXISTS conversations (
    id TEXT PRIMARY KEY,
    session_id TEXT NOT NULL,
    title TEXT NOT NULL,
    created_at REAL NOT NULL,
    updated_at REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_conversations_session ON conversations(session_id, updated_at DESC);

CREATE TABLE IF NOT EXISTS messages (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    conversation_id TEXT NOT NULL,
    role TEXT NOT NULL,
    content TEXT NOT NULL,
    created_at REAL NOT NULL,
    FOREIGN KEY (conversation_id) REFERENCES conversations(id)
);
CREATE INDEX IF NOT EXISTS idx_messages_conversation ON messages(conversation_id, id ASC);
"""


def _connect() -> sqlite3.Connection:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(DB_PATH), timeout=10.0)
    conn.row_factory = sqlite3.Row
    return conn


def init_db() -> None:
    """Creates the tables if they don't exist. Safe to call on every startup."""
    with _lock:
        conn = _connect()
        try:
            conn.executescript(_SCHEMA)
            conn.commit()
        finally:
            conn.close()


def new_session_id() -> str:
    """A fresh anonymous session id. Callers persist this in the browser
    (see app.py's gr.BrowserState usage) so it survives a page refresh but
    never crosses browsers/devices -- there is no login to tie it to."""
    return str(uuid.uuid4())


def _make_title(first_message: str, max_len: int = 60) -> str:
    text = " ".join((first_message or "").split())
    if len(text) <= max_len:
        return text or "New conversation"
    return text[:max_len].rstrip() + "..."


def create_conversation(session_id: str, first_message: str) -> str:
    """Creates a new conversation row, titled from the first user message.
    Returns the new conversation id."""
    conv_id = str(uuid.uuid4())
    now = time.time()
    title = _make_title(first_message)
    with _lock:
        conn = _connect()
        try:
            conn.execute(
                "INSERT INTO conversations (id, session_id, title, created_at, updated_at) VALUES (?, ?, ?, ?, ?)",
                (conv_id, session_id, title, now, now)
            )
            conn.commit()
        finally:
            conn.close()
    return conv_id


def add_message(conversation_id: str, role: str, content: str) -> None:
    """Appends one message and bumps the conversation's updated_at (used for
    sidebar ordering -- most recently active conversation first)."""
    now = time.time()
    with _lock:
        conn = _connect()
        try:
            conn.execute(
                "INSERT INTO messages (conversation_id, role, content, created_at) VALUES (?, ?, ?, ?)",
                (conversation_id, role, content, now)
            )
            conn.execute(
                "UPDATE conversations SET updated_at = ? WHERE id = ?",
                (now, conversation_id)
            )
            conn.commit()
        finally:
            conn.close()


def list_conversations(session_id: str, limit: int = 50) -> List[Dict]:
    """Most-recently-active conversations for one browser session, for the sidebar."""
    with _lock:
        conn = _connect()
        try:
            rows = conn.execute(
                "SELECT id, title, updated_at FROM conversations "
                "WHERE session_id = ? ORDER BY updated_at DESC LIMIT ?",
                (session_id, limit)
            ).fetchall()
        finally:
            conn.close()
    return [dict(r) for r in rows]


def get_messages(conversation_id: str) -> List[Dict]:
    """Full message history for one conversation, oldest first -- ready to
    hand straight to gr.Chatbot(type='messages')."""
    with _lock:
        conn = _connect()
        try:
            rows = conn.execute(
                "SELECT role, content FROM messages WHERE conversation_id = ? ORDER BY id ASC",
                (conversation_id,)
            ).fetchall()
        finally:
            conn.close()
    return [{"role": r["role"], "content": r["content"]} for r in rows]


def delete_conversation(conversation_id: str, session_id: str) -> bool:
    """Deletes a conversation and its messages. Scoped to session_id so one
    browser can't delete another's conversation by guessing an id."""
    with _lock:
        conn = _connect()
        try:
            owned = conn.execute(
                "SELECT 1 FROM conversations WHERE id = ? AND session_id = ?",
                (conversation_id, session_id)
            ).fetchone()
            if not owned:
                return False
            conn.execute("DELETE FROM messages WHERE conversation_id = ?", (conversation_id,))
            conn.execute("DELETE FROM conversations WHERE id = ?", (conversation_id,))
            conn.commit()
            return True
        finally:
            conn.close()
