"""SQLite cache for target responses and judge verdicts. Keys are content hashes.
Target generation and judge classification are hashed.
"""

import hashlib
import os
import sqlite3
import threading

_lock = threading.Lock()
_conn = None

def _get_conn(cache_dir):
    global _conn
    if _conn is None:
        os.makedirs(cache_dir, exist_ok=True)
        db_path = os.path.join(cache_dir, "jb_cache.sqlite")
        _conn = sqlite3.connect(db_path, check_same_thread=False, timeout=30)
        _conn.execute("PRAGMA journal_mode=WAL;")
        _conn.execute(
            "CREATE TABLE IF NOT EXISTS kv (key TEXT PRIMARY KEY, value TEXT NOT NULL)"
        )
        _conn.commit()
    return _conn

def _hash(*parts):
    h = hashlib.sha256()
    for part in parts:
        h.update(str(part).encode("utf-8"))
        h.update(b"\x00")
    return h.hexdigest()

def response_key(model_name, messages, max_tokens=512, temperature=0.0):
    # messages is a list of {"role","content"} dicts; serialize deterministically.
    flat = "|".join(f"{m.get('role','')}:{m.get('content','')}" for m in messages)
    return _hash("resp:v2", model_name, max_tokens, temperature, flat)

def verdict_key(judge_model_name, judge_mode, rubric_version, behavior, response):
    return _hash(
        "verdict:v2", judge_model_name, judge_mode, rubric_version,
        behavior, response,
    )

def get(cache_dir, key):
    conn = _get_conn(cache_dir)
    with _lock:
        row = conn.execute("SELECT value FROM kv WHERE key = ?", (key,)).fetchone()
    return row[0] if row else None

def put(cache_dir, key, value):
    conn = _get_conn(cache_dir)
    with _lock:
        conn.execute(
            "INSERT OR REPLACE INTO kv (key, value) VALUES (?, ?)", (key, str(value))
        )
        conn.commit()
