from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Any

from .paths import DB_PATH
from .utils import now_iso


SCHEMA = """
PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS writers (
  id TEXT PRIMARY KEY,
  created_at TEXT NOT NULL,
  notes TEXT NOT NULL DEFAULT ''
);

CREATE TABLE IF NOT EXISTS style_refs (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  writer_id TEXT NOT NULL REFERENCES writers(id),
  path TEXT NOT NULL,
  sha256 TEXT NOT NULL,
  width INTEGER,
  height INTEGER,
  created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS worksheets (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  writer_id TEXT NOT NULL REFERENCES writers(id),
  source_path TEXT NOT NULL,
  manifest_path TEXT NOT NULL,
  created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS glyphs (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  worksheet_id INTEGER NOT NULL REFERENCES worksheets(id),
  label TEXT NOT NULL,
  path TEXT NOT NULL,
  sha256 TEXT NOT NULL,
  width INTEGER,
  height INTEGER,
  created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS documents (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  writer_id TEXT NOT NULL REFERENCES writers(id),
  source_path TEXT NOT NULL,
  source_sha256 TEXT NOT NULL,
  created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS render_jobs (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  document_id INTEGER NOT NULL REFERENCES documents(id),
  out_dir TEXT NOT NULL,
  status TEXT NOT NULL,
  created_at TEXT NOT NULL,
  completed_at TEXT
);

CREATE TABLE IF NOT EXISTS render_spans (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  job_id INTEGER NOT NULL REFERENCES render_jobs(id),
  span_id TEXT NOT NULL,
  kind TEXT NOT NULL,
  route TEXT NOT NULL,
  text TEXT NOT NULL,
  artifact_path TEXT,
  exact_text_certified INTEGER NOT NULL DEFAULT 0
);

CREATE TABLE IF NOT EXISTS artifacts (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  job_id INTEGER REFERENCES render_jobs(id),
  kind TEXT NOT NULL,
  path TEXT NOT NULL,
  sha256 TEXT,
  created_at TEXT NOT NULL
);
"""


def connect(db_path: Path | None = None) -> sqlite3.Connection:
    db_path = db_path or DB_PATH
    db_path.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(db_path)
    con.row_factory = sqlite3.Row
    con.execute("PRAGMA foreign_keys = ON")
    return con


def init_db(db_path: Path | None = None) -> None:
    db_path = db_path or DB_PATH
    with connect(db_path) as con:
        con.executescript(SCHEMA)


def create_writer(writer_id: str) -> None:
    init_db()
    with connect() as con:
        con.execute(
            "INSERT OR IGNORE INTO writers(id, created_at) VALUES (?, ?)",
            (writer_id, now_iso()),
        )


def require_writer(con: sqlite3.Connection, writer_id: str) -> None:
    row = con.execute("SELECT id FROM writers WHERE id = ?", (writer_id,)).fetchone()
    if row is None:
        raise RuntimeError(f"unknown writer {writer_id!r}; run `handgen writer create {writer_id}` first")


def latest_style_ref(con: sqlite3.Connection, writer_id: str) -> sqlite3.Row | None:
    return con.execute(
        "SELECT * FROM style_refs WHERE writer_id = ? ORDER BY id DESC LIMIT 1",
        (writer_id,),
    ).fetchone()


def latest_worksheet(con: sqlite3.Connection, writer_id: str) -> sqlite3.Row | None:
    return con.execute(
        "SELECT * FROM worksheets WHERE writer_id = ? ORDER BY id DESC LIMIT 1",
        (writer_id,),
    ).fetchone()


def insert_artifact(con: sqlite3.Connection, job_id: int | None, kind: str, path: str, sha256: str | None = None) -> None:
    con.execute(
        "INSERT INTO artifacts(job_id, kind, path, sha256, created_at) VALUES (?, ?, ?, ?, ?)",
        (job_id, kind, path, sha256, now_iso()),
    )


def row_to_dict(row: sqlite3.Row | None) -> dict[str, Any] | None:
    return dict(row) if row is not None else None
