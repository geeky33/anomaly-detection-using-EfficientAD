"""Inspection history stored in a local SQLite file."""

from __future__ import annotations

import os
import sqlite3
import time
from pathlib import Path

import pandas as pd

# MVP_HISTORY_DB lets the tests use a throw-away database.
DB_PATH = Path(os.environ.get("MVP_HISTORY_DB", Path(__file__).resolve().parent / "results" / "history.sqlite3"))
# Saved overlays and reports live next to the database, so a test database never
# writes into the real results folder.
OUTPUT_DIR = DB_PATH.parent


def _connect() -> sqlite3.Connection:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(DB_PATH)
    connection.execute(
        """CREATE TABLE IF NOT EXISTS inspections (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT, category TEXT, source TEXT, image TEXT,
            score REAL, image_threshold REAL, prediction TEXT,
            defect_area_pct REAL, latency_ms REAL, ground_truth TEXT, overlay_path TEXT)"""
    )
    # Columns added for the model comparison / workspace (older databases are migrated).
    existing = {row[1] for row in connection.execute("PRAGMA table_info(inspections)")}
    for column, kind in (("model", "TEXT"), ("image_id", "TEXT"), ("iou", "REAL"), ("dice", "REAL"),
                         ("num_regions", "INTEGER")):
        if column not in existing:
            connection.execute(f"ALTER TABLE inspections ADD COLUMN {column} {kind}")
    return connection


def add(record: dict) -> None:
    columns = ["category", "source", "image", "score", "image_threshold", "prediction",
               "defect_area_pct", "latency_ms", "ground_truth", "overlay_path", "model", "image_id", "iou",
               "dice", "num_regions"]
    with _connect() as connection:
        connection.execute(
            f"INSERT INTO inspections (timestamp, {', '.join(columns)}) VALUES (?, {', '.join('?' * len(columns))})",
            [time.strftime("%Y-%m-%d %H:%M:%S")] + [record.get(c) for c in columns],
        )


def load() -> pd.DataFrame:
    with _connect() as connection:
        return pd.read_sql_query("SELECT * FROM inspections ORDER BY id DESC", connection)


def clear() -> None:
    with _connect() as connection:
        connection.execute("DELETE FROM inspections")
