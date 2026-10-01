from __future__ import annotations

import sqlite3
from contextlib import closing
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional


DEFAULT_DB = Path("data") / "bills.db"


class BillDatabase:
    """Small SQLite database for storing every scanned bill."""

    def __init__(self, db_path: str | Path = DEFAULT_DB):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._create_tables()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _create_tables(self) -> None:
        with closing(self._connect()) as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS bills (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    denomination INTEGER,
                    serial_number TEXT,
                    serial_left_read TEXT,
                    serial_right_read TEXT,
                    serial_match INTEGER,
                    series TEXT,
                    federal_reserve_letter TEXT,
                    federal_reserve_number INTEGER,
                    federal_reserve_district TEXT,
                    federal_reserve_city TEXT,
                    federal_reserve_match INTEGER,
                    confidence REAL,
                    status TEXT NOT NULL,
                    warnings TEXT,
                    scanned_at TEXT NOT NULL
                )
                """
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_bills_serial "
                "ON bills(serial_number)"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_bills_scanned_at "
                "ON bills(scanned_at)"
            )
            conn.commit()

    def save_scan(self, result: Dict[str, Any]) -> int:
        """Save one scan result and return its database ID."""
        fed = result.get("federal_reserve") or {}
        warnings = result.get("warnings") or []

        with closing(self._connect()) as conn:
            cur = conn.execute(
                """
                INSERT INTO bills (
                    denomination,
                    serial_number,
                    serial_left_read,
                    serial_right_read,
                    serial_match,
                    series,
                    federal_reserve_letter,
                    federal_reserve_number,
                    federal_reserve_district,
                    federal_reserve_city,
                    federal_reserve_match,
                    confidence,
                    status,
                    warnings,
                    scanned_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    result.get("denomination"),
                    result.get("serial_number"),
                    result.get("serial_left_read"),
                    result.get("serial_right_read"),
                    None if result.get("serial_match") is None
                    else int(bool(result.get("serial_match"))),
                    result.get("series"),
                    fed.get("letter"),
                    fed.get("number"),
                    fed.get("district"),
                    fed.get("city"),
                    None if fed.get("match") is None
                    else int(bool(fed.get("match"))),
                    result.get("confidence"),
                    result.get("status", "needs_review"),
                    "; ".join(str(w) for w in warnings),
                    datetime.now(timezone.utc).isoformat(),
                ),
            )
            conn.commit()
            return int(cur.lastrowid)

    def recent(self, limit: int = 50) -> List[Dict[str, Any]]:
        with closing(self._connect()) as conn:
            rows = conn.execute(
                "SELECT * FROM bills ORDER BY id DESC LIMIT ?",
                (limit,),
            ).fetchall()
            return [dict(row) for row in rows]

    def find_by_serial(self, serial_number: str) -> List[Dict[str, Any]]:
        with closing(self._connect()) as conn:
            rows = conn.execute(
                """
                SELECT * FROM bills
                WHERE UPPER(serial_number) = UPPER(?)
                ORDER BY id DESC
                """,
                (serial_number.strip(),),
            ).fetchall()
            return [dict(row) for row in rows]

    def count(self) -> int:
        with closing(self._connect()) as conn:
            return int(conn.execute("SELECT COUNT(*) FROM bills").fetchone()[0])
