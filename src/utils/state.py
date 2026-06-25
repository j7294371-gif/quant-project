import json
import os
import sqlite3
import time
from decimal import Decimal
from loguru import logger


class StateStore:
    def __init__(self, state_dir: str):
        os.makedirs(state_dir, exist_ok=True)
        self.state_dir = state_dir

        # Open SQLite in WAL mode
        db_path = os.path.join(state_dir, "equity.db")
        self.conn = sqlite3.connect(db_path, check_same_thread=False)
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS equity_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp INTEGER NOT NULL,
                equity TEXT NOT NULL,
                created_at TEXT DEFAULT (datetime('now'))
            )
        """)
        self.conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_equity_timestamp ON equity_history(timestamp)
        """)
        self.conn.commit()

    # === JSON persistence ===

    def _path(self, key: str) -> str:
        return os.path.join(self.state_dir, f"{key}.json")

    def save(self, key: str, data: dict) -> None:
        data = dict(data)
        data["updated_at"] = int(time.time() * 1000)
        tmp_path = self._path(key) + ".tmp"
        final_path = self._path(key)
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        os.replace(tmp_path, final_path)

    def load(self, key: str) -> dict | None:
        path = self._path(key)
        if not os.path.exists(path):
            return None
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except (json.JSONDecodeError, UnicodeDecodeError) as e:
            logger.warning(f"状态文件损坏 ({key}): {e}")
            return None

    def delete(self, key: str) -> None:
        path = self._path(key)
        if os.path.exists(path):
            os.remove(path)

    # === SQLite equity history ===

    def append_equity(self, timestamp: int, equity: Decimal) -> None:
        self.conn.execute(
            "INSERT INTO equity_history (timestamp, equity) VALUES (?, ?)",
            (timestamp, str(equity)),
        )
        self.conn.commit()

    def get_equity_history(self, since: int) -> list[tuple[int, Decimal]]:
        cursor = self.conn.execute(
            "SELECT timestamp, equity FROM equity_history WHERE timestamp >= ? ORDER BY timestamp ASC",
            (since,),
        )
        return [(row[0], Decimal(row[1])) for row in cursor.fetchall()]

    def get_today_starting_equity(self) -> Decimal | None:
        from datetime import datetime, timezone
        today_start = datetime.now(timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0)
        today_start_ms = int(today_start.timestamp() * 1000)
        cursor = self.conn.execute(
            "SELECT equity FROM equity_history WHERE timestamp >= ? ORDER BY timestamp ASC LIMIT 1",
            (today_start_ms,),
        )
        row = cursor.fetchone()
        return Decimal(row[0]) if row else None

    # === Shutdown markers ===

    def mark_shutdown(self, status: str) -> None:
        self.save("shutdown", {"status": status})

    def is_clean_shutdown(self) -> bool:
        data = self.load("shutdown")
        return data is not None and data.get("status") == "clean"

    def close(self) -> None:
        self.conn.close()
