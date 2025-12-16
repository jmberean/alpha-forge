"""SQLite-backed persistence for AlphaForge API results."""

from __future__ import annotations

import json
import logging
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


class NumpyEncoder(json.JSONEncoder):
    """JSON encoder that handles numpy types."""

    def default(self, obj: Any) -> Any:
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.bool_):
            return bool(obj)
        return super().default(obj)


class Storage:
    def __init__(self, db_path: str = "data/alphaforge.db") -> None:
        self.db_path = db_path
        self._init_db()

    def _init_db(self) -> None:
        """Initialize database schema."""
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)

        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS validations (
                    id TEXT PRIMARY KEY,
                    strategy_name TEXT,
                    status TEXT,
                    passed INTEGER,
                    metrics_json TEXT,
                    equity_curve_json TEXT,
                    logs_json TEXT,
                    timestamp TEXT
                )
                """
            )

            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS discovery_runs (
                    id TEXT PRIMARY KEY,
                    status TEXT,
                    config_json TEXT,
                    result_json TEXT,
                    timestamp TEXT
                )
                """
            )

            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS factory_runs (
                    id TEXT PRIMARY KEY,
                    status TEXT,
                    config_json TEXT,
                    result_json TEXT,
                    timestamp TEXT
                )
                """
            )

            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS pipeline_stats (
                    key TEXT PRIMARY KEY,
                    value INTEGER,
                    updated_at TEXT
                )
                """
            )

    def save_validation(self, validation_id: str, data: dict[str, Any]) -> None:
        """Save or update validation result."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO validations
                (id, strategy_name, status, passed, metrics_json, equity_curve_json, logs_json, timestamp)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    validation_id,
                    data.get("strategy_name", ""),
                    data.get("status", "unknown"),
                    1 if data.get("passed", False) else 0,
                    json.dumps(data.get("metrics", {}), cls=NumpyEncoder),
                    json.dumps(data.get("equity_curve", []), cls=NumpyEncoder),
                    json.dumps(data.get("logs", []), cls=NumpyEncoder),
                    data.get("timestamp", datetime.now().isoformat()),
                ),
            )

    def get_validation(self, validation_id: str) -> dict[str, Any] | None:
        """Get validation result."""
        with sqlite3.connect(self.db_path) as conn:
            row = conn.execute(
                "SELECT * FROM validations WHERE id = ?", (validation_id,)
            ).fetchone()
            if not row:
                return None

            return {
                "validation_id": row[0],
                "strategy_name": row[1],
                "status": row[2],
                "passed": bool(row[3]),
                "metrics": json.loads(row[4]),
                "equity_curve": json.loads(row[5]),
                "logs": json.loads(row[6]),
                "timestamp": row[7],
            }

    def list_validations(self) -> list[dict[str, Any]]:
        """List all validations (most recent first)."""
        with sqlite3.connect(self.db_path) as conn:
            rows = conn.execute("SELECT * FROM validations ORDER BY timestamp DESC").fetchall()

        return [
            {
                "validation_id": row[0],
                "strategy_name": row[1],
                "status": row[2],
                "passed": bool(row[3]),
                "metrics": json.loads(row[4]),
                "equity_curve": json.loads(row[5]),
                "logs": json.loads(row[6]),
                "timestamp": row[7],
            }
            for row in rows
        ]

    def save_discovery_run(
        self, discovery_id: str, status: str, config: dict[str, Any], result: dict[str, Any]
    ) -> None:
        """Save or update a discovery run."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO discovery_runs
                (id, status, config_json, result_json, timestamp)
                VALUES (?, ?, ?, ?, ?)
                """,
                (
                    discovery_id,
                    status,
                    json.dumps(config, cls=NumpyEncoder),
                    json.dumps(result, cls=NumpyEncoder),
                    result.get("timestamp", datetime.now().isoformat()),
                ),
            )

    def get_discovery_run(self, discovery_id: str) -> dict[str, Any] | None:
        """Get a discovery run."""
        with sqlite3.connect(self.db_path) as conn:
            row = conn.execute(
                "SELECT * FROM discovery_runs WHERE id = ?", (discovery_id,)
            ).fetchone()
            if not row:
                return None

            result = json.loads(row[3])
            result.setdefault("discovery_id", row[0])
            result.setdefault("status", row[1])
            result.setdefault("timestamp", row[4])
            return result

    def list_discovery_runs(self) -> list[dict[str, Any]]:
        with sqlite3.connect(self.db_path) as conn:
            rows = conn.execute("SELECT * FROM discovery_runs ORDER BY timestamp DESC").fetchall()
        return [json.loads(row[3]) for row in rows]

    def save_factory_run(
        self, factory_id: str, status: str, config: dict[str, Any], result: dict[str, Any]
    ) -> None:
        """Save or update a factory run."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO factory_runs
                (id, status, config_json, result_json, timestamp)
                VALUES (?, ?, ?, ?, ?)
                """,
                (
                    factory_id,
                    status,
                    json.dumps(config, cls=NumpyEncoder),
                    json.dumps(result, cls=NumpyEncoder),
                    result.get("timestamp", datetime.now().isoformat()),
                ),
            )

    def get_factory_run(self, factory_id: str) -> dict[str, Any] | None:
        """Get a factory run."""
        with sqlite3.connect(self.db_path) as conn:
            row = conn.execute("SELECT * FROM factory_runs WHERE id = ?", (factory_id,)).fetchone()
            if not row:
                return None

            result = json.loads(row[3])
            result.setdefault("factory_id", row[0])
            result.setdefault("status", row[1])
            result.setdefault("timestamp", row[4])
            return result

    def list_factory_runs(self) -> list[dict[str, Any]]:
        with sqlite3.connect(self.db_path) as conn:
            rows = conn.execute("SELECT * FROM factory_runs ORDER BY timestamp DESC").fetchall()
        return [json.loads(row[3]) for row in rows]

    def save_pipeline_stats(self, stats: dict[str, int]) -> None:
        """Save or update pipeline statistics.

        Args:
            stats: Dictionary of stat keys to integer values
        """
        with sqlite3.connect(self.db_path) as conn:
            for key, value in stats.items():
                conn.execute(
                    """
                    INSERT OR REPLACE INTO pipeline_stats (key, value, updated_at)
                    VALUES (?, ?, ?)
                    """,
                    (key, value, datetime.now().isoformat()),
                )

    def load_pipeline_stats(self) -> dict[str, int]:
        """Load pipeline statistics from database.

        Returns:
            Dictionary of stat keys to integer values
        """
        with sqlite3.connect(self.db_path) as conn:
            rows = conn.execute("SELECT key, value FROM pipeline_stats").fetchall()

        if not rows:
            # Return default values if no stats exist
            return {
                "total_generated": 0,
                "total_validated": 0,
                "total_passed": 0,
                "total_deployed": 0,
            }

        return {row[0]: row[1] for row in rows}
