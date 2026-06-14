"""
Monitoring site registry — tracks which locations are actively monitored.
"""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import aiosqlite
from pydantic import BaseModel, Field

from .schema import EcosystemState, Location

logger = logging.getLogger(__name__)

_DEFAULT_DB_DIR = Path.home() / ".kinship-earth"
_DEFAULT_DB_PATH = _DEFAULT_DB_DIR / "monitoring.db"

_CREATE_TABLES = """
CREATE TABLE IF NOT EXISTS monitoring_sites (
    site_id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    lat REAL NOT NULL,
    lng REAL NOT NULL,
    enabled INTEGER NOT NULL DEFAULT 1,
    created_at TEXT NOT NULL,
    last_checked TEXT,
    check_interval_hours INTEGER NOT NULL DEFAULT 24,
    metadata TEXT NOT NULL DEFAULT '{}'
);

CREATE TABLE IF NOT EXISTS ecosystem_states (
    id TEXT NOT NULL,
    site_id TEXT NOT NULL,
    timestamp TEXT NOT NULL,
    state_json TEXT NOT NULL,
    PRIMARY KEY (site_id, timestamp)
);

CREATE TABLE IF NOT EXISTS subscriptions (
    site_id TEXT NOT NULL,
    user_id TEXT NOT NULL DEFAULT 'anonymous',
    severity_min TEXT NOT NULL DEFAULT 'warning',
    created_at TEXT NOT NULL,
    last_notified TEXT,
    PRIMARY KEY (site_id, user_id)
);

CREATE INDEX IF NOT EXISTS idx_states_site ON ecosystem_states(site_id);
CREATE INDEX IF NOT EXISTS idx_states_time ON ecosystem_states(timestamp);
"""


class Subscription(BaseModel):
    """An alert subscription for a monitored location."""

    site_id: str
    user_id: str = Field(default="anonymous")
    severity_min: str = Field(default="warning")
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    last_notified: Optional[datetime] = None


class MonitoringSite(BaseModel):
    """A location being actively monitored for ecosystem state changes."""

    site_id: str
    name: str
    location: Location
    enabled: bool = True
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    last_checked: Optional[datetime] = None
    check_interval_hours: int = 24


class MonitoringRegistry:
    """Manages monitored sites and their ecosystem state history."""

    def __init__(self, db_path: str | None = None):
        self._db_path = db_path or str(
            os.environ.get("KINSHIP_MONITORING_DB_PATH", _DEFAULT_DB_PATH)
        )
        self._initialized = False

    async def initialize(self) -> None:
        if self._initialized:
            return
        Path(self._db_path).parent.mkdir(parents=True, exist_ok=True)
        async with aiosqlite.connect(self._db_path) as db:
            await db.executescript(_CREATE_TABLES)
            await db.commit()
        self._initialized = True

    async def add_site(self, site: MonitoringSite) -> None:
        await self.initialize()
        async with aiosqlite.connect(self._db_path) as db:
            await db.execute(
                """INSERT OR REPLACE INTO monitoring_sites
                (site_id, name, lat, lng, enabled, created_at, last_checked, check_interval_hours)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                (site.site_id, site.name, site.location.lat, site.location.lng,
                 int(site.enabled), site.created_at.isoformat(),
                 site.last_checked.isoformat() if site.last_checked else None,
                 site.check_interval_hours),
            )
            await db.commit()

    async def get_site(self, site_id: str) -> MonitoringSite | None:
        await self.initialize()
        async with aiosqlite.connect(self._db_path) as db:
            cursor = await db.execute(
                "SELECT site_id, name, lat, lng, enabled, created_at, last_checked, check_interval_hours FROM monitoring_sites WHERE site_id = ?",
                (site_id,),
            )
            row = await cursor.fetchone()
            if not row:
                return None
            return MonitoringSite(
                site_id=row[0], name=row[1],
                location=Location(lat=row[2], lng=row[3]),
                enabled=bool(row[4]),
                created_at=datetime.fromisoformat(row[5]),
                last_checked=datetime.fromisoformat(row[6]) if row[6] else None,
                check_interval_hours=row[7],
            )

    async def list_sites(self, enabled_only: bool = True) -> list[MonitoringSite]:
        await self.initialize()
        async with aiosqlite.connect(self._db_path) as db:
            query = "SELECT site_id, name, lat, lng, enabled, created_at, last_checked, check_interval_hours FROM monitoring_sites"
            if enabled_only:
                query += " WHERE enabled = 1"
            cursor = await db.execute(query)
            rows = await cursor.fetchall()
            return [
                MonitoringSite(
                    site_id=row[0], name=row[1],
                    location=Location(lat=row[2], lng=row[3]),
                    enabled=bool(row[4]),
                    created_at=datetime.fromisoformat(row[5]),
                    last_checked=datetime.fromisoformat(row[6]) if row[6] else None,
                    check_interval_hours=row[7],
                )
                for row in rows
            ]

    async def remove_site(self, site_id: str) -> bool:
        await self.initialize()
        async with aiosqlite.connect(self._db_path) as db:
            cursor = await db.execute("DELETE FROM monitoring_sites WHERE site_id = ?", (site_id,))
            await db.commit()
            return cursor.rowcount > 0

    async def add_subscription(self, sub: Subscription) -> None:
        await self.initialize()
        async with aiosqlite.connect(self._db_path) as db:
            await db.execute(
                "INSERT OR REPLACE INTO subscriptions (site_id, user_id, severity_min, created_at, last_notified) VALUES (?, ?, ?, ?, ?)",
                (sub.site_id, sub.user_id, sub.severity_min, sub.created_at.isoformat(), sub.last_notified.isoformat() if sub.last_notified else None),
            )
            await db.commit()

    async def remove_subscription(self, site_id: str, user_id: str) -> bool:
        await self.initialize()
        async with aiosqlite.connect(self._db_path) as db:
            cursor = await db.execute(
                "DELETE FROM subscriptions WHERE site_id = ? AND user_id = ?",
                (site_id, user_id),
            )
            await db.commit()
            return cursor.rowcount > 0

    async def list_subscriptions(self, user_id: str = "anonymous") -> list[Subscription]:
        await self.initialize()
        async with aiosqlite.connect(self._db_path) as db:
            cursor = await db.execute(
                "SELECT site_id, user_id, severity_min, created_at, last_notified FROM subscriptions WHERE user_id = ?",
                (user_id,),
            )
            rows = await cursor.fetchall()
            return [
                Subscription(
                    site_id=row[0],
                    user_id=row[1],
                    severity_min=row[2],
                    created_at=datetime.fromisoformat(row[3]),
                    last_notified=datetime.fromisoformat(row[4]) if row[4] else None,
                )
                for row in rows
            ]

    async def get_pending_alerts(self, user_id: str = "anonymous") -> list[dict]:
        await self.initialize()
        subs = await self.list_subscriptions(user_id)
        alerts = []
        for sub in subs:
            state = await self.get_latest_state(sub.site_id)
            if state:
                alerts.append({
                    "site_id": sub.site_id,
                    "severity_min": sub.severity_min,
                    "last_notified": sub.last_notified.isoformat() if sub.last_notified else None,
                    "latest_state_timestamp": state.timestamp.isoformat(),
                    "health_score": state.overall_health_score,
                })
        return alerts

    async def store_state(self, state: EcosystemState) -> None:
        await self.initialize()
        async with aiosqlite.connect(self._db_path) as db:
            await db.execute(
                "INSERT OR REPLACE INTO ecosystem_states (id, site_id, timestamp, state_json) VALUES (?, ?, ?, ?)",
                (state.id, state.id, state.timestamp.isoformat(), state.model_dump_json()),
            )
            await db.execute(
                "UPDATE monitoring_sites SET last_checked = ? WHERE site_id = ?",
                (state.timestamp.isoformat(), state.id),
            )
            await db.commit()

    async def get_latest_state(self, site_id: str) -> EcosystemState | None:
        await self.initialize()
        async with aiosqlite.connect(self._db_path) as db:
            cursor = await db.execute(
                "SELECT state_json FROM ecosystem_states WHERE site_id = ? ORDER BY timestamp DESC LIMIT 1",
                (site_id,),
            )
            row = await cursor.fetchone()
            if not row:
                return None
            return EcosystemState.model_validate_json(row[0])

    async def get_state_history(self, site_id: str, limit: int = 30) -> list[EcosystemState]:
        await self.initialize()
        async with aiosqlite.connect(self._db_path) as db:
            cursor = await db.execute(
                "SELECT state_json FROM ecosystem_states WHERE site_id = ? ORDER BY timestamp DESC LIMIT ?",
                (site_id, limit),
            )
            rows = await cursor.fetchall()
            return [EcosystemState.model_validate_json(row[0]) for row in rows]
