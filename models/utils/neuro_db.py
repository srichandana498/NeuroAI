"""
NeuroAI - neuro_db.py
Backbone data layer for the new modules: Behavior Timeline, Parent Journal
Intelligence, Unified Neuro Score, and Daily Planner.

PLACE AT: NeuroAI_Project\\models\\utils\\neuro_db.py

Uses the same neuroai.db SQLite file as the rest of the app. This module
only CREATEs new tables — it never touches your existing tables/data, so
it's safe to drop in alongside session_store.py etc.
"""

import sqlite3
import json
import os
import re
from datetime import datetime

# neuroai.db lives at the project root, three levels up from this file
DB_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "neuroai.db",
)

DOMAINS = [
    "attention", "executive_function", "communication", "language",
    "motor_skills", "sensory_processing", "learning", "social_interaction",
    "emotional_regulation", "sleep", "behavior",
]

# ── Rule-based journal extraction ─────────────────────────────────────────
# Deliberately simple keyword matching so Parent Journal Intelligence works
# out of the box with no API key / no LLM dependency, and never asserts a
# diagnosis or a cause — only flags patterns for the parent to review.
# Swap in extract_with_llm() later without changing callers (same shape).
TRIGGER_WORDS = ["school", "loud", "noise", "transition", "homework", "crowd",
                  "sibling", "screen", "bedtime", "change of plan", "party"]
MOOD_WORDS = {"meltdown": "distressed", "tantrum": "distressed", "happy": "positive",
              "calm": "positive", "cried": "distressed", "anxious": "anxious",
              "excited": "positive", "frustrated": "distressed", "proud": "positive"}
SLEEP_WORDS = ["slept", "nap", "bedtime", "woke up", "insomnia", "night waking", "tired"]
FOOD_WORDS = ["ate", "meal", "snack", "breakfast", "lunch", "dinner", "hungry", "refused food"]


class NeuroDB:
    def __init__(self, db_path=None):
        self.db_path = db_path or DB_PATH
        self._init_schema()

    def _conn(self):
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_schema(self):
        conn = self._conn()
        conn.executescript("""
        CREATE TABLE IF NOT EXISTS events (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id TEXT NOT NULL,
            type TEXT NOT NULL,
            source TEXT DEFAULT 'parent',
            payload TEXT,
            ts TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS journal_entries (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id TEXT NOT NULL,
            text TEXT NOT NULL,
            extracted TEXT,
            ts TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS neuro_scores (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id TEXT NOT NULL,
            domain TEXT NOT NULL,
            score REAL NOT NULL,
            confidence REAL DEFAULT 0.6,
            ts TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS planner_blocks (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id TEXT NOT NULL,
            day TEXT NOT NULL,
            block_json TEXT NOT NULL,
            ts TEXT NOT NULL
        );

        CREATE INDEX IF NOT EXISTS idx_events_user ON events(user_id);
        CREATE INDEX IF NOT EXISTS idx_journal_user ON journal_entries(user_id);
        CREATE INDEX IF NOT EXISTS idx_scores_user ON neuro_scores(user_id);
        """)
        conn.commit()
        conn.close()

    # ---------- Events / Behavior Timeline ----------
    def add_event(self, user_id, event_type, payload=None, source="parent"):
        conn = self._conn()
        conn.execute(
            "INSERT INTO events (user_id, type, source, payload, ts) VALUES (?,?,?,?,?)",
            (user_id, event_type, source, json.dumps(payload or {}), datetime.utcnow().isoformat()),
        )
        conn.commit()
        conn.close()

    def get_timeline(self, user_id, limit=100):
        conn = self._conn()
        rows = conn.execute(
            "SELECT type, source, payload, ts FROM events WHERE user_id=? ORDER BY ts DESC LIMIT ?",
            (user_id, limit),
        ).fetchall()
        conn.close()
        return [
            {"type": r["type"], "source": r["source"],
             "payload": json.loads(r["payload"] or "{}"), "ts": r["ts"]}
            for r in rows
        ]

    def get_events_by_type(self, user_id, event_type, limit=50):
        """Used by Teacher Portal / Therapist Dashboard to pull just their
        own log entries out of the shared timeline."""
        conn = self._conn()
        rows = conn.execute(
            "SELECT type, source, payload, ts FROM events WHERE user_id=? AND type=? ORDER BY ts DESC LIMIT ?",
            (user_id, event_type, limit),
        ).fetchall()
        conn.close()
        return [
            {"type": r["type"], "source": r["source"],
             "payload": json.loads(r["payload"] or "{}"), "ts": r["ts"]}
            for r in rows
        ]

    # ---------- Parent Journal Intelligence ----------
    def add_journal_entry(self, user_id, text):
        extracted = self._extract(text)
        conn = self._conn()
        ts = datetime.utcnow().isoformat()
        conn.execute(
            "INSERT INTO journal_entries (user_id, text, extracted, ts) VALUES (?,?,?,?)",
            (user_id, text, json.dumps(extracted), ts),
        )
        conn.commit()
        conn.close()
        # every journal entry is also a timeline event
        self.add_event(user_id, "journal", {"summary": extracted, "text": text[:200]}, source="parent")
        return extracted

    def get_journal_entries(self, user_id, limit=50):
        conn = self._conn()
        rows = conn.execute(
            "SELECT text, extracted, ts FROM journal_entries WHERE user_id=? ORDER BY ts DESC LIMIT ?",
            (user_id, limit),
        ).fetchall()
        conn.close()
        return [{"text": r["text"], "extracted": json.loads(r["extracted"] or "{}"), "ts": r["ts"]} for r in rows]

    def _extract(self, text):
        t = text.lower()
        triggers = [w for w in TRIGGER_WORDS if w in t]
        mood_hits = [label for word, label in MOOD_WORDS.items() if word in t]
        mood = mood_hits[0] if mood_hits else "unspecified"
        sleep_mentioned = any(w in t for w in SLEEP_WORDS)
        food_mentioned = any(w in t for w in FOOD_WORDS)
        meltdown_count = len(re.findall(r"meltdown|tantrum", t))
        return {
            "triggers": triggers,
            "mood": mood,
            "sleep_mentioned": sleep_mentioned,
            "food_mentioned": food_mentioned,
            "meltdown_mentions": meltdown_count,
            "note": "Auto-extracted with keyword matching — a pattern to review, not a conclusion.",
        }

    def extract_with_llm(self, text, llm_call_fn):
        """Optional upgrade: pass any function(text) -> dict with the same
        keys as _extract() to swap in LLM-based extraction later."""
        return llm_call_fn(text)

    # ---------- Unified Neuro Score ----------
    def save_domain_scores(self, user_id, scores: dict, confidence=0.6):
        """scores: {"attention": 0.7, "sleep": 0.4, ...} — values 0-1"""
        conn = self._conn()
        ts = datetime.utcnow().isoformat()
        for domain, score in scores.items():
            conn.execute(
                "INSERT INTO neuro_scores (user_id, domain, score, confidence, ts) VALUES (?,?,?,?,?)",
                (user_id, domain, float(score), float(confidence), ts),
            )
        conn.commit()
        conn.close()
        self.add_event(user_id, "neuro_score_update", {"scores": scores}, source="system")

    def get_latest_neuro_score(self, user_id):
        conn = self._conn()
        rows = conn.execute(
            """
            SELECT domain, score, confidence, ts FROM neuro_scores AS ns
            WHERE user_id=? AND ts = (
                SELECT MAX(ts) FROM neuro_scores AS n2
                WHERE n2.user_id = ns.user_id AND n2.domain = ns.domain
            )
            ORDER BY domain
            """,
            (user_id,),
        ).fetchall()
        conn.close()
        return {r["domain"]: {"score": r["score"], "confidence": r["confidence"], "ts": r["ts"]} for r in rows}

    def get_score_trend(self, user_id, domain, limit=20):
        conn = self._conn()
        rows = conn.execute(
            "SELECT score, ts FROM neuro_scores WHERE user_id=? AND domain=? ORDER BY ts ASC LIMIT ?",
            (user_id, domain, limit),
        ).fetchall()
        conn.close()
        return [{"score": r["score"], "ts": r["ts"]} for r in rows]

    # ---------- Daily Planner ----------
    def save_planner(self, user_id, day, blocks):
        conn = self._conn()
        conn.execute(
            "INSERT INTO planner_blocks (user_id, day, block_json, ts) VALUES (?,?,?,?)",
            (user_id, day, json.dumps(blocks), datetime.utcnow().isoformat()),
        )
        conn.commit()
        conn.close()

    def get_planner(self, user_id, day):
        conn = self._conn()
        row = conn.execute(
            "SELECT block_json FROM planner_blocks WHERE user_id=? AND day=? ORDER BY ts DESC LIMIT 1",
            (user_id, day),
        ).fetchone()
        conn.close()
        return json.loads(row["block_json"]) if row else None