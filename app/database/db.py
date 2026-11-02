"""
Database Layer for SpeechBench
Initializes SQLite database and tables for persisting benchmark runs and evaluation records.
"""

import os
import sqlite3
import json
from typing import Dict, Any, List, Optional
from datetime import datetime

DB_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "results", "speechbench.db")


def get_connection(db_path: str = DB_PATH) -> sqlite3.Connection:
    """Returns a SQLite connection with dict-like row factory."""
    os.makedirs(os.path.dirname(os.path.abspath(db_path)), exist_ok=True)
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    return conn


def init_db(db_path: str = DB_PATH) -> None:
    """Initializes tables if they do not exist."""
    conn = get_connection(db_path)
    cursor = conn.cursor()

    cursor.execute("""
    CREATE TABLE IF NOT EXISTS benchmark_runs (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        run_name TEXT NOT NULL,
        timestamp TEXT NOT NULL,
        device TEXT,
        gpu_name TEXT,
        models TEXT,
        total_samples INTEGER,
        summary_json TEXT
    );
    """)

    cursor.execute("""
    CREATE TABLE IF NOT EXISTS evaluation_records (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        run_id INTEGER,
        model_name TEXT NOT NULL,
        audio_path TEXT NOT NULL,
        condition TEXT NOT NULL,
        snr_db REAL,
        accent TEXT,
        domain_category TEXT,
        duration_seconds REAL,
        latency_seconds REAL,
        rtf REAL,
        raw_reference TEXT,
        raw_prediction TEXT,
        normalized_reference TEXT,
        normalized_prediction TEXT,
        wer REAL,
        cer REAL,
        substitutions INTEGER,
        deletions INTEGER,
        insertions INTEGER,
        hits INTEGER,
        reference_words INTEGER,
        entity_error_rate REAL,
        alignments_json TEXT,
        created_at TEXT NOT NULL,
        FOREIGN KEY (run_id) REFERENCES benchmark_runs (id)
    );
    """)

    conn.commit()
    conn.close()


class BenchmarkRepository:
    """Repository class for inserting and querying benchmark evaluation results."""

    def __init__(self, db_path: str = DB_PATH):
        self.db_path = db_path
        init_db(self.db_path)

    def create_run(
        self,
        run_name: str,
        device: str,
        gpu_name: str,
        models: List[str],
        total_samples: int,
        summary: Optional[Dict[str, Any]] = None,
    ) -> int:
        """Creates a benchmark run record and returns run ID."""
        conn = get_connection(self.db_path)
        cursor = conn.cursor()
        now = datetime.utcnow().isoformat()

        cursor.execute(
            """
            INSERT INTO benchmark_runs (run_name, timestamp, device, gpu_name, models, total_samples, summary_json)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                run_name,
                now,
                device,
                gpu_name,
                json.dumps(models),
                total_samples,
                json.dumps(summary or {}),
            ),
        )
        run_id = cursor.lastrowid
        conn.commit()
        conn.close()
        return run_id

    def insert_record(self, run_id: Optional[int], record: Dict[str, Any]) -> int:
        """Inserts an evaluation record for a single audio file and model."""
        conn = get_connection(self.db_path)
        cursor = conn.cursor()
        now = datetime.utcnow().isoformat()

        cursor.execute(
            """
            INSERT INTO evaluation_records (
                run_id, model_name, audio_path, condition, snr_db, accent, domain_category,
                duration_seconds, latency_seconds, rtf, raw_reference, raw_prediction,
                normalized_reference, normalized_prediction, wer, cer, substitutions,
                deletions, insertions, hits, reference_words, entity_error_rate,
                alignments_json, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                run_id,
                record.get("model_name"),
                record.get("audio_path"),
                record.get("condition", "clean"),
                record.get("snr_db"),
                record.get("accent", "standard"),
                record.get("domain_category", "general"),
                record.get("duration_seconds", 0.0),
                record.get("latency_seconds", 0.0),
                record.get("rtf", 0.0),
                record.get("raw_reference", ""),
                record.get("raw_prediction", ""),
                record.get("normalized_reference", ""),
                record.get("normalized_prediction", ""),
                record.get("wer", 0.0),
                record.get("cer", 0.0),
                record.get("substitutions", 0),
                record.get("deletions", 0),
                record.get("insertions", 0),
                record.get("hits", 0),
                record.get("reference_words", 0),
                record.get("entity_error_rate", 0.0),
                json.dumps(record.get("alignments", [])),
                now,
            ),
        )
        record_id = cursor.lastrowid
        conn.commit()
        conn.close()
        return record_id

    def insert_records_batch(self, run_id: Optional[int], records: List[Dict[str, Any]]) -> None:
        """Batch inserts multiple evaluation records."""
        for r in records:
            self.insert_record(run_id, r)

    def get_all_records(self) -> List[Dict[str, Any]]:
        """Fetches all evaluation records as dictionaries."""
        conn = get_connection(self.db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM evaluation_records ORDER BY id DESC")
        rows = [dict(row) for row in cursor.fetchall()]
        conn.close()
        return rows

    def get_runs(self) -> List[Dict[str, Any]]:
        """Fetches all benchmark runs."""
        conn = get_connection(self.db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM benchmark_runs ORDER BY id DESC")
        rows = [dict(row) for row in cursor.fetchall()]
        conn.close()
        return rows
