"""Shared helpers for the Firestore -> Supabase ETL (full + incremental sync)."""

import io
import os
import csv
import json
import sys
import time

import firebase_admin
from firebase_admin import credentials, firestore
import psycopg2
from dotenv import dotenv_values

# ---- paths (relative to repo root) -----------------------------------------
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CRED_PATH = os.path.join(REPO, "firestore", "serviceAccountKey.json")
ENV_PATH = os.path.join(REPO, "server", ".env")

# ---- config -----------------------------------------------------------------
COLLECTION_GROUP = "db_launch"
STATS_COLLECTION = "statistics_per_state"
LIVE_TABLE = "all_npi_states"
META_TABLE = "sync_state_meta"
PG_REF = "jcbircleopydnrtikisa"
PG_HOST = "aws-0-us-east-1.pooler.supabase.com"
PG_PORT = 5432

# Typed columns lifted from each doc (everything also lands in `raw`).
COLUMNS = [
    "fs_path", "document_id", "person_nbr",
    "first_name", "middle_name", "last_name",
    "agency_name", "state",
    "start_date", "end_date", "start_date_iso", "end_date_iso",
    "separation_reason", "search_queries", "raw",
]


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def firestore_client():
    try:
        firebase_admin.get_app()
    except ValueError:
        firebase_admin.initialize_app(credentials.Certificate(CRED_PATH))
    return firestore.client()


def pg_connect():
    pwd = dotenv_values(ENV_PATH).get("db")
    if not pwd:
        sys.exit(f"Postgres password ('db') not found in {ENV_PATH}")
    return psycopg2.connect(
        host=PG_HOST, port=PG_PORT,
        user=f"postgres.{PG_REF}", password=pwd,
        dbname="postgres", sslmode="require", connect_timeout=15,
    )


def to_iso(v):
    """Firestore ISO fields come back as datetime or str; normalize to str or None."""
    if v in (None, ""):
        return None
    return str(v)


def doc_to_row(snap):
    """Map a Firestore db_launch snapshot to a row matching COLUMNS."""
    d = snap.to_dict() or {}
    sq = d.get("searchQueries")
    return [
        snap.reference.path,                 # fs_path (globally unique)
        d.get("document_id", ""),
        d.get("person_nbr", ""),
        d.get("first_name", ""),
        d.get("middle_name", ""),
        d.get("last_name", ""),
        d.get("agency_name", ""),
        d.get("state", ""),
        d.get("start_date", ""),
        d.get("end_date", ""),
        to_iso(d.get("start_date_iso")),
        to_iso(d.get("end_date_iso")),
        d.get("separation_reason", ""),
        json.dumps(sq) if sq is not None else None,
        json.dumps(d, default=str),
    ]


def rows_to_csv_buffer(rows):
    buf = io.StringIO()
    w = csv.writer(buf)
    for r in rows:
        w.writerow(["" if c is None else c for c in r])
    buf.seek(0)
    return buf


def copy_into(conn, table, rows):
    """COPY rows into `table` (must have COLUMNS), idempotent on fs_path."""
    if not rows:
        return
    buf = rows_to_csv_buffer(rows)
    cols = ",".join(COLUMNS)
    with conn.cursor() as cur:
        cur.execute(f"CREATE TEMP TABLE _b (LIKE {table} INCLUDING DEFAULTS) ON COMMIT DROP;")
        cur.copy_expert(f"COPY _b ({cols}) FROM STDIN WITH (FORMAT csv, NULL '')", buf)
        cur.execute(f"""
            INSERT INTO {table} ({cols})
            SELECT {cols} FROM _b
            ON CONFLICT (fs_path) DO NOTHING;
        """)
    conn.commit()


def copy_rows(conn, table, rows):
    """Plain COPY of rows straight into `table` (no dedup). Caller controls the txn/commit.
    Use for staging into a temp table whose fs_paths are already unique."""
    if not rows:
        return
    buf = rows_to_csv_buffer(rows)
    cols = ",".join(COLUMNS)
    with conn.cursor() as cur:
        cur.copy_expert(f"COPY {table} ({cols}) FROM STDIN WITH (FORMAT csv, NULL '')", buf)


def state_doc_count(cg, state):
    """Live count of db_launch docs for one state (cheap aggregation)."""
    from google.cloud.firestore_v1.base_query import FieldFilter
    q = cg.where(filter=FieldFilter("state", "==", state))
    return q.count().get()[0][0].value
