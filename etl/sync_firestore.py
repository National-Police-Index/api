"""
Firestore -> Supabase FULL RELOAD (bootstrap / rebuild).

Mirrors the NPI's live Firestore collection-group `db_launch` (all states, all
employment records) into a Supabase Postgres table `all_npi_states`.

Strategy: full reload via staging table + atomic swap.
  1. (re)create `all_npi_states_staging`
  2. stream db_launch ordered by __name__, paginated + checkpointed (resumable)
  3. COPY each batch into staging
  4. atomic swap: drop old live table, rename staging -> live, build indexes

For routine refreshes use `sync_incremental.py` instead (per-state, cron-friendly).
This full reload is for the first build or a from-scratch rebuild.

Run from the repo root with the project venv:
    venv/bin/python etl/sync_firestore.py            # resumes from checkpoint if present
    venv/bin/python etl/sync_firestore.py --fresh    # ignore checkpoint, rebuild staging
"""

import os
import time
import argparse

from common import (
    log, firestore_client, pg_connect, doc_to_row, copy_into,
    COLUMNS, COLLECTION_GROUP, LIVE_TABLE, REPO,
)

STAGING_TABLE = "all_npi_states_staging"
BATCH = 5000
CHECKPOINT = os.path.join(REPO, "etl", ".sync_checkpoint")


def create_staging(conn, fresh):
    with conn.cursor() as cur:
        if fresh:
            cur.execute(f"DROP TABLE IF EXISTS {STAGING_TABLE};")
        cur.execute(f"""
            CREATE TABLE IF NOT EXISTS {STAGING_TABLE} (
                fs_path           text PRIMARY KEY,
                document_id       text,
                person_nbr        text,
                first_name        text,
                middle_name       text,
                last_name         text,
                agency_name       text,
                state             text,
                start_date        text,
                end_date          text,
                start_date_iso    timestamptz,
                end_date_iso      timestamptz,
                separation_reason text,
                search_queries    jsonb,
                raw               jsonb,
                synced_at         timestamptz DEFAULT now()
            );
        """)
    conn.commit()


def read_checkpoint():
    if os.path.exists(CHECKPOINT):
        with open(CHECKPOINT) as f:
            return f.read().strip() or None
    return None


def write_checkpoint(path):
    with open(CHECKPOINT, "w") as f:
        f.write(path)


def swap(conn):
    log("Building indexes on staging + atomic swap to live table...")
    with conn.cursor() as cur:
        # pg_trgm trigram GIN indexes make the entity-res ILIKE name queries fast enough
        # to survive the pipeline's concurrency (without them, queries time out under load).
        cur.execute("CREATE EXTENSION IF NOT EXISTS pg_trgm;")
        cur.execute(f"CREATE INDEX IF NOT EXISTS idx_{STAGING_TABLE}_last ON {STAGING_TABLE} (last_name);")
        cur.execute(f"CREATE INDEX IF NOT EXISTS idx_{STAGING_TABLE}_last_lc ON {STAGING_TABLE} (lower(last_name));")
        cur.execute(f"CREATE INDEX IF NOT EXISTS idx_{STAGING_TABLE}_person ON {STAGING_TABLE} (person_nbr);")
        cur.execute(f"CREATE INDEX IF NOT EXISTS idx_{STAGING_TABLE}_state ON {STAGING_TABLE} (state);")
        cur.execute(f"CREATE INDEX IF NOT EXISTS idx_{STAGING_TABLE}_agency ON {STAGING_TABLE} (agency_name);")
        cur.execute(f"CREATE INDEX IF NOT EXISTS idx_{STAGING_TABLE}_fn_trgm ON {STAGING_TABLE} USING gin (first_name gin_trgm_ops);")
        cur.execute(f"CREATE INDEX IF NOT EXISTS idx_{STAGING_TABLE}_ln_trgm ON {STAGING_TABLE} USING gin (last_name gin_trgm_ops);")
        cur.execute(f"DROP TABLE IF EXISTS {LIVE_TABLE} CASCADE;")
        cur.execute(f"ALTER TABLE {STAGING_TABLE} RENAME TO {LIVE_TABLE};")
        for suf in ["last", "last_lc", "person", "state", "agency", "fn_trgm", "ln_trgm"]:
            cur.execute(
                f"ALTER INDEX IF EXISTS idx_{STAGING_TABLE}_{suf} RENAME TO idx_{LIVE_TABLE}_{suf};"
            )
    conn.commit()
    log("Swap complete.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fresh", action="store_true", help="ignore checkpoint, rebuild staging from scratch")
    args = ap.parse_args()

    if args.fresh and os.path.exists(CHECKPOINT):
        os.remove(CHECKPOINT)

    log("Connecting to Firestore...")
    fs = firestore_client()
    cg = fs.collection_group(COLLECTION_GROUP)

    log("Counting source docs...")
    total = cg.count().get()[0][0].value
    log(f"Source db_launch docs: {total:,}")

    log("Connecting to Postgres (session pooler)...")
    conn = pg_connect()
    create_staging(conn, fresh=args.fresh)

    last_path = read_checkpoint()
    if last_path:
        log(f"Resuming after checkpoint: {last_path}")

    done = 0
    t0 = time.time()
    last_snap = None

    if last_path:
        last_snap = fs.document(last_path).get()
        with conn.cursor() as cur:
            cur.execute(f"SELECT count(*) FROM {STAGING_TABLE};")
            done = cur.fetchone()[0]

    while True:
        q = cg.order_by("__name__").limit(BATCH)
        if last_snap is not None:
            q = q.start_after(last_snap)
        snaps = list(q.stream())
        if not snaps:
            break

        copy_into(conn, STAGING_TABLE, [doc_to_row(s) for s in snaps])

        last_snap = snaps[-1]
        write_checkpoint(last_snap.reference.path)
        done += len(snaps)

        rate = done / max(time.time() - t0, 1e-6)
        eta = (total - done) / rate if rate else 0
        log(f"  loaded {done:,}/{total:,} ({100*done/total:.1f}%) | {rate:.0f} rows/s | ETA {eta/60:.1f} min")

    log(f"All batches loaded ({done:,} rows). Swapping into {LIVE_TABLE}...")
    swap(conn)

    with conn.cursor() as cur:
        cur.execute(f"SELECT count(*) FROM {LIVE_TABLE};")
        final = cur.fetchone()[0]
    log(f"DONE. {LIVE_TABLE} now has {final:,} rows (source had {total:,}).")

    if os.path.exists(CHECKPOINT):
        os.remove(CHECKPOINT)
    conn.close()


if __name__ == "__main__":
    main()
