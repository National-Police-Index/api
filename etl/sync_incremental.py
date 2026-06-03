"""
Incremental Firestore -> Supabase sync (per-STATE). Cron-friendly.

Why per-state, not per-row: db_launch docs have NO modified timestamp, so there is
no per-row change signal. But `statistics_per_state.<state>.last_updated` IS a reliable
per-state signal, and upstream publishes whole-state CSV files. So we detect change at
the state level and re-pull only changed states.

Each run:
  1. read every state's `last_updated` from statistics_per_state
  2. compare to what we last synced (sync_state_meta table)
  3. a state needs resync if: new, OR last_updated advanced, OR live doc count != stored count
  4. for each changed state: pull its db_launch docs, diff against the table by content,
     and apply ONLY the deltas (insert new / update genuinely-changed / delete removed)
  5. unchanged states do ZERO Firestore doc reads and ZERO DB writes

On a typical night nothing has changed (NPI updates ~annually per state), so the run is
~50 tiny stat reads + ~50 count() aggregations and finishes in seconds.

Run from the repo root with the project venv:
    venv/bin/python etl/sync_incremental.py                 # normal incremental run
    venv/bin/python etl/sync_incremental.py --init-baseline # record current state as baseline (no data pull)
    venv/bin/python etl/sync_incremental.py --dry-run       # report what WOULD change, do nothing
"""

import time
import argparse

from common import (
    log, firestore_client, pg_connect, doc_to_row, copy_rows, state_doc_count,
    COLUMNS, COLLECTION_GROUP, STATS_COLLECTION, LIVE_TABLE, META_TABLE,
)

BATCH = 5000


def ensure_meta_table(conn):
    with conn.cursor() as cur:
        cur.execute(f"""
            CREATE TABLE IF NOT EXISTS {META_TABLE} (
                state        text PRIMARY KEY,
                last_updated timestamptz,
                row_count    bigint,
                synced_at    timestamptz DEFAULT now()
            );
        """)
    conn.commit()


def read_stats_last_updated(fs):
    """{state -> last_updated} from statistics_per_state."""
    out = {}
    for doc in fs.collection(STATS_COLLECTION).stream():
        out[doc.id] = (doc.to_dict() or {}).get("last_updated")
    return out


def read_meta(conn):
    with conn.cursor() as cur:
        cur.execute(f"SELECT state, last_updated, row_count FROM {META_TABLE};")
        return {s: (lu, rc) for s, lu, rc in cur.fetchall()}


def table_state_counts(conn):
    with conn.cursor() as cur:
        cur.execute(f"SELECT state, count(*) FROM {LIVE_TABLE} GROUP BY state;")
        return {s: c for s, c in cur.fetchall()}


def upsert_meta(conn, state, last_updated, row_count):
    with conn.cursor() as cur:
        cur.execute(f"""
            INSERT INTO {META_TABLE} (state, last_updated, row_count, synced_at)
            VALUES (%s, %s, %s, now())
            ON CONFLICT (state) DO UPDATE
              SET last_updated = EXCLUDED.last_updated,
                  row_count    = EXCLUDED.row_count,
                  synced_at    = now();
        """, (state, last_updated, row_count))
    conn.commit()


def resync_state(conn, cg, state):
    """Pull all db_launch docs for `state`, diff against the table, apply only deltas.
    Returns (inserted_or_updated_estimate, deleted)."""
    from google.cloud.firestore_v1.base_query import FieldFilter

    cols = ",".join(COLUMNS)
    update_set = ",\n".join(f"{c}=EXCLUDED.{c}" for c in COLUMNS if c != "fs_path")

    # Whole state resync runs as ONE transaction: the diff applies atomically, and
    # _incoming (ON COMMIT DROP) is cleaned up by the final commit. We COPY straight
    # into _incoming (fs_paths are unique within a state) without per-batch commits.
    with conn.cursor() as cur:
        cur.execute("DROP TABLE IF EXISTS _incoming;")
        cur.execute(f"CREATE TEMP TABLE _incoming (LIKE {LIVE_TABLE} INCLUDING DEFAULTS) ON COMMIT DROP;")

    # stream the state's docs into _incoming, paginated (no commit between batches)
    last_snap = None
    pulled = 0
    base_q = cg.where(filter=FieldFilter("state", "==", state)).order_by("__name__")
    while True:
        q = base_q.limit(BATCH)
        if last_snap is not None:
            q = q.start_after(last_snap)
        snaps = list(q.stream())
        if not snaps:
            break
        copy_rows(conn, "_incoming", [doc_to_row(s) for s in snaps])
        last_snap = snaps[-1]
        pulled += len(snaps)

    with conn.cursor() as cur:
        # DELETE rows no longer present upstream
        cur.execute(f"""
            DELETE FROM {LIVE_TABLE} a
            WHERE a.state = %s
              AND NOT EXISTS (SELECT 1 FROM _incoming i WHERE i.fs_path = a.fs_path);
        """, (state,))
        deleted = cur.rowcount
        # INSERT new + UPDATE only genuinely-changed rows (raw IS DISTINCT FROM)
        cur.execute(f"""
            INSERT INTO {LIVE_TABLE} ({cols})
            SELECT {cols} FROM _incoming
            ON CONFLICT (fs_path) DO UPDATE
              SET {update_set}, synced_at = now()
              WHERE {LIVE_TABLE}.raw IS DISTINCT FROM EXCLUDED.raw;
        """)
        changed = cur.rowcount  # rows inserted or actually updated
    conn.commit()
    return pulled, changed, deleted


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--init-baseline", action="store_true",
                    help="record current table+stats as the baseline; pull NO data")
    ap.add_argument("--dry-run", action="store_true",
                    help="report which states would resync; change nothing")
    args = ap.parse_args()

    log("Connecting...")
    fs = firestore_client()
    cg = fs.collection_group(COLLECTION_GROUP)
    conn = pg_connect()
    ensure_meta_table(conn)

    stats_lu = read_stats_last_updated(fs)
    log(f"statistics_per_state has {len(stats_lu)} states.")

    if args.init_baseline:
        counts = table_state_counts(conn)
        for state, lu in stats_lu.items():
            upsert_meta(conn, state, lu, counts.get(state, 0))
        log(f"Baseline recorded for {len(stats_lu)} states (no data pulled). Nightly runs will now be incremental.")
        conn.close()
        return

    meta = read_meta(conn)
    if not meta:
        log("WARNING: sync_state_meta is empty. Run with --init-baseline first "
            "(or every state will be treated as new and fully re-pulled).")

    # Decide which states need a resync.
    to_sync = []
    for state, live_lu in stats_lu.items():
        stored = meta.get(state)
        if stored is None:
            to_sync.append((state, "new state"))
            continue
        stored_lu, stored_rc = stored
        if live_lu is not None and stored_lu is not None and live_lu > stored_lu:
            to_sync.append((state, f"last_updated advanced ({stored_lu} -> {live_lu})"))
            continue
        # fallback: count check catches a missed last_updated bump
        live_rc = state_doc_count(cg, state)
        if live_rc != (stored_rc or 0):
            to_sync.append((state, f"row count changed ({stored_rc} -> {live_rc})"))

    if not to_sync:
        log("No states changed. Nothing to do.")
        conn.close()
        return

    log(f"{len(to_sync)} state(s) need resync:")
    for state, reason in to_sync:
        log(f"  - {state}: {reason}")

    if args.dry_run:
        log("Dry run — no changes made.")
        conn.close()
        return

    t0 = time.time()
    tot_pulled = tot_changed = tot_deleted = 0
    for state, reason in to_sync:
        log(f"Resyncing '{state}' ({reason})...")
        pulled, changed, deleted = resync_state(conn, cg, state)
        new_rc = state_doc_count(cg, state)
        upsert_meta(conn, state, stats_lu.get(state), new_rc)
        log(f"  '{state}': pulled {pulled:,} | written(new+changed) {changed:,} | deleted {deleted:,}")
        tot_pulled += pulled; tot_changed += changed; tot_deleted += deleted

    log(f"DONE in {(time.time()-t0)/60:.1f} min. "
        f"Across {len(to_sync)} state(s): pulled {tot_pulled:,}, "
        f"written {tot_changed:,}, deleted {tot_deleted:,}.")
    conn.close()


if __name__ == "__main__":
    main()
