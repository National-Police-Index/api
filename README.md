# National Police Index (NPI) API

A REST API over POST (Peace Officer Standards and Training) officer employment records for
**all loaded states**, plus a modular **entity-resolution** library that matches officer
mentions in incident reports to verified employment records.

The repo is **API-centric**: the FastAPI app (`api/`) is the primary entry point, and the
entity-resolution pipeline (`resolve/`) is a standalone, importable library that works either
from an input CSV (batch) or directly from a name (single), fetching candidates from the API.

```
api/        FastAPI app over the all_npi_states table (port 8001)
              app.py · database.py (AllStatesClient) · config.py
resolve/    entity-resolution library + CLI
              pipeline.py (PostMatcher)  candidates.py  features.py  scoring.py
              agency.py  validation.py  llm.py  client.py (NPIClient)
              explain.py (gate sections)  io.py  cli.py
              data/ · models/ (XGBoost + scaler)
shared/     models.py (Pydantic) · env.py (.env discovery)
etl/        Firestore → Supabase sync (full + incremental)
tools/      explore_tui.py · pipeline_runner.py  (manual exploration)
tests/      pytest suite (unit + opt-in integration)
notebooks/  EDA / scratch notebooks
run.py      single launcher (api | resolve)
archive/legacy_ca/   the deprecated CA-only `postie` stack (server/, database/, …)
```

> **Data:** `all_npi_states` (~3.7M rows, ~24 states) is mirrored from the NPI's Firestore by
> `etl/`. See **[PIPELINE.md](./PIPELINE.md)** for the full fetch → sync → serve → resolve flow.

## Quick start

> **Use the project venv (`venv/bin/python`) for everything** — API *and* pipeline. The venv has
> fastapi/uvicorn/supabase *and* the ML deps, and (unlike the global Python) **no tensorflow**, so
> it avoids the `sentence-transformers` import-deadlock. Don't run any of this with the global Python.

### 1. Serve the API

```bash
venv/bin/python run.py api    # :8001
# docs at http://localhost:8001/docs
```

`SUPABASE_KEY` is read from `.env` (any of repo-root `.env`, `resolve/.env`, or the archived
`server/.env`) via `shared/env.py`.

### 2. Run entity resolution

**From a name (direct, candidates fetched from the API):**

```bash
venv/bin/python run.py resolve from-name \
    --first Scott --last Lunger --state CA --year 2015 \
    --source-agency "Hayward Police Department" --api http://localhost:8001
# -> AUTO-MATCHED -> b04-j30 (Hayward Police Department)
```

**From a CSV (batch):**

```bash
venv/bin/python run.py resolve from-csv \
    --input resolve/data/input/involved_officers_2-2-2026.csv \
    --api http://localhost:8001 --default-state CA \
    --output-dir resolve/data/output/all_states
```

Outputs: `auto_matched.jsonl`, `early_filtered.jsonl`, `failed_entity_resolution.jsonl`,
`results.csv`.

### As a library

```python
from resolve import PostMatcher, build_mention

matcher = PostMatcher(api_url="http://localhost:8001")
verdict = matcher.resolve_one(build_mention({
    "first_name": "Scott", "last_name": "Lunger", "state": "CA",
    "incident_year": 2015, "source_agency": "Hayward Police Department",
}))
print(verdict.status, verdict.match)        # "auto_matched", {...}

results = matcher.resolve_batch(list_of_mentions)
```

`PostMatcher` takes injectable `client`, `scorer`, and `validator` dependencies, so it is easy
to test and reuse.

## How matching works (precision gates)

`OfficerMention` → candidates from the API → XGBoost scoring → gates → verdict.

| Stage | Rule |
|---|---|
| Stage 0 — early filter | route to review on: no state (when required), common surname, or ≥2 same-name persons (state-scoped) |
| Stage 1 — candidates | name net + temporal (±1yr). **Optional**: county filter (if a county is known & not CORRECTIONS), agency_type mask (if the data carries real types) |
| Stage 2 — scoring | XGBoost match probability; threshold |
| Stage 3 — exact-name gate + ambiguity guard | require exact first+last; ≥2 distinct exact-name persons → review |
| Stage 4 — agency validation | deterministic non-LE guard (DA/Coroner/… ✗ LE) then LLM agency-equivalence check |

The county / agency_type filters are **optional** — used when the data has them (the rich CA
data), silently skipped otherwise (the lean all-states data). Same code, both regimes.

**Gate visibility.** Every result records which gates fired. `MentionResult` carries a `gates`
checklist (mention-level stages: state, common-name, same-name-in-state, candidates-found,
exact-name, ambiguity, agency) and `ambiguous`, and each entry in `candidates` is annotated with
`above_threshold`, `exact_name`, `is_best`, `agency_valid`. `resolve.explain.gate_sections()`
groups candidates by how many gates they cleared (most-central = passed all = the auto-match).

## Interactive TUI

`tools/explore_tui.py` — **Search** mode (direct API queries) and **Pipeline** mode (runs the
real `resolve` pipeline on one mention). Pipeline mode renders each candidate as a row with
**per-gate columns** (`common · uniq · thr · exact · ambig · best · agency`; ✓ passed / ⚑ flagged
/ ✗ failed / · n/a), best-first. Early-stop cases (e.g. common surname) show one mention row with
the flagged gate.

```bash
venv/bin/python run.py api                                          # :8001 (terminal 1)
NPI_API_URL=http://localhost:8001 venv/bin/python tools/explore_tui.py   # (terminal 2)
```

## Testing

```bash
venv/bin/python -m pytest                    # fast unit suite (no network)
venv/bin/python -m pytest -m integration     # opt-in: hits live Supabase / API / LLM
```

Integration tests skip cleanly when the API isn't reachable. Parity tests (`test_parity.py`)
compare the all-states API against the legacy postie API when both are up.

## API endpoints

`GET /post/employment` · `GET /post/employment/count` · `GET /post/candidates` ·
`GET /post/officers/by-name` (optional `state`) · `GET /post/agency/county` (null for
all-states) · `GET /post/stats` · batch variants under `/post/.../batch`.

## Environment

| var | purpose |
|---|---|
| `NPI_API_URL` | which API the resolve pipeline hits (default `http://localhost:8000`; use `:8001`) |
| `NPI_ALL_STATES_PORT` | API port (default 8001) |
| `OPENAI_API_KEY` | agency-validation LLM (in `resolve/.env`) |
| `SUPABASE_KEY` | Supabase REST key (in a discoverable `.env`) |
| `DEFAULT_STATE` | fallback state for CSV inputs without a `state` column |

## Legacy CA stack

The original California-only `postie` stack (`server/`, `database/`, the old `match.py`) has
been superseded by the all-states stack and moved to `archive/legacy_ca/` (validated head-to-head:
0 false positives, equal/better recall — see `plans/` and PIPELINE.md). It is kept for reference
only and is not part of the active build.
