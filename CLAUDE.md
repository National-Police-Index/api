# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

This is the **National Police Index (NPI) API** - a multi-component system for accessing POST (Peace Officer Standards and Training) officer employment records and performing entity resolution to match officer mentions in incident reports to verified employment records.

The system consists of:
1. **FastAPI server** - REST API for querying POST employment data
2. **Entity resolution pipeline** - Machine learning-based matching system
3. **Supabase database** - PostgreSQL backend storing employment records
4. **Rust feature extraction** (optional) - Performance-optimized string similarity calculations

### One consolidated stack (2026-06 refactor) — read [PIPELINE.md](./PIPELINE.md)

The repo was consolidated onto a **single, API-centric stack**. The legacy CA `postie` stack was
archived to `archive/legacy_ca/` (gitignored). Current layout:

| | **Current (all-states)** |
|---|---|
| Table | `all_npi_states` (~3.7M rows, ~24 states; synced from NPI Firestore) |
| API | `api/` (FastAPI, port **8001**) — `app.py`, `database.py` (AllStatesClient), `config.py` |
| Pipeline | `resolve/` package — `pipeline.py` (`PostMatcher`), `candidates.py`, `scoring.py`, `validation.py`, `agency.py`, `client.py`, `io.py`, `cli.py` |
| Models | `shared/models.py` (Pydantic) ; `shared/env.py` (.env discovery) |
| Sync | `etl/` (Firestore → Supabase, full + incremental) |
| Tools | `tools/explore_tui.py`, `tools/pipeline_runner.py` |
| Launcher | `run.py` (`python run.py api` / `venv/bin/python run.py resolve …`) |

**The CA-specific richness (county filter, agency_type mask) is preserved but OPTIONAL** in the
consolidated `resolve.candidates.select_candidates` — applied when the data has it, skipped when
it doesn't. The pipeline points at an API via **`NPI_API_URL`**. Legacy CA code lives in
`archive/legacy_ca/` for reference only. **[PIPELINE.md](./PIPELINE.md)** documents the full flow.

## Development Commands

### Starting the API Server

```bash
venv/bin/python run.py api       # :8001
# docs at http://localhost:8001/docs
```

**Use `venv/bin/python` for everything** (API + pipeline). The venv now has
fastapi/uvicorn/supabase *and* the ML deps, with **no tensorflow** — so it avoids the
`sentence-transformers` import-deadlock that the global Python (which has tensorflow) triggers.
Do not run any component with the global Python.

### Running the Entity Resolution Pipeline

The API must be running first.

```bash
# from a CSV (batch)
venv/bin/python run.py resolve from-csv \
    --input resolve/data/input/involved_officers_2-2-2026.csv \
    --api http://localhost:8001 --default-state CA \
    --output-dir resolve/data/output/all_states [--sample-n 100 --sample-seed 42]

# from a name (single; candidates fetched from the API)
venv/bin/python run.py resolve from-name \
    --first Scott --last Lunger --state CA --year 2015 \
    --source-agency "Hayward Police Department" --api http://localhost:8001
```

`--default-state` supplies a fallback for inputs lacking a `state` column. Outputs:
`auto_matched.jsonl`, `early_filtered.jsonl`, `failed_entity_resolution.jsonl`, `results.csv`.

As a library: `from resolve import PostMatcher, build_mention` → `matcher.resolve_one(...)` /
`resolve_batch(...)`. `PostMatcher(client=…, scorer=…, validator=…)` are injectable for testing.

### Syncing the data

```bash
venv/bin/python etl/sync_firestore.py            # full reload (bootstrap/rebuild)
venv/bin/python etl/sync_incremental.py          # per-state incremental (cron-friendly)
```

### Running tests

```bash
venv/bin/python -m pytest                    # fast unit suite (no network)
venv/bin/python -m pytest -m integration     # opt-in: live Supabase / API / LLM (skips if down)
```

### Interactive TUI (manual exploration)

`tools/explore_tui.py` — **Search** mode (direct API queries) and **Pipeline** mode (runs the real
`resolve` pipeline on one typed-in mention via `tools/pipeline_runner.py`, an isolated subprocess
so heavy library output goes to the log pane, not the terminal). Pipeline mode renders each
candidate as a row with **per-gate columns** (`common · uniq · thr · exact · ambig · best ·
agency`; ✓ passed / ⚑ flagged / ✗ failed / · n/a), best-first; early-stop cases show one mention
row with the flagged gate.

```bash
venv/bin/python run.py api                                          # :8001
NPI_API_URL=http://localhost:8001 venv/bin/python tools/explore_tui.py
```

Run the TUI under the venv (pipeline mode imports `sentence_transformers`; needs `OPENAI_API_KEY`
in `resolve/.env`). The incident year must fall within the officer's years of service (candidates
are filtered to incident year ± 1).

### Building Rust Components (Optional)

```bash
cargo build --release
```

Rust features are optional - the Python pipeline uses PyO3 to call Rust implementations of string similarity functions (jellyfish compatibility) for performance.

### Running Tests

```bash
cd test
python3 src.py
```

## Architecture

### Component Structure

```
api/
├── PIPELINE.md      # ★ all-states flow (fetch → sync → serve → resolve)
├── run.py           # launcher: `python run.py api` | `venv/bin/python run.py resolve …`
├── api/             # FastAPI app over all_npi_states (port 8001)
│   ├── app.py       # endpoints
│   ├── database.py  # AllStatesClient (case-insensitive, state code↔name, no county/agency_type)
│   └── config.py
├── resolve/         # entity-resolution library + CLI
│   ├── pipeline.py  # PostMatcher (resolve_one/resolve_batch) + stage logic; injectable deps
│   ├── candidates.py# date/name/county/agency_type filters — county+agency_type OPTIONAL
│   ├── scoring.py   # XGBoost scorer (make_scorer; model+featurize injectable)
│   ├── features.py  # feature engineering (package-relative scaler path)
│   ├── agency.py    # deterministic non-LE guard (import-clean, no LLM)
│   ├── validation.py# agency validation: guard + LLM (injectable llm_fn)
│   ├── llm.py       # OpenAI wrapper, lazy client (gpt-5.4-nano)
│   ├── client.py    # NPIClient (base URL from NPI_API_URL)
│   ├── explain.py   # gate_sections() — group annotated candidates by gates cleared
│   ├── io.py        # CSV read, build_mention, write_outputs, common-name loader
│   ├── cli.py       # `from-csv` / `from-name` subcommands
│   └── data/, models/  # inputs + pickled XGBoost / scaler
├── shared/          # models.py (Pydantic) · env.py (.env discovery)
├── etl/             # Firestore → Supabase sync (sync_firestore / sync_incremental / common)
├── tools/           # explore_tui.py · pipeline_runner.py (manual exploration)
├── tests/           # pytest suite (unit + opt-in integration)
├── notebooks/       # EDA / scratch
├── archive/legacy_ca/  # deprecated CA `postie` stack (server/, database/, models/, resolve_src/, test/)
└── src/             # Rust performance optimizations (optional)
```

### Key Data Models

**PostEmploymentRecord**: Employment stint from POST database
- `post_person_nbr` - Unique officer identifier
- `post_first_name`, `post_middle_name`, `post_last_name`, `post_suffix` - Name fields
- `post_agency_name`, `post_agency_type` - Agency info
- `post_start_date`, `post_end_date` - Employment period
- `state`, `county` - Geographic location

**OfficerMention**: Officer mention from incident report
- `mention_uid` - Unique mention identifier
- `mention_first_name`, `mention_last_name`, etc. - Name fields
- `mention_agency` - Source agency
- `mention_incident_date` - Incident date
- `mentioned_agencies` - List of agencies mentioned in report

**AgencyType**: Enum with values `POLICE` or `CORRECTIONS`

### Database Layer (database/src.py)

The `SupabaseClient` class provides all database access:

**Primary query methods:**
- `get_post_employment_records(query)` - General POST data search
- `get_candidates_for_mention(query)` - Targeted candidate generation for entity resolution
- `get_officers_by_name(first_name, last_name)` - Full database name search (for manual review)
- `get_county_for_agency(agency_name)` - Geographic lookup for agency filtering

**Critical implementation detail**: `get_candidates_for_mention()` uses TWO queries to cast a wide net:
1. First name 2-char prefix match + exact last name
2. Exact first name + last name 2-char prefix match

Results are merged and deduplicated. This handles nickname variations and data entry inconsistencies.

### Entity Resolution Pipeline (resolve/src/match.py)

The pipeline runs through multiple stages:

**Stage 1: Candidate Generation** (`generate_candidates`)
1. Query API for candidates using name prefixes
2. Apply geographic filtering (county-based, POLICE only)
3. Apply temporal filtering (±1 year buffer from incident)
4. Filter by agency type (POLICE vs CORRECTIONS)
5. Return both filtered candidates AND full employment history

**Stage 2: Common Name Flagging**
- Check against top 100 CA common surnames (`data/input/common_last_names.csv`)
- Flag mentions with 2+ unique persons sharing exact same name
- These are routed to manual review, NOT auto-matched

**Stage 3: Model Scoring**
- Load XGBoost model from `models/best_model_xgboost.pkl`
- Engineer features via `featurize()` function (features.py)
- Predict match probability for all candidates
- Filter by threshold (>0.8)

**Stage 4: Agency Validation**
- For best match per mention, validate agency alignment via `validate_agency_match()` helper
- First applies a deterministic **non-LE guard**: rejects when every agency to compare against is non-LE (DA/Coroner/ME/Public Defender/AG) and the POST agency is LE (Police/Sheriff/Marshal/Patrol/etc.); bypassed when any LE agency is in `mentioned_agencies`
- Otherwise an **LLM** decides whether `post_agency_name` matches `mention_agency` OR any of `mentioned_agencies`. Uses the standard OpenAI API, model `gpt-5.4-nano` (`llm.py`); LLM response caching is disabled (always fresh)

**Output Files** (all in `resolve/data/output/interface/`):
- `df.csv` - All mentions with match results
- `unmatched.xlsx` - Officers needing manual review (multi-sheet Excel with candidates)
- `matched_clean_with_conflicts.xlsx` - Auto-matched but other officers share same name
- `matched_clean_no_conflicts.xlsx` - High-confidence auto-matches with unique names
- `matched_clean_no_conflicts.jsonl` - JSONL export for web interface integration
- `matching_summary_stats.txt` - Summary statistics

### Geographic Filtering Strategy

**POLICE agencies**: County-based filtering is applied
- Source agency is resolved to county via `get_county_for_agency()`
- Only candidates with ANY employment history in that county are considered
- Handles Office/Department variations (e.g., "Los Angeles County Sheriff's Office" vs "Department")

**CORRECTIONS agencies**: No geographic filtering
- Corrections officers may work across multiple facilities statewide
- County filtering is explicitly skipped when `agency_type == CORRECTIONS`

### Feature Engineering (resolve/src/features.py)

The `featurize()` function creates ML features from mention-candidate pairs:
- Name similarity scores (Jaro-Winkler, Levenshtein, fuzzy ratio)
- Date overlap calculations
- Agency name matching scores
- Middle name/suffix comparisons

Features must match the training set used for the pickled models in `models/`.

### Manual Review Triggers

Mentions are flagged for manual review when:
1. **Common last name** - Top 100 CA surnames
2. **Multiple plausible persons** - 2+ officers with exact same name at relevant agencies
3. **Agency validation failed** - POST agency doesn't match source/mentioned agencies
4. **No candidates found** - No matching records in database

## Important Implementation Notes

### Date Handling Edge Cases

Empty string end dates and dates before 1950 are treated as NaT, then filled with today's date:

```python
end_dates_cleaned = post.post_end_date.replace("", pd.NaT)
end_dates_cleaned = pd.to_datetime(end_dates_cleaned, errors='coerce')
end_dates_cleaned = end_dates_cleaned.where(
    (end_dates_cleaned.isna()) | (end_dates_cleaned.dt.year >= 1950),
    pd.NaT
)
end_dates_filled = end_dates_cleaned.fillna(pd.Timestamp.today())
```

This ensures current officers (null end date) and data quality issues are handled consistently.

### Officer UID Generation

If the input CSV lacks an `officer_uid` column, UIDs are generated via SHA256 hash:

```python
def generate_officer_uid(row: pd.Series) -> str:
    combined = f"{first_name}|{last_name}|{provisional_case_name}|{incident_year}|{incident_month}|{incident_date}|{source_agency}"
    return hashlib.sha256(combined.encode('utf-8')).hexdigest()
```

### Multi-Sheet Excel Output Structure

The `unmatched.xlsx` file contains:
- **Summary sheet** - Statistics breakdown
- **Per-officer sheets** - Each unmatched officer gets a dedicated sheet showing:
  - Officer info header with document link
  - All candidates with match probabilities
  - Full employment history for each candidate
  - Review reason (why auto-match failed)

## Input Data Requirements

The entity resolution pipeline expects a CSV at `resolve/data/input/involved_officers.csv` with columns:
- `first_name`, `middle_name`, `last_name` - Officer name
- `source_agency` - Agency from incident report
- `agency_type` - "POLICE" or "CORRECTIONS"
- `incident_year` - Year of incident (required)
- `state` - State code (e.g., "CA")
- `mentioned_agencies` - String representation of list of agency names from report
- `provisional_case_name` - Case identifier for linking to case documents
- `officer_uid` - Unique ID (auto-generated if missing)

## Environment Variables

Create a `.env` file in the `server/` directory:

```
SUPABASE_KEY=your_supabase_api_key_here
```

The Supabase URL is hardcoded in `server/config.py` but can be moved to .env if needed.

The entity-resolution pipeline needs a separate `.env` in `resolve/src/` for LLM agency validation:

```
OPENAI_API_KEY=your_openai_api_key_here
```

## Dependencies

**Python packages:**
- fastapi, uvicorn - API server
- supabase - Database client
- pandas, scikit-learn, xgboost - ML pipeline
- pydantic - Data validation
- python-dotenv - Environment management
- openpyxl - Excel file generation
- openai - LLM agency validation (`resolve/src/llm.py`)

**Interpreter note:** run the server with the system Python that has `fastapi`; run the entity-resolution pipeline (`resolve/src/match.py`) from the project **venv** (`venv/bin/python`). The pipeline imports `sentence_transformers`; if the interpreter also has TensorFlow installed it can deadlock at import in non-interactive shells, so prefer the venv which does not.

**Rust dependencies** (optional, for performance):
- polars - DataFrame operations
- rust-bert - Embeddings (experimental)
- pyo3 - Python interop

## API Endpoints Reference

### Core Endpoints

**GET /post/employment** - Query employment records
- Query params: `first_name`, `last_name`, `agency`, `state`, `limit`, `offset`

**GET /post/candidates** - Get targeted candidates for entity resolution
- Query params: `first_name`, `last_name`, `agency_type`, `start_year`, `end_year`, `state`

**GET /post/officers/by-name** - Get all officers with matching name
- Query params: `first_name`, `last_name`
- Used for manual review to find all potential matches

**GET /post/agency/county** - Look up county for an agency
- Query param: `agency_name`

**GET /post/employment/count** - Count records matching filters

**GET /post/stats** - Database statistics

## Troubleshooting

**Pipeline fails with connection error**: Ensure API server is running at localhost:8000

**No candidates found for obvious matches**: Check that the state filter isn't too restrictive and that agency_type is set correctly

**Agency validation rejecting valid matches**: Check that mentioned_agencies list includes the POST agency name. Validation is LLM-based (`helpers.py:validate_agency_match()` → `llm.py`, OpenAI `gpt-5.4-nano`); a deterministic non-LE guard runs first and is bypassed when an LE agency is present in mentioned_agencies. Requires `OPENAI_API_KEY` in `resolve/src/.env`.

**Empty Excel files**: Ensure there is data to write. The code skips creating files when there are no sheets to write.

**CORRECTIONS officers not matching**: Verify county filtering is being skipped. The code explicitly checks `if agency_type != "CORRECTIONS"` before applying county filters.

## Pipeline Layout (June 2026)

**Current / general flow = the all-states stack — see [PIPELINE.md](./PIPELINE.md).** It syncs the
NPI's live Firestore into Supabase (`all_npi_states`, ~24 states) and resolves any person in any
state, without `county` or `agency_type`. Validated head-to-head vs the legacy CA pipeline on 350
random rows: **0 false positives, 0 different-person matches**, recall slightly higher (the old
county/agency_type filters were over-excluding correct matches). Requires pg_trgm indexes (baked
into `etl/sync_firestore.py`) to avoid query timeouts under pipeline concurrency.

**All-states entry points:**
- `etl/sync_firestore.py` (full), `etl/sync_incremental.py` (per-state, cron) — Firestore → Supabase
- `server_all_states/` — API over `all_npi_states` (port 8001)
- `resolve/src/match_all_states.py` — entity resolution (outputs → `resolve/data/output/all_states/`)

**Legacy CA stack (`postie`):** the sequential pipeline in `resolve/` (against `postie`) remains
for reference. The previous batch-optimization experiments live under `resolve_archive/resolve_batch/`
(broken batch county lookup; see `resolve_archive/resolve_batch/COUNTY_LOOKUP_TODO.md`).
- `resolve/src/match.py` — legacy CA pipeline (chunked + checkpointed; formerly `match_fast.py`)
- `resolve/src/post_processing.py`, `filter_auto_matched.py` — post-match cleanup
- `resolve/src/tests.py`, `test_names.py` — groundtruth validation against `resolve/data/output/auto_matched.jsonl`

## Agent and Subagent Usage

### Subagent Configuration

Custom agents are defined in `.claude/agents/`. Each agent has:
- **One clear goal** - Single responsibility
- **Defined inputs/outputs** - What it receives and produces
- **Tool permissions** - Scoped to what it needs

| Agent | Purpose | Tools | Files Owned |
|-------|---------|-------|-------------|
| `@orchestrator` | Coordinate multi-phase work | Read, Grep, Glob, Edit, Task | None (delegates) |
| `@database-engineer` | Async Supabase, connection pooling | Read, Write, Edit, Bash | `database/src.py` |
| `@api-architect` | FastAPI batch endpoints | Read, Write, Edit, Bash | `server/src.py`, `models/src.py` |
| `@test-engineer` | Tests and benchmarks | Read, Write, Edit, Bash | `test/*.py` |
| `@code-reviewer` | Quality review | Read, Grep, Glob | None (read-only) |

### Parallel Execution via Task Tool

**IMPORTANT:** Subagents can run in parallel using the Task tool. Max 10 concurrent; additional tasks queue automatically.

**Invoke parallel execution:**
```
Deploy subagents in parallel:
- Task 1 (@database-engineer): Implement AsyncSupabaseClient in database/src.py
- Task 2 (@api-architect): Implement batch Pydantic models in models/src.py
```

**Safe to parallelize:**
- Different files with no dependencies
- Read-only analysis tasks
- Test files for different modules

**MUST be sequential:**
- Implementation depending on another agent's output
- Multiple agents writing to same file
- Tests before implementation exists

### When to Use Agents vs Direct Tools

**Use `Explore` agent when:**
- Understanding unfamiliar code: "How does candidate generation work?"
- Searching across many files for a concept
- Thoroughness levels: `"quick"` (1-2 files), `"medium"` (3-5), `"very thorough"` (10+)

**Use direct tools when:**
- Reading specific known files → `Read` tool
- Simple keyword search → `Grep` tool  
- Finding files by pattern → `Glob` tool
- Single-file edits → `Edit` tool

### Context Management

- Each subagent has isolated 200k token context window
- ~20k token overhead per subagent startup
- Use `/clear` or new session when switching major tasks
- Subagents return only results, not full context

### Workflow Pattern

```
1. claude --plan          # Define task, generate plan
2. Add plan to CLAUDE.md  # Single source of truth
3. @orchestrator          # Execute with parallel subagents where safe
```

### Documentation Fetching

Agents should fetch latest docs before implementing:
```
use context7 to get documentation for /supabase/supabase-py
use context7 to get documentation for /tiangolo/fastapi
```

### Pipeline-Specific Parallelization

**CAN run in parallel (independent operations):**

1. **Candidate queries** (per person)
   - ✅ Already batch-optimized
   - Each person's candidate generation is independent

2. **Feature engineering** (per mention-candidate pair)
   - Name similarity calculations
   - Date overlap computations
   - Agency matching scores
   - All embarrassingly parallel

3. **County lookups** (per agency)
   - Each agency's county lookup is independent
   - ⚠️ Not yet batch-optimized (low priority)

4. **Name validation queries** (per unique name)
   - Each name uniqueness check is independent
   - ⚠️ **Next optimization target** - should be batched

5. **ML model predictions** (batch scoring)
   - XGBoost already supports batch prediction
   - No additional optimization needed

**CANNOT run in parallel (sequential dependencies):**

1. **Pipeline stages must run in order:**
   - Stage 1: Candidate generation
   - Stage 2: Common name flagging (needs Stage 1 results)
   - Stage 3: ML scoring (needs Stage 2 filtered candidates)
   - Stage 4: Agency validation (needs Stage 3 best match)

2. **Chunked processing is sequential:**
   - Must complete chunk N before chunk N+1
   - Results appended incrementally to shared files
   - File locking would cause issues with parallel writes

3. **Database writes are sequential:**
   - CSV/JSONL appending must be serialized
   - Concurrent writes would corrupt output files

### Optimization Strategy for New Features

When adding database-intensive operations:

1. **First**: Implement sequentially (get it working)
2. **Profile**: Identify if it's a bottleneck (>10% of total time)
3. **Design**: Create batch endpoint (input: list, output: dict)
4. **Implement**: Use hash-based partitioning (O(n+m) not O(n×m))
5. **Test**: Validate with 100-item chunks, not full dataset
6. **Validate**: Compare outputs AND timing

## Profiling Best Practices (Summary)

**Critical Rule**: Always compare outputs AND timing, not just timing alone.

**Checklist for any optimization:**
1. Use identical input data (same random seed)
2. Profile on realistic size (1000+ persons)
3. Compare match counts, candidate counts, final IDs
4. Check for zero-result bugs
5. Document discrepancies
6. Test with chunking (100-item chunks)

**Common Pitfalls**:
- Using `set()` with Pydantic models (breaks) → Use `dict` with tuple keys
- Batching 1000+ items in single call (fails) → Use 100-item chunks
- Only comparing timing (insufficient) → Always validate outputs

**Template**:
```python
seq_results = profile_approach(mentions, "sequential")
batch_results = profile_approach(mentions, "batch")

# CRITICAL: Validate outputs match!
assert seq_results['match_ids'] == batch_results['match_ids']
print(f"Speedup: {seq_results['time']/batch_results['time']:.2f}x")
```

**📚 Full guide**: See "Performance Profiling Best Practices" in [BATCH_PROCESSING.md](./BATCH_PROCESSING.md)