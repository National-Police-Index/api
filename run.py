#!/usr/bin/env python3
"""Single entry point for the NPI API repo.

    # serve the API (FastAPI over all_npi_states) — run with a Python that has fastapi
    python run.py api                          # :8001 (NPI_ALL_STATES_PORT to override)

    # entity resolution — run with the project venv (has the ML deps, no tensorflow)
    venv/bin/python run.py resolve from-csv  --input data.csv --api http://localhost:8001 --default-state CA
    venv/bin/python run.py resolve from-name --first Scott --last Lunger --state CA \
        --year 2015 --source-agency "Hayward Police Department" --api http://localhost:8001

The two stacks intentionally use different interpreters (see CLAUDE.md): the API needs
fastapi/uvicorn; the resolve pipeline needs sentence-transformers/xgboost and must avoid
the tensorflow-bearing global Python (import deadlock).
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def _run_api(argv):
    import uvicorn
    from api.app import app
    from api.config import PORT
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("NPI_ALL_STATES_PORT", PORT)))


def _run_resolve(argv):
    from resolve.cli import main
    main(argv)


def main():
    if len(sys.argv) < 2 or sys.argv[1] not in ("api", "resolve"):
        print(__doc__)
        sys.exit(2)
    target, rest = sys.argv[1], sys.argv[2:]
    if target == "api":
        _run_api(rest)
    else:
        _run_resolve(rest)


if __name__ == "__main__":
    main()
