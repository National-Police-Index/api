"""Shared pytest configuration for the NPI repo.

Puts the repo root on sys.path so tests import the new `resolve.*`, `api.*`,
and `shared.*` packages, and loads the project .env files so anything that
needs OPENAI_API_KEY / SUPABASE_KEY (integration tests) finds them.
"""
import os
import sys

import pytest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

# Load env (OPENAI_API_KEY, SUPABASE_KEY) from all known .env locations.
from shared.env import load_env  # noqa: E402

load_env()


def _has_openai_key() -> bool:
    return bool(os.getenv("OPENAI_API_KEY"))


def _has_supabase_key() -> bool:
    return bool(os.getenv("SUPABASE_KEY"))


@pytest.fixture
def repo_root() -> str:
    return REPO_ROOT
