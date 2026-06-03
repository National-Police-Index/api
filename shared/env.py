"""Single source of truth for .env discovery.

Keys live in a few historical locations (resolve/.env for OPENAI_API_KEY, the legacy
server/.env for SUPABASE_KEY). This loads all known candidates so any entry point —
the API, the resolve CLI, tests — finds the keys regardless of where it was launched.
Existing environment variables are never overridden.
"""
import os

from dotenv import load_dotenv

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Order = preference. A repo-root .env (if present) wins; the rest are back-compat.
_CANDIDATES = (
    ".env",
    "resolve/.env",
    "server/.env",
    "archive/legacy_ca/server/.env",
    "archive/legacy_ca/resolve/src/.env",
)


def load_env() -> None:
    """Load every known .env (override=False, so real env vars take precedence)."""
    for rel in _CANDIDATES:
        load_dotenv(os.path.join(_ROOT, rel), override=False)
