"""Config for the ISOLATED all_npi_states API (separate from the postie server)."""
import os
from dotenv import load_dotenv

# Reuse the same Supabase project creds as the postie server (same project, new table).
_ENV = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "server", ".env")
load_dotenv(_ENV)

SUPABASE_URL = "https://jcbircleopydnrtikisa.supabase.co"
SUPABASE_KEY = os.getenv("SUPABASE_KEY", "")

TABLE_NAME = "all_npi_states"

API_TITLE = "NPI All-States Employment API"
API_VERSION = "1.0.0"

# Distinct default port so this can run alongside the postie server (8000).
PORT = int(os.getenv("NPI_ALL_STATES_PORT", "8001"))
