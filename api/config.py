"""Config for the all_npi_states API."""
import os

from shared.env import load_env

load_env()

SUPABASE_URL = os.getenv("SUPABASE_URL", "https://jcbircleopydnrtikisa.supabase.co")
SUPABASE_KEY = os.getenv("SUPABASE_KEY", "")

TABLE_NAME = "all_npi_states"

API_TITLE = "NPI All-States Employment API"
API_VERSION = "1.0.0"

# Distinct default port so this can run alongside the postie server (8000).
PORT = int(os.getenv("NPI_ALL_STATES_PORT", "8001"))
