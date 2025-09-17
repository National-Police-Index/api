import os
from dotenv import find_dotenv, load_dotenv

load_dotenv(find_dotenv())

SUPABASE_URL = "https://jcbircleopydnrtikisa.supabase.co"
SUPABASE_KEY = os.getenv("SUPABASE_KEY", "")

# API settings
API_TITLE = "Officer Data API"
API_DESCRIPTION = "API for searching officers and retrieving case information"
API_VERSION = "1.0.0"

# Client settings
DEFAULT_TIMEOUT = 30
DEFAULT_BASE_URL = "http://localhost:8000"
