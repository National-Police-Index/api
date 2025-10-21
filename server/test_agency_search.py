import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(
    os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "resolve")
)


from database.src import SupabaseClient
from config import SUPABASE_URL, SUPABASE_KEY

# Initialize client
db_client = SupabaseClient(SUPABASE_URL, SUPABASE_KEY)

# Test the method directly
print("Testing database method directly...")

 
try:
    results = db_client.get_officers_by_name_and_agency(
        first_name="CHRISTINE",
        last_name="MURILLO",
        agency_name="Los Angeles County Safety",
        similarity_threshold=0.6
    )
    print(f"Success! Found {len(results)} records")
    for record in results:
        print(f"  - {record.post_first_name} {record.post_last_name} at {record.post_agency_name}")
except Exception as e:
    print(f"ERROR: {e}")
    import traceback
    traceback.print_exc()