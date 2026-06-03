"""
Export the entire postie table to CSV using batched fetching.
Handles large tables (600k+ rows) by fetching in chunks and writing incrementally.
"""

import sys
import os
import csv

# Add parent directories to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from server.config import SUPABASE_URL, SUPABASE_KEY
from supabase import create_client

# Configuration
# Note: Supabase has a hard limit of 1000 rows per request
BATCH_SIZE = 1000  # Supabase's max rows per request
OUTPUT_FILE = "../data/output/postie_export.csv"
TABLE_NAME = "postie"


def export_postie_table():
    """Export entire postie table to CSV in batches"""

    print(f"Connecting to Supabase...")
    client = create_client(SUPABASE_URL, SUPABASE_KEY)

    # Get total count first
    print(f"Counting rows in {TABLE_NAME} table...")
    count_response = client.table(TABLE_NAME).select("*", count="exact").limit(1).execute()
    total_rows = count_response.count
    print(f"Total rows to export: {total_rows:,}")

    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)

    # Initialize CSV file with headers
    first_batch = True
    total_exported = 0
    offset = 0

    print(f"\nStarting export to {OUTPUT_FILE}...")
    print(f"Batch size: {BATCH_SIZE:,} rows\n")

    while offset < total_rows:
        print(f"Fetching rows {offset:,} to {min(offset + BATCH_SIZE, total_rows):,}...", end=" ")

        try:
            # Fetch batch with offset and limit
            response = client.table(TABLE_NAME)\
                .select("*")\
                .range(offset, offset + BATCH_SIZE - 1)\
                .execute()

            batch_data = response.data

            if not batch_data:
                print("No more data.")
                break

            # Write to CSV
            mode = 'w' if first_batch else 'a'
            with open(OUTPUT_FILE, mode, newline='', encoding='utf-8') as f:
                if batch_data:
                    writer = csv.DictWriter(f, fieldnames=batch_data[0].keys())

                    if first_batch:
                        writer.writeheader()
                        first_batch = False

                    writer.writerows(batch_data)

            batch_count = len(batch_data)
            total_exported += batch_count
            print(f"✓ Wrote {batch_count:,} rows (Total: {total_exported:,}/{total_rows:,} - {total_exported/total_rows*100:.1f}%)")

            # Move to next batch - use actual batch_count to handle Supabase's limits
            offset += batch_count

        except Exception as e:
            print(f"\n✗ Error fetching batch at offset {offset}: {e}")
            print(f"Exported {total_exported:,} rows before error.")
            raise

    print(f"\n{'='*60}")
    print(f"Export complete!")
    print(f"Total rows exported: {total_exported:,}")
    print(f"Output file: {OUTPUT_FILE}")
    print(f"{'='*60}")


if __name__ == "__main__":
    try:
        export_postie_table()
    except KeyboardInterrupt:
        print("\n\nExport interrupted by user.")
        sys.exit(1)
    except Exception as e:
        print(f"\n\nExport failed: {e}")
        sys.exit(1)
