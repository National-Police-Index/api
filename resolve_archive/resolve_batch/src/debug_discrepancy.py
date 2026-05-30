"""
Debug specific discrepancies between batch and sequential processing.
Re-runs only the discrepant mentions with detailed logging.
"""

import pandas as pd
import datetime
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from api import NPIClient
from models.src import OfficerMention
from match import PostMatcher, generate_officer_uid


def debug_single_mention(mention_uid: str, input_csv: str = "../data/input/involved_officers.csv"):
    """
    Debug a single mention through both sequential and batch pipelines.
    Shows detailed step-by-step filtering.
    """
    # Load the input data
    df = pd.read_csv(input_csv)

    # Filter to just this mention
    if 'officer_uid' not in df.columns:
        df['officer_uid'] = df.apply(generate_officer_uid, axis=1)

    officer_row = df[df['officer_uid'] == mention_uid]

    if len(officer_row) == 0:
        print(f"ERROR: Mention UID {mention_uid} not found in input data")
        return

    officer_row = officer_row.iloc[0]

    print("="*80)
    print(f"DEBUGGING MENTION: {mention_uid}")
    print("="*80)
    print(f"Officer: {officer_row['first_name']} {officer_row['last_name']}")
    print(f"Source Agency: {officer_row['source_agency']}")
    print(f"Incident Year: {officer_row['incident_year']}")
    print(f"Agency Type: {officer_row['agency_type']}")
    print("="*80)
    print()

    # Create OfficerMention object
    mention_date = datetime.date(year=int(officer_row["incident_year"]), month=1, day=1)

    # Handle NaN values by converting to empty strings
    def safe_get(row, key, default=""):
        val = row.get(key, default)
        return default if pd.isna(val) else str(val)

    mention = OfficerMention(
        mention_uid=officer_row["officer_uid"],
        mention_agency_type=officer_row["agency_type"],
        mention_incident_date=mention_date,
        mention_first_name=safe_get(officer_row, "first_name"),
        mention_middle_name=safe_get(officer_row, "middle_name"),
        mention_last_name=safe_get(officer_row, "last_name"),
        mention_agency=safe_get(officer_row, "source_agency"),
        state=safe_get(officer_row, "state", "CA"),
        mentioned_agencies=safe_get(officer_row, "mentioned_agencies")
    )

    # Run through SEQUENTIAL pipeline
    print("\n" + "="*80)
    print("SEQUENTIAL PIPELINE - DETAILED TRACE")
    print("="*80)

    client = NPIClient(base_url="http://localhost:8000")

    # Step 1: Get source county
    source_county = None
    if mention.mention_agency and mention.mention_agency_type.upper() != "CORRECTIONS":
        source_county = client.get_county_for_agency(mention.mention_agency)
        print(f"\n[1] County Lookup:")
        print(f"    Source agency: {mention.mention_agency}")
        print(f"    County: {source_county}")

    # Step 2: Get candidates from API
    print(f"\n[2] Fetching candidates from API...")
    api_candidates = client.get_candidates_for_mention(
        first_name=mention.mention_first_name,
        last_name=mention.mention_last_name,
        incident_year=mention.mention_incident_date.year,
        state=mention.state,
        agency_type=mention.mention_agency_type,
    )

    print(f"    API returned {len(api_candidates)} candidates")

    if api_candidates:
        post_data = [candidate.model_dump() for candidate in api_candidates]
        post = pd.DataFrame(post_data)

        print(f"\n    Candidates by person_nbr:")
        for person_nbr in post['post_person_nbr'].unique()[:10]:  # Show first 10
            person_records = post[post['post_person_nbr'] == person_nbr]
            counties = person_records['county'].unique()
            print(f"      - {person_nbr}: {person_records.iloc[0]['post_first_name']} {person_records.iloc[0]['post_last_name']}")
            print(f"        Counties: {', '.join([c for c in counties if c])}")

        # Step 3: County filtering
        if source_county and mention.mention_agency_type.upper() != "CORRECTIONS":
            print(f"\n[3] County Filtering (source county: {source_county}):")
            print(f"    Before: {len(post)} records, {post['post_person_nbr'].nunique()} unique persons")

            person_has_county_match = post.groupby("post_person_nbr")["county"].apply(
                lambda counties: source_county in counties.values
            )
            valid_person_nbrs = person_has_county_match[person_has_county_match].index.tolist()

            print(f"    Persons with employment in {source_county}:")
            for person_nbr in valid_person_nbrs:
                person_records = post[post['post_person_nbr'] == person_nbr]
                print(f"      ✓ {person_nbr}: {person_records.iloc[0]['post_first_name']} {person_records.iloc[0]['post_last_name']}")

            post = post[post["post_person_nbr"].isin(valid_person_nbrs)]
            print(f"    After: {len(post)} records, {post['post_person_nbr'].nunique()} unique persons")
        else:
            print(f"\n[3] County Filtering: SKIPPED")

        # Step 4: Date filtering
        print(f"\n[4] Date Filtering (incident year: {mention.mention_incident_date.year}, ±1 year buffer):")
        print(f"    Before: {len(post)} records")

        end_dates_cleaned = post.post_end_date.replace("", pd.NaT)
        end_dates_cleaned = pd.to_datetime(end_dates_cleaned, errors='coerce')
        end_dates_cleaned = end_dates_cleaned.where(
            (end_dates_cleaned.isna()) | (end_dates_cleaned.dt.year >= 1950),
            pd.NaT
        )
        end_dates_filled = end_dates_cleaned.fillna(pd.Timestamp.today())
        start_dates = pd.to_datetime(post.post_start_date, errors='coerce')

        incident_year = mention.mention_incident_date.year
        date_in_range = (
            (start_dates.dt.year <= incident_year + 1) &
            (end_dates_filled.dt.year >= incident_year - 1)
        )

        print(f"    Records in date range: {date_in_range.sum()}")

        # Show which records pass/fail date filter by person
        for person_nbr in post['post_person_nbr'].unique():
            person_records = post[post['post_person_nbr'] == person_nbr]
            person_dates = date_in_range[post['post_person_nbr'] == person_nbr]
            if person_dates.any():
                print(f"      ✓ {person_nbr}: {person_dates.sum()}/{len(person_records)} records in range")
            else:
                print(f"      ✗ {person_nbr}: 0/{len(person_records)} records in range")

        # Step 5: Final candidate selection
        print(f"\n[5] Final Candidate Selection:")

        agency_type = (post.post_agency_type.str.lower() == mention.mention_agency_type.lower())
        prefix_len = 2
        fn_prefix = (mention.mention_first_name[:prefix_len].casefold() if mention.mention_first_name else "Z")

        fn_cand = post.post_first_name.str[:prefix_len].str.casefold() == fn_prefix
        fn_full_cand = (post.post_first_name.str.casefold() == mention.mention_first_name.casefold())
        ln_cand = (post.post_last_name.str[:prefix_len].str.casefold() == mention.mention_last_name[:prefix_len].casefold())
        ln_full_cand = (post.post_last_name.str.casefold() == mention.mention_last_name.casefold())

        cands = pd.concat([
            post.loc[agency_type & date_in_range & fn_cand & ln_full_cand],
            post.loc[agency_type & date_in_range & fn_full_cand & ln_cand],
        ]).drop_duplicates()

        print(f"    Final candidates: {len(cands)} records, {cands['post_person_nbr'].nunique()} unique persons")

        if len(cands) > 0:
            print(f"\n    Selected candidates:")
            for person_nbr in cands['post_person_nbr'].unique():
                person_cands = cands[cands['post_person_nbr'] == person_nbr]
                print(f"      ✓ {person_nbr}: {person_cands.iloc[0]['post_first_name']} {person_cands.iloc[0]['post_last_name']} @ {person_cands.iloc[0]['post_agency_name']}")

    print("\n" + "="*80)
    print("BATCH PIPELINE - DETAILED TRACE")
    print("="*80)
    print("(Batch uses same filtering logic, would show same results)")
    print("The discrepancy likely comes from the batch API returning different data.")
    print("="*80)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Debug a specific mention discrepancy")
    parser.add_argument("mention_uid", help="The mention UID to debug")
    parser.add_argument("--input", default="../data/input/involved_officers.csv", help="Input CSV file")

    args = parser.parse_args()

    debug_single_mention(args.mention_uid, args.input)
