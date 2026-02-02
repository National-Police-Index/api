"""
Comprehensive comparison of batch vs sequential processing - FULL PIPELINE.
Tests the entire end-to-end flow including name uniqueness validation.
"""

import time
import pandas as pd
import sys
import os
import datetime
import logging

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from api import NPIClient
from models.src import OfficerMention

from match import PostMatcher, generate_officer_uid, fetch_all_candidates_with_same_name as fetch_seq

# Setup logging
def setup_logger(name, log_file, level=logging.INFO):
    """Function to setup logger with file handler"""
    handler = logging.FileHandler(log_file, mode='w')
    formatter = logging.Formatter('%(message)s')
    handler.setFormatter(formatter)

    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.addHandler(handler)

    return logger


def load_sample_data(n_samples=100, random_seed=42):
    """Load and prepare sample data from input CSV."""
    df = pd.read_csv("../data/input/involved_officers.csv")

    # Filter to valid records
    df = df.fillna("")
    df = df[~(df.incident_year == "")]
    df = df[~(df.provisional_case_name.fillna("") == "")]

    # Sample with fixed seed for reproducibility
    if len(df) > n_samples:
        df = df.sample(n=n_samples, random_state=random_seed)

    # Generate officer_uid if missing
    if 'officer_uid' not in df.columns:
        df['officer_uid'] = df.apply(generate_officer_uid, axis=1)

    return df


def convert_to_mentions(df):
    """Convert DataFrame rows to OfficerMention objects."""
    mentions = []
    for _, row in df.iterrows():
        try:
            mention_date = datetime.date(year=int(row["incident_year"]), month=1, day=1)
            mention = OfficerMention(
                mention_uid=row["officer_uid"],
                mention_agency_type=row["agency_type"],
                mention_incident_date=mention_date,
                mention_first_name=row["first_name"],
                mention_middle_name=row.get("middle_name", ""),
                mention_last_name=row["last_name"],
                mention_agency=row["source_agency"],
                state=row.get("state", "CA"),
                mentioned_agencies=row.get("mentioned_agencies", "")
            )
            mentions.append(mention)
        except Exception as e:
            print(f"Warning: Failed to create mention for row: {e}")
            continue

    return mentions


def run_sequential_full_pipeline(mentions, input_df):
    """
    Run the FULL sequential pipeline including name uniqueness checks.
    Mimics the flow from match.py's __main__ block.
    """
    print("\n" + "="*80)
    print("SEQUENTIAL PIPELINE - FULL END-TO-END")
    print("="*80)

    # Setup logger for sequential processing
    seq_logger = setup_logger('sequential', '../data/output/sequential_candidates.log')

    client = NPIClient(base_url="http://localhost:8000")
    matcher = PostMatcher(api_base_url="http://localhost:8000")

    # Track timing for different stages
    timings = {
        "matching": 0,
        "name_uniqueness": 0,
        "total": 0
    }

    total_start = time.time()

    # Stage 1: Core matching pipeline
    print("\nStage 1: Running core matching...")
    match_start = time.time()
    results, all_candidates, invalid_candidates, full_employment_histories = \
        matcher.find_canonical_stint(mentions)
    timings["matching"] = time.time() - match_start
    print(f"  Matching completed in {timings['matching']:.2f}s")
    print(f"  Matches found: {len(results)}")

    # Log candidate details for debugging
    seq_logger.info("="*80)
    seq_logger.info("SEQUENTIAL PIPELINE - CANDIDATE DETAILS")
    seq_logger.info("="*80)
    seq_logger.info(f"Total mentions processed: {len(mentions)}")
    seq_logger.info(f"Total candidates found: {len(all_candidates)}")
    seq_logger.info(f"Total matches: {len(results)}")
    seq_logger.info("")

    # Create a map of mention details
    mention_details = {}
    for mention in mentions:
        mention_details[mention.mention_uid] = {
            'name': f"{mention.mention_first_name} {mention.mention_last_name}",
            'agency': mention.mention_agency,
            'year': mention.mention_incident_date.year,
            'agency_type': mention.mention_agency_type.value if hasattr(mention.mention_agency_type, 'value') else str(mention.mention_agency_type)
        }

    # Log candidates per mention with detailed filtering info
    # Note: County info already fetched inside find_canonical_stint()
    if len(all_candidates) > 0:
        for mention_uid in all_candidates['mention_uid'].unique():
            mention_candidates = all_candidates[all_candidates['mention_uid'] == mention_uid]
            mention_info = mention_details.get(mention_uid, {})

            seq_logger.info(f"Mention: {mention_uid}")
            seq_logger.info(f"  Officer: {mention_info.get('name', 'Unknown')}")
            seq_logger.info(f"  Source Agency: {mention_info.get('agency', 'Unknown')}")
            seq_logger.info(f"  Incident Year: {mention_info.get('year', 'Unknown')}")
            seq_logger.info(f"  Agency Type: {mention_info.get('agency_type', 'Unknown')}")
            seq_logger.info(f"  Final Candidates: {len(mention_candidates)}")

            for idx, cand in mention_candidates.iterrows():
                seq_logger.info(f"    - {cand['post_first_name']} {cand['post_last_name']} (ID: {cand['post_person_nbr']}, Agency: {cand['post_agency_name']}, County: {cand.get('county', 'N/A')})")
            seq_logger.info("")

    # Stage 2: Name uniqueness validation (SEQUENTIAL - N API calls)
    print("\nStage 2: Name uniqueness validation (SEQUENTIAL)...")
    uniqueness_start = time.time()

    api_call_count = 0
    officers_with_conflicts = []
    officers_without_conflicts = []

    if len(results) > 0:
        for idx, (_, match) in enumerate(results.iterrows()):
            first_name = match["post_first_name"]
            last_name = match["post_last_name"]
            post_person_nbr = match["post_person_nbr"]

            # SEQUENTIAL: One API call per matched officer
            all_name_candidates = fetch_seq(
                first_name=first_name,
                last_name=last_name,
                client=client
            )
            api_call_count += 1

            # Check for conflicts
            if len(all_name_candidates) > 0:
                all_name_candidates = all_name_candidates[
                    all_name_candidates['post_person_nbr'] != post_person_nbr
                ]
                if len(all_name_candidates) > 0:
                    officers_with_conflicts.append(match["mention_uid"])
                else:
                    officers_without_conflicts.append(match["mention_uid"])
            else:
                officers_without_conflicts.append(match["mention_uid"])

    timings["name_uniqueness"] = time.time() - uniqueness_start
    print(f"  Name uniqueness completed in {timings['name_uniqueness']:.2f}s")
    print(f"  API calls made: {api_call_count}")
    print(f"  Officers with conflicts: {len(officers_with_conflicts)}")
    print(f"  Officers without conflicts: {len(officers_without_conflicts)}")

    timings["total"] = time.time() - total_start

    print(f"\n{'='*80}")
    print(f"SEQUENTIAL TOTAL TIME: {timings['total']:.2f}s")
    print(f"  - Matching: {timings['matching']:.2f}s ({timings['matching']/timings['total']*100:.1f}%)")
    print(f"  - Name uniqueness: {timings['name_uniqueness']:.2f}s ({timings['name_uniqueness']/timings['total']*100:.1f}%)")
    print(f"{'='*80}\n")

    return {
        "results": results,
        "all_candidates": all_candidates,
        "invalid_candidates": invalid_candidates,
        "full_histories": full_employment_histories,
        "officers_with_conflicts": officers_with_conflicts,
        "officers_without_conflicts": officers_without_conflicts,
        "timings": timings,
        "api_calls": api_call_count
    }


def run_batch_full_pipeline(mentions, input_df):
    """
    Run the FULL batch pipeline including name uniqueness checks.
    Mimics the flow from match_v2.py's __main__ block with batch optimization.
    """
    print("\n" + "="*80)
    print("BATCH PIPELINE - FULL END-TO-END")
    print("="*80)

    # Setup logger for batch processing
    batch_logger = setup_logger('batch', '../data/output/batch_candidates.log')

    client = NPIClient(base_url="http://localhost:8000")
    matcher = PostMatcher(api_base_url="http://localhost:8000")

    # Track timing for different stages
    timings = {
        "matching": 0,
        "name_uniqueness": 0,
        "total": 0
    }

    total_start = time.time()

    # Stage 1: Core matching pipeline (with batch candidate/county fetching)
    print("\nStage 1: Running core matching (BATCH)...")
    match_start = time.time()
    results, all_candidates, invalid_candidates, full_employment_histories = \
        matcher.find_canonical_stint_batch(mentions)
    timings["matching"] = time.time() - match_start
    print(f"  Matching completed in {timings['matching']:.2f}s")
    print(f"  Matches found: {len(results)}")

    # Log candidate details for debugging
    batch_logger.info("="*80)
    batch_logger.info("BATCH PIPELINE - CANDIDATE DETAILS")
    batch_logger.info("="*80)
    batch_logger.info(f"Total mentions processed: {len(mentions)}")
    batch_logger.info(f"Total candidates found: {len(all_candidates)}")
    batch_logger.info(f"Total matches: {len(results)}")
    batch_logger.info("")

    # Create a map of mention details
    mention_details = {}
    for mention in mentions:
        mention_details[mention.mention_uid] = {
            'name': f"{mention.mention_first_name} {mention.mention_last_name}",
            'agency': mention.mention_agency,
            'year': mention.mention_incident_date.year,
            'agency_type': mention.mention_agency_type.value if hasattr(mention.mention_agency_type, 'value') else str(mention.mention_agency_type)
        }

    # Log candidates per mention with detailed filtering info
    # Note: County info already fetched inside find_canonical_stint_batch()
    if len(all_candidates) > 0:
        for mention_uid in all_candidates['mention_uid'].unique():
            mention_candidates = all_candidates[all_candidates['mention_uid'] == mention_uid]
            mention_info = mention_details.get(mention_uid, {})

            batch_logger.info(f"Mention: {mention_uid}")
            batch_logger.info(f"  Officer: {mention_info.get('name', 'Unknown')}")
            batch_logger.info(f"  Source Agency: {mention_info.get('agency', 'Unknown')}")
            batch_logger.info(f"  Incident Year: {mention_info.get('year', 'Unknown')}")
            batch_logger.info(f"  Agency Type: {mention_info.get('agency_type', 'Unknown')}")
            batch_logger.info(f"  Final Candidates: {len(mention_candidates)}")

            for idx, cand in mention_candidates.iterrows():
                batch_logger.info(f"    - {cand['post_first_name']} {cand['post_last_name']} (ID: {cand['post_person_nbr']}, Agency: {cand['post_agency_name']}, County: {cand.get('county', 'N/A')})")
            batch_logger.info("")

    # Stage 2: Name uniqueness validation (BATCH - 1 API call for all)
    print("\nStage 2: Name uniqueness validation (BATCH)...")
    uniqueness_start = time.time()

    api_call_count = 0
    officers_with_conflicts = []
    officers_without_conflicts = []

    if len(results) > 0:
        # BATCH: Collect all unique names first
        unique_names = list(set(
            (match["post_first_name"], match["post_last_name"])
            for _, match in results.iterrows()
        ))
        print(f"  Checking {len(unique_names)} unique names...")

        # BATCH: Single API call for all names
        name_uniqueness_counts = client.get_batch_name_uniqueness(unique_names)
        api_call_count = 1  # Only 1 batch API call!

        print(f"  Retrieved uniqueness counts in 1 batch API call")
        print(f"  DEBUG: API returned {len(name_uniqueness_counts)} entries")
        print(f"  DEBUG: Sample of returned data: {dict(list(name_uniqueness_counts.items())[:3])}")

        # Now determine conflicts using batch-fetched uniqueness data
        for idx, (_, match) in enumerate(results.iterrows()):
            first_name = match["post_first_name"]
            last_name = match["post_last_name"]
            name_key = f"{first_name}|{last_name}"  # Use pipe-separated string to match API response format

            # Get uniqueness count - if >= 2 unique persons with same name, there's a conflict
            unique_person_count = name_uniqueness_counts.get(name_key, 1)
            has_conflicts = unique_person_count >= 2

            # Debug first few lookups
            if idx < 3:
                print(f"  DEBUG: Looking up '{name_key}', got count={unique_person_count}, has_conflicts={has_conflicts}")

            if has_conflicts:
                officers_with_conflicts.append(match["mention_uid"])
            else:
                officers_without_conflicts.append(match["mention_uid"])

    timings["name_uniqueness"] = time.time() - uniqueness_start
    print(f"  Name uniqueness completed in {timings['name_uniqueness']:.2f}s")
    print(f"  API calls made: {api_call_count}")
    print(f"  Officers with conflicts: {len(officers_with_conflicts)}")
    print(f"  Officers without conflicts: {len(officers_without_conflicts)}")

    timings["total"] = time.time() - total_start

    print(f"\n{'='*80}")
    print(f"BATCH TOTAL TIME: {timings['total']:.2f}s")
    print(f"  - Matching: {timings['matching']:.2f}s ({timings['matching']/timings['total']*100:.1f}%)")
    print(f"  - Name uniqueness: {timings['name_uniqueness']:.2f}s ({timings['name_uniqueness']/timings['total']*100:.1f}%)")
    print(f"{'='*80}\n")

    return {
        "results": results,
        "all_candidates": all_candidates,
        "invalid_candidates": invalid_candidates,
        "full_histories": full_employment_histories,
        "officers_with_conflicts": officers_with_conflicts,
        "officers_without_conflicts": officers_without_conflicts,
        "timings": timings,
        "api_calls": api_call_count
    }


def log_county_discrepancies(mentions):
    """
    Compare county lookups between sequential (one-by-one) and batch (parallel) approaches.
    Both should find the same counties for each agency.
    """
    county_logger = setup_logger('county_comparison', '../data/output/county_comparison.log')

    county_logger.info("="*80)
    county_logger.info("COUNTY LOOKUP COMPARISON: SEQUENTIAL VS BATCH")
    county_logger.info("="*80)
    county_logger.info("")

    from api import NPIClient
    from concurrent.futures import ThreadPoolExecutor, as_completed
    import time

    client = NPIClient(base_url="http://localhost:8000")
    unique_agencies = list(set(m.mention_agency for m in mentions if m.mention_agency and m.mention_agency_type.upper() != "CORRECTIONS"))

    county_logger.info(f"Testing {len(unique_agencies)} unique agencies...")
    county_logger.info("")

    # SEQUENTIAL APPROACH (how find_canonical_stint does it)
    county_logger.info("SEQUENTIAL APPROACH (one-by-one):")
    seq_start = time.time()
    seq_counties = {}
    for agency in unique_agencies:
        county = client.get_county_for_agency(agency)
        seq_counties[agency] = county
    seq_time = time.time() - seq_start
    county_logger.info(f"  Time: {seq_time:.2f}s")
    county_logger.info(f"  Found: {sum(1 for c in seq_counties.values() if c is not None)}/{len(unique_agencies)}")
    county_logger.info("")

    # BATCH APPROACH (how find_canonical_stint_batch does it)
    county_logger.info("BATCH APPROACH (parallel with ThreadPoolExecutor):")
    batch_start = time.time()

    def fetch_county(agency):
        return agency, client.get_county_for_agency(agency)

    batch_counties = {}
    with ThreadPoolExecutor(max_workers=10) as executor:
        futures = {executor.submit(fetch_county, agency): agency for agency in unique_agencies}
        for future in as_completed(futures):
            try:
                agency, county = future.result()
                batch_counties[agency] = county
            except Exception as e:
                county_logger.info(f"  ERROR: {e}")

    batch_time = time.time() - batch_start
    county_logger.info(f"  Time: {batch_time:.2f}s")
    county_logger.info(f"  Found: {sum(1 for c in batch_counties.values() if c is not None)}/{len(unique_agencies)}")
    county_logger.info(f"  Speedup: {seq_time/batch_time:.2f}x")
    county_logger.info("")

    # COMPARE RESULTS
    discrepancies = []
    for agency in sorted(unique_agencies):
        seq_county = seq_counties.get(agency)
        batch_county = batch_counties.get(agency)
        if seq_county != batch_county:
            discrepancies.append((agency, seq_county, batch_county))

    if discrepancies:
        county_logger.info(f"⚠️  DISCREPANCIES FOUND: {len(discrepancies)}")
        county_logger.info("")
        for agency, seq, batch in discrepancies:
            county_logger.info(f"  Agency: {agency}")
            county_logger.info(f"    Sequential: {seq or 'NOT FOUND'}")
            county_logger.info(f"    Batch: {batch or 'NOT FOUND'}")
            county_logger.info("")
    else:
        county_logger.info("✓ NO DISCREPANCIES - Both approaches found identical counties!")
        county_logger.info("")

    # List agencies without counties (regardless of approach)
    not_found = [agency for agency in sorted(unique_agencies) if seq_counties.get(agency) is None]
    if not_found:
        county_logger.info(f"AGENCIES WITHOUT COUNTIES ({len(not_found)}):")
        for agency in not_found:
            county_logger.info(f"  ✗ {agency}")
        county_logger.info("")

    county_logger.info("="*80)

    return len(discrepancies), len(not_found)


def log_candidate_discrepancies(seq_data, batch_data):
    """
    Log detailed comparison of candidates between sequential and batch pipelines.
    Only logs mentions where there are differences.
    """
    discrepancy_logger = setup_logger('discrepancies', '../data/output/candidate_discrepancies.log')

    seq_all_cands = seq_data["all_candidates"]
    batch_all_cands = batch_data["all_candidates"]

    discrepancy_logger.info("="*80)
    discrepancy_logger.info("CANDIDATE DISCREPANCY ANALYSIS")
    discrepancy_logger.info("="*80)
    discrepancy_logger.info("")

    # Get all unique mention UIDs from both pipelines
    all_mention_uids = set()
    if len(seq_all_cands) > 0:
        all_mention_uids.update(seq_all_cands['mention_uid'].unique())
    if len(batch_all_cands) > 0:
        all_mention_uids.update(batch_all_cands['mention_uid'].unique())

    discrepancy_count = 0

    for mention_uid in sorted(all_mention_uids):
        # Get candidates from each pipeline
        seq_cands = seq_all_cands[seq_all_cands['mention_uid'] == mention_uid] if len(seq_all_cands) > 0 else pd.DataFrame()
        batch_cands = batch_all_cands[batch_all_cands['mention_uid'] == mention_uid] if len(batch_all_cands) > 0 else pd.DataFrame()

        seq_count = len(seq_cands)
        batch_count = len(batch_cands)

        # Only log if there's a discrepancy
        if seq_count != batch_count:
            discrepancy_count += 1
            discrepancy_logger.info(f"DISCREPANCY #{discrepancy_count}")
            discrepancy_logger.info(f"Mention UID: {mention_uid}")
            discrepancy_logger.info(f"Sequential candidates: {seq_count}, Batch candidates: {batch_count}")
            discrepancy_logger.info("")

            # Get candidate person_nbrs from each
            seq_person_nbrs = set(seq_cands['post_person_nbr'].tolist()) if seq_count > 0 else set()
            batch_person_nbrs = set(batch_cands['post_person_nbr'].tolist()) if batch_count > 0 else set()

            only_in_seq = seq_person_nbrs - batch_person_nbrs
            only_in_batch = batch_person_nbrs - seq_person_nbrs
            in_both = seq_person_nbrs & batch_person_nbrs

            if in_both:
                discrepancy_logger.info(f"  Candidates in BOTH ({len(in_both)}):")
                for person_nbr in sorted(in_both):
                    cand = seq_cands[seq_cands['post_person_nbr'] == person_nbr].iloc[0]
                    discrepancy_logger.info(f"    ✓ {cand['post_first_name']} {cand['post_last_name']} (ID: {person_nbr}, Agency: {cand['post_agency_name']})")
                discrepancy_logger.info("")

            if only_in_seq:
                discrepancy_logger.info(f"  Candidates ONLY in SEQUENTIAL ({len(only_in_seq)}):")
                for person_nbr in sorted(only_in_seq):
                    cand = seq_cands[seq_cands['post_person_nbr'] == person_nbr].iloc[0]
                    discrepancy_logger.info(f"    ⚠ {cand['post_first_name']} {cand['post_last_name']} (ID: {person_nbr}, Agency: {cand['post_agency_name']})")
                discrepancy_logger.info("")

            if only_in_batch:
                discrepancy_logger.info(f"  Candidates ONLY in BATCH ({len(only_in_batch)}):")
                for person_nbr in sorted(only_in_batch):
                    cand = batch_cands[batch_cands['post_person_nbr'] == person_nbr].iloc[0]
                    discrepancy_logger.info(f"    ⚠ {cand['post_first_name']} {cand['post_last_name']} (ID: {person_nbr}, Agency: {cand['post_agency_name']})")
                discrepancy_logger.info("")

            discrepancy_logger.info("-" * 80)
            discrepancy_logger.info("")

    discrepancy_logger.info("="*80)
    discrepancy_logger.info(f"SUMMARY: Found {discrepancy_count} mentions with candidate discrepancies")
    discrepancy_logger.info("="*80)

    return discrepancy_count


def compare_full_pipeline_results(seq_data, batch_data):
    """
    Compare sequential vs batch FULL pipeline results.

    Returns:
        dict with comparison metrics and discrepancies
    """
    # Log detailed candidate discrepancies
    discrepancy_count = log_candidate_discrepancies(seq_data, batch_data)

    comparison = {
        "identical": True,
        "discrepancies": [],
        "stats": {},
        "timings": {},
        "candidate_discrepancy_count": discrepancy_count
    }

    # Extract results
    seq_results = seq_data["results"]
    batch_results = batch_data["results"]
    seq_all_cands = seq_data["all_candidates"]
    batch_all_cands = batch_data["all_candidates"]

    # Compare match counts
    seq_uids = set(seq_results['mention_uid'].tolist()) if len(seq_results) > 0 else set()
    batch_uids = set(batch_results['mention_uid'].tolist()) if len(batch_results) > 0 else set()

    comparison["stats"]["sequential_matches"] = len(seq_uids)
    comparison["stats"]["batch_matches"] = len(batch_uids)
    comparison["stats"]["sequential_candidates"] = len(seq_all_cands)
    comparison["stats"]["batch_candidates"] = len(batch_all_cands)

    # Compare conflict detection
    seq_conflicts = set(seq_data["officers_with_conflicts"])
    batch_conflicts = set(batch_data["officers_with_conflicts"])

    comparison["stats"]["sequential_conflicts"] = len(seq_conflicts)
    comparison["stats"]["batch_conflicts"] = len(batch_conflicts)

    # Compare timings
    comparison["timings"]["sequential"] = seq_data["timings"]
    comparison["timings"]["batch"] = batch_data["timings"]

    # Compare API calls for name uniqueness
    comparison["stats"]["sequential_uniqueness_api_calls"] = seq_data["api_calls"]
    comparison["stats"]["batch_uniqueness_api_calls"] = batch_data["api_calls"]

    # Check if same mentions matched
    only_in_seq = seq_uids - batch_uids
    only_in_batch = batch_uids - seq_uids
    matched_in_both = seq_uids & batch_uids

    if only_in_seq or only_in_batch:
        comparison["identical"] = False
        comparison["discrepancies"].append({
            "type": "different_matches",
            "only_in_sequential": list(only_in_seq),
            "only_in_batch": list(only_in_batch),
            "matched_in_both_count": len(matched_in_both)
        })

    # Compare conflict detection results
    only_conflicts_in_seq = seq_conflicts - batch_conflicts
    only_conflicts_in_batch = batch_conflicts - seq_conflicts

    if only_conflicts_in_seq or only_conflicts_in_batch:
        comparison["identical"] = False
        comparison["discrepancies"].append({
            "type": "different_conflict_detection",
            "only_in_sequential": list(only_conflicts_in_seq),
            "only_in_batch": list(only_conflicts_in_batch)
        })

    # Compare matched persons for officers matched in both
    for uid in matched_in_both:
        seq_match = seq_results[seq_results['mention_uid'] == uid].iloc[0]
        batch_match = batch_results[batch_results['mention_uid'] == uid].iloc[0]

        if seq_match['post_person_nbr'] != batch_match['post_person_nbr']:
            comparison["identical"] = False
            comparison["discrepancies"].append({
                "type": "different_person_matched",
                "mention_uid": uid,
                "sequential_person": seq_match['post_person_nbr'],
                "batch_person": batch_match['post_person_nbr']
            })

    return comparison


def print_full_comparison_report(comparison):
    """Print detailed comparison report for full pipeline."""
    print("\n" + "="*80)
    print("FULL PIPELINE COMPARISON REPORT")
    print("="*80)

    # Performance comparison
    seq_timings = comparison["timings"]["sequential"]
    batch_timings = comparison["timings"]["batch"]

    print(f"\nPERFORMANCE BREAKDOWN:")
    print(f"\n  SEQUENTIAL:")
    print(f"    Total time: {seq_timings['total']:.2f}s")
    print(f"    - Matching: {seq_timings['matching']:.2f}s")
    print(f"    - Name uniqueness: {seq_timings['name_uniqueness']:.2f}s")

    print(f"\n  BATCH:")
    print(f"    Total time: {batch_timings['total']:.2f}s")
    print(f"    - Matching: {batch_timings['matching']:.2f}s")
    print(f"    - Name uniqueness: {batch_timings['name_uniqueness']:.2f}s")

    print(f"\n  SPEEDUP:")
    total_speedup = seq_timings['total'] / batch_timings['total'] if batch_timings['total'] > 0 else 0
    matching_speedup = seq_timings['matching'] / batch_timings['matching'] if batch_timings['matching'] > 0 else 0
    uniqueness_speedup = seq_timings['name_uniqueness'] / batch_timings['name_uniqueness'] if batch_timings['name_uniqueness'] > 0 else 0

    print(f"    Total: {total_speedup:.2f}x")
    print(f"    - Matching: {matching_speedup:.2f}x")
    print(f"    - Name uniqueness: {uniqueness_speedup:.2f}x")

    # API call comparison
    stats = comparison["stats"]
    print(f"\n  API CALL EFFICIENCY (Name Uniqueness):")
    print(f"    Sequential: {stats['sequential_uniqueness_api_calls']} calls")
    print(f"    Batch: {stats['batch_uniqueness_api_calls']} call(s)")
    print(f"    Reduction: {stats['sequential_uniqueness_api_calls'] - stats['batch_uniqueness_api_calls']} calls saved")

    # Results comparison
    print(f"\nRESULTS COMPARISON:")
    print(f"  Matches:")
    print(f"    Sequential: {stats['sequential_matches']}")
    print(f"    Batch: {stats['batch_matches']}")
    print(f"  Candidates:")
    print(f"    Sequential: {stats['sequential_candidates']}")
    print(f"    Batch: {stats['batch_candidates']}")
    print(f"  Conflicts detected:")
    print(f"    Sequential: {stats['sequential_conflicts']}")
    print(f"    Batch: {stats['batch_conflicts']}")

    # Correctness check
    print(f"\nCORRECTNESS:")
    if comparison["identical"]:
        print("  ✓ PASS: Batch and sequential produce identical results")
    else:
        print("  ✗ FAIL: Discrepancies found between batch and sequential")
        print(f"  Number of discrepancies: {len(comparison['discrepancies'])}")

        print("\n  DISCREPANCY DETAILS:")
        for i, disc in enumerate(comparison["discrepancies"][:10], 1):
            print(f"\n  {i}. {disc['type']}:")
            for key, value in disc.items():
                if key != 'type':
                    if isinstance(value, list) and len(value) > 5:
                        print(f"     {key}: {len(value)} items (showing first 5): {value[:5]}")
                    else:
                        print(f"     {key}: {value}")

        if len(comparison['discrepancies']) > 10:
            print(f"\n  ... and {len(comparison['discrepancies']) - 10} more discrepancies")

    print("\n" + "="*80)


def main(sample_size=100):
    print("="*80)
    print("FULL PIPELINE COMPARISON: BATCH VS SEQUENTIAL")
    print("Includes name uniqueness validation optimization")
    print("="*80)

    # Check API
    import requests
    try:
        response = requests.get("http://localhost:8000/", timeout=5)
        print("✓ API server is running\n")
    except:
        print("✗ API server not running at localhost:8000")
        print("Please start the API server first: cd ../../../server && python3 src.py")
        sys.exit(1)

    # Load data with fixed seed
    print(f"Loading sample data (n={sample_size}, seed=42)...")
    df = load_sample_data(n_samples=sample_size, random_seed=42)
    print(f"Loaded {len(df)} officers\n")

    # Convert to mentions
    print("Converting to OfficerMention objects...")
    mentions = convert_to_mentions(df)
    print(f"Successfully created {len(mentions)} mentions\n")

    if len(mentions) == 0:
        print("No valid mentions to process!")
        sys.exit(1)

    # Log county lookup comparison BEFORE running pipelines
    print("Comparing county lookup approaches (sequential vs batch)...")
    discrepancies, not_found = log_county_discrepancies(mentions)
    if discrepancies == 0:
        print(f"  ✓ Both approaches found identical counties!")
    else:
        print(f"  ⚠️  Found {discrepancies} discrepancies!")
    print(f"  ✓ County comparison saved to: ../data/output/county_comparison.log\n")

    # Run SEQUENTIAL full pipeline
    seq_data = run_sequential_full_pipeline(mentions, df)

    # Run BATCH full pipeline
    batch_data = run_batch_full_pipeline(mentions, df)

    # Compare results
    print("\n" + "="*80)
    print("COMPARING FULL PIPELINE RESULTS...")
    print("="*80)
    comparison = compare_full_pipeline_results(seq_data, batch_data)
    print_full_comparison_report(comparison)

    # Save detailed report
    report_path = "../data/output/batch_vs_sequential_comparison.txt"
    os.makedirs(os.path.dirname(report_path), exist_ok=True)

    with open(report_path, 'w') as f:
        f.write("FULL PIPELINE COMPARISON: BATCH VS SEQUENTIAL\n")
        f.write("="*80 + "\n\n")
        f.write(f"Sample Size: {len(mentions)} officers\n")
        f.write(f"Random Seed: 42\n\n")

        seq_timings = comparison["timings"]["sequential"]
        batch_timings = comparison["timings"]["batch"]
        stats = comparison["stats"]

        f.write("PERFORMANCE:\n")
        f.write(f"  Sequential total: {seq_timings['total']:.2f}s\n")
        f.write(f"    - Matching: {seq_timings['matching']:.2f}s\n")
        f.write(f"    - Name uniqueness: {seq_timings['name_uniqueness']:.2f}s\n")
        f.write(f"  Batch total: {batch_timings['total']:.2f}s\n")
        f.write(f"    - Matching: {batch_timings['matching']:.2f}s\n")
        f.write(f"    - Name uniqueness: {batch_timings['name_uniqueness']:.2f}s\n")
        f.write(f"  Total speedup: {seq_timings['total']/batch_timings['total']:.2f}x\n\n")

        f.write("API CALLS (Name Uniqueness):\n")
        f.write(f"  Sequential: {stats['sequential_uniqueness_api_calls']} calls\n")
        f.write(f"  Batch: {stats['batch_uniqueness_api_calls']} call(s)\n\n")

        f.write("RESULTS:\n")
        f.write(f"  Sequential matches: {stats['sequential_matches']}\n")
        f.write(f"  Batch matches: {stats['batch_matches']}\n")
        f.write(f"  Sequential conflicts: {stats['sequential_conflicts']}\n")
        f.write(f"  Batch conflicts: {stats['batch_conflicts']}\n\n")

        f.write(f"CORRECTNESS: {'PASS' if comparison['identical'] else 'FAIL'}\n")
        if not comparison['identical']:
            f.write(f"  Discrepancies: {len(comparison['discrepancies'])}\n")
            for disc in comparison['discrepancies']:
                f.write(f"\n  {disc}\n")

    print(f"\n✓ Comparison report saved to: {report_path}")

    # Summary
    total_speedup = seq_timings['total'] / batch_timings['total']
    print(f"\n{'='*80}")
    print(f"SUMMARY")
    print(f"{'='*80}")
    print(f"Sample size: {sample_size} officers")
    print(f"Total speedup: {total_speedup:.2f}x")
    print(f"Sequential time: {seq_timings['total']:.2f}s")
    print(f"Batch time: {batch_timings['total']:.2f}s")
    print(f"Time saved: {seq_timings['total'] - batch_timings['total']:.2f}s")
    print(f"{'='*80}")

    # Exit with appropriate code
    if not comparison["identical"]:
        print("\n⚠️  WARNING: Results are NOT identical!")
        return 1
    else:
        print("\n✓ SUCCESS: Batch and sequential produce identical results!")
        return 0


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Compare full batch vs sequential pipeline")
    parser.add_argument("--sample-size", type=int, default=100, help="Number of officers to test")
    args = parser.parse_args()

    exit_code = main(sample_size=args.sample_size)
    sys.exit(exit_code)
