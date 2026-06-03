from typing import List
import pandas as pd
import pickle
import datetime
import sys
import os
import hashlib
import json

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from api import NPIClient
from models.src import OfficerMention
from concurrent.futures import ThreadPoolExecutor, as_completed
from features import featurize
from helpers import validate_agency_match, ensure_incident_year_column


#TODO output is missing the full employment history. we can fetch it with post id

CHUNK_SIZE = 100  # Process this many officers per checkpoint
# Agency validation calls the (Azure) LLM. 100 concurrent calls saturate the
# connection pool / rate limit and surface as RetryError[APIConnectionError],
# which wrongly drops valid matches. Keep this modest.
VALIDATION_MAX_WORKERS = 8
CHECKPOINT_DIR = "../data/output/checkpoints"
PROGRESS_FILE = os.path.join(CHECKPOINT_DIR, ".progress.json")


def generate_officer_uid(row: pd.Series) -> str:
    """Generate a unique officer_uid using SHA256 hash based on officer details."""
    # Extract values and convert to strings, handling NaN/None values
    first_name = str(row.get('first_name', '')).strip()
    last_name = str(row.get('last_name', '')).strip()
    provisional_case_name = str(row.get('provisional_case_name', '')).strip()
    incident_year = str(row.get('incident_year', '')).strip()
    incident_month = str(row.get('incident_month', '')).strip()
    incident_date = str(row.get('incident_date', '')).strip()
    source_agency = str(row.get('source_agency', '')).strip()

    # Concatenate all fields with a separator
    combined_string = f"{first_name}|{last_name}|{provisional_case_name}|{incident_year}|{incident_month}|{incident_date}|{source_agency}"

    # Generate SHA256 hash
    hash_object = hashlib.sha256(combined_string.encode('utf-8'))
    return hash_object.hexdigest()


def check_name_for_early_filtering(mention: OfficerMention, common_last_names: set) -> tuple[bool, str]:
    """
    Early filtering check: determine if mention should skip expensive pipeline.

    Returns:
        (should_skip_pipeline, review_reason)
        - should_skip_pipeline: True if officer should go to manual review
        - review_reason: Explanation of why officer needs review
    """
    client = NPIClient()  # base_url from NPI_API_URL env (default localhost:8000)

    mention_last_name = mention.mention_last_name.strip().upper()
    mention_first_name = mention.mention_first_name.strip().upper()

    # Check 1: Common last name
    if mention_last_name in common_last_names:
        return True, f"Common last name ({mention_last_name}) - requires manual verification"

    # Check 2: Multiple unique persons with same name in entire database
    try:
        all_records = client.get_officers_by_name(
            first_name=mention.mention_first_name,
            last_name=mention.mention_last_name
        )

        if all_records:
            # Count unique person_nbrs
            unique_persons = set(record.post_person_nbr for record in all_records)

            if len(unique_persons) >= 2:
                return True, f"Multiple persons ({len(unique_persons)}) with same name in database - needs verification"

    except Exception as e:
        print(f"DEBUG: Error checking name uniqueness for {mention_first_name} {mention_last_name}: {e}")
        # On error, don't skip - proceed with normal pipeline
        return False, ""

    # Pass all checks - proceed to expensive pipeline
    return False, ""


def generate_candidates(mention: OfficerMention) -> tuple[pd.DataFrame, pd.DataFrame]:
    """for a given mention, return a list of possibly matching stints from the post employment history
    Returns: (filtered_candidates, full_employment_history)"""

    incident_year = mention.mention_incident_date.year
    prefix_len = 2

    print(f"\nDEBUG: Processing mention: {mention}")
    print(f"DEBUG: Incident year: {incident_year}")
    print(f"DEBUG: Agency type: {mention.mention_agency_type}")

    client = NPIClient()  # base_url from NPI_API_URL env (default localhost:8000)

    source_county = None
    # Only look up county for non-CORRECTIONS agencies
    if mention.mention_agency and mention.mention_agency_type.upper() != "CORRECTIONS":
        source_county = client.get_county_for_agency(mention.mention_agency)
        print(
            f"DEBUG: Source agency '{mention.mention_agency}' is in county: {source_county}"
        )
    elif mention.mention_agency_type.upper() == "CORRECTIONS":
        print(f"DEBUG: CORRECTIONS agency type - skipping county lookup")

    print(f"first_name: {mention.mention_first_name}")
    print(f"last_name: {mention.mention_last_name}")
    print(f"year: {mention.mention_incident_date.year}")

    print(
        f"DEBUG: Fetching targeted candidates for: {mention.mention_first_name} {mention.mention_last_name}"
    )

    # Pass agency_type to the API call
    api_candidates = client.get_candidates_for_mention(
        first_name=mention.mention_first_name,
        last_name=mention.mention_last_name,
        incident_year=mention.mention_incident_date.year,
        state=mention.state,
        agency_type=mention.mention_agency_type,
    )

    if not api_candidates:
        print("DEBUG: No candidates found from API")
        return pd.DataFrame(), pd.DataFrame()

    post_data = [candidate.dict() for candidate in api_candidates]
    post = pd.DataFrame(post_data)

    # PRESERVE FULL EMPLOYMENT HISTORY BEFORE FILTERING
    full_employment_history = post.copy()

    print(f"DEBUG: Retrieved {len(post)} candidates from API")
    print(post)

    # Only apply county filtering for non-CORRECTIONS agencies
    if source_county and mention.mention_agency_type.upper() != "CORRECTIONS":
        print(f"DEBUG: Filtering candidates by county: {source_county}")
        # Group by post_person_nbr and check if they have ANY record in the source county
        person_has_county_match = post.groupby("post_person_nbr")["county"].apply(
            lambda counties: source_county in counties.values
        )
        valid_person_nbrs = person_has_county_match[
            person_has_county_match
        ].index.tolist()

        print(
            f"DEBUG: Found {len(valid_person_nbrs)} unique officers with employment in {source_county} county"
        )
        post = post[post["post_person_nbr"].isin(valid_person_nbrs)]
        print(f"DEBUG: After county filtering: {len(post)} candidate records remain")

        if len(post) == 0:
            print(
                f"DEBUG: No candidates found with employment history in {source_county} county"
            )
            return pd.DataFrame(), full_employment_history
    elif mention.mention_agency_type.upper() == "CORRECTIONS":
        print(
            f"DEBUG: CORRECTIONS agency - skipping county filtering, keeping all {len(post)} candidates"
        )
    else:
        print(f"DEBUG: No source county found - keeping candidates from all counties ({len(post)} candidates)")

    agency_type = (
        post.post_agency_type.str.lower() == mention.mention_agency_type.lower()
    )
    fn_prefix = (
        mention.mention_first_name[:prefix_len].casefold()
        if mention.mention_first_name
        else "Z"
    )

    # Fix empty string handling in end dates
    # Replace empty strings with NaN first, then fill with today's date
    end_dates_cleaned = post.post_end_date.replace("", pd.NaT)

# Also treat obviously invalid dates as NaT (dates before 1950)
    end_dates_cleaned = pd.to_datetime(end_dates_cleaned, errors='coerce')
    end_dates_cleaned = end_dates_cleaned.where(
        (end_dates_cleaned.isna()) | (end_dates_cleaned.dt.year >= 1950),
        pd.NaT
    )

    # Fill NaT with today's date
    end_dates_filled = end_dates_cleaned.fillna(pd.Timestamp.today())

    # Compare years instead of exact dates with buffer
    date_in_range = (
        pd.to_datetime(post.post_start_date).dt.year <= incident_year + 1
    ) & (end_dates_filled.dt.year >= incident_year - 1)

    fn_cand = post.post_first_name.str[:prefix_len].str.casefold() == fn_prefix
    fn_full_cand = (
        post.post_first_name.str.casefold() == mention.mention_first_name.casefold()
    )
    ln_cand = (
        post.post_last_name.str[:prefix_len].str.casefold()
        == mention.mention_last_name[:prefix_len].casefold()
    )
    ln_full_cand = (
        post.post_last_name.str.casefold() == mention.mention_last_name.casefold()
    )

    print(f"DEBUG: Initial filter counts:")
    print(f"- Agency type matches: {agency_type.sum()}")
    print(f"- Date in range: {date_in_range.sum()}")
    print(f"- First name prefix matches: {fn_cand.sum()}")
    print(f"- Full first name matches: {fn_full_cand.sum()}")
    print(f"- Last name prefix matches: {ln_cand.sum()}")
    print(f"- Full last name matches: {ln_full_cand.sum()}")

    cands = pd.concat(
        [
            post.loc[agency_type & date_in_range & fn_cand & ln_full_cand],
            post.loc[agency_type & date_in_range & fn_full_cand & ln_cand],
        ]
    ).drop_duplicates()

    print(f"DEBUG: Found {len(cands)} candidate matches")
    if len(cands) > 0:
        print("DEBUG: Sample candidates:")
        print(
            cands[
                [
                    "post_first_name",
                    "post_last_name",
                    "post_agency_name",
                    "post_start_date",
                    "post_end_date",
                ]
            ]
        )

    mention_df = pd.DataFrame([mention.dict() for _ in range(len(cands))])
    # Remove any columns in cands that exist in mention_df
    cands_clean = cands[[col for col in cands.columns if col not in mention_df.columns]]
    filtered_candidates = pd.concat(
        [mention_df.reset_index(drop=True), cands_clean.reset_index(drop=True)], axis=1
    )

    return filtered_candidates, full_employment_history



def fetch_all_candidates_with_same_name(
    first_name: str,
    last_name: str,
    client: NPIClient
) -> pd.DataFrame:
    """
    Fetch all officers with matching first/last name across the entire database.
    Returns records regardless of middle name, suffix, or agency.
    """
    try:
        records = client.get_officers_by_name(
            first_name=first_name,
            last_name=last_name
        )
        if records:
            record_data = [record.dict() for record in records]
            df = pd.DataFrame(record_data)
            return df.drop_duplicates()
        else:
            return pd.DataFrame()
    except Exception as e:
        print(f"DEBUG: Error fetching candidates by name: {e}")
        return pd.DataFrame()


def load_model():
    model_path = "../models/best_model_xgboost.pkl"
    """Load a pickled model from a local file path"""
    with open(model_path, "rb") as f:
        model = pickle.load(f)
    return model


# ============================================================================
# CHECKPOINT FUNCTIONS
# ============================================================================

def save_checkpoint(chunk_idx: int, results, all_candidates, invalid_candidates, full_employment_histories):
    """Save checkpoint for a completed chunk."""
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    checkpoint_file = os.path.join(CHECKPOINT_DIR, f"checkpoint_chunk_{chunk_idx}.pkl")
    checkpoint_data = {
        'results': results,
        'all_candidates': all_candidates,
        'invalid_candidates': invalid_candidates,
        'full_employment_histories': full_employment_histories
    }

    with open(checkpoint_file, 'wb') as f:
        pickle.dump(checkpoint_data, f)

    print(f"  ✓ Saved checkpoint for chunk {chunk_idx}")


def load_checkpoint(chunk_idx: int):
    """Load checkpoint for a specific chunk."""
    checkpoint_file = os.path.join(CHECKPOINT_DIR, f"checkpoint_chunk_{chunk_idx}.pkl")

    if not os.path.exists(checkpoint_file):
        return None

    with open(checkpoint_file, 'rb') as f:
        return pickle.load(f)


def save_progress(last_completed_chunk: int, total_chunks: int):
    """Save progress to JSON file."""
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    progress = {
        'last_completed_chunk': last_completed_chunk,
        'total_chunks': total_chunks,
        'timestamp': datetime.datetime.now().isoformat()
    }

    with open(PROGRESS_FILE, 'w') as f:
        json.dump(progress, f, indent=2)


def load_progress():
    """Load progress from JSON file. Returns None if no progress file exists."""
    if not os.path.exists(PROGRESS_FILE):
        return None

    with open(PROGRESS_FILE, 'r') as f:
        return json.load(f)


def clear_checkpoints():
    """Delete all checkpoint files and progress file."""
    if os.path.exists(CHECKPOINT_DIR):
        for file in os.listdir(CHECKPOINT_DIR):
            file_path = os.path.join(CHECKPOINT_DIR, file)
            if os.path.isfile(file_path):
                os.remove(file_path)
        print(f"✓ Cleared all checkpoints from {CHECKPOINT_DIR}")


def merge_checkpoints(total_chunks: int):
    """Merge all checkpoint files into final results."""
    all_results = []
    all_candidates_list = []
    all_invalid_candidates = []
    merged_employment_histories = {}

    print(f"\nMerging {total_chunks} checkpoints...")

    for chunk_idx in range(total_chunks):
        checkpoint = load_checkpoint(chunk_idx)
        if checkpoint is None:
            print(f"WARNING: Checkpoint {chunk_idx} not found - results may be incomplete")
            continue

        if len(checkpoint['results']) > 0:
            all_results.append(checkpoint['results'])

        if len(checkpoint['all_candidates']) > 0:
            all_candidates_list.append(checkpoint['all_candidates'])

        if len(checkpoint['invalid_candidates']) > 0:
            all_invalid_candidates.append(checkpoint['invalid_candidates'])

        merged_employment_histories.update(checkpoint['full_employment_histories'])

    # Concatenate all DataFrames
    final_results = pd.concat(all_results) if all_results else pd.DataFrame()
    final_all_candidates = pd.concat(all_candidates_list) if all_candidates_list else pd.DataFrame()
    final_invalid_candidates = pd.concat(all_invalid_candidates) if all_invalid_candidates else pd.DataFrame()

    print(f"  ✓ Merged {len(final_results)} matches, {len(final_all_candidates)} candidates, {len(final_invalid_candidates)} invalid")

    return final_results, final_all_candidates, final_invalid_candidates, merged_employment_histories


class PostMatcher:
    def __init__(self, api_base_url: str = None):
        api_base_url = api_base_url or os.environ.get("NPI_API_URL", "http://localhost:8000")
        self.logistic_fileid = "1s7ho9DMJSxLuESCgqZ-pJIJ3NwL4n3Lq"
        self.xgboost_fileid = "1CReLGw8s5j6agneqyymKpC6vkp9-bdaJ"
        self.logistic = None
        self.xgboost = None
        self.post = None
        self.api_base_url = api_base_url

    def classify_pairs(self, pairs, classifier):
        assert classifier in ["logistic", "xgboost", "llm"], "Invalid model"
        print(f"\nDEBUG: Classifying {len(pairs)} pairs using {classifier} classifier")

        if classifier == "logistic":
            features = featurize(pairs)
            print("\nDEBUG: Generated features:")
            print(f"- Feature columns: {features.columns.tolist()}")
            print(f"- Feature shape: {features.shape}")

            model = self._logistic_model()
            cols = [c for c in features.columns if c in model.feature_names_in_]
            print(f"\nDEBUG: Using {len(cols)} features for prediction:")
            print(f"- Selected features: {cols}")

            probabilities = model.predict_proba(features[cols])
            predictions = model.predict(features[cols])

            print(f"\nDEBUG: Predictions and probabilities summary:")
            print(f"- Total predictions: {len(predictions)}")
            print(f"- Positive predictions: {sum(predictions == 1)}")
            print("\nDEBUG: Detailed comparison and probability scores:")
            for i in range(len(predictions)):
                print(f"\nPair {i + 1}:")
                print("Mention data:")
                print(f"- First name:    '{pairs.iloc[i].mention_first_name}'")
                print(f"- Middle name:   '{pairs.iloc[i].mention_middle_name}'")
                print(f"- Last name:     '{pairs.iloc[i].mention_last_name}'")
                print(f"- Agency:        '{pairs.iloc[i].mention_agency}'")
                print(f"- Agency type:   '{pairs.iloc[i].mention_agency_type}'")
                print(f"- Incident date: '{pairs.iloc[i].mention_incident_date}'")
                print("\nPOST record data:")
                print(f"- First name:    '{pairs.iloc[i].post_first_name}'")
                print(f"- Middle name:   '{pairs.iloc[i].post_middle_name}'")
                print(f"- Last name:     '{pairs.iloc[i].post_last_name}'")
                print(f"- Agency:        '{pairs.iloc[i].post_agency_name}'")
                print(f"- Agency type:   '{pairs.iloc[i].post_agency_type}'")
                print(f"- Start date:    '{pairs.iloc[i].post_start_date}'")
                print(f"- End date:      '{pairs.iloc[i].post_end_date}'")
                print("\nProbabilities:")
                print(f"- Negative class (0) probability: {probabilities[i][0]:.4f}")
                print(f"- Positive class (1) probability: {probabilities[i][1]:.4f}")
                print(f"- Final prediction: {predictions[i]}")

            return predictions

        elif classifier == "xgboost":
            features = featurize(pairs)
            model = self._xgboost_model()
            cols = [c for c in features.columns if c in model.feature_names_in_]
            return model.predict(features[cols])
        else:
            raise NotImplementedError()

    def read_common_tbl():
            df = pd.read_csv("../data/input/common_last_names.csv")
            return df


    def find_canonical_stint(
        self, mentions: List[OfficerMention]
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
        """for a list of mentions, return the most likely matching stint from the post employment history
        Also returns all candidates with their scores for debugging, invalid candidates with validation reasons,
        and a dict mapping mention_uid to full employment history"""

        print(f"\n{'='*80}")
        print(f"STAGE 0: EARLY FILTERING - Common names and database-wide uniqueness check")
        print(f"{'='*80}")

        # Load common last names once
        common_ln_df = pd.read_csv("../data/input/common_last_names.csv")
        common_last_names = set(common_ln_df['last_name'].str.strip().str.upper())
        print(f"DEBUG: Loaded {len(common_last_names)} common last names")

        # Run early filtering checks in parallel
        print(f"DEBUG: Running early filtering checks for {len(mentions)} mentions in parallel...")

        with ThreadPoolExecutor(max_workers=100) as executor:
            # Create partial function with common_last_names bound
            check_func = lambda m: check_name_for_early_filtering(m, common_last_names)
            early_check_results = list(executor.map(check_func, mentions))

        # Split mentions into two groups
        mentions_for_pipeline = []
        mentions_for_review = []
        early_review_reasons = {}

        # Categorize early filtering reasons
        common_name_count = 0
        multiple_persons_count = 0

        for mention, (should_skip, reason) in zip(mentions, early_check_results):
            if should_skip:
                mentions_for_review.append(mention)
                early_review_reasons[mention.mention_uid] = reason

                # Count by category
                if "Common last name" in reason:
                    common_name_count += 1
                elif "Multiple persons" in reason:
                    multiple_persons_count += 1
            else:
                mentions_for_pipeline.append(mention)

        print(f"\n{'='*80}")
        print(f"EARLY FILTERING RESULTS:")
        print(f"{'='*80}")
        print(f"  Total officers: {len(mentions)}")
        print(f"  Early filtered (SKIPPED entity resolution): {len(mentions_for_review)}")
        print(f"    • Common last names: {common_name_count}")
        print(f"    • Multiple persons in database: {multiple_persons_count}")
        print(f"  Proceeding to entity resolution: {len(mentions_for_pipeline)}")
        print(f"{'='*80}\n")

        # Store full employment histories
        full_employment_histories = {}

        # Only generate candidates for mentions that passed early filtering
        if len(mentions_for_pipeline) > 0:
            print(f"DEBUG: Generating candidates for {len(mentions_for_pipeline)} mentions...")
            with ThreadPoolExecutor(max_workers=100) as executor:
                results = list(executor.map(generate_candidates, mentions_for_pipeline))

            candidate_dfs = []
            for mention, (filtered_cands, full_history) in zip(mentions_for_pipeline, results):
                candidate_dfs.append(filtered_cands)
                if len(full_history) > 0:
                    full_employment_histories[mention.mention_uid] = full_history

            candidates = pd.concat(candidate_dfs) if candidate_dfs else pd.DataFrame()
        else:
            print("DEBUG: All mentions flagged in early filtering - skipping candidate generation")
            candidates = pd.DataFrame()

        print(f"\nDEBUG: Initial candidates: {len(candidates)}")

        # Track officers that were early-filtered
        early_filtered_invalid = pd.DataFrame()
        if mentions_for_review:
            # Create invalid_candidates entries for early-filtered mentions
            early_filtered_data = []
            for mention in mentions_for_review:
                early_filtered_data.append({
                    'mention_uid': mention.mention_uid,
                    'mention_first_name': mention.mention_first_name,
                    'mention_last_name': mention.mention_last_name,
                    'validation_reason': early_review_reasons[mention.mention_uid]
                })
            early_filtered_invalid = pd.DataFrame(early_filtered_data)

        # Early return if no candidates found at all
        if len(candidates) == 0:
            print("DEBUG: No candidates found for any mentions - skipping model scoring")
            return pd.DataFrame(), pd.DataFrame(), early_filtered_invalid, full_employment_histories

        model = self._xgboost_model()
        features = featurize(candidates)
        cols = [c for c in features.columns if c in model.feature_names_in_]

        features_subset = features[cols].copy()
        # Explicitly ensure DataFrame structure with column names for XGBoost
        features_subset = pd.DataFrame(
            features_subset.values,
            columns=cols,
            index=features_subset.index
        )
        probabilities = model.predict_proba(features_subset)

        candidates["match_probability"] = probabilities[:, 1]

        # Stage 1: Filter by probability threshold
        candidates_after_prob = candidates[candidates["match_probability"] > 0.5]

        print(f"DEBUG: After probability filter (>0.8): {len(candidates_after_prob)} candidates")

        # Stage 1b: Filter by exact first and last name match
        def has_exact_name_match(row):
            """Check if first name and last name match exactly (case-insensitive)"""
            mention_first = str(row.get("mention_first_name", "")).strip().upper()
            mention_last = str(row.get("mention_last_name", "")).strip().upper()
            post_first = str(row.get("post_first_name", "")).strip().upper()
            post_last = str(row.get("post_last_name", "")).strip().upper()

            return (mention_first == post_first) and (mention_last == post_last)

        # Identify candidates that failed exact name match (for flagging)
        if len(candidates_after_prob) > 0:
            candidates_after_prob["has_exact_name_match"] = candidates_after_prob.apply(has_exact_name_match, axis=1)
            failed_exact_name_match = candidates_after_prob[~candidates_after_prob["has_exact_name_match"]].copy()

            # Flag these for review
            if len(failed_exact_name_match) > 0:
                failed_exact_name_match["validation_reason"] = "High similarity score but no exact first+last name match"

            candidates = candidates_after_prob[candidates_after_prob["has_exact_name_match"]].copy()
        else:
            failed_exact_name_match = pd.DataFrame()
            candidates = pd.DataFrame()

        print(f"DEBUG: After exact name match filter: {len(candidates)} candidates")
        failed_count = len(failed_exact_name_match['mention_uid'].unique()) if len(failed_exact_name_match) > 0 else 0
        print(f"DEBUG: Flagged {failed_count} mentions for review (no exact name match)")

        # IMPORTANT: Save all candidates before filtering for debugging purposes
        all_candidates_for_debug = candidates.copy()

        # Combine early-filtered with exact-name-match failures
        invalid_candidates = pd.concat([early_filtered_invalid, failed_exact_name_match])

        if len(candidates) == 0:
            return pd.DataFrame(), all_candidates_for_debug, invalid_candidates, full_employment_histories

        # Stage 2: Select best match per mention
        print(f"\n{'='*80}")
        print(f"STAGE 2: Selecting best match per mention")
        print(f"{'='*80}")

        # Sort by: mention_uid, match_probability (desc)
        candidates_sorted = candidates.sort_values(
            by=["mention_uid", "match_probability"], ascending=[True, False]
        )

        # For each mention + person, keep best match
        best_per_person = candidates_sorted.drop_duplicates(
            subset=["mention_uid", "post_person_nbr"], keep="first"
        )

        print(f"\nAfter deduplicating by person: {len(best_per_person)} candidates")

        # For each mention, keep highest probability match
        best_matches = best_per_person.drop_duplicates(
            subset="mention_uid", keep="first"
        )

        print(f"\nBest matches to validate: {len(best_matches)}")

        # Stage 3: Apply agency validation ONLY to best matches
        print(f"\n{'='*80}")
        print(f"STAGE 3: Applying agency validation to best matches")
        print(f"{'='*80}")

        # If no best matches, return early
        if len(best_matches) == 0:
            print("DEBUG: No best matches to validate")
            return pd.DataFrame(), all_candidates_for_debug, invalid_candidates, full_employment_histories

        def is_valid_agency_match(row):
            """Check if post_agency matches mention_agency or any mentioned_agencies"""
            mention_agency = row.get("mention_agency", "")
            mentioned_agencies = row.get("mentioned_agencies", "")
            post_agency = row.get("post_agency_name", "")

            # Use the validation helper
            is_valid, reason = validate_agency_match(
                mention_agency=mention_agency,
                mentioned_agencies=mentioned_agencies,
                post_agency=post_agency,
                threshold=0.8,
            )

            if is_valid:
                print(f"    {row['mention_uid']}: VALID")
            else:
                print(f"    {row['mention_uid']}: INVALID - {reason}")

            return is_valid, reason

        # Apply validation to best matches in parallel
        print(f"DEBUG: Validating {len(best_matches)} matches in parallel...")

        with ThreadPoolExecutor(max_workers=VALIDATION_MAX_WORKERS) as executor:
            # Convert rows to list for parallel processing
            rows_list = [row for _, row in best_matches.iterrows()]
            validation_results_list = list(executor.map(is_valid_agency_match, rows_list))

        # Convert results to DataFrame columns
        is_valid_list = [result[0] for result in validation_results_list]
        reason_list = [result[1] for result in validation_results_list]

        best_matches_with_validation = best_matches.copy()
        best_matches_with_validation['is_agency_valid'] = is_valid_list
        best_matches_with_validation['validation_reason'] = reason_list

        valid_matches = best_matches_with_validation[best_matches_with_validation["is_agency_valid"] == True].copy()
        agency_invalid_matches = best_matches_with_validation[best_matches_with_validation["is_agency_valid"] == False].copy()

        # Combine all invalid candidates
        invalid_candidates = pd.concat([invalid_candidates, agency_invalid_matches])

        print(f"\n{'='*80}")
        print(f"ENTITY RESOLUTION RESULTS:")
        print(f"{'='*80}")
        print(f"Auto-matched: {len(valid_matches)}")
        failed_exact_count = len(failed_exact_name_match['mention_uid'].unique()) if len(failed_exact_name_match) > 0 else 0
        print(f"Failed entity resolution: {len(agency_invalid_matches) + failed_exact_count}")
        print(f"  • Agency validation failed: {len(agency_invalid_matches)}")
        print(f"  • No exact name match: {failed_exact_count}")
        print(f"{'='*80}\n")

        return valid_matches, all_candidates_for_debug, invalid_candidates, full_employment_histories

    def _logistic_model(self):
        if self.logistic is not None:
            return self.logistic
        else:
            logistic = load_model()
            self.logistic = logistic
            return logistic

    def _xgboost_model(self):
        if self.xgboost is not None:
            return self.xgboost
        else:
            xgboost = load_model()
            self.xgboost = xgboost
            return xgboost


if __name__ == "__main__":
    print("\nDEBUG: Starting matching process...")
    import time 
    start_time = time.time()

    input_df = pd.read_csv("../data/input/involved_officers_2-2-2026.csv")
    _sample_n = int(os.environ.get("SAMPLE_N", "100"))
    _sample_seed = os.environ.get("SAMPLE_SEED")
    input_df = input_df.sample(n=_sample_n, random_state=int(_sample_seed) if _sample_seed else None)
    print(input_df.head())

    # Check if officer_uid column exists, if not create it
    if 'officer_uid' not in input_df.columns:
        print("\nDEBUG: 'officer_uid' column not found, generating UIDs using SHA256 hash...")
        input_df['officer_uid'] = input_df.apply(generate_officer_uid, axis=1)
        print(f"DEBUG: Generated {len(input_df)} officer UIDs")
        print(f"DEBUG: Sample UIDs: {input_df['officer_uid'].head(3).tolist()}")

    # Ensure incident_year column exists (generate from incident_date if needed)
    input_df = ensure_incident_year_column(input_df)

    # input_df = input_df.fillna("")
    # input_df = input_df[~(input_df.incident_year == "")]
    # input_df = input_df[~(input_df["authoritative_caseid"].fillna("") == "")]


    input_df = input_df.fillna("")
    input_df = input_df[~(input_df.incident_year == "")]
    input_df = input_df[~(input_df["authoritative_caseid"].fillna("") == "")]
    input_df.loc[:, "agency_type"] = "POLICE"
    input_df.loc[:, "first_name"] = input_df.first_name.str.upper()
    input_df.loc[:, "last_name"] = input_df.last_name.str.upper()
    input_df.loc[:, "middle_name"] = input_df.middle_name.str.upper()
    input_df.loc[:, "suffix"] = input_df.suffix.str.upper()
    input_df.loc[:, "provisional_case_name"] = input_df.old_case_name
    # input_df = input_df.rename(columns={"old_case_name": "provisional_case_name"})
    input_df = input_df.rename(columns={"source agencies": "source_agency"})


    print("\nDEBUG: Columns before rename:")
    print(input_df.columns.tolist())

    print("\nDEBUG: Columns after rename:")
    print(input_df.columns.tolist())

    # Check if source_agency exists, if not, show what columns might be similar
    if 'source_agency' not in input_df.columns:
        print("\nWARNING: 'source_agency' column not found!")
        print("Columns that contain 'agency' or 'source':")
        for col in input_df.columns:
            if 'agency' in col.lower() or 'source' in col.lower():
                print(f"  - '{col}'")

    print(input_df)
    print(f"\nDEBUG: Read {len(input_df)} records from input CSV")
    print(input_df.shape)
    print("\nDEBUG: Input data sample:")
    print(input_df.head(1))

    # Convert to list of OfficerMention objects
    mentions = []
    for _, row in input_df.iterrows():
        mention_date = datetime.date(year=int(row["incident_year"]), month=1, day=1)
        mention = OfficerMention(
            mention_uid=row["officer_uid"],
            mention_agency_type=row["agency_type"],
            mention_incident_date=mention_date,
            mention_first_name=row["first_name"],
            mention_middle_name=row["middle_name"],
            mention_last_name=row["last_name"],
            mention_agency=row["source_agency"],
            state=row.get("state", None),
            mentioned_agencies=row.get("mentioned_agencies", ""),
        )
        mentions.append(mention)

    # ========================================================================
    # CHECKPOINT-BASED PROCESSING
    # ========================================================================

    # Check for existing progress
    progress = load_progress()
    total_chunks = (len(mentions) + CHUNK_SIZE - 1) // CHUNK_SIZE  # Ceiling division

    if progress:
        start_chunk = progress['last_completed_chunk'] + 1
        print(f"\n{'='*80}")
        print(f"RESUMING FROM CHECKPOINT")
        print(f"  Last completed chunk: {progress['last_completed_chunk']}")
        print(f"  Resuming from chunk: {start_chunk}")
        print(f"  Total chunks: {total_chunks}")
        print(f"  Last checkpoint: {progress['timestamp']}")
        print(f"{'='*80}\n")
    else:
        start_chunk = 0
        print(f"\n{'='*80}")
        print(f"STARTING FRESH RUN")
        print(f"  Total officers: {len(mentions)}")
        print(f"  Chunk size: {CHUNK_SIZE}")
        print(f"  Total chunks: {total_chunks}")
        print(f"{'='*80}\n")

    # Initialize matcher
    matcher = PostMatcher()  # api_base_url from NPI_API_URL env (default localhost:8000)
    print("DEBUG: Initialized PostMatcher with API backend\n")

    # Process each chunk
    for chunk_idx in range(start_chunk, total_chunks):
        chunk_start = chunk_idx * CHUNK_SIZE
        chunk_end = min((chunk_idx + 1) * CHUNK_SIZE, len(mentions))
        mentions_chunk = mentions[chunk_start:chunk_end]

        print(f"\n{'='*80}")
        print(f"PROCESSING CHUNK {chunk_idx + 1}/{total_chunks}")
        print(f"  Officers: {chunk_start} to {chunk_end - 1} ({len(mentions_chunk)} total)")
        print(f"{'='*80}")

        chunk_start_time = time.time()

        # Process this chunk
        results_chunk, all_candidates_chunk, invalid_candidates_chunk, histories_chunk = \
            matcher.find_canonical_stint(mentions_chunk)

        chunk_elapsed = time.time() - chunk_start_time

        print(f"\nChunk {chunk_idx + 1} completed in {chunk_elapsed:.1f}s")
        print(f"  - Matches: {len(results_chunk)}")
        print(f"  - Candidates: {len(all_candidates_chunk)}")
        print(f"  - Invalid: {len(invalid_candidates_chunk)}")

        # Save checkpoint
        save_checkpoint(chunk_idx, results_chunk, all_candidates_chunk,
                       invalid_candidates_chunk, histories_chunk)
        save_progress(chunk_idx, total_chunks)

    print(f"\n{'='*80}")
    print(f"ALL CHUNKS COMPLETED - Merging results...")
    print(f"{'='*80}\n")

    # Merge all checkpoints
    results, all_candidates, invalid_candidates, full_employment_histories = \
        merge_checkpoints(total_chunks)

    print(f"\nDEBUG: Got {len(results)} results from matcher")
    print(f"\nDEBUG: Got {len(all_candidates)} total candidates")
    print(f"\nDEBUG: Got {len(invalid_candidates)} invalid candidates (failed validation or early filtering)")
    if len(results) > 0:
        print("\nDEBUG: Sample results:")
        print(results.head(1))

    # Initialize tracking for review reasons for ALL officers
    review_reasons = {}

    # Track officers with no candidates found
    for officer_uid in input_df['officer_uid']:
        if officer_uid not in all_candidates['mention_uid'].values and officer_uid not in invalid_candidates['mention_uid'].values:
            review_reasons[officer_uid] = "No candidates found"

    if len(invalid_candidates) > 0:
        # Use validation_reason from invalid_candidates
        for _, row in invalid_candidates.iterrows():
            officer_uid = row['mention_uid']
            if officer_uid not in review_reasons:
                review_reasons[officer_uid] = row.get('validation_reason', 'Validation failed')

    # Initialize variables for tracking different match types
    auto_match_results = pd.DataFrame()

    if len(results) > 0:
        auto_match_results = results.copy()

        # Merge auto-match results with original data
        output_df = input_df.merge(
            auto_match_results[
                [
                    "mention_uid",
                    "post_person_nbr",
                    "post_separation_reason",
                    "post_first_name",
                    "post_middle_name",
                    "post_last_name",
                    "post_suffix",
                    "post_agency_name",
                    "post_agency_type",
                    "post_start_date",
                    "post_end_date",
                    "state",
                    "county",
                    "match_probability",
                ]
            ],
            left_on="officer_uid",
            right_on="mention_uid",
            how="left",
        )

        # Add review reasons for all non-matched officers
        output_df["review_reason"] = output_df["officer_uid"].map(review_reasons)

        output_df = output_df.rename(
            columns={
                "post_person_nbr": "post_uid",
                "post_separation_reason": "separation_reason",
            }
        )
    else:
        # If no matches found, add empty columns
        output_df = input_df.copy()
        output_df["post_uid"] = None
        output_df["separation_reason"] = None
        output_df["post_first_name"] = None
        output_df["post_middle_name"] = None
        output_df["post_last_name"] = None
        output_df["post_suffix"] = None
        output_df["post_agency_name"] = None
        output_df["post_agency_type"] = None
        output_df["post_start_date"] = None
        output_df["post_end_date"] = None
        output_df["state"] = None
        output_df["county"] = None
        output_df["match_probability"] = None
        output_df["review_reason"] = output_df["officer_uid"].map(review_reasons)

    column_order = [
        # Original officer info
        "first_name",
        "middle_name",
        "last_name",
        "source_agency",
        "incident_year",
        # POST match info
        "post_uid",
        "post_first_name",
        "post_middle_name",
        "post_last_name",
        "post_agency_name",
        "post_start_date",
        "post_end_date",
        # Other info
        "provisional_case_name",
        "mentioned_agencies",
        "match_probability",
        "review_reason",
    ]

    remaining_cols = [col for col in output_df.columns if col not in column_order]
    column_order.extend(remaining_cols)

    output_df = output_df[column_order]
    output_df.to_csv("../data/output/df_fast.csv", index=False)

    # ========================================================================
    # CREATE JSONL OUTPUT FILES
    # ========================================================================
    print(f"\n{'='*80}")
    print(f"CREATING JSONL OUTPUT FILES")
    print(f"{'='*80}\n")

    def format_post_person_nbr_for_url(person_nbr: str) -> str:
        """Convert POST person number to URL format (lowercase with underscores)."""
        if not person_nbr or pd.isna(person_nbr):
            return ""
        # Convert to lowercase and replace hyphens with underscores if needed
        # e.g., "B97-O96" -> "b97-o96"
        return str(person_nbr).lower().replace("-", "-")

    def create_officer_record(row, include_post_match=False):
        """Create a standardized officer record dict."""
        record = {
            "input_officer": {
                "old_case_name": row.get("old_case_name", ""),
                "case_resourceid": row.get("case_resourceid", ""),
                "tileid": row.get("tileid", ""),
                "source_agency": row.get("source_agency", ""),
                "incident_date": str(row.get("incident_date", "")),
                "officer_name": row.get("officer_name", ""),
                "first_name": row.get("first_name", ""),
                "last_name": row.get("last_name", ""),
                "middle_name": row.get("middle_name", "") if pd.notna(row.get("middle_name")) else "",
                "suffix": row.get("suffix", "") if pd.notna(row.get("suffix")) else "",
                "mentioned_agencies": row.get("mentioned_agencies", "") if pd.notna(row.get("mentioned_agencies")) else "",
                "authoritative_caseid": row.get("authoritative_caseid", ""),
                "document_link": f"https://clean.calmatters.org/cases/{row.get('authoritative_caseid', '')}" if row.get('authoritative_caseid') else ""
            }
        }

        if include_post_match and pd.notna(row.get("post_uid")):
            # Format officer URL
            person_nbr = row.get("post_uid", "")
            officer_url = ""
            if person_nbr:
                formatted_nbr = format_post_person_nbr_for_url(person_nbr)
                officer_url = f"https://national.cpdp.co/officers/california/california_{formatted_nbr}/z"

            record["post_match"] = {
                "post_person_nbr": row.get("post_uid", ""),
                "post_first_name": row.get("post_first_name", ""),
                "post_middle_name": row.get("post_middle_name", "") if pd.notna(row.get("post_middle_name")) else "",
                "post_last_name": row.get("post_last_name", ""),
                "post_suffix": row.get("post_suffix", "") if pd.notna(row.get("post_suffix")) else "",
                "post_agency_name": row.get("post_agency_name", ""),
                "post_agency_type": row.get("post_agency_type", "") if pd.notna(row.get("post_agency_type")) else "POLICE",
                "post_start_date": str(row.get("post_start_date", "")) if pd.notna(row.get("post_start_date")) else "",
                "post_end_date": str(row.get("post_end_date", "")) if pd.notna(row.get("post_end_date")) else "",
                "post_separation_reason": row.get("separation_reason", "") if pd.notna(row.get("separation_reason")) else "",
                "state": row.get("state", "") if pd.notna(row.get("state")) else "CA",
                "county": row.get("county", "") if pd.notna(row.get("county")) else "",
                "match_probability": float(row.get("match_probability", 0.0)) if pd.notna(row.get("match_probability")) else 0.0,
                "officer_link": officer_url
            }

        return record

    # Categorize all officers
    auto_matched = output_df[output_df["post_uid"].notna()].copy()
    officers_for_review = output_df[output_df["post_uid"].isna()].copy()

    # Split review officers into early filtered vs failed entity resolution
    early_filtered = officers_for_review[
        officers_for_review['review_reason'].str.contains('Common last name|Multiple persons', na=False, regex=True)
    ]
    failed_entity_resolution = officers_for_review[
        ~officers_for_review['review_reason'].str.contains('Common last name|Multiple persons', na=False, regex=True)
    ]

    # Create JSONL for auto-matched officers
    auto_matched_records = []
    for _, row in auto_matched.iterrows():
        record = create_officer_record(row, include_post_match=True)
        auto_matched_records.append(record)

    auto_matched_path = "../data/output/auto_matched.jsonl"
    with open(auto_matched_path, 'w') as f:
        for record in auto_matched_records:
            f.write(json.dumps(record, default=str) + '\n')
    print(f"✓ Wrote {len(auto_matched_records)} auto-matched records to {auto_matched_path}")

    # Create JSONL for early filtered officers
    early_filtered_records = []
    for _, row in early_filtered.iterrows():
        record = create_officer_record(row, include_post_match=False)
        record["review_reason"] = row.get("review_reason", "")

        # Get candidates for this officer if available
        officer_uid = row.get("officer_uid")
        if officer_uid and len(all_candidates) > 0:
            officer_candidates = all_candidates[all_candidates["mention_uid"] == officer_uid]
            if len(officer_candidates) > 0:
                candidates_list = []
                for _, cand in officer_candidates.iterrows():
                    candidates_list.append({
                        "post_person_nbr": cand.get("post_person_nbr", ""),
                        "post_first_name": cand.get("post_first_name", ""),
                        "post_middle_name": cand.get("post_middle_name", ""),
                        "post_last_name": cand.get("post_last_name", ""),
                        "post_agency_name": cand.get("post_agency_name", ""),
                        "post_start_date": str(cand.get("post_start_date", "")),
                        "post_end_date": str(cand.get("post_end_date", "")),
                        "match_probability": float(cand.get("match_probability", 0.0)) if pd.notna(cand.get("match_probability")) else 0.0
                    })
                record["candidates"] = candidates_list

        early_filtered_records.append(record)

    early_filtered_path = "../data/output/early_filtered.jsonl"
    with open(early_filtered_path, 'w') as f:
        for record in early_filtered_records:
            f.write(json.dumps(record, default=str) + '\n')
    print(f"✓ Wrote {len(early_filtered_records)} early filtered records to {early_filtered_path}")

    # Create JSONL for failed entity resolution officers
    failed_er_records = []
    for _, row in failed_entity_resolution.iterrows():
        record = create_officer_record(row, include_post_match=False)
        record["review_reason"] = row.get("review_reason", "")

        # Get candidates for this officer
        officer_uid = row.get("officer_uid")
        if officer_uid and len(all_candidates) > 0:
            officer_candidates = all_candidates[all_candidates["mention_uid"] == officer_uid]
            if len(officer_candidates) > 0:
                # Sort by match probability
                officer_candidates = officer_candidates.sort_values("match_probability", ascending=False)

                candidates_list = []
                for idx, cand in enumerate(officer_candidates.iterrows()):
                    _, cand = cand
                    cand_dict = {
                        "post_person_nbr": cand.get("post_person_nbr", ""),
                        "post_first_name": cand.get("post_first_name", ""),
                        "post_middle_name": cand.get("post_middle_name", ""),
                        "post_last_name": cand.get("post_last_name", ""),
                        "post_agency_name": cand.get("post_agency_name", ""),
                        "post_start_date": str(cand.get("post_start_date", "")),
                        "post_end_date": str(cand.get("post_end_date", "")),
                        "match_probability": float(cand.get("match_probability", 0.0)) if pd.notna(cand.get("match_probability")) else 0.0
                    }

                    # First candidate is the best match attempt
                    if idx == 0:
                        record["best_match_attempt"] = cand_dict

                    candidates_list.append(cand_dict)

                record["all_candidates"] = candidates_list

        failed_er_records.append(record)

    failed_er_path = "../data/output/failed_entity_resolution.jsonl"
    with open(failed_er_path, 'w') as f:
        for record in failed_er_records:
            f.write(json.dumps(record, default=str) + '\n')
    print(f"✓ Wrote {len(failed_er_records)} failed entity resolution records to {failed_er_path}")

    print(f"\n{'='*80}")
    print(f"JSONL OUTPUT SUMMARY")
    print(f"{'='*80}")
    total_input_officers = len(input_df)
    print(f"Auto-matched: {len(auto_matched_records)} ({len(auto_matched_records)/total_input_officers*100:.1f}%)")
    print(f"Early filtered: {len(early_filtered_records)} ({len(early_filtered_records)/total_input_officers*100:.1f}%)")
    print(f"Failed entity resolution: {len(failed_er_records)} ({len(failed_er_records)/total_input_officers*100:.1f}%)")
    print(f"{'='*80}\n")

    # Create debug Excel file for all officers needing review
    officers_for_review = output_df[output_df["post_uid"].isna()].copy()

    # Categorize review reasons
    early_filtered = officers_for_review[
        officers_for_review['review_reason'].str.contains('Common last name|Multiple persons', na=False, regex=True)
    ]
    no_candidates = officers_for_review[
        officers_for_review['review_reason'].str.contains('No candidates found', na=False)
    ]
    agency_failed = officers_for_review[
        officers_for_review['review_reason'].str.contains('Agency', na=False) &
        ~officers_for_review['review_reason'].str.contains('Common last name|Multiple persons', na=False, regex=True)
    ]
    other_failed = officers_for_review[
        ~officers_for_review.index.isin(early_filtered.index) &
        ~officers_for_review.index.isin(no_candidates.index) &
        ~officers_for_review.index.isin(agency_failed.index)
    ]

    # Calculate summary statistics for use in logs and Excel
    total_records = len(input_df)
    matched_records = output_df["post_uid"].notna().sum()
    early_filtered_count = len(early_filtered)
    entity_res_failed_count = len(officers_for_review) - early_filtered_count

    print(f"\n{'='*80}")
    print(f"REVIEW CATEGORIZATION")
    print(f"{'='*80}")
    print(f"Early filtered (SKIPPED entity resolution): {early_filtered_count}")
    print(f"  • Common last names: {len(early_filtered[early_filtered['review_reason'].str.contains('Common last name', na=False)])}")
    print(f"  • Multiple persons in DB: {len(early_filtered[early_filtered['review_reason'].str.contains('Multiple persons', na=False)])}")
    print(f"\nWent through entity resolution but FAILED: {entity_res_failed_count}")
    print(f"  • No candidates found: {len(no_candidates)}")
    print(f"  • Agency validation failed: {len(agency_failed)}")
    print(f"  • Other reasons: {len(other_failed)}")
    print(f"{'='*80}\n")

    with pd.ExcelWriter(
        "../data/output/unmatched_fast.xlsx", engine="openpyxl"
    ) as writer:
        summary_data = {
            "Metric": [
                "Total Officers",
                "Auto-Matched",
                "Skipped (Early Filtered)",
                "  • Common Last Names",
                "  • Multiple Persons in DB",
                "Failed Entity Resolution",
                "  • No Candidates Found",
                "  • Agency Validation Failed",
                "  • Other Reasons",
                "Total Needing Manual Review",
            ],
            "Count": [
                len(input_df),
                matched_records,
                early_filtered_count,
                len(early_filtered[early_filtered['review_reason'].str.contains('Common last name', na=False)]),
                len(early_filtered[early_filtered['review_reason'].str.contains('Multiple persons', na=False)]),
                entity_res_failed_count,
                len(no_candidates),
                len(agency_failed),
                len(other_failed),
                len(officers_for_review),
            ],
            "Percentage": [
                "100.0%",
                f"{matched_records/total_records*100:.1f}%",
                f"{early_filtered_count/total_records*100:.1f}%",
                f"{len(early_filtered[early_filtered['review_reason'].str.contains('Common last name', na=False)])/total_records*100:.1f}%",
                f"{len(early_filtered[early_filtered['review_reason'].str.contains('Multiple persons', na=False)])/total_records*100:.1f}%",
                f"{entity_res_failed_count/total_records*100:.1f}%",
                f"{len(no_candidates)/total_records*100:.1f}%",
                f"{len(agency_failed)/total_records*100:.1f}%",
                f"{len(other_failed)/total_records*100:.1f}%",
                f"{len(officers_for_review)/total_records*100:.1f}%",
            ]
        }

        summary_df = pd.DataFrame(summary_data)
        summary_df.to_excel(writer, sheet_name="Summary", index=False)

        # Create a sheet for each officer needing review
        for idx, (_, officer) in enumerate(officers_for_review.iterrows()):
            officer_uid = officer["officer_uid"]

            # Get all candidates for this officer
            officer_candidates = all_candidates[
                all_candidates["mention_uid"] == officer_uid
            ].copy()

            # Sort by match probability descending
            if len(officer_candidates) > 0:
                officer_candidates = officer_candidates.sort_values(
                    "match_probability", ascending=False
                )

            # Create sheet name (Excel has 31 char limit)
            sheet_name = f"{officer['last_name'][:20]}_{idx}"

            # Get review reason
            review_reason = officer.get("review_reason", "Unknown reason")

            # Create officer info dataframe
            provisional_case_name = officer.get("provisional_case_name", "")
            document_link = f"https://clean-juno-dev-cmhrd4e2fef3hrd5.westus2-01.azurewebsites.net/demo/cases/{provisional_case_name}?query_string=" if provisional_case_name else ""

            officer_info = pd.DataFrame(
                {
                    "Field": [
                        "Officer UID",
                        "First Name",
                        "Middle Name",
                        "Last Name",
                        "Agency",
                        "Agency Type",
                        "Incident Year",
                        "Mentioned Agencies",
                        "Provisional Case Name",
                        "Document Link",
                        "Review Reason",
                    ],
                    "Value": [
                        officer_uid,
                        officer["first_name"],
                        officer["middle_name"],
                        officer["last_name"],
                        officer["source_agency"],
                        officer["agency_type"],
                        officer["incident_year"],
                        officer.get("mentioned_agencies", ""),
                        provisional_case_name,
                        document_link,
                        review_reason,
                    ],
                }
            )

            officer_info.to_excel(
                writer, sheet_name=sheet_name, index=False, startrow=0
            )

            if len(officer_candidates) > 0:
                current_row = len(officer_info) + 2

                # Get unique candidate POST IDs
                unique_candidates = officer_candidates['post_person_nbr'].unique()

                # For each unique candidate, show their info
                for candidate_nbr in unique_candidates:
                    # Get the candidate's match info
                    candidate_info = officer_candidates[officer_candidates['post_person_nbr'] == candidate_nbr].iloc[0]

                    # Add header for this candidate
                    pd.DataFrame({"": ["", f"CANDIDATE: {candidate_info['post_first_name']} {candidate_info['post_last_name']} (POST ID: {candidate_nbr}) - Match Probability: {candidate_info['match_probability']:.2%}"]}).to_excel(
                        writer,
                        sheet_name=sheet_name,
                        index=False,
                        header=False,
                        startrow=current_row,
                    )
                    current_row += 3

                    # Get full employment history for this candidate
                    candidate_history = all_candidates[
                        (all_candidates['post_person_nbr'] == candidate_nbr) &
                        (all_candidates['mention_uid'] == officer_uid)
                    ].copy()

                    if len(candidate_history) > 0:
                        # Display key columns for employment history
                        history_display_cols = [
                            "post_person_nbr",
                            "post_first_name",
                            "post_middle_name",
                            "post_last_name",
                            "post_agency_name",
                            "post_agency_type",
                            "post_start_date",
                            "post_end_date",
                        ]
                        # Only include columns that exist
                        history_display_cols = [col for col in history_display_cols if col in candidate_history.columns]

                        candidate_history[history_display_cols].to_excel(
                            writer,
                            sheet_name=sheet_name,
                            index=False,
                            startrow=current_row,
                        )
                        current_row += len(candidate_history) + 3
                    else:
                        pd.DataFrame({"": ["No employment history found"]}).to_excel(
                            writer,
                            sheet_name=sheet_name,
                            index=False,
                            header=False,
                            startrow=current_row,
                        )
                        current_row += 3

            else:
                # Check if this was early filtered
                review_reason = officer.get("review_reason", "")
                if "Common last name" in review_reason or "Multiple persons" in review_reason:
                    message = f"SKIPPED ENTITY RESOLUTION - {review_reason}\n\nThis officer was flagged during early filtering and did not proceed through the entity resolution pipeline."
                else:
                    message = "NO CANDIDATES FOUND\n\nNo matching records were found in the POST database after entity resolution."

                pd.DataFrame({"": ["", message]}).to_excel(
                    writer,
                    sheet_name=sheet_name,
                    index=False,
                    header=False,
                    startrow=len(officer_info) + 2,
                )

    print(f"\n{'='*80}")
    print(f"FINAL STATISTICS")
    print(f"{'='*80}")
    print(f"Total officers processed: {total_records}")
    print(f"\nAuto-matched: {matched_records} ({matched_records/total_records*100:.1f}%)")
    print(f"Skipped (early filtered): {early_filtered_count} ({early_filtered_count/total_records*100:.1f}%)")
    print(f"Failed entity resolution: {entity_res_failed_count} ({entity_res_failed_count/total_records*100:.1f}%)")
    print(f"\nTotal needing manual review: {len(officers_for_review)} ({len(officers_for_review)/total_records*100:.1f}%)")
    print(f"\nOutput files:")
    print(f"  • df_fast.csv - All results")
    print(f"  • unmatched_fast.xlsx - Officers needing review")
    print(f"{'='*80}")

    # Note: Post-matching conflict checking (lines 941-1261 from original) is omitted
    # for this fast version since early filtering already handles database-wide uniqueness
    print(f"\nNOTE: This fast version uses early filtering to skip expensive operations.")
    print(f"Post-matching conflict detection has been replaced by upfront database-wide uniqueness checks.")

    # Clean up checkpoints after successful completion
    print(f"\n{'='*80}")
    print("Cleaning up checkpoints...")
    print(f"{'='*80}")
    clear_checkpoints()

    end_time = time.time()
    print(f"\n{'='*80}")
    print(f"TOTAL TIME ELAPSED: {end_time - start_time:.1f}s ({(end_time - start_time)/60:.1f} minutes)")
    print(f"{'='*80}")
