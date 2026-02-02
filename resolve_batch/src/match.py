from typing import List
import pandas as pd
import pickle
import datetime
import sys
import os
import hashlib

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from api import NPIClient
from models.src import OfficerMention
from concurrent.futures import ThreadPoolExecutor, as_completed
from features import featurize
from helpers import validate_agency_match


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


def generate_candidates(mention: OfficerMention) -> tuple[pd.DataFrame, pd.DataFrame]:
    """for a given mention, return a list of possibly matching stints from the post employment history
    Returns: (filtered_candidates, full_employment_history)"""

    incident_year = mention.mention_incident_date.year
    prefix_len = 2

    print(f"\nDEBUG: Processing mention: {mention}")
    print(f"DEBUG: Incident year: {incident_year}")
    print(f"DEBUG: Agency type: {mention.mention_agency_type}")

    client = NPIClient(base_url="http://localhost:8000")

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

    post_data = [candidate.model_dump() for candidate in api_candidates]
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

    # Parse start dates with error handling
    start_dates = pd.to_datetime(post.post_start_date, errors='coerce')

    # Compare years instead of exact dates with buffer
    date_in_range = (
        start_dates.dt.year <= incident_year + 1
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

    mention_df = pd.DataFrame([mention.model_dump() for _ in range(len(cands))])
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
            record_data = [record.model_dump() for record in records]
            df = pd.DataFrame(record_data)
            return df.drop_duplicates()
        else:
            return pd.DataFrame()
    except Exception as e:
        print(f"DEBUG: Error fetching candidates by name: {e}")
        return pd.DataFrame()


def process_mention_with_prefetched_data(
    mention: OfficerMention,
    counties_map: dict,
    candidates_map: dict,
    mention_idx: int
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Process a single mention using pre-fetched batch data.

    Args:
        mention: The officer mention to process
        counties_map: Pre-fetched county mappings {agency_name: county}
        candidates_map: Pre-fetched candidates {mention_idx: [PostEmploymentRecord, ...]}
        mention_idx: Index of this mention in the batch

    Returns:
        (filtered_candidates, full_employment_history) tuple
    """
    incident_year = mention.mention_incident_date.year
    prefix_len = 2

    print(f"\nDEBUG: Processing mention: {mention}")
    print(f"DEBUG: Incident year: {incident_year}")
    print(f"DEBUG: Agency type: {mention.mention_agency_type}")

    # Get pre-fetched county
    source_county = None
    if mention.mention_agency and mention.mention_agency_type.upper() != "CORRECTIONS":
        source_county = counties_map.get(mention.mention_agency)
        print(f"DEBUG: Source agency '{mention.mention_agency}' is in county: {source_county}")
    elif mention.mention_agency_type.upper() == "CORRECTIONS":
        print(f"DEBUG: CORRECTIONS agency type - skipping county lookup")

    # Get pre-fetched candidates
    api_candidates = candidates_map.get(str(mention_idx), [])

    if not api_candidates:
        print("DEBUG: No candidates found from batch")
        return pd.DataFrame(), pd.DataFrame()

    post_data = [candidate.model_dump() for candidate in api_candidates]
    post = pd.DataFrame(post_data)

    # PRESERVE FULL EMPLOYMENT HISTORY BEFORE FILTERING
    full_employment_history = post.copy()

    print(f"DEBUG: Retrieved {len(post)} candidates from batch")

    # Only apply county filtering for non-CORRECTIONS agencies
    if source_county and mention.mention_agency_type.upper() != "CORRECTIONS":
        print(f"DEBUG: Filtering candidates by county: {source_county}")
        person_has_county_match = post.groupby("post_person_nbr")["county"].apply(
            lambda counties: source_county in counties.values
        )
        valid_person_nbrs = person_has_county_match[person_has_county_match].index.tolist()

        print(f"DEBUG: Found {len(valid_person_nbrs)} unique officers with employment in {source_county} county")
        post = post[post["post_person_nbr"].isin(valid_person_nbrs)]
        print(f"DEBUG: After county filtering: {len(post)} candidate records remain")

        if len(post) == 0:
            print(f"DEBUG: No candidates found with employment history in {source_county} county")
            return pd.DataFrame(), full_employment_history
    elif mention.mention_agency_type.upper() == "CORRECTIONS":
        print(f"DEBUG: CORRECTIONS agency - skipping county filtering, keeping all {len(post)} candidates")
    else:
        print(f"DEBUG: No source county found - keeping candidates from all counties ({len(post)} candidates)")

    # Apply temporal and name filtering (same as original generate_candidates)
    agency_type = (post.post_agency_type.str.lower() == mention.mention_agency_type.lower())
    fn_prefix = (mention.mention_first_name[:prefix_len].casefold() if mention.mention_first_name else "Z")

    # Fix empty string handling in end dates
    end_dates_cleaned = post.post_end_date.replace("", pd.NaT)
    end_dates_cleaned = pd.to_datetime(end_dates_cleaned, errors='coerce')
    end_dates_cleaned = end_dates_cleaned.where(
        (end_dates_cleaned.isna()) | (end_dates_cleaned.dt.year >= 1950),
        pd.NaT
    )
    end_dates_filled = end_dates_cleaned.fillna(pd.Timestamp.today())

    start_dates = pd.to_datetime(post.post_start_date, errors='coerce')

    # Temporal filtering: ±1 year buffer
    date_in_range = (
        (start_dates.dt.year <= incident_year + 1) &
        (end_dates_filled.dt.year >= incident_year - 1)
    )

    # Name filtering (matching sequential version)
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

    # Apply all filters with name matching patterns (same as sequential)
    cands = pd.concat(
        [
            post.loc[agency_type & date_in_range & fn_cand & ln_full_cand],
            post.loc[agency_type & date_in_range & fn_full_cand & ln_cand],
        ]
    ).drop_duplicates()

    print(f"DEBUG: After all filtering: {len(cands)} candidates remain")

    if len(cands) == 0:
        return pd.DataFrame(), full_employment_history

    if len(cands) > 0:
        print("DEBUG: Sample candidates:")
        print(cands[["post_first_name", "post_last_name", "post_agency_name", "post_start_date", "post_end_date"]])

    mention_df = pd.DataFrame([mention.model_dump() for _ in range(len(cands))])
    cands_clean = cands[[col for col in cands.columns if col not in mention_df.columns]]
    filtered_candidates = pd.concat(
        [mention_df.reset_index(drop=True), cands_clean.reset_index(drop=True)], axis=1
    )

    return filtered_candidates, full_employment_history


def load_model():
    model_path = "../models/best_model_xgboost.pkl"
    """Load a pickled model from a local file path"""
    with open(model_path, "rb") as f:
        model = pickle.load(f)
    return model


class PostMatcher:
    def __init__(self, api_base_url: str = "http://localhost:8000"):
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

            # Ensure we pass a DataFrame with feature names (required by newer sklearn/xgboost)
            features_subset = pd.DataFrame(features[cols], columns=cols)
            probabilities = model.predict_proba(features_subset)
            predictions = model.predict(features_subset)

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

        # Store full employment histories
        full_employment_histories = {}
        
        with ThreadPoolExecutor() as executor:
            results = list(executor.map(generate_candidates, mentions))
        
        candidate_dfs = []
        for mention, (filtered_cands, full_history) in zip(mentions, results):
            candidate_dfs.append(filtered_cands)
            if len(full_history) > 0:
                full_employment_histories[mention.mention_uid] = full_history

        candidates = pd.concat(candidate_dfs)

        model = self._xgboost_model()
        features = featurize(candidates)
        cols = [c for c in features.columns if c in model.feature_names_in_]
        # Ensure we pass a DataFrame with feature names (required by XGBoost 1.6+)
        features_subset = pd.DataFrame(features[cols], columns=cols)
        probabilities = model.predict_proba(features_subset)

        candidates["match_probability"] = probabilities[:, 1]

        print(f"\nDEBUG: Initial candidates: {len(candidates)}")

        # Stage 1: Filter by probability threshold
        candidates_after_prob = candidates[candidates["match_probability"] > 0.8]

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
        candidates_after_prob["has_exact_name_match"] = candidates_after_prob.apply(has_exact_name_match, axis=1)
        failed_exact_name_match = candidates_after_prob[~candidates_after_prob["has_exact_name_match"]].copy()

        # Flag these for review
        failed_exact_name_match["validation_reason"] = "High similarity score but no exact first+last name match"

        candidates = candidates_after_prob[candidates_after_prob["has_exact_name_match"]].copy()

        print(f"DEBUG: After exact name match filter: {len(candidates)} candidates")
        if len(failed_exact_name_match) > 0:
            print(f"DEBUG: Flagged {len(failed_exact_name_match['mention_uid'].unique())} mentions for review (no exact name match)")
        else:
            print(f"DEBUG: No mentions flagged for exact name mismatch (candidates_after_prob was empty)")

        if len(candidates) == 0 and len(failed_exact_name_match) > 0:
            # All candidates failed exact name match - return them as invalid
            return pd.DataFrame(), all_candidates_for_debug, failed_exact_name_match, full_employment_histories
        elif len(candidates) == 0:
            return candidates, candidates, pd.DataFrame(), full_employment_histories

        # IMPORTANT: Save all candidates before filtering for debugging purposes
        all_candidates_for_debug = candidates.copy()

        # Stage 2: Check for common last names FIRST
        print(f"\n{'='*80}")
        print(f"STAGE 2: Checking for common last names and other issues requiring human review")
        print(f"{'='*80}")

        # Load common last names
        common_ln_df = pd.read_csv("../data/input/common_last_names.csv")
        common_last_names = set(common_ln_df['last_name'].str.strip().str.upper())
        print(f"DEBUG: Loaded {len(common_last_names)} common last names")

        mention_groups = candidates.groupby('mention_uid')
        needs_review_list = []
        # Initialize invalid_candidates with those that failed exact name match
        invalid_candidates = failed_exact_name_match if len(failed_exact_name_match) > 0 else pd.DataFrame()

        for mention_uid, group in mention_groups:
            review_reasons = []
            
            mention_last_name = group.iloc[0]['mention_last_name'].strip().upper()
            
            # Check 1: Common last name - flag immediately
            if mention_last_name in common_last_names:
                review_reasons.append(f"Common last name ({mention_last_name}) - requires manual verification")
                print(f"  NEEDS REVIEW: {mention_uid} - Common last name: {mention_last_name}")
            
            # Check 2: Multiple persons with exact same name (only check if not already flagged)
            if not review_reasons:  # Only check if we haven't already flagged for common name
                mention_first_name = group.iloc[0]['mention_first_name'].strip().upper()

                # Check if there are multiple different persons with the EXACT same first AND last name as the mention
                exact_name_matches = group[
                    (group['post_first_name'].str.strip().str.upper() == mention_first_name) &
                    (group['post_last_name'].str.strip().str.upper() == mention_last_name)
                ]

                # Count unique person_nbrs with this exact full name
                unique_persons_with_exact_name = exact_name_matches['post_person_nbr'].nunique()

                # Flag if 2+ different persons share the exact first AND last name
                if unique_persons_with_exact_name >= 2:
                    review_reasons.append("Same name, different persons - needs verification")
                    print(f"  NEEDS REVIEW: {mention_uid} - Multiple persons with same name")
            
            # If there are any review reasons, flag this mention
            if review_reasons:
                group_copy = group.copy()
                group_copy['validation_reason'] = "; ".join(review_reasons) + " - needs human review"
                needs_review_list.append(group_copy)

        if needs_review_list:
            needs_review_candidates = pd.concat(needs_review_list)
            
            # Add to invalid
            invalid_candidates = needs_review_candidates.copy()
            
            # Remove from candidates pool for auto-matching
            candidates = candidates[~candidates['mention_uid'].isin(needs_review_candidates['mention_uid'])]
            
            print(f"DEBUG: Flagged {len(needs_review_candidates['mention_uid'].unique())} mentions for human review")
        else:
            print(f"DEBUG: No mentions flagged for review")

        print(f"{'='*80}\n")

        # Stage 3: Select best match per mention (from non-flagged candidates)
        print(f"\n{'='*80}")
        print(f"STAGE 3: Selecting best match per mention")
        print(f"{'='*80}")

        # Check if there are any candidates left after filtering
        if len(candidates) == 0:
            print("DEBUG: No candidates remaining after review filtering")
            print(f"\n{'='*80}")
            print(f"DEBUG: Final results:")
            print(f"- Valid auto-matches: 0")
            print(f"- Invalid (agency validation failed): 0")
            print(f"- Needs review (common name/ambiguous): {len(needs_review_list) if needs_review_list else 0}")
            print(f"- Total invalid/review: {len(invalid_candidates)}")
            print(f"{'='*80}\n")
            return pd.DataFrame(), all_candidates_for_debug, invalid_candidates, full_employment_histories

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

        # Stage 4: Apply agency validation ONLY to best matches
        print(f"\n{'='*80}")
        print(f"STAGE 4: Applying agency validation to best matches")
        print(f"{'='*80}")

        # If no best matches, return early
        if len(best_matches) == 0:
            print("DEBUG: No best matches to validate")
            print(f"\n{'='*80}")
            print(f"DEBUG: Final results:")
            print(f"- Valid auto-matches: 0")
            print(f"- Invalid (agency validation failed): 0")
            print(f"- Needs review (common name/ambiguous): {len(needs_review_list) if needs_review_list else 0}")
            print(f"- Total invalid/review: {len(invalid_candidates)}")
            print(f"{'='*80}\n")
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

        # Apply validation to best matches only (SEQUENTIAL)
        validation_results = best_matches.apply(
            lambda row: pd.Series(is_valid_agency_match(row), index=['is_agency_valid', 'validation_reason']),
            axis=1
        )
        best_matches_with_validation = pd.concat([best_matches, validation_results], axis=1)
        
        valid_matches = best_matches_with_validation[best_matches_with_validation["is_agency_valid"] == True].copy()
        agency_invalid_matches = best_matches_with_validation[best_matches_with_validation["is_agency_valid"] == False].copy()

        # Combine all invalid candidates
        invalid_candidates = pd.concat([invalid_candidates, agency_invalid_matches])

        print(f"\n{'='*80}")
        print(f"DEBUG: Final results:")
        print(f"- Valid auto-matches: {len(valid_matches)}")
        print(f"- Invalid (agency validation failed): {len(agency_invalid_matches)}")
        print(f"- Needs review (common name/ambiguous): {len(needs_review_list) if needs_review_list else 0}")
        print(f"- Needs review (no exact name match): {len(failed_exact_name_match['mention_uid'].unique()) if len(failed_exact_name_match) > 0 else 0}")
        print(f"- Total invalid/review: {len(invalid_candidates)}")
        print(f"{'='*80}\n")

        print(f"\nFinal valid matches:")
        for idx, row in valid_matches.iterrows():
            print(
                f"  - {row['mention_uid']}: {row['post_first_name']} {row['post_last_name']} @ {row['post_agency_name']} (prob: {row['match_probability']:.4f})"
            )

        return valid_matches, all_candidates_for_debug, invalid_candidates, full_employment_histories

    def find_canonical_stint_batch(
        self, mentions: List[OfficerMention]
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
        """
        Batch-optimized version of find_canonical_stint using batch API endpoints.

        Pre-fetches all data in batch requests then processes in parallel.
        Significantly faster than sequential approach for large batches.

        Returns same format as find_canonical_stint:
        (valid_matches, all_candidates_for_debug, invalid_candidates, full_employment_histories)
        """
        import time

        start_time = time.time()
        print(f"\n{'='*80}")
        print(f"BATCH PROCESSING: Processing {len(mentions)} mentions")
        print(f"{'='*80}\n")

        # Step 1: Fetch counties in parallel (using individual API calls that work)
        batch_start = time.time()
        client = NPIClient(base_url=self.api_base_url)

        # Collect unique agency names (excluding CORRECTIONS)
        agency_names = []
        for mention in mentions:
            if mention.mention_agency and mention.mention_agency_type.upper() != "CORRECTIONS":
                agency_names.append(mention.mention_agency)

        unique_agencies = list(set(agency_names))
        print(f"DEBUG: Fetching counties for {len(unique_agencies)} unique agencies (parallel)...")

        # Parallelize individual county lookups for speed
        def fetch_county(agency):
            return agency, client.get_county_for_agency(agency)

        counties_map = {}
        with ThreadPoolExecutor(max_workers=10) as executor:
            futures = {executor.submit(fetch_county, agency): agency for agency in unique_agencies}
            for future in as_completed(futures):
                try:
                    agency, county = future.result()
                    if county:
                        counties_map[agency] = county
                except Exception as e:
                    print(f"DEBUG: Error fetching county for agency: {e}")

        print(f"DEBUG: Retrieved {len(counties_map)} county mappings in {time.time() - batch_start:.2f}s")

        # Step 2: Pre-fetch all candidates in batch (with chunking to avoid overload)
        batch_start = time.time()
        print(f"\nDEBUG: Fetching candidates for {len(mentions)} mentions...")

        mentions_data = []
        for mention in mentions:
            mentions_data.append({
                "first_name": mention.mention_first_name,
                "last_name": mention.mention_last_name,
                "agency_type": mention.mention_agency_type.value if hasattr(mention.mention_agency_type, 'value') else mention.mention_agency_type,
                "start_year": mention.mention_incident_date.year,
                "end_year": mention.mention_incident_date.year,
                "state": mention.state
            })

        # IMPORTANT: Chunk requests to avoid overloading server (100 mentions per request)
        chunk_size = 100
        candidates_map = {}

        for i in range(0, len(mentions_data), chunk_size):
            chunk = mentions_data[i:i+chunk_size]
            chunk_start_idx = i

            print(f"DEBUG: Fetching chunk {i//chunk_size + 1}/{(len(mentions_data)-1)//chunk_size + 1} ({len(chunk)} mentions)...")
            chunk_results = client.get_candidates_batch(chunk)

            # Remap indices to account for chunk offset
            for local_idx, candidates in chunk_results.items():
                global_idx = str(int(local_idx) + chunk_start_idx)
                candidates_map[global_idx] = candidates

            print(f"DEBUG:   -> Got {len(chunk_results)} results from chunk")

        print(f"DEBUG: Retrieved candidates for {len(candidates_map)} mentions in {time.time() - batch_start:.2f}s")

        # Step 3: Process mentions in parallel using pre-fetched data
        print(f"\nDEBUG: Processing mentions with pre-fetched data...")
        process_start = time.time()

        full_employment_histories = {}

        # Use ThreadPoolExecutor to process in parallel
        with ThreadPoolExecutor() as executor:
            # Create partial function with pre-fetched data
            from functools import partial
            process_func = partial(
                process_mention_with_prefetched_data,
                counties_map=counties_map,
                candidates_map=candidates_map
            )

            # Map mentions with their indices
            results = list(executor.map(
                lambda item: process_func(mention=item[1], mention_idx=item[0]),
                enumerate(mentions)
            ))

        print(f"DEBUG: Processed all mentions in {time.time() - process_start:.2f}s")

        # Collect results (same as original find_canonical_stint)
        candidate_dfs = []
        for mention, (filtered_cands, full_history) in zip(mentions, results):
            candidate_dfs.append(filtered_cands)
            if len(full_history) > 0:
                full_employment_histories[mention.mention_uid] = full_history

        if not candidate_dfs or all(len(df) == 0 for df in candidate_dfs):
            print("DEBUG: No candidates found for any mentions")
            return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), full_employment_histories

        candidates = pd.concat([df for df in candidate_dfs if len(df) > 0], ignore_index=True)

        # Rest of the processing is identical to original find_canonical_stint
        model = self._xgboost_model()
        features = featurize(candidates)
        cols = [c for c in features.columns if c in model.feature_names_in_]
        # Ensure we pass a DataFrame with feature names (required by XGBoost 1.6+)
        features_subset = pd.DataFrame(features[cols], columns=cols)
        probabilities = model.predict_proba(features_subset)

        candidates["match_probability"] = probabilities[:, 1]

        print(f"\nDEBUG: Initial candidates: {len(candidates)}")

        # Stage 1: Filter by probability threshold
        candidates_after_prob = candidates[candidates["match_probability"] > 0.8]
        print(f"DEBUG: After probability filter (>0.8): {len(candidates_after_prob)} candidates")

        # Stage 1b: Filter by exact first and last name match
        def has_exact_name_match(row):
            mention_first = str(row.get("mention_first_name", "")).strip().upper()
            mention_last = str(row.get("mention_last_name", "")).strip().upper()
            post_first = str(row.get("post_first_name", "")).strip().upper()
            post_last = str(row.get("post_last_name", "")).strip().upper()
            return (mention_first == post_first) and (mention_last == post_last)

        candidates_after_prob["has_exact_name_match"] = candidates_after_prob.apply(has_exact_name_match, axis=1)
        failed_exact_name_match = candidates_after_prob[~candidates_after_prob["has_exact_name_match"]].copy()
        failed_exact_name_match["validation_reason"] = "High similarity score but no exact first+last name match"

        candidates = candidates_after_prob[candidates_after_prob["has_exact_name_match"]].copy()

        print(f"DEBUG: After exact name match filter: {len(candidates)} candidates")
        if len(failed_exact_name_match) > 0:
            print(f"DEBUG: Flagged {len(failed_exact_name_match['mention_uid'].unique())} mentions for review (no exact name match)")
        else:
            print(f"DEBUG: No mentions flagged for exact name mismatch (candidates_after_prob was empty)")

        if len(candidates) == 0 and len(failed_exact_name_match) > 0:
            all_candidates_for_debug = candidates_after_prob.copy()
            return pd.DataFrame(), all_candidates_for_debug, failed_exact_name_match, full_employment_histories
        elif len(candidates) == 0:
            return candidates, candidates, pd.DataFrame(), full_employment_histories

        all_candidates_for_debug = candidates.copy()

        # Stage 2: Common last names and validation (identical to original)
        print(f"\n{'='*80}")
        print(f"STAGE 2: Checking for common last names and other issues requiring human review")
        print(f"{'='*80}")

        common_ln_df = pd.read_csv("../data/input/common_last_names.csv")
        common_last_names = set(common_ln_df['last_name'].str.strip().str.upper())
        print(f"DEBUG: Loaded {len(common_last_names)} common last names")

        mention_groups = candidates.groupby('mention_uid')
        needs_review_list = []
        invalid_candidates = failed_exact_name_match if len(failed_exact_name_match) > 0 else pd.DataFrame()

        for mention_uid, group in mention_groups:
            review_reasons = []
            mention_first_name = group.iloc[0]['mention_first_name'].strip()
            mention_last_name = group.iloc[0]['mention_last_name'].strip().upper()

            # Check 1: Common last name from CSV
            if mention_last_name in common_last_names:
                review_reasons.append(f"Common last name ({mention_last_name}) - requires manual verification")
                print(f"  NEEDS REVIEW: {mention_uid} - Common last name: {mention_last_name}")

            # Check 2: Multiple persons with exact same name (only check if not already flagged)
            if not review_reasons:  # Only check if we haven't already flagged for common name
                # Check if there are multiple different persons with the EXACT same first AND last name as the mention
                exact_name_matches = group[
                    (group['post_first_name'].str.strip().str.upper() == mention_first_name) &
                    (group['post_last_name'].str.strip().str.upper() == mention_last_name.upper())
                ]

                # Count unique person_nbrs with this exact full name
                unique_persons_with_exact_name = exact_name_matches['post_person_nbr'].nunique()

                # Flag if 2+ different persons share the exact first AND last name
                if unique_persons_with_exact_name >= 2:
                    review_reasons.append("Same name, different persons - needs verification")
                    print(f"  NEEDS REVIEW: {mention_uid} - Multiple persons with same name")

            if review_reasons:
                group_copy = group.copy()
                group_copy['validation_reason'] = '; '.join(review_reasons)
                needs_review_list.append(group_copy)

        if needs_review_list:
            needs_review_df = pd.concat(needs_review_list, ignore_index=True)
            invalid_candidates = pd.concat([invalid_candidates, needs_review_df], ignore_index=True) if len(invalid_candidates) > 0 else needs_review_df
            candidates = candidates[~candidates['mention_uid'].isin(needs_review_df['mention_uid'])]
            print(f"DEBUG: Flagged {len(needs_review_df['mention_uid'].unique())} additional mentions for review")

        print(f"DEBUG: {len(candidates['mention_uid'].unique())} mentions remaining for automatic matching")

        if len(candidates) == 0:
            print("DEBUG: No candidates remaining after review filters")
            return pd.DataFrame(), all_candidates_for_debug, invalid_candidates, full_employment_histories

        # Stage 3: Agency validation (PARALLEL)
        print(f"\n{'='*80}")
        print(f"STAGE 3: Agency Validation (PARALLEL)")
        print(f"{'='*80}")

        # Prepare best matches for each mention
        best_matches = []
        for mention_uid, group in candidates.groupby('mention_uid'):
            best_match = group.loc[group['match_probability'].idxmax()]
            best_matches.append(best_match)

        best_matches_df = pd.DataFrame(best_matches)

        def validate_match(row):
            """Validate agency match for a single row"""
            source_agency = row['mention_agency']
            post_agency = row['post_agency_name']
            mentioned_agencies_str = row.get('mentioned_agencies', '[]')
            mention_uid = row['mention_uid']  # FIX: Use column value, not index

            try:
                import ast
                mentioned_agencies = ast.literal_eval(mentioned_agencies_str) if isinstance(mentioned_agencies_str, str) else []
            except:
                mentioned_agencies = []

            is_valid, reason = validate_agency_match(
                mention_agency=source_agency,
                mentioned_agencies=str(mentioned_agencies),
                post_agency=post_agency,
                threshold=0.8
            )

            return mention_uid, is_valid, reason

        # PARALLEL validation using ThreadPoolExecutor
        print(f"Validating {len(best_matches_df)} agency matches in parallel...")
        validation_map = {}

        with ThreadPoolExecutor(max_workers=10) as executor:
            futures = {
                executor.submit(validate_match, row): idx
                for idx, row in best_matches_df.iterrows()
            }

            for future in as_completed(futures):
                try:
                    mention_uid, is_valid, reason = future.result()
                    validation_map[mention_uid] = (is_valid, reason)
                except Exception as e:
                    row_idx = futures[future]
                    print(f"    Error validating row {row_idx}: {e}")
                    # Try to get mention_uid from the dataframe
                    try:
                        mention_uid = best_matches_df.loc[row_idx, 'mention_uid']
                        validation_map[mention_uid] = (False, f"Validation error: {e}")
                    except:
                        validation_map[row_idx] = (False, f"Validation error: {e}")

        # Split into valid and invalid based on results
        valid_matches_list = []
        agency_failed_list = []

        for idx, row in best_matches_df.iterrows():
            mention_uid = row['mention_uid']  # FIX: Use mention_uid column, not index
            is_valid, reason = validation_map.get(mention_uid, (False, "Unknown error"))
            if is_valid:
                valid_matches_list.append(row)
            else:
                row_copy = row.copy()
                row_copy['validation_reason'] = reason
                agency_failed_list.append(row_copy)

        if valid_matches_list:
            valid_matches = pd.DataFrame(valid_matches_list)
        else:
            valid_matches = pd.DataFrame()

        if agency_failed_list:
            agency_failed_df = pd.DataFrame(agency_failed_list)
            invalid_candidates = pd.concat([invalid_candidates, agency_failed_df], ignore_index=True) if len(invalid_candidates) > 0 else agency_failed_df

        total_time = time.time() - start_time
        print(f"\n{'='*80}")
        print(f"BATCH PROCESSING COMPLETE: {len(mentions)} mentions in {total_time:.2f}s ({len(mentions)/total_time:.2f} mentions/sec)")
        print(f"  - Valid matches: {len(valid_matches)}")
        print(f"  - Needs review: {len(invalid_candidates['mention_uid'].unique()) if len(invalid_candidates) > 0 else 0}")
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

    input_df = pd.read_csv("../data/input/involved_officers.csv")
    input_df = input_df.sample(n=100)
    print(input_df.head())

    # Check if officer_uid column exists, if not create it
    if 'officer_uid' not in input_df.columns:
        print("\nDEBUG: 'officer_uid' column not found, generating UIDs using SHA256 hash...")
        input_df['officer_uid'] = input_df.apply(generate_officer_uid, axis=1)
        print(f"DEBUG: Generated {len(input_df)} officer UIDs")
        print(f"DEBUG: Sample UIDs: {input_df['officer_uid'].head(3).tolist()}")
    # input_df = input_df[input_df.fillna("").last_name.str.contains(r"CANELA")]



    # input_df = input_df.sample(n=10)

    # input_df = input_df[input_df.fillna("").source_agency.str.contains(r"Los Angeles County Sheriff\'s Office")]
    # input_df = input_df[input_df.fillna("").first_name.str.contains(r"RUBEN")]

    input_df = input_df.fillna("")
    input_df = input_df[~(input_df.incident_year == "")]
    input_df = input_df[~(input_df.provisional_case_name.fillna("") == "")]

    # input_df = input_df.sample(n=10)
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
        print(f"\nDEBUG: Created mention object:")
        print(mention)

    # Initialize matcher and process all mentions
    matcher = PostMatcher(api_base_url="http://localhost:8000")
    print("\nDEBUG: Initialized PostMatcher with API backend")

    # Get matches, all candidates, and invalid candidates
    results, all_candidates, invalid_candidates, full_employment_histories = matcher.find_canonical_stint_batch(mentions)

    print(f"\nDEBUG: Got {len(results)} results from matcher")
    print(f"\nDEBUG: Got {len(all_candidates)} total candidates")
    print(f"\nDEBUG: Got {len(invalid_candidates)} invalid candidates (failed agency validation)")
    if len(results) > 0:
        print("\nDEBUG: Sample results:")
        print(results.head(1))

    # Initialize tracking for review reasons for ALL officers
    review_reasons = {}
    
    # Track officers with no candidates found
    for officer_uid in input_df['officer_uid']:
        if officer_uid not in all_candidates['mention_uid'].values:
            review_reasons[officer_uid] = "No candidates found"

    if len(invalid_candidates) > 0:
        # Group by mention_uid and take the highest probability match's validation reason
        invalid_by_officer = invalid_candidates.sort_values('match_probability', ascending=False).groupby('mention_uid').first()
        for officer_uid, row in invalid_by_officer.iterrows():
            # Only set if not already set (e.g., by "no candidates")
            if officer_uid not in review_reasons:
                reason = row.get('validation_reason', 'Validation failed')
                # Don't prepend "Agency validation failed:" if the reason already explains itself
                if "Multiple plausible persons" in reason or "needs human review" in reason:
                    review_reasons[officer_uid] = reason
                else:
                    review_reasons[officer_uid] = f"Agency validation failed: {reason}"

    # Initialize variables for tracking different match types
    auto_match_results = pd.DataFrame()
    needs_review_results = pd.DataFrame()

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
                    "post_agency_name",
                    "post_start_date",
                    "post_end_date",
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
        output_df["post_agency_name"] = None
        output_df["post_start_date"] = None
        output_df["post_end_date"] = None
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
    output_df.to_csv("../data/output/df.csv", index=False)

    # Create debug Excel file for all officers needing review (unmatched + common last name)
    officers_for_review = output_df[output_df["post_uid"].isna()].copy()
    
    print(f"\nDEBUG: Creating debug Excel for {len(officers_for_review)} officers needing review")
    
    # Count by reason
    reason_counts = officers_for_review['review_reason'].value_counts()
    print("\nDEBUG: Review reasons breakdown:")
    for reason, count in reason_counts.items():
        print(f"  - {reason}: {count}")

    with pd.ExcelWriter(
        "../data/output/unmatched.xlsx", engine="openpyxl"
    ) as writer:
        summary_data = {
            "Total Officers": [len(input_df)],
            "Auto-Matched Officers": [output_df["post_uid"].notna().sum()],
            "Officers Needing Review": [len(officers_for_review)],
            "Match Rate": [
                f"{output_df['post_uid'].notna().sum() / len(input_df) * 100:.1f}%"
            ],
        }
        
        # Add breakdown by review reason
        for reason, count in reason_counts.items():
            summary_data[f"  - {reason}"] = [count]
        
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
                
                # For each unique candidate, show their full employment history
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
                    
                    # Get full employment history for this candidate from all_candidates
                    candidate_history = all_candidates[
                        (all_candidates['post_person_nbr'] == candidate_nbr) & 
                        (all_candidates['mention_uid'] == officer_uid)
                    ].copy()
                    
                    # If no history in all_candidates, try to fetch it
                    if len(candidate_history) == 0:
                        # Initialize client if needed
                        client = NPIClient(base_url="http://localhost:8000")
                        
                        # Get full employment history for this person
                        person_history_data = client.get_employment_history(candidate_nbr)
                        if person_history_data:
                            candidate_history = pd.DataFrame([h.model_dump() for h in person_history_data])
                    
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
                pd.DataFrame({"": ["", "NO CANDIDATES FOUND"]}).to_excel(
                    writer,
                    sheet_name=sheet_name,
                    index=False,
                    header=False,
                    startrow=len(officer_info) + 2,
                )

    total_records = len(input_df)
    matched_records = output_df["post_uid"].notna().sum()
    print(f"\nDEBUG: Final statistics:")
    print(f"Processed {total_records} records")
    print(
        f"Found auto-matches for {matched_records} records ({matched_records/total_records*100:.1f}%)"
    )
    print(f"Officers needing review: {len(officers_for_review)}")
    print(f"Created debug Excel file: data/output/unmatched_debug.xlsx")

    if len(auto_match_results) > 0:
        print(f"\nDEBUG: Creating employment history Excel for {len(auto_match_results)} matched officers")

        # Initialize API client for fetching all name candidates
        client = NPIClient(base_url="http://localhost:8000")

        # Track officers with and without same-name conflicts
        officers_with_conflicts = []
        officers_without_conflicts = []

        # Store sheet data to write later (to avoid empty workbook errors)
        conflicts_sheets = []
        clean_sheets = []

        # BATCH: Pre-fetch name uniqueness counts for all matched officers
        print(f"\nDEBUG: Batch checking name uniqueness for {len(auto_match_results)} matched officers")
        unique_names = list(set(
            (match["post_first_name"], match["post_last_name"])
            for _, match in auto_match_results.iterrows()
        ))
        print(f"DEBUG: Found {len(unique_names)} unique names to check")

        # Chunk API calls to avoid overloading (100 names per request)
        name_uniqueness_counts = {}
        uniqueness_chunk_size = 100

        for i in range(0, len(unique_names), uniqueness_chunk_size):
            chunk = unique_names[i:i+uniqueness_chunk_size]
            print(f"DEBUG: Checking uniqueness chunk {i//uniqueness_chunk_size + 1}/{(len(unique_names)-1)//uniqueness_chunk_size + 1} ({len(chunk)} names)...")

            chunk_counts = client.get_batch_name_uniqueness(chunk)
            name_uniqueness_counts.update(chunk_counts)

            print(f"DEBUG:   -> Got {len(chunk_counts)} uniqueness counts from chunk")

        print(f"DEBUG: Retrieved uniqueness counts for {len(name_uniqueness_counts)} names")

        for idx, (_, match) in enumerate(auto_match_results.iterrows()):
            mention_uid = match["mention_uid"]
            post_person_nbr = match["post_person_nbr"]
            
            # Get the mentioned agencies for this officer from the original input
            officer_row = input_df[input_df["officer_uid"] == mention_uid].iloc[0]
            mentioned_agencies_str = officer_row.get("mentioned_agencies", "")
            provisional_case_name = officer_row.get("provisional_case_name", "")
            
            # Get the officer's name from the match
            first_name = match["post_first_name"]
            last_name = match["post_last_name"]
            
            # Parse the mentioned agencies (they're stored as a string representation of a list)
            import ast
            try:
                if mentioned_agencies_str and mentioned_agencies_str != "":
                    mentioned_agencies = ast.literal_eval(mentioned_agencies_str)
                else:
                    mentioned_agencies = []
            except:
                mentioned_agencies = []
            
            # Get full employment history for this officer
            if mention_uid in full_employment_histories:
                full_history = full_employment_histories[mention_uid]
                
                # Filter to just this person's employment history
                person_history = full_history[full_history["post_person_nbr"] == post_person_nbr].copy()
                
                # Sort by start date
                person_history = person_history.sort_values("post_start_date")

                # Check pre-fetched uniqueness count (from batch API call)
                name_key = f"{first_name}|{last_name}"
                unique_person_count = name_uniqueness_counts.get(name_key, 1)

                print(f"DEBUG: Name '{first_name} {last_name}' appears {unique_person_count} times in database")

                # Determine if there are conflicts (2+ unique persons with same name)
                has_conflicts = unique_person_count >= 2
                all_name_candidates = pd.DataFrame()

                if has_conflicts:
                    # Only fetch full candidate details if there ARE conflicts
                    # (for populating Excel with other officers with same name)
                    print(f"DEBUG: Fetching full candidate details for '{first_name} {last_name}' (has conflicts)")
                    all_name_candidates = fetch_all_candidates_with_same_name(
                        first_name=first_name,
                        last_name=last_name,
                        client=client
                    )

                    # Remove the matched officer from the candidates to avoid duplication
                    all_name_candidates = all_name_candidates[
                        all_name_candidates['post_person_nbr'] != post_person_nbr
                    ]

                    if len(all_name_candidates) > 0:
                        all_name_candidates = all_name_candidates.sort_values("post_start_date")
                        print(f"DEBUG: Found {len(all_name_candidates)} other records with same name (excluding matched officer)")
                        print(f"DEBUG: Unique officers found: {all_name_candidates['post_person_nbr'].nunique()}")

                    officers_with_conflicts.append(mention_uid)
                else:
                    print(f"DEBUG: No conflicts - unique name")
                    officers_without_conflicts.append(mention_uid)
                
                # Create sheet name (Excel has 31 char limit)
                sheet_name = f"{match['post_last_name'][:15]}_{post_person_nbr}"[:31]

                # Create officer info header (use pre-fetched count, subtract 1 for matched officer)
                unique_officers_with_same_name = max(0, unique_person_count - 1)
                
                # Create document link
                doc_link = f"https://clean-juno-dev-cmhrd4e2fef3hrd5.westus2-01.azurewebsites.net/demo/cases/{provisional_case_name}?query_string=" if provisional_case_name else "No case link available"
                
                officer_info = pd.DataFrame({
                    "Field": [
                        "Mention UID",
                        "Provisional Case Name",
                        "Document Link",
                        "Matched POST Person Number",
                        "Match Probability",
                        "First Name",
                        "Middle Name", 
                        "Last Name",
                        "Matched Agency",
                        "Mentioned Agencies",
                        "Total Employment Stints (Matched Officer)",
                        "Other Officers with Same Name in Database",
                        "CORRECT (Enter 1 for Yes, 0 for No)"
                    ],
                    "Value": [
                        mention_uid,
                        provisional_case_name,
                        doc_link,
                        post_person_nbr,
                        f"{match['match_probability']:.4f}",
                        match['post_first_name'],
                        match['post_middle_name'],
                        match['post_last_name'],
                        match['post_agency_name'],
                        ", ".join(mentioned_agencies) if mentioned_agencies else "None",
                        len(person_history),
                        f"{unique_officers_with_same_name} unique officer(s), {len(all_name_candidates)} total record(s)",
                        ""  # Empty cell for user input
                    ]
                })
                
                # Store sheet data for later writing
                sheet_data = {
                    'sheet_name': sheet_name,
                    'officer_info': officer_info,
                    'all_name_candidates': all_name_candidates if has_conflicts else pd.DataFrame(),
                    'person_history': person_history,
                    'first_name': first_name,
                    'last_name': last_name,
                    'post_person_nbr': post_person_nbr,
                    'has_conflicts': has_conflicts,
                    'provisional_case_name': provisional_case_name
                }
                
                if has_conflicts:
                    conflicts_sheets.append(sheet_data)
                else:
                    clean_sheets.append(sheet_data)
        
        # Write conflicts file only if there are sheets to write
        if len(conflicts_sheets) > 0:
            with pd.ExcelWriter("../data/output/matched_clean_with_conflicts.xlsx", engine="openpyxl") as writer:
                for sheet_data in conflicts_sheets:
                    current_row = 0
                    sheet_data['officer_info'].to_excel(writer, sheet_name=sheet_data['sheet_name'], index=False, startrow=current_row)
                    current_row += len(sheet_data['officer_info']) + 2
                    
                    # Add officers with same name from entire database
                    if len(sheet_data['all_name_candidates']) > 0:
                        pd.DataFrame({"": ["", f"OTHER OFFICERS NAMED '{sheet_data['first_name'].upper()} {sheet_data['last_name'].upper()}' IN DATABASE (FOR MANUAL REVIEW):"]}).to_excel(
                            writer, sheet_name=sheet_data['sheet_name'], index=False, header=False, 
                            startrow=current_row
                        )
                        current_row += 3
                        
                        # Display key columns for comparison
                        display_cols = [
                            "post_person_nbr",
                            "post_first_name",
                            "post_middle_name",
                            "post_last_name",
                            "post_agency_name",
                            "post_start_date",
                            "post_end_date"
                        ]
                        # Only include columns that exist
                        display_cols = [col for col in display_cols if col in sheet_data['all_name_candidates'].columns]
                        
                        sheet_data['all_name_candidates'][display_cols].to_excel(
                            writer, sheet_name=sheet_data['sheet_name'], index=False, 
                            startrow=current_row
                        )
                        current_row += len(sheet_data['all_name_candidates']) + 3
                    
                    # Add full employment history for the MATCHED officer
                    pd.DataFrame({"": ["", f"FULL EMPLOYMENT HISTORY FOR MATCHED OFFICER (POST ID: {sheet_data['post_person_nbr']}):"]}).to_excel(
                        writer, sheet_name=sheet_data['sheet_name'], index=False, header=False, 
                        startrow=current_row
                    )
                    current_row += 3
                    
                    sheet_data['person_history'].to_excel(
                        writer, sheet_name=sheet_data['sheet_name'], index=False, 
                        startrow=current_row
                    )
            
            print(f"Created employment history Excel: data/output/matched_employment_histories.xlsx")
        else:
            print("No officers with name conflicts found - skipping matched_employment_histories.xlsx")
        
        # Write clean file only if there are sheets to write
        if len(clean_sheets) > 0:
            with pd.ExcelWriter("../data/output/matched_clean_no_conflicts.xlsx", engine="openpyxl") as writer:
                for sheet_data in clean_sheets:
                    current_row = 0
                    sheet_data['officer_info'].to_excel(writer, sheet_name=sheet_data['sheet_name'], index=False, startrow=current_row)
                    current_row += len(sheet_data['officer_info']) + 2
                    
                    # Add full employment history for the MATCHED officer
                    pd.DataFrame({"": ["", f"FULL EMPLOYMENT HISTORY FOR MATCHED OFFICER (POST ID: {sheet_data['post_person_nbr']}):"]}).to_excel(
                        writer, sheet_name=sheet_data['sheet_name'], index=False, header=False, 
                        startrow=current_row
                    )
                    current_row += 3
                    
                    sheet_data['person_history'].to_excel(
                        writer, sheet_name=sheet_data['sheet_name'], index=False, 
                        startrow=current_row
                    )
            
            print(f"Created clean matches Excel: data/output/matched_clean_no_conflicts.xlsx")
            
            # Generate JSONL file for matched_clean_no_conflicts
            import json
            
            jsonl_output_path = "../data/output/matched_clean_no_conflicts.jsonl"
            with open(jsonl_output_path, 'w') as jsonl_file:
                for sheet_data in clean_sheets:
                    # Extract officer information
                    first_name = sheet_data['first_name'].strip()
                    last_name = sheet_data['last_name'].strip()
                    post_person_nbr = sheet_data['post_person_nbr']
                    provisional_case_name = sheet_data["provisional_case_name"]
                    
                    # Lowercase the UID for the URL
                    post_person_nbr_lower = post_person_nbr.lower()
                    
                    # Hard-code state as california
                    state = "california"
                    
                    # Format the CPDP URL with lowercase names and UID
                    cpdp_url = f"https://national.cpdp.co/officers/{state}_{post_person_nbr_lower}/{last_name.lower()}-{first_name.lower()}"
                    
                    # Get the most recent work stint from person_history
                    person_history = sheet_data['person_history'].sort_values('post_start_date', ascending=False)
                    
                    if len(person_history) > 0:
                        most_recent_stint = person_history.iloc[0]
                        
                        # Convert dates to strings, handling NaT/None values
                        start_date = str(most_recent_stint['post_start_date']) if pd.notna(most_recent_stint['post_start_date']) else ""
                        end_date = str(most_recent_stint['post_end_date']) if pd.notna(most_recent_stint['post_end_date']) else ""
                        
                        work_stint = {
                            "employer": most_recent_stint['post_agency_name'],
                            "start_date": start_date,
                            "end_date": end_date
                        }
                    else:
                        # Fallback if no history (shouldn't happen but just in case)
                        work_stint = {
                            "employer": "",
                            "start_date": "",
                            "end_date": ""
                        }
                    
                    # Create JSON object in the required format
                    json_obj = {
                        "officer_first_name": f"{first_name.upper()}",
                        "officer_first_name": f"{last_name.upper()}",
                        "provisional_case_name": f"{provisional_case_name}",
                        "certification_id": post_person_nbr.upper(),
                        "npi_url": cpdp_url,
                        "work_stint": work_stint
                    }
                    
                    # Write as single line to JSONL file
                    jsonl_file.write(json.dumps(json_obj) + '\n')
            
            print(f"Created JSONL file: {jsonl_output_path}")
            print(f"Generated {len(clean_sheets)} JSONL entries for matched_clean_no_conflicts")
            
        else:
            print("No officers with clean matches found - skipping matched_clean_no_conflicts.xlsx")
        
        # Generate summary statistics
        total_officers = len(input_df)
        officers_needing_review = len(officers_for_review)
        auto_matched_with_conflicts = len(officers_with_conflicts)
        auto_matched_clean = len(officers_without_conflicts)
        
        summary_stats = f"""
    MATCHING SUMMARY STATISTICS
    {'='*60}

    TOTAL OFFICERS PROCESSED: {total_officers}

    BREAKDOWN BY CATEGORY:
    {'='*60}

    1. Officers Requiring Manual Review: {officers_needing_review} ({officers_needing_review/total_officers*100:.2f}%)
    - These officers failed validation or had ambiguous matches

    2. Auto-Matched with Name Conflicts: {auto_matched_with_conflicts} ({auto_matched_with_conflicts/total_officers*100:.2f}%)
    - These officers passed all filters but have other officers 
        with the same name in the database
    - Output file: data/output/matched_employment_histories.xlsx

    3. Auto-Matched Clean (No Conflicts): {auto_matched_clean} ({auto_matched_clean/total_officers*100:.2f}%)
    - These officers passed all filters and have unique names 
        in the database
    - Output file: data/output/matched_clean_no_conflicts.xlsx

    TOTALS:
    {'='*60}
    Total Auto-Matched: {auto_matched_with_conflicts + auto_matched_clean} ({(auto_matched_with_conflicts + auto_matched_clean)/total_officers*100:.2f}%)
    Overall Match Rate: {(auto_matched_with_conflicts + auto_matched_clean)/total_officers*100:.2f}%

    REVIEW REASONS BREAKDOWN:
    {'='*60}
    """
        
        # Add review reasons breakdown
        if len(officers_for_review) > 0:
            reason_counts = officers_for_review['review_reason'].value_counts()
            for reason, count in reason_counts.items():
                summary_stats += f"  - {reason}: {count} ({count/total_officers*100:.2f}%)\n"
        
        # Write to file
        with open("../data/output/matching_summary_stats.txt", "w") as f:
            f.write(summary_stats)
        
        print("\n" + summary_stats)
        print(f"\nSummary statistics saved to: data/output/matching_summary_stats.txt")

# ============================================================================
# CHUNKED PROCESSING FOR UNLIMITED SCALABILITY
# ============================================================================

def process_file_in_chunks(
    input_file: str,
    chunk_size: int = 1000,
    start_from: int = 0,
    output_dir: str = "../data/output"
):
    """
    Process input file in chunks for unlimited scalability.

    Processes the input CSV in chunks of chunk_size, appending results to output files.
    Can handle 100k+ persons with constant memory usage.

    Args:
        input_file: Path to input CSV file
        chunk_size: Number of persons per chunk (default: 1000)
        start_from: Row number to start from for resuming (default: 0)
        output_dir: Directory for output files

    Output Files (all appended):
        - df.csv: All candidates with match results
        - matched_clean_no_conflicts.jsonl: High-confidence auto-matches
        - matching_summary_stats.txt: Summary statistics
    """
    import pandas as pd
    import os
    from pathlib import Path

    # Output files
    output_files = {
        "all": f"{output_dir}/df.csv",
        "matched_jsonl": f"{output_dir}/matched_clean_no_conflicts.jsonl",
        "stats": f"{output_dir}/matching_summary_stats.txt"
    }

    print(f"Reading input file: {input_file}")
    df = pd.read_csv(input_file)
    total_rows = len(df)

    print(f"Total rows: {total_rows}")
    print(f"Chunk size: {chunk_size}")
    print(f"Starting from row: {start_from}")

    # Initialize output files (clear if starting from 0)
    os.makedirs(output_dir, exist_ok=True)
    if start_from == 0:
        for output_file in output_files.values():
            if os.path.exists(output_file):
                os.remove(output_file)
                print(f"Cleared existing file: {output_file}")

    # Initialize matcher
    matcher = PostMatcher(api_base_url="http://localhost:8000")

    # Process chunks
    chunk_num = start_from // chunk_size + 1
    total_stats = {'total': 0, 'matched': 0, 'unmatched': 0}

    for chunk_start in range(start_from, total_rows, chunk_size):
        chunk_end = min(chunk_start + chunk_size, total_rows)
        chunk_df = df.iloc[chunk_start:chunk_end]

        print(f"\n{'='*80}")
        print(f"Processing CHUNK {chunk_num}: Rows {chunk_start} to {chunk_end-1}")
        print(f"{'='*80}")

        # Ensure officer_uid exists
        if 'officer_uid' not in chunk_df.columns:
            chunk_df['officer_uid'] = chunk_df.apply(generate_officer_uid, axis=1)

        # Convert to OfficerMention objects
        mentions = []
        for _, row in chunk_df.iterrows():
            try:
                # Parse date
                year = int(row['incident_year'])
                month = int(row.get('incident_month', 1))
                day = int(row.get('incident_day', 1))
                incident_date = datetime.date(year, month, day)

                mention = OfficerMention(
                    mention_uid=row.get('officer_uid', ''),
                    mention_first_name=row['first_name'],
                    mention_middle_name=row.get('middle_name', ''),
                    mention_last_name=row['last_name'],
                    mention_suffix=row.get('suffix', ''),
                    mention_agency=row['source_agency'],
                    mention_agency_type=row['agency_type'],
                    mention_incident_date=incident_date,
                    mentioned_agencies=str(row.get('mentioned_agencies', '[]')),
                    state='CA',
                    provisional_case_name=row.get('provisional_case_name', '')
                )
                mentions.append(mention)
            except Exception as e:
                print(f"WARNING: Skipping row due to error: {e}")
                continue

        if not mentions:
            print("WARNING: No valid mentions in this chunk")
            continue

        # Run batch matching pipeline
        valid_matches, all_candidates, invalid_candidates, histories = \
            matcher.find_canonical_stint_batch(mentions)

        # Calculate stats
        chunk_stats = {
            'total': len(mentions),
            'matched': len(valid_matches) if len(valid_matches) > 0 else 0,
            'unmatched': len(invalid_candidates) if len(invalid_candidates) > 0 else 0
        }

        # Append to CSV
        if len(all_candidates) > 0:
            all_candidates.to_csv(
                output_files["all"],
                mode='a',
                header=not os.path.exists(output_files["all"]),
                index=False
            )

        # Append to JSONL (matched clean)
        if len(valid_matches) > 0:
            with open(output_files["matched_jsonl"], 'a') as f:
                for _, row in valid_matches.iterrows():
                    f.write(row.to_json() + '\n')

        # Append stats
        with open(output_files["stats"], 'a') as f:
            f.write(f"\n{'='*80}\n")
            f.write(f"CHUNK {chunk_num} (Rows {chunk_start}-{chunk_end-1})\n")
            f.write(f"{'='*80}\n")
            f.write(f"Matched: {chunk_stats['matched']}\n")
            f.write(f"Unmatched: {chunk_stats['unmatched']}\n")
            f.write(f"Total processed: {chunk_stats['total']}\n")

        print(f"\nChunk {chunk_num} complete:")
        print(f"  Total: {chunk_stats['total']}")
        print(f"  Matched: {chunk_stats['matched']}")
        print(f"  Unmatched: {chunk_stats['unmatched']}")

        # Accumulate stats
        for key in total_stats:
            total_stats[key] += chunk_stats.get(key, 0)

        chunk_num += 1

    # Write final summary
    print(f"\n{'='*80}")
    print("FINAL SUMMARY")
    print(f"{'='*80}")
    print(f"Total processed: {total_stats['total']}")
    print(f"Matched: {total_stats['matched']}")
    print(f"Unmatched: {total_stats['unmatched']}")

    with open(output_files["stats"], 'a') as f:
        f.write(f"\n\n{'='*80}\n")
        f.write("FINAL SUMMARY\n")
        f.write(f"{'='*80}\n")
        f.write(f"Total processed: {total_stats['total']}\n")
        f.write(f"Matched: {total_stats['matched']}\n")
        f.write(f"Unmatched: {total_stats['unmatched']}\n")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Process NPI matches with optional chunking")
    parser.add_argument("--chunk-size", type=int, default=100, help="Number of persons per chunk (default: 100)")
    parser.add_argument("--start-from", type=int, default=0, help="Row number to start from (for resuming)")
    parser.add_argument("--input", type=str, default="../data/input/involved_officers.csv", help="Input CSV file path")

    args = parser.parse_args()

    # Always use chunked processing
    print(f"Running in CHUNKED mode with chunk size {args.chunk_size}")
    process_file_in_chunks(
        input_file=args.input,
        chunk_size=args.chunk_size,
        start_from=args.start_from
    )
