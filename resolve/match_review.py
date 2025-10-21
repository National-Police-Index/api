from typing import List
import pandas as pd
import pickle
import datetime
from api import NPIClient

# from features_rust import featurize
from models.src import OfficerMention
from concurrent.futures import ThreadPoolExecutor, as_completed
from features import featurize
from helpers import validate_agency_match

# TODO we should be matching on agency name? quizas
## 2 things to do:
# 1) if no county is found, then gen candidates over entire table,
# 1) we should always prioritize the highest proba match that also is in mentioned agencies


def generate_candidates(mention: OfficerMention) -> pd.DataFrame:
    """for a given mention, return a list of possibly matching stints from the post employment history"""

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
        return pd.DataFrame()

    post_data = [candidate.dict() for candidate in api_candidates]
    post = pd.DataFrame(post_data)
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
            return pd.DataFrame()
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
    return pd.concat(
        [mention_df.reset_index(drop=True), cands_clean.reset_index(drop=True)], axis=1
    )


def load_model():
    model_path = "models/best_model_xgboost.pkl"
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
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """for a list of mentions, return the most likely matching stint from the post employment history
        Also returns all candidates with their scores for debugging, and invalid candidates with validation reasons"""

        with ThreadPoolExecutor() as executor:
            candidate_dfs = list(executor.map(generate_candidates, mentions))

        candidates = pd.concat(candidate_dfs)

        model = self._xgboost_model()
        features = featurize(candidates)
        cols = [c for c in features.columns if c in model.feature_names_in_]
        probabilities = model.predict_proba(features[cols])

        candidates["match_probability"] = probabilities[:, 1]

        print(f"\nDEBUG: Initial candidates: {len(candidates)}")

        # Stage 1: Filter by probability threshold
        candidates = candidates[candidates["match_probability"] > 0.2]

        print(f"DEBUG: After probability filter (>0.5): {len(candidates)} candidates")

        if len(candidates) == 0:
            return candidates, candidates, pd.DataFrame()

        # IMPORTANT: Save all candidates before filtering for debugging purposes
        all_candidates_for_debug = candidates.copy()

        # Stage 2: Check for issues requiring human review
        print(f"\n{'='*80}")
        print(f"STAGE 2: Checking for issues requiring human review")
        print(f"{'='*80}")

        # Load common last names
        common_ln_df = pd.read_csv("data/input/common_last_names.csv")
        common_last_names = set(common_ln_df['last_name'].str.strip().str.upper())
        print(f"DEBUG: Loaded {len(common_last_names)} common last names")

        mention_groups = candidates.groupby('mention_uid')
        needs_review_list = []
        invalid_candidates = pd.DataFrame()

        for mention_uid, group in mention_groups:
            review_reasons = []
            
            unique_persons = group['post_person_nbr'].nunique()
            mention_last_name = group.iloc[0]['mention_last_name'].strip().upper()
            
            # # Check 1: Multiple plausible persons
            # if unique_persons >= 2:
            #     mention_first_name = group.iloc[0]['mention_first_name'].strip().upper()
                
            #     # Filter to exact last name matches
            #     exact_lastname_matches = group[
            #         group['post_last_name'].str.strip().str.upper() == mention_last_name
            #     ]
                
            #     unique_persons_exact_ln = exact_lastname_matches['post_person_nbr'].nunique()
                
            #     if unique_persons_exact_ln >= 2:
            #         high_prob_matches = exact_lastname_matches[
            #             exact_lastname_matches['match_probability'] > 0.75
            #         ]
                    
            #         unique_high_prob_persons = high_prob_matches['post_person_nbr'].nunique()
                    
            #         if unique_high_prob_persons >= 2:
            #             review_reasons.append(
            #                 f"Multiple plausible persons ({unique_high_prob_persons} different persons with exact last name match)"
            #             )

           # Check 2: Multiple persons with exact same name
            mention_first_name = group.iloc[0]['mention_first_name'].strip().upper()
            mention_last_name = group.iloc[0]['mention_last_name'].strip().upper()

            # Check if there are multiple different persons with the EXACT same first AND last name as the mention
            exact_name_matches = group[
                (group['post_first_name'].str.strip().str.upper() == mention_first_name) &
                (group['post_last_name'].str.strip().str.upper() == mention_last_name)
            ]

            # Count unique person_nbrs with this exact full name
            unique_persons_with_exact_name = exact_name_matches['post_person_nbr'].nunique()

            # Flag if 2+ different persons share the exact first AND last name
            if unique_persons_with_exact_name >= 2:
                if mention_last_name in common_last_names:
                    review_reasons.append("Common name - multiple persons with exact match")
                else:
                    review_reasons.append("Same name, different persons - needs verification")
            
            # If there are any review reasons, flag this mention
            if review_reasons:
                group_copy = group.copy()
                group_copy['validation_reason'] = "; ".join(review_reasons) + " - needs human review"
                needs_review_list.append(group_copy)
                print(f"  NEEDS REVIEW: {mention_uid} - {'; '.join(review_reasons)}")

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
            return pd.DataFrame(), all_candidates_for_debug, invalid_candidates

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
            return pd.DataFrame(), all_candidates_for_debug, invalid_candidates

        def is_valid_agency_match(row):
            """Check if post_agency matches mention_agency or any mentioned_agencies"""
            mention_agency = row.get("mention_agency", "")
            mentioned_agencies = row.get("mentioned_agencies", "")
            post_agency = row.get("post_agency_name", "")

            # Skip validation for corrections agencies
            if mention_agency and "corrections" in mention_agency.lower():
                print(f"    {row['mention_uid']}: VALID (corrections agency - skip validation)")
                return True, ""

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

        # Apply validation to best matches only
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
        print(f"- Total invalid/review: {len(invalid_candidates)}")
        print(f"{'='*80}\n")

        print(f"\nFinal valid matches:")
        for idx, row in valid_matches.iterrows():
            print(
                f"  - {row['mention_uid']}: {row['post_first_name']} {row['post_last_name']} @ {row['post_agency_name']} (prob: {row['match_probability']:.4f})"
            )

        return valid_matches, all_candidates_for_debug, invalid_candidates
    

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

    input_df = pd.read_csv("data/input/mentioned_agencies_agency_type_v2.csv")

    # input_df = input_df[input_df.fillna("").source_agency.str.contains(r"Los Angeles County Sheriff\'s Office")]
    # input_df = input_df[input_df.fillna("").last_name.str.contains(r"GARCIA")]
    # input_df = input_df[input_df.fillna("").last_name.str.contains(r"PANTOJA")]

    # input_df = input_df.sample(n=200)
    print(input_df)
    print(f"\nDEBUG: Read {len(input_df)} records from input CSV")
    print("\nDEBUG: Input data sample:")
    print(input_df.head(1))

    input_df = input_df.fillna("")
    input_df = input_df[~(input_df.incident_year == "")]

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
    results, all_candidates, invalid_candidates = matcher.find_canonical_stint(mentions)
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
    output_df.to_csv("data/output/df.csv", index=False)

    # Create debug Excel file for all officers needing review (unmatched + common last name)
    officers_for_review = output_df[output_df["post_uid"].isna()].copy()
    
    print(f"\nDEBUG: Creating debug Excel for {len(officers_for_review)} officers needing review")
    
    # Count by reason
    reason_counts = officers_for_review['review_reason'].value_counts()
    print("\nDEBUG: Review reasons breakdown:")
    for reason, count in reason_counts.items():
        print(f"  - {reason}: {count}")

    with pd.ExcelWriter(
        "data/output/unmatched_debug.xlsx", engine="openpyxl"
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
                        review_reason,
                    ],
                }
            )

            officer_info.to_excel(
                writer, sheet_name=sheet_name, index=False, startrow=0
            )

            if len(officer_candidates) > 0:
                pd.DataFrame({"": ["", "CANDIDATE MATCHES:"]}).to_excel(
                    writer,
                    sheet_name=sheet_name,
                    index=False,
                    header=False,
                    startrow=len(officer_info) + 2,
                )

                display_cols = [
                    "match_probability",
                    "post_person_nbr",
                    "post_first_name",
                    "post_middle_name",
                    "post_last_name",
                    "post_agency_name",
                    "post_agency_type",
                    "post_start_date",
                    "post_end_date",
                ]

                officer_candidates[display_cols].to_excel(
                    writer,
                    sheet_name=sheet_name,
                    index=False,
                    startrow=len(officer_info) + 5,
                )
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