from typing import List
import pandas as pd
import pickle
import datetime
from api import NPIClient

from features import featurize
from models.src import OfficerMention

# TODO we should be matching on agency name? quizas


def generate_candidates(mention: OfficerMention) -> pd.DataFrame:
    """for a given mention, return a list of possibly matching stints from the post employment history"""

    incident_year = mention.mention_incident_date.year
    prefix_len = 2

    print(f"\nDEBUG: Processing mention: {mention}")
    print(f"DEBUG: Incident year: {incident_year}")

    client = NPIClient(base_url="http://localhost:8000")

    print(f"first_name: {mention.mention_first_name}")
    print(f"last_name: {mention.mention_last_name}")
    print(f"year: {mention.mention_incident_date.year}")

    print(
        f"DEBUG: Fetching targeted candidates for: {mention.mention_first_name} {mention.mention_last_name}"
    )
    api_candidates = client.get_candidates_for_mention(
        first_name=mention.mention_first_name,
        last_name=mention.mention_last_name,
        incident_year=mention.mention_incident_date.year,
        state=mention.state,
    )

    if not api_candidates:
        print("DEBUG: No candidates found from API")
        return pd.DataFrame()

    post_data = [candidate.dict() for candidate in api_candidates]
    post = pd.DataFrame(post_data)
    print(f"DEBUG: Retrieved {len(post)} candidates from API")
    print(post)

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
    end_dates_filled = end_dates_cleaned.fillna(pd.Timestamp.today())

    # Compare years instead of exact dates with buffer
    date_in_range = (
        pd.to_datetime(post.post_start_date).dt.year <= incident_year + 1
    ) & (pd.to_datetime(end_dates_filled).dt.year >= incident_year - 1)

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
    return pd.concat(
        [mention_df.reset_index(drop=True), cands.reset_index(drop=True)], axis=1
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

    def find_canonical_stint(
        self, mentions: List[OfficerMention]
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """for a list of mentions, return the most likely matching stint from the post employment history
        Also returns all candidates with their scores for debugging"""
        candidates = pd.concat(generate_candidates(mention) for mention in mentions)

        # Get probabilities instead of just predictions
        # model = self._logistic_model()

        model = self._xgboost_model()
        features = featurize(candidates)
        cols = [c for c in features.columns if c in model.feature_names_in_]
        probabilities = model.predict_proba(features[cols])

        candidates["match_probability"] = probabilities[:, 1]

        # For each mention_uid, keep only the highest probability match
        best_matches = candidates[candidates["match_probability"] > 0.5].sort_values(
            "match_probability", ascending=False
        )
        best_matches = best_matches.drop_duplicates(subset="mention_uid", keep="first")

        # Return both best matches and all candidates for debugging
        return best_matches, candidates

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

    input_df = pd.read_csv("data/input/df_lapd_short_ma.csv")
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
        )
        mentions.append(mention)
        print(f"\nDEBUG: Created mention object:")
        print(mention)

    # Initialize matcher and process all mentions
    matcher = PostMatcher(api_base_url="http://localhost:8000")
    print("\nDEBUG: Initialized PostMatcher with API backend")

    # Get both matches and all candidates
    results, all_candidates = matcher.find_canonical_stint(mentions)
    print(f"\nDEBUG: Got {len(results)} results from matcher")
    print(f"\nDEBUG: Got {len(all_candidates)} total candidates")
    if len(results) > 0:
        print("\nDEBUG: Sample results:")
        print(results.head(1))

    if len(results) > 0:
        # Merge back with original data
        output_df = input_df.merge(
            results[
                [
                    "mention_uid",
                    "post_person_nbr",
                    "post_separation_reason",
                    "post_first_name",
                    "post_middle_name",
                    "post_last_name",
                ]
            ],
            left_on="officer_uid",
            right_on="mention_uid",
            how="left",
        )
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

    column_order = [
        # Original officer info
        "officer_uid",
        "first_name",
        "middle_name",
        "last_name",
        # POST match info
        "post_uid",
        "post_first_name",
        "post_middle_name",
        "post_last_name",
        # Other info
        "source_agency",
        "agency_type",
        "incident_year",
        "separation_reason",
    ]

    remaining_cols = [col for col in output_df.columns if col not in column_order]
    column_order.extend(remaining_cols)

    output_df = output_df[column_order]
    output_df.to_csv("data/output/df.csv", index=False)

    # Create debug Excel file for unmatched officers
    unmatched_df = output_df[output_df["post_uid"].isna()]
    print(f"\nDEBUG: Creating debug Excel for {len(unmatched_df)} unmatched officers")

    with pd.ExcelWriter(
        "data/output/unmatched_debug.xlsx", engine="openpyxl"
    ) as writer:
        summary_df = pd.DataFrame(
            {
                "Total Officers": [len(input_df)],
                "Matched Officers": [output_df["post_uid"].notna().sum()],
                "Unmatched Officers": [len(unmatched_df)],
                "Match Rate": [
                    f"{output_df['post_uid'].notna().sum() / len(input_df) * 100:.1f}%"
                ],
            }
        )
        summary_df.to_excel(writer, sheet_name="Summary", index=False)

        # Create a sheet for each unmatched officer
        for idx, (_, unmatched_officer) in enumerate(unmatched_df.iterrows()):
            officer_uid = unmatched_officer["officer_uid"]

            # Get all candidates for this officer
            officer_candidates = all_candidates[
                all_candidates["mention_uid"] == officer_uid
            ].copy()

            # Sort by match probability descending
            officer_candidates = officer_candidates.sort_values(
                "match_probability", ascending=False
            )

            # Create sheet name (Excel has 31 char limit)
            sheet_name = f"{unmatched_officer['last_name'][:20]}_{idx}"

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
                    ],
                    "Value": [
                        officer_uid,
                        unmatched_officer["first_name"],
                        unmatched_officer["middle_name"],
                        unmatched_officer["last_name"],
                        unmatched_officer["source_agency"],
                        unmatched_officer["agency_type"],
                        unmatched_officer["incident_year"],
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
        f"Found matches for {matched_records} records ({matched_records/total_records*100:.1f}%)"
    )
    print(f"Created debug Excel file: data/output/unmatched_debug.xlsx")
