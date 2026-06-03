# from Levenshtein import distance as levenshtein_distance


# def validate_agency_match(mention_agency: str, mentioned_agencies: str, post_agency: str, threshold: float = 0.8) -> tuple[bool, str]:
#     """
#     Validate if the POST agency matches either the source agency or one of the mentioned agencies.
#     """
    
#     def similarity_score(str1: str, str2: str) -> float:
#         """Calculate normalized similarity score (1.0 = identical, 0.0 = completely different)"""
#         if not str1 or not str2:
#             return 0.0
        
#         str1 = str1.lower().strip()
#         str2 = str2.lower().strip()
        
#         max_len = max(len(str1), len(str2))
#         if max_len == 0:
#             return 1.0
        
#         dist = levenshtein_distance(str1, str2)
#         return 1.0 - (dist / max_len)
    
#     if not post_agency:
#         return False, "POST agency is empty"
    
#     # Normalize post_agency once for all comparisons
#     post_agency_normalized = post_agency.lower().strip()
    
#     # Check source agency
#     if mention_agency:
#         mention_agency_normalized = mention_agency.lower().strip()
#         score = similarity_score(mention_agency_normalized, post_agency_normalized)
#         if score >= threshold:
#             return True, ""
    
#     # Parse and check mentioned agencies list
#     if mentioned_agencies and mentioned_agencies.strip():
#         try:
#             # Handle string representation of list
#             import ast
#             if isinstance(mentioned_agencies, str):
#                 agencies_list = ast.literal_eval(mentioned_agencies)
#             else:
#                 agencies_list = mentioned_agencies
            
#             if isinstance(agencies_list, list):
#                 for agency in agencies_list:
#                     if agency:  # Skip empty strings
#                         # Normalize the agency string
#                         agency_normalized = str(agency).lower().strip()
#                         score = similarity_score(agency_normalized, post_agency_normalized)
#                         if score >= threshold:
#                             return True, ""
#         except (ValueError, SyntaxError):
#             # If parsing fails, treat as single string
#             mentioned_normalized = mentioned_agencies.lower().strip()
#             score = similarity_score(mentioned_normalized, post_agency_normalized)
#             if score >= threshold:
#                 return True, ""
    
#     return False, "Agency cannot be validated"


from llm import prompt_gpt
import logging
import ast
from datetime import datetime
import pandas as pd


# Keywords that mark a source agency as non-law-enforcement (prosecutorial,
# coronial, defense, etc.). These show up as `source_agency` in incident reports
# but should never be auto-matched to a POST police/sheriff record without a
# corroborating LE agency in `mentioned_agencies`.
_NON_LE_KEYWORDS = (
    "district attorney",
    "attorney general",
    "public defender",
    "coroner",
    "medical examiner",
    " da ",
    " me ",
    "office of the da",
)

# Keywords that mark a POST agency as a law-enforcement employer.
_LE_KEYWORDS = (
    "police",
    "sheriff",
    "marshal",
    "patrol",
    "highway patrol",
    "probation",
    "corrections",
    "department of public safety",
    "public safety",
)


def _is_non_le_agency(name: str) -> bool:
    if not name:
        return False
    n = f" {name.lower().strip()} "
    if any(kw in n for kw in _LE_KEYWORDS):
        # Mixed strings like "Sheriff's Office / DA" are LE, not non-LE.
        return False
    return any(kw in n for kw in _NON_LE_KEYWORDS)


def _all_non_le(agencies: list) -> bool:
    return bool(agencies) and all(_is_non_le_agency(a) for a in agencies)


def _is_le_agency(name: str) -> bool:
    if not name:
        return False
    return any(kw in name.lower() for kw in _LE_KEYWORDS)

logger = logging.getLogger('idonea.' + __name__)


def ensure_incident_year_column(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ensure the DataFrame has an 'incident_year' column.
    If it doesn't exist, create it from 'incident_date'.
    Handles comma-separated dates by choosing the most recent one.

    Args:
        df: DataFrame with incident data

    Returns:
        DataFrame with 'incident_year' column
    """
    if 'incident_year' in df.columns:
        print("DEBUG: 'incident_year' column already exists")
        return df

    if 'incident_date' not in df.columns:
        raise ValueError("Neither 'incident_year' nor 'incident_date' column found in DataFrame")

    print("\nDEBUG: 'incident_year' column not found, generating from 'incident_date'...")

    def extract_most_recent_year(date_str):
        """Extract the year from the most recent date in a comma-separated list of dates"""
        if pd.isna(date_str) or str(date_str).strip() == "":
            return None

        date_str = str(date_str).strip()

        # Split by comma to handle multiple dates
        dates = [d.strip() for d in date_str.split(',') if d.strip()]

        if not dates:
            return None

        # Parse all dates and find the most recent
        parsed_dates = []
        for date in dates:
            try:
                # Try different date formats
                for fmt in ['%Y-%m-%d', '%m/%d/%Y', '%Y/%m/%d', '%m-%d-%Y', '%Y']:
                    try:
                        parsed_date = datetime.strptime(date, fmt)
                        parsed_dates.append(parsed_date)
                        break
                    except ValueError:
                        continue
            except Exception as e:
                print(f"DEBUG: Could not parse date '{date}': {e}")
                continue

        if not parsed_dates:
            # If no dates could be parsed, try to extract just the year
            for date in dates:
                try:
                    # Try to find a 4-digit year in the string
                    import re
                    year_match = re.search(r'\b(19|20)\d{2}\b', date)
                    if year_match:
                        return int(year_match.group())
                except:
                    continue
            return None

        # Return the year of the most recent date
        most_recent = max(parsed_dates)
        return most_recent.year

    df['incident_year'] = df['incident_date'].apply(extract_most_recent_year)

    # Count how many were successfully parsed
    parsed_count = df['incident_year'].notna().sum()
    print(f"DEBUG: Generated {parsed_count} incident years from {len(df)} records")
    print(f"DEBUG: Sample incident_year values: {df['incident_year'].head(3).tolist()}")

    return df


def validate_agency_match(
    mention_agency: str,
    mentioned_agencies: str,
    post_agency: str,
    threshold: float = 0.8  # Kept for API compatibility but not used
) -> tuple[bool, str]:
    """
    Validate if the POST agency matches either the source agency or one of the mentioned agencies.
    Uses an LLM to determine if agencies refer to the same organization.
    """
    
    # Open debug file
    debug_file = "agency_validation_debug.txt"
    with open(debug_file, "a") as f:
        f.write(f"\n{'='*80}\n")
        f.write(f"Validation at {datetime.now().isoformat()}\n")
        f.write(f"{'='*80}\n")
        f.write(f"POST agency: '{post_agency}'\n")
        f.write(f"Mention agency: '{mention_agency}'\n")
        f.write(f"Mentioned agencies: '{mentioned_agencies}'\n")
        f.write(f"\n")
    
    if not post_agency:
        with open(debug_file, "a") as f:
            f.write("Result: POST agency is empty\n")
        return False, "POST agency is empty"
    
    # Parse mentioned_agencies if it's a string representation of a list
    agencies_to_check = []
    
    if mention_agency:
        agencies_to_check.append(mention_agency)
        with open(debug_file, "a") as f:
            f.write(f"Added mention_agency to check list: '{mention_agency}'\n")
    
    if mentioned_agencies and mentioned_agencies.strip():
        try:
            if isinstance(mentioned_agencies, str):
                agencies_list = ast.literal_eval(mentioned_agencies)
            else:
                agencies_list = mentioned_agencies
            
            if isinstance(agencies_list, list):
                valid_agencies = [str(a) for a in agencies_list if a]
                agencies_to_check.extend(valid_agencies)
                with open(debug_file, "a") as f:
                    f.write(f"Added {len(valid_agencies)} agencies from list: {valid_agencies}\n")
        except (ValueError, SyntaxError) as e:
            # If parsing fails, treat as single string
            agencies_to_check.append(mentioned_agencies)
            with open(debug_file, "a") as f:
                f.write(f"Failed to parse mentioned_agencies as list ({e}), treating as single string: '{mentioned_agencies}'\n")
    
    if not agencies_to_check:
        with open(debug_file, "a") as f:
            f.write("Result: No agencies to compare against\n")
        return False, "No agencies to compare against"

    # Pre-LLM guard: if every agency we'd compare against is a non-LE source
    # (DA, Coroner, Medical Examiner, Public Defender, Attorney General, etc.)
    # and the POST agency is a Police/Sheriff/Marshal/Patrol-style LE org,
    # reject deterministically. The LLM has historically said MATCH here
    # (e.g. "Alameda County DA" -> "Hayward PD"), producing bad auto-matches.
    if _all_non_le(agencies_to_check) and _is_le_agency(post_agency):
        reason = (
            "Source agency is non-LE (DA/Coroner/ME/etc.) with no mentioned "
            "LE agencies; cannot auto-validate against an LE POST agency"
        )
        with open(debug_file, "a") as f:
            f.write(f"Result: {reason}\n")
        return False, reason

    with open(debug_file, "a") as f:
        f.write(f"Total agencies to check: {len(agencies_to_check)}\n")
        f.write(f"Agencies list: {agencies_to_check}\n\n")
    
    # Format the agencies list
    agencies_list = "\n".join(f"- {agency}" for agency in agencies_to_check)
    
    # Create the LLM prompt with California-specific few-shot examples
    prompt = f"""<task>
Determine if the POST agency matches any of the provided agencies. Agencies match ONLY if they refer to the EXACT SAME organization.
</task>

<matching_criteria>
Agencies match when:
- They are the same organization with different abbreviations (e.g., "Sacramento PD" vs "Sacramento Police Department")
- They have minor spelling differences or punctuation (e.g., "Sheriff's Office" vs "Sheriffs Office")
- They have slightly different formatting but same meaning (e.g., "Napa County Sheriff's Office" vs "Napa County Sheriff")

Agencies DO NOT match when:
- They are different police departments, even in nearby cities (e.g., "Corona Police Department" vs "Riverside Police Department")
- They are different sheriff departments from different counties (e.g., "Sacramento County Sheriff" vs "Riverside County Sheriff")
- They are different state prisons or correctional facilities (e.g., "Pelican Bay State Prison" vs "San Quentin State Prison")
- They are different types of agencies within the corrections system (e.g., "Correctional Training Center" vs "State Prison")
- One is a police department and another is a sheriff's department, even in the same area
</matching_criteria>

<examples>
Example 1:
POST agency: "CORONA POLICE DEPARTMENT"
Agencies to compare: ["ANTIOCH POLICE DEPARTMENT", "SAN JOSE POLICE DEPARTMENT", "SACRAMENTO COUNTY SHERIFF'S DEPARTMENT"]
Answer: NO_MATCH (these are completely different law enforcement agencies in different cities)

Example 2:
POST agency: "SACRAMENTO COUNTY SHERIFF'S DEPARTMENT"
Agencies to compare: ["Sacramento County Sheriff", "Sacramento Sheriff's Office"]
Answer: MATCH (same organization, minor formatting differences)

Example 3:
POST agency: "NAPA COUNTY SHERIFF'S OFFICE"
Agencies to compare: ["Napa County Sheriff", "Napa Co Sheriff's Dept"]
Answer: MATCH (same organization, different abbreviations/formatting)

Example 4:
POST agency: "PELICAN BAY STATE PRISON"
Agencies to compare: ["California Department of Corrections and Rehabilitation", "Correctional Training Center", "San Quentin State Prison"]
Answer: NO_MATCH (these are different facilities within the CA corrections system)

Example 5:
POST agency: "ELK GROVE POLICE DEPARTMENT"
Agencies to compare: ["Elk Grove PD", "City of Elk Grove Police"]
Answer: MATCH (same organization, abbreviated)

Example 6:
POST agency: "RIVERSIDE COUNTY SHERIFF'S DEPARTMENT"
Agencies to compare: ["Riverside County Sheriff", "RCSD"]
Answer: MATCH (same organization, different formats)

Example 7:
POST agency: "SAN JOSE POLICE DEPARTMENT"
Agencies to compare: ["San Jose PD", "SJPD"]
Answer: MATCH (same organization, abbreviated)

Example 8:
POST agency: "CALIFORNIA DEPARTMENT OF CORRECTIONS AND REHABILITATION"
Agencies to compare: ["CDCR", "CA Dept of Corrections"]
Answer: MATCH (same organization, abbreviated)

Example 9:
POST agency: "SAN QUENTIN STATE PRISON"
Agencies to compare: ["San Quentin State Prison", "095: SAN QUENTIN STATE PRISON"]
Answer: MATCH (same prison, one includes facility code)

Example 10:
POST agency: "California Department of Corrections and Rehabilitation Office of Internal Affairs"
Agencies to compare: ["Office of Internal Affairs", "Office of Internal Affairs - Northern Region", "California Department of Corrections and Rehabilitation"]
Answer: MATCH (Office of Internal Affairs is a division within CDCR, these refer to the same parent organization)
</examples>

<post_agency>
{post_agency}
</post_agency>

<agencies_to_compare>
{agencies_list}
</agencies_to_compare>

<instructions>
Return ONLY "MATCH" if the POST agency matches any of the agencies to compare (meaning they are the EXACT SAME organization).
Return ONLY "NO_MATCH" if the POST agency does not match any of them (meaning they are different organizations).
Do not include any explanation or other text.
</instructions>"""

    with open(debug_file, "a") as f:
        f.write("PROMPT SENT TO LLM:\n")
        f.write("-" * 80 + "\n")
        f.write(prompt)
        f.write("\n" + "-" * 80 + "\n\n")

    try:
        response = prompt_gpt(
            prompt,
            model="gpt-5.4-nano",
            cached=True,
            logger=logger
        )
        
        with open(debug_file, "a") as f:
            f.write(f"LLM raw response: '{response}'\n")
        
        response_clean = response.strip().upper()
        
        with open(debug_file, "a") as f:
            f.write(f"LLM cleaned response: '{response_clean}'\n")
        
        if "MATCH" in response_clean and "NO_MATCH" not in response_clean:
            with open(debug_file, "a") as f:
                f.write("Result: MATCH found - validation successful\n")
            return True, ""
        else:
            with open(debug_file, "a") as f:
                f.write("Result: NO_MATCH - validation failed\n")
            return False, "Agency cannot be validated"
            
    except Exception as e:
        logger.error(f"Error in LLM-based agency validation: {str(e)}")
        with open(debug_file, "a") as f:
            f.write(f"EXCEPTION: {type(e).__name__}: {str(e)}\n")
        return False, f"Validation error: {str(e)}"