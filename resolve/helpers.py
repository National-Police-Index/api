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

logger = logging.getLogger('idonea.' + __name__)


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
            model="gpt-4.1-mini",
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