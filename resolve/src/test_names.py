import json
import pandas as pd
from collections import defaultdict
from difflib import SequenceMatcher
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor
from llm import prompt_gpt

FIRST_NAME_THRESHOLD = 0.7
OUTPUT_FILE = "../data/output/tests/similar_names_in_post.txt"
NEEDS_REVIEW_CSV = "../data/output/tests/names_need_review.csv"
NO_REVIEW_CSV = "../data/output/tests/names_no_review_needed.csv"

# Load POST database
post_df = pd.read_csv("../data/output/post_export.csv")

# Get unique persons (dedupe by person_nbr)
post_persons = post_df.drop_duplicates(subset=["person_nbr"])[["person_nbr", "first_name", "last_name"]].copy()
post_persons["first_name"] = post_persons["first_name"].str.strip().str.lower()
post_persons["last_name"] = post_persons["last_name"].str.strip().str.lower()

print(f"Unique POST persons: {len(post_persons)}")

# Load auto_matched
matched = []
with open("../data/output/auto_matched.jsonl", 'r') as f:
    for line_num, line in enumerate(f, 1):
        row = json.loads(line.strip())
        matched.append({
            "line": line_num,
            "post_id": row["post_match"]["post_person_nbr"],
            "input_first": row["input_officer"]["first_name"].strip().lower(),
            "input_last": row["input_officer"]["last_name"].strip().lower(),
            "row": row
        })

print(f"Auto-matched officers: {len(matched)}")

# Group POST persons by last name
by_last_name = defaultdict(list)
for _, row in post_persons.iterrows():
    by_last_name[row["last_name"]].append((row["person_nbr"], row["first_name"]))

# For each matched officer, check if there are similar names with different POST IDs
flagged = []

for officer in matched:
    post_id = officer["post_id"]
    input_first = officer["input_first"]
    input_last = officer["input_last"]
    matched_first = officer["row"]["post_match"]["post_first_name"].strip().lower()

    # Similarity of the matched POST person's first name to input. If this is
    # an exact match, the auto-matcher had no name ambiguity to resolve, and
    # the mere existence of other "similar" people doesn't make the match wrong.
    matched_similarity = SequenceMatcher(None, input_first, matched_first).ratio()
    if matched_first == input_first:
        continue

    # Get all POST persons with same last name
    same_last = by_last_name.get(input_last, [])

    # Only flag when another POST person is at least as close to the input as
    # the matched POST person — i.e., there's genuine ambiguity.
    similar_persons = []
    for other_id, other_first in same_last:
        if other_id == post_id:
            continue

        ratio = SequenceMatcher(None, input_first, other_first).ratio()
        if ratio >= FIRST_NAME_THRESHOLD and ratio >= matched_similarity:
            similar_persons.append({
                "post_id": other_id,
                "first_name": other_first,
                "similarity": ratio
            })

    if similar_persons:
        flagged.append({
            "line": officer["line"],
            "matched_post_id": post_id,
            "input_first": input_first,
            "input_last": input_last,
            "similar_persons": similar_persons,
            "row": officer["row"]
        })

print(f"Officers with similar names in POST (different POST ID): {len(flagged)}")

# ============================================================================
# LLM REVIEW OF SIMILAR NAMES
# ============================================================================
print(f"\nRunning LLM review on {len(flagged)} flagged officers...")

def review_similarity_with_llm(item):
    """Use LLM to determine if similar names are confusable"""
    input_first = item['input_first']
    input_last = item['input_last']
    matched_post_id = item['matched_post_id']
    similar_persons = item['similar_persons']

    # Build list of similar persons for prompt
    similar_list = "\n".join([
        f"  • {sim['first_name'].upper()} {input_last.upper()} (POST ID: {sim['post_id']}, string similarity: {sim['similarity']:.2f})"
        for sim in similar_persons
    ])

    prompt = f"""Review this officer name matching scenario:

INPUT OFFICER: {input_first.upper()} {input_last.upper()}
AUTO-MATCHED TO: POST ID {matched_post_id}

OTHER OFFICERS IN DATABASE WITH SIMILAR NAMES:
{similar_list}

Task: Determine if these other officers have names similar enough to the input officer that a human should manually verify the match.

Consider:
- Are these likely to be the same person (nicknames, data entry variations, middle names used as first names)?
- Could the auto-match have selected the wrong person?
- Are the names confusable or clearly distinct?

Respond with ONE of:
NEEDS_REVIEW - Names are confusable, human should verify
NOT_SIMILAR - Names are clearly different, no review needed

Format your response as:
<DECISION>
<One sentence explanation>

Example:
NEEDS_REVIEW
The names "Bob" and "Robert" are common variations that could refer to the same person."""

    response = prompt_gpt(prompt, model='gpt-5.4-nano', cached=True)

    # Parse response
    lines = response.strip().split('\n', 1)
    decision = lines[0].strip()
    explanation = lines[1].strip() if len(lines) > 1 else ""

    return decision, explanation

# Run LLM review for each flagged officer in parallel
print(f"Running {len(flagged)} LLM reviews in parallel...")
with ThreadPoolExecutor(max_workers=100) as executor:
    llm_results = list(executor.map(review_similarity_with_llm, flagged))

# Attach results to flagged items
for item, (decision, explanation) in zip(flagged, llm_results):
    item['llm_decision'] = decision
    item['llm_explanation'] = explanation

# Split into two groups based on LLM decision
needs_review = [item for item in flagged if 'NEEDS_REVIEW' in item['llm_decision']]
no_review_needed = [item for item in flagged if 'NOT_SIMILAR' in item['llm_decision']]

print(f"LLM Review Results:")
print(f"  - NEEDS_REVIEW: {len(needs_review)}")
print(f"  - NOT_SIMILAR: {len(no_review_needed)}")

# ============================================================================
# CREATE CSV OUTPUTS
# ============================================================================

def create_csv_row(item):
    """Convert flagged item to CSV row"""
    row = item['row']
    input_officer = row['input_officer']
    post_match = row['post_match']

    similar_names_str = "; ".join([
        f"{sim['first_name'].upper()} {item['input_last'].upper()} (POST ID: {sim['post_id']}, sim: {sim['similarity']:.2f})"
        for sim in item['similar_persons']
    ])

    return {
        'line_number': item['line'],
        'input_first_name': input_officer['first_name'],
        'input_middle_name': input_officer.get('middle_name', ''),
        'input_last_name': input_officer['last_name'],
        'source_agency': input_officer['source_agency'],
        'incident_date': input_officer['incident_date'],
        'matched_post_id': post_match['post_person_nbr'],
        'matched_post_name': f"{post_match['post_first_name']} {post_match.get('post_middle_name', '')} {post_match['post_last_name']}".strip(),
        'matched_agency': post_match['post_agency_name'],
        'match_probability': post_match['match_probability'],
        'similar_persons': similar_names_str,
        'llm_decision': item['llm_decision'],
        'llm_explanation': item['llm_explanation'],
        'document_link': input_officer.get('document_link', '')
    }

# Create DataFrames and save CSVs
if needs_review:
    needs_review_df = pd.DataFrame([create_csv_row(item) for item in needs_review])
    needs_review_df.to_csv(NEEDS_REVIEW_CSV, index=False)
    print(f"\n✓ Wrote {len(needs_review)} officers needing review to {NEEDS_REVIEW_CSV}")
else:
    # Create empty file with headers
    pd.DataFrame(columns=['line_number', 'input_first_name', 'input_middle_name', 'input_last_name',
                          'source_agency', 'incident_date', 'matched_post_id', 'matched_post_name',
                          'matched_agency', 'match_probability', 'similar_persons', 'llm_decision',
                          'llm_explanation', 'document_link']).to_csv(NEEDS_REVIEW_CSV, index=False)
    print(f"\n✓ No officers need review - created empty {NEEDS_REVIEW_CSV}")

if no_review_needed:
    no_review_df = pd.DataFrame([create_csv_row(item) for item in no_review_needed])
    no_review_df.to_csv(NO_REVIEW_CSV, index=False)
    print(f"✓ Wrote {len(no_review_needed)} officers passing review to {NO_REVIEW_CSV}")
else:
    # Create empty file with headers
    pd.DataFrame(columns=['line_number', 'input_first_name', 'input_middle_name', 'input_last_name',
                          'source_agency', 'incident_date', 'matched_post_id', 'matched_post_name',
                          'matched_agency', 'match_probability', 'similar_persons', 'llm_decision',
                          'llm_explanation', 'document_link']).to_csv(NO_REVIEW_CSV, index=False)
    print(f"✓ No officers passed review - created empty {NO_REVIEW_CSV}")

# ============================================================================
# Write to file
with open(OUTPUT_FILE, 'w') as f:
    f.write("SIMILAR NAMES CHECK\n")
    f.write("="*80 + "\n")
    f.write(f"Generated: {datetime.now().isoformat()}\n")
    f.write(f"First name similarity threshold: {FIRST_NAME_THRESHOLD}\n")
    f.write(f"\nTotal auto-matched officers: {len(matched)}\n")
    f.write(f"Officers with similar names in POST: {len(flagged)}\n")
    f.write(f"Percentage flagged: {len(flagged)/len(matched)*100:.1f}%\n")
    f.write("="*80 + "\n\n")
    
    f.write("These officers were auto-matched, but there are other people in POST\n")
    f.write("with the same last name and similar first name (different POST ID).\n")
    f.write("These should have been flagged for manual review.\n")
    f.write("="*80 + "\n\n")
    
    for item in flagged:
        row = item["row"]
        input_officer = row["input_officer"]
        post_match = row["post_match"]
        
        f.write("-"*80 + "\n")
        f.write(f"Line {item['line']}: {item['input_first'].upper()} {item['input_last'].upper()}\n")
        f.write("-"*80 + "\n\n")
        
        f.write("INPUT OFFICER:\n")
        f.write(f"  Name: {input_officer['first_name']} {input_officer.get('middle_name', '')} {input_officer['last_name']}\n")
        f.write(f"  Source Agency: {input_officer['source_agency']}\n")
        f.write(f"  Incident Date: {input_officer['incident_date']}\n")
        f.write(f"  Document: {input_officer.get('document_link', '')}\n\n")
        
        f.write("MATCHED TO:\n")
        f.write(f"  POST ID: {post_match['post_person_nbr']}\n")
        f.write(f"  Name: {post_match['post_first_name']} {post_match.get('post_middle_name', '')} {post_match['post_last_name']}\n")
        f.write(f"  Agency: {post_match['post_agency_name']}\n")
        f.write(f"  Employment: {post_match['post_start_date']} to {post_match.get('post_end_date') or 'present'}\n")
        f.write(f"  Match Probability: {post_match['match_probability']:.4f}\n\n")
        
        f.write("SIMILAR PERSONS IN POST (potential conflicts):\n")
        for sim in item["similar_persons"]:
            f.write(f"  - {sim['first_name'].upper()} {item['input_last'].upper()} (POST ID: {sim['post_id']}) - similarity: {sim['similarity']:.2f}\n")
        
        f.write("\n\n")

print(f"✓ Wrote results to {OUTPUT_FILE}")