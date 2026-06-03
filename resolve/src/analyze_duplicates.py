import json
from collections import defaultdict
import pandas as pd

# Load original input CSV
print("Loading original input CSV...")
input_df = pd.read_csv("../data/input/involved_officers_2-2-2026.csv")
print(f"Loaded {len(input_df)} rows from input CSV")

# Create lookup dictionary using tileid as key (unique per officer mention)
input_lookup = {}
for _, row in input_df.iterrows():
    key = row.get('tileid', '')
    if key:
        input_lookup[key] = row.to_dict()

print(f"Created lookup for {len(input_lookup)} unique tileids")

# Read auto_matched.jsonl and group by post_person_nbr
post_id_to_mentions = defaultdict(list)

print("\nAnalyzing auto_matched.jsonl...")
with open("../data/output/auto_matched.jsonl", 'r') as f:
    for line in f:
        line = line.strip()
        if not line:  # Skip empty lines
            continue
        record = json.loads(line)
        post_person_nbr = record['post_match']['post_person_nbr']
        post_id_to_mentions[post_person_nbr].append(record)

# Find duplicates (POST IDs with more than 1 mention)
duplicates = {post_id: mentions for post_id, mentions in post_id_to_mentions.items() if len(mentions) > 1}

print(f"\nTotal unique POST IDs: {len(post_id_to_mentions)}")
print(f"POST IDs with duplicates: {len(duplicates)}")
print(f"Total duplicate mentions: {sum(len(mentions) - 1 for mentions in duplicates.values())}")

# Analyze duplicates in detail - structure as JSON
duplicate_analysis_json = []

for post_id, mentions in sorted(duplicates.items(), key=lambda x: len(x[1]), reverse=True):
    # Get POST officer info from first mention (same for all)
    first_mention = mentions[0]
    post_match = first_mention['post_match']

    # Check if duplicates are within same case or across cases
    case_ids = [m['input_officer'].get('authoritative_caseid', '') for m in mentions]
    unique_cases = len(set(case_ids))
    is_same_case = unique_cases == 1

    post_officer = {
        "post_person_nbr": post_id,
        "post_name": f"{post_match['post_first_name']} {post_match.get('post_middle_name', '')} {post_match['post_last_name']}".strip(),
        "post_first_name": post_match['post_first_name'],
        "post_middle_name": post_match.get('post_middle_name', ''),
        "post_last_name": post_match['post_last_name'],
        "post_suffix": post_match.get('post_suffix', ''),
        "post_agency_name": post_match['post_agency_name'],
        "post_agency_type": post_match['post_agency_type'],
        "post_start_date": post_match['post_start_date'],
        "post_end_date": post_match.get('post_end_date', ''),
        "duplicate_count": len(mentions),
        "unique_case_count": unique_cases,
        "is_same_case_duplicate": is_same_case,
        "input_mentions": []
    }

    # Add all input officer mentions
    for mention in mentions:
        input_off = mention['input_officer']

        # Look up original CSV row for additional context using tileid
        tileid = input_off.get('tileid', '')
        original_row = input_lookup.get(tileid, {})

        input_mention = {
            "old_case_name": input_off.get('old_case_name', ''),
            "authoritative_caseid": input_off.get('authoritative_caseid', ''),
            "case_resourceid": input_off.get('case_resourceid', ''),
            "tileid": input_off.get('tileid', ''),
            "input_name": f"{input_off['first_name']} {input_off.get('middle_name', '')} {input_off['last_name']}".strip(),
            "first_name": input_off['first_name'],
            "middle_name": input_off.get('middle_name', ''),
            "last_name": input_off['last_name'],
            "suffix": input_off.get('suffix', ''),
            "source_agency": input_off['source_agency'],
            "incident_date": input_off['incident_date'],
            "officer_name": input_off.get('officer_name', ''),
            "match_probability": mention['post_match']['match_probability'],
            "document_link": input_off.get('document_link', ''),
            # Include any additional fields from original CSV
            "original_csv_data": original_row
        }

        post_officer["input_mentions"].append(input_mention)

    duplicate_analysis_json.append(post_officer)

# Save JSON output
output_json = "../data/output/tests/duplicate_analysis.json"
with open(output_json, 'w') as f:
    json.dump(duplicate_analysis_json, f, indent=2)

print(f"\nSaved detailed analysis to {output_json}")

# Summary statistics
print(f"\nDuplicate count distribution:")
from collections import Counter
duplicate_counts = Counter([officer['duplicate_count'] for officer in duplicate_analysis_json])
for count in sorted(duplicate_counts.keys()):
    print(f"  {count} mentions: {duplicate_counts[count]} POST IDs")

# Same-case vs cross-case analysis
same_case_dupes = [o for o in duplicate_analysis_json if o['is_same_case_duplicate']]
cross_case_dupes = [o for o in duplicate_analysis_json if not o['is_same_case_duplicate']]

print(f"\n{'='*80}")
print(f"SAME-CASE vs CROSS-CASE ANALYSIS")
print(f"{'='*80}")
print(f"Same-case duplicates (same officer in SAME case): {len(same_case_dupes)}")
print(f"Cross-case duplicates (same officer in DIFFERENT cases): {len(cross_case_dupes)}")
print(f"\n⚠️  Same-case duplicates are likely bugs - same person shouldn't match multiple times in one case")
print(f"✓  Cross-case duplicates are expected - same officer appearing in multiple incidents")

if same_case_dupes:
    print(f"\n{'='*80}")
    print(f"SAME-CASE DUPLICATES (POTENTIAL BUGS):")
    print(f"{'='*80}")
    for officer in same_case_dupes[:20]:  # Show first 20
        case_id = officer['input_mentions'][0]['authoritative_caseid']
        print(f"  {officer['post_person_nbr']}: {officer['post_name']} - {officer['duplicate_count']} mentions in case {case_id}")

# Show top 10 most duplicated POST IDs
print(f"\n{'='*80}")
print(f"Top 10 most duplicated POST IDs (all types):")
print(f"{'='*80}")
top_10 = duplicate_analysis_json[:10]  # Already sorted by duplicate count descending
for officer in top_10:
    dupe_type = "SAME-CASE" if officer['is_same_case_duplicate'] else "CROSS-CASE"
    print(f"  {officer['post_person_nbr']}: {officer['duplicate_count']} mentions ({officer['unique_case_count']} cases) - {officer['post_name']} [{dupe_type}]")
