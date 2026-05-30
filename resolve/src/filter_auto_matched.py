import json

# Load POST IDs to exclude
with open("../data/output/tests/post_ids_need_review.txt", 'r') as f:
    exclude_post_ids = set(line.strip() for line in f if line.strip())

print(f"Loaded {len(exclude_post_ids)} POST IDs to exclude")

# Read auto_matched.jsonl and filter
filtered_records = []
seen_post_ids = set()
total_count = 0
excluded_count = 0
duplicate_count = 0

with open("../data/output/auto_matched.jsonl", 'r') as f:
    for line in f:
        total_count += 1
        record = json.loads(line.strip())

        post_person_nbr = record['post_match']['post_person_nbr']

        # Skip if in exclusion list
        if post_person_nbr in exclude_post_ids:
            excluded_count += 1
            continue

        # # Skip if already seen (dedupe)
        # if post_person_nbr in seen_post_ids:
        #     duplicate_count += 1
        #     continue

        # Keep this record
        # seen_post_ids.add(post_person_nbr)
        filtered_records.append(record)

# Write filtered records
output_file = "../data/output/auto_matched_deduped_filtered.jsonl"
with open(output_file, 'w') as f:
    for record in filtered_records:
        f.write(json.dumps(record) + '\n')

print(f"\nProcessing complete:")
print(f"  Total records: {total_count}")
print(f"  Excluded (in review list): {excluded_count}")
print(f"  Excluded (duplicates): {duplicate_count}")
print(f"  Kept: {len(filtered_records)}")
print(f"\nSaved to {output_file}")
