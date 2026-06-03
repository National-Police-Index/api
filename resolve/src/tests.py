import json
import os
from datetime import datetime
from difflib import SequenceMatcher


OUTPUT_DIR = "../data/output/tests"


def parse_date(date_str: str) -> datetime | None:
    """Parse various date formats, return None if invalid/empty."""
    if not date_str or date_str.strip() == "":
        return None
    
    # Handle multiple comma-separated dates - take the first one
    if "," in date_str:
        date_str = date_str.split(",")[0].strip()
    
    for fmt in ["%Y-%m-%d", "%Y-%m-%d %H:%M:%S"]:
        try:
            return datetime.strptime(date_str.strip(), fmt)
        except ValueError:
            continue
    return None


def parse_all_dates(date_str: str) -> list[datetime]:
    """Parse all dates from a potentially comma-separated string."""
    if not date_str or date_str.strip() == "":
        return []
    
    dates = []
    for part in date_str.split(","):
        part = part.strip()
        for fmt in ["%Y-%m-%d", "%Y-%m-%d %H:%M:%S"]:
            try:
                dates.append(datetime.strptime(part, fmt))
                break
            except ValueError:
                continue
    
    return dates


def test_exact_name_match(row: dict) -> tuple[bool, str]:
    """Test that first and last names match exactly (case-insensitive)."""
    input_first = row["input_officer"]["first_name"].strip().upper()
    input_last = row["input_officer"]["last_name"].strip().upper()
    post_first = row["post_match"]["post_first_name"].strip().upper()
    post_last = row["post_match"]["post_last_name"].strip().upper()
    
    if input_first != post_first:
        return False, f"First name mismatch: '{input_first}' vs '{post_first}'"
    if input_last != post_last:
        return False, f"Last name mismatch: '{input_last}' vs '{post_last}'"
    
    return True, "Names match exactly"


def test_incident_date_within_employment(row: dict) -> tuple[bool, str]:
    """Test that at least one incident date falls within employment window (with ±1 year buffer)."""
    incident_dates = parse_all_dates(row["input_officer"]["incident_date"])
    if not incident_dates:
        return False, f"Could not parse any incident dates: {row['input_officer']['incident_date']}"
    
    start_date = parse_date(row["post_match"]["post_start_date"])
    if not start_date:
        return False, f"Could not parse start date: {row['post_match']['post_start_date']}"
    
    end_date = parse_date(row["post_match"]["post_end_date"])
    # Pre-1950 end dates are sentinel values in the source data; the pipeline
    # treats them as null and fills with today. Mirror that here.
    if end_date is None or end_date.year < 1950:
        end_year = datetime.now().year
    else:
        end_year = end_date.year
    
    start_year = start_date.year
    
    # Check if ANY incident date falls within the employment window
    for incident_date in incident_dates:
        incident_year = incident_date.year
        if start_year <= incident_year + 1 and end_year >= incident_year - 1:
            return True, f"Incident year {incident_year} within employment {start_year}-{end_year}"
    
    # None matched - report all incident years
    incident_years = [d.year for d in incident_dates]
    return False, f"No incident years {incident_years} within employment {start_year}-{end_year}"


_AGENCY_EQUIVALENTS = [
    ("DEPARTMENT OF PUBLIC SAFETY", "POLICE DEPARTMENT"),
    ("PUBLIC SAFETY", "POLICE"),
    ("SHERIFF-CORONER", "SHERIFF"),
    ("SHERIFFS", "SHERIFF"),
    ("DEPT", "DEPARTMENT"),
    ("CSU", "CALIFORNIA STATE UNIVERSITY"),
    (" REG ", " REGIONAL "),
    (" SER ", " SERVICES "),
    ("PUBLICLIC", "PUBLIC"),
]
_AGENCY_DROP_TOKENS = {
    "OFFICE", "DEPARTMENT", "CORONER", "S", "THE", "A", "AN", "OF",
}


def _normalize_agency(name: str) -> set[str]:
    """Normalize an agency name to a token set for comparison."""
    s = name.upper()
    s = s.replace("’", "'").replace("`", "'")
    for src, dst in _AGENCY_EQUIVALENTS:
        s = s.replace(src, dst)
    # Strip punctuation
    for ch in "'./,()-":
        s = s.replace(ch, " ")
    s = s.replace("/", " ")
    tokens = {t for t in s.split() if t and t not in _AGENCY_DROP_TOKENS}
    return tokens


def _agency_token_similarity(a: str, b: str) -> float:
    ta, tb = _normalize_agency(a), _normalize_agency(b)
    if not ta or not tb:
        return 0.0
    # Jaccard over the smaller set — captures "Alameda County Sheriff" ⊂
    # "Alameda County Sheriff's Department/Coroner" as a full match.
    intersection = ta & tb
    smaller = min(len(ta), len(tb))
    return len(intersection) / smaller


def test_agency_name_similarity(row: dict) -> tuple[bool, str]:
    """Test that agency names are sufficiently similar after normalization.

    Normalizes equivalents (Office=Department, drops /Coroner suffix,
    Public Safety=Police, etc.) before comparison so that real spelling
    variants don't fail the test.
    """
    threshold = 0.8

    source_agencies_raw = row["input_officer"]["source_agency"].strip().upper()
    mentioned_agencies_raw = row["input_officer"].get("mentioned_agencies", "").strip().upper()
    post_agency = row["post_match"]["post_agency_name"].strip().upper()

    all_input_agencies = []
    for agency in source_agencies_raw.split(","):
        agency = agency.strip()
        if agency:
            all_input_agencies.append(agency)
    for agency in mentioned_agencies_raw.split(","):
        agency = agency.strip()
        if agency:
            all_input_agencies.append(agency)

    best_ratio = 0.0
    best_match = ""

    for agency in all_input_agencies:
        # Token similarity catches structural matches; raw SequenceMatcher
        # is the fallback for genuine string variants.
        token_ratio = _agency_token_similarity(agency, post_agency)
        char_ratio = SequenceMatcher(None, agency, post_agency).ratio()
        ratio = max(token_ratio, char_ratio)
        if ratio > best_ratio:
            best_ratio = ratio
            best_match = agency
        if ratio >= threshold:
            return True, f"Agency match: '{agency}' ~ '{post_agency}' (similarity {ratio:.2f})"

    return False, f"No agency match >= {threshold}: best was '{best_match}' vs '{post_agency}' ({best_ratio:.2f})"


def test_probability_threshold(row: dict) -> tuple[bool, str]:
    """Test that match probability exceeds threshold."""
    prob = row["post_match"]["match_probability"]
    threshold = 0.5
    
    if prob > threshold:
        return True, f"Probability {prob:.4f} > {threshold}"
    
    return False, f"Probability {prob:.4f} <= {threshold}"


def test_state_is_california(row: dict) -> tuple[bool, str]:
    """Test that the match is in California."""
    state = row["post_match"].get("state", "").strip().upper()
    
    if state in ["CA", "CALIFORNIA"]:
        return True, "State is California"
    
    return False, f"Unexpected state: '{state}'"


def test_valid_post_person_nbr(row: dict) -> tuple[bool, str]:
    """Test that POST person number is present and valid."""
    person_nbr = row["post_match"]["post_person_nbr"]
    
    if not person_nbr or person_nbr.strip() == "":
        return False, "Empty POST person number"
    
    if not any(c.isalnum() for c in person_nbr):
        return False, f"Invalid POST person number format: '{person_nbr}'"
    
    return True, f"Valid POST person number: {person_nbr}"


def test_no_empty_critical_fields(row: dict) -> tuple[bool, str]:
    """Test that critical fields are not empty."""
    critical_input = ["first_name", "last_name"]
    critical_post = ["post_first_name", "post_last_name", "post_agency_name", "post_start_date"]

    errors = []

    for field in critical_input:
        val = row["input_officer"].get(field, "")
        if not val or str(val).strip() == "":
            errors.append(f"input_officer.{field} is empty")

    # source_agency may be empty if mentioned_agencies covers it.
    src = str(row["input_officer"].get("source_agency", "")).strip()
    mentioned = str(row["input_officer"].get("mentioned_agencies", "")).strip()
    if not src and not mentioned:
        errors.append("input_officer has neither source_agency nor mentioned_agencies")
    
    for field in critical_post:
        val = row["post_match"].get(field, "")
        if not val or str(val).strip() == "":
            errors.append(f"post_match.{field} is empty")
    
    if errors:
        return False, "; ".join(errors)
    
    return True, "All critical fields populated"


def run_all_tests(row: dict) -> list[tuple[str, bool, str]]:
    """Run all validation tests on a single row."""
    tests = [
        ("exact_name_match", test_exact_name_match),
        ("incident_date_within_employment", test_incident_date_within_employment),
        ("agency_name_similarity", test_agency_name_similarity),
        ("probability_threshold", test_probability_threshold),
        ("state_is_california", test_state_is_california),
        ("valid_post_person_nbr", test_valid_post_person_nbr),
        ("no_empty_critical_fields", test_no_empty_critical_fields),
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            passed, message = test_func(row)
            results.append((test_name, passed, message))
        except Exception as e:
            results.append((test_name, False, f"Test error: {str(e)}"))
    
    return results


def format_row_details(row: dict, line_num: int, message: str) -> str:
    """Format full details of a failed row for the error file."""
    input_officer = row["input_officer"]
    post_match = row["post_match"]
    
    return f"""
{'='*80}
Line {line_num}: {input_officer['first_name']} {input_officer['last_name']}
{'='*80}
Error: {message}

INPUT OFFICER:
  Name: {input_officer['first_name']} {input_officer.get('middle_name', '')} {input_officer['last_name']} {input_officer.get('suffix', '')}
  Source Agency: {input_officer['source_agency']}
  Mentioned Agencies: {input_officer.get('mentioned_agencies', '')}
  Incident Date: {input_officer['incident_date']}
  Case: {input_officer.get('old_case_name', '')}
  Document Link: {input_officer.get('document_link', '')}

POST MATCH:
  Name: {post_match['post_first_name']} {post_match.get('post_middle_name', '')} {post_match['post_last_name']} {post_match.get('post_suffix', '')}
  POST Person #: {post_match['post_person_nbr']}
  Agency: {post_match['post_agency_name']}
  Agency Type: {post_match.get('post_agency_type', '')}
  Employment: {post_match['post_start_date']} to {post_match.get('post_end_date', 'present')}
  County: {post_match.get('county', '')}
  State: {post_match.get('state', '')}
  Match Probability: {post_match['match_probability']:.4f}
  Officer Link: {post_match.get('officer_link', '')}

RAW JSON:
{json.dumps(row, indent=2, default=str)}
"""


if __name__ == "__main__":
    filepath = "../data/output/auto_matched.jsonl"
    
    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    total_rows = 0
    failed_rows = 0
    test_failures = {}
    
    with open(filepath, 'r') as f:
        for line_num, line in enumerate(f, 1):
            row = json.loads(line.strip())
            total_rows += 1
            
            results = run_all_tests(row)
            row_failed = False
            
            for test_name, passed, message in results:
                if not passed:
                    row_failed = True
                    if test_name not in test_failures:
                        test_failures[test_name] = []
                    test_failures[test_name].append({
                        "line": line_num,
                        "message": message,
                        "officer": f"{row['input_officer']['first_name']} {row['input_officer']['last_name']}",
                        "row": row
                    })
            
            if row_failed:
                failed_rows += 1
    
    # Write each test's failures to a separate file
    for test_name, failures in test_failures.items():
        output_file = os.path.join(OUTPUT_DIR, f"{test_name}_failures.txt")
        
        with open(output_file, 'w') as f:
            f.write(f"TEST: {test_name}\n")
            f.write(f"Total failures: {len(failures)}\n")
            f.write(f"Generated: {datetime.now().isoformat()}\n")
            f.write("="*80 + "\n\n")
            
            for fail in failures:
                f.write(format_row_details(fail["row"], fail["line"], fail["message"]))
                f.write("\n")
        
        print(f"✓ Wrote {len(failures)} failures to {output_file}")
    
    # Write summary file
    summary_file = os.path.join(OUTPUT_DIR, "summary.txt")
    with open(summary_file, 'w') as f:
        f.write(f"VALIDATION SUMMARY\n")
        f.write(f"Generated: {datetime.now().isoformat()}\n")
        f.write(f"Input file: {filepath}\n")
        f.write("="*60 + "\n\n")
        f.write(f"Total rows: {total_rows}\n")
        f.write(f"Failed rows: {failed_rows}\n")
        f.write(f"Pass rate: {(total_rows - failed_rows) / total_rows * 100:.1f}%\n\n")
        
        f.write("FAILURES BY TEST:\n")
        f.write("-"*40 + "\n")
        
        all_tests = [
            "exact_name_match",
            "incident_date_within_employment", 
            "agency_name_similarity",
            "probability_threshold",
            "state_is_california",
            "valid_post_person_nbr",
            "no_empty_critical_fields",
        ]
        
        for test_name in all_tests:
            count = len(test_failures.get(test_name, []))
            status = "✓" if count == 0 else "✗"
            f.write(f"{status} {test_name}: {count} failures\n")
    
    print(f"✓ Wrote summary to {summary_file}")
    
    # Print summary to console
    print(f"\n{'='*60}")
    print(f"VALIDATION SUMMARY")
    print(f"{'='*60}")
    print(f"Total rows: {total_rows}")
    print(f"Failed rows: {failed_rows}")
    print(f"Pass rate: {(total_rows - failed_rows) / total_rows * 100:.1f}%")
    print(f"\nOutput files written to: {OUTPUT_DIR}")
    
    if test_failures:
        print(f"\nFAILURES BY TEST:")
        for test_name, failures in test_failures.items():
            print(f"  ✗ {test_name}: {len(failures)} failures")
    else:
        print("\n✓ All tests passed!")