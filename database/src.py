import sys
import os
from typing import List, Dict, Any, Optional
from supabase import create_client, Client

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.src import PostEmploymentRecord, CandidateQuery


class SupabaseClient:
    """Wrapper for Supabase database operations focused on POST employment data"""

    def __init__(self, url: str, key: str):
        self.supabase: Client = create_client(url, key)

    def _apply_filters(self, query, candidate_query: CandidateQuery):
        """Apply filters to a Supabase query based on CandidateQuery parameters"""
        if candidate_query.first_name:
            query = query.ilike("first_name", f"%{candidate_query.first_name}%")
        if candidate_query.last_name:
            query = query.ilike("last_name", f"%{candidate_query.last_name}%")
        if candidate_query.agency:
            query = query.ilike("agency_name", f"%{candidate_query.agency}%")
        if candidate_query.state:
            query = query.ilike("state", f"%{candidate_query.state}%")
        return query

    def _transform_record(self, record: Dict[str, Any]) -> PostEmploymentRecord:
        start_date = record.get("start_date")
        if start_date == "":
            start_date = None

        end_date = record.get("end_date")
        if end_date == "":
            end_date = None

        return PostEmploymentRecord(
            post_person_nbr=record.get("person_nbr", ""),
            post_first_name=record.get("first_name", ""),
            post_middle_name=record.get("middle_name", ""),
            post_last_name=record.get("last_name", ""),
            post_suffix="",
            post_agency_name=record.get("agency_name", ""),
            post_agency_type=record.get("agency_type", "POLICE"),
            post_start_date=start_date,
            post_end_date=end_date,
            post_separation_reason=record.get("separation_reason", ""),
            state=record.get("state", ""),
            county=record.get("county", ""),
        )

    def get_post_employment_records(
        self, query_params: CandidateQuery
    ) -> List[PostEmploymentRecord]:
        """
        Get POST employment records with optional filtering.

        Args:
            query_params: CandidateQuery with filtering parameters

        Returns:
            List of PostEmploymentRecord models
        """
        # Start with base query selecting all columns from post table
        query = self.supabase.table("postie").select("*")

        # Apply filters
        query = self._apply_filters(query, query_params)

        # Apply pagination
        if query_params.offset > 0:
            query = query.range(
                query_params.offset,
                query_params.offset + (query_params.limit or 1000) - 1,
            )
        elif query_params.limit:
            query = query.limit(query_params.limit)

        response = query.execute()

        if not response.data:
            return []

        # Transform to PostEmploymentRecord models
        return [self._transform_record(record) for record in response.data]

    def get_post_employment_count(
        self, query_params: Optional[CandidateQuery] = None
    ) -> int:
        """Get total count of POST employment records with optional filters"""
        query = self.supabase.table("postie").select("person_nbr", count="exact")

        if query_params:
            query = self._apply_filters(query, query_params)

        response = query.execute()
        return response.count or 0
    
    def get_county_for_agency(self, agency_name: str) -> Optional[str]:
        """
        Get the county for a given agency name.
        For county sheriff agencies, matches both Office and Department variants.
        Returns the first matching county found, or None if agency not found.
        """
        if not agency_name:
            return None

        # Write debug output to file
        import os
        debug_file = os.path.join(os.path.dirname(__file__), "..", "resolve", "data", "output", "sequential_county_debug.log")
        os.makedirs(os.path.dirname(debug_file), exist_ok=True)

        with open(debug_file, 'a') as f:
            f.write(f"\n{'='*60}\n")
            f.write(f"SEQUENTIAL COUNTY LOOKUP: '{agency_name}'\n")
            f.write(f"{'='*60}\n")

        # First try exact match
        response = (
            self.supabase.table("postie")
            .select("county,agency_name")
            .ilike("agency_name", f"%{agency_name}%")
            .limit(10)  # Get more records for debugging
            .execute()
        )

        with open(debug_file, 'a') as f:
            f.write(f"Query: agency_name ILIKE '%{agency_name}%'\n")
            f.write(f"Results: {len(response.data or [])} records\n")
            if response.data:
                for i, record in enumerate(response.data[:5]):  # Show first 5
                    f.write(f"  {i+1}. '{record.get('agency_name')}' -> {record.get('county')}\n")
                if len(response.data) > 5:
                    f.write(f"  ... and {len(response.data) - 5} more\n")

        if response.data and len(response.data) > 0:
            county = response.data[0].get("county")
            with open(debug_file, 'a') as f:
                f.write(f"✓ FOUND: {county}\n")
            return county

        # If no match and it's a county sheriff agency, try swapping office/department
        normalized = agency_name.lower()
        if "county sheriff" in normalized:
            if "office" in normalized:
                alternate_name = agency_name.replace("Office", "Department").replace("office", "department")
            elif "department" in normalized:
                alternate_name = agency_name.replace("Department", "Office").replace("department", "office")
            else:
                with open(debug_file, 'a') as f:
                    f.write(f"✗ NOT FOUND (no alternate name to try)\n")
                return None

            with open(debug_file, 'a') as f:
                f.write(f"Trying alternate: agency_name ILIKE '%{alternate_name}%'\n")

            response = (
                self.supabase.table("postie")
                .select("county,agency_name")
                .ilike("agency_name", f"%{alternate_name}%")
                .limit(10)
                .execute()
            )

            with open(debug_file, 'a') as f:
                f.write(f"Results: {len(response.data or [])} records\n")
                if response.data:
                    for i, record in enumerate(response.data[:5]):
                        f.write(f"  {i+1}. '{record.get('agency_name')}' -> {record.get('county')}\n")

            if response.data and len(response.data) > 0:
                county = response.data[0].get("county")
                with open(debug_file, 'a') as f:
                    f.write(f"✓ FOUND (alternate): {county}\n")
                return county

        with open(debug_file, 'a') as f:
            f.write(f"✗ NOT FOUND\n")

        return None

    def get_post_stats(self) -> Dict[str, Any]:
        """Get POST employment statistics"""

        # Total employment records
        total_response = (
            self.supabase.table("postie").select("person_nbr", count="exact").execute()
        )
        total_records = total_response.count or 0

        # Unique officers (distinct post_uid)
        officers_response = self.supabase.table("postie").select("person_nbr").execute()
        unique_officers = len(
            set(
                record.get("person_nbr")
                for record in officers_response.data
                if record.get("person_nbr")
            )
        )

        # Unique agencies
        agencies_response = self.supabase.table("postie").select("agency_name").execute()
        unique_agencies = len(
            set(
                record.get("agency_name")
                for record in agencies_response.data
                if record.get("agency_name")
            )
        )

        # Unique states
        states_response = self.supabase.table("postie").select("state").execute()
        unique_states = len(
            set(
                record.get("state")
                for record in states_response.data
                if record.get("state")
            )
        )

        # Active officers (no end date or end date is None/null)
        active_response = (
            self.supabase.table("postie")
            .select("person_nbr", count="exact")
            .is_("end_date", "null")
            .execute()
        )
        active_officers = active_response.count or 0

        # Separated officers (has end date)
        separated_response = (
            self.supabase.table("postie")
            .select("person_nbr", count="exact")
            .not_.is_("end_date", "null")
            .execute()
        )
        separated_officers = separated_response.count or 0

        return {
            "total_employment_records": total_records,
            "unique_officers": unique_officers,
            "unique_agencies": unique_agencies,
            "unique_states": unique_states,
            "active_officers": active_officers,
            "separated_officers": separated_officers,
        }

    def search_post_employment_by_name(
        self, name: str, state: Optional[str] = None
    ) -> List[PostEmploymentRecord]:
        """
        Search POST employment records by officer name.
        This is a convenience method that handles name parsing.
        """
        name_parts = name.strip().split()

        if len(name_parts) == 1:
            # Single name - search both first and last name
            first_query = CandidateQuery(first_name=name_parts[0], state=state)
            last_query = CandidateQuery(last_name=name_parts[0], state=state)
            return self.get_post_employment_records(
                first_query
            ) + self.get_post_employment_records(last_query)
        else:
            # Multiple names - assume first and last
            first_name = name_parts[0]
            last_name = " ".join(name_parts[1:])
            query = CandidateQuery(
                first_name=first_name, last_name=last_name, state=state
            )
            return self.get_post_employment_records(query)

    def get_employment_record_by_person_nbr(
        self, person_nbr: str
    ) -> List[PostEmploymentRecord]:
        """Get all employment records for a specific person number"""
        response = (
            self.supabase.table("postie").select("*").eq("person_nbr", person_nbr).execute()
        )

        return [self._transform_record(record) for record in response.data]

    def get_candidates_for_mention(
        self, query_params: CandidateQuery
    ) -> List[PostEmploymentRecord]:
        """Get candidates matching specific entity resolution criteria"""

        # Base query filters
        base_filters = []

        # Add state filter if provided
        if query_params.state:
            base_filters.append(("state", "ilike", f"%{query_params.state}%"))

        if query_params.agency_type:
            base_filters.append(("agency_type", "eq", query_params.agency_type.value))

        # Query 1: first name prefix + exact last name
        query1 = self.supabase.table("postie").select("*")
        if query_params.first_name and len(query_params.first_name) >= 2:
            query1 = query1.ilike("first_name", f"{query_params.first_name[:2]}%")
        if query_params.last_name:
            query1 = query1.eq("last_name", query_params.last_name)

        # Apply base filters to query1
        for field, operator, value in base_filters:
            query1 = query1.ilike(field, value)

        # Query 2: exact first name + last name prefix
        query2 = self.supabase.table("postie").select("*")
        if query_params.first_name:
            query2 = query2.eq("first_name", query_params.first_name)
        if query_params.last_name and len(query_params.last_name) >= 2:
            query2 = query2.ilike("last_name", f"{query_params.last_name[:2]}%")

        # Apply base filters to query2
        for field, operator, value in base_filters:
            query2 = query2.ilike(field, value)

        # Execute both queries and combine
        results1 = query1.execute().data or []
        results2 = query2.execute().data or []

        # Combine and deduplicate by unique employment record
        all_results = results1 + results2
        seen = set()
        unique_results = []

        for record in all_results:
            key = (
                record.get("person_nbr"),
                record.get("start_date"),
                record.get("end_date"),
                record.get("agency_name"),
                record.get("state"),
            )

            if key not in seen:
                seen.add(key)
                unique_results.append(self._transform_record(record))

        return unique_results

    def get_officers_by_name(
    self,
    first_name: str,
    last_name: str
) -> List[PostEmploymentRecord]:
        """
        Get all officers with matching first/last name across the entire database.
        Returns records regardless of middle name, suffix, or agency.
        Uses prefix matching on last name to catch suffixes (e.g., "JR", "SR", "II").

        Args:
            first_name: Officer first name (case-insensitive exact match)
            last_name: Officer last name (case-insensitive, prefix match to catch suffixes)

        Returns:
            List of PostEmploymentRecord models
        """
        if not first_name or not last_name:
            return []

        try:
            # Get all records matching first name (exact) and last name (prefix)
            # This will catch variations like "CHAVEZ", "CHAVEZ JR", "CHAVEZ SR", etc.
            response = (
                self.supabase.table("postie")
                .select("*")
                .ilike("first_name", first_name)
                .ilike("last_name", f"{last_name}%")
                .execute()
            )

            if not response.data:
                return []

            # Transform all records
            return [self._transform_record(record) for record in response.data]

        except Exception as e:
            print(f"Error in get_officers_by_name: {e}")
            import traceback
            traceback.print_exc()
            return []

    def get_officers_by_name_and_agency(
        self,
        first_name: str,
        last_name: str,
        agency_name: str,
        similarity_threshold: float = 0.6
    ) -> List[PostEmploymentRecord]:
        """
        Get officers matching first/last name and agency with fuzzy matching.

        Args:
            first_name: Officer first name (case-insensitive exact match)
            last_name: Officer last name (case-insensitive, prefix match to catch suffixes)
            agency_name: Agency name for fuzzy matching
            similarity_threshold: Minimum similarity score (0.0 to 1.0) for agency matching

        Returns:
            List of PostEmploymentRecord models matching name and agency
        """
        if not first_name or not last_name or not agency_name:
            return []

        try:
            # First get all officers with matching name
            all_records = self.get_officers_by_name(first_name, last_name)

            if not all_records:
                return []

            # Apply fuzzy matching on agency name
            from rapidfuzz import fuzz

            filtered_records = []
            agency_name_lower = agency_name.lower()

            for record in all_records:
                post_agency_lower = record.post_agency_name.lower()

                # Calculate similarity score using token_set_ratio (handles word order differences)
                similarity = fuzz.token_set_ratio(agency_name_lower, post_agency_lower) / 100.0

                if similarity >= similarity_threshold:
                    filtered_records.append(record)

            return filtered_records

        except Exception as e:
            print(f"Error in get_officers_by_name_and_agency: {e}")
            import traceback
            traceback.print_exc()
            return []

    def get_counties_for_agencies_batch(self, agency_names: List[str]) -> Dict[str, Optional[str]]:
        """
        Batch lookup counties for multiple agencies using a single query.
        Uses fuzzy matching (ILIKE) to handle case differences and partial matches.

        Args:
            agency_names: List of agency names to look up

        Returns:
            Dict mapping agency_name -> county (or None if not found)
        """
        if not agency_names:
            return {}

        # Build list including Office/Department variations for county sheriffs
        agency_variations = {}  # Maps variation -> original_name

        for name in agency_names:
            agency_variations[name] = name
            normalized = name.lower()
            if 'county sheriff' in normalized:
                if 'office' in normalized:
                    alternate = name.replace('Office', 'Department').replace('office', 'department')
                    agency_variations[alternate] = name
                elif 'department' in normalized:
                    alternate = name.replace('Department', 'Office').replace('department', 'office')
                    agency_variations[alternate] = name

        # NEW STRATEGY: Fetch ALL agency-county pairs once, then do fuzzy matching in Python
        # This is MUCH faster than N individual queries

        # Fetch all distinct agency-county pairs using pagination
        # Supabase has a default limit of 1000 rows, must paginate to get all records
        all_db_records = []
        batch_size = 1000
        page = 0

        # DEBUG: Log pagination progress to file
        import os
        pagination_debug_file = os.path.join(os.path.dirname(__file__), "..", "resolve", "data", "output", "pagination_debug.log")
        os.makedirs(os.path.dirname(pagination_debug_file), exist_ok=True)

        with open(pagination_debug_file, 'a') as debug_log:
            debug_log.write("\n" + "="*80 + "\n")
            debug_log.write("PAGINATION DEBUG - Starting fetch\n")
            debug_log.write("="*80 + "\n")

            while True:
                start = page * batch_size
                end = start + batch_size - 1

                debug_log.write(f"Page {page}: Fetching range {start}-{end}...\n")

                response = (
                    self.supabase.table("postie")
                    .select("agency_name,county")
                    .range(start, end)
                    .execute()
                )

                batch = response.data or []
                debug_log.write(f"  Got {len(batch)} records\n")

                if not batch:
                    debug_log.write(f"  Empty batch - stopping\n")
                    break

                all_db_records.extend(batch)

                # If we got fewer than batch_size records, we've reached the end
                if len(batch) < batch_size:
                    debug_log.write(f"  Batch size ({len(batch)}) < {batch_size} - reached end\n")
                    break

                page += 1

            debug_log.write(f"\nTotal records fetched: {len(all_db_records)}\n")
            debug_log.write(f"Unique agencies: {len(set(r.get('agency_name', '') for r in all_db_records if r.get('agency_name')))}\n")
            debug_log.write("="*80 + "\n")

        # Build a lookup map: agency_name -> county (taking first match)
        db_agency_to_county = {}
        for record in all_db_records:
            agency_name = record.get('agency_name', '').strip()
            county = record.get('county')
            if agency_name and agency_name not in db_agency_to_county:
                db_agency_to_county[agency_name] = county

        # Now do fuzzy matching in Python using the fetched data
        agency_to_county = {}

        # Debug output
        import os
        debug_file = os.path.join(os.path.dirname(__file__), "..", "resolve", "data", "output", "batch_county_debug.log")
        os.makedirs(os.path.dirname(debug_file), exist_ok=True)

        with open(debug_file, 'a') as f:
            f.write("\n" + "="*80 + "\n")
            f.write(f"BATCH COUNTY LOOKUP - {len(agency_names)} agencies\n")
            f.write("="*80 + "\n")
            f.write(f"STRATEGY: Fetch ALL agency-county pairs once, fuzzy match in Python\n")
            f.write(f"  1. Single DB query for all records (limit: 100,000)\n")
            f.write(f"  2. Build in-memory lookup: agency_name -> county\n")
            f.write(f"  3. Fuzzy match requested agencies against lookup\n")
            f.write(f"\n")
            f.write(f"DATABASE FETCH:\n")
            f.write(f"  Total records fetched: {len(all_db_records)}\n")
            f.write(f"  Unique agencies in DB: {len(db_agency_to_county)}\n")
            f.write(f"\n")
            f.write(f"REQUESTED AGENCIES ({len(agency_names)}):\n")
            for name in agency_names:
                f.write(f"  - {name}\n")
            f.write(f"\n")
            f.write(f"MATCHING PROCESS:\n")

        # For each requested agency (and its variations), find matching DB agency
        for variant, original in agency_variations.items():
            if original in agency_to_county:
                continue  # Already found

            # Case-insensitive substring match
            variant_lower = variant.lower()
            for db_agency, county in db_agency_to_county.items():
                db_agency_lower = db_agency.lower()

                # Same logic as ILIKE %pattern%: check if variant is substring of DB agency
                if variant_lower in db_agency_lower or db_agency_lower in variant_lower:
                    agency_to_county[original] = county
                    with open(debug_file, 'a') as f:
                        f.write(f"✓ MATCHED: '{variant}' -> '{db_agency}' (County: {county})\n")
                    break

        # Ensure all requested agencies are in result (with None if not found)
        result = {name: agency_to_county.get(name) for name in agency_names}

        # Log results
        with open(debug_file, 'a') as f:
            not_found = [name for name, county in result.items() if county is None]
            f.write(f"\nRESULTS:\n")
            f.write(f"  Found: {len(agency_to_county)}/{len(agency_names)}\n")
            if not_found:
                f.write(f"\n  NOT FOUND ({len(not_found)} agencies):\n")
                for name in not_found:
                    f.write(f"    - '{name}'\n")
            f.write("\n")

        return result

    def get_candidates_for_mentions_batch(self, mentions: List[Dict]) -> Dict[str, List[PostEmploymentRecord]]:
        """
        True batch candidate generation using combined OR queries.

        Strategy: Instead of 200 individual queries (2 per mention × 100 mentions),
        build 2 massive OR queries covering all mentions, then partition results.

        Args:
            mentions: List of mention dicts with keys:
                - first_name, last_name, agency_type, start_year, end_year, state

        Returns:
            Dict mapping mention index (as string) -> list of PostEmploymentRecords
        """
        if not mentions:
            return {}

        from models.src import AgencyType
        from collections import defaultdict

        # Group mentions by (state, agency_type) for efficient querying
        groups = defaultdict(list)
        for idx, mention_dict in enumerate(mentions):
            agency_type_str = mention_dict.get('agency_type', 'POLICE')
            if isinstance(agency_type_str, str):
                agency_type = AgencyType.POLICE if agency_type_str.upper() == 'POLICE' else AgencyType.CORRECTIONS
            else:
                agency_type = agency_type_str

            state = mention_dict.get('state', 'CA')
            key = (state, agency_type.value)
            groups[key].append((idx, mention_dict, agency_type))

        # Process each group with true batch queries
        all_results = {}

        for (state, agency_type_val), group_mentions in groups.items():
            # Build OR conditions for both query patterns
            or_conditions_pattern1 = []  # fn_prefix + ln_exact
            or_conditions_pattern2 = []  # fn_exact + ln_prefix

            for idx, mention_dict, agency_type in group_mentions:
                first_name = mention_dict.get('first_name', '')
                last_name = mention_dict.get('last_name', '')

                if first_name and last_name:
                    # Pattern 1: first name prefix (2 chars) + exact last name
                    if len(first_name) >= 2:
                        fn_prefix = first_name[:2]
                        ln_exact = last_name
                        or_conditions_pattern1.append(
                            f"and(first_name.ilike.{fn_prefix}%,last_name.eq.{ln_exact})"
                        )

                    # Pattern 2: exact first name + last name prefix (2 chars)
                    if len(last_name) >= 2:
                        fn_exact = first_name
                        ln_prefix = last_name[:2]
                        or_conditions_pattern2.append(
                            f"and(first_name.eq.{fn_exact},last_name.ilike.{ln_prefix}%)"
                        )

            # Execute both queries if we have conditions
            group_records = []

            if or_conditions_pattern1:
                try:
                    # Apply OR conditions (smaller batches to avoid query timeout)
                    batch_size = 10
                    for i in range(0, len(or_conditions_pattern1), batch_size):
                        batch = or_conditions_pattern1[i:i+batch_size]
                        or_string = ",".join(batch)

                        # Create fresh query for each batch to avoid filter accumulation
                        query1 = self.supabase.table("postie").select("*")

                        # Apply state filter
                        if state:
                            query1 = query1.ilike("state", f"%{state}%")

                        # Apply agency_type filter
                        query1 = query1.eq("agency_type", agency_type_val)

                        # Apply OR conditions for this batch
                        query1 = query1.or_(or_string)

                        # Remove default limit to get all results
                        query1 = query1.limit(100000)

                        response1 = query1.execute()
                        print(f"[DEBUG] Pattern1 batch {i//batch_size + 1}: {len(batch)} conditions -> {len(response1.data or [])} records")
                        group_records.extend(response1.data or [])

                except Exception as e:
                    print(f"Error in batch query pattern 1: {e}")

            if or_conditions_pattern2:
                try:
                    # Apply OR conditions (smaller batches to avoid query timeout)
                    batch_size = 10
                    for i in range(0, len(or_conditions_pattern2), batch_size):
                        batch = or_conditions_pattern2[i:i+batch_size]
                        or_string = ",".join(batch)

                        # Create fresh query for each batch to avoid filter accumulation
                        query2 = self.supabase.table("postie").select("*")

                        # Apply state filter
                        if state:
                            query2 = query2.ilike("state", f"%{state}%")

                        # Apply agency_type filter
                        query2 = query2.eq("agency_type", agency_type_val)

                        # Apply OR conditions for this batch
                        query2 = query2.or_(or_string)

                        # Remove default limit to get all results
                        query2 = query2.limit(100000)

                        response2 = query2.execute()
                        print(f"[DEBUG] Pattern2 batch {i//batch_size + 1}: {len(batch)} conditions -> {len(response2.data or [])} records")
                        group_records.extend(response2.data or [])

                except Exception as e:
                    print(f"Error in batch query pattern 2: {e}")

            # Transform to PostEmploymentRecord objects (no deduplication - let pipeline handle it)
            transformed_records = [self._transform_record(r) for r in group_records]

            print(f"[DEBUG] Group ({state}, {agency_type_val}): {len(group_mentions)} mentions, {len(group_records)} total records")

            # Build hash index for O(mentions + records) partitioning instead of O(mentions × records)
            from collections import defaultdict
            record_index_p1 = defaultdict(list)  # Pattern 1: (fn_prefix, ln_exact)
            record_index_p2 = defaultdict(list)  # Pattern 2: (fn_exact, ln_prefix)

            for record in transformed_records:
                fn = record.post_first_name.upper()
                ln = record.post_last_name.upper()

                # Index for Pattern 1 lookups: fn_prefix + ln_exact
                if len(fn) >= 2:
                    record_index_p1[(fn[:2], ln)].append(record)

                # Index for Pattern 2 lookups: fn_exact + ln_prefix
                if len(ln) >= 2:
                    record_index_p2[(fn, ln[:2])].append(record)

            # Partition results by mention using hash index (O(mentions) instead of O(mentions × records))
            for idx, mention_dict, agency_type in group_mentions:
                first_name = mention_dict.get('first_name', '').upper()
                last_name = mention_dict.get('last_name', '').upper()

                # Collect candidates from both patterns using hash lookups
                # Use dict to track seen records by ID to avoid duplicates
                seen_records = {}

                # Pattern 1: first name prefix + exact last name
                if len(first_name) >= 2:
                    fn_prefix = first_name[:2]
                    key = (fn_prefix, last_name)
                    for record in record_index_p1.get(key, []):
                        # Use person_nbr + start_date + agency as unique key
                        record_key = (record.post_person_nbr, record.post_start_date, record.post_agency_name)
                        seen_records[record_key] = record

                # Pattern 2: exact first name + last name prefix
                if len(last_name) >= 2:
                    ln_prefix = last_name[:2]
                    key = (first_name, ln_prefix)
                    for record in record_index_p2.get(key, []):
                        record_key = (record.post_person_nbr, record.post_start_date, record.post_agency_name)
                        seen_records[record_key] = record

                all_results[str(idx)] = list(seen_records.values())

        return all_results

    def get_batch_name_uniqueness(self, names: List[List[str]]) -> Dict[str, int]:
        """
        Get count of unique persons for each (first_name, last_name) pair.
        Uses hash-based partitioning for O(n+m) complexity.

        Args:
            names: List of [first_name, last_name] pairs

        Returns:
            Dict with keys "FirstName|LastName" and values as person counts
        """
        if not names:
            return {}

        from collections import defaultdict

        # Deduplicate input names
        unique_names = set(tuple(name) for name in names)

        # Group names by state (all CA for now, but keeping pattern consistent)
        # Query in batches to avoid overwhelming database
        all_records = []
        batch_size = 50
        names_list = list(unique_names)

        for i in range(0, len(names_list), batch_size):
            batch = names_list[i:i + batch_size]

            # Build OR conditions for this batch
            or_conditions = []
            for first_name, last_name in batch:
                if first_name and last_name:
                    # Case-insensitive exact match on both first and last name
                    or_conditions.append(
                        f"and(first_name.ilike.{first_name},last_name.ilike.{last_name})"
                    )

            if not or_conditions:
                continue

            try:
                or_string = ",".join(or_conditions)

                # Query for person_nbr, first_name, last_name only (not full records)
                query = self.supabase.table("postie").select("person_nbr,first_name,last_name")
                query = query.or_(or_string)
                query = query.limit(100000)  # Remove default limit

                response = query.execute()
                all_records.extend(response.data or [])

            except Exception as e:
                print(f"Error in batch name uniqueness query: {e}")
                # Continue with other batches even if one fails

        # Hash-based partitioning: count unique person_nbr per name
        # Build a mapping from uppercase name keys to original case for result keys
        name_key_mapping = {}
        for first, last in unique_names:
            upper_key = f"{first.upper()}|{last.upper()}"
            original_key = f"{first}|{last}"
            name_key_mapping[upper_key] = original_key

        name_to_persons = defaultdict(set)

        # O(n) single pass through records with O(1) hash lookup
        for record in all_records:
            fn = record.get("first_name", "").strip().upper()
            ln = record.get("last_name", "").strip().upper()
            person_nbr = record.get("person_nbr")

            if not person_nbr:
                continue

            # Check if this record matches any requested name (O(1) lookup)
            upper_key = f"{fn}|{ln}"
            if upper_key in name_key_mapping:
                original_key = name_key_mapping[upper_key]
                name_to_persons[original_key].add(person_nbr)

        # Convert sets to counts
        result = {}
        for first_name, last_name in names:
            key = f"{first_name}|{last_name}"
            if key in name_to_persons:
                result[key] = len(name_to_persons[key])
            else:
                result[key] = 0

        return result