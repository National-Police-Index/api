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
        
        # First try exact match
        response = (
            self.supabase.table("postie")
            .select("county")
            .ilike("agency_name", f"%{agency_name}%")
            .limit(1)
            .execute()
        )
        
        if response.data and len(response.data) > 0:
            return response.data[0].get("county")
        
        # If no match and it's a county sheriff agency, try swapping office/department
        normalized = agency_name.lower()
        if "county sheriff" in normalized:
            if "office" in normalized:
                alternate_name = agency_name.replace("Office", "Department").replace("office", "department")
            elif "department" in normalized:
                alternate_name = agency_name.replace("Department", "Office").replace("department", "office")
            else:
                return None
            
            response = (
                self.supabase.table("postie")
                .select("county")
                .ilike("agency_name", f"%{alternate_name}%")
                .limit(1)
                .execute()
            )
            
            if response.data and len(response.data) > 0:
                return response.data[0].get("county")
        
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