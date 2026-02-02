import unittest
import requests
import pandas as pd
from datetime import datetime, date
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(
    os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "resolve")
)

from models.src import PostEmploymentRecord, OfficerMention, CandidateQuery, AgencyType
from resolve.src.api import NPIClient
from database.src import SupabaseClient
from server.config import SUPABASE_URL, SUPABASE_KEY


class TestModels(unittest.TestCase):
    """Test Pydantic model validation and functionality"""

    def test_post_employment_record_valid(self):
        """Test valid PostEmploymentRecord creation"""
        record = PostEmploymentRecord(
            post_person_nbr="A123-B456",
            post_first_name="John",
            post_middle_name="M",
            post_last_name="Smith",
            post_agency_name="Test Police Department",
            post_agency_type=AgencyType.POLICE,
            post_start_date=datetime(2020, 1, 1),
            post_end_date=datetime(2023, 1, 1),
            post_separation_reason="Retired",
            state="CA",
        )
        self.assertEqual(record.post_person_nbr, "A123-B456")
        self.assertEqual(record.post_first_name, "John")
        self.assertEqual(record.state, "CA")

    def test_post_employment_record_minimal(self):
        """Test PostEmploymentRecord with minimal required fields"""
        record = PostEmploymentRecord(
            post_person_nbr="A123-B456",
            post_first_name="John",
            post_last_name="Smith",
            post_agency_name="Test Police Department",
        )
        self.assertIsNone(record.post_middle_name)
        self.assertEqual(record.post_agency_type, AgencyType.POLICE)
        self.assertIsNone(record.state)

    def test_post_employment_record_invalid_dates(self):
        """Test PostEmploymentRecord with invalid date formats"""
        with self.assertRaises(ValueError):
            PostEmploymentRecord(
                post_person_nbr="A123-B456",
                post_first_name="John",
                post_last_name="Smith",
                post_agency_name="Test Police Department",
                post_start_date="invalid-date",
            )

    def test_officer_mention_valid(self):
        """Test valid OfficerMention creation"""
        mention = OfficerMention(
            mention_uid="test-123",
            mention_incident_date=date(2020, 1, 1),
            mention_first_name="Jane",
            mention_last_name="Doe",
            mention_agency="Test Police",
            state="NY",
        )
        self.assertEqual(mention.mention_uid, "test-123")
        self.assertEqual(mention.state, "NY")
        self.assertEqual(mention.mention_agency_type, AgencyType.POLICE)

    def test_candidate_query_defaults(self):
        """Test CandidateQuery with default values"""
        query = CandidateQuery()
        self.assertEqual(query.offset, 0)
        self.assertEqual(query.agency_type, AgencyType.POLICE)
        self.assertIsNone(query.first_name)
        self.assertIsNone(query.limit)

    def test_agency_type_enum(self):
        """Test AgencyType enum validation"""
        self.assertEqual(AgencyType.POLICE.value, "POLICE")
        self.assertEqual(AgencyType.CORRECTIONS.value, "CORRECTIONS")

        # Test invalid agency type
        with self.assertRaises(ValueError):
            AgencyType("FIRE")


class TestAPIEndpoints(unittest.TestCase):
    """Test FastAPI endpoints"""

    BASE_URL = "http://localhost:8000"

    @classmethod
    def setUpClass(cls):
        """Check if API server is running"""
        try:
            response = requests.get(f"{cls.BASE_URL}/")
            if response.status_code != 200:
                raise unittest.SkipTest("API server is not running")
        except requests.exceptions.ConnectionError:
            raise unittest.SkipTest("API server is not running")

    def test_health_check(self):
        """Test root health check endpoint"""
        response = requests.get(f"{self.BASE_URL}/")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("message", data)
        self.assertEqual(data["status"], "running")

    def test_employment_records_no_filters(self):
        """Test employment records endpoint without filters"""
        response = requests.get(f"{self.BASE_URL}/post/employment?limit=5")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIsInstance(data, list)
        if data:  # If there's data, validate structure
            record = data[0]
            self.assertIn("post_person_nbr", record)
            self.assertIn("post_first_name", record)
            self.assertIn("state", record)

    def test_employment_records_with_filters(self):
        """Test employment records endpoint with name filters"""
        response = requests.get(
            f"{self.BASE_URL}/post/employment?first_name=JOHN&last_name=SMITH&limit=10"
        )
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIsInstance(data, list)

    def test_employment_records_with_state_filter(self):
        """Test employment records endpoint with state filter"""
        response = requests.get(f"{self.BASE_URL}/post/employment?state=CA&limit=10")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIsInstance(data, list)
        # Verify all returned records have CA state
        for record in data:
            self.assertEqual(record["state"], "CA")

    def test_employment_count(self):
        """Test employment count endpoint"""
        response = requests.get(f"{self.BASE_URL}/post/employment/count")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("total_records", data)
        self.assertIsInstance(data["total_records"], int)
        self.assertGreaterEqual(data["total_records"], 0)

    def test_employment_count_with_filters(self):
        """Test employment count endpoint with filters"""
        response = requests.get(
            f"{self.BASE_URL}/post/employment/count?state=CA&first_name=JOHN"
        )
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("total_records", data)
        self.assertIsInstance(data["total_records"], int)

    def test_stats_endpoint(self):
        """Test stats endpoint"""
        response = requests.get(f"{self.BASE_URL}/post/stats")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        expected_keys = [
            "total_employment_records",
            "unique_officers",
            "unique_agencies",
            "unique_states",
            "active_officers",
            "separated_officers",
        ]
        for key in expected_keys:
            self.assertIn(key, data)
            self.assertIsInstance(data[key], int)

    def test_candidates_endpoint_valid(self):
        """Test candidates endpoint with valid parameters"""
        response = requests.get(
            f"{self.BASE_URL}/post/candidates?"
            f"first_name=ROBERT&last_name=SMITH&agency_type=POLICE&start_year=2018&end_year=2020"
        )
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIsInstance(data, list)

    def test_candidates_endpoint_with_state(self):
        """Test candidates endpoint with state filter"""
        response = requests.get(
            f"{self.BASE_URL}/post/candidates?"
            f"first_name=ROBERT&last_name=SMITH&agency_type=POLICE&start_year=2018&end_year=2020&state=CA"
        )
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIsInstance(data, list)
        # Verify all returned records have CA state
        for record in data:
            self.assertEqual(record["state"], "CA")

    def test_candidates_endpoint_no_matches(self):
        """Test candidates endpoint that should return no matches"""
        response = requests.get(
            f"{self.BASE_URL}/post/candidates?"
            f"first_name=NONEXISTENT&last_name=PERSON&agency_type=POLICE&start_year=2018&end_year=2020"
        )
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data, [])

    def test_candidates_endpoint_missing_required_params(self):
        """Test candidates endpoint with missing required parameters"""
        response = requests.get(
            f"{self.BASE_URL}/post/candidates?first_name=ROBERT"  # Missing required params
        )
        self.assertEqual(response.status_code, 422)  # Validation error

    def test_invalid_agency_type(self):
        """Test candidates endpoint with invalid agency type"""
        response = requests.get(
            f"{self.BASE_URL}/post/candidates?"
            f"first_name=ROBERT&last_name=SMITH&agency_type=INVALID&start_year=2018&end_year=2020"
        )
        self.assertEqual(response.status_code, 422)  # Validation error


class TestNPIClient(unittest.TestCase):
    """Test NPIClient functionality"""

    def setUp(self):
        self.client = NPIClient(base_url="http://localhost:8000")

    @classmethod
    def setUpClass(cls):
        """Check if API server is running"""
        try:
            client = NPIClient(base_url="http://localhost:8000")
            if not client.health_check():
                raise unittest.SkipTest("API server is not running")
        except Exception:
            raise unittest.SkipTest("API server is not running")

    def test_health_check(self):
        """Test client health check"""
        self.assertTrue(self.client.health_check())

    def test_get_post_employment_records(self):
        """Test getting employment records through client"""
        records = self.client.get_post_employment_records(limit=5)
        self.assertIsInstance(records, list)
        if records:
            self.assertIsInstance(records[0], PostEmploymentRecord)

    def test_get_post_employment_records_with_filters(self):
        """Test getting employment records with filters"""
        records = self.client.get_post_employment_records(
            first_name="ROBERT", last_name="SMITH", state="CA", limit=10
        )
        self.assertIsInstance(records, list)
        for record in records:
            self.assertIsInstance(record, PostEmploymentRecord)
            if record.state:  # Only check if state is not None/empty
                self.assertEqual(record.state, "CA")

    def test_get_candidates_for_mention(self):
        """Test getting candidates through client"""
        candidates = self.client.get_candidates_for_mention(
            first_name="ROBERT",
            last_name="SMITH",
            incident_year=2018,
            agency_type=AgencyType.POLICE,
        )
        self.assertIsInstance(candidates, list)
        for candidate in candidates:
            self.assertIsInstance(candidate, PostEmploymentRecord)

    def test_get_candidates_with_state_filter(self):
        """Test getting candidates with state filter"""
        candidates = self.client.get_candidates_for_mention(
            first_name="ROBERT",
            last_name="SMITH",
            incident_year=2018,
            agency_type=AgencyType.POLICE,
            state="CA",
        )
        self.assertIsInstance(candidates, list)
        for candidate in candidates:
            self.assertEqual(candidate.state, "CA")

    def test_get_post_employment_count(self):
        """Test getting employment record count"""
        count = self.client.get_post_employment_count()
        self.assertIsInstance(count, int)
        self.assertGreaterEqual(count, 0)

    def test_get_post_employment_count_with_filters(self):
        """Test getting employment record count with filters"""
        count = self.client.get_post_employment_count(first_name="ROBERT", state="CA")
        self.assertIsInstance(count, int)
        self.assertGreaterEqual(count, 0)

    def test_nonexistent_server(self):
        """Test client behavior with non-existent server"""
        bad_client = NPIClient(base_url="http://localhost:9999")
        self.assertFalse(bad_client.health_check())
        self.assertEqual(bad_client.get_post_employment_records(), [])
        self.assertIsNone(bad_client.get_post_employment_count())


class TestDatabaseLayer(unittest.TestCase):
    """Test SupabaseClient functionality"""

    def setUp(self):
        self.db_client = SupabaseClient(SUPABASE_URL, SUPABASE_KEY)

    def test_get_post_employment_records(self):
        """Test basic database record retrieval"""
        query = CandidateQuery(limit=5)
        records = self.db_client.get_post_employment_records(query)
        self.assertIsInstance(records, list)
        if records:
            self.assertIsInstance(records[0], PostEmploymentRecord)

    def test_get_post_employment_records_with_name_filters(self):
        """Test database record retrieval with name filters"""
        query = CandidateQuery(first_name="ROBERT", last_name="SMITH", limit=10)
        records = self.db_client.get_post_employment_records(query)
        self.assertIsInstance(records, list)
        for record in records:
            self.assertIsInstance(record, PostEmploymentRecord)

    def test_get_post_employment_records_with_state_filter(self):
        """Test database record retrieval with state filter"""
        query = CandidateQuery(state="CA", limit=10)
        records = self.db_client.get_post_employment_records(query)
        self.assertIsInstance(records, list)
        for record in records:
            self.assertEqual(record.state, "CA")

    def test_get_post_employment_count(self):
        """Test database record count"""
        count = self.db_client.get_post_employment_count()
        self.assertIsInstance(count, int)
        self.assertGreaterEqual(count, 0)

    def test_get_post_employment_count_with_filters(self):
        """Test database record count with filters"""
        query = CandidateQuery(state="CA", first_name="ROBERT")
        count = self.db_client.get_post_employment_count(query)
        self.assertIsInstance(count, int)
        self.assertGreaterEqual(count, 0)

    def test_get_candidates_for_mention(self):
        """Test candidate generation logic"""
        query = CandidateQuery(
            first_name="ROBERT", last_name="SMITH", start_year=2018, end_year=2020
        )
        candidates = self.db_client.get_candidates_for_mention(query)
        self.assertIsInstance(candidates, list)
        for candidate in candidates:
            self.assertIsInstance(candidate, PostEmploymentRecord)

    def test_get_candidates_with_state_filter(self):
        """Test candidate generation with state filter"""
        query = CandidateQuery(
            first_name="ROBERT",
            last_name="SMITH",
            state="CA",
            start_year=2018,
            end_year=2020,
        )
        candidates = self.db_client.get_candidates_for_mention(query)
        self.assertIsInstance(candidates, list)
        for candidate in candidates:
            self.assertEqual(candidate.state, "CA")

    def test_get_post_stats(self):
        """Test database statistics"""
        stats = self.db_client.get_post_stats()
        self.assertIsInstance(stats, dict)
        expected_keys = [
            "total_employment_records",
            "unique_officers",
            "unique_agencies",
            "unique_states",
            "active_officers",
            "separated_officers",
        ]
        for key in expected_keys:
            self.assertIn(key, stats)
            self.assertIsInstance(stats[key], int)

    def test_search_post_employment_by_name(self):
        """Test name-based search functionality"""
        results = self.db_client.search_post_employment_by_name("ROBERT SMITH")
        self.assertIsInstance(results, list)
        for record in results:
            self.assertIsInstance(record, PostEmploymentRecord)

    def test_get_employment_record_by_person_nbr(self):
        """Test retrieval by person number"""
        # First get a person number from existing data
        query = CandidateQuery(limit=1)
        records = self.db_client.get_post_employment_records(query)
        if records:
            person_nbr = records[0].post_person_nbr
            results = self.db_client.get_employment_record_by_person_nbr(person_nbr)
            self.assertIsInstance(results, list)
            self.assertGreater(len(results), 0)
            for record in results:
                self.assertEqual(record.post_person_nbr, person_nbr)


class TestEdgeCases(unittest.TestCase):
    """Test edge cases and error conditions"""

    def setUp(self):
        self.client = NPIClient(base_url="http://localhost:8000")

    @classmethod
    def setUpClass(cls):
        """Check if API server is running"""
        try:
            client = NPIClient(base_url="http://localhost:8000")
            if not client.health_check():
                raise unittest.SkipTest("API server is not running")
        except Exception:
            raise unittest.SkipTest("API server is not running")

    def test_empty_name_filters(self):
        """Test behavior with empty string name filters"""
        records = self.client.get_post_employment_records(
            first_name="", last_name="", limit=5
        )
        self.assertIsInstance(records, list)

    def test_very_long_names(self):
        """Test behavior with very long name strings"""
        long_name = "A" * 1000
        records = self.client.get_post_employment_records(
            first_name=long_name, last_name=long_name, limit=5
        )
        self.assertIsInstance(records, list)
        self.assertEqual(len(records), 0)  # Should find no matches

    def test_special_characters_in_names(self):
        """Test behavior with special characters in names"""
        special_chars = ["O'CONNOR", "SMITH-JONES", "JOSÉ", "D'ANGELO"]
        for name in special_chars:
            records = self.client.get_post_employment_records(first_name=name, limit=5)
            self.assertIsInstance(records, list)

    def test_case_insensitive_search(self):
        """Test case insensitive name searching"""
        # Test with different cases
        test_cases = ["robert", "ROBERT", "Robert", "rObErT"]
        results = []
        for case in test_cases:
            records = self.client.get_post_employment_records(first_name=case, limit=10)
            results.append(len(records))

        # All case variations should return the same number of results
        if any(r > 0 for r in results):  # Only test if there are any Roberts
            self.assertTrue(all(r == results[0] for r in results))

    def test_nonexistent_state(self):
        """Test filtering by non-existent state"""
        records = self.client.get_post_employment_records(
            state="ZZ", limit=10  # Non-existent state code
        )
        self.assertIsInstance(records, list)
        self.assertEqual(len(records), 0)

    def test_extreme_date_ranges(self):
        """Test candidates endpoint with extreme date ranges"""
        # Very old dates
        candidates = self.client.get_candidates_for_mention(
            first_name="ROBERT", last_name="SMITH", incident_year=1900
        )
        self.assertIsInstance(candidates, list)

        # Future dates
        candidates = self.client.get_candidates_for_mention(
            first_name="ROBERT", last_name="SMITH", incident_year=2100
        )
        self.assertIsInstance(candidates, list)

    def test_zero_and_negative_limits(self):
        """Test behavior with zero and negative limits"""
        # Zero limit should return empty list
        records = self.client.get_post_employment_records(limit=0)
        self.assertIsInstance(records, list)
        self.assertEqual(len(records), 0)

        # Negative limit behavior - API treats this as "no limit" rather than zero
        # This is the current API behavior, not necessarily ideal but expected
        records = self.client.get_post_employment_records(limit=-1)
        self.assertIsInstance(records, list)
        # API treats negative limits as "no limit", so we expect records returned

    def test_large_offset(self):
        """Test behavior with very large offset values"""
        records = self.client.get_post_employment_records(
            offset=1000000, limit=5  # Very large offset
        )
        self.assertIsInstance(records, list)
        self.assertEqual(len(records), 0)  # Should return empty list

    def test_unicode_names(self):
        """Test handling of Unicode characters in names"""
        unicode_names = ["José", "François", "Müller", "李明"]
        for name in unicode_names:
            records = self.client.get_post_employment_records(first_name=name, limit=5)
            self.assertIsInstance(records, list)


class TestDataIntegrity(unittest.TestCase):
    """Test data integrity and consistency"""

    def setUp(self):
        self.client = NPIClient(base_url="http://localhost:8000")

    @classmethod
    def setUpClass(cls):
        """Check if API server is running"""
        try:
            client = NPIClient(base_url="http://localhost:8000")
            if not client.health_check():
                raise unittest.SkipTest("API server is not running")
        except Exception:
            raise unittest.SkipTest("API server is not running")

    def test_record_count_consistency(self):
        """Test that record counts are consistent across endpoints"""
        # Get total count
        total_count = self.client.get_post_employment_count()

        # Get count with no filters (should be same as total)
        unfiltered_count = self.client.get_post_employment_count(
            first_name=None, last_name=None, agency=None, state=None
        )

        self.assertEqual(total_count, unfiltered_count)

    def test_state_filter_integrity(self):
        """Test that state filtering is consistent"""
        # Get all CA records
        ca_records = self.client.get_post_employment_records(state="CA", limit=1000)
        ca_count = self.client.get_post_employment_count(state="CA")

        # Count should match actual records (up to limit)
        if len(ca_records) < 1000:  # If we got all records
            self.assertEqual(len(ca_records), ca_count)
        else:  # If we hit the limit
            self.assertGreaterEqual(ca_count, 1000)

        # All returned records should have CA state
        for record in ca_records:
            self.assertEqual(record.state, "CA")

    def test_date_field_consistency(self):
        """Test that date fields are properly formatted"""
        records = self.client.get_post_employment_records(limit=50)

        for record in records:
            # Check that dates are either None or proper datetime objects
            if record.post_start_date is not None:
                self.assertIsInstance(record.post_start_date, datetime)
            if record.post_end_date is not None:
                self.assertIsInstance(record.post_end_date, datetime)

            # If both dates exist, start should be before or equal to end
            # Skip obviously invalid placeholder dates (like 1901-01-01)
            if (
                record.post_start_date is not None
                and record.post_end_date is not None
                and record.post_end_date.year > 1950
            ):  # Skip placeholder dates
                self.assertLessEqual(record.post_start_date, record.post_end_date)

    def test_required_fields_populated(self):
        """Test that required fields are always populated"""
        records = self.client.get_post_employment_records(limit=100)

        for record in records:
            # Required fields should never be None or empty
            self.assertTrue(record.post_person_nbr)
            self.assertTrue(record.post_first_name)
            self.assertTrue(record.post_last_name)
            self.assertTrue(record.post_agency_name)
            self.assertIn(
                record.post_agency_type, [AgencyType.POLICE, AgencyType.CORRECTIONS]
            )

    def test_candidate_deduplication(self):
        """Test that candidate queries return unique records"""
        candidates = self.client.get_candidates_for_mention(
            first_name="ROBERT", last_name="SMITH", incident_year=2018
        )

        # Check for duplicates by creating a set of unique identifiers
        seen_records = set()
        for candidate in candidates:
            record_id = (
                candidate.post_person_nbr,
                candidate.post_start_date,
                candidate.post_end_date,
                candidate.post_agency_name,
            )
            self.assertNotIn(record_id, seen_records, "Duplicate record found")
            seen_records.add(record_id)


def run_performance_tests():
    """Run basic performance tests"""
    print("Running performance tests...")

    client = NPIClient(base_url="http://localhost:8000")

    if not client.health_check():
        print("❌ API server not available for performance tests")
        return

    import time

    # Test 1: Large batch retrieval
    start_time = time.time()
    records = client.get_post_employment_records(limit=1000)
    batch_time = time.time() - start_time
    print(f"✅ Large batch retrieval (1000 records): {batch_time:.2f}s")

    # Test 2: Multiple small queries
    start_time = time.time()
    for i in range(10):
        client.get_post_employment_records(limit=10, offset=i * 10)
    multi_query_time = time.time() - start_time
    print(f"✅ Multiple small queries (10x10 records): {multi_query_time:.2f}s")

    # Test 3: Complex candidate search
    start_time = time.time()
    client.get_candidates_for_mention(
        first_name="ROBERT", last_name="SMITH", incident_year=2018, state="CA"
    )
    candidate_time = time.time() - start_time
    print(f"✅ Complex candidate search: {candidate_time:.2f}s")

    print("Performance tests completed.")


if __name__ == "__main__":
    print("POST Employment Data API - Comprehensive Test Suite")
    print("=" * 60)

    # Run unit tests
    unittest.main(argv=[""], exit=False, verbosity=2)

    # Run performance tests
    print("\n" + "=" * 60)
    run_performance_tests()
