import sys
import os
import requests
import pandas as pd
from typing import Optional, List
import time
import logging

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.src import PostEmploymentRecord, AgencyType

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class NPIClient:
    """POST Employment API Client with Pydantic model support"""

    def __init__(self, base_url: str = "http://localhost:8000", timeout: int = 30):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.session = requests.Session()

    def health_check(self) -> bool:
        """Check if the API is running"""
        try:
            response = self.session.get(f"{self.base_url}/", timeout=self.timeout)
            return response.status_code == 200
        except requests.exceptions.RequestException as e:
            logger.error(f"Health check failed: {e}")
            return False

    def get_post_employment_records(
        self,
        limit: Optional[int] = None,
        offset: int = 0,
        first_name: Optional[str] = None,
        last_name: Optional[str] = None,
        agency: Optional[str] = None,
        state: Optional[str] = None,
    ) -> List[PostEmploymentRecord]:
        """
        Get POST employment records with structured name fields.

        Args:
            limit: Maximum number of records to return
            offset: Number of records to skip (for pagination)
            first_name: Filter by first name (partial match)
            last_name: Filter by last name (partial match)
            agency: Filter by agency name (partial match)
            state: Filter by state (partial match)

        Returns:
            List of PostEmploymentRecord models
        """
        try:
            params = {"offset": offset}
            if limit:
                params["limit"] = limit
            if first_name:
                params["first_name"] = first_name
            if last_name:
                params["last_name"] = last_name
            if agency:
                params["agency"] = agency
            if state:
                params["state"] = state

            response = self.session.get(
                f"{self.base_url}/post/employment", params=params, timeout=self.timeout
            )
            response.raise_for_status()

            # Convert response to PostEmploymentRecord models
            raw_data = response.json()
            return [PostEmploymentRecord(**record) for record in raw_data]

        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to get POST employment records: {e}")
            return []

    def get_candidates_for_mention(
        self,
        first_name: str,
        last_name: str,
        incident_year: int = 2018,
        agency_type: AgencyType = AgencyType.POLICE,
        state: Optional[str] = None,
    ) -> List[PostEmploymentRecord]:
        """
        Get candidates for entity resolution.

        Args:
            first_name: Officer first name
            last_name: Officer last name
            incident_year: Year of incident
            agency_type: Type of agency (POLICE/CORRECTIONS)
            state: Optional state filter

        Returns:
            List of PostEmploymentRecord models
        """
        try:
            params = {
                "first_name": first_name,
                "last_name": last_name,
                "agency_type": agency_type.value,
                "start_year": incident_year,
                "end_year": incident_year,
            }

            if state:
                params["state"] = state

            logger.info(
                f"Calling /post/candidates endpoint for {first_name} {last_name}"
                + (f" in {state}" if state else "")
            )

            response = self.session.get(
                f"{self.base_url}/post/candidates", params=params, timeout=self.timeout
            )
            response.raise_for_status()

            raw_data = response.json()
            candidates = [PostEmploymentRecord(**record) for record in raw_data]

            logger.info(
                f"Received {len(candidates)} candidates from dedicated endpoint"
            )
            return candidates

        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to get candidates from dedicated endpoint: {e}")
            return []
        
    def get_county_for_agency(self, agency_name: str) -> Optional[str]:
        """
        Get the county for a given agency name.
        
        Args:
            agency_name: Name of the agency to lookup
            
        Returns:
            County name or None if not found
        """
        try:
            params = {"agency_name": agency_name}
            response = self.session.get(
                f"{self.base_url}/post/agency/county",
                params=params,
                timeout=self.timeout
            )
            response.raise_for_status()
            data = response.json()
            return data.get("county")
        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to get county for agency: {e}")
            return None

    def get_all_post_employment_records(
        self, batch_size: int = 1000, state: Optional[str] = None
    ) -> List[PostEmploymentRecord]:
        """
        Get all POST employment records by paginating through the API.

        Args:
            batch_size: Number of records to fetch per request
            state: Optional state filter

        Returns:
            List of PostEmploymentRecord models
        """
        all_records = []
        offset = 0

        logger.info("Fetching all POST employment records...")
        if state:
            logger.info(f"Filtering by state: {state}")

        while True:
            batch = self.get_post_employment_records(
                limit=batch_size, offset=offset, state=state
            )

            if not batch:
                break

            all_records.extend(batch)
            logger.info(f"Fetched {len(batch)} records (total: {len(all_records)})")

            # If we got fewer records than requested, we've reached the end
            if len(batch) < batch_size:
                break

            offset += batch_size

            # Small delay to be nice to the server
            time.sleep(0.1)

        logger.info(f"Fetched {len(all_records)} total POST employment records")
        return all_records

    def get_post_employment_count(
        self,
        first_name: Optional[str] = None,
        last_name: Optional[str] = None,
        agency: Optional[str] = None,
        state: Optional[str] = None,
    ) -> Optional[int]:
        """Get total count of POST employment records with optional filters"""
        try:
            params = {}
            if first_name:
                params["first_name"] = first_name
            if last_name:
                params["last_name"] = last_name
            if agency:
                params["agency"] = agency
            if state:
                params["state"] = state

            response = self.session.get(
                f"{self.base_url}/post/employment/count",
                params=params,
                timeout=self.timeout,
            )
            response.raise_for_status()
            data = response.json()
            return data.get("total_records")
        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to get POST employment count: {e}")
            return None
        
    def get_officers_by_name(
    self, 
    first_name: str, 
    last_name: str
) -> List[PostEmploymentRecord]:
        """
        Get all officers with matching first/last name across the entire database.
        Returns all records regardless of middle name, suffix, or agency.
        
        Args:
            first_name: Officer first name
            last_name: Officer last name
            
        Returns:
            List of PostEmploymentRecord models
        """
        try:
            params = {
                "first_name": first_name,
                "last_name": last_name
            }
            
            response = self.session.get(
                f"{self.base_url}/post/officers/by-name",
                params=params,
                timeout=self.timeout
            )
            response.raise_for_status()
            
            raw_data = response.json()
            records = [PostEmploymentRecord(**record) for record in raw_data]
            
            logger.info(
                f"Found {len(records)} records for {first_name} {last_name}"
            )
            return records
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to get officers by name: {e}")
            return []


def test_post_employment_api(api_base_url: str = "http://localhost:8000"):
    """Test the POST employment API connection and print some basic info"""
    print(f"Testing POST employment API at {api_base_url}...")

    client = NPIClient(base_url=api_base_url)

    # Health check
    if client.health_check():
        print("✓ API is running")
    else:
        print("✗ API is not accessible")
        return False

    count = client.get_post_employment_count()
    if count:
        print(f"✓ Total POST employment records: {count:,}")

    # Test getting a small sample
    sample_records = client.get_post_employment_records(limit=5)
    print(f"✓ Retrieved {len(sample_records)} sample records")

    if sample_records:
        print("Sample record:")
        record = sample_records[0]
        print(f"  Person NBR: {record.post_person_nbr}")
        print(
            f"  Name: {record.post_first_name} {record.post_middle_name} {record.post_last_name}"
        )
        print(f"  Agency: {record.post_agency_name}")
        print(f"  State: {record.state}")
        print(f"  Start Date: {record.post_start_date}")
        print(f"  End Date: {record.post_end_date}")

    # Test state filtering if there are records
    if sample_records and sample_records[0].state:
        test_state = sample_records[0].state
        state_count = client.get_post_employment_count(state=test_state)
        print(f"✓ Records in {test_state}: {state_count:,}")

    return True


if __name__ == "__main__":
    if test_post_employment_api():
        print("\nFetching POST data from API...")
        try:
            post_data = get_post_data_from_api()
            print(f"Successfully fetched {len(post_data)} POST employment records")
            print("\nSample data:")
            print(post_data.head())

            post_data.to_csv("post_employment_data_sample.csv", index=False)
            print("\nSaved sample data to 'post_employment_data_sample.csv'")

        except Exception as e:
            print(f"Error fetching POST data: {e}")
    else:
        print(
            "Please make sure the CLEAN-POST API server is running with the new POST employment endpoint"
        )
