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


def get_post_data_from_api(
    api_base_url: str = "http://localhost:8000", state: Optional[str] = None
) -> pd.DataFrame:
    """
    Updated replacement for the original get_post_data() function.
    Fetches POST employment data from the new dedicated API endpoint.

    Args:
        api_base_url: Base URL for the API
        state: Optional state filter

    Returns:
        DataFrame with POST employment records in the same format as the original function
    """
    logger.info(f"Fetching POST data from API at {api_base_url}")
    if state:
        logger.info(f"Filtering by state: {state}")

    client = NPIClient(base_url=api_base_url)

    # Check if API is available
    if not client.health_check():
        raise ConnectionError(
            f"Cannot connect to API at {api_base_url}. Make sure the server is running."
        )

    # Get all employment records
    employment_records = client.get_all_post_employment_records(state=state)

    if not employment_records:
        logger.warning("No employment records found from API")
        return pd.DataFrame()

    # Convert PostEmploymentRecord models to DataFrame
    post_data = []
    for record in employment_records:
        post_data.append(record.dict())

    post = pd.DataFrame(post_data)

    post.loc[:, "post_start_date"] = pd.to_datetime(
        post["post_start_date"], errors="coerce"
    )
    post.loc[:, "post_end_date"] = pd.to_datetime(
        post["post_end_date"], errors="coerce"
    ).fillna(pd.Timestamp.today().round(freq="D"))

    # Filter out records with "withheld" in person_nbr
    selected = (
        ~post["post_person_nbr"].str.casefold().str.contains("withheld", na=False)
    )

    outcols = [
        "post_person_nbr",
        "post_first_name",
        "post_middle_name",
        "post_last_name",
        "post_suffix",
        "post_agency_name",
        "post_agency_type",
        "post_start_date",
        "post_end_date",
        "post_separation_reason",
        "state", 
    ]

    # Ensure all columns exist
    for col in outcols:
        if col not in post.columns:
            post[col] = ""

    result = post.loc[selected, outcols]
    logger.info(f"Returning {len(result)} POST employment records")

    return result


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
