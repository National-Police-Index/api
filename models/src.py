from pydantic import BaseModel, Field, field_validator
from typing import Optional, List, Dict
from datetime import date, datetime
from enum import Enum


class AgencyType(str, Enum):
    POLICE = "POLICE"
    CORRECTIONS = "CORRECTIONS"


class PostEmploymentRecord(BaseModel):
    """POST employment record"""

    post_person_nbr: str
    post_first_name: str
    post_middle_name: Optional[str] = None
    post_last_name: str
    post_suffix: Optional[str] = None
    post_agency_name: str
    post_agency_type: AgencyType = AgencyType.POLICE
    post_start_date: Optional[datetime] = None
    post_end_date: Optional[datetime] = None
    post_separation_reason: Optional[str] = None
    state: Optional[str] = None
    county: Optional[str] = None  
    

class OfficerMention(BaseModel):
    """Officer mention from incident reports"""

    mention_uid: str
    mention_agency_type: AgencyType = AgencyType.POLICE
    mention_incident_date: date
    mention_first_name: str = None
    mention_middle_name: Optional[str] = None
    mention_suffix: Optional[str] = None
    mention_last_name: str
    mention_rank: Optional[str] = None
    mention_agency: Optional[str] = None
    state: Optional[str] = None
    mentioned_agencies: Optional[str] = ""


class EntityMatch(BaseModel):
    """Entity resolution match result"""

    mention: OfficerMention
    post_record: PostEmploymentRecord
    match_probability: float = Field(ge=0.0, le=1.0)
    classifier_used: str


class CandidateQuery(BaseModel):
    """Query parameters for candidate generation"""

    first_name: Optional[str] = None
    last_name: Optional[str] = None
    agency_type: AgencyType = AgencyType.POLICE
    agency: Optional[str] = None
    start_year: Optional[int] = None
    end_year: Optional[int] = None
    state: Optional[str] = None
    limit: Optional[int] = None
    offset: int = 0


class BatchCandidatesRequest(BaseModel):
    """Request for batch candidate generation"""

    mentions: List[OfficerMention]


class BatchNameUniquenessRequest(BaseModel):
    """Request model for batch name uniqueness checking.

    Accepts a list of [first_name, last_name] pairs and returns
    the count of unique POST person numbers for each name combination.
    """
    names: List[List[str]] = Field(
        ...,
        description="List of [first_name, last_name] pairs to check for uniqueness",
        example=[["John", "Smith"], ["Jane", "Doe"], ["Michael", "Johnson"]]
    )

    @field_validator('names')
    @classmethod
    def validate_names(cls, v):
        """Validate that each name pair has exactly 2 elements."""
        for name_pair in v:
            if len(name_pair) != 2:
                raise ValueError(f"Each name pair must have exactly 2 elements (first_name, last_name), got: {name_pair}")
            if not all(isinstance(n, str) for n in name_pair):
                raise ValueError(f"All name components must be strings, got: {name_pair}")
        return v


class BatchNameUniquenessResponse(BaseModel):
    """Response model for batch name uniqueness checking.

    Returns a mapping of "FirstName|LastName" to the count of unique
    POST person numbers with that exact name combination.
    """
    name_counts: Dict[str, int] = Field(
        ...,
        description="Mapping of 'FirstName|LastName' to count of unique officers with that name",
        example={"John|Smith": 15, "Jane|Doe": 3, "Michael|Johnson": 8}
    )
    processing_time_ms: Optional[float] = Field(
        None,
        description="Time taken to process the batch request in milliseconds"
    )
