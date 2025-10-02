from pydantic import BaseModel, Field
from typing import Optional, List
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
