import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi import FastAPI, HTTPException, Query, Depends
from fastapi.middleware.cors import CORSMiddleware
from typing import List, Optional, Dict
import uvicorn
import time

from database.src import SupabaseClient
from config import SUPABASE_URL, SUPABASE_KEY
from models.src import (
    PostEmploymentRecord,
    CandidateQuery,
    AgencyType,
    BatchNameUniquenessRequest,
    BatchNameUniquenessResponse
)
import datetime

app = FastAPI(
    title="POST Employment Data API",
    description="API for accessing POST officer employment records",
    version="1.0.0",
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

db_client = SupabaseClient(SUPABASE_URL, SUPABASE_KEY)


@app.get("/")
async def root():
    """Health check endpoint"""
    return {"message": "POST Employment Data API", "status": "running"}


@app.get("/post/employment", response_model=List[PostEmploymentRecord])
async def get_post_employment_records(
    limit: Optional[int] = Query(None, description="Limit number of records returned"),
    offset: Optional[int] = Query(0, description="Offset for pagination"),
    first_name: Optional[str] = Query(None, description="Filter by first name"),
    last_name: Optional[str] = Query(None, description="Filter by last name"),
    agency: Optional[str] = Query(None, description="Filter by agency name"),
    state: Optional[str] = Query(None, description="Filter by state"),
):
    """
    Get POST employment records with structured name fields.
    This endpoint returns the actual POST employment data needed for entity resolution.
    """
    try:
        query = CandidateQuery(
            first_name=first_name,
            last_name=last_name,
            agency=agency,
            state=state,
            limit=limit,
            offset=offset,
        )
        return db_client.get_post_employment_records(query)
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Failed to fetch POST employment records: {str(e)}"
        )


@app.get("/post/employment/count")
async def get_post_employment_count(
    first_name: Optional[str] = Query(None, description="Filter by first name"),
    last_name: Optional[str] = Query(None, description="Filter by last name"),
    agency: Optional[str] = Query(None, description="Filter by agency name"),
    state: Optional[str] = Query(None, description="Filter by state"),
):
    """Get total count of POST employment records with optional filters"""
    try:
        query = CandidateQuery(
            first_name=first_name, last_name=last_name, agency=agency, state=state
        )
        count = db_client.get_post_employment_count(query)
        return {"total_records": count}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get count: {str(e)}")


@app.get("/post/stats")
async def get_post_stats():
    """Get basic statistics about POST employment data"""
    try:
        return db_client.get_post_stats()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Stats retrieval failed: {str(e)}")


@app.get("/post/candidates", response_model=List[PostEmploymentRecord])
async def get_candidates_for_mention(
    first_name: str = Query(..., description="Officer first name"),
    last_name: str = Query(..., description="Officer last name"),
    agency_type: AgencyType = Query(AgencyType.POLICE, description="Agency type"),
    start_year: int = Query(..., description="Incident start year"),
    end_year: int = Query(..., description="Incident end year"),
    state: Optional[str] = Query(None, description="Filter by state"),
):
    """Get targeted candidates for entity resolution"""
    try:
        query = CandidateQuery(
            first_name=first_name,
            last_name=last_name,
            agency_type=agency_type,
            start_year=start_year,
            end_year=end_year,
            state=state,
        )
        return db_client.get_candidates_for_mention(query)
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Failed to get candidates: {str(e)}"
        )
    
@app.get("/post/agency/county")
async def get_county_for_agency(
    agency_name: str = Query(..., description="Agency name to lookup")
):
    """Get the county for a given agency name"""
    try:
        county = db_client.get_county_for_agency(agency_name)
        return {"agency_name": agency_name, "county": county}
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Failed to get county for agency: {str(e)}"
        )
    
@app.get("/post/officers/by-name", response_model=List[PostEmploymentRecord])
async def get_officers_by_name(
    first_name: str = Query(..., description="Officer first name"),
    last_name: str = Query(..., description="Officer last name"),
):
    """
    Get all officers with matching first/last name across the entire database.
    Returns all records regardless of middle name, suffix, or agency.
    Uses prefix matching on last name to catch suffixes like JR, SR, II, etc.
    
    This endpoint is useful for manual review to find all potential matches
    with the same name, including variations in middle names and suffixes.
    """
    try:
        records = db_client.get_officers_by_name(
            first_name=first_name,
            last_name=last_name
        )
        return records
    except Exception as e:
        # Write detailed error to file
        import traceback
        from datetime import datetime
        with open("api_error_log.txt", "a") as f:
            f.write(f"\n{'='*80}\n")
            f.write(f"Error at {datetime.now()}\n")
            f.write(f"Endpoint: /post/officers/by-name\n")
            f.write(f"Params: first_name={first_name}, last_name={last_name}\n")
            f.write(f"Error: {str(e)}\n")
            f.write(f"Traceback:\n{traceback.format_exc()}\n")
        
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get officers by name: {str(e)}"
        )


@app.post(
    "/post/officers/batch-name-uniqueness",
    response_model=BatchNameUniquenessResponse,
    summary="Batch check name uniqueness",
    description="""
    Check how many unique officers exist for multiple name combinations.

    This endpoint accepts a list of [first_name, last_name] pairs and returns
    the count of unique POST person numbers for each name. This is useful for
    identifying common names that may require manual review in entity resolution.

    **Example use case**: Before auto-matching officers in the entity resolution
    pipeline, check if names like "John Smith" appear 15+ times, indicating a
    common name that should be flagged for manual review.

    **Performance**: Processes names in batch using hash-based partitioning for
    O(n+m) complexity instead of O(n×m).
    """,
    tags=["POST Data - Batch Operations"]
)
async def batch_check_name_uniqueness(
    request: BatchNameUniquenessRequest
) -> BatchNameUniquenessResponse:
    """
    Get the count of unique officers for multiple name combinations.

    Args:
        request: BatchNameUniquenessRequest containing list of [first_name, last_name] pairs

    Returns:
        BatchNameUniquenessResponse with name_counts dict mapping "FirstName|LastName" to count

    Example:
        Request: {"names": [["John", "Smith"], ["Jane", "Doe"]]}
        Response: {"name_counts": {"John|Smith": 15, "Jane|Doe": 3}, "processing_time_ms": 45.2}
    """
    start_time = time.time()

    try:
        # Call database method to get uniqueness counts
        name_counts_dict = db_client.get_batch_name_uniqueness(request.names)

        processing_time = (time.time() - start_time) * 1000  # Convert to milliseconds

        return BatchNameUniquenessResponse(
            name_counts=name_counts_dict,
            processing_time_ms=round(processing_time, 2)
        )

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error processing batch name uniqueness check: {str(e)}"
        )


# Batch Processing Endpoints

from pydantic import BaseModel

class BatchCountyRequest(BaseModel):
    agency_names: List[str]

class BatchCountyResponse(BaseModel):
    counties: dict

class MentionRequest(BaseModel):
    first_name: str
    last_name: str
    agency_type: str
    start_year: int
    end_year: int
    state: Optional[str] = None

class BatchCandidatesRequest(BaseModel):
    mentions: List[MentionRequest]

class BatchCandidatesResponse(BaseModel):
    results: dict
    metadata: Optional[dict] = None


@app.post("/post/agencies/counties/batch", response_model=BatchCountyResponse)
async def batch_get_counties(request: BatchCountyRequest):
    """
    Batch lookup counties for multiple agencies in a single request.

    This endpoint is optimized for bulk operations, reducing network overhead
    by fetching all county mappings in one request instead of individual calls.

    Args:
        request: BatchCountyRequest containing list of agency names

    Returns:
        BatchCountyResponse with dict mapping agency_name -> county
    """
    try:
        counties = db_client.get_counties_for_agencies_batch(request.agency_names)
        return BatchCountyResponse(counties=counties)
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to batch fetch counties: {str(e)}"
        )


@app.post("/post/candidates/batch", response_model=BatchCandidatesResponse)
async def batch_get_candidates(request: BatchCandidatesRequest):
    """
    Batch candidate generation for multiple officer mentions in a single request.

    Significantly faster than sequential requests for entity resolution pipelines.
    Processes all mentions in parallel and returns candidates indexed by position.

    Args:
        request: BatchCandidatesRequest containing list of mention dictionaries

    Returns:
        BatchCandidatesResponse with results dict (mention_idx -> candidates)
        and metadata about the batch operation
    """
    try:
        # Convert Pydantic models to dicts for database layer
        mentions_dicts = [mention.dict() for mention in request.mentions]

        # Get candidates for all mentions
        results = db_client.get_candidates_for_mentions_batch(mentions_dicts)

        # Calculate metadata
        total_candidates = sum(len(candidates) for candidates in results.values())
        mentions_with_candidates = sum(1 for candidates in results.values() if candidates)

        return BatchCandidatesResponse(
            results=results,
            metadata={
                "total_mentions": len(request.mentions),
                "total_candidates": total_candidates,
                "mentions_with_candidates": mentions_with_candidates,
            }
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to batch fetch candidates: {str(e)}"
        )


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
