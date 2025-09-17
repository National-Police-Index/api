import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi import FastAPI, HTTPException, Query, Depends
from fastapi.middleware.cors import CORSMiddleware
from typing import List, Optional
import uvicorn

from database.src import SupabaseClient
from config import SUPABASE_URL, SUPABASE_KEY
from models.src import PostEmploymentRecord, CandidateQuery, AgencyType

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


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
