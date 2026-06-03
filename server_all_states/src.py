"""
ISOLATED FastAPI server for the all_npi_states table (synced from Firestore).

Mirrors the postie server's endpoint contract so the entity-res pipeline can point at
EITHER API via the NPI_API_URL env var. The postie server (server/src.py, port 8000)
is left completely untouched. This one defaults to port 8001.

Run:
    cd server_all_states && python3 src.py
    # or:  NPI_ALL_STATES_PORT=8002 python3 server_all_states/src.py
"""
import sys
import os
import time

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from typing import List, Optional
from pydantic import BaseModel
import uvicorn

from database import AllStatesClient
from config import SUPABASE_URL, SUPABASE_KEY, TABLE_NAME, API_TITLE, API_VERSION, PORT
from models.src import (
    PostEmploymentRecord, CandidateQuery, AgencyType,
    BatchNameUniquenessRequest, BatchNameUniquenessResponse,
)

app = FastAPI(title=API_TITLE, description="Isolated API over all_npi_states", version=API_VERSION)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True,
                   allow_methods=["*"], allow_headers=["*"])

db_client = AllStatesClient(SUPABASE_URL, SUPABASE_KEY, table=TABLE_NAME)


@app.get("/")
async def root():
    return {"message": API_TITLE, "table": TABLE_NAME, "status": "running"}


@app.get("/post/employment", response_model=List[PostEmploymentRecord])
async def get_post_employment_records(
    limit: Optional[int] = Query(None),
    offset: Optional[int] = Query(0),
    first_name: Optional[str] = Query(None),
    last_name: Optional[str] = Query(None),
    agency: Optional[str] = Query(None),
    state: Optional[str] = Query(None),
):
    try:
        q = CandidateQuery(first_name=first_name, last_name=last_name, agency=agency,
                           state=state, limit=limit, offset=offset)
        return db_client.get_post_employment_records(q)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to fetch employment records: {e}")


@app.get("/post/employment/count")
async def get_post_employment_count(
    first_name: Optional[str] = Query(None),
    last_name: Optional[str] = Query(None),
    agency: Optional[str] = Query(None),
    state: Optional[str] = Query(None),
):
    try:
        q = CandidateQuery(first_name=first_name, last_name=last_name, agency=agency, state=state)
        return {"total_records": db_client.get_post_employment_count(q)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get count: {e}")


@app.get("/post/stats")
async def get_post_stats():
    try:
        return db_client.get_post_stats()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Stats retrieval failed: {e}")


@app.get("/post/candidates", response_model=List[PostEmploymentRecord])
async def get_candidates_for_mention(
    first_name: str = Query(...),
    last_name: str = Query(...),
    agency_type: AgencyType = Query(AgencyType.POLICE),
    start_year: int = Query(...),
    end_year: int = Query(...),
    state: Optional[str] = Query(None),
):
    try:
        q = CandidateQuery(first_name=first_name, last_name=last_name, agency_type=agency_type,
                           start_year=start_year, end_year=end_year, state=state)
        return db_client.get_candidates_for_mention(q)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get candidates: {e}")


@app.get("/post/agency/county")
async def get_county_for_agency(agency_name: str = Query(...)):
    """county is not available in all_npi_states; always returns null."""
    return {"agency_name": agency_name, "county": db_client.get_county_for_agency(agency_name)}


@app.get("/post/officers/by-name", response_model=List[PostEmploymentRecord])
async def get_officers_by_name(first_name: str = Query(...), last_name: str = Query(...),
                               state: Optional[str] = Query(None, description="Optional state filter (code or name)")):
    try:
        return db_client.get_officers_by_name(first_name=first_name, last_name=last_name, state=state)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get officers by name: {e}")


@app.post("/post/officers/batch-name-uniqueness", response_model=BatchNameUniquenessResponse)
async def batch_check_name_uniqueness(request: BatchNameUniquenessRequest) -> BatchNameUniquenessResponse:
    start = time.time()
    try:
        counts = db_client.get_batch_name_uniqueness(request.names)
        return BatchNameUniquenessResponse(name_counts=counts,
                                           processing_time_ms=round((time.time() - start) * 1000, 2))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error in batch name uniqueness: {e}")


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
    return BatchCountyResponse(counties=db_client.get_counties_for_agencies_batch(request.agency_names))


@app.post("/post/candidates/batch", response_model=BatchCandidatesResponse)
async def batch_get_candidates(request: BatchCandidatesRequest):
    try:
        results = db_client.get_candidates_for_mentions_batch([m.dict() for m in request.mentions])
        total = sum(len(v) for v in results.values())
        with_c = sum(1 for v in results.values() if v)
        return BatchCandidatesResponse(results=results, metadata={
            "total_mentions": len(request.mentions), "total_candidates": total,
            "mentions_with_candidates": with_c})
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to batch fetch candidates: {e}")


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=PORT)
