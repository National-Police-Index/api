# NPI Employment Data API

A REST API and interactive search tool for accessing POST (Peace Officer Standards and Training) officer employment records through the National Police Index. D

## Quick Start

### 1. Setup Environment

```bash
# Install dependencies
pip install fastapi uvicorn supabase python-dotenv requests pandas

# Create .env file with your Supabase key
echo "SUPABASE_KEY=your_supabase_key_here" > .env
```

### 2. Start the API Server

```bash
# Terminal 1: Start the API server
cd server
python3 src.py
```

The API will be available at `http://localhost:8000`

## Features

### API Endpoints
- **POST Employment Records**: Retrieve officer employment records with structured name fields
- **Employment Count**: Get total number of employment records
- **Employment Statistics**: View database statistics and employment distributions
- **Health Check**: Verify API status

### Search Tool
- Interactive terminal interface for browsing POST employment data
- Search by first name, last name, or agency
- View detailed employment records with structured name fields
- Navigate through employment history and separation details

### Entity Resolution Support
- Optimized for entity resolution pipelines
- Structured name fields (first_name, middle_name, last_name, suffix)
- Efficient querying by name components
- Compatible with machine learning matching algorithms

## API Usage

### Get Employment Records
```bash
# Basic name searches
curl "http://localhost:8000/post/employment?last_name=Smith"
curl "http://localhost:8000/post/employment?first_name=John&last_name=Smith"

# State-based filtering
curl "http://localhost:8000/post/employment?state=CA&limit=10"
curl "http://localhost:8000/post/employment?first_name=Robert&state=MA"

# Agency filtering
curl "http://localhost:8000/post/employment?agency=Los Angeles Police Department"
curl "http://localhost:8000/post/employment?agency=Police&state=NY"

# Pagination and limits
curl "http://localhost:8000/post/employment?limit=100&offset=0"
curl "http://localhost:8000/post/employment?limit=50&offset=100"

# Combined filtering (multiple criteria)
curl "http://localhost:8000/post/employment?first_name=John&last_name=Smith&state=CA&limit=20"

# Basic candidate search for entity resolution
curl "http://localhost:8000/post/candidates?first_name=Robert&last_name=Smith&agency_type=POLICE&start_year=2018&end_year=2020"

# Candidates with state filter for entity res
curl "http://localhost:8000/post/candidates?first_name=Robert&last_name=Smith&agency_type=POLICE&start_year=2018&end_year=2020&state=CA"

# Different agency types
curl "http://localhost:8000/post/candidates?first_name=John&last_name=Doe&agency_type=CORRECTIONS&start_year=2015&end_year=2023"
```

### Get Employment Count
```bash
# Total count
curl "http://localhost:8000/post/employment/count"

# Filtered counts
curl "http://localhost:8000/post/employment/count?state=CA"
curl "http://localhost:8000/post/employment/count?first_name=Robert&state=NY"
curl "http://localhost:8000/post/employment/count?agency=Police"
```

### View Statistics
```bash
curl "http://localhost:8000/post/stats"
```

## Data Structure

Employment records contain the following structured fields:

- **post_person_nbr**: Unique POST identifier
- **post_first_name**: Officer's first name
- **post_middle_name**: Officer's middle name
- **post_last_name**: Officer's last name
- **post_suffix**: Name suffix (Jr., Sr., etc.)
- **post_agency_name**: Name of employing agency
- **post_agency_type**: Type of agency (typically "POLICE")
- **post_start_date**: Employment start date
- **post_end_date**: Employment end date (null if current)
- **post_separation_reason**: Reason for separation if applicable
- **post_state**: Reason for separation if applicable

## Project Structure

```
api/
├── server/
│   ├── src.py          # FastAPI server
│   └── config.py       # Configuration settings
├── search/
│   └── src.py          # Interactive search tool
├── database/
│   └── src.py          # Database client
├── test/
│   └── src.py          # API test script
├── resolve/
│   ├── src/
│   │   ├── match.py    # Entity resolution pipeline
│   │   └── api.py      # API client for entity resolution
│   └── data/
│       ├── input/      # Input data for matching
│       └── output/     # Matching results
└── .env                # Environment variables
```

## Entity Resolution Pipeline

The API supports an entity resolution pipeline for matching officer mentions to POST employment records:

```bash
# Run entity resolution
cd resolve/src
python3 match.py
```

The pipeline:
1. Reads officer mentions from input CSV
2. Fetches targeted POST employment records by last name (efficient API usage)
3. Applies machine learning models to match mentions to employment records
4. Outputs matched results with confidence scores

## Documentation

- **API Documentation**: Available at `http://localhost:8000/docs` when server is running
- **OpenAPI Schema**: Available at `http://localhost:8000/openapi.json`

## Requirements

- Python 3.7+
- FastAPI
- Supabase account and API key
- pandas (for entity resolution)
- scikit-learn (for entity resolution models)

## Testing the API

Run the comprehensive test script:

```bash
cd test
python3 src.py
```

The test script will:
- Verify API health check
- Test employment record search functionality
- Check employment count endpoint
- Display database statistics
- Test specific officer searches

## Troubleshooting

**Server won't start**: Check that your `.env` file contains a valid `SUPABASE_KEY`

**Search tool connection error**: Ensure the API server is running on `http://localhost:8000`

**No search results**: Try partial names or different spelling variations

**Entity resolution performance**: The pipeline fetches only relevant records by last name and state, if provided

## Example API Response

```json
[
  {
    "post_person_nbr": "B01-Y73",
    "post_first_name": "John",
    "post_middle_name": "A",
    "post_last_name": "Smith",
    "post_suffix": "",
    "post_agency_name": "LOS ANGELES POLICE DEPARTMENT",
    "post_agency_type": "POLICE",
    "post_start_date": "2003-06-30",
    "post_end_date": "2024-09-19",
    "post_separation_reason": "Retired",
    "post_state": "CA"
  }
]
```