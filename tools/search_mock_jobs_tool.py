"""Tool: returns a small set of fabricated job listings for a keyword/location.

Deterministic and fully offline — no real job board is queried. This is a
stand-in for wiring up a real job-search API later; the same
keyword/location always produces the same listings.
"""

import hashlib

from pydantic import BaseModel

from tools.base import Tool, tool_error

_COMPANIES = ["Acme Corp", "Globex", "Initech", "Umbrella Inc", "Stark Industries"]
_LEVELS = ["Junior", "Mid-level", "Senior", "Staff"]
_RESULTS_PER_SEARCH = 3


class SearchMockJobsArgs(BaseModel):
    keyword: str
    location: str


def _seed_for(keyword: str, location: str) -> int:
    digest = hashlib.sha256(f"{keyword.lower()}|{location.lower()}".encode()).hexdigest()
    return int(digest, 16)


def _search_mock_jobs(args: SearchMockJobsArgs) -> dict:
    keyword = args.keyword.strip()
    location = args.location.strip()
    if not keyword or not location:
        return tool_error("invalid_arguments", "keyword and location must not be empty.")

    seed = _seed_for(keyword, location)
    jobs = [
        {
            "job_id": f"MOCK-{(seed + i) % 100_000:05d}",
            "title": f"{_LEVELS[(seed // (i + 1)) % len(_LEVELS)]} {keyword.title()}",
            "company": _COMPANIES[(seed + i) % len(_COMPANIES)],
            "location": location,
        }
        for i in range(_RESULTS_PER_SEARCH)
    ]

    return {"keyword": keyword, "location": location, "jobs": jobs}


search_mock_jobs_tool = Tool(
    name="search_mock_jobs",
    description=(
        "Search for job postings by keyword and location. Returns fabricated "
        "listings for demo/testing purposes, not real job postings."
    ),
    args_model=SearchMockJobsArgs,
    handler=_search_mock_jobs,
)
