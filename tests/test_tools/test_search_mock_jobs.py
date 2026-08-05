from tools.search_mock_jobs_tool import search_mock_jobs_tool


def test_search_mock_jobs_returns_structured_listings():
    result = search_mock_jobs_tool.run({"keyword": "Backend Engineer", "location": "Chicago"})

    assert result["keyword"] == "Backend Engineer"
    assert result["location"] == "Chicago"
    assert len(result["jobs"]) == 3
    for job in result["jobs"]:
        assert set(job) == {"job_id", "title", "company", "location"}
        assert job["location"] == "Chicago"


def test_search_mock_jobs_is_deterministic_for_the_same_inputs():
    first = search_mock_jobs_tool.run({"keyword": "Data Scientist", "location": "Austin"})
    second = search_mock_jobs_tool.run({"keyword": "Data Scientist", "location": "Austin"})

    assert first == second


def test_search_mock_jobs_differs_for_different_inputs():
    chicago = search_mock_jobs_tool.run({"keyword": "Data Scientist", "location": "Chicago"})
    austin = search_mock_jobs_tool.run({"keyword": "Data Scientist", "location": "Austin"})

    assert chicago["jobs"] != austin["jobs"]


def test_search_mock_jobs_returns_structured_error_on_empty_keyword():
    result = search_mock_jobs_tool.run({"keyword": "   ", "location": "Chicago"})

    assert result["error"] == "invalid_arguments"


def test_search_mock_jobs_returns_structured_error_on_empty_location():
    result = search_mock_jobs_tool.run({"keyword": "Backend Engineer", "location": ""})

    assert result["error"] == "invalid_arguments"
