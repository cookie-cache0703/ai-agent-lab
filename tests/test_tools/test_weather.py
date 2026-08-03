from unittest.mock import MagicMock

import httpx
import pytest

from tools import weather_tool
from tools.weather_tool import get_weather_tool


class _FakeResponse:
    def __init__(self, json_data: dict) -> None:
        self._json_data = json_data

    def raise_for_status(self) -> None:
        pass

    def json(self) -> dict:
        return self._json_data


def _geocode_response(name="Chicago", country="United States", lat=41.85, lon=-87.65) -> _FakeResponse:
    return _FakeResponse({"results": [{"name": name, "country": country, "latitude": lat, "longitude": lon}]})


def _forecast_response(temperature=22.5, windspeed=10.0, weathercode=1) -> _FakeResponse:
    return _FakeResponse(
        {"current_weather": {"temperature": temperature, "windspeed": windspeed, "weathercode": weathercode}}
    )


def _http_error() -> httpx.HTTPError:
    return httpx.ConnectError("connection failed")


def test_get_weather_returns_structured_data_on_success(monkeypatch):
    mock_get = MagicMock(side_effect=[_geocode_response(), _forecast_response()])
    monkeypatch.setattr(weather_tool.httpx, "get", mock_get)

    result = get_weather_tool.run({"city": "Chicago"})

    assert result == {
        "city": "Chicago",
        "country": "United States",
        "temperature_c": 22.5,
        "windspeed_kmh": 10.0,
        "weather_code": 1,
    }
    assert mock_get.call_count == 2


def test_get_weather_returns_structured_error_when_city_not_found(monkeypatch):
    mock_get = MagicMock(return_value=_FakeResponse({"results": []}))
    monkeypatch.setattr(weather_tool.httpx, "get", mock_get)

    result = get_weather_tool.run({"city": "Atlantis"})

    assert result["error"] == "city_not_found"
    assert "Atlantis" in result["message"]
    assert mock_get.call_count == 1


def test_get_weather_returns_structured_error_on_geocoding_failure(monkeypatch):
    mock_get = MagicMock(side_effect=_http_error())
    monkeypatch.setattr(weather_tool.httpx, "get", mock_get)

    result = get_weather_tool.run({"city": "Chicago"})

    assert result["error"] == "geocoding_request_failed"
    assert "Chicago" in result["message"]


def test_get_weather_returns_structured_error_on_forecast_failure(monkeypatch):
    mock_get = MagicMock(side_effect=[_geocode_response(), _http_error()])
    monkeypatch.setattr(weather_tool.httpx, "get", mock_get)

    result = get_weather_tool.run({"city": "Chicago"})

    assert result["error"] == "forecast_request_failed"


@pytest.mark.parametrize("field", ["error", "message"])
def test_structured_errors_always_have_error_and_message_keys(monkeypatch, field):
    mock_get = MagicMock(return_value=_FakeResponse({"results": []}))
    monkeypatch.setattr(weather_tool.httpx, "get", mock_get)

    result = get_weather_tool.run({"city": "Nowhere"})

    assert field in result
