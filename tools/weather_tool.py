"""Tool: looks up the current weather for a city via the free Open-Meteo API.

No API key is required. On any failure (city not found, network error, bad
response) the handler returns a structured {"error": ..., "message": ...}
dict instead of raising, so the model can read it back and explain the
failure to the user rather than the whole turn crashing.
"""

import httpx
from pydantic import BaseModel

from tools.base import Tool

GEOCODING_URL = "https://geocoding-api.open-meteo.com/v1/search"
FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
REQUEST_TIMEOUT_SECONDS = 10


class GetWeatherArgs(BaseModel):
    city: str


def _error(code: str, message: str) -> dict:
    return {"error": code, "message": message}


def _get_weather(args: GetWeatherArgs) -> dict:
    try:
        geocode_response = httpx.get(
            GEOCODING_URL,
            params={"name": args.city, "count": 1},
            timeout=REQUEST_TIMEOUT_SECONDS,
        )
        geocode_response.raise_for_status()
    except httpx.HTTPError as e:
        return _error("geocoding_request_failed", f"Could not look up {args.city!r}: {e}")

    results = geocode_response.json().get("results")
    if not results:
        return _error("city_not_found", f"No location found matching {args.city!r}.")

    location = results[0]

    try:
        forecast_response = httpx.get(
            FORECAST_URL,
            params={
                "latitude": location["latitude"],
                "longitude": location["longitude"],
                "current_weather": "true",
            },
            timeout=REQUEST_TIMEOUT_SECONDS,
        )
        forecast_response.raise_for_status()
    except httpx.HTTPError as e:
        return _error("forecast_request_failed", f"Could not fetch weather for {args.city!r}: {e}")

    current = forecast_response.json().get("current_weather")
    if not current:
        return _error("forecast_unavailable", f"No current weather data available for {args.city!r}.")

    return {
        "city": location.get("name", args.city),
        "country": location.get("country"),
        "temperature_c": current["temperature"],
        "windspeed_kmh": current["windspeed"],
        "weather_code": current["weathercode"],
    }


get_weather_tool = Tool(
    name="get_weather",
    description="Get the current weather for a city (temperature in Celsius, windspeed in km/h).",
    args_model=GetWeatherArgs,
    handler=_get_weather,
)
