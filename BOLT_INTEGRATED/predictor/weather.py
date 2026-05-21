"""
Fetch hourly ambient temperature + shortwave irradiance from Open-Meteo.

Why Open-Meteo over OpenWeatherMap:
  - No API key required (OWM free tier blocks historical data anyway).
  - Free historical archive back to 1940 (archive-api.open-meteo.com).
  - Free 16-day forecast (api.open-meteo.com).
  - timezone=auto infers from lat/lon, so 'timestamp' values come back
    in the site's local time — matching how training data is indexed.

The module caches every response to <package>/../data/weather_cache/ keyed by
(provider, lat, lon, date-range). Repeat training runs are a no-op
network-wise.

Failure mode: any HTTP / parse error returns None from get_temps_30min /
get_irradiance_30min, and the caller (features.add_temperature_features /
features.add_solar_features) falls back to the synthetic models so training
never breaks because of weather.
"""
from __future__ import annotations
import json
from pathlib import Path
from typing import Optional
import urllib.request
import urllib.error
import pandas as pd

# Cache lives next to BOLT_INTEGRATED/data/ at runtime (created on first write).
CACHE_DIR = Path(__file__).resolve().parent.parent / "data" / "weather_cache"

ARCHIVE_URL  = "https://archive-api.open-meteo.com/v1/archive"
FORECAST_URL = "https://api.open-meteo.com/v1/forecast"

# Archive lags by ~2 days. Beyond that we hit the forecast endpoint instead.
_ARCHIVE_LAG_DAYS = 2


def _cache_path(kind: str, lat: float, lon: float, start: str, end: str) -> Path:
    return CACHE_DIR / f"{kind}_{lat:.4f}_{lon:.4f}_{start}_{end}.json"


def _http_get_json(url: str, timeout: int = 20) -> dict:
    req = urllib.request.Request(url, headers={"User-Agent": "BOLT-Predictor/1.0"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read())


def _fetch_block(
    base_url: str, kind: str,
    lat: float, lon: float,
    start: pd.Timestamp, end: pd.Timestamp,
    timezone: str,
) -> pd.DataFrame:
    """Fetch a date range's hourly weather. Returns DataFrame with columns:
    timestamp, temp_c, shortwave_w_m2. shortwave_radiation is a free proxy
    for cloud cover (drops on overcast days) — operationally the biggest
    win for solar sites because their net load is dominated by it."""
    start_str = start.strftime("%Y-%m-%d")
    end_str   = end.strftime("%Y-%m-%d")
    empty = pd.DataFrame({"timestamp": [], "temp_c": [], "shortwave_w_m2": []})
    if start > end:
        return empty

    cache_p = _cache_path(kind, lat, lon, start_str, end_str)
    if cache_p.is_file():
        data = json.loads(cache_p.read_text())
    else:
        # Request both temperature and shortwave_radiation in one call.
        # Open-Meteo bundles them in a single hourly block free of charge.
        url = (f"{base_url}?latitude={lat:.4f}&longitude={lon:.4f}"
               f"&start_date={start_str}&end_date={end_str}"
               f"&hourly=temperature_2m,shortwave_radiation"
               f"&timezone={timezone}")
        data = _http_get_json(url)
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        cache_p.write_text(json.dumps(data))

    hh = data.get("hourly") or {}
    ts = hh.get("time") or []
    t  = hh.get("temperature_2m") or []
    sr = hh.get("shortwave_radiation") or [None] * len(ts)
    if not ts:
        return empty
    return pd.DataFrame({
        "timestamp": pd.to_datetime(ts),
        "temp_c": t,
        "shortwave_w_m2": sr,
    })


def fetch_hourly_temps(
    lat: float, lon: float,
    start: pd.Timestamp, end: pd.Timestamp,
    timezone: str = "auto",
) -> pd.DataFrame:
    """Hourly temps in site-local time, [start, end] inclusive."""
    today = pd.Timestamp.utcnow().tz_localize(None).normalize()
    archive_end = today - pd.Timedelta(days=_ARCHIVE_LAG_DAYS)

    parts: list[pd.DataFrame] = []
    if start <= archive_end:
        parts.append(_fetch_block(
            ARCHIVE_URL, "archive", lat, lon,
            start, min(end, archive_end), timezone,
        ))
    if end > archive_end:
        fc_start = max(start, archive_end + pd.Timedelta(days=1))
        parts.append(_fetch_block(
            FORECAST_URL, "forecast", lat, lon,
            fc_start, end, timezone,
        ))

    parts = [p for p in parts if not p.empty]
    if not parts:
        return pd.DataFrame({"timestamp": [], "temp_c": []})
    df = (pd.concat(parts, ignore_index=True)
            .drop_duplicates(subset="timestamp")
            .sort_values("timestamp")
            .reset_index(drop=True))
    return df


def _get_weather_30min(
    timestamps: pd.Series,
    lat: float, lon: float,
    column: str,
    timezone: str = "auto",
) -> Optional[pd.Series]:
    """Generic: fetch one hourly column from Open-Meteo and interpolate to
    the requested 30-min timestamps. Returns None on failure so the caller
    falls back to a synth model and training never breaks."""
    if timestamps is None or len(timestamps) == 0:
        return None
    try:
        ts_min = pd.to_datetime(timestamps.min())
        ts_max = pd.to_datetime(timestamps.max())
        start = (ts_min - pd.Timedelta(days=1)).normalize()
        end   = (ts_max + pd.Timedelta(days=1)).normalize()
        hourly = fetch_hourly_temps(lat, lon, start, end, timezone=timezone)
    except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, OSError, ValueError):
        return None
    if hourly.empty or column not in hourly.columns:
        return None

    h = hourly.set_index("timestamp")
    h.index = pd.to_datetime(h.index)
    target_idx = pd.DatetimeIndex(pd.to_datetime(timestamps.values))

    union = h.index.union(target_idx).sort_values()
    interp = (h.reindex(union)
                .interpolate(method="time")
                .reindex(target_idx)[column])
    interp = pd.Series(interp.values, index=timestamps.index, dtype=float)
    if interp.isna().all():
        return None
    interp = interp.ffill().bfill()
    return interp


def get_temps_30min(
    timestamps: pd.Series,
    lat: float, lon: float,
    timezone: str = "auto",
) -> Optional[pd.Series]:
    """Ambient temperature (°C) aligned to the requested 30-min timestamps."""
    return _get_weather_30min(timestamps, lat, lon, "temp_c", timezone)


def get_irradiance_30min(
    timestamps: pd.Series,
    lat: float, lon: float,
    timezone: str = "auto",
) -> Optional[pd.Series]:
    """Shortwave radiation (W/m²) at the surface — Open-Meteo's free proxy
    for cloud cover. Drops sharply on overcast days, follows the sun curve
    on clear days. The single most useful feature for solar-site forecasts."""
    return _get_weather_30min(timestamps, lat, lon, "shortwave_w_m2", timezone)
