"""
Feature engineering for load forecasting.
Input:  normalized DataFrame from data_loader (timestamp, kw_import, kw_export, ...)
Output: DataFrame of features + target ready for LightGBM.
"""
from __future__ import annotations
import numpy as np
import pandas as pd


# Lags in units of 30-min steps (the data frequency)
DEFAULT_LAGS = [1, 2, 4, 8, 16, 24, 48, 96, 336]  # 30min, 1h, 2h, 4h, 8h, 12h, 24h, 48h, 1 week


# Malaysian fixed public holidays — plus the major lunar holidays. Lunar
# dates vary by year; we list the actual dates for 2024-2026 (the span of
# the synth datasets). For real production use, swap in the `holidays`
# package via `import holidays; my_h = holidays.MY()`.
_MY_PUBLIC_HOLIDAYS = frozenset({
    # Fixed federal
    (1, 1),    # New Year
    (2, 1),    # Federal Territory Day
    (5, 1),    # Labour Day
    (8, 31),   # Merdeka
    (9, 16),   # Malaysia Day
    (12, 25),  # Christmas
})
# Lunar/movable holidays — (year, month, day)
_MY_LUNAR_HOLIDAYS = frozenset({
    # 2024
    (2024, 1, 25), (2024, 2, 10), (2024, 2, 11), (2024, 2, 12),
    (2024, 4, 10), (2024, 4, 11), (2024, 6, 17), (2024, 7, 7),
    (2024, 10, 31), (2024, 12, 11),
    # 2025
    (2025, 1, 29), (2025, 1, 30), (2025, 3, 11), (2025, 3, 31),
    (2025, 4, 1), (2025, 5, 12), (2025, 6, 2), (2025, 6, 7), (2025, 6, 8),
    (2025, 10, 20), (2025, 10, 29),
    # 2026
    (2026, 1, 18), (2026, 2, 17), (2026, 2, 18), (2026, 2, 19),
    (2026, 3, 21), (2026, 4, 20), (2026, 5, 1), (2026, 5, 31), (2026, 5, 27),
    (2026, 10, 9), (2026, 11, 7),
})


def _is_my_holiday(ts: pd.Series) -> pd.Series:
    """Boolean series: True on Malaysian public holiday (fixed or lunar)."""
    md = list(zip(ts.dt.month.values, ts.dt.day.values))
    ymd = list(zip(ts.dt.year.values, ts.dt.month.values, ts.dt.day.values))
    return pd.Series(
        [(m in _MY_PUBLIC_HOLIDAYS) or (y in _MY_LUNAR_HOLIDAYS)
         for m, y in zip(md, ymd)],
        index=ts.index,
    )


def add_time_features(df: pd.DataFrame, ts_col: str = "timestamp") -> pd.DataFrame:
    """Cyclical and categorical time-of-day / day-of-week features.

    Includes Malaysian public-holiday flags + day-before/after — recurring
    nightly anomalies in the baseline (e.g. E's Wednesday 00:00 spikes) are
    often holiday-eve patterns the schedule features can't catch.
    """
    out = df.copy()
    ts = out[ts_col]
    out["hour"] = ts.dt.hour + ts.dt.minute / 60.0
    out["dow"] = ts.dt.dayofweek
    out["month"] = ts.dt.month
    out["is_weekend"] = (out["dow"] >= 5).astype(int)
    out["is_business_hours"] = ((out["hour"] >= 8) & (out["hour"] < 18) & (out["is_weekend"] == 0)).astype(int)
    out["hour_sin"] = np.sin(2 * np.pi * out["hour"] / 24.0)
    out["hour_cos"] = np.cos(2 * np.pi * out["hour"] / 24.0)
    out["dow_sin"] = np.sin(2 * np.pi * out["dow"] / 7.0)
    out["dow_cos"] = np.cos(2 * np.pi * out["dow"] / 7.0)

    # Holiday flags — capture eve-of and morning-after effects too. Many
    # commercial sites have a slow-down on holiday eve and a slow ramp the
    # day after, which is invisible to dow_hour_mean alone.
    is_h = _is_my_holiday(ts).astype(int)
    out["is_holiday"] = is_h.values
    # Day before / after holiday (shifted by 48 = 24h)
    out["is_pre_holiday"]  = is_h.shift(-48).fillna(0).astype(int).values
    out["is_post_holiday"] = is_h.shift(48).fillna(0).astype(int).values
    return out


def add_lag_features(df: pd.DataFrame, target: str = "kw_import", lags: list[int] = None) -> pd.DataFrame:
    """Historical lags — crucial for any time-series forecasting."""
    if lags is None:
        lags = DEFAULT_LAGS
    out = df.copy()
    for L in lags:
        out[f"lag_{L}"] = out[target].shift(L)
    return out


def add_rolling_features(df: pd.DataFrame, target: str = "kw_import") -> pd.DataFrame:
    """Rolling stats give the model a sense of recent trend + volatility."""
    out = df.copy()
    s = out[target].shift(1)
    out["roll_mean_24"]  = s.rolling(24,  min_periods=1).mean()
    out["roll_mean_48"]  = s.rolling(48,  min_periods=1).mean()
    out["roll_std_48"]   = s.rolling(48,  min_periods=1).std().fillna(0)
    out["roll_max_48"]   = s.rolling(48,  min_periods=1).max()
    out["roll_min_48"]   = s.rolling(48,  min_periods=1).min()
    out["roll_mean_336"] = s.rolling(336, min_periods=48).mean()  # 7-day average
    return out


def add_schedule_features(df: pd.DataFrame, target: str = "kw_import") -> pd.DataFrame:
    """
    Historical mean and std for each (day-of-week, hour) bucket.
    Gives the model a strong weekly schedule prior — the single most
    predictive signal for long-horizon forecasts on commercial sites.

    Adds two flavours of the schedule prior:
      - dow_hour_mean (all-history): stable but slow to adapt to regime shifts
      - same_dow_hour_4w_mean (last 4 weeks): adapts to recent operating
        changes (new tenant, equipment swap, seasonal schedule). Catches
        anomalies like E's recurring Wednesday-night spikes when they only
        started a few months in.
    """
    out = df.copy()
    hour_bin = out["timestamp"].dt.hour
    dow = out["timestamp"].dt.dayofweek
    # 168 unique keys (7 days × 24 hours)
    out["_sched_key"] = dow * 24 + hour_bin
    out["dow_hour_mean"] = out.groupby("_sched_key")[target].transform("mean")
    out["dow_hour_std"]  = out.groupby("_sched_key")[target].transform("std").fillna(0)

    # 4-week rolling same-DOW-same-hour mean. Each (dow, hour) bucket has
    # one sample per week (~24 over the typical training window), so a
    # 4-sample rolling mean = "last 4 same weekday-hours". Use the SHIFTED
    # series so the current row isn't included (no leakage).
    s_lag = out[target].shift(48)   # shift one day so we never include now
    # Group-aware rolling: within each (dow, hour) bucket, take 4-sample mean
    out["same_dow_hour_4w_mean"] = (
        s_lag.groupby(out["_sched_key"])
             .transform(lambda x: x.rolling(4, min_periods=1).mean())
    )
    out["same_dow_hour_4w_mean"] = out["same_dow_hour_4w_mean"].fillna(out["dow_hour_mean"])

    out.drop(columns=["_sched_key"], inplace=True)
    return out


def add_regime_features(df: pd.DataFrame, target: str = "kw_import") -> pd.DataFrame:
    """
    Regime-shift signals: tell the trees how today compares to history.

    Without these, on bimodal sites (some days busy, some quiet), the model's
    median prediction collapses to the historical mean of the dow_hour bucket,
    which is between the two regimes — so it over-predicts on quiet days and
    under-predicts on busy ones. The deltas + recent-vs-schedule give a
    splittable feature that says "we are in a quiet/loud regime right now",
    which the trees can use to pick the appropriate leaf.
    """
    out = df.copy()
    s = out[target]
    out["delta_vs_24h"] = s.shift(1) - s.shift(48)
    out["delta_vs_7d"]  = s.shift(1) - s.shift(336)
    # Recent rolling mean vs the long-run dow_hour baseline.
    # Both inputs are already shifted-1 (rolling) and historical (groupby
    # transform), so no current-row leakage.
    if "roll_mean_24" in out.columns and "dow_hour_mean" in out.columns:
        out["recent_vs_schedule"] = out["roll_mean_24"] - out["dow_hour_mean"]
    return out


# ---- Tropical temperature model (Malaysia) ----------------------------------
# Half-hourly temperature (°C) for a typical Malaysian day.
# Sinusoidal: mean 29 °C, amplitude 4 °C, trough ~06:00 (hh=12), peak ~14:00 (hh=28).
_HH = np.arange(48)
_TROPICAL_TEMP_C = 29.0 + 4.0 * np.sin(2 * np.pi * (_HH - 12) / 48.0)


def estimated_temp_c(ts: pd.Series) -> pd.Series:
    """Synthetic ambient temperature (°C) based on time-of-day for tropical Malaysia."""
    hh = (ts.dt.hour * 2 + ts.dt.minute // 30).astype(int) % 48
    return pd.Series(_TROPICAL_TEMP_C[hh.values], index=ts.index)


def add_temperature_features(
    df: pd.DataFrame,
    temps: pd.Series | None = None,
) -> pd.DataFrame:
    """Add ambient temperature + 'feels hot' flag.

    If `temps` is provided (real measurements from weather.get_temps_30min),
    it MUST be aligned to df['timestamp'] index-wise. NaNs are filled from
    the synthetic tropical fallback so feature rows are never dropped.

    The column name stays `est_temp_c` for save/load backward-compat with
    forecasters trained before weather was wired in.
    """
    out = df.copy()
    synth = estimated_temp_c(out["timestamp"])
    if temps is not None:
        real = pd.Series(temps.values, index=out.index, dtype=float)
        out["est_temp_c"] = real.where(real.notna(), synth)
    else:
        out["est_temp_c"] = synth
    out["is_hot_period"] = (out["est_temp_c"] > 31.0).astype(int)
    return out


# ---- Solar generation model -------------------------------------------------
_SOLAR_CURVE_PER_HH = np.array([
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0.01, 0.03,
    0.06, 0.09, 0.13, 0.17, 0.20, 0.22, 0.24, 0.25,
    0.25, 0.24, 0.23, 0.21,
    0.19, 0.16, 0.12, 0.08, 0.05, 0.03, 0.02, 0.01,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
], dtype=float)
_SOLAR_CURVE_PER_HH = _SOLAR_CURVE_PER_HH * (4.0 / _SOLAR_CURVE_PER_HH.sum())


def estimated_solar_gen_kw(ts: pd.Series, capacity_kwp: float) -> pd.Series:
    """Estimate instantaneous solar kW output at each timestamp given installed capacity."""
    hh = (ts.dt.hour * 2 + ts.dt.minute // 30).astype(int)
    kwh_per_hh = pd.Series(_SOLAR_CURVE_PER_HH[hh.values], index=ts.index)
    return kwh_per_hh * capacity_kwp * 2.0


def add_solar_features(
    df: pd.DataFrame,
    capacity_kwp: float,
    irradiance_w_m2: pd.Series | None = None,
) -> pd.DataFrame:
    """Append solar-related features. If `irradiance_w_m2` is provided
    (real W/m² from Open-Meteo shortwave_radiation), also include:

      - solar_irrad_w_m2:        real measured/forecast irradiance
      - solar_irrad_ratio:       real / theoretical (a 1.0 = clear sky;
                                 ~0.2 = heavy overcast). This is the
                                 cloud-cover proxy the model actually sees
                                 as predictive — it's normalized so
                                 LightGBM can split on "clear vs cloudy"
                                 regardless of absolute hour-of-day.

    Falls back gracefully to synth when irradiance_w_m2 is None.
    """
    out = df.copy()
    out["solar_capacity_kwp"] = float(capacity_kwp)
    out["est_solar_gen_kw"] = estimated_solar_gen_kw(out["timestamp"], capacity_kwp)
    out["has_solar"] = int(capacity_kwp > 0)
    if irradiance_w_m2 is not None:
        irr = pd.Series(irradiance_w_m2.values, index=out.index, dtype=float)
        out["solar_irrad_w_m2"] = irr.fillna(0.0).clip(lower=0.0)
        # Theoretical clear-sky irradiance at this hour (tropical max ≈ 1000)
        # — same shape as _SOLAR_CURVE_PER_HH but normalized to W/m² scale.
        hh = (out["timestamp"].dt.hour * 2 + out["timestamp"].dt.minute // 30).astype(int)
        theoretical = _SOLAR_CURVE_PER_HH[hh.values] * 4000.0   # peak ≈ 1000 W/m²
        # ratio = 1 on clear day, ~0 on heavy overcast. Clip + safe divide.
        with np.errstate(divide="ignore", invalid="ignore"):
            ratio = np.where(theoretical > 50, out["solar_irrad_w_m2"].values / theoretical, 1.0)
        out["solar_irrad_ratio"] = np.clip(ratio, 0.0, 1.5)
    else:
        out["solar_irrad_w_m2"] = 0.0
        out["solar_irrad_ratio"] = 1.0
    return out


# Horizons (in 30-min intervals) at which we expose future-weather snapshots
# to every per-horizon booster.  Picked to span the operationally relevant
# range: 1h, 3h, 6h, 12h, 24h ahead.
FUTURE_WEATHER_HORIZONS = (2, 6, 12, 24, 48)

# Lookahead window for the future-weather *rolling* aggregates (in intervals).
# 6h, 12h, 24h forward windows.
FUTURE_WEATHER_WINDOWS = (12, 24, 48)


def add_future_weather_features(
    out: pd.DataFrame,
    extended_temp_c: pd.Series | None,
    extended_irrad_w_m2: pd.Series | None,
    capacity_kwp: float,
) -> pd.DataFrame:
    """Add forward-looking weather features so each per-horizon model can
    answer 'what will conditions be h intervals from now?'.

    Columns added (every row, regardless of weather source — shape stable):
      fut_irrad_h{H}        irradiance W/m² at t+H intervals
      fut_temp_h{H}         ambient °C       at t+H intervals
      fut_irrad_{W*30}min_mean   rolling mean of irradiance over [t+1, t+W]
      fut_temp_{W*30}min_max     rolling MAX  of temp        over [t+1, t+W]
      fut_irrad_delta_h12        irrad now − irrad at t+12  (regime-change cue)

    Real-weather path (when extended_irrad_w_m2 / extended_temp_c are non-None,
    each indexed by the same row numbers as `out`) reads the actual forecast
    from Open-Meteo.

    Synth fallback path (otherwise) uses the deterministic tropical
    temperature sinusoid + clear-sky irradiance curve evaluated at t+H.
    Both paths produce IDENTICAL column names so saved boosters remain
    compatible across runs.
    """
    n = len(out)
    ts = out["timestamp"]

    # ---- Helpers to fetch synth values at offset horizons (deterministic) ----
    def _synth_temp_at_offset(h: int) -> pd.Series:
        future_ts = ts + pd.Timedelta(minutes=30 * h)
        return estimated_temp_c(future_ts)

    def _synth_irrad_at_offset(h: int) -> pd.Series:
        # Convert synth solar GENERATION (kW) into an equivalent IRRAD (W/m²)
        # using the clear-sky curve at 100 % cap (peak ≈ 1000 W/m²).
        future_ts = ts + pd.Timedelta(minutes=30 * h)
        hh = (future_ts.dt.hour * 2 + future_ts.dt.minute // 30).astype(int)
        return pd.Series(_SOLAR_CURVE_PER_HH[hh.values] * 4000.0, index=ts.index)

    have_real = (extended_temp_c is not None and extended_irrad_w_m2 is not None
                 and len(extended_temp_c) >= n and len(extended_irrad_w_m2) >= n)

    # ---- Per-horizon snapshots ------------------------------------------------
    for h in FUTURE_WEATHER_HORIZONS:
        if have_real and len(extended_irrad_w_m2) >= n + h:
            irrad_h = extended_irrad_w_m2.shift(-h).iloc[:n].reset_index(drop=True)
            temp_h  = extended_temp_c.shift(-h).iloc[:n].reset_index(drop=True)
            # Forward-fill the very last edge if any NaN slipped in
            irrad_h = irrad_h.fillna(method="ffill").fillna(0.0)
            temp_h  = temp_h.fillna(method="ffill").fillna(_synth_temp_at_offset(h))
            out[f"fut_irrad_h{h}"] = irrad_h.values
            out[f"fut_temp_h{h}"]  = temp_h.values
        else:
            out[f"fut_irrad_h{h}"] = _synth_irrad_at_offset(h).values
            out[f"fut_temp_h{h}"]  = _synth_temp_at_offset(h).values

    # ---- Rolling future-window aggregates ------------------------------------
    # At row index i, fut_irrad_{W*30}min_mean = mean of irradiance at
    # intervals [i+1, i+W].  Implemented via right-aligned rolling on the
    # extended series, then shifted backward by W.
    for w in FUTURE_WEATHER_WINDOWS:
        if have_real and len(extended_irrad_w_m2) >= n + w:
            ahead_mean  = extended_irrad_w_m2.rolling(w, min_periods=1).mean().shift(-w)
            ahead_tmax  = extended_temp_c.rolling(w, min_periods=1).max().shift(-w)
            out[f"fut_irrad_{w*30}min_mean"] = (
                ahead_mean.iloc[:n].fillna(method="ffill").fillna(0.0).values
            )
            out[f"fut_temp_{w*30}min_max"] = (
                ahead_tmax.iloc[:n].fillna(method="ffill")
                          .fillna(_synth_temp_at_offset(w)).values
            )
        else:
            # Synth: average synth irradiance over the future window
            stack = np.column_stack([
                _synth_irrad_at_offset(off).values
                for off in range(1, w + 1)
            ])
            out[f"fut_irrad_{w*30}min_mean"] = stack.mean(axis=1)
            tstack = np.column_stack([
                _synth_temp_at_offset(off).values
                for off in range(1, w + 1)
            ])
            out[f"fut_temp_{w*30}min_max"] = tstack.max(axis=1)

    # Regime-change cue: how different will conditions be 6 h from now?
    # Positive value = clouds rolling in (irrad dropping), negative = clearing.
    out["fut_irrad_delta_h12"] = (
        out.get("solar_irrad_w_m2", pd.Series(0.0, index=out.index)).values
        - out["fut_irrad_h12"].values
    )

    return out


def build_feature_matrix(
    df: pd.DataFrame,
    capacity_kwp: float = 0.0,
    target: str = "kw_import",
    lat: float | None = None,
    lon: float | None = None,
    timezone: str = "auto",
) -> tuple[pd.DataFrame, list[str]]:
    """One-call pipeline.

    If `lat` AND `lon` are provided, real hourly temperatures + shortwave
    irradiance are fetched from Open-Meteo (cached) and interpolated to
    30-min — both for the current-time feature columns AND for the new
    fut_* columns that look 1 h to 24 h ahead.  The forecast endpoint
    supplies any timestamps beyond the archive's 2-day lag, so the model
    can react to actual rain-tomorrow conditions, not just current ones.

    Network/parse failures silently fall back to the synth (deterministic
    time-of-day) — same column shape, model still trains.
    """
    out = add_time_features(df)
    out = add_lag_features(out, target=target)
    out = add_rolling_features(out, target=target)
    out = add_schedule_features(out, target=target)
    out = add_regime_features(out, target=target)

    # Pull real weather over an EXTENDED range:
    #   [df.timestamp.min(), df.timestamp.max() + LOOKAHEAD intervals]
    # The lookahead must cover the longest future-weather horizon plus the
    # longest rolling window (max(FUTURE_WEATHER_HORIZONS, FUTURE_WEATHER_WINDOWS)
    # = 48), with a small safety buffer.
    LOOKAHEAD_INTERVALS = 50          # ≈ 25 h ahead
    temps_ext, irrad_ext = None, None
    if lat is not None and lon is not None:
        from predictor.weather import get_temps_30min, get_irradiance_30min
        last_ts = out["timestamp"].iloc[-1]
        ext_future = pd.date_range(
            last_ts + pd.Timedelta(minutes=30),
            periods=LOOKAHEAD_INTERVALS, freq="30min",
        )
        ext_ts = pd.Series(
            list(out["timestamp"].values) + list(ext_future.values),
            index=range(len(out) + LOOKAHEAD_INTERVALS),
        )
        try:
            temps_ext = get_temps_30min(ext_ts, lat=lat, lon=lon, timezone=timezone)
            irrad_ext = get_irradiance_30min(ext_ts, lat=lat, lon=lon, timezone=timezone)
        except Exception:
            temps_ext = irrad_ext = None

    # Current-time weather features use the same series, sliced to history rows.
    n = len(out)
    temps_curr = temps_ext.iloc[:n].reset_index(drop=True) if temps_ext is not None else None
    irrad_curr = irrad_ext.iloc[:n].reset_index(drop=True) if irrad_ext is not None else None

    out = add_solar_features(out, capacity_kwp=capacity_kwp, irradiance_w_m2=irrad_curr)
    out = add_temperature_features(out, temps=temps_curr)

    # NEW: future-weather features (always emitted; synth fallback if no real)
    out = add_future_weather_features(out, temps_ext, irrad_ext, capacity_kwp=capacity_kwp)

    feature_cols = [
        "hour", "dow", "month", "is_weekend", "is_business_hours",
        "hour_sin", "hour_cos", "dow_sin", "dow_cos",
        "is_holiday", "is_pre_holiday", "is_post_holiday",
        *[f"lag_{L}" for L in DEFAULT_LAGS],
        "roll_mean_24", "roll_mean_48", "roll_std_48", "roll_max_48", "roll_min_48",
        "roll_mean_336",
        "dow_hour_mean", "dow_hour_std", "same_dow_hour_4w_mean",
        "delta_vs_24h", "delta_vs_7d", "recent_vs_schedule",
        "solar_capacity_kwp", "est_solar_gen_kw", "has_solar",
        "solar_irrad_w_m2", "solar_irrad_ratio",
        "est_temp_c", "is_hot_period",
        # Future-weather columns
        *[f"fut_irrad_h{h}" for h in FUTURE_WEATHER_HORIZONS],
        *[f"fut_temp_h{h}"  for h in FUTURE_WEATHER_HORIZONS],
        *[f"fut_irrad_{w*30}min_mean" for w in FUTURE_WEATHER_WINDOWS],
        *[f"fut_temp_{w*30}min_max"   for w in FUTURE_WEATHER_WINDOWS],
        "fut_irrad_delta_h12",
    ]
    out = out.dropna(subset=feature_cols + [target]).reset_index(drop=True)
    return out, feature_cols
