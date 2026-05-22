"""
BOLT Integrated — Unified AI Energy Management Platform
Combines: Predictor → Manager → SAM Calculator → PowerRECO

Run with:  streamlit run app.py
"""
from __future__ import annotations
import io
import math
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import traceback

# ── Package imports ────────────────────────────────────────────────────────────
from predictor.data_loader import auto_load, load_csv, summarize
from predictor.solar_estimator import detect_has_solar, estimate_solar_capacity_kwp
from predictor.forecaster import DirectMultiStepForecaster
from predictor.cv import expanding_window_cv, format_cv_report
from predictor.simulation import (
    initialize_simulation, advance_one_tick, compute_running_accuracy,
    build_forecast_vs_actual_df, split_historical_for_simulation,
)

from manager.optimizer import run_ai_manager, parse_uploaded_data, calc_kva

from calculator.tnb_tariffs import (
    auto_detect_tariff, compute_monthly_stats, calculate_bill,
    compute_nem_credit, TARIFF_META,
)

from powerreco.solar_sizing import calculate_solar_sizing
from powerreco.battery_sizing import calculate_battery_sizing
from powerreco.roi_engine import calculate_roi, assess_feasibility
from powerreco.optimizer import find_optimum_from_existing_run
from powerreco.energy_flows import decompose_flows, representative_day

from pipeline.data_bridge import (
    forecast_to_manager_df,
    historical_to_manager_df,
    manager_results_to_sam_df,
    manager_results_to_original_df,
    manager_results_to_powerreco_df,
    manager_results_to_csv,
    forecast_result_to_csv,
)

from site_profiles import (
    PROFILES, list_profiles, get_profile,
    profile_loads_for_manager, profile_predictor_kwargs,
    DEFAULT_PROFILE_ID,
)

from persistence import (
    load_overrides, save_overrides, OVERRIDES_PATH,
    save_load_profile, load_load_profile,
    save_forecaster,   load_forecaster,
    site_state_info,   all_sites_state_info, clear_site_state,
)

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="BOLT Integrated",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Session state init ─────────────────────────────────────────────────────────
for key, default in [
    ("df", None),
    ("file_summary", None),
    ("solar_info", None),
    ("forecaster", None),
    ("forecast_result", None),
    ("manager_results", None),
    ("manager_df_optimized", None),
    ("manager_df_original", None),
    ("powerreco_df", None),
    ("solar_sizing", None),
    ("battery_sizing", None),
    ("roi", None),
    ("tariff_code", "C1"),
    ("tariff_schedule_key", "new_2025"),  # 'new_2025' or 'legacy_2014'
    ("site_profile_id", DEFAULT_PROFILE_ID),
    ("sizing_sweep", None),
    # Live simulation state — owned by the Live Simulation tab.  Lazy-inited
    # when the user opens the tab with a trained forecaster + data present.
    ("sim_state", None),
    ("sim_playing", False),       # autoplay flag
    ("sim_history_actuals", None),  # cached "what was already in the forecaster's
                                     # history" so we can plot revealed-vs-historical
    # Forward-looking Manager output, re-computed every sim tick.  Read by
    # the AI Manager tab's live fragment.  Distinct from manager_results
    # (which is the user's manual-mode run on historical data).
    ("live_manager_results", None),
    ("live_manager_last_tick", -1),
    # Live battery state-of-charge — carried across ticks so the Manager's
    # next plan starts from the SOC reached after executing the previous
    # tick's first-interval decision.  Without this the Manager re-plans
    # from a fresh 50 % every tick, ignoring battery state evolution.
    ("live_battery_soc_pct", None),
    # Accumulates the FIRST row of each tick's Manager output — the decision
    # actually executed on the live actual reading, not the forecast plan.
    ("executed_ticks", []),
    # User scenario factor — scales the forecast for a known operational event.
    # Persisted across reruns so the Manager always reads the active factor.
    ("forecast_user_factor",     1.0),
    ("forecast_user_confidence", 0.80),
    ("forecast_factor_steps",    48),
    # Tracks which site_id the auto historical Manager run was last completed
    # for. Reset to None by _invalidate_downstream() so a new data upload or
    # site switch triggers a fresh auto-run.
    ("_auto_manager_site", None),
    # Completed-month bills derived from executed_ticks. Each entry is:
    # {"month": "YYYY-MM", "ticks": int, "before": {...}, "after": {...}, "savings_rm": float}
    ("live_bill_months", []),
    # Per-site override dict: {site_id: {field: user_value}}.  Any field not
    # in here falls back to the site profile preset.  Editing in the Site
    # Setup tab populates these; switching site profile in the sidebar
    # preserves each site's edits independently.  On first run we hydrate
    # from data/site_overrides.json so a browser refresh doesn't wipe edits.
    ("site_overrides", load_overrides()),
    ("use_real_weather", True),
]:
    if key not in st.session_state:
        st.session_state[key] = default


# ── Helpers ────────────────────────────────────────────────────────────────────
def _fmt_rm(v: float) -> str:
    return f"RM {v:,.2f}"


def _fmt_kw(v: float) -> str:
    return f"{v:,.1f} kW"


def _plot_load(df: pd.DataFrame, title: str = "Load Profile") -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df["timestamp"], y=df["kw_import"],
        mode="lines", name="kW Import", line=dict(color="#1f77b4"),
    ))
    if "kw_export" in df.columns and df["kw_export"].max() > 0:
        fig.add_trace(go.Scatter(
            x=df["timestamp"], y=df["kw_export"],
            mode="lines", name="kW Export", line=dict(color="#ff7f0e"),
        ))
    fig.update_layout(title=title, xaxis_title="Time", yaxis_title="kW",
                      hovermode="x unified", height=350)
    return fig


def _site_overrides_for(site_id: str) -> dict:
    """Return the override dict for the given site, creating it lazily."""
    if site_id not in st.session_state.site_overrides:
        st.session_state.site_overrides[site_id] = {}
    return st.session_state.site_overrides[site_id]


def _so(field: str, fallback):
    """Get an override for the *active* site if set, else `fallback`."""
    ov = _site_overrides_for(st.session_state.site_profile_id)
    return ov.get(field, fallback)


def _set_so(field: str, value):
    """Set an override for the active site."""
    ov = _site_overrides_for(st.session_state.site_profile_id)
    ov[field] = value


def _check_data_quality(df: pd.DataFrame) -> list[dict]:
    """Auto-detect common data-quality issues that hurt forecast accuracy.

    Returns a list of {severity, title, detail} dicts ready for the UI.
    severity: 'error' (model can't train reliably), 'warning' (likely to
    hurt accuracy), or 'info' (FYI).
    """
    issues: list[dict] = []
    if df is None or len(df) == 0:
        return [{"severity": "error", "title": "Empty dataset",
                 "detail": "No rows after parsing."}]

    n = len(df)
    span_days = (df["timestamp"].max() - df["timestamp"].min()).days

    # 1) Length
    if span_days < 30:
        issues.append({
            "severity": "warning", "title": f"Only {span_days} days of data",
            "detail": "Forecaster needs 60+ days to learn weekly patterns "
                      "reliably, 90+ for monthly seasonality. Long-horizon "
                      "MAPE (h=24, h=48) will be capped no matter how you tune.",
        })

    # 2) Duplicate timestamps
    dup = df["timestamp"].duplicated().sum()
    if dup > 0:
        issues.append({
            "severity": "warning",
            "title": f"{dup} duplicate timestamp(s)",
            "detail": "Same timestamp appears more than once — silently "
                      "kept the first occurrence during parsing. Inspect "
                      "the raw file if this is unexpected.",
        })

    # 3) Gaps — expected interval 30 min
    deltas = df["timestamp"].sort_values().diff().dt.total_seconds() / 60.0
    expected = 30.0
    big_gaps = int(((deltas > expected * 1.5)).sum())
    if big_gaps > 0:
        max_gap_min = float(deltas.max() or 0)
        issues.append({
            "severity": "warning" if big_gaps > 5 else "info",
            "title": f"{big_gaps} gap(s) > 45 min (max {max_gap_min:.0f} min)",
            "detail": "Missing intervals break lag features for the rows "
                      "right after the gap. Consider forward-filling or "
                      "splitting the dataset into continuous chunks.",
        })

    # 4) Negative kw_import
    neg = int((df["kw_import"] < 0).sum())
    if neg > 0:
        issues.append({
            "severity": "warning",
            "title": f"{neg} row(s) with negative kw_import",
            "detail": "Likely sensor errors or net-metering bookkeeping. "
                      "These pollute training. Auto-clipping to 0 on use, "
                      "but consider cleaning at the source.",
        })

    # 5) Sudden 10× spikes — likely sensor glitches, not real peaks
    med = float(df["kw_import"].median())
    if med > 0:
        glitches = int((df["kw_import"] > 10 * med).sum())
        if glitches > 0:
            max_spike = float(df["kw_import"].max())
            issues.append({
                "severity": "warning",
                "title": f"{glitches} row(s) > 10× median ({max_spike:.0f} kW vs median {med:.0f})",
                "detail": "Either real peak events (festival ops, equipment "
                          "startup) — informative, the outlier filter handles "
                          "those — or sensor instrumentation glitches. "
                          "Spot-check those rows.",
            })

    # 6) Long zero stretches (potentially missing data masked as 0)
    zero_run = (df["kw_import"] == 0).rolling(48, min_periods=1).sum().max()
    if zero_run >= 24:
        issues.append({
            "severity": "warning",
            "title": f"Up to {int(zero_run)} consecutive zero readings",
            "detail": "Could be a real outage, but often it's a parser "
                      "issue (NaN → 0). Verify against the source file.",
        })

    # 7) Constant runs (sensor stuck)
    const_run = 0
    if len(df) > 12:
        diffs_zero = (df["kw_import"].diff().abs() < 1e-6).astype(int)
        const_run = int(diffs_zero.rolling(12, min_periods=1).sum().max())
    if const_run >= 12:
        issues.append({
            "severity": "info",
            "title": f"Up to {const_run} consecutive constant readings",
            "detail": "Sensor may be reporting a hold value. Tree-based "
                      "models can handle this, but it's worth flagging.",
        })

    if not issues:
        issues.append({
            "severity": "ok", "title": "Data looks clean",
            "detail": f"{n:,} rows · {span_days} days · no anomalies detected.",
        })
    return issues


def _run_manager_on_live_forecast() -> list | None:
    """Run the AI Manager on the *current 24-h forecast* to produce a
    forward-looking dispatch plan that updates every sim tick.

    Returns None when prerequisites are missing (sim not initialised,
    forecaster not trained, battery_kwh=0, etc.).  Errors are swallowed —
    a stale live_manager_results is preferable to crashing the fragment.
    """
    sim = st.session_state.get("sim_state")
    fc  = st.session_state.get("forecaster")
    if sim is None or fc is None or getattr(sim, "current_forecast", None) is None:
        return None

    profile = get_profile(st.session_state.site_profile_id)
    battery_kwh = float(_so("battery_kwh", profile.battery_kwh) or profile.battery_kwh)
    if battery_kwh <= 0:
        battery_kwh = max(profile.battery_kwh, 100.0)

    try:
        mgr_df = forecast_to_manager_df(sim.current_forecast, historical_df=fc.history)
    except Exception:
        return None
    if mgr_df is None or mgr_df.empty:
        return None

    # ── #1: PREPEND THE LATEST ACTUAL ─────────────────────────────────────
    # Ground the Manager's first dispatch decision in reality.  Without
    # this, even the "right now" decision is based on a prediction made
    # 30 min ago.  We add the most-recently-revealed row from the
    # forecaster's history (which advance_one_tick has been appending to)
    # as the first interval of the Manager input.  Subsequent rows are
    # the 24-h forecast as before.
    if fc.history is not None and len(fc.history) > 0:
        try:
            last_row = fc.history.iloc[[-1]]
            actual_row = pd.DataFrame({
                "timestamp":   pd.to_datetime(last_row["timestamp"].values),
                "kw_import":   last_row["kw_import"].astype(float).values,
                "kw_export":   last_row.get("kw_export",   pd.Series([0.0])).astype(float).values,
                "kvar_import": last_row.get("kvar_import", pd.Series([0.0])).astype(float).values,
                "kvar_export": last_row.get("kvar_export", pd.Series([0.0])).astype(float).values,
            })
            actual_row["kw_net"]   = actual_row["kw_import"]   - actual_row["kw_export"]
            actual_row["kvar_net"] = actual_row["kvar_import"] - actual_row["kvar_export"]
            actual_row["kva"]      = np.sqrt(actual_row["kw_net"]**2 + actual_row["kvar_net"]**2)
            # Skip if the actual's timestamp is already inside the forecast
            # window (shouldn't happen, but defensive).
            if actual_row["timestamp"].iloc[0] < mgr_df["timestamp"].iloc[0]:
                mgr_df = pd.concat([actual_row, mgr_df], ignore_index=True)
        except Exception:
            pass    # fall back to forecast-only if shape mismatch

    # Apply user scenario factor to forecast rows only.
    # Row 0 is the actual reading — never scaled.
    # Rows 1..n_steps are the forecast — scaled by the user's factor so the
    # Manager plans battery dispatch for the expected operational change.
    _u_factor = float(st.session_state.get("forecast_user_factor", 1.0))
    _u_steps  = int(st.session_state.get("forecast_factor_steps", 48))
    if abs(_u_factor - 1.0) > 1e-3 and len(mgr_df) > 1:
        _end = min(1 + _u_steps, len(mgr_df))
        for _col in ("kw_import", "kw_net", "kvar_import", "kvar_net"):
            if _col in mgr_df.columns:
                _v = mgr_df[_col].values.astype(float).copy()
                _v[1:_end] = _v[1:_end] * _u_factor
                mgr_df[_col] = _v
        if "kw_net" in mgr_df.columns and "kvar_net" in mgr_df.columns:
            mgr_df["kva"] = np.sqrt(
                mgr_df["kw_net"].values ** 2 + mgr_df["kvar_net"].values ** 2
            )

    # Build loads dict from profile, then layer Site Setup overrides on top
    loads = profile_loads_for_manager(profile)
    ov = _site_overrides_for(profile.id)
    if "ev" in loads:
        ev = loads["ev"]
        if any(k in ov for k in ("ev_count", "ev_kw_each", "ev_kind")):
            c   = int(ov.get("ev_count",   4))
            kwe = float(ov.get("ev_kw_each", 22.0))
            kd  = str(ov.get("ev_kind",    "AC"))
            ev["ev_chargers"] = [{"count": c, "kw_each": kwe, "kind": kd}]
            ev["ev_total_kw"] = c * kwe
        if any(k in ov for k in ("ev_window_start", "ev_window_end")):
            ev["allowed_window"] = [int(ov.get("ev_window_start", 18)),
                                     int(ov.get("ev_window_end",   8))]
    if "hvac" in loads:
        hv = loads["hvac"]
        if any(k in ov for k in ("hvac_protect_start", "hvac_protect_end")):
            hv["protected_window"] = [int(ov.get("hvac_protect_start", 9)),
                                       int(ov.get("hvac_protect_end",  17))]
        if "hvac_max_cut_pct" in ov:
            hv["max_cut_pct"] = float(ov["hvac_max_cut_pct"]) / 100.0

    # ── #2: SOC CONTINUITY ────────────────────────────────────────────────
    # If a live battery SOC has been tracked across previous ticks, use
    # IT as initial_soc_pct.  This makes the Manager's strategy actually
    # COMPOUND — pre-charging on tick 5 affects available capacity at
    # tick 30.  Without this, every tick re-plans from a fresh 50 % SOC.
    live_soc = st.session_state.get("live_battery_soc_pct")
    if live_soc is None:
        init_soc = float(_so("init_soc_pct", profile.initial_soc_pct))
    else:
        init_soc = float(live_soc)

    try:
        return run_ai_manager(
            mgr_df, loads,
            battery_capacity_kwh=battery_kwh,
            priority_order=list(loads.keys()),
            peak_target_pct=float(_so("md_target_pct",      profile.md_target_pct)),
            bat_charge_upper_pct=float(_so("charge_upper_pct", profile.charge_upper_pct)),
            c_rate=float(_so("c_rate", profile.c_rate)),
            initial_soc_pct=init_soc,
            bat_efficiency=0.95,
            peak_reference_kva=None,
        )
    except Exception:
        return None


def _compute_live_month_bill(
    month_str: str,
    ticks: list[dict],
    tariff: str,
    sched_key: str,
    icpt_sen: float,
    nem_rate: float,
) -> dict:
    """
    Compute before/after TNB monthly bill from a list of executed Manager ticks
    belonging to a single calendar month.

    Each tick dict (from executed_ticks) contains:
        kw_original, kvar_original  — raw pre-optimization readings
        kw_managed,  kvar_managed   — post-optimization values

    Returns a dict ready to append to session_state.live_bill_months.
    """
    orig_rows, mgd_rows = [], []
    for t in ticks:
        ts = pd.to_datetime(t["timestamp"])
        orig_rows.append({
            "timestamp":   ts,
            "kw_import":   float(t.get("kw_original",  t.get("kw_managed", 0))),
            "kw_export":   0.0,
            "kvar_import": float(t.get("kvar_original", t.get("kvar_managed", 0))),
            "kvar_export": 0.0,
        })
        mgd_rows.append({
            "timestamp":   ts,
            "kw_import":   float(t.get("kw_managed",  0)),
            "kw_export":   0.0,
            "kvar_import": float(t.get("kvar_managed", 0)),
            "kvar_export": 0.0,
        })

    def _bill_for(rows: list[dict]) -> dict:
        if not rows:
            return {}
        df_m = pd.DataFrame(rows)
        stats = compute_monthly_stats(df_m, schedule_key=sched_key)
        if stats.empty:
            return {}
        row = stats.iloc[0]
        b   = calculate_bill(
            tariff,
            monthly_kwh     = float(row["total_kwh"]),
            peak_kwh        = float(row["peak_kwh"]),
            offpeak_kwh     = float(row["offpeak_kwh"]),
            max_demand_kw   = float(row["max_demand_kw"]),
            icpt_sen_per_kwh= icpt_sen,
            schedule_key    = sched_key,
        )
        nem = compute_nem_credit(float(row["export_kwh"]), nem_rate)
        return {
            "total_kwh":     round(float(row["total_kwh"]), 1),
            "peak_kw_md":    round(float(row["max_demand_kw"]), 1),
            "energy_rm":     b["energy_charge"],
            "md_rm":         b["md_charge"],
            "icpt_rm":       b["icpt_charge"],
            "kwtbb_rm":      b["kwtbb_charge"],
            "tax_rm":        b["service_tax"],
            "nem_credit_rm": nem["nem_credit_rm"],
            "net_bill_rm":   round(b["total_bill"] - nem["nem_credit_rm"], 2),
        }

    before = _bill_for(orig_rows)
    after  = _bill_for(mgd_rows)
    savings = round(
        before.get("net_bill_rm", 0) - after.get("net_bill_rm", 0), 2
    ) if before and after else 0.0

    return {
        "month":      month_str,
        "ticks":      len(ticks),
        "before":     before,
        "after":      after,
        "savings_rm": savings,
    }


def _invalidate_downstream():
    """Clear cached pipeline state so the next Train re-runs everything.

    Called when the user changes a site parameter — we don't auto-retrain
    (that would surprise them mid-edit), but we DO clear the stale results
    so they don't keep showing after the inputs have changed.
    """
    for key in ("forecaster", "forecast_result", "manager_results",
                "manager_df_optimized", "manager_df_original", "powerreco_df",
                "solar_sizing", "battery_sizing", "roi", "sizing_sweep"):
        st.session_state[key] = None
    # Reset the auto-run flag so a new upload or site switch triggers a fresh run.
    st.session_state["_auto_manager_site"] = None


def _ensure_historical_manager() -> bool:
    """
    Run the AI Manager on the full uploaded dataset if it hasn't been run yet
    for the current site in this session.  Uses the site profile defaults
    layered with whatever overrides the user has set in Site Setup.

    Called automatically after data upload and on app start-up so the
    Calculator and PowerRECO always have reference data without requiring
    the user to manually click 'Run on full history'.

    Returns True if a run was performed, False if skipped (already done,
    no data, or an error occurred).
    """
    site_id = st.session_state.site_profile_id
    if st.session_state.df is None:
        return False
    if st.session_state.manager_results is not None:
        return False
    if st.session_state.get("_auto_manager_site") == site_id:
        return False   # already ran for this site this session

    # Mark immediately so a Streamlit rerun doesn't trigger a second run.
    st.session_state["_auto_manager_site"] = site_id

    try:
        profile = get_profile(site_id)
        loads   = profile_loads_for_manager(profile)

        # Apply any overrides the user may have set (same logic as live Manager).
        ov = _site_overrides_for(site_id)
        if "ev" in loads:
            ev = loads["ev"]
            if any(k in ov for k in ("ev_count", "ev_kw_each", "ev_kind")):
                c   = int(ov.get("ev_count",   4))
                kwe = float(ov.get("ev_kw_each", 22.0))
                kd  = str(ov.get("ev_kind",    "AC"))
                ev["ev_chargers"] = [{"count": c, "kw_each": kwe, "kind": kd}]
                ev["ev_total_kw"] = c * kwe
            if any(k in ov for k in ("ev_window_start", "ev_window_end")):
                ev["allowed_window"] = [
                    int(ov.get("ev_window_start", 18)),
                    int(ov.get("ev_window_end",    8)),
                ]
        if "hvac" in loads:
            hv = loads["hvac"]
            if any(k in ov for k in ("hvac_protect_start", "hvac_protect_end")):
                hv["protected_window"] = [
                    int(ov.get("hvac_protect_start",  9)),
                    int(ov.get("hvac_protect_end",   17)),
                ]
            if "hvac_max_cut_pct" in ov:
                hv["max_cut_pct"] = float(ov["hvac_max_cut_pct"]) / 100.0

        mgr_df  = historical_to_manager_df(st.session_state.df)
        results = run_ai_manager(
            mgr_df, loads,
            battery_capacity_kwh  = float(_so("battery_kwh",      profile.battery_kwh)),
            priority_order        = list(loads.keys()),
            peak_target_pct       = float(_so("md_target_pct",    profile.md_target_pct)),
            bat_charge_upper_pct  = float(_so("charge_upper_pct", profile.charge_upper_pct)),
            c_rate                = float(_so("c_rate",            profile.c_rate)),
            initial_soc_pct       = float(_so("init_soc_pct",     profile.initial_soc_pct)),
            bat_efficiency        = 0.95,
        )
        st.session_state.manager_results      = results
        st.session_state.manager_df_optimized = manager_results_to_sam_df(results)
        st.session_state.manager_df_original  = manager_results_to_original_df(results)
        st.session_state.powerreco_df         = manager_results_to_powerreco_df(results)
        return True
    except Exception:
        # Silent failure — user can always run manually from the Manager tab.
        st.session_state["_auto_manager_site"] = None  # allow retry on next render
        return False


def _apply_user_factor(
    median: np.ndarray,
    p10: np.ndarray,
    p90: np.ndarray,
    factor: float,
    confidence: float,
    n_steps: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Scale the first `n_steps` of a forecast by `factor`, widening the
    uncertainty bands to reflect the user's confidence in their estimate.

    Engineering rationale
    ─────────────────────
    A user-supplied factor ("big order → ×1.5") is an operational judgement,
    not a statistical guarantee.  The estimation error (1 − confidence)
    adds to the existing model uncertainty so the Manager plans
    conservatively rather than treating the factor as exact.

    Band widening formula:
        extra[i] = adj_median[i] × (1 − confidence) × |factor − 1|
        adj_p90[i] += extra[i]
        adj_p10[i]  = max(0, adj_p10[i] − extra[i])

    At confidence = 100 %: bands scale proportionally, no additional width.
    At confidence =  50 %: 50 % of the scaling delta added to each side.
    """
    n          = min(int(n_steps), len(median))
    adj_median = median.copy().astype(float)
    adj_p10    = p10.copy().astype(float)
    adj_p90    = p90.copy().astype(float)

    adj_median[:n] = median[:n] * factor
    adj_p10[:n]    = p10[:n]   * factor
    adj_p90[:n]    = p90[:n]   * factor

    # Uncertainty widening from estimation error
    extra          = adj_median[:n] * (1.0 - confidence) * abs(factor - 1.0)
    adj_p90[:n]   += extra
    adj_p10[:n]    = np.maximum(0.0, adj_p10[:n] - extra)

    return adj_median, adj_p10, adj_p90


def _collect_assumptions() -> list[dict]:
    """Walk the current session state and build a transparency table of every
    value the system is using, tagged with provenance:

        auto-detect  — derived from the uploaded data
        profile      — set by the active site profile
        default      — hard-coded sensible default
        user         — the user explicitly entered/overrode this value
    """
    ss = st.session_state
    summ      = ss.get("file_summary") or {}
    solar_inf = ss.get("solar_info") or {}
    profile   = get_profile(ss.get("site_profile_id"))
    sched_key = ss.get("tariff_schedule_key", "new_2025")

    rows: list[dict] = []
    def add(label, value, provenance, note=""):
        rows.append({
            "Parameter": label,
            "Value":     value,
            "Source":    provenance,
            "Note":      note,
        })

    # ── Site / location ─────────────────────────────────────────────
    add("Site profile",          profile.name,                  "user",        "Sidebar selector")
    add("Latitude",               f"{profile.lat:.4f}",          "profile",     "Default Kuala Lumpur — override in Predictor tab")
    add("Longitude",              f"{profile.lon:.4f}",          "profile",     "Default Kuala Lumpur — override in Predictor tab")

    # ── Data summary (from upload) ──────────────────────────────────
    if summ:
        add("Data interval",      "30 min",                       "default",
            "Hourly / 15-min input is auto-resampled to 30 min")
        add("Period days",        summ.get("days", "?"),          "auto-detect", "")
        add("Peak kW (observed)", f"{summ.get('max_kw_import', 0):.1f}", "auto-detect", "")
        add("Mean kW",            f"{summ.get('mean_kw_import', 0):.1f}", "auto-detect", "")

    # ── Tariff ──────────────────────────────────────────────────────
    add("TNB tariff",             ss.get("tariff_code", "C1"),
        "auto-detect", "Inferred from peak kW + load shape; overridable in Calculator tab")
    add("Tariff schedule",        sched_key,
        "user", "Pre/Post-July 2025 — sidebar toggle")

    # ── Solar / weather ─────────────────────────────────────────────
    if solar_inf.get("has_solar"):
        add("Solar present",      "Yes (detected)", "auto-detect",
            solar_inf.get("reason", ""))
        add("Solar capacity",     f"{solar_inf.get('capacity_kwp', '?')} kWp",
            "auto-detect", "Estimated from midday dip vs morning baseline + max export")
    else:
        kwp_hint = profile.expected_solar_kwp
        if kwp_hint:
            add("Solar capacity",  f"{kwp_hint} kWp",  "profile",
                "From site profile preset — actual data showed no solar signature")
        else:
            add("Solar present",   "No",               "auto-detect", "")
    add("Peak Sun Hours",        "4.5 h/day",          "default",
        "Malaysia average — override in PowerRECO tab")

    # ── Manager / battery defaults ──────────────────────────────────
    add("Battery default",       f"{profile.battery_kwh:.0f} kWh", "profile", "")
    add("MD target",             f"{profile.md_target_pct*100:.0f}% of ref peak", "profile", "")
    add("Charge upper threshold", f"{profile.charge_upper_pct*100:.0f}% of ref peak", "profile", "")
    add("Round-trip efficiency", "90% (45 % one-way × 2)", "default", "")
    add("MD billing window",     "14:00 — 22:00",          "default",
        "Manager treats this as the high-MD-risk band; the TOU peak window for billing is separately 08:00 — 22:00 Mon-Sat")

    # ── Power factor / kVAR (from data bridge) ──────────────────────
    df_raw = ss.get("df")
    if df_raw is not None and len(df_raw):
        try:
            kw_mean = float(df_raw["kw_import"].mean())
            kvar_mean = float(df_raw["kvar_import"].mean()) if "kvar_import" in df_raw.columns else 0.0
            if kw_mean > 1.0:
                kva_mean = (kw_mean**2 + kvar_mean**2) ** 0.5
                pf = kw_mean / kva_mean if kva_mean > 0 else 0.85
                add("Power factor", f"{pf:.2f}", "auto-detect",
                    "Derived from kW / kVAR readings; used by the data bridge when forecast kVAR is unavailable")
        except Exception:
            pass

    # ── Forecaster tuning ───────────────────────────────────────────
    add("Forecaster horizons",    "24 (1 → 48 steps)",  "default", "")
    add("Quantile bands",         "P10 / P50 / P90",    "default", "")
    add("bias_window",           f"{profile.predictor.bias_window}", "profile",
        "Per-site tuned — wider = more stable; narrower = tracks regime shifts faster")
    add("n_estimators",           f"{profile.predictor.n_estimators}", "profile", "")

    return rows


def _provenance_color(p: str) -> str:
    return {
        "auto-detect": "#22c55e",   # green
        "user":        "#0ea5e9",   # blue
        "profile":     "#a78bfa",   # purple
        "default":     "#f59e0b",   # amber
    }.get(p, "#9ca3af")


def _run_manager_on_df(df: pd.DataFrame, loads: dict, priority_order: list,
                        battery_kwh: float, peak_target_pct: float,
                        charge_upper_pct: float, c_rate: float,
                        init_soc_pct: float, bat_eff: float,
                        peak_ref_kva: float | None) -> list[dict]:
    mgr_df = historical_to_manager_df(df)
    return run_ai_manager(
        mgr_df, loads, battery_kwh, priority_order,
        peak_target_pct, charge_upper_pct,
        c_rate=c_rate, initial_soc_pct=init_soc_pct,
        bat_efficiency=bat_eff,
        peak_reference_kva=peak_ref_kva if peak_ref_kva and peak_ref_kva > 0 else None,
    )


def _bootstrap_active_site():
    """Hydrate session_state from disk for the active site.

    Triggered:
      • First page render of a new session (no `_loaded_site_id` yet)
      • Sidebar profile dropdown changes (mismatch with `_loaded_site_id`)

    Side-effects:
      • Clears stale Manager / Bill / ROI state when switching sites.
      • Loads saved load profile → `st.session_state.df` + summary + tariff
      • Loads saved forecaster → `st.session_state.forecaster` + initial forecast

    Idempotent: when `_loaded_site_id` already matches, this no-ops cheaply
    (the function returns after the first guard), so Streamlit's frequent
    page re-renders don't keep re-reading disk.
    """
    site_id = st.session_state.site_profile_id
    if st.session_state.get("_loaded_site_id") == site_id:
        return

    # Switching sites — clear stale per-site pipeline state first
    for key in ("df", "file_summary", "solar_info", "forecaster", "forecast_result",
                "manager_results", "manager_df_optimized", "manager_df_original",
                "powerreco_df", "solar_sizing", "battery_sizing", "roi", "sizing_sweep"):
        st.session_state[key] = None

    loaded_bits: list[str] = []

    # 1) Load saved load profile
    saved_df = load_load_profile(site_id)
    if saved_df is not None and len(saved_df) > 0:
        st.session_state.df = saved_df
        try:
            summ = summarize(saved_df)
            summ["max_kw_import"] = float(saved_df["kw_import"].max())
            st.session_state.file_summary = summ
        except Exception:
            pass
        try:
            has_solar, reason = detect_has_solar(saved_df)
            solar_info = {"has_solar": has_solar, "reason": reason}
            if has_solar:
                solar_info.update(estimate_solar_capacity_kwp(saved_df))
            st.session_state.solar_info = solar_info
        except Exception:
            pass
        try:
            tc, _, _ = auto_detect_tariff(saved_df)
            st.session_state.tariff_code = tc
        except Exception:
            pass
        loaded_bits.append(f"load profile ({len(saved_df):,} rows)")

    # 2) Load saved trained forecaster
    saved_fc = load_forecaster(site_id)
    if saved_fc is not None:
        st.session_state.forecaster = saved_fc
        try:
            st.session_state.forecast_result = saved_fc.forecast(output_steps=48)
            loaded_bits.append("trained forecaster")
        except Exception:
            # Saved model exists but predict() failed (likely feature mismatch).
            # Drop it so the UI prompts a fresh train.
            st.session_state.forecaster = None
            st.session_state.forecast_result = None

    st.session_state._loaded_site_id = site_id
    st.session_state._bootstrap_msg = (
        f"✅ Restored {' + '.join(loaded_bits)} for "
        f"{get_profile(site_id).name}." if loaded_bits else None
    )


_bootstrap_active_site()

# Auto-run the historical Manager if data is loaded but results are missing.
# This fires once per site per session so the Calculator and PowerRECO always
# have reference data without requiring a manual button click.
if (st.session_state.df is not None
        and st.session_state.manager_results is None
        and st.session_state.get("_auto_manager_site") != st.session_state.site_profile_id):
    with st.spinner(
        f"Auto-running historical Manager for "
        f"{get_profile(st.session_state.site_profile_id).name}…"
    ):
        _ensure_historical_manager()


# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.title("⚡ BOLT Integrated")
    st.caption("AI Energy Management Platform")

    # ── Site profile picker ───────────────────────────────────────────────
    _profile_ids = list(PROFILES.keys())
    _profile_idx = _profile_ids.index(st.session_state.site_profile_id) \
                   if st.session_state.site_profile_id in _profile_ids else 0
    _picked_id = st.selectbox(
        "Site profile",
        options=_profile_ids,
        index=_profile_idx,
        format_func=lambda pid: PROFILES[pid].name,
        help=(
            "Loads tuned forecaster + Manager defaults for one of the four "
            "case-study sites, or 'Custom' for a generic upload. Override "
            "any field downstream — this just pre-fills."
        ),
    )
    if _picked_id != st.session_state.site_profile_id:
        st.session_state.site_profile_id = _picked_id
    _active_profile = get_profile(st.session_state.site_profile_id)
    st.caption(_active_profile.description)
    if _active_profile.data_hint:
        st.caption(f"📂 Expected file: *{_active_profile.data_hint}*")

    st.divider()

    if st.session_state.df is not None:
        summ = st.session_state.file_summary or {}
        st.success(f"Data loaded: {summ.get('rows', '?')} rows")
        st.caption(f"{summ.get('start', '?')} → {summ.get('end', '?')}")
        st.caption(f"Peak: {summ.get('max_kw_import', 0):.1f} kW")
        if st.button("Clear data", use_container_width=True):
            for key in ("df", "file_summary", "solar_info", "forecaster",
                        "forecast_result", "manager_results", "manager_df_optimized",
                        "manager_df_original", "powerreco_df",
                        "solar_sizing", "battery_sizing", "roi"):
                st.session_state[key] = None
            st.rerun()
    else:
        st.info("Upload a load profile to begin.")

    st.divider()
    st.caption("TNB tariff schedule")
    _sched_label = st.radio(
        "Schedule",
        options=["new_2025", "legacy_2014"],
        format_func=lambda k: {
            "new_2025":    "Post-July 2025  (MD RM 97.06/kW)",
            "legacy_2014": "Pre-July 2025   (MD RM 30.30/kW)",
        }[k],
        index=0 if st.session_state.tariff_schedule_key == "new_2025" else 1,
        key="tariff_schedule_radio",
        help="Switches BOTH the Bill Calculator and PowerRECO ROI engine to "
             "the same rate schedule. Use this for sensitivity analysis "
             "between the legacy and post-July 2025 tariffs.",
    )
    if _sched_label != st.session_state.tariff_schedule_key:
        st.session_state.tariff_schedule_key = _sched_label

    st.divider()
    if st.session_state.df is not None:
        with st.expander("🧾 Assumptions (auto-filled)", expanded=False):
            _summary_rows = _collect_assumptions()
            # Compact 2-column rendering — Parameter / Value with provenance dot
            for r in _summary_rows[:12]:                # top 12 to keep sidebar tight
                colA, colB = st.columns([5, 3])
                dot = f"<span style='display:inline-block;width:8px;height:8px;border-radius:50%;background:{_provenance_color(r['Source'])};margin-right:6px;vertical-align:middle'></span>"
                colA.markdown(f"{dot}{r['Parameter']}", unsafe_allow_html=True)
                colB.markdown(f"**{r['Value']}**")
            if len(_summary_rows) > 12:
                st.caption(f"…and {len(_summary_rows) - 12} more — see Upload tab for full table")
        st.divider()

    st.caption("Pipeline status")
    _statuses = [
        ("Data", st.session_state.df is not None),
        ("Predictor", st.session_state.forecaster is not None),
        ("Manager", st.session_state.manager_results is not None),
        ("Calculator", st.session_state.manager_df_optimized is not None),
        ("PowerRECO", st.session_state.roi is not None),
    ]
    for label, done in _statuses:
        st.write(f"{'✅' if done else '⬜'} {label}")


# ══════════════════════════════════════════════════════════════════════════════
# TABS
# ══════════════════════════════════════════════════════════════════════════════
# Show one-time bootstrap message at the top (cleared once shown)
_boot_msg = st.session_state.pop("_bootstrap_msg", None)
if _boot_msg:
    st.success(_boot_msg + "  Open the Predictor tab to forecast, or edit "
                            "parameters in Site Setup → Apply & Re-run.")

tab0, tab1, tab2, tab_live, tab3, tab4, tab5 = st.tabs([
    "🛠️ Site Setup",
    "📂 Data Upload",
    "📈 Predictor",
    "🔴 Live Simulation",
    "⚙️ AI Manager",
    "💰 Bill Calculator",
    "🌞 PowerRECO",
])


# ─────────────────────────────────────────────────────────────────────────────
# TAB 0: SITE SETUP — single editable form for every per-site parameter
# ─────────────────────────────────────────────────────────────────────────────
with tab0:
    st.header("Site Setup")
    _active = get_profile(st.session_state.site_profile_id)
    st.caption(
        f"Configuring **{_active.name}**.  Values come from the site profile "
        f"preset; any edit is stored per-site so switching the profile dropdown "
        f"preserves each site's settings independently.  After editing, click "
        f"**Apply & Re-run** to invalidate cached results and update the strategy."
    )

    # ── Persisted state overview ───────────────────────────────────────────
    st.markdown("##### 💾 Persisted state across all sites")
    st.caption(
        "Uploaded load profiles and trained forecasters are saved per site to "
        "`data/sites/<id>/`.  They auto-load when you re-open the app — no "
        "re-upload or re-train needed.  Site Setup overrides save to "
        f"`{OVERRIDES_PATH.name}`."
    )
    _all_state = all_sites_state_info(list(PROFILES.keys()))
    _state_rows = []
    for s in _all_state:
        _pname = PROFILES[s["site_id"]].name
        _state_rows.append({
            "Site": _pname,
            "Data":       (f"✅ {s['data_rows']:,} rows · {s['data_kb']} KB · {s['data_saved']}"
                            if s["has_data"] else "—"),
            "Forecaster": (f"✅ {s['model_mb']} MB · {s['model_saved']}"
                            if s["has_model"] else "—"),
        })
    st.dataframe(pd.DataFrame(_state_rows), use_container_width=True, hide_index=True)

    _cur_state = site_state_info(_active.id)
    cs1, cs2 = st.columns([3, 1])
    if _cur_state["has_data"] or _cur_state["has_model"]:
        with cs1:
            st.caption(
                f"For **{_active.name}**: "
                + ("data ✅ · "    if _cur_state["has_data"]  else "data ❌ · ")
                + ("model ✅"      if _cur_state["has_model"] else "model ❌")
            )
        with cs2:
            if st.button("🗑️ Clear saved data + model", use_container_width=True,
                          help="Removes the per-site disk files for the active site. "
                               "Overrides (the Site Setup edits below) are kept."):
                clear_site_state(_active.id, also_clear_overrides=False)
                # Drop session_state mirror so a refresh re-bootstraps cleanly
                st.session_state._loaded_site_id = None
                st.session_state.df = None
                st.session_state.forecaster = None
                st.success(f"Cleared disk state for **{_active.name}**.")
                st.rerun()
    else:
        st.caption(f"No saved state yet for **{_active.name}** — upload a load "
                    f"profile in the Data Upload tab and the system will start "
                    f"remembering it.")

    st.divider()

    # Use a per-site key suffix so switching profiles re-mounts widgets with
    # fresh defaults (without this, Streamlit holds the OLD site's values).
    _wk = f"ss_{_active.id}_"
    with st.expander("📍 Location (drives weather + irradiance lookup)", expanded=True):
        l1, l2, l3 = st.columns(3)
        new_lat = l1.number_input(
            "Latitude",  -90.0, 90.0,
            float(_so("lat", _active.lat)), step=0.001, format="%.4f",
            key=_wk + "lat",
            help="Used to pull real ambient temperature + shortwave irradiance "
                 "from Open-Meteo for both training and live forecasts. "
                 "Defaults to Kuala Lumpur.",
        )
        new_lon = l2.number_input(
            "Longitude", -180.0, 180.0,
            float(_so("lon", _active.lon)), step=0.001, format="%.4f",
            key=_wk + "lon",
        )
        new_tz  = l3.text_input(
            "Timezone", _so("timezone", _active.timezone),
            key=_wk + "tz",
            help="Either an IANA name (e.g. 'Asia/Kuala_Lumpur') or 'auto' "
                 "to infer from lat/lon.",
        )
        new_use_weather = st.checkbox(
            "Use real weather data (Open-Meteo)",
            value=bool(st.session_state.use_real_weather),
            key=_wk + "real_weather",
            help="When on, the forecaster pulls actual temperature + irradiance "
                 "from Open-Meteo's archive (past) and forecast (future) endpoints. "
                 "Falls back to a synthetic tropical model on network failure. "
                 "First run takes 1-2 minutes; results are cached on disk.",
        )

    with st.expander("☀️ PV system", expanded=False):
        s1, s2, s3 = st.columns(3)
        new_solar_kwp = s1.number_input(
            "Existing solar capacity (kWp)", min_value=0.0,
            value=float(_so("solar_kwp",
                            _active.expected_solar_kwp if _active.expected_solar_kwp is not None
                            else 0.0)),
            step=10.0, key=_wk + "solar_kwp",
            help="0 if no solar installed yet. Used by the forecaster (for "
                 "net-load shape) and by PowerRECO (as the starting point "
                 "for sizing recommendations).",
        )
        new_roof_area = s2.number_input(
            "Roof area (m²)", 50.0, 10000.0,
            float(_so("roof_area_m2", _active.expected_roof_area_m2)),
            step=50.0, key=_wk + "roof_area",
            help="Caps the maximum solar PV size PowerRECO will recommend.",
        )
        new_panel_w = s3.number_input(
            "Panel wattage (W)", 300, 700,
            int(_so("panel_w", 415)), step=5, key=_wk + "panel_w",
        )
        s4, s5 = st.columns(2)
        new_psh = s4.number_input(
            "Peak sun hours/day", 3.5, 6.0, float(_so("psh", 4.5)), step=0.1,
            key=_wk + "psh",
            help="Malaysia average ~4.5 h/day. Override with site-specific PVGIS value if known.",
        )
        new_solar_cost = s5.number_input(
            "Solar cost (RM/kWp)", 2000.0, 6000.0,
            float(_so("solar_cost", 3500.0)), step=100.0, key=_wk + "solar_cost",
        )

    with st.expander("🔋 Battery & dispatch", expanded=False):
        b1, b2, b3 = st.columns(3)
        new_battery_kwh = b1.number_input(
            "Battery capacity (kWh)", 0.0, 5000.0,
            float(_so("battery_kwh", _active.battery_kwh)), step=10.0,
            key=_wk + "battery_kwh",
            help="0 = no battery. Used by the Manager for dispatch and by "
                 "PowerRECO as the starting battery size.",
        )
        new_c_rate = b2.number_input(
            "C-rate", 0.1, 1.0, float(_so("c_rate", _active.c_rate)), step=0.1,
            key=_wk + "c_rate",
        )
        new_bat_cost = b3.number_input(
            "Battery cost (RM/kWh)", 1500.0, 5000.0,
            float(_so("batt_cost", 2500.0)), step=100.0, key=_wk + "batt_cost",
        )
        b4, b5, b6 = st.columns(3)
        new_md_target = b4.slider(
            "MD target (% of ref peak)", 70, 95,
            int(_so("md_target_pct", _active.md_target_pct) * 100),
            key=_wk + "md_target",
        ) / 100.0
        new_charge_upper = b5.slider(
            "Charge upper threshold (%)", 50, 90,
            int(_so("charge_upper_pct", _active.charge_upper_pct) * 100),
            key=_wk + "charge_upper",
        ) / 100.0
        new_init_soc = b6.slider(
            "Initial SOC (%)", 20, 80,
            int(_so("init_soc_pct", _active.initial_soc_pct) * 100),
            key=_wk + "init_soc",
        ) / 100.0

    with st.expander("🚙 EV chargers", expanded=False):
        _ev_default = _active.loads.get("ev")
        _ev_charger = _ev_default.ev_chargers[0] if (_ev_default and _ev_default.ev_chargers) else None
        e1, e2, e3 = st.columns(3)
        new_ev_count = e1.number_input(
            "# chargers", 0, 100,
            int(_so("ev_count", _ev_charger.count if _ev_charger else 4)),
            key=_wk + "ev_count",
        )
        new_ev_kw = e2.number_input(
            "kW each", 3.6, 250.0,
            float(_so("ev_kw_each", _ev_charger.kw_each if _ev_charger else 22.0)),
            step=1.0, key=_wk + "ev_kw",
        )
        new_ev_kind = e3.selectbox(
            "Connector",
            ["AC", "DC"],
            index=0 if _so("ev_kind", _ev_charger.kind if _ev_charger else "AC") == "AC" else 1,
            key=_wk + "ev_kind",
        )
        e4, e5 = st.columns(2)
        _ev_window = _ev_default.allowed_window if (_ev_default and _ev_default.allowed_window) else (18, 8)
        new_ev_ws = e4.number_input(
            "Charge window start (h)", 0, 23,
            int(_so("ev_window_start", _ev_window[0])),
            key=_wk + "ev_ws",
        )
        new_ev_we = e5.number_input(
            "Charge window end (h)",   0, 23,
            int(_so("ev_window_end",   _ev_window[1])),
            key=_wk + "ev_we",
            help="Window wraps overnight: 18 → 8 means 18:00 to 08:00 next morning.",
        )

    with st.expander("❄️ HVAC", expanded=False):
        _hvac = _active.loads.get("hvac")
        _hwin = _hvac.protected_window if (_hvac and _hvac.protected_window) else (9, 17)
        h1, h2, h3 = st.columns(3)
        new_hvac_protect_start = h1.number_input(
            "Protected from (h)", 0, 23,
            int(_so("hvac_protect_start", _hwin[0])),
            key=_wk + "hvac_ws",
        )
        new_hvac_protect_end = h2.number_input(
            "Protected to (h)", 0, 23,
            int(_so("hvac_protect_end", _hwin[1])),
            key=_wk + "hvac_we",
        )
        new_hvac_cut = h3.number_input(
            "Max cut % outside protect", 0, 100,
            int(_so("hvac_max_cut_pct", (_hvac.max_cut_pct * 100) if _hvac else 15)),
            key=_wk + "hvac_cut",
        )

    with st.expander("💰 Tariff & financial", expanded=False):
        t1, t2, t3 = st.columns(3)
        tariff_options = list(TARIFF_META.keys())
        _cur_tariff = _so("tariff_code", st.session_state.tariff_code)
        new_tariff = t1.selectbox(
            "Tariff code", tariff_options,
            index=tariff_options.index(_cur_tariff) if _cur_tariff in tariff_options else 2,
            format_func=lambda c: f"{c} — {TARIFF_META[c]['name']}",
            key=_wk + "tariff",
            help="Auto-detected from your data, overridable here.",
        )
        new_icpt = t2.number_input(
            "ICPT (sen/kWh)", -10.0, 20.0,
            float(_so("icpt_sen", 0.0)), step=0.5, key=_wk + "icpt",
        )
        new_nem = t3.number_input(
            "NEM buyback rate (RM/kWh)", 0.20, 0.50,
            float(_so("nem_rate", 0.31)), step=0.01, key=_wk + "nem",
        )
        new_budget = st.number_input(
            "Max CAPEX budget (RM, 0 = no cap)", 0, 5_000_000,
            int(_so("budget_rm", 0)), step=50_000, key=_wk + "budget",
            help="Used by the PowerRECO optimizer to find the best NPV within budget.",
        )

    # ── Apply + Re-run ────────────────────────────────────────────────────
    st.divider()
    a1, a2, a3 = st.columns([2, 2, 1])

    apply_btn = a1.button("💾 Apply changes", use_container_width=True,
                           help="Store edits for this site (kept in memory). "
                                "Cached pipeline state is cleared so the next "
                                "Train picks up the new values.")
    rerun_btn = a2.button("🔁 Apply & Re-run full pipeline", type="primary",
                           use_container_width=True,
                           help="Apply + cascading Train → Manager → Bill → PowerRECO. "
                                "Requires data to already be uploaded.")
    reset_btn = a3.button("Reset", use_container_width=True,
                           help="Discard overrides for this site (revert to profile preset).")

    def _commit_overrides():
        _set_so("lat",                new_lat)
        _set_so("lon",                new_lon)
        _set_so("timezone",           new_tz)
        _set_so("solar_kwp",          new_solar_kwp)
        _set_so("roof_area_m2",       new_roof_area)
        _set_so("panel_w",            int(new_panel_w))
        _set_so("psh",                new_psh)
        _set_so("solar_cost",         new_solar_cost)
        _set_so("battery_kwh",        new_battery_kwh)
        _set_so("c_rate",             new_c_rate)
        _set_so("batt_cost",          new_bat_cost)
        _set_so("md_target_pct",      new_md_target)
        _set_so("charge_upper_pct",   new_charge_upper)
        _set_so("init_soc_pct",       new_init_soc)
        _set_so("ev_count",           int(new_ev_count))
        _set_so("ev_kw_each",         new_ev_kw)
        _set_so("ev_kind",            new_ev_kind)
        _set_so("ev_window_start",    int(new_ev_ws))
        _set_so("ev_window_end",      int(new_ev_we))
        _set_so("hvac_protect_start", int(new_hvac_protect_start))
        _set_so("hvac_protect_end",   int(new_hvac_protect_end))
        _set_so("hvac_max_cut_pct",   int(new_hvac_cut))
        _set_so("tariff_code",        new_tariff)
        _set_so("icpt_sen",           new_icpt)
        _set_so("nem_rate",           new_nem)
        _set_so("budget_rm",          int(new_budget))
        st.session_state.use_real_weather = bool(new_use_weather)
        st.session_state.tariff_code = new_tariff
        _invalidate_downstream()
        # Persist to disk so a browser refresh doesn't wipe edits
        save_overrides(st.session_state.site_overrides)

    if apply_btn:
        _commit_overrides()
        st.success(f"Applied {len(_site_overrides_for(_active.id))} parameters "
                    f"to **{_active.name}**. Saved to `{OVERRIDES_PATH.name}` — "
                    f"survives browser refresh. Re-train the Predictor to update the strategy.")

    if reset_btn:
        st.session_state.site_overrides[_active.id] = {}
        _invalidate_downstream()
        save_overrides(st.session_state.site_overrides)
        st.info(f"Overrides cleared for **{_active.name}** — back to profile preset.")
        st.rerun()

    if rerun_btn:
        if st.session_state.df is None:
            st.error("Upload a load profile first (Data Upload tab).")
        else:
            _commit_overrides()
            try:
                # 1. Train predictor with new params (with progress bar)
                fc_kwargs = profile_predictor_kwargs(_active)
                fc_kwargs["lat"] = new_lat if new_use_weather else None
                fc_kwargs["lon"] = new_lon if new_use_weather else None
                fc_kwargs["timezone"] = new_tz if new_use_weather else "auto"
                fc = DirectMultiStepForecaster(
                    capacity_kwp=float(new_solar_kwp), **fc_kwargs,
                )
                import time
                _prog = st.progress(0.0)
                _msg  = st.empty()
                _t0 = time.time()
                def _on_h(i, n, h):
                    _prog.progress(i / n)
                    elapsed = time.time() - _t0
                    eta = elapsed * (n - i) / max(i, 1)
                    _msg.caption(
                        f"Training horizon {i}/{n} (h={h}) · "
                        f"{elapsed:.0f}s elapsed · ~{eta:.0f}s remaining"
                    )
                metrics = fc.fit(st.session_state.df, verbose=False,
                                  progress_callback=_on_h)
                _prog.progress(1.0)
                _msg.caption(f"Train done in {time.time()-_t0:.1f}s · "
                              f"now running Manager + Bill + PowerRECO…")
                st.session_state.forecaster = fc
                save_forecaster(st.session_state.site_profile_id, fc)
                with st.spinner("Running Manager + Bill + PowerRECO…"):
                    fr = fc.forecast(output_steps=48)
                    st.session_state.forecast_result = fr

                    # 2. Manager — build loads dict from overrides
                    loads_dict = {
                        "ev": {
                            "name": "EV Chargers",
                            "proportion": 0.30, "max_cut_pct": 0.40,
                            "kind": "ev",
                            "allowed_window": [int(new_ev_ws), int(new_ev_we)],
                            "ev_chargers": [{
                                "count": int(new_ev_count),
                                "kw_each": float(new_ev_kw),
                                "kind": new_ev_kind,
                            }],
                            "ev_total_kw": int(new_ev_count) * float(new_ev_kw),
                        },
                        "hvac": {
                            "name": "HVAC",
                            "proportion": 0.40,
                            "max_cut_pct": int(new_hvac_cut) / 100,
                            "kind": "hvac",
                            "protected_window": [int(new_hvac_protect_start), int(new_hvac_protect_end)],
                        },
                        "misc": {
                            "name": "Misc Loads", "proportion": 0.30,
                            "max_cut_pct": 0.20, "kind": "misc",
                        },
                    }
                    mgr_results = _run_manager_on_df(
                        st.session_state.df, loads_dict, list(loads_dict.keys()),
                        battery_kwh=float(new_battery_kwh) if new_battery_kwh > 0 else 200.0,
                        peak_target_pct=new_md_target,
                        charge_upper_pct=new_charge_upper,
                        c_rate=new_c_rate, init_soc_pct=new_init_soc,
                        bat_eff=0.95, peak_ref_kva=None,
                    )
                    st.session_state.manager_results      = mgr_results
                    st.session_state.manager_df_optimized = manager_results_to_sam_df(mgr_results)
                    st.session_state.manager_df_original  = manager_results_to_original_df(mgr_results)
                    st.session_state.powerreco_df         = manager_results_to_powerreco_df(mgr_results)

                    # 3. PowerRECO sizing + ROI
                    solar = calculate_solar_sizing(new_roof_area, int(new_panel_w), new_psh)
                    st.session_state.solar_sizing = solar
                    batt = calculate_battery_sizing(st.session_state.powerreco_df) \
                            if st.session_state.powerreco_df is not None else \
                            {"min_capacity_kwh_commercial": float(new_battery_kwh) or 100.0,
                             "md_reduction_kw": 0.0, "n_days_analyzed": 0,
                             "spike_note": "No Manager output."}
                    st.session_state.battery_sizing = batt
                    roi = calculate_roi(
                        solar_kwp=solar["system_kwp"],
                        battery_kwh=float(batt["min_capacity_kwh_commercial"]),
                        monthly_generation_kwh=solar["monthly_generation_kwh_avg"],
                        md_reduction_kw=float(batt.get("md_reduction_kw", 0.0)),
                        schedule_key=st.session_state.tariff_schedule_key,
                        tariff=new_tariff,
                        solar_cost_per_kwp=new_solar_cost,
                        battery_cost_per_kwh=new_bat_cost,
                    )
                    st.session_state.roi = roi
                st.success(
                    f"✅ Re-run complete · forecaster mean MAPE {metrics['mean_mape']:.2f}% · "
                    f"Manager processed {len(mgr_results):,} intervals · "
                    f"new payback {roi['simple_payback_years']:.1f} yrs · "
                    f"new NPV RM {roi['npv_25yr_rm']:,.0f}"
                )
            except Exception as e:
                st.error(f"Re-run failed: {e}")
                st.code(traceback.format_exc())

    # Live preview of what'll be used
    st.divider()
    st.subheader("Current effective values for this site")
    st.caption("Yellow = override · grey = profile preset.")
    _ov = _site_overrides_for(_active.id)
    eff_rows = [
        ("Latitude",           f"{_so('lat', _active.lat):.4f}",                 "lat" in _ov),
        ("Longitude",          f"{_so('lon', _active.lon):.4f}",                 "lon" in _ov),
        ("Real weather",       "On" if st.session_state.use_real_weather else "Off", False),
        ("Solar capacity",     f"{_so('solar_kwp', _active.expected_solar_kwp or 0):.1f} kWp", "solar_kwp" in _ov),
        ("Roof area",          f"{_so('roof_area_m2', _active.expected_roof_area_m2):.0f} m²", "roof_area_m2" in _ov),
        ("Battery capacity",   f"{_so('battery_kwh', _active.battery_kwh):.0f} kWh",            "battery_kwh" in _ov),
        ("MD target",          f"{_so('md_target_pct', _active.md_target_pct)*100:.0f}%",      "md_target_pct" in _ov),
        ("EV chargers",        f"{int(_so('ev_count', 4))} × {_so('ev_kw_each', 22):.0f} kW {_so('ev_kind','AC')}", any(k in _ov for k in ('ev_count','ev_kw_each','ev_kind'))),
        ("EV charge window",   f"{int(_so('ev_window_start', 18)):02d}:00 → {int(_so('ev_window_end', 8)):02d}:00", any(k in _ov for k in ('ev_window_start','ev_window_end'))),
        ("HVAC protected",     f"{int(_so('hvac_protect_start', 9)):02d}:00 → {int(_so('hvac_protect_end', 17)):02d}:00", any(k in _ov for k in ('hvac_protect_start','hvac_protect_end'))),
        ("Tariff",             _so("tariff_code", st.session_state.tariff_code),               "tariff_code" in _ov),
        ("Budget cap",          f"RM {int(_so('budget_rm', 0)):,}" if _so('budget_rm', 0) else "No cap", "budget_rm" in _ov),
    ]
    eff_df = pd.DataFrame(eff_rows, columns=["Parameter", "Value", "Overridden"])
    st.dataframe(eff_df, use_container_width=True, hide_index=True)


# ─────────────────────────────────────────────────────────────────────────────
# TAB 1: DATA UPLOAD
# ─────────────────────────────────────────────────────────────────────────────
with tab1:
    st.header("Load Profile Upload")
    st.markdown(
        "Upload an Excel or CSV file containing your site's half-hourly load data. "
        "Supported formats: all four TNB meter export formats (SOL, E, SUN, MI2) and plain CSV."
    )

    uploaded = st.file_uploader(
        "Choose a load profile file",
        type=["csv", "xlsx", "xls"],
        help="Half-hourly (30-min) interval data. Columns auto-detected.",
    )

    if uploaded is not None:
        try:
            with st.spinner("Parsing file…"):
                content = uploaded.read()
                fname   = uploaded.name
                # parse_uploaded_data handles all formats via bytes — avoids Path() issues
                df = parse_uploaded_data(content, fname)
                # Trim to canonical predictor columns (drop kw_net / kvar_net / kva)
                for col in ("kw_net", "kvar_net", "kva"):
                    if col in df.columns:
                        df = df.drop(columns=[col])

            st.session_state.df = df
            summ = summarize(df)
            summ["max_kw_import"] = float(df["kw_import"].max())
            st.session_state.file_summary = summ

            has_solar, solar_reason = detect_has_solar(df)
            solar_info = {"has_solar": has_solar, "reason": solar_reason}
            if has_solar:
                solar_est = estimate_solar_capacity_kwp(df)
                solar_info.update(solar_est)
            st.session_state.solar_info = solar_info

            tariff_code, tariff_reason, tariff_stats = auto_detect_tariff(df)
            st.session_state.tariff_code = tariff_code

            # Persist for the active site — survives browser refresh + restart.
            _active_site = st.session_state.site_profile_id
            save_load_profile(_active_site, df)

            # Clear auto-run flag so the Manager re-runs on the fresh data.
            st.session_state["_auto_manager_site"] = None

            st.success(f"Loaded {summ['rows']:,} intervals  |  "
                       f"{summ['days']} days  |  "
                       f"Peak {summ['max_kw_import']:.1f} kW · "
                       f"Saved to `data/sites/{_active_site}/load_profile.joblib`")

            col1, col2, col3 = st.columns(3)
            col1.metric("Mean kW", f"{summ['mean_kw_import']:.1f}")
            col2.metric("Peak kW", f"{summ['max_kw_import']:.1f}")
            col3.metric("Days of data", summ["days"])

            st.plotly_chart(_plot_load(df, "Raw Load Profile"), use_container_width=True)

            with st.expander("Tariff auto-detection"):
                st.write(f"**Detected tariff:** {tariff_code} — {TARIFF_META[tariff_code]['name']}")
                st.write(tariff_reason)
                st.json(tariff_stats)

            if has_solar:
                with st.expander("Solar detection"):
                    st.warning(solar_reason)
                    if "capacity_kwp" in solar_info:
                        st.write(f"Estimated capacity: **{solar_info['capacity_kwp']} kWp**")

            # ── Data quality checks (run on the just-parsed df) ────────
            st.divider()
            st.subheader("🩺 Data Quality Check")
            st.caption(
                "Auto-detected issues that hurt forecast accuracy. Fix these "
                "at the source if possible — they explain a lot of the "
                "'MAPE is bad' problem."
            )
            _dq_issues = _check_data_quality(df)
            for _iss in _dq_issues:
                sev = _iss["severity"]
                msg = f"**{_iss['title']}** — {_iss['detail']}"
                if sev == "error":
                    st.error(msg)
                elif sev == "warning":
                    st.warning(msg)
                elif sev == "ok":
                    st.success(msg)
                else:
                    st.info(msg)

            # ── Pre-filled assumptions (transparency) ──────────────────
            st.divider()
            st.subheader("🧾 Pre-filled Assumptions")
            st.caption(
                "Every value the system is using and where it came from. "
                "Green = derived from your data · Blue = you set it · "
                "Purple = site profile preset · Amber = hard default."
            )
            _assump_rows = _collect_assumptions()
            _assump_df = pd.DataFrame(_assump_rows)
            st.dataframe(_assump_df, use_container_width=True, hide_index=True)

        except Exception as e:
            st.error(f"Failed to parse file: {e}")
            with st.expander("Traceback"):
                st.code(traceback.format_exc())

    elif st.session_state.df is not None:
        st.info("File already loaded. See sidebar for summary.")
        st.plotly_chart(_plot_load(st.session_state.df, "Loaded Load Profile"),
                        use_container_width=True)


# ─────────────────────────────────────────────────────────────────────────────
# TAB 2: PREDICTOR
# ─────────────────────────────────────────────────────────────────────────────
with tab2:
    st.header("AI Load Predictor")

    if st.session_state.df is None:
        st.warning("Upload a load profile in the Data Upload tab first.")
        st.stop()

    df = st.session_state.df
    solar_info = st.session_state.solar_info or {}
    profile = get_profile(st.session_state.site_profile_id)

    # Override > profile preset > auto-detect > 0
    default_kwp = _so("solar_kwp",
        profile.expected_solar_kwp if profile.expected_solar_kwp is not None
        else solar_info.get("capacity_kwp", 0.0))
    default_n_est = profile.predictor.n_estimators
    default_lr    = profile.predictor.learning_rate
    use_real_weather = bool(st.session_state.use_real_weather)

    st.caption(
        f"Active site profile: **{profile.name}** · "
        f"tuned bias_window={profile.predictor.bias_window}, "
        f"n_estimators={profile.predictor.n_estimators}. "
        f"Solar capacity + location come from **Site Setup** (Tab 0); "
        f"override model knobs below."
    )

    with st.expander("Model configuration", expanded=True):
        col1, col2, col3 = st.columns(3)
        capacity_kwp = col1.number_input(
            "Solar capacity (kWp)", min_value=0.0, value=float(default_kwp), step=10.0,
            help="From Site Setup → override there to persist per-site.")
        n_estimators = col2.number_input("LightGBM rounds", 100, 500, int(default_n_est), step=50)
        lr           = col3.number_input("Learning rate", 0.01, 0.2, float(default_lr), step=0.01)

        # Real-weather + lat/lon now controlled from Site Setup tab.
        lat = float(_so("lat", profile.lat))
        lon = float(_so("lon", profile.lon))
        st.caption(
            f"📍 Site location: **{lat:.4f}, {lon:.4f}** · "
            f"Real weather: **{'ON' if use_real_weather else 'OFF'}** "
            f"(edit in Site Setup tab)."
        )
        if not use_real_weather:
            st.warning(
                "⚠️  **Real weather is OFF — forecast accuracy is being capped.**  \n"
                "Synthetic temperature + irradiance are *deterministic functions "
                "of time-of-day*, so they encode the same information as the "
                "`hour_sin / cos` features the model already has — they add **zero "
                "extra signal**.  All 17 forward-looking weather features "
                "(`fut_irrad_h{2,6,12,24,48}`, rolling future means, etc.) are "
                "currently fed synthetic values too.  \n\n"
                "Toggle **'Use real weather (Open-Meteo)'** in the 🛠️ Site Setup tab "
                "to fetch actual ambient temp + shortwave irradiance for your "
                "lat/lon.  First fetch caches to `data/weather_cache/`; subsequent "
                "runs are offline.  **Expected MAPE improvement: 10–30 % on solar "
                "sites, 5–15 % on HVAC-driven sites.**"
            )

    col_train, col_cv = st.columns([2, 1])
    with col_train:
        if st.button("Train Forecaster", type="primary", use_container_width=True):
            try:
                fc_kwargs = profile_predictor_kwargs(profile)
                fc_kwargs.update({
                    "n_estimators":  int(n_estimators),
                    "learning_rate": lr,
                })
                if use_real_weather:
                    fc_kwargs["lat"]      = float(lat)
                    fc_kwargs["lon"]      = float(lon)
                    fc_kwargs["timezone"] = _so("timezone", profile.timezone)
                else:
                    fc_kwargs["lat"] = None
                    fc_kwargs["lon"] = None
                fc = DirectMultiStepForecaster(
                    capacity_kwp=capacity_kwp,
                    **fc_kwargs,
                )

                # Visible progress so the user knows training is alive.
                # 24 horizons × 3 quantiles = 72 boosters + 24 baselines.
                import time
                _prog_bar = st.progress(0.0)
                _prog_msg = st.empty()
                _t0 = time.time()
                def _on_horizon(i_done, n_total, h):
                    pct = i_done / n_total
                    elapsed = time.time() - _t0
                    eta = elapsed * (n_total - i_done) / max(i_done, 1)
                    _prog_bar.progress(pct)
                    _prog_msg.caption(
                        f"Training horizon {i_done}/{n_total} (h={h} = "
                        f"{h*30} min ahead) · {elapsed:.0f}s elapsed · "
                        f"~{eta:.0f}s remaining"
                    )
                metrics = fc.fit(df, verbose=False, progress_callback=_on_horizon)
                _prog_bar.progress(1.0)
                _prog_msg.caption(f"Done in {time.time()-_t0:.1f}s")

                st.session_state.forecaster = fc
                save_forecaster(st.session_state.site_profile_id, fc)
                st.success(
                    f"Trained {metrics['n_models_trained']} models in "
                    f"{time.time()-_t0:.1f}s. "
                    f"Mean MAPE: {metrics['mean_mape']:.2f}%  |  "
                    f"MAPE@24h: {metrics.get('mape_at_h24', 0):.2f}%  ·  "
                    f"Saved to `data/sites/{st.session_state.site_profile_id}/forecaster.joblib`"
                )
            except Exception as e:
                st.error(f"Training failed: {e}")
                st.code(traceback.format_exc())

    with col_cv:
        if st.button("Run Cross-Validation", use_container_width=True):
            if st.session_state.forecaster is None:
                st.warning("Train the forecaster first.")
            else:
                cap = st.session_state.forecaster.capacity_kwp
                with st.spinner("Running expanding-window CV…"):
                    try:
                        _cv_kwargs = profile_predictor_kwargs(profile)
                        _cv_kwargs.update({"n_estimators": int(n_estimators),
                                           "learning_rate": lr})
                        # CV refits per fold; skip real-weather to avoid N×API calls.
                        _cv_kwargs["lat"] = None
                        _cv_kwargs["lon"] = None
                        cv_summary = expanding_window_cv(
                            df,
                            forecaster_factory=lambda: DirectMultiStepForecaster(
                                capacity_kwp=cap, **_cv_kwargs),
                            n_splits=4, min_train_days=21, val_block_days=5,
                        )
                        st.code(format_cv_report(cv_summary))
                    except Exception as e:
                        st.error(f"CV failed: {e}")

    if st.session_state.forecaster is not None:
        st.divider()
        st.subheader("24-Hour Forecast")
        fc = st.session_state.forecaster

        try:
            fr = fc.forecast(output_steps=48)
            st.session_state.forecast_result = fr

            fig = go.Figure()
            # Historical tail (last 48 readings)
            hist_tail = fc.history.tail(48)
            fig.add_trace(go.Scatter(
                x=hist_tail["timestamp"], y=hist_tail["kw_import"],
                mode="lines", name="Historical", line=dict(color="#666"),
            ))
            # Forecast bands
            fig.add_trace(go.Scatter(
                x=list(fr.timestamps) + list(fr.timestamps[::-1]),
                y=list(fr.p90) + list(fr.p10[::-1]),
                fill="toself", fillcolor="rgba(31,119,180,0.15)",
                line=dict(color="rgba(255,255,255,0)"), name="P10–P90 band",
            ))
            fig.add_trace(go.Scatter(
                x=fr.timestamps, y=fr.median,
                mode="lines", name="Median forecast",
                line=dict(color="#1f77b4", width=2),
            ))
            fig.update_layout(title="48-Step Ahead Forecast (24 hours)",
                              xaxis_title="Time", yaxis_title="kW",
                              hovermode="x unified", height=400)
            st.plotly_chart(fig, use_container_width=True)

            # Peak table
            peaks = fc.detect_peaks(fr)
            st.write("**Top 3 Predicted Peaks**")
            st.dataframe(peaks.style.format({
                "predicted_kw": "{:.1f}",
                "lower_bound_kw": "{:.1f}",
                "upper_bound_kw": "{:.1f}",
            }), use_container_width=True)

            # Download forecast CSV
            csv_str = forecast_result_to_csv(fr)
            st.download_button(
                "Download Forecast CSV",
                data=csv_str.encode(),
                file_name="forecast_output.csv",
                mime="text/csv",
            )

            # ── SCENARIO ADJUSTMENT FACTOR ────────────────────────────────
            st.divider()
            st.subheader("🎛️ Scenario Adjustment Factor")
            st.caption(
                "Apply an operational scaling factor to account for a known upcoming "
                "event that the model cannot see (e.g. a large production order, a "
                "partial closure, a holiday).  The adjusted forecast is shown visually "
                "and fed to the live Manager so battery dispatch is planned for the "
                "expected conditions, not just the historical pattern."
            )

            _fac_c1, _fac_c2, _fac_c3 = st.columns(3)

            with _fac_c1:
                _user_factor = st.number_input(
                    "Scaling factor",
                    min_value=0.10, max_value=5.00,
                    value=float(st.session_state.get("forecast_user_factor", 1.0)),
                    step=0.05, format="%.2f",
                    help=(
                        "1.00 = no adjustment (model used as-is).\n"
                        "1.50 = expect 50% MORE load (big order, event, extra shift).\n"
                        "0.30 = expect 70% LESS load (closure, holiday, equipment off)."
                    ),
                    key="pred_user_factor_input",
                )

            _horizon_opts = {
                "2 h  (4 steps)":  4,
                "4 h  (8 steps)":  8,
                "8 h  (16 steps)": 16,
                "12 h (24 steps)": 24,
                "24 h (48 steps)": 48,
            }
            with _fac_c2:
                _horizon_label = st.selectbox(
                    "Applies to next",
                    list(_horizon_opts.keys()),
                    index=4,
                    key="pred_factor_horizon_sel",
                    help="How far ahead the factor is applied. Beyond this window the model's raw forecast resumes.",
                )
            _n_steps_fac = _horizon_opts[_horizon_label]

            with _fac_c3:
                _user_conf = st.slider(
                    "Confidence in estimate", 50, 100,
                    int(st.session_state.get("forecast_user_confidence", 0.80) * 100),
                    step=5, format="%d%%",
                    key="pred_user_confidence_sl",
                    help=(
                        "How certain are you in the factor?\n"
                        "100% = certain → bands scale proportionally only.\n"
                        "50%  = uncertain → bands widen significantly to cover estimation error."
                    ),
                ) / 100.0

            # Persist to session state for Manager propagation
            st.session_state["forecast_user_factor"]     = _user_factor
            st.session_state["forecast_user_confidence"] = _user_conf
            st.session_state["forecast_factor_steps"]    = _n_steps_fac

            if abs(_user_factor - 1.0) < 0.01:
                st.caption(
                    "Factor = 1.00 — no adjustment.  "
                    "Raw model forecast used for display and Manager dispatch."
                )
            else:
                _change_pct = abs(_user_factor - 1.0) * 100
                if _user_factor > 1.0:
                    st.info(
                        f"**Load scaled UP by {_change_pct:.0f}%** for the next "
                        f"{_horizon_label.strip()}.  "
                        "Typical: large production order, additional shift, onsite event.  "
                        f"Manager will pre-charge the battery more aggressively to handle "
                        "the higher expected peak."
                    )
                else:
                    st.warning(
                        f"**Load scaled DOWN by {_change_pct:.0f}%** for the next "
                        f"{_horizon_label.strip()}.  "
                        "Typical: partial or full site closure, public holiday, "
                        "major equipment shutdown.  "
                        "Manager will hold back on charging to avoid over-reserving capacity."
                    )

                # Compute adjusted arrays
                _adj_median, _adj_p10, _adj_p90 = _apply_user_factor(
                    fr.median, fr.p10, fr.p90,
                    _user_factor, _user_conf, _n_steps_fac,
                )

                # Adjusted forecast chart — raw as dashed grey reference
                _ts_list = list(fr.timestamps)
                _fig_adj = go.Figure()
                _fig_adj.add_trace(go.Scatter(
                    x=hist_tail["timestamp"], y=hist_tail["kw_import"],
                    mode="lines", name="Historical",
                    line=dict(color="#6b7280", width=1.5),
                ))
                _fig_adj.add_trace(go.Scatter(
                    x=_ts_list, y=fr.median,
                    mode="lines", name="Raw model forecast (reference)",
                    line=dict(color="#9ca3af", width=1.5, dash="dot"),
                ))
                _fig_adj.add_trace(go.Scatter(
                    x=_ts_list + _ts_list[::-1],
                    y=list(_adj_p90) + list(_adj_p10[::-1]),
                    fill="toself",
                    fillcolor=(
                        "rgba(239,68,68,0.13)" if _user_factor > 1.0
                        else "rgba(59,130,246,0.13)"
                    ),
                    line=dict(color="rgba(255,255,255,0)"),
                    name="Adjusted P10–P90 band",
                ))
                _fig_adj.add_trace(go.Scatter(
                    x=_ts_list, y=_adj_median,
                    mode="lines",
                    name=f"Adjusted forecast (×{_user_factor:.2f}, {_user_conf*100:.0f}% conf.)",
                    line=dict(
                        color="#ef4444" if _user_factor > 1.0 else "#3b82f6",
                        width=2.5,
                    ),
                ))
                # Mark the end of the factor window if not full horizon
                if _n_steps_fac < len(_ts_list):
                    _win_end_ts = _ts_list[_n_steps_fac - 1]
                    _y_top = float(_adj_p90.max()) * 1.05
                    _fig_adj.add_trace(go.Scatter(
                        x=[_win_end_ts, _win_end_ts], y=[0, _y_top],
                        mode="lines", name="Factor window end",
                        line=dict(color="#f59e0b", width=2, dash="dash"),
                        hoverinfo="skip",
                    ))
                _fig_adj.update_layout(
                    title=(
                        f"Adjusted forecast — ×{_user_factor:.2f} for next "
                        f"{_horizon_label.strip()}  |  "
                        f"confidence {_user_conf*100:.0f}%"
                    ),
                    xaxis_title="Time", yaxis_title="kW",
                    hovermode="x unified", height=420,
                    legend=dict(orientation="h", yanchor="bottom", y=1.02),
                )
                st.plotly_chart(_fig_adj, use_container_width=True)

                # Comparison summary
                _sc1, _sc2, _sc3, _sc4 = st.columns(4)
                _raw_pk  = float(fr.median.max())
                _adj_pk  = float(_adj_median.max())
                _raw_avg = float(fr.median.mean())
                _adj_avg = float(_adj_median.mean())
                _sc1.metric("Raw peak",     f"{_raw_pk:.1f} kW")
                _sc2.metric("Adjusted peak", f"{_adj_pk:.1f} kW",
                            delta=f"{_adj_pk - _raw_pk:+.1f} kW", delta_color="off")
                _sc3.metric("Raw mean",     f"{_raw_avg:.1f} kW")
                _sc4.metric("Adjusted mean", f"{_adj_avg:.1f} kW",
                            delta=f"{_adj_avg - _raw_avg:+.1f} kW", delta_color="off")

                _conf_note = (
                    "Uncertainty bands are widened to cover your estimation error "
                    f"(confidence {_user_conf*100:.0f}%)."
                    if _user_conf < 1.0 else
                    "Bands scale proportionally (100% confidence → no additional widening)."
                )
                st.caption(
                    f"**Manager dispatch**: the live Manager sees the adjusted load "
                    f"for the next {_horizon_label.strip()} and plans battery "
                    f"{'pre-charging' if _user_factor > 1.0 else 'hold-back'} accordingly.  "
                    f"{_conf_note}"
                )

        except Exception as e:
            st.error(f"Forecast failed: {e}")
            st.code(traceback.format_exc())


# ─────────────────────────────────────────────────────────────────────────────
# TAB LIVE: LIVE SIMULATION  (the "smarter every day" demo)
# ─────────────────────────────────────────────────────────────────────────────
with tab_live:
    st.header("🔴 Live Forecast Simulation")
    st.caption(
        "Replays the held-out 30 % tail of your uploaded data tick-by-tick, "
        "feeding each new reading to the trained model.  Every 12 ticks (6 h "
        "of revealed data) a warm-start retrain fires.  The live KPIs below "
        "should *sharpen visibly* as more data arrives — that's the "
        "*'smarter every day'* story made real."
    )

    if st.session_state.df is None or st.session_state.forecaster is None:
        st.warning(
            "Need both an uploaded load profile and a trained forecaster. "
            "Use the Data Upload + Predictor tabs first — or pick a site "
            "with a saved model in the sidebar."
        )
        st.stop()

    _fc_live = st.session_state.forecaster
    _df_live = st.session_state.df

    # ── Initialize the simulation lazily on first entry ─────────────────────
    if st.session_state.sim_state is None:
        # Make sure live SOC + manager cache start fresh too
        st.session_state.live_battery_soc_pct  = None
        st.session_state.live_manager_results  = None
        st.session_state.live_manager_last_tick = -1
        try:
            with st.spinner("Initialising live simulation: re-fitting forecaster on "
                             "just the 70% slice (honest train/eval — model genuinely "
                             "hasn't seen the future stream)…"):
                train_df, future_df = split_historical_for_simulation(
                    _df_live, train_fraction=0.7,
                )
                # Honest split: refit on ONLY the 70% so the sim's accuracy
                # numbers reflect real forecasting performance.  The earlier
                # full-fit (used on the Predictor tab + persisted to disk) is
                # untouched — it still represents the "best estimate using
                # all data" for downstream tabs.
                profile = get_profile(st.session_state.site_profile_id)
                fc_kwargs = profile_predictor_kwargs(profile)
                fc_kwargs["n_estimators"]  = _fc_live.n_estimators
                fc_kwargs["learning_rate"] = _fc_live.learning_rate
                if st.session_state.use_real_weather:
                    fc_kwargs["lat"]      = float(_so("lat", profile.lat))
                    fc_kwargs["lon"]      = float(_so("lon", profile.lon))
                else:
                    fc_kwargs["lat"] = None
                    fc_kwargs["lon"] = None
                _fc_honest = DirectMultiStepForecaster(
                    capacity_kwp=_fc_live.capacity_kwp, **fc_kwargs,
                )
                _fc_honest.fit(train_df, verbose=False)
                # Swap into session_state for the sim — leave disk-persisted
                # model alone.
                st.session_state.forecaster = _fc_honest
                _fc_live = _fc_honest
                st.session_state.sim_history_actuals = train_df.copy()
                _bw = profile.predictor.bias_window
                st.session_state.sim_state = initialize_simulation(
                    _fc_honest, future_df,
                    retrain_every_n=12,
                    warm_start_rounds=50,
                    bias_window=_bw,         # per-site tuned (SoL=8, E=24, …)
                )
            st.success(
                f"Sim ready · refit forecaster on **{len(train_df):,}** rows "
                f"(honest eval, no train/test contamination) · "
                f"**{len(future_df):,}** rows held out as the future stream "
                f"({len(future_df) // 48} days)"
            )
        except Exception as e:
            st.error(f"Sim init failed: {e}")
            st.code(traceback.format_exc())
            st.stop()

    sim = st.session_state.sim_state

    # ── Controls (outside the live fragment — toggle state, not redraw constantly)
    st.markdown("##### Playback")
    b1, b2, b3, b4, b5, b6 = st.columns(6)

    if b1.button("⏭ Next tick",     use_container_width=True,
                  disabled=sim.is_finished, key="lsim_next"):
        advance_one_tick(_fc_live, sim)
        st.rerun()
    if b2.button("⏩ +12 (6 h)",     use_container_width=True,
                  disabled=sim.is_finished, key="lsim_12"):
        for _ in range(12):
            if sim.is_finished: break
            advance_one_tick(_fc_live, sim)
        st.rerun()
    if b3.button("⏩ +48 (24 h)",    use_container_width=True,
                  disabled=sim.is_finished, key="lsim_48"):
        with st.spinner("Advancing 24 h of simulated time…"):
            for _ in range(48):
                if sim.is_finished: break
                advance_one_tick(_fc_live, sim)
        st.rerun()
    if b4.button("⏯ Play/Pause",    use_container_width=True,
                  disabled=sim.is_finished, key="lsim_play"):
        st.session_state.sim_playing = not st.session_state.sim_playing
        st.rerun()
    if b5.button("🏁 Run to end",   use_container_width=True,
                  disabled=sim.is_finished, key="lsim_end"):
        with st.spinner(f"Replaying {sim.total_ticks - sim.tick:,} remaining ticks…"):
            while not sim.is_finished:
                advance_one_tick(_fc_live, sim)
        st.rerun()
    if b6.button("🔄 Reset sim",    use_container_width=True, key="lsim_reset"):
        st.session_state.sim_state = None
        st.session_state.sim_playing = False
        # Reset live SOC + Manager cache so the next sim starts fresh.
        st.session_state.live_battery_soc_pct  = None
        st.session_state.live_manager_results  = None
        st.session_state.live_manager_last_tick = -1
        st.session_state.executed_ticks         = []
        st.session_state.live_bill_months       = []
        st.rerun()

    sp1, sp2 = st.columns([3, 2])
    tick_speed = sp1.slider(
        "Auto-play tick rate (seconds per refresh)",
        min_value=0.2, max_value=2.0, value=0.5, step=0.1,
        help="How often the live view refreshes when Play is on. "
             "0.5 s ≈ 2 sim ticks per real second; 2.0 s ≈ 1 tick every 2 s "
             "(easier to read for a live audience).",
        key="lsim_tickrate",
    )
    sp2.markdown(
        f"<div style='padding-top:24px'>"
        f"{'<span style=\"color:#ef4444;font-weight:700\">● LIVE</span>'  + f' — updates every {tick_speed:.1f} s' if st.session_state.sim_playing and not sim.is_finished else '<span style=\"color:#9ca3af\">○ Paused</span>'}"
        f"</div>",
        unsafe_allow_html=True,
    )

    # ── LIVE VIEW (in a fragment so only THIS section re-runs on the timer) ──
    @st.fragment(run_every=f"{tick_speed:.1f}s")
    def _live_view():
        # Always pull the latest sim from session_state inside the fragment —
        # the buttons above may have mutated it between renders.
        sim = st.session_state.sim_state
        fc  = st.session_state.forecaster
        if sim is None or fc is None:
            st.info("Sim was reset — initialising again on next render.")
            return

        # Auto-advance ONE tick per fragment cycle while Play is on.
        # One tick / cycle keeps the chart moving smoothly; the tick_rate
        # slider above controls the cycle interval.
        if st.session_state.sim_playing and not sim.is_finished:
            try:
                advance_one_tick(fc, sim)
            except Exception as e:
                st.session_state.sim_playing = False
                st.error(f"Tick failed: {e}")

        # Re-run the AI Manager on the FRESH forecast so the Manager tab's
        # live fragment sees an up-to-date dispatch plan.  Only recompute
        # when the sim has actually advanced — saves CPU on idle frames.
        if sim.tick != st.session_state.get("live_manager_last_tick"):
            try:
                _lm = _run_manager_on_live_forecast()
                if _lm is not None and len(_lm) > 0:
                    st.session_state.live_manager_results = _lm
                    st.session_state.live_manager_last_tick = sim.tick
                    # ── Accumulate executed decisions ──────────────────────
                    # The first row of the Manager output is always the actual
                    # reading (prepended in _run_manager_on_live_forecast).
                    # Save it as the "executed" decision for this tick so the
                    # Manager tab can show history vs forward plan separately.
                    if len(_lm) > 0:
                        _exec = st.session_state.get("executed_ticks", [])
                        _new_ts = _lm[0].get("timestamp")
                        if not _exec or _exec[-1].get("timestamp") != _new_ts:
                            _exec.append(dict(_lm[0]))
                            st.session_state.executed_ticks = _exec

                            # Month-boundary detection: when the sim crosses into
                            # a new calendar month, compute + store the completed
                            # month's bill so the Calculator tab can display it.
                            if len(_exec) >= 2:
                                try:
                                    _prev_m = pd.to_datetime(_exec[-2]["timestamp"]).to_period("M")
                                    _curr_m = pd.to_datetime(_exec[-1]["timestamp"]).to_period("M")
                                    if _curr_m > _prev_m:
                                        _bms  = st.session_state.get("live_bill_months", [])
                                        _done = {b["month"] for b in _bms}
                                        if str(_prev_m) not in _done:
                                            _m_ticks = [
                                                t for t in _exec
                                                if pd.to_datetime(t["timestamp"]).to_period("M") == _prev_m
                                            ]
                                            if _m_ticks:
                                                _mb = _compute_live_month_bill(
                                                    str(_prev_m), _m_ticks,
                                                    tariff    = st.session_state.tariff_code,
                                                    sched_key = st.session_state.tariff_schedule_key,
                                                    icpt_sen  = float(_so("icpt_sen", 0.0)),
                                                    nem_rate  = float(_so("nem_rate", 0.31)),
                                                )
                                                _bms.append(_mb)
                                                st.session_state.live_bill_months = _bms
                                except Exception:
                                    pass
                    # ── #2 SOC CONTINUITY ──────────────────────────────────
                    # Save the SOC AFTER the first interval (which is the
                    # decision we're "executing" this tick) so the next
                    # Manager run starts from the post-execution state.
                    try:
                        _first = _lm[0]
                        # Manager outputs `battery_soc_pct` as 0–100 — convert
                        # to fraction for initial_soc_pct
                        st.session_state.live_battery_soc_pct = float(
                            _first.get("battery_soc_pct", 50.0)
                        ) / 100.0
                    except Exception:
                        pass
            except Exception:
                pass

        # ── Status metrics ───────────────────────────────────────────────
        s_c1, s_c2, s_c3, s_c4 = st.columns(4)
        s_c1.metric("Tick",                f"{sim.tick:,} / {sim.total_ticks:,}")
        s_c2.metric("Progress",            f"{sim.progress * 100:.1f} %")
        s_c3.metric("Warm-start retrains", f"{len(sim.retrain_log)}")
        if sim.is_finished:
            s_c4.metric("Status", "✅ Finished")
        elif st.session_state.sim_playing:
            s_c4.metric("Status", "🔴 LIVE")
        else:
            s_c4.metric("Status", "⏸ Paused")
        st.progress(float(sim.progress))

        # ── Chart 1: Real vs predicted, real-time as it streams in ───────
        st.subheader("📡 Real vs Predicted — live stream (24-h window)")
        st.caption(
            "Sliding 24-h window centered on **now** (the last revealed tick): "
            "12 h of revealed actuals on the left, 12 h of forward forecast on "
            "the right.  Window advances automatically as the sim ticks forward."
        )

        revealed = sim.revealed_data()
        fr       = sim.current_forecast

        # Anchor "now" to the last revealed timestamp (or the end of training
        # history if no ticks yet).  Window = [now - 12h, now + 12h] = 24 h.
        if fc.history is not None and len(fc.history) > 0:
            now_ts = pd.to_datetime(fc.history["timestamp"].max())
        elif fr is not None:
            now_ts = pd.to_datetime(fr.last_history_ts)
        else:
            now_ts = pd.to_datetime(_df_live["timestamp"].max())
        x_min = now_ts - pd.Timedelta(hours=12)
        x_max = now_ts + pd.Timedelta(hours=12)

        try:
            fig_live = go.Figure()
            # Revealed actuals — clipped to the visible 24-h window
            if len(revealed) > 0:
                rev_in = revealed[
                    (revealed["timestamp"] >= x_min)
                    & (revealed["timestamp"] <= now_ts)
                ]
                if len(rev_in) > 0:
                    fig_live.add_trace(go.Scatter(
                        x=rev_in["timestamp"], y=rev_in["kw_import"],
                        mode="lines+markers", name="Revealed actuals",
                        line=dict(color="#22c55e", width=2.5),
                        marker=dict(size=3),
                    ))
            # If the window's left half isn't yet filled by sim ticks, blend in
            # the training history that falls inside [x_min, now_ts].
            train_full = (st.session_state.sim_history_actuals
                          if st.session_state.sim_history_actuals is not None
                          else _df_live)
            train_in = train_full[
                (train_full["timestamp"] >= x_min)
                & (train_full["timestamp"] <= now_ts)
            ]
            if len(train_in) > 0:
                fig_live.add_trace(go.Scatter(
                    x=train_in["timestamp"], y=train_in["kw_import"],
                    mode="lines", name="Training history (pre-sim)",
                    line=dict(color="#6b7a9a", width=1, dash="dot"),
                ))
            # Forecast: P10-P90 band + median — clipped to next 12 h
            if fr is not None:
                fts = pd.to_datetime(list(fr.timestamps))
                mask = (fts >= now_ts) & (fts <= x_max)
                fts_v = fts[mask]
                med_v = fr.median[mask]
                p10_v = fr.p10[mask]
                p90_v = fr.p90[mask]
                if len(fts_v) > 0:
                    fig_live.add_trace(go.Scatter(
                        x=list(fts_v) + list(fts_v[::-1]),
                        y=list(p90_v) + list(p10_v[::-1]),
                        fill="toself", fillcolor="rgba(0,212,255,0.15)",
                        line=dict(color="rgba(255,255,255,0)"),
                        name="P10–P90 band",
                    ))
                    fig_live.add_trace(go.Scatter(
                        x=fts_v, y=med_v,
                        mode="lines", name="Median forecast",
                        line=dict(color="#00d4ff", width=2.5),
                    ))
            # Vertical "now" marker.  Plotly's add_vline does internal
            # date arithmetic that breaks on numpy 2 + pandas 3, and mixing
            # ISO strings with Timestamp axes throws a different error.
            # Drawing it as a regular Scatter trace sidesteps both code paths.
            _now_yvals = []
            if len(revealed) > 0:
                _now_yvals.append(float(revealed["kw_import"].max()))
            if fr is not None:
                _now_yvals.append(float(fr.p90.max()))
            y_top = (max(_now_yvals) * 1.05) if _now_yvals else 100.0
            fig_live.add_trace(go.Scatter(
                x=[now_ts, now_ts], y=[0, y_top],
                mode="lines", name="● now",
                line=dict(color="#ef4444", width=2),
                hoverinfo="skip",
            ))
            # Retrain markers — only the ones inside the visible window
            if sim.retrain_log:
                try:
                    ret_pairs = []
                    for r in sim.retrain_log:
                        ts = pd.to_datetime(r["reveal_ts"])
                        if not (x_min <= ts <= now_ts):
                            continue
                        match = _df_live.loc[_df_live["timestamp"] == ts, "kw_import"]
                        if len(match):
                            ret_pairs.append((ts, float(match.iloc[0])))
                    if ret_pairs:
                        fig_live.add_trace(go.Scatter(
                            x=[p[0] for p in ret_pairs],
                            y=[p[1] for p in ret_pairs],
                            mode="markers", name="Warm-start retrain",
                            marker=dict(symbol="star", size=15, color="#f59e0b",
                                        line=dict(color="#92400e", width=1.5)),
                        ))
                except Exception:
                    pass
            fig_live.update_layout(
                # Pass Timestamps directly — Plotly's Scatter-x route handles
                # them.  (The earlier add_vline path was the broken one; the
                # axis-range path is fine.)
                xaxis=dict(title="Time", range=[x_min, x_max]),
                yaxis_title="kW",
                hovermode="x unified", height=420,
                legend=dict(orientation="h", yanchor="bottom", y=1.02),
            )
            st.plotly_chart(fig_live, use_container_width=True,
                             key=f"live_chart_{sim.tick}")
        except Exception as e:
            st.warning(f"Live chart render failed: {e}")

        # ── Chart 2: Predicted-vs-actual log (each tick adds verified points)
        try:
            verified = pd.DataFrame([
                r for r in sim.forecast_log if r["actual"] is not None
                and r["horizon_steps"] == 1     # the "30-min-ahead" view
            ])
            if len(verified) > 0:
                verified = verified.sort_values("target_ts").reset_index(drop=True)
                st.subheader("🎯 Forecast accuracy log — predicted vs actual")
                st.caption(
                    f"For every revealed actual, the chart shows the model's "
                    f"h=1 forecast made 30 min earlier (cyan) vs what really "
                    f"happened (green). Lines hugging each other = high accuracy. "
                    f"This is what your live audience cares about. ({len(verified)} "
                    f"verified pairs so far.)"
                )
                fig_pa = go.Figure()
                fig_pa.add_trace(go.Scatter(
                    x=verified["target_ts"], y=verified["actual"],
                    mode="lines+markers", name="Actual",
                    line=dict(color="#22c55e", width=2.5),
                    marker=dict(size=4),
                ))
                fig_pa.add_trace(go.Scatter(
                    x=verified["target_ts"], y=verified["median"],
                    mode="lines+markers", name="Predicted (h=1 forecast)",
                    line=dict(color="#00d4ff", width=2, dash="dash"),
                    marker=dict(size=3, symbol="diamond"),
                ))
                fig_pa.add_trace(go.Scatter(
                    x=list(verified["target_ts"]) + list(verified["target_ts"][::-1]),
                    y=list(verified["p90"]) + list(verified["p10"][::-1]),
                    fill="toself", fillcolor="rgba(0,212,255,0.10)",
                    line=dict(color="rgba(255,255,255,0)"),
                    name="P10–P90 (predicted band)",
                ))
                fig_pa.update_layout(
                    xaxis_title="Time", yaxis_title="kW",
                    hovermode="x unified", height=340,
                    legend=dict(orientation="h", yanchor="bottom", y=1.02),
                )
                st.plotly_chart(fig_pa, use_container_width=True,
                                 key=f"pa_chart_{sim.tick}")
        except Exception as e:
            st.caption(f"(Predicted-vs-actual chart unavailable: {e})")

        # ── KPIs ──────────────────────────────────────────────────────────
        acc = compute_running_accuracy(sim)
        st.subheader("🎯 Live Accuracy")
        if not acc.get("n_verified"):
            st.info("Advance the simulation to accumulate verified rows.")
        else:
            ak1, ak2, ak3, ak4 = st.columns(4)
            ak1.metric("Verified rows",   f"{acc['n_verified']:,}")
            ak2.metric("Overall MAE",      f"{acc['overall_mae']:.1f} kW")
            ak3.metric("Overall MAPE",     f"{acc['overall_mape']:.2f} %")
            cov = acc["within_80ci_pct"] or 0
            ak4.metric("P10–P90 coverage", f"{cov:.1f} %",
                        delta=f"{cov - 80:+.1f} vs 80% target",
                        delta_color="normal" if 75 <= cov <= 90 else "inverse")

            by_h = acc.get("by_horizon", {})
            if by_h:
                hkeys = sorted(by_h.keys())
                fig_h = go.Figure(go.Bar(
                    x=[f"h={h} ({h*30}min)" for h in hkeys],
                    y=[by_h[h]["mape"] for h in hkeys],
                    marker_color="#00d4ff",
                ))
                fig_h.update_layout(
                    title="Per-horizon MAPE — closer in time = more accurate",
                    xaxis_title="Forecast horizon", yaxis_title="MAPE %",
                    height=300,
                )
                st.plotly_chart(fig_h, use_container_width=True,
                                 key=f"h_chart_{sim.tick}")

            # Rolling MAPE-over-time (smarter every day proof)
            try:
                verified_all = pd.DataFrame([
                    r for r in sim.forecast_log if r["actual"] is not None
                ])
                if len(verified_all) >= 24:
                    verified_all = verified_all.sort_values("target_ts").reset_index(drop=True)
                    verified_all["err_pct"] = (
                        (verified_all["median"] - verified_all["actual"]).abs()
                        / verified_all["actual"].clip(lower=1.0)
                    ) * 100
                    verified_all["mape_rolling"] = (
                        verified_all["err_pct"].rolling(24, min_periods=12).mean()
                    )
                    fig_t = go.Figure(go.Scatter(
                        x=verified_all["target_ts"], y=verified_all["mape_rolling"],
                        mode="lines", line=dict(color="#22c55e", width=2.5),
                    ))
                    for r in sim.retrain_log:
                        fig_t.add_vline(x=r["reveal_ts"], line_dash="dot",
                                          line_color="#f59e0b", line_width=1)
                    fig_t.update_layout(
                        title="Live MAPE over time — should trend down with each retrain",
                        xaxis_title="Time of revealed actual",
                        yaxis_title="Rolling MAPE %",
                        height=300, showlegend=False,
                    )
                    st.plotly_chart(fig_t, use_container_width=True,
                                     key=f"t_chart_{sim.tick}")
            except Exception:
                pass

    # Mount the fragment
    _live_view()

    # ── Replace the future stream with a fresh CSV ──────────────────────────
    with st.expander("📥 Replace future stream from a CSV (advanced)"):
        st.caption(
            "Upload a separate file containing only the *future* readings you "
            "want to feed in tick-by-tick.  Use the same canonical schema as "
            "the main upload (timestamp + kw_import + kvar_import etc.). "
            "This resets the current sim."
        )
        live_csv = st.file_uploader(
            "Future-stream CSV/Excel",
            type=["csv", "xlsx", "xls"], key="live_stream_upload",
        )
        if live_csv is not None:
            try:
                future_bytes = live_csv.read()
                from manager.optimizer import parse_uploaded_data
                future_df_user = parse_uploaded_data(future_bytes, live_csv.name)
                for col in ("kw_net", "kvar_net", "kva"):
                    if col in future_df_user.columns:
                        future_df_user = future_df_user.drop(columns=[col])
                # Restart the sim with the uploaded future stream.  Use an
                # explicit None check — `df_a or df_b` raises on DataFrames
                # because they have no single truth value.
                _hist_for_sim = (
                    st.session_state.sim_history_actuals
                    if st.session_state.sim_history_actuals is not None
                    else _df_live
                )
                _fc_live.history = (_hist_for_sim
                                     .sort_values("timestamp")
                                     .reset_index(drop=True))
                _bw = get_profile(st.session_state.site_profile_id).predictor.bias_window
                st.session_state.sim_state = initialize_simulation(
                    _fc_live, future_df_user,
                    retrain_every_n=12, warm_start_rounds=50,
                    bias_window=_bw,
                )
                st.success(
                    f"Future stream replaced — {len(future_df_user):,} rows "
                    f"({len(future_df_user) // 48} days).  Hit Play to advance."
                )
                st.rerun()
            except Exception as e:
                st.error(f"Could not parse future-stream file: {e}")


# ─────────────────────────────────────────────────────────────────────────────
# TAB 3: AI MANAGER
# ─────────────────────────────────────────────────────────────────────────────
with tab3:
    st.header("AI Load Manager")
    st.caption(
        "Configure parameters first, then activate the live simulation or run a "
        "historical analysis.  Discharge formula: dis_kw = kW − √(trigger² − kVAR²)."
    )

    if st.session_state.df is None:
        st.warning("Upload a load profile first.")
        st.stop()

    df      = st.session_state.df
    profile = get_profile(st.session_state.site_profile_id)

    # ═════════════════════════════════════════════════════════════════════
    # SECTION A: CONFIGURATION — set ALL parameters before anything runs
    # ═════════════════════════════════════════════════════════════════════
    st.subheader("⚙️ Manager Configuration")
    st.caption(
        "Set battery capacity, dispatch thresholds, and load priorities here.  "
        "Click **Apply to live simulation** to activate — the live view below "
        "will use these settings from the next sim tick onwards."
    )

    with st.expander("Battery & dispatch settings", expanded=True):
        c1, c2, c3 = st.columns(3)
        battery_kwh  = c1.number_input(
            "Battery capacity (kWh)", 50.0, 5000.0,
            float(_so("battery_kwh", profile.battery_kwh)) or profile.battery_kwh,
            step=50.0,
        )
        c_rate       = c2.number_input(
            "C-rate", 0.1, 1.0, float(_so("c_rate", profile.c_rate)), step=0.1,
            help="Max charge/discharge rate as fraction of capacity per hour",
        )
        bat_eff      = c3.number_input("One-way efficiency", 0.80, 1.00, 0.95, step=0.01)

        c4, c5, c6 = st.columns(3)
        peak_target  = c4.slider(
            "Peak target (% of ref peak)", 70, 95,
            int(_so("md_target_pct", profile.md_target_pct) * 100),
        ) / 100
        charge_upper = c5.slider(
            "Charge upper threshold (% of ref peak)", 50, 90,
            int(_so("charge_upper_pct", profile.charge_upper_pct) * 100),
        ) / 100
        init_soc     = c6.slider(
            "Initial SOC (%)", 20, 80,
            int(_so("init_soc_pct", profile.initial_soc_pct) * 100),
        ) / 100

        c7, c8 = st.columns(2)
        peak_ref_kva = c7.number_input(
            "Reference peak kVA (0 = auto rolling 30-day)", 0.0, 10000.0, 0.0, step=10.0,
        )
        pre_md_hours = c8.number_input("Pre-MD boost window (hours)", 0, 4, 2)

    st.subheader("Load configuration")
    st.caption(
        f"Loads pre-populated from **{profile.name}**.  "
        "EV charging window and HVAC protect hours feed the window-aware dispatch logic."
    )

    loads: dict          = {}
    priority_order: list = []

    _profile_load_keys = list(profile.loads.keys())
    _load_cols         = st.columns(len(_profile_load_keys))
    for _col, lk in zip(_load_cols, _profile_load_keys):
        ld = profile.loads[lk]
        with _col:
            st.markdown(f"**{ld.name}**  \n*{ld.kind}*")
            lname = st.text_input(f"Name [{lk}]", ld.name, key=f"mgr_name_{lk}",
                                   label_visibility="collapsed")
            lprop = st.number_input(f"Proportion [{lk}]", 0.0, 1.0,
                                     float(ld.proportion), step=0.05, key=f"mgr_prop_{lk}")
            lcut  = st.number_input(f"Max cut % [{lk}]", 0, 100,
                                     int(ld.max_cut_pct * 100), key=f"mgr_cut_{lk}")
            entry = {
                "name": lname, "proportion": lprop,
                "max_cut_pct": lcut / 100, "kind": ld.kind,
            }

            if ld.kind == "ev":
                st.caption("🚙 EV chargers")
                _ev0 = ld.ev_chargers[0] if ld.ev_chargers else None
                e_count = st.number_input("# chargers", 0, 50,
                                           int(_so("ev_count", _ev0.count if _ev0 else 4)),
                                           key=f"mgr_evcount_{lk}")
                e_kw    = st.number_input("kW each", 3.6, 250.0,
                                           float(_so("ev_kw_each", _ev0.kw_each if _ev0 else 22.0)),
                                           step=1.0, key=f"mgr_evkw_{lk}")
                _kind_def = _so("ev_kind", _ev0.kind if _ev0 else "AC")
                e_kind  = st.selectbox("AC / DC", ["AC", "DC"],
                                        index=0 if _kind_def == "AC" else 1,
                                        key=f"mgr_evkind_{lk}")
                w_start_d, w_end_d = (ld.allowed_window or (18, 8))
                e_ws = st.number_input("Charge window start (h)", 0, 23,
                                        int(_so("ev_window_start", w_start_d)),
                                        key=f"mgr_evws_{lk}")
                e_we = st.number_input("Charge window end (h)", 0, 23,
                                        int(_so("ev_window_end", w_end_d)),
                                        key=f"mgr_evwe_{lk}")
                entry["ev_chargers"]    = [{"count": int(e_count), "kw_each": float(e_kw), "kind": e_kind}]
                entry["ev_total_kw"]    = int(e_count) * float(e_kw)
                entry["allowed_window"] = [int(e_ws), int(e_we)]

            if ld.kind == "hvac" and ld.protected_window:
                st.caption("❄️ Protected business hours")
                w_start_d, w_end_d = ld.protected_window
                p_ws = st.number_input("Protect from (h)", 0, 23,
                                        int(_so("hvac_protect_start", w_start_d)),
                                        key=f"mgr_hws_{lk}")
                p_we = st.number_input("Protect to (h)", 0, 23,
                                        int(_so("hvac_protect_end", w_end_d)),
                                        key=f"mgr_hwe_{lk}")
                entry["protected_window"] = [int(p_ws), int(p_we)]

            loads[lk] = entry
            priority_order.append(lk)

    # Action buttons — Apply (live) and Run (historical) side by side
    _btn1, _btn2 = st.columns(2)
    _apply_live = _btn1.button(
        "▶ Apply to live simulation", type="primary", use_container_width=True,
        help="Writes settings to session state so the live Manager picks them up from "
             "the next sim tick.  Does not re-run history.",
    )
    _run_hist = _btn2.button(
        "🔁 Run on full history", use_container_width=True,
        help="Run Manager over the entire uploaded dataset for a historical "
             "'what-if' analysis.  Independent of the live simulation.",
    )

    if _apply_live:
        _set_so("battery_kwh",      battery_kwh)
        _set_so("c_rate",           c_rate)
        _set_so("md_target_pct",    peak_target)
        _set_so("charge_upper_pct", charge_upper)
        _set_so("init_soc_pct",     init_soc)
        for lk, entry in loads.items():
            if entry.get("kind") == "ev":
                _ev_ch = (entry.get("ev_chargers") or [{}])[0]
                _set_so("ev_count",        int(_ev_ch.get("count",   4)))
                _set_so("ev_kw_each",      float(_ev_ch.get("kw_each", 22.0)))
                _set_so("ev_kind",         _ev_ch.get("kind", "AC"))
                if "allowed_window" in entry:
                    _set_so("ev_window_start", int(entry["allowed_window"][0]))
                    _set_so("ev_window_end",   int(entry["allowed_window"][1]))
            if entry.get("kind") == "hvac" and "protected_window" in entry:
                _set_so("hvac_protect_start", int(entry["protected_window"][0]))
                _set_so("hvac_protect_end",   int(entry["protected_window"][1]))
                _set_so("hvac_max_cut_pct",   int(entry.get("max_cut_pct", 0.15) * 100))
        save_overrides(st.session_state.site_overrides)
        st.session_state.live_manager_results = None  # force recompute on next tick
        st.success("✅ Settings applied — live simulation will use them from the next tick.")
        st.rerun()

    if _run_hist:
        try:
            with st.spinner("Optimizing full load profile…"):
                _hist_results = _run_manager_on_df(
                    df, loads, priority_order,
                    battery_kwh, peak_target, charge_upper,
                    c_rate, init_soc, bat_eff,
                    peak_ref_kva if peak_ref_kva > 0 else None,
                )
            st.session_state.manager_results      = _hist_results
            st.session_state.manager_df_optimized = manager_results_to_sam_df(_hist_results)
            st.session_state.manager_df_original  = manager_results_to_original_df(_hist_results)
            st.session_state.powerreco_df         = manager_results_to_powerreco_df(_hist_results)
            # Mark as done so the auto-run doesn't fire again this session.
            st.session_state["_auto_manager_site"] = st.session_state.site_profile_id
            st.success(f"✅ Historical run complete — {len(_hist_results):,} intervals processed.")
        except Exception as e:
            st.error(f"Manager failed: {e}")
            st.code(traceback.format_exc())

    st.divider()

    # ═════════════════════════════════════════════════════════════════════
    # SECTION B: LIVE VIEW — re-renders every sim tick
    # ═════════════════════════════════════════════════════════════════════
    _mgr_tick_rate = float(st.session_state.get("lsim_tickrate", 0.5))

    @st.fragment(run_every=f"{_mgr_tick_rate:.1f}s")
    def _live_manager_view():
        # Ensure a plan exists — try once if missing (sim just initialised).
        results = st.session_state.get("live_manager_results")
        if results is None or len(results) == 0:
            _maybe = _run_manager_on_live_forecast()
            if _maybe:
                st.session_state.live_manager_results = _maybe
                results = _maybe

        if results is None or len(results) == 0:
            st.info(
                "Live strategy will appear once you (1) train a forecaster, "
                "(2) open the **🔴 Live Simulation** tab to initialise the sim, "
                "and (3) hit Play.  Every tick the Manager re-optimises "
                "against the fresh forecast."
            )
            return

        sim            = st.session_state.get("sim_state")
        executed_ticks = st.session_state.get("executed_ticks", [])
        palette        = ["#00b4d8", "#f59e0b", "#a78bfa", "#22c55e", "#f97316", "#ec4899"]

        # ══════════════════════════════════════════════════════════════════
        # SECTION 1 — THIS TICK: what the Manager actually did right now
        # ══════════════════════════════════════════════════════════════════
        st.markdown("### ⚡ This Tick — What the Manager just did")
        st.caption(
            "The first interval of each Manager run is grounded in the latest "
            "**revealed actual** reading, not a forecast.  This section shows "
            "the decision the Manager took on that real reading."
        )

        if executed_ticks:
            cur    = executed_ticks[-1]
            cur_ts = pd.to_datetime(cur["timestamp"]).strftime("%Y-%m-%d %H:%M")
            st.caption(f"Executed at **{cur_ts}**")

            # KPIs
            k1, k2, k3, k4, k5 = st.columns(5)
            kva_orig = float(cur.get("kva_original", 0))
            kva_mgd  = float(cur.get("kva_managed",  0))
            delta_kva = kva_mgd - kva_orig
            k1.metric("Actual kVA",   f"{kva_orig:.1f}")
            k2.metric("Managed kVA",  f"{kva_mgd:.1f}",
                      delta=f"{delta_kva:+.1f}", delta_color="inverse")
            bat_act = float(cur.get("battery_action_kw", 0))
            bat_soc = float(cur.get("battery_soc_pct", 0))
            if bat_act < -0.1:
                k3.metric("Battery", f"Discharge {-bat_act:.1f} kW",
                          delta=f"SOC {bat_soc:.0f}%", delta_color="inverse")
            elif bat_act > 0.1:
                k3.metric("Battery", f"Charge {bat_act:.1f} kW",
                          delta=f"SOC {bat_soc:.0f}%")
            else:
                k3.metric("Battery", "Idle", delta=f"SOC {bat_soc:.0f}%", delta_color="off")
            k4.metric("MD Window",   "🔴 Active" if cur.get("in_md_hours")  else "⬜ Off-peak")
            k5.metric("Pre-MD Boost","🟡 Active" if cur.get("in_pre_md")    else "—")

            # Per-load conditions
            load_factor_keys = [
                c for c in cur.keys()
                if c.endswith("_factor")
                and c not in ("load_factor",)
                and not c.startswith(("kw_", "kvar_", "kva_", "bat"))
            ]
            if load_factor_keys:
                st.markdown("**Load conditions this tick:**")
                lcols = st.columns(len(load_factor_keys))
                for i, fk in enumerate(load_factor_keys):
                    lk       = fk.replace("_factor", "")
                    factor   = float(cur.get(fk, 1.0))
                    load_kva = float(cur.get(f"{lk}_kva",     0))
                    mgd_kva  = float(cur.get(f"{lk}_managed", 0))
                    cut_pct  = (1 - factor) * 100
                    color    = "#ef4444" if cut_pct > 1 else "#22c55e"
                    status   = f"Shed {cut_pct:.0f}%" if cut_pct > 0.5 else "Normal"
                    with lcols[i]:
                        st.markdown(
                            f"**{lk.replace('_',' ').title()}**  \n"
                            f"<span style='color:{color}'>{status}</span>  \n"
                            f"{mgd_kva:.1f} / {load_kva:.1f} kVA",
                            unsafe_allow_html=True,
                        )

            # Actions taken this tick
            actions_now = cur.get("actions", [])
            if actions_now:
                st.markdown("**Actions taken:**")
                for a in actions_now:
                    kind = a.get("type", "?")
                    if kind == "battery_discharge":
                        st.success(
                            f"🔻 **Discharged** {a.get('discharge_kw', 0):.1f} kW  "
                            f"· SOC {a.get('soc_before_kwh', 0):.0f} → "
                            f"{a.get('soc_after_kwh', 0):.0f} kWh"
                            + ("  · look-ahead trigger" if a.get("lookahead_triggered") else "")
                            + ("  · in MD window"        if a.get("md_hours")            else "")
                        )
                    elif kind == "battery_charge":
                        st.info(
                            f"🔺 **Charged** {a.get('charge_kw', 0):.1f} kW  "
                            f"· {a.get('charge_trigger', '')}  "
                            f"· SOC {a.get('soc_before_kwh', 0):.0f} → "
                            f"{a.get('soc_after_kwh', 0):.0f} kWh"
                        )
                    elif kind == "load_reduction":
                        st.warning(
                            f"✂️ **Cut** {a.get('cut_kva', 0):.1f} kVA  "
                            f"from **{a.get('load', '?')}**  "
                            f"({a.get('reason', 'normal')})  "
                            f"→ factor {a.get('factor_pct', 0):.0f}% remaining"
                        )
            else:
                st.caption("No interventions needed — load was within the discharge trigger.")
        else:
            st.info("Advance the simulation to see per-tick decisions here.")

        st.divider()

        # ══════════════════════════════════════════════════════════════════
        # SECTION 2 — HISTORY: accumulated executed decisions
        # ══════════════════════════════════════════════════════════════════
        st.markdown("### 📜 History — What has happened")
        st.caption(
            "Every tick the Manager executed an actual reading.  "
            "This chart and log accumulate those real decisions — not the forecast plan."
        )

        if len(executed_ticks) > 1:
            hist_df = pd.DataFrame([
                {k: v for k, v in t.items() if k != "actions"} for t in executed_ticks
            ])
            hist_df["timestamp"] = pd.to_datetime(hist_df["timestamp"])

            # Summary KPIs
            h1, h2, h3, h4 = st.columns(4)
            n_dis       = sum(1 for t in executed_ticks if float(t.get("battery_discharge_kw", 0)) > 0.1)
            n_chg       = sum(1 for t in executed_ticks if float(t.get("battery_charge_kw",    0)) > 0.1)
            n_cuts      = sum(
                len([a for a in t.get("actions", []) if a.get("type") == "load_reduction"])
                for t in executed_ticks
            )
            tot_dis_kwh = sum(float(t.get("battery_discharge_kw", 0)) * 0.5 for t in executed_ticks)
            h1.metric("Ticks executed",    len(executed_ticks))
            h2.metric("Discharge events",  n_dis, delta=f"{tot_dis_kwh:.1f} kWh total")
            h3.metric("Charge events",     n_chg)
            h4.metric("Load cuts applied", n_cuts)

            # History chart: actual vs managed kVA + SOC overlay
            fig_hist = make_subplots(
                rows=2, cols=1, shared_xaxes=True,
                vertical_spacing=0.06, row_heights=[0.65, 0.35],
                subplot_titles=("Actual vs managed kVA (executed)", "Battery SOC % (executed)"),
            )
            fig_hist.add_trace(go.Scatter(
                x=hist_df["timestamp"], y=hist_df["kva_original"],
                mode="lines+markers", name="Actual kVA",
                line=dict(color="#ef4444", width=2), marker=dict(size=3),
            ), row=1, col=1)
            fig_hist.add_trace(go.Scatter(
                x=hist_df["timestamp"], y=hist_df["kva_managed"],
                mode="lines+markers", name="Managed kVA",
                line=dict(color="#00d4ff", width=2.5), marker=dict(size=3),
            ), row=1, col=1)
            if "target_peak" in hist_df.columns:
                fig_hist.add_trace(go.Scatter(
                    x=hist_df["timestamp"], y=hist_df["target_peak"],
                    mode="lines", name="Target",
                    line=dict(color="#22c55e", width=1.5, dash="dash"),
                ), row=1, col=1)
            fig_hist.add_trace(go.Scatter(
                x=hist_df["timestamp"], y=hist_df.get("battery_soc_pct", pd.Series(dtype=float)),
                mode="lines+markers", name="SOC %",
                line=dict(color="#a78bfa", width=2),
                fill="tozeroy", fillcolor="rgba(167,139,250,0.15)",
                marker=dict(size=3),
            ), row=2, col=1)
            fig_hist.update_layout(
                height=420, hovermode="x unified",
                legend=dict(orientation="h", yanchor="bottom", y=1.04),
            )
            fig_hist.update_yaxes(title_text="kVA",  row=1, col=1)
            fig_hist.update_yaxes(title_text="SOC %", row=2, col=1, range=[0, 105])
            fig_hist.update_xaxes(title_text="Time",  row=2, col=1)
            st.plotly_chart(fig_hist, use_container_width=True,
                            key=f"hist_chart_{len(executed_ticks)}")

            # Executed action log
            with st.expander(f"Executed action log — {len(executed_ticks)} ticks"):
                elog: list[dict] = []
                for t in reversed(executed_ticks[-60:]):
                    ts_lbl = pd.to_datetime(t["timestamp"]).strftime("%Y-%m-%d %H:%M")
                    for a in t.get("actions", []):
                        kind = a.get("type", "?")
                        if kind == "battery_discharge":
                            elog.append({"Time": ts_lbl, "": "🔻", "Action": "Discharge",
                                          "Load": "Battery",
                                          "Detail": (
                                              f"{a.get('discharge_kw', 0):.1f} kW · "
                                              f"SOC {a.get('soc_before_kwh', 0):.0f}→"
                                              f"{a.get('soc_after_kwh', 0):.0f} kWh"
                                              + (" · look-ahead" if a.get("lookahead_triggered") else "")
                                              + (" · MD-hrs"     if a.get("md_hours")            else "")
                                          )})
                        elif kind == "battery_charge":
                            elog.append({"Time": ts_lbl, "": "🔺", "Action": "Charge",
                                          "Load": "Battery",
                                          "Detail": (
                                              f"{a.get('charge_kw', 0):.1f} kW · "
                                              f"{a.get('charge_trigger', '')} · "
                                              f"SOC {a.get('soc_before_kwh', 0):.0f}→"
                                              f"{a.get('soc_after_kwh', 0):.0f} kWh"
                                          )})
                        elif kind == "load_reduction":
                            elog.append({"Time": ts_lbl, "": "✂️", "Action": "Load cut",
                                          "Load": a.get("load", "?"),
                                          "Detail": (
                                              f"Cut {a.get('cut_kva', 0):.1f} kVA · "
                                              f"{a.get('reason', 'normal')} · "
                                              f"factor {a.get('factor_pct', 0):.0f}%"
                                          )})
                if elog:
                    st.dataframe(pd.DataFrame(elog), use_container_width=True,
                                 hide_index=True, height=240)
                else:
                    st.caption("No interventions in executed history.")
        else:
            st.caption("History populates as the simulation advances tick by tick.")

        st.divider()

        # ══════════════════════════════════════════════════════════════════
        # SECTION 3 — FORWARD PLAN: what the Manager intends to do
        # The forward plan is based on the 24-h forecast — it updates every
        # tick as new actuals arrive.  It is a PLAN, not executed history.
        # The first interval (index 0) is the actual reading already shown
        # above; we skip it and show only the future-facing intervals.
        # ══════════════════════════════════════════════════════════════════
        future_results = results[1:] if len(results) > 1 else results
        sim_label = (
            f"sim tick {sim.tick:,}/{sim.total_ticks:,}" if sim is not None else "static plan"
        )
        st.markdown(
            f"### 🗺️ Forward Plan — Manager's intent  "
            f"<span style='color:#9ca3af;font-size:13px;font-weight:400'>"
            f"({len(future_results)} intervals · {sim_label})</span>",
            unsafe_allow_html=True,
        )
        st.caption(
            "Based on the **current 24-h forecast**.  Updates every tick as new actuals "
            "arrive and refine the forecast.  This is a living plan — reality may diverge, "
            "at which point the next tick will revise it."
        )

        if future_results:
            fut_df = pd.DataFrame([
                {k: v for k, v in r.items() if k != "actions"} for r in future_results
            ])
            fut_df["timestamp"] = pd.to_datetime(fut_df["timestamp"])

            # Plan KPIs
            fc_peak  = float(fut_df["kva_original"].max()) if "kva_original" in fut_df.columns else 0
            pln_peak = float(fut_df["kva_managed"].max())  if "kva_managed"  in fut_df.columns else 0
            pk_red   = (fc_peak - pln_peak) / fc_peak * 100 if fc_peak > 0 else 0
            pln_dis  = float(fut_df.get("battery_discharge_kw", pd.Series([0])).sum()) * 0.5
            cur_soc  = float(executed_ticks[-1].get("battery_soc_pct", 50)) if executed_ticks else 50.0

            pk1, pk2, pk3, pk4 = st.columns(4)
            pk1.metric("Forecast peak (plan)",   f"{fc_peak:.1f} kVA")
            pk2.metric("Planned managed peak",   f"{pln_peak:.1f} kVA",
                       delta=f"−{pk_red:.1f}%", delta_color="inverse")
            pk3.metric("Planned discharge",      f"{pln_dis:.1f} kWh")
            pk4.metric("Current SOC",            f"{cur_soc:.0f}%")

            # Forward load + battery chart
            fwd_load_keys = [
                c.replace("_managed", "")
                for c in fut_df.columns
                if c.endswith("_managed")
                and c not in ("kw_managed", "kvar_managed", "kva_managed")
            ]
            fig_fwd = make_subplots(
                rows=2, cols=1, shared_xaxes=True,
                vertical_spacing=0.06, row_heights=[0.65, 0.35],
                subplot_titles=("Planned load (kVA)", "Planned battery action (kW)"),
            )
            for i, lk in enumerate(fwd_load_keys):
                col_name = f"{lk}_managed"
                if col_name in fut_df.columns:
                    fig_fwd.add_trace(go.Bar(
                        x=fut_df["timestamp"], y=fut_df[col_name],
                        name=f"{lk} (planned)",
                        marker_color=palette[i % len(palette)], opacity=0.55,
                    ), row=1, col=1)
            fig_fwd.add_trace(go.Scatter(
                x=fut_df["timestamp"], y=fut_df["kva_original"],
                mode="lines", name="Forecast kVA",
                line=dict(color="#ef4444", width=2.5),
            ), row=1, col=1)
            if "target_peak" in fut_df.columns:
                fig_fwd.add_trace(go.Scatter(
                    x=fut_df["timestamp"], y=fut_df["target_peak"],
                    mode="lines", name="Target",
                    line=dict(color="#22c55e", width=2, dash="dash"),
                ), row=1, col=1)
            fig_fwd.add_trace(go.Scatter(
                x=fut_df["timestamp"], y=fut_df["kva_managed"],
                mode="lines", name="Planned managed kVA",
                line=dict(color="#00d4ff", width=2.5),
            ), row=1, col=1)
            if "battery_charge_kw" in fut_df.columns:
                fig_fwd.add_trace(go.Bar(
                    x=fut_df["timestamp"], y=fut_df["battery_charge_kw"],
                    name="Charge (planned)", marker_color="#22c55e",
                ), row=2, col=1)
            if "battery_discharge_kw" in fut_df.columns:
                fig_fwd.add_trace(go.Bar(
                    x=fut_df["timestamp"], y=-fut_df["battery_discharge_kw"],
                    name="Discharge (planned)", marker_color="#ef4444",
                ), row=2, col=1)
            fig_fwd.update_layout(
                height=500, barmode="stack", hovermode="x unified",
                legend=dict(orientation="h", yanchor="bottom", y=1.06, xanchor="left", x=0),
            )
            fig_fwd.update_xaxes(title_text="Time", row=2, col=1)
            fig_fwd.update_yaxes(title_text="kVA",  row=1, col=1)
            fig_fwd.update_yaxes(title_text="kW",   row=2, col=1)
            st.plotly_chart(fig_fwd, use_container_width=True,
                            key=f"mgr_fwd_{len(results)}_{st.session_state.live_manager_last_tick}")

            # Planned action log
            with st.expander("Planned action log — what the Manager intends to do"):
                flog: list[dict] = []
                for r in future_results:
                    ts_lbl = pd.to_datetime(r["timestamp"]).strftime("%H:%M")
                    for a in r.get("actions", []):
                        kind = a.get("type", "?")
                        if kind == "battery_discharge":
                            flog.append({"Time": ts_lbl, "": "🔻", "Action": "Discharge",
                                          "Detail": f"{a.get('discharge_kw', 0):.1f} kW · SOC {a.get('soc_before_kwh', 0):.0f}→{a.get('soc_after_kwh', 0):.0f} kWh"})
                        elif kind == "battery_charge":
                            flog.append({"Time": ts_lbl, "": "🔺", "Action": "Charge",
                                          "Detail": f"{a.get('charge_kw', 0):.1f} kW · {a.get('charge_trigger', '')}"})
                        elif kind == "load_reduction":
                            flog.append({"Time": ts_lbl, "": "✂️", "Action": "Load cut",
                                          "Detail": f"{a.get('load', '?')} · {a.get('cut_kva', 0):.1f} kVA · {a.get('reason', 'normal')}"})
                if flog:
                    st.dataframe(pd.DataFrame(flog), use_container_width=True,
                                 hide_index=True, height=200)
                else:
                    st.caption("No planned interventions — forecast stays within the discharge trigger.")

        # ── Forecast vs Actual Divergence ──────────────────────────────────
        sim_d = st.session_state.get("sim_state")
        if sim_d is not None and len(sim_d.forecast_log) > 0:
            verified = pd.DataFrame([
                r for r in sim_d.forecast_log
                if r.get("actual") is not None and r.get("horizon_steps") == 1
            ])
            if len(verified) > 0:
                st.divider()
                st.markdown("##### 📉 Forecast vs Actual Divergence")
                st.caption(
                    "How much actual load deviated from the h=1 forecast each tick.  "
                    "Positive bars = underpredicted (less buffer available than planned).  "
                    "Negative = overpredicted (excess reserve held unnecessarily).  "
                    "Flat line = Manager's plan tracked reality closely."
                )
                verified = verified.sort_values("target_ts").reset_index(drop=True)
                verified["divergence_kw"] = verified["actual"] - verified["median"]
                d1, d2, d3, d4 = st.columns(4)
                d1.metric("Verified ticks",    f"{len(verified):,}")
                d2.metric("Mean |error|",      f"{verified['divergence_kw'].abs().mean():.1f} kW")
                d3.metric("Max underpredict",  f"+{verified['divergence_kw'].max():.0f} kW")
                d4.metric("Max overpredict",   f"{verified['divergence_kw'].min():.0f} kW")
                colors = ["#ef4444" if v > 0 else "#3b82f6" for v in verified["divergence_kw"]]
                fig_dv = go.Figure()
                fig_dv.add_trace(go.Bar(
                    x=verified["target_ts"], y=verified["divergence_kw"],
                    marker_color=colors, name="actual − forecast",
                    hovertemplate="%{x}<br>Δ %{y:.1f} kW<extra></extra>",
                ))
                fig_dv.add_trace(go.Scatter(
                    x=[verified["target_ts"].min(), verified["target_ts"].max()],
                    y=[0, 0], mode="lines",
                    line=dict(color="#9ca3af", width=1), showlegend=False,
                ))
                fig_dv.update_layout(
                    xaxis_title="Time", yaxis_title="kW (actual − h=1 forecast)",
                    height=300, showlegend=False, bargap=0.05,
                )
                st.plotly_chart(fig_dv, use_container_width=True,
                                key=f"mgr_dv_{len(verified)}_{st.session_state.live_manager_last_tick}")

    _live_manager_view()

    # ═════════════════════════════════════════════════════════════════════
    # SECTION C: HISTORICAL RUN RESULTS
    # ═════════════════════════════════════════════════════════════════════
    if st.session_state.manager_results:
        results = st.session_state.manager_results
        res_df  = pd.DataFrame([
            {k: v for k, v in r.items() if k != "actions"} for r in results
        ])
        res_df["timestamp"] = pd.to_datetime(res_df["timestamp"])

        # KPI metrics
        orig_peak    = float(res_df["kva_original"].max())
        managed_peak = float(res_df["kva_managed"].max())
        peak_red_pct = (orig_peak - managed_peak) / orig_peak * 100 if orig_peak else 0
        total_dis    = float(res_df["battery_discharge_kw"].sum()) * 0.5
        total_chg    = float(res_df["battery_charge_kw"].sum()) * 0.5

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Original Peak", f"{orig_peak:.1f} kVA")
        c2.metric("Managed Peak", f"{managed_peak:.1f} kVA", delta=f"-{peak_red_pct:.1f}%", delta_color="inverse")
        c3.metric("Total Discharge", f"{total_dis:.1f} kWh")
        c4.metric("Total Charge", f"{total_chg:.1f} kWh")

        # Before/after chart
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                            subplot_titles=("Load Profile kVA", "Battery SOC"),
                            row_heights=[0.65, 0.35])
        fig.add_trace(go.Scatter(x=res_df["timestamp"], y=res_df["kva_original"],
                                  mode="lines", name="Original kVA", line=dict(color="#d62728")), row=1, col=1)
        fig.add_trace(go.Scatter(x=res_df["timestamp"], y=res_df["kva_managed"],
                                  mode="lines", name="Managed kVA", line=dict(color="#1f77b4")), row=1, col=1)
        if "target_peak" in res_df.columns:
            fig.add_trace(go.Scatter(x=res_df["timestamp"], y=res_df["target_peak"],
                                      mode="lines", name="Target", line=dict(color="#2ca02c", dash="dash")), row=1, col=1)
        fig.add_trace(go.Scatter(x=res_df["timestamp"], y=res_df["battery_soc_pct"],
                                  mode="lines", name="SOC %", line=dict(color="#9467bd")), row=2, col=1)
        fig.update_layout(height=550, hovermode="x unified")
        fig.update_yaxes(title_text="kVA", row=1, col=1)
        fig.update_yaxes(title_text="SOC %", row=2, col=1)
        st.plotly_chart(fig, use_container_width=True)

        # Download
        csv_str = manager_results_to_csv(results)
        st.download_button(
            "Download Manager Results CSV",
            data=csv_str.encode(),
            file_name="manager_results.csv",
            mime="text/csv",
        )


# ─────────────────────────────────────────────────────────────────────────────
# TAB 4: BILL CALCULATOR
# ─────────────────────────────────────────────────────────────────────────────
with tab4:
    st.header("TNB Bill Calculator")
    st.caption(
        "Live monthly bills are computed automatically as each calendar month "
        "completes in the simulation.  A historical batch mode is also available below."
    )

    # ── Shared billing settings (used by both live and historical modes) ──
    with st.expander("Tariff & billing settings", expanded=True):
        c1, c2, c3 = st.columns(3)
        tariff_options  = list(TARIFF_META.keys())
        _tariff_default = _so("tariff_code", st.session_state.tariff_code)
        default_idx     = tariff_options.index(_tariff_default) if _tariff_default in tariff_options else 2
        tariff          = c1.selectbox("Tariff", tariff_options, index=default_idx,
                                        format_func=lambda t: f"{t} — {TARIFF_META[t]['name']}")
        icpt_sen        = c2.number_input("ICPT (sen/kWh)", -10.0, 20.0,
                                           float(_so("icpt_sen", 0.0)), step=0.5,
                                           help="Positive = surcharge, negative = rebate")
        nem_rate        = c3.number_input("NEM buyback rate (RM/kWh)", 0.20, 0.50,
                                           float(_so("nem_rate", 0.31)), step=0.01)

    sched_key = st.session_state.tariff_schedule_key
    st.caption(
        f"Schedule in use: **{sched_key}** "
        f"(toggle in sidebar to switch between July 2025 and legacy rates)."
    )

    # ══════════════════════════════════════════════════════════════════════
    # SECTION A: LIVE MONTHLY BILL TRACKER
    # Populated automatically each time a calendar month completes in the
    # live simulation.  Bills are computed from executed_ticks (actual
    # Manager decisions on real readings), not from the forecast.
    # ══════════════════════════════════════════════════════════════════════
    st.subheader("📅 Live Monthly Bill Tracker")

    _live_bills = st.session_state.get("live_bill_months", [])

    if _live_bills:
        # Summary KPIs
        _tot_before  = sum(m["before"].get("net_bill_rm", 0) for m in _live_bills)
        _tot_after   = sum(m["after"].get("net_bill_rm",  0) for m in _live_bills)
        _tot_savings = _tot_before - _tot_after
        _n_months    = len(_live_bills)

        lk1, lk2, lk3, lk4 = st.columns(4)
        lk1.metric("Months completed",       f"{_n_months}")
        lk2.metric("Total before (all)",      _fmt_rm(_tot_before))
        lk3.metric("Total after (all)",        _fmt_rm(_tot_after),
                   delta=f"−{_fmt_rm(_tot_savings)}", delta_color="inverse")
        lk4.metric("Avg monthly savings",     _fmt_rm(_tot_savings / _n_months))

        # Before vs After bar chart per month
        _months_lbl  = [m["month"] for m in _live_bills]
        _before_vals = [m["before"].get("net_bill_rm", 0) for m in _live_bills]
        _after_vals  = [m["after"].get("net_bill_rm",  0) for m in _live_bills]
        _sav_vals    = [m["savings_rm"] for m in _live_bills]

        fig_live_bill = go.Figure()
        fig_live_bill.add_trace(go.Bar(
            name="Before optimisation", x=_months_lbl, y=_before_vals,
            marker_color="#d62728",
        ))
        fig_live_bill.add_trace(go.Bar(
            name="After optimisation",  x=_months_lbl, y=_after_vals,
            marker_color="#1f77b4",
        ))
        fig_live_bill.add_trace(go.Scatter(
            name="Savings (RM)", x=_months_lbl, y=_sav_vals,
            mode="lines+markers+text",
            line=dict(color="#22c55e", width=2.5),
            marker=dict(size=8),
            text=[f"RM {v:,.0f}" for v in _sav_vals],
            textposition="top center",
            yaxis="y2",
        ))
        fig_live_bill.update_layout(
            barmode="group",
            title="Monthly bill — before vs after AI Manager optimisation",
            yaxis_title="Net bill (RM)",
            yaxis2=dict(title="Savings (RM)", overlaying="y", side="right", showgrid=False),
            height=360,
            hovermode="x unified",
            legend=dict(orientation="h", yanchor="bottom", y=1.02),
        )
        st.plotly_chart(fig_live_bill, use_container_width=True)

        # Month selector — detailed breakdown
        st.markdown("**Detailed breakdown — select a month:**")
        _sel_month = st.selectbox(
            "Month", _months_lbl, index=len(_months_lbl) - 1,
            key="calc_live_month_sel",
            label_visibility="collapsed",
        )
        _sel = next(m for m in _live_bills if m["month"] == _sel_month)

        _dc1, _dc2 = st.columns(2)
        with _dc1:
            st.markdown(f"**Before optimisation — {_sel_month}**")
            _b = _sel["before"]
            _b_rows = [
                ("Total kWh",       f"{_b.get('total_kwh', 0):,.1f} kWh"),
                ("Peak MD",         f"{_b.get('peak_kw_md', 0):.1f} kW"),
                ("Energy charge",   _fmt_rm(_b.get("energy_rm", 0))),
                ("MD charge",       _fmt_rm(_b.get("md_rm", 0))),
                ("ICPT",            _fmt_rm(_b.get("icpt_rm", 0))),
                ("KWTBB",           _fmt_rm(_b.get("kwtbb_rm", 0))),
                ("Service tax",     _fmt_rm(_b.get("tax_rm", 0))),
                ("NEM credit",      f"−{_fmt_rm(_b.get('nem_credit_rm', 0))}"),
                ("**Net bill**",    f"**{_fmt_rm(_b.get('net_bill_rm', 0))}**"),
            ]
            st.table(pd.DataFrame(_b_rows, columns=["Item", "Amount"]))

        with _dc2:
            st.markdown(f"**After optimisation — {_sel_month}**")
            _a = _sel["after"]
            _a_rows = [
                ("Total kWh",       f"{_a.get('total_kwh', 0):,.1f} kWh"),
                ("Peak MD",         f"{_a.get('peak_kw_md', 0):.1f} kW"),
                ("Energy charge",   _fmt_rm(_a.get("energy_rm", 0))),
                ("MD charge",       _fmt_rm(_a.get("md_rm", 0))),
                ("ICPT",            _fmt_rm(_a.get("icpt_rm", 0))),
                ("KWTBB",           _fmt_rm(_a.get("kwtbb_rm", 0))),
                ("Service tax",     _fmt_rm(_a.get("tax_rm", 0))),
                ("NEM credit",      f"−{_fmt_rm(_a.get('nem_credit_rm', 0))}"),
                ("**Net bill**",    f"**{_fmt_rm(_a.get('net_bill_rm', 0))}**"),
            ]
            st.table(pd.DataFrame(_a_rows, columns=["Item", "Amount"]))

        _savings_this = _sel["savings_rm"]
        _savings_pct  = (
            _savings_this / _sel["before"].get("net_bill_rm", 1) * 100
            if _sel["before"].get("net_bill_rm", 0) > 0 else 0
        )
        st.info(
            f"**{_sel_month} savings: {_fmt_rm(_savings_this)}**  "
            f"({_savings_pct:.1f}% reduction)  ·  "
            f"{_sel['ticks']} intervals captured  "
            f"({'full month' if _sel['ticks'] >= 1344 else 'partial month — sim may not cover full calendar month'})"
        )

        # Download live bill CSV
        _live_bill_rows = []
        for m in _live_bills:
            _live_bill_rows.append({
                "Month":              m["month"],
                "Ticks":              m["ticks"],
                "Before kWh":         m["before"].get("total_kwh", 0),
                "Before Peak kW":     m["before"].get("peak_kw_md", 0),
                "Before Net Bill":    m["before"].get("net_bill_rm", 0),
                "After kWh":          m["after"].get("total_kwh", 0),
                "After Peak kW":      m["after"].get("peak_kw_md", 0),
                "After Net Bill":     m["after"].get("net_bill_rm", 0),
                "Savings RM":         m["savings_rm"],
            })
        _lb_csv = io.StringIO()
        pd.DataFrame(_live_bill_rows).to_csv(_lb_csv, index=False)
        st.download_button(
            "Download live monthly bill CSV",
            _lb_csv.getvalue().encode(),
            "live_monthly_bills.csv", "text/csv",
        )

    else:
        st.info(
            "Live monthly bills appear here automatically each time a full calendar month "
            "completes in the **🔴 Live Simulation**.  Start the simulation, hit Play, and "
            "advance until the first month boundary passes — the bill will populate here "
            "with before/after comparison from the Manager's actual executed decisions."
        )

    st.divider()

    # ══════════════════════════════════════════════════════════════════════
    # SECTION B: HISTORICAL BATCH CALCULATION
    # Calculates from the full uploaded dataset (or Manager historical run).
    # ══════════════════════════════════════════════════════════════════════
    st.subheader("📊 Historical Batch Calculation")
    st.caption(
        "Processes the entire uploaded load profile at once.  "
        "Run the AI Manager first (Manager tab → Run on full history) to see "
        "before/after comparison; otherwise uses raw uploaded data."
    )

    if st.session_state.df is None:
        st.warning("Upload a load profile first.")
        st.stop()

    if st.session_state.manager_df_optimized is None:
        st.info("Run the AI Manager on full history to see before/after comparison.  "
                "Showing raw uploaded data only for now.")
        df_for_bill  = st.session_state.df
        df_orig_bill = st.session_state.df
        show_comparison = False
    else:
        df_for_bill  = st.session_state.manager_df_optimized
        df_orig_bill = st.session_state.manager_df_original
        show_comparison = True

    try:
        monthly_stats_opt  = compute_monthly_stats(df_for_bill,  schedule_key=sched_key)
        monthly_stats_orig = compute_monthly_stats(df_orig_bill, schedule_key=sched_key) if show_comparison else monthly_stats_opt

        bill_rows_opt, bill_rows_orig = [], []
        for _, row in monthly_stats_opt.iterrows():
            bill = calculate_bill(
                tariff,
                monthly_kwh=row["total_kwh"],
                peak_kwh=row["peak_kwh"],
                offpeak_kwh=row["offpeak_kwh"],
                max_demand_kw=row["max_demand_kw"],
                icpt_sen_per_kwh=icpt_sen,
                schedule_key=sched_key,
            )
            nem = compute_nem_credit(row["export_kwh"], nem_rate)
            bill_rows_opt.append({
                "Month": str(row["month"]),
                "Total kWh": row["total_kwh"],
                "Peak kW (MD)": round(row["max_demand_kw"], 1),
                "Energy (RM)": bill["energy_charge"],
                "MD Charge (RM)": bill["md_charge"],
                "ICPT (RM)": bill["icpt_charge"],
                "KWTBB (RM)": bill["kwtbb_charge"],
                "Svc Tax (RM)": bill["service_tax"],
                "NEM Credit (RM)": nem["nem_credit_rm"],
                "Net Bill (RM)": round(bill["total_bill"] - nem["nem_credit_rm"], 2),
            })

        if show_comparison:
            for _, row in monthly_stats_orig.iterrows():
                bill = calculate_bill(
                    tariff,
                    monthly_kwh=row["total_kwh"],
                    peak_kwh=row["peak_kwh"],
                    offpeak_kwh=row["offpeak_kwh"],
                    max_demand_kw=row["max_demand_kw"],
                    icpt_sen_per_kwh=icpt_sen,
                    schedule_key=sched_key,
                )
                nem = compute_nem_credit(row["export_kwh"], nem_rate)
                bill_rows_orig.append({
                    "Month": str(row["month"]),
                    "Net Bill (RM)": round(bill["total_bill"] - nem["nem_credit_rm"], 2),
                })

        opt_df     = pd.DataFrame(bill_rows_opt)
        total_opt  = opt_df["Net Bill (RM)"].sum()

        if show_comparison:
            orig_df    = pd.DataFrame(bill_rows_orig)
            total_orig = orig_df["Net Bill (RM)"].sum()
            savings    = total_orig - total_opt

            c1, c2, c3 = st.columns(3)
            c1.metric("Total Bill (Before)", _fmt_rm(total_orig))
            c2.metric("Total Bill (After)",  _fmt_rm(total_opt),
                      delta=f"−{_fmt_rm(savings)}", delta_color="inverse")
            c3.metric("Annual Savings (est.)",
                      _fmt_rm(savings * 12 / max(len(monthly_stats_opt), 1)))

            if len(opt_df) > 0:
                fig_hist_bill = go.Figure()
                if len(orig_df):
                    fig_hist_bill.add_trace(go.Bar(
                        name="Before", x=orig_df["Month"], y=orig_df["Net Bill (RM)"],
                        marker_color="#d62728",
                    ))
                fig_hist_bill.add_trace(go.Bar(
                    name="After",  x=opt_df["Month"],  y=opt_df["Net Bill (RM)"],
                    marker_color="#1f77b4",
                ))
                fig_hist_bill.update_layout(
                    barmode="group", title="Historical monthly bill comparison",
                    yaxis_title="RM", height=340,
                )
                st.plotly_chart(fig_hist_bill, use_container_width=True)
        else:
            c1, c2 = st.columns(2)
            c1.metric("Total Bill (period)", _fmt_rm(total_opt))
            c2.metric("Avg Monthly",         _fmt_rm(total_opt / max(len(opt_df), 1)))

        st.subheader("Monthly Bill Breakdown (After Optimisation)")
        st.dataframe(opt_df, use_container_width=True, hide_index=True)

        buf = io.StringIO()
        opt_df.to_csv(buf, index=False)
        st.download_button("Download historical bill CSV", buf.getvalue().encode(),
                           "bill_breakdown_historical.csv", "text/csv")

        # ─────────────────────────────────────────────────────────────
        # MD TARIFF SENSITIVITY — same period, both schedules
        # ─────────────────────────────────────────────────────────────
        st.divider()
        st.subheader("📊 MD Tariff Sensitivity")
        st.caption(
            "Compares the same bill period priced under both schedules. "
            "Highlights how much the July 2025 MD jump costs (or, after "
            "optimization, how much it saves)."
        )
        try:
            def _bill_total(d_in, sk):
                stats = compute_monthly_stats(d_in, schedule_key=sk)
                tot_energy, tot_md, tot_icpt, tot_kwtbb, tot_tax, tot_total, tot_nem = (
                    0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
                )
                for _, row in stats.iterrows():
                    b = calculate_bill(
                        tariff,
                        monthly_kwh=row["total_kwh"],
                        peak_kwh=row["peak_kwh"],
                        offpeak_kwh=row["offpeak_kwh"],
                        max_demand_kw=row["max_demand_kw"],
                        icpt_sen_per_kwh=icpt_sen,
                        schedule_key=sk,
                    )
                    n = compute_nem_credit(row["export_kwh"], nem_rate)
                    tot_energy += b["energy_charge"]
                    tot_md     += b["md_charge"]
                    tot_icpt   += b["icpt_charge"]
                    tot_kwtbb  += b["kwtbb_charge"]
                    tot_tax    += b["service_tax"]
                    tot_total  += b["total_bill"]
                    tot_nem    += n["nem_credit_rm"]
                return {
                    "Energy (RM)":  round(tot_energy, 2),
                    "MD (RM)":      round(tot_md,     2),
                    "ICPT (RM)":    round(tot_icpt,   2),
                    "KWTBB (RM)":   round(tot_kwtbb,  2),
                    "Svc Tax (RM)": round(tot_tax,    2),
                    "Gross (RM)":   round(tot_total,  2),
                    "NEM Credit (RM)": round(tot_nem, 2),
                    "Net Bill (RM)":   round(tot_total - tot_nem, 2),
                }

            # After-optimization rows for both schedules
            rows = []
            for sk, label in [("legacy_2014","Pre-July 2025"), ("new_2025","Post-July 2025")]:
                r = _bill_total(df_for_bill, sk)
                r = {"Schedule": label, "Stage": "After optimisation", **r}
                rows.append(r)
                if show_comparison:
                    rb = _bill_total(df_orig_bill, sk)
                    rb = {"Schedule": label, "Stage": "Before optimisation", **rb}
                    rows.append(rb)
            sens_df = pd.DataFrame(rows)
            # Reorder so Before/After pair under each schedule
            sens_df = sens_df.sort_values(["Schedule", "Stage"]).reset_index(drop=True)
            st.dataframe(sens_df, use_container_width=True, hide_index=True)

            # Headline delta
            new_net    = next((r["Net Bill (RM)"] for r in rows if r["Schedule"]=="Post-July 2025" and r["Stage"]=="After optimisation"), None)
            legacy_net = next((r["Net Bill (RM)"] for r in rows if r["Schedule"]=="Pre-July 2025"  and r["Stage"]=="After optimisation"), None)
            if new_net is not None and legacy_net is not None:
                delta = new_net - legacy_net
                pct   = delta / legacy_net * 100 if legacy_net else 0
                st.info(
                    f"**Tariff-change impact (after optimisation):** "
                    f"bill rises **RM {delta:,.2f}** ({pct:+.1f}%) when switching "
                    f"from legacy to July 2025 rates. "
                    f"All else equal — this is the slice of your bill the new MD rate adds, "
                    f"and the slice BOLT's peak-shaving directly attacks."
                )
        except Exception as e:
            st.warning(f"Sensitivity table failed: {e}")

    except Exception as e:
        st.error(f"Bill calculation failed: {e}")
        st.code(traceback.format_exc())


# ─────────────────────────────────────────────────────────────────────────────
# TAB 5: POWERRECO
# ─────────────────────────────────────────────────────────────────────────────
with tab5:
    st.header("PowerRECO — Solar & Battery Sizing")
    st.caption(
        "Sizes solar PV and battery storage using Manager optimization results, "
        "then calculates a 25-year financial return."
    )

    with st.expander("Site & system parameters", expanded=True):
        c1, c2, c3 = st.columns(3)
        _active_p = get_profile(st.session_state.site_profile_id)
        roof_area   = c1.number_input("Roof area (m²)", 50.0, 10000.0,
                                        float(_so("roof_area_m2", _active_p.expected_roof_area_m2)),
                                        step=50.0)
        panel_w     = c2.number_input("Panel wattage (W)", 300, 700,
                                        int(_so("panel_w", 415)), step=5)
        psh         = c3.number_input("Peak sun hours/day", 3.5, 6.0,
                                        float(_so("psh", 4.5)), step=0.1,
                                       help="Malaysia average: 4.5 h/day")

        c4, c5, c6 = st.columns(3)
        solar_cost  = c4.number_input("Solar cost (RM/kWp)", 2000.0, 6000.0,
                                        float(_so("solar_cost", 3500.0)), step=100.0)
        batt_cost   = c5.number_input("Battery cost (RM/kWh)", 1500.0, 5000.0,
                                        float(_so("batt_cost",  2500.0)), step=100.0)

        # Self-consumption: computed from Manager output if available, with
        # a manual override toggle. The slider used to be the only input;
        # now it acts as a fallback when there's no Manager run yet.
        _computed_sc = None
        if st.session_state.manager_results and st.session_state.solar_info:
            _detected_kwp = st.session_state.solar_info.get("capacity_kwp", 0.0) or 0.0
            try:
                _ef = decompose_flows(st.session_state.manager_results, capacity_kwp=_detected_kwp)
                _computed_sc = _ef["metrics"]["self_consumption_pct"]
            except Exception:
                _computed_sc = None

        with c6:
            if _computed_sc is not None and _computed_sc > 0:
                st.caption(f"📐 Computed self-consumption: **{_computed_sc*100:.1f}%**")
                _override_sc = st.checkbox("Override", value=False, key="sc_override")
                if _override_sc:
                    self_cons = st.slider("Self-consumption (%)", 40, 90, int(_computed_sc * 100),
                                            key="sc_override_slider") / 100.0
                else:
                    self_cons = float(_computed_sc)
            else:
                self_cons = st.slider("Self-consumption (%)", 40, 90, 65) / 100.0

        sched_key = st.session_state.tariff_schedule_key
        st.caption(
            f"Tariff schedule: **{sched_key}** "
            f"(toggle in sidebar to switch). MD savings will be priced against "
            f"this schedule for the selected tariff."
        )

    if st.button("Run PowerRECO Analysis", type="primary", use_container_width=True):
        try:
            with st.spinner("Sizing solar and battery…"):
                solar = calculate_solar_sizing(roof_area, int(panel_w), psh)
                st.session_state.solar_sizing = solar

                if st.session_state.powerreco_df is not None:
                    batt = calculate_battery_sizing(st.session_state.powerreco_df)
                    md_reduction_kw = batt["md_reduction_kw"]
                else:
                    st.warning(
                        "No Manager results — battery sized conservatively. "
                        "Run the AI Manager for a data-driven battery recommendation."
                    )
                    daily_kwh = solar["daily_generation_kwh_avg"]
                    batt = {
                        "min_capacity_kwh_commercial": max(10.0, round(daily_kwh * 0.10 / 50) * 50),
                        "md_reduction_kw": 0.0,
                        "n_days_analyzed": 0,
                        "spike_note": "Estimated — no Manager data.",
                    }
                    md_reduction_kw = 0.0
                st.session_state.battery_sizing = batt

                battery_kwh_rec = float(batt["min_capacity_kwh_commercial"])

                # Run feasibility check — determines if installation is worth it
                _summ = st.session_state.file_summary or {}
                _avg_monthly = float(_summ.get("mean_kw_import", 0)) * 24 * 30 / 2
                _peak_kw     = float(_summ.get("max_kw_import",  0))
                feasibility = assess_feasibility(
                    avg_monthly_kwh=max(_avg_monthly, float(solar["monthly_generation_kwh_avg"])),
                    peak_demand_kw=_peak_kw,
                    tariff=st.session_state.tariff_code,
                    md_reduction_kw=md_reduction_kw,
                    schedule_key=sched_key,
                    monthly_gen_kwh_per_kwp=float(solar["monthly_generation_kwh_avg"]) / max(solar["system_kwp"], 1),
                    solar_cost_per_kwp=solar_cost,
                    battery_cost_per_kwh=batt_cost,
                    battery_kwh=battery_kwh_rec,
                )
                st.session_state["powerreco_feasibility"] = feasibility

                roi = calculate_roi(
                    solar_kwp=solar["system_kwp"],
                    battery_kwh=battery_kwh_rec,
                    monthly_generation_kwh=solar["monthly_generation_kwh_avg"],
                    md_reduction_kw=md_reduction_kw,
                    self_consumption_pct=self_cons,
                    schedule_key=sched_key,
                    tariff=st.session_state.tariff_code,
                    solar_cost_per_kwp=solar_cost,
                    battery_cost_per_kwh=batt_cost,
                )
                st.session_state.roi = roi

            st.success("Analysis complete.")
        except Exception as e:
            st.error(f"PowerRECO failed: {e}")
            st.code(traceback.format_exc())

    if st.session_state.solar_sizing and st.session_state.roi:
        solar       = st.session_state.solar_sizing
        batt        = st.session_state.battery_sizing
        roi         = st.session_state.roi
        feasibility = st.session_state.get("powerreco_feasibility")

        # ─────────────────────────────────────────────────────────────
        # 1. FEASIBILITY GATE — the first thing you see
        # ─────────────────────────────────────────────────────────────
        st.divider()
        st.subheader("🔍 Investment Feasibility")
        st.caption(
            "Pre-screening verdict BEFORE committing to sizing or ROI details.  "
            "VIABLE = confident payback within threshold.  "
            "MARGINAL = borderline — sensitive to tariff escalation.  "
            "NOT VIABLE = unlikely to recover investment."
        )

        if feasibility:
            _VERDICT_COLOR = {
                "VIABLE":             "#22c55e",
                "MARGINAL":           "#f59e0b",
                "NOT_VIABLE":         "#ef4444",
                "INSUFFICIENT_DATA":  "#9ca3af",
            }
            fc1, fc2 = st.columns(2)

            with fc1:
                sv = feasibility["solar_verdict"]
                sc = _VERDICT_COLOR.get(sv, "#9ca3af")
                st.markdown(
                    f"<div style='border:1px solid {sc};border-radius:8px;padding:14px'>"
                    f"<span style='font-size:18px;font-weight:700;color:{sc}'>☀️ Solar — {sv}</span><br>"
                    + "".join(f"<p style='margin:4px 0;font-size:13px'>{r}</p>" for r in feasibility["solar_reasons"])
                    + (f"<p style='margin-top:8px;font-size:13px'>"
                       f"Estimated payback: <b>{feasibility['estimated_payback_solar_yrs']} yrs</b></p>"
                       if feasibility.get("estimated_payback_solar_yrs") else "")
                    + "</div>",
                    unsafe_allow_html=True,
                )

            with fc2:
                bv = feasibility["battery_verdict"]
                bc = _VERDICT_COLOR.get(bv, "#9ca3af")
                st.markdown(
                    f"<div style='border:1px solid {bc};border-radius:8px;padding:14px'>"
                    f"<span style='font-size:18px;font-weight:700;color:{bc}'>🔋 Battery — {bv}</span><br>"
                    + "".join(f"<p style='margin:4px 0;font-size:13px'>{r}</p>" for r in feasibility["battery_reasons"])
                    + (f"<p style='margin-top:8px;font-size:13px'>"
                       f"Estimated payback: <b>{feasibility['estimated_payback_battery_yrs']} yrs</b></p>"
                       if feasibility.get("estimated_payback_battery_yrs") else "")
                    + "</div>",
                    unsafe_allow_html=True,
                )

            st.info(f"**Recommendation:** {feasibility['recommendation']}")
            for cav in feasibility.get("caveats", []):
                st.caption(f"ℹ️ {cav}")

            # If neither component is viable, stop here — don't show misleading ROI
            if not feasibility["solar_viable"] and not feasibility["battery_viable"]:
                st.warning(
                    "Neither solar nor battery investment appears financially viable under "
                    "current parameters.  Review load profile, tariff schedule, and cost inputs.  "
                    "Detailed ROI figures are hidden to avoid misleading analysis."
                )
                st.stop()

        st.divider()

        # ─────────────────────────────────────────────────────────────
        # 2. RECOMMENDED SYSTEM — single best option
        # ─────────────────────────────────────────────────────────────
        st.subheader("⭐ Recommended System")

        r1, r2 = st.columns(2)
        with r1:
            st.markdown("**Solar PV**")
            rc1, rc2, rc3 = st.columns(3)
            rc1.metric("System size",   f"{solar['system_kwp']} kWp")
            rc2.metric("Panels",        f"{solar['n_panels']} × {solar['panel_wattage_w']} W")
            rc3.metric("Annual output", f"{solar['annual_generation_kwh']:,} kWh")
        with r2:
            st.markdown("**Battery Storage**")
            rb1, rb2, rb3 = st.columns(3)
            rb1.metric("Capacity",      f"{batt['min_capacity_kwh_commercial']} kWh")
            rb2.metric("MD reduction",  f"{batt.get('md_reduction_kw', 0):.1f} kW")
            rb3.metric("Days analysed", batt.get("n_days_analyzed", "N/A"))
        if "spike_note" in batt:
            st.caption(batt["spike_note"])

        # ─────────────────────────────────────────────────────────────
        # 3. CAPEX BREAKDOWN — per component, not a single lump sum
        # ─────────────────────────────────────────────────────────────
        st.divider()
        st.subheader("💰 CAPEX Breakdown")

        solar_panels_rm  = roi["solar_capex_rm"]
        inverter_rm      = roi["inverter_capex_rm"]
        battery_rm       = roi["battery_capex_rm"]
        total_capex_rm   = roi["total_capex_rm"]

        capex_c1, capex_c2, capex_c3, capex_c4 = st.columns(4)
        capex_c1.metric(
            "Solar panels",
            _fmt_rm(solar_panels_rm),
            delta=f"RM {solar_cost:,.0f}/kWp × {solar['system_kwp']} kWp",
            delta_color="off",
        )
        capex_c2.metric(
            "Inverter",
            _fmt_rm(inverter_rm),
            delta=f"RM {roi.get('inverter_cost_per_kwp_rm', solar['inverter_cost_per_kwp_rm']):,.0f}/kWp · {solar.get('inverter_tier','')[:25]}",
            delta_color="off",
        )
        capex_c3.metric(
            "Battery",
            _fmt_rm(battery_rm),
            delta=f"RM {batt_cost:,.0f}/kWh × {batt['min_capacity_kwh_commercial']} kWh",
            delta_color="off",
        )
        capex_c4.metric("Total CAPEX", _fmt_rm(total_capex_rm))

        # CAPEX pie chart
        if solar_panels_rm + inverter_rm + battery_rm > 0:
            fig_pie = go.Figure(go.Pie(
                labels=["Solar panels", "Inverter", "Battery"],
                values=[solar_panels_rm, inverter_rm, battery_rm],
                hole=0.45,
                marker=dict(colors=["#f59e0b", "#00b4d8", "#a78bfa"]),
                textinfo="label+percent",
            ))
            fig_pie.update_layout(
                title="CAPEX composition",
                height=280,
                showlegend=False,
                margin=dict(t=40, b=10, l=10, r=10),
            )
            st.plotly_chart(fig_pie, use_container_width=True)

        st.caption(
            f"Inverter note: {solar.get('inverter_note', '')}"
        )

        # ─────────────────────────────────────────────────────────────
        # 4. FINANCIAL RETURNS
        # ─────────────────────────────────────────────────────────────
        st.divider()
        st.subheader("📈 25-Year Financial Return")

        fin1, fin2, fin3, fin4 = st.columns(4)
        fin1.metric("Simple payback",  f"{roi['simple_payback_years']} yrs")
        fin2.metric("NPV (25 yr)",     _fmt_rm(roi["npv_25yr_rm"]))
        fin3.metric("IRR",             f"{roi['irr_pct']}%" if roi["irr_pct"] else "N/A")
        fin4.metric("Monthly net benefit", _fmt_rm(roi["monthly_net_benefit_rm"]))

        ann1, ann2, ann3 = st.columns(3)
        ann1.metric("Annual energy savings", _fmt_rm(roi["annual_energy_savings_rm"]))
        ann2.metric("Annual MD savings",     _fmt_rm(roi["annual_md_savings_rm"]))
        ann3.metric("Annual NEM credit",     _fmt_rm(roi["annual_nem_credit_rm"]))

        st.caption(
            f"CO₂ offset: **{roi['co2_offset_tonnes_yr']:.1f} t/year**  ·  "
            f"Self-consumption: {self_cons*100:.0f}%  ·  "
            f"MD rate: RM {roi['md_rate_used']:.2f}/kW/month  ·  "
            f"Tax multiplier: {roi.get('tax_multiplier_used', 1.077):.3f}"
        )

        # Cumulative NPV chart
        years = list(range(1, len(roi["cumulative_npv"]) + 1))
        fig2 = go.Figure()
        fig2.add_hline(y=0, line_dash="dash", line_color="gray")
        # Battery replacement annotation at year 10
        fig2.add_vline(x=10, line_dash="dot", line_color="#f59e0b",
                       annotation_text="Battery replacement", annotation_position="top right")
        fig2.add_trace(go.Scatter(
            x=years, y=roi["cumulative_npv"],
            mode="lines+markers", name="Cumulative NPV",
            line=dict(color="#2ca02c", width=2),
            fill="tozeroy", fillcolor="rgba(44,160,44,0.15)",
            marker=dict(size=4),
        ))
        fig2.update_layout(
            title="Cumulative NPV over 25 Years",
            xaxis_title="Year", yaxis_title="RM",
            height=380,
        )
        st.plotly_chart(fig2, use_container_width=True)

        # Monthly solar generation chart
        with st.expander("Monthly solar generation profile"):
            fig_sol = go.Figure(go.Bar(
                x=solar["month_labels"],
                y=solar["monthly_breakdown_kwh"],
                marker_color="#f59e0b",
                name="Monthly kWh",
            ))
            fig_sol.update_layout(
                title="Monthly generation (Malaysia seasonal factors applied)",
                yaxis_title="kWh", height=280,
            )
            st.plotly_chart(fig_sol, use_container_width=True)

        # Summary download
        summary = {
            "solar_kwp":              solar["system_kwp"],
            "battery_kwh":            batt["min_capacity_kwh_commercial"],
            "solar_capex_rm":         roi["solar_capex_rm"],
            "inverter_capex_rm":      roi["inverter_capex_rm"],
            "battery_capex_rm":       roi["battery_capex_rm"],
            "total_capex_rm":         roi["total_capex_rm"],
            "feasibility_solar":      feasibility["solar_verdict"]   if feasibility else "N/A",
            "feasibility_battery":    feasibility["battery_verdict"] if feasibility else "N/A",
            **{k: v for k, v in roi.items() if k != "cumulative_npv"},
        }
        import json
        st.download_button(
            "Download ROI Summary (JSON)",
            data=json.dumps(summary, indent=2).encode(),
            file_name="powerreco_roi.json",
            mime="application/json",
        )

        # ─────────────────────────────────────────────────────────────
        # ENERGY FLOWS — stacked daily chart + annual Sankey + KPIs
        # ─────────────────────────────────────────────────────────────
        if st.session_state.manager_results:
            st.divider()
            st.subheader("⚡ Energy Flow Analysis")
            st.caption(
                "Decomposes the optimised period into the five energy flows. "
                "Solar self-consumption + grid independence are derived directly "
                "from the Manager's interval-by-interval dispatch."
            )

            _detected_kwp = (st.session_state.solar_info or {}).get("capacity_kwp", 0.0) or 0.0
            efp_kwp = st.number_input(
                "Solar capacity used for flow attribution (kWp)",
                min_value=0.0, value=float(_detected_kwp), step=10.0,
                help="Defaults to auto-detected capacity. Set 0 for a no-solar site."
            )

            try:
                ef = decompose_flows(st.session_state.manager_results, capacity_kwp=efp_kwp)
                m   = ef["metrics"]
                tot = ef["totals"]
                ann = ef["annual"]

                # Sustainability KPI cards (replaces the slider-as-input)
                k1, k2, k3, k4 = st.columns(4)
                k1.metric("Self-consumption", f"{m['self_consumption_pct']*100:.1f}%",
                          help="Share of solar generation consumed on-site (load + battery), "
                               "not exported to grid.")
                k2.metric("Grid independence", f"{m['grid_independence_pct']*100:.1f}%",
                          help="Share of total load NOT covered by grid imports.")
                k3.metric("Solar share of load", f"{m['solar_share_of_load_pct']*100:.1f}%",
                          help="Share of load met by solar (directly + via battery).")
                k4.metric("Battery throughput", f"{m['battery_throughput_kwh']:,.0f} kWh",
                          help="One-way energy moved through the battery over the period.")

                # Stacked area chart — representative day
                rd = representative_day(ef["df"])
                if not rd.empty:
                    fig_sa = go.Figure()
                    fig_sa.add_trace(go.Scatter(
                        x=rd["timestamp"], y=rd["solar_to_load"],
                        name="Solar → Load", stackgroup="one",
                        mode="lines", line=dict(width=0), fillcolor="#f59e0b",
                    ))
                    fig_sa.add_trace(go.Scatter(
                        x=rd["timestamp"], y=rd["battery_to_load"],
                        name="Battery → Load", stackgroup="one",
                        mode="lines", line=dict(width=0), fillcolor="#a78bfa",
                    ))
                    fig_sa.add_trace(go.Scatter(
                        x=rd["timestamp"], y=rd["grid_to_load"],
                        name="Grid → Load", stackgroup="one",
                        mode="lines", line=dict(width=0), fillcolor="#6b7a9a",
                    ))
                    # Solar generation reference line (the curve being attributed)
                    fig_sa.add_trace(go.Scatter(
                        x=rd["timestamp"], y=rd["solar_gen_kwh"],
                        name="Total solar generation", mode="lines",
                        line=dict(color="#f97316", width=2, dash="dot"),
                    ))
                    fig_sa.update_layout(
                        title=f"Representative day energy stack ({rd['timestamp'].iloc[0].date()})",
                        xaxis_title="Time", yaxis_title="kWh per 30-min interval",
                        height=380, hovermode="x unified",
                    )
                    st.plotly_chart(fig_sa, use_container_width=True)

                # Annual Sankey
                # Build nodes + links. Nodes are 0=Solar, 1=Grid, 2=Battery, 3=Load, 4=Export
                nodes = ["Solar", "Grid", "Battery", "Load", "Export"]
                flows = [
                    ("Solar",   "Load",    ann.get("solar_to_load",    0)),
                    ("Solar",   "Battery", ann.get("solar_to_battery", 0)),
                    ("Solar",   "Export",  ann.get("solar_to_grid",    0)),
                    ("Battery", "Load",    ann.get("battery_to_load",  0)),
                    ("Grid",    "Load",    ann.get("grid_to_load",     0)),
                ]
                src_idx, tgt_idx, vals, labels = [], [], [], []
                for src, tgt, v in flows:
                    if v <= 0.5:
                        continue
                    src_idx.append(nodes.index(src))
                    tgt_idx.append(nodes.index(tgt))
                    vals.append(float(v))
                    labels.append(f"{v:,.0f} kWh/yr")
                if vals:
                    fig_sk = go.Figure(go.Sankey(
                        node=dict(
                            pad=18, thickness=20, line=dict(color="black", width=0.4),
                            label=nodes,
                            color=["#f59e0b", "#6b7a9a", "#a78bfa", "#22c55e", "#ec4899"],
                        ),
                        link=dict(source=src_idx, target=tgt_idx,
                                   value=vals, label=labels,
                                   color=["rgba(245,158,11,.3)",
                                          "rgba(245,158,11,.3)",
                                          "rgba(245,158,11,.3)",
                                          "rgba(167,139,250,.4)",
                                          "rgba(107,122,154,.35)"][:len(vals)]),
                    ))
                    fig_sk.update_layout(
                        title=f"Annual energy flows (scaled from {m['period_days']:.0f} measured days)",
                        height=380, font=dict(size=12),
                    )
                    st.plotly_chart(fig_sk, use_container_width=True)
                else:
                    st.info("No non-trivial energy flows to display — try a longer Manager period.")

                # Period totals table
                with st.expander("Energy flow totals (period + annualised)"):
                    flow_keys = [
                        "solar_to_load", "solar_to_battery", "solar_to_grid",
                        "battery_to_load", "grid_to_load", "load_kwh", "solar_gen_kwh",
                    ]
                    flow_df = pd.DataFrame({
                        "Flow":      [k.replace("_", " ").title() for k in flow_keys],
                        "Period kWh":   [tot.get(k, 0) for k in flow_keys],
                        "Annual kWh":   [ann.get(k, 0) for k in flow_keys],
                    })
                    st.dataframe(flow_df, use_container_width=True, hide_index=True)

            except Exception as e:
                st.warning(f"Energy flow analysis failed: {e}")
                with st.expander("Traceback"):
                    st.code(traceback.format_exc())
