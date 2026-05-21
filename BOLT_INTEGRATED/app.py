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

from manager.optimizer import run_ai_manager, parse_uploaded_data, calc_kva

from calculator.tnb_tariffs import (
    auto_detect_tariff, compute_monthly_stats, calculate_bill,
    compute_nem_credit, TARIFF_META,
)

from powerreco.solar_sizing import calculate_solar_sizing
from powerreco.battery_sizing import calculate_battery_sizing
from powerreco.roi_engine import calculate_roi
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

from persistence import load_overrides, save_overrides, OVERRIDES_PATH

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
tab0, tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🛠️ Site Setup",
    "📂 Data Upload",
    "📈 Predictor",
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

    with st.expander("📍 Location (drives weather + irradiance lookup)", expanded=True):
        l1, l2, l3 = st.columns(3)
        new_lat = l1.number_input(
            "Latitude",  -90.0, 90.0,
            float(_so("lat", _active.lat)), step=0.001, format="%.4f",
            help="Used to pull real ambient temperature + shortwave irradiance "
                 "from Open-Meteo for both training and live forecasts. "
                 "Defaults to Kuala Lumpur.",
        )
        new_lon = l2.number_input(
            "Longitude", -180.0, 180.0,
            float(_so("lon", _active.lon)), step=0.001, format="%.4f",
        )
        new_tz  = l3.text_input(
            "Timezone", _so("timezone", _active.timezone),
            help="Either an IANA name (e.g. 'Asia/Kuala_Lumpur') or 'auto' "
                 "to infer from lat/lon.",
        )
        new_use_weather = st.checkbox(
            "Use real weather data (Open-Meteo)",
            value=bool(st.session_state.use_real_weather),
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
            step=10.0,
            help="0 if no solar installed yet. Used by the forecaster (for "
                 "net-load shape) and by PowerRECO (as the starting point "
                 "for sizing recommendations).",
        )
        new_roof_area = s2.number_input(
            "Roof area (m²)", 50.0, 10000.0,
            float(_so("roof_area_m2", _active.expected_roof_area_m2)),
            step=50.0,
            help="Caps the maximum solar PV size PowerRECO will recommend.",
        )
        new_panel_w = s3.number_input(
            "Panel wattage (W)", 300, 700,
            int(_so("panel_w", 415)), step=5,
        )
        s4, s5 = st.columns(2)
        new_psh = s4.number_input(
            "Peak sun hours/day", 3.5, 6.0, float(_so("psh", 4.5)), step=0.1,
            help="Malaysia average ~4.5 h/day. Override with site-specific PVGIS value if known.",
        )
        new_solar_cost = s5.number_input(
            "Solar cost (RM/kWp)", 2000.0, 6000.0,
            float(_so("solar_cost", 3500.0)), step=100.0,
        )

    with st.expander("🔋 Battery & dispatch", expanded=False):
        b1, b2, b3 = st.columns(3)
        new_battery_kwh = b1.number_input(
            "Battery capacity (kWh)", 0.0, 5000.0,
            float(_so("battery_kwh", _active.battery_kwh)), step=10.0,
            help="0 = no battery. Used by the Manager for dispatch and by "
                 "PowerRECO as the starting battery size.",
        )
        new_c_rate = b2.number_input(
            "C-rate", 0.1, 1.0, float(_so("c_rate", _active.c_rate)), step=0.1,
        )
        new_bat_cost = b3.number_input(
            "Battery cost (RM/kWh)", 1500.0, 5000.0,
            float(_so("batt_cost", 2500.0)), step=100.0,
        )
        b4, b5, b6 = st.columns(3)
        new_md_target = b4.slider(
            "MD target (% of ref peak)", 70, 95,
            int(_so("md_target_pct", _active.md_target_pct) * 100),
        ) / 100.0
        new_charge_upper = b5.slider(
            "Charge upper threshold (%)", 50, 90,
            int(_so("charge_upper_pct", _active.charge_upper_pct) * 100),
        ) / 100.0
        new_init_soc = b6.slider(
            "Initial SOC (%)", 20, 80,
            int(_so("init_soc_pct", _active.initial_soc_pct) * 100),
        ) / 100.0

    with st.expander("🚙 EV chargers", expanded=False):
        _ev_default = _active.loads.get("ev")
        _ev_charger = _ev_default.ev_chargers[0] if (_ev_default and _ev_default.ev_chargers) else None
        e1, e2, e3 = st.columns(3)
        new_ev_count = e1.number_input(
            "# chargers", 0, 100,
            int(_so("ev_count", _ev_charger.count if _ev_charger else 4)),
        )
        new_ev_kw = e2.number_input(
            "kW each", 3.6, 250.0,
            float(_so("ev_kw_each", _ev_charger.kw_each if _ev_charger else 22.0)),
            step=1.0,
        )
        new_ev_kind = e3.selectbox(
            "Connector",
            ["AC", "DC"],
            index=0 if _so("ev_kind", _ev_charger.kind if _ev_charger else "AC") == "AC" else 1,
        )
        e4, e5 = st.columns(2)
        _ev_window = _ev_default.allowed_window if (_ev_default and _ev_default.allowed_window) else (18, 8)
        new_ev_ws = e4.number_input(
            "Charge window start (h)", 0, 23,
            int(_so("ev_window_start", _ev_window[0])),
        )
        new_ev_we = e5.number_input(
            "Charge window end (h)",   0, 23,
            int(_so("ev_window_end",   _ev_window[1])),
            help="Window wraps overnight: 18 → 8 means 18:00 to 08:00 next morning.",
        )

    with st.expander("❄️ HVAC", expanded=False):
        _hvac = _active.loads.get("hvac")
        _hwin = _hvac.protected_window if (_hvac and _hvac.protected_window) else (9, 17)
        h1, h2, h3 = st.columns(3)
        new_hvac_protect_start = h1.number_input(
            "Protected from (h)", 0, 23,
            int(_so("hvac_protect_start", _hwin[0])),
        )
        new_hvac_protect_end = h2.number_input(
            "Protected to (h)", 0, 23,
            int(_so("hvac_protect_end", _hwin[1])),
        )
        new_hvac_cut = h3.number_input(
            "Max cut % outside protect", 0, 100,
            int(_so("hvac_max_cut_pct", (_hvac.max_cut_pct * 100) if _hvac else 15)),
        )

    with st.expander("💰 Tariff & financial", expanded=False):
        t1, t2, t3 = st.columns(3)
        tariff_options = list(TARIFF_META.keys())
        _cur_tariff = _so("tariff_code", st.session_state.tariff_code)
        new_tariff = t1.selectbox(
            "Tariff code", tariff_options,
            index=tariff_options.index(_cur_tariff) if _cur_tariff in tariff_options else 2,
            format_func=lambda c: f"{c} — {TARIFF_META[c]['name']}",
            help="Auto-detected from your data, overridable here.",
        )
        new_icpt = t2.number_input(
            "ICPT (sen/kWh)", -10.0, 20.0,
            float(_so("icpt_sen", 0.0)), step=0.5,
        )
        new_nem = t3.number_input(
            "NEM buyback rate (RM/kWh)", 0.20, 0.50,
            float(_so("nem_rate", 0.31)), step=0.01,
        )
        new_budget = st.number_input(
            "Max CAPEX budget (RM, 0 = no cap)", 0, 5_000_000,
            int(_so("budget_rm", 0)), step=50_000,
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
                with st.spinner("Re-running full pipeline…"):
                    # 1. Train predictor with new params
                    fc_kwargs = profile_predictor_kwargs(_active)
                    fc_kwargs["lat"] = new_lat if new_use_weather else None
                    fc_kwargs["lon"] = new_lon if new_use_weather else None
                    fc_kwargs["timezone"] = new_tz if new_use_weather else "auto"
                    fc = DirectMultiStepForecaster(
                        capacity_kwp=float(new_solar_kwp), **fc_kwargs,
                    )
                    metrics = fc.fit(st.session_state.df, verbose=False)
                    st.session_state.forecaster = fc
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

            st.success(f"Loaded {summ['rows']:,} intervals  |  "
                       f"{summ['days']} days  |  "
                       f"Peak {summ['max_kw_import']:.1f} kW")

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

    col_train, col_cv = st.columns([2, 1])
    with col_train:
        if st.button("Train Forecaster", type="primary", use_container_width=True):
            try:
                with st.spinner("Training LightGBM models for all horizons…"):
                    fc_kwargs = profile_predictor_kwargs(profile)
                    # User-overridable values take precedence over profile defaults
                    fc_kwargs.update({
                        "n_estimators":  int(n_estimators),
                        "learning_rate": lr,
                    })
                    # lat/lon + real-weather flag from Site Setup overrides
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
                    metrics = fc.fit(df, verbose=False)
                st.session_state.forecaster = fc
                st.success(
                    f"Trained {metrics['n_models_trained']} models. "
                    f"Mean MAPE: {metrics['mean_mape']:.2f}%  |  "
                    f"MAPE@24h: {metrics.get('mape_at_h24', 0):.2f}%"
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

        except Exception as e:
            st.error(f"Forecast failed: {e}")
            st.code(traceback.format_exc())


# ─────────────────────────────────────────────────────────────────────────────
# TAB 3: AI MANAGER
# ─────────────────────────────────────────────────────────────────────────────
with tab3:
    st.header("AI Load Manager")
    st.caption(
        "Runs peak-shaving and load-shifting optimization on the loaded data. "
        "Uses the exact discharge formula: dis_kw = kW − √(trigger² − kVAR²)."
    )

    if st.session_state.df is None:
        st.warning("Upload a load profile first.")
        st.stop()

    df = st.session_state.df
    profile = get_profile(st.session_state.site_profile_id)

    with st.expander("Battery & optimization settings", expanded=True):
        c1, c2, c3 = st.columns(3)
        battery_kwh   = c1.number_input("Battery capacity (kWh)", 50.0, 5000.0,
                                          float(_so("battery_kwh", profile.battery_kwh)) or profile.battery_kwh, step=50.0)
        c_rate        = c2.number_input("C-rate", 0.1, 1.0,
                                          float(_so("c_rate", profile.c_rate)), step=0.1,
                                         help="Max charge/discharge rate as fraction of capacity per hour")
        bat_eff       = c3.number_input("One-way efficiency", 0.80, 1.00, 0.95, step=0.01)

        c4, c5, c6 = st.columns(3)
        peak_target   = c4.slider("Peak target (% of ref peak)", 70, 95,
                                    int(_so("md_target_pct", profile.md_target_pct) * 100)) / 100
        charge_upper  = c5.slider("Charge upper threshold (% of ref peak)", 50, 90,
                                    int(_so("charge_upper_pct", profile.charge_upper_pct) * 100)) / 100
        init_soc      = c6.slider("Initial SOC (%)", 20, 80,
                                    int(_so("init_soc_pct", profile.initial_soc_pct) * 100)) / 100

        c7, c8 = st.columns(2)
        peak_ref_kva  = c7.number_input(
            "Reference peak kVA (0 = auto from rolling 30-day)", 0.0, 10000.0, 0.0, step=10.0)
        pre_md_hours  = c8.number_input("Pre-MD boost window (hours)", 0, 4, 2)

    st.subheader("Load configuration")
    st.caption(
        f"Loads pre-populated from **{profile.name}**. EV charger details are "
        f"first-class (count, kW, charging window) and feed the Manager's "
        f"window-aware cut logic."
    )

    loads: dict = {}
    priority_order: list[str] = []

    # Profile-derived loads, rendered as per-load editable cards
    profile_load_keys = list(profile.loads.keys())
    cols = st.columns(len(profile_load_keys))
    for col, lk in zip(cols, profile_load_keys):
        ld = profile.loads[lk]
        with col:
            st.markdown(f"**{ld.name}**  \n*{ld.kind}*")
            lname = st.text_input(f"Name [{lk}]", ld.name, key=f"mgr_name_{lk}",
                                   label_visibility="collapsed")
            lprop = st.number_input(f"Proportion [{lk}]", 0.0, 1.0,
                                     float(ld.proportion), step=0.05,
                                     key=f"mgr_prop_{lk}")
            lcut  = st.number_input(f"Max cut % [{lk}]", 0, 100,
                                     int(ld.max_cut_pct * 100),
                                     key=f"mgr_cut_{lk}")
            entry = {
                "name":        lname,
                "proportion":  lprop,
                "max_cut_pct": lcut / 100,
                "kind":        ld.kind,
            }

            # EV-specific block — reads from Site Setup overrides if present
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
                e_we = st.number_input("Charge window end (h)",   0, 23,
                                        int(_so("ev_window_end",   w_end_d)),
                                        key=f"mgr_evwe_{lk}")
                entry["ev_chargers"]   = [{"count": int(e_count), "kw_each": float(e_kw), "kind": e_kind}]
                entry["ev_total_kw"]   = int(e_count) * float(e_kw)
                entry["allowed_window"] = [int(e_ws), int(e_we)]

            # HVAC protected-hours block
            if ld.kind == "hvac" and ld.protected_window:
                st.caption("❄️ Protected business hours")
                w_start_d, w_end_d = ld.protected_window
                p_ws = st.number_input("Protect from (h)", 0, 23,
                                        int(_so("hvac_protect_start", w_start_d)),
                                        key=f"mgr_hws_{lk}")
                p_we = st.number_input("Protect to (h)",   0, 23,
                                        int(_so("hvac_protect_end", w_end_d)),
                                        key=f"mgr_hwe_{lk}")
                entry["protected_window"] = [int(p_ws), int(p_we)]

            loads[lk] = entry
            priority_order.append(lk)

    if st.button("Run AI Manager", type="primary", use_container_width=True):
        try:
            with st.spinner("Optimizing load profile…"):
                results = _run_manager_on_df(
                    df, loads, priority_order,
                    battery_kwh, peak_target, charge_upper,
                    c_rate, init_soc, bat_eff,
                    peak_ref_kva if peak_ref_kva > 0 else None,
                )
            st.session_state.manager_results    = results
            st.session_state.manager_df_optimized = manager_results_to_sam_df(results)
            st.session_state.manager_df_original  = manager_results_to_original_df(results)
            st.session_state.powerreco_df         = manager_results_to_powerreco_df(results)
            st.success(f"Optimization complete — {len(results):,} intervals processed.")
        except Exception as e:
            st.error(f"Manager failed: {e}")
            st.code(traceback.format_exc())

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
    st.caption("Compares monthly electricity bills before and after AI Manager optimization.")

    if st.session_state.df is None:
        st.warning("Upload a load profile first.")
        st.stop()

    if st.session_state.manager_df_optimized is None:
        st.info("Run the AI Manager first to see before/after bill comparison. "
                "The calculator will use raw uploaded data for now.")
        df_for_bill  = st.session_state.df
        df_orig_bill = st.session_state.df
        show_comparison = False
    else:
        df_for_bill  = st.session_state.manager_df_optimized
        df_orig_bill = st.session_state.manager_df_original
        show_comparison = True

    with st.expander("Tariff & billing settings", expanded=True):
        c1, c2, c3 = st.columns(3)
        tariff_options = list(TARIFF_META.keys())
        _tariff_default = _so("tariff_code", st.session_state.tariff_code)
        default_idx    = tariff_options.index(_tariff_default) if _tariff_default in tariff_options else 2
        tariff         = c1.selectbox("Tariff", tariff_options, index=default_idx,
                                       format_func=lambda t: f"{t} — {TARIFF_META[t]['name']}")
        icpt_sen       = c2.number_input("ICPT (sen/kWh)", -10.0, 20.0,
                                          float(_so("icpt_sen", 0.0)), step=0.5,
                                          help="Positive = surcharge, negative = rebate")
        nem_rate       = c3.number_input("NEM buyback rate (RM/kWh)", 0.20, 0.50,
                                          float(_so("nem_rate", 0.31)), step=0.01)

    sched_key = st.session_state.tariff_schedule_key
    st.caption(
        f"Schedule in use: **{sched_key}** "
        f"(toggle in sidebar to switch between July 2025 and legacy rates)."
    )

    try:
        monthly_stats_opt  = compute_monthly_stats(df_for_bill, schedule_key=sched_key)
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

        opt_df  = pd.DataFrame(bill_rows_opt)
        total_opt  = opt_df["Net Bill (RM)"].sum()

        if show_comparison:
            orig_df   = pd.DataFrame(bill_rows_orig)
            total_orig = orig_df["Net Bill (RM)"].sum()
            savings    = total_orig - total_opt

            c1, c2, c3 = st.columns(3)
            c1.metric("Total Bill (Before)", _fmt_rm(total_orig))
            c2.metric("Total Bill (After)",  _fmt_rm(total_opt), delta=f"-{_fmt_rm(savings)}", delta_color="inverse")
            c3.metric("Annual Savings",       _fmt_rm(savings * 12 / max(len(monthly_stats_opt), 1)))

            # Chart
            if len(opt_df) > 0 and "Month" in opt_df.columns:
                fig = go.Figure()
                if len(orig_df):
                    fig.add_trace(go.Bar(name="Before", x=orig_df["Month"], y=orig_df["Net Bill (RM)"],
                                         marker_color="#d62728"))
                fig.add_trace(go.Bar(name="After",  x=opt_df["Month"],  y=opt_df["Net Bill (RM)"],
                                     marker_color="#1f77b4"))
                fig.update_layout(barmode="group", title="Monthly Bill Comparison",
                                  yaxis_title="RM", height=350)
                st.plotly_chart(fig, use_container_width=True)
        else:
            c1, c2 = st.columns(2)
            c1.metric("Total Bill (period)", _fmt_rm(total_opt))
            c2.metric("Avg Monthly",         _fmt_rm(total_opt / max(len(opt_df), 1)))

        st.subheader("Monthly Bill Breakdown (After Optimization)")
        st.dataframe(opt_df, use_container_width=True, hide_index=True)

        # Download
        buf = io.StringIO()
        opt_df.to_csv(buf, index=False)
        st.download_button("Download Bill CSV", buf.getvalue().encode(),
                           "bill_breakdown.csv", "text/csv")

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

                # Battery sizing from Manager results if available
                if st.session_state.powerreco_df is not None:
                    batt = calculate_battery_sizing(st.session_state.powerreco_df)
                    md_reduction_kw = batt["md_reduction_kw"]
                else:
                    st.warning(
                        "No Manager results — battery sized conservatively to 10% of daily solar. "
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

                roi = calculate_roi(
                    solar_kwp=solar["system_kwp"],
                    battery_kwh=battery_kwh_rec,
                    monthly_generation_kwh=solar["monthly_generation_kwh_avg"],
                    md_reduction_kw=float(batt.get("md_reduction_kw", 0.0)),
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
        solar = st.session_state.solar_sizing
        batt  = st.session_state.battery_sizing
        roi   = st.session_state.roi

        st.divider()
        st.subheader("Solar System")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("System Size",     f"{solar['system_kwp']} kWp")
        c2.metric("Panels",          f"{solar['n_panels']} × {solar['panel_wattage_w']}W")
        c3.metric("Annual Output",   f"{solar['annual_generation_kwh']:,} kWh")
        c4.metric("Usable Roof",     f"{solar['usable_area_m2']} m²")

        # Monthly generation chart
        fig = go.Figure(go.Bar(
            x=solar["month_labels"],
            y=solar["monthly_breakdown_kwh"],
            marker_color="#ff7f0e",
            name="Monthly kWh",
        ))
        fig.update_layout(title="Monthly Solar Generation (Malaysia seasonal factors)",
                          yaxis_title="kWh", height=300)
        st.plotly_chart(fig, use_container_width=True)

        st.subheader("Battery Sizing")
        c1, c2, c3 = st.columns(3)
        c1.metric("Recommended",     f"{batt['min_capacity_kwh_commercial']} kWh")
        c2.metric("MD Reduction",    f"{batt.get('md_reduction_kw', 0):.1f} kW")
        c3.metric("Days Analysed",   batt.get("n_days_analyzed", "N/A"))
        if "spike_note" in batt:
            st.caption(batt["spike_note"])

        st.subheader("25-Year Financial Return")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Total CAPEX",     _fmt_rm(roi["total_capex_rm"]))
        c2.metric("Simple Payback",  f"{roi['simple_payback_years']} yrs")
        c3.metric("NPV (25 yr)",     _fmt_rm(roi["npv_25yr_rm"]))
        c4.metric("IRR",             f"{roi['irr_pct']}%" if roi["irr_pct"] else "N/A")

        c1, c2, c3 = st.columns(3)
        c1.metric("Annual Energy Savings", _fmt_rm(roi["annual_energy_savings_rm"]))
        c2.metric("Annual MD Savings",     _fmt_rm(roi["annual_md_savings_rm"]))
        c3.metric("Annual NEM Credit",     _fmt_rm(roi["annual_nem_credit_rm"]))

        # Cumulative NPV chart
        years = list(range(1, len(roi["cumulative_npv"]) + 1))
        fig2 = go.Figure()
        fig2.add_hline(y=0, line_dash="dash", line_color="gray")
        fig2.add_trace(go.Scatter(
            x=years, y=roi["cumulative_npv"],
            mode="lines+markers", name="Cumulative NPV",
            line=dict(color="#2ca02c", width=2),
            fill="tozeroy", fillcolor="rgba(44,160,44,0.15)",
        ))
        fig2.update_layout(title="Cumulative NPV over 25 Years",
                           xaxis_title="Year", yaxis_title="RM",
                           height=380)
        st.plotly_chart(fig2, use_container_width=True)

        st.caption(
            f"CO₂ offset: **{roi['co2_offset_tonnes_yr']:.1f} tonnes/year**  |  "
            f"Self-consumption: {self_cons*100:.0f}%  |  "
            f"MD rate: RM {roi['md_rate_used']}/kW/month"
        )

        # Summary download
        summary = {
            "solar_kwp": solar["system_kwp"],
            "battery_kwh_recommended": batt["min_capacity_kwh_commercial"],
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
        # OPTIMUM SIZING — sweep (solar × battery) grid for best NPV
        # ─────────────────────────────────────────────────────────────
        st.divider()
        st.subheader("🔍 Find optimum sizing")
        st.caption(
            "Sweeps a grid of solar + battery sizes against the same tariff "
            "schedule and reports the configuration with the highest 25-yr NPV. "
            "MD reduction at non-measured battery sizes is approximated from "
            "the measured Manager result with diminishing-returns scaling."
        )

        ob1, ob2 = st.columns([2, 1])
        with ob1:
            budget_rm = st.number_input(
                "Budget ceiling (RM, 0 = no cap)", 0, 5_000_000,
                int(_so("budget_rm", 0)), step=50_000,
                help="If set, also reports the best NPV configuration with total CAPEX ≤ this. "
                     "Defaults to the Site Setup value.",
            )
        with ob2:
            run_opt = st.button("Run sweep", use_container_width=True)

        if run_opt:
            try:
                opt = find_optimum_from_existing_run(
                    solar_sizing=solar,
                    battery_sizing=batt,
                    schedule_key=sched_key,
                    tariff=st.session_state.tariff_code,
                    self_consumption_pct=self_cons,
                    solar_cost_per_kwp=solar_cost,
                    battery_cost_per_kwh=batt_cost,
                    budget_rm=float(budget_rm) if budget_rm > 0 else None,
                )
                st.session_state.sizing_sweep = opt
            except Exception as e:
                st.error(f"Optimizer failed: {e}")
                st.code(traceback.format_exc())

        if st.session_state.get("sizing_sweep"):
            opt = st.session_state.sizing_sweep
            best = opt.get("best_npv")
            best_b = opt.get("best_under_budget")
            best_pb = opt.get("best_payback")

            cb1, cb2, cb3 = st.columns(3)
            if best:
                cb1.metric(
                    "Best NPV config",
                    f"{best['solar_kwp']:.0f} kWp · {best['battery_kwh']:.0f} kWh",
                    delta=f"NPV {best['npv_25yr_rm']:,.0f} RM",
                )
            if best_pb:
                cb2.metric(
                    "Shortest payback",
                    f"{best_pb['solar_kwp']:.0f} kWp · {best_pb['battery_kwh']:.0f} kWh",
                    delta=f"{best_pb['payback_yrs']:.1f} yrs",
                )
            if best_b:
                cb3.metric(
                    f"Best under RM {opt['budget_rm']:,.0f}" if opt.get('budget_rm') else "Best under budget",
                    f"{best_b['solar_kwp']:.0f} kWp · {best_b['battery_kwh']:.0f} kWh",
                    delta=f"NPV {best_b['npv_25yr_rm']:,.0f} RM",
                )
            elif opt.get('budget_rm'):
                cb3.warning("No config fits the budget — try increasing it or lowering unit costs.")

            # NPV heatmap
            try:
                import plotly.express as px
                import numpy as np
                z = np.array(opt['npv_matrix']) / 1000.0  # show as k-RM
                fig_hm = go.Figure(data=go.Heatmap(
                    z=z,
                    x=[f"{int(b)} kWh" for b in opt['grid_battery_kwh']],
                    y=[f"{int(s)} kWp" for s in opt['grid_solar_kwp']],
                    colorscale="Viridis",
                    colorbar=dict(title="NPV (k RM)"),
                    hovertemplate="Solar %{y}<br>Battery %{x}<br>NPV %{z:.0f}k RM<extra></extra>",
                ))
                fig_hm.update_layout(
                    title="25-yr NPV by (Solar × Battery)",
                    xaxis_title="Battery capacity",
                    yaxis_title="Solar capacity",
                    height=380,
                )
                st.plotly_chart(fig_hm, use_container_width=True)
            except Exception as e:
                st.warning(f"Heatmap render failed: {e}")

            # Full grid table
            with st.expander("Full sweep results"):
                pts_df = pd.DataFrame(opt['points'])
                st.dataframe(pts_df, use_container_width=True, hide_index=True)

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
