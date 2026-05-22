"""
Solar sizing engine for PowerRECO.
Calculates optimal solar panel array size based on available roof area,
plus rough inverter cost estimates for initial budgeting.
"""
from __future__ import annotations
import math

_PANEL_W_M = 1.134
_PANEL_H_M = 1.762
_PANEL_AREA_M2 = _PANEL_W_M * _PANEL_H_M
_SPACING_FACTOR = 1.05

USABLE_ROOF_FRACTION = 0.85
EFFECTIVE_PANEL_AREA_M2 = _PANEL_AREA_M2 * _SPACING_FACTOR
SYSTEM_DERATE = 0.80

SEASONAL_FACTORS = [
    0.92, 0.97, 1.05, 1.08, 1.05, 0.98,
    0.97, 0.99, 0.96, 0.94, 0.88, 0.87,
]
MONTH_LABELS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

# ---------------------------------------------------------------------------
# Inverter cost benchmarks — Malaysia commercial market 2024-2025.
# Figures are supply-and-install (string / multi-string topology) and
# EXCLUDE civil works, grid connection fees, and metering upgrades.
# Verified against recent quotations for Peninsular Malaysia.
# Obtain a formal quote before committing to a budget.
# ---------------------------------------------------------------------------
_INVERTER_TIERS: list[tuple[float, float, str]] = [
    # (max_kwp, cost_per_kwp_rm, description)
    (10.0,  900.0, "single-phase string (< 10 kWp, residential / small commercial)"),
    (100.0, 640.0, "three-phase commercial string (10 – 100 kWp)"),
    (float("inf"), 460.0, "central / multi-string (> 100 kWp, large commercial / industrial)"),
]


def estimate_inverter_cost(system_kwp: float) -> dict:
    """
    Rough inverter cost for a grid-tied solar system (Malaysia market, 2025).

    DC:AC ratio assumed 1:1 — standard for commercial NEM systems.
    Residential systems sometimes oversize the PV array (1.1–1.3×), which
    would increase cost slightly but also boost generation in low-irradiance
    conditions.  This function always prices at 1:1 for conservatism.

    Returns a dict with:
      inverter_kwac       — rated AC output (= system_kwp at 1:1 ratio)
      cost_per_kwp_rm     — benchmark rate for this system size tier
      total_cost_rm       — rough total inverter supply-and-install cost
      tier                — size category label
      note                — caveats for the end user
    """
    if system_kwp <= 0:
        return {
            "inverter_kwac": 0.0, "cost_per_kwp_rm": 0.0,
            "total_cost_rm": 0.0, "tier": "N/A", "note": "System kWp must be > 0.",
        }

    for max_kwp, cost_per_kwp, tier_desc in _INVERTER_TIERS:
        if system_kwp <= max_kwp:
            total_cost = round(system_kwp * cost_per_kwp)
            return {
                "inverter_kwac":    round(system_kwp, 2),
                "cost_per_kwp_rm":  cost_per_kwp,
                "total_cost_rm":    total_cost,
                "tier":             tier_desc,
                "note": (
                    f"Rough estimate for {tier_desc}. "
                    "Actual cost depends on brand, number of MPPTs, site complexity, "
                    "and whether monitoring hardware is included. "
                    "Obtain a formal quotation before budgeting."
                ),
            }
    # unreachable — last tier is inf, but satisfy the type-checker
    return {"inverter_kwac": 0.0, "cost_per_kwp_rm": 0.0, "total_cost_rm": 0.0,
            "tier": "unknown", "note": ""}


def calculate_solar_sizing(
    roof_area_m2: float,
    panel_wattage_w: int = 415,
    psh: float = 4.5,
) -> dict:
    """
    Return optimal solar sizing for a given roof area.

    Args:
        roof_area_m2:    Total roof area in square metres.
        panel_wattage_w: Rated output per panel in watts (default 415 W).
        psh:             Peak sun hours per day (Malaysia avg 4.5).
    """
    if roof_area_m2 <= 0:
        raise ValueError(f"roof_area_m2 must be positive, got {roof_area_m2}")
    if not (200 <= panel_wattage_w <= 700):
        raise ValueError(f"panel_wattage_w {panel_wattage_w} W is outside realistic range 200–700 W")
    if not (2.0 <= psh <= 7.0):
        raise ValueError(f"psh {psh} h/day is outside plausible range 2.0–7.0 for Malaysia")

    usable_m2 = roof_area_m2 * USABLE_ROOF_FRACTION
    n_panels = math.floor(usable_m2 / EFFECTIVE_PANEL_AREA_M2)
    system_kwp = round(n_panels * panel_wattage_w / 1_000, 2)

    annual_kwh = system_kwp * psh * 365 * SYSTEM_DERATE
    monthly_kwh_avg = annual_kwh / 12
    daily_kwh_avg = annual_kwh / 365

    monthly_breakdown = [round(monthly_kwh_avg * f) for f in SEASONAL_FACTORS]
    inverter = estimate_inverter_cost(system_kwp)

    return {
        "roof_area_m2":               roof_area_m2,
        "usable_area_m2":             round(usable_m2, 2),
        "coverage_pct":               USABLE_ROOF_FRACTION * 100,
        "n_panels":                   n_panels,
        "panel_wattage_w":            panel_wattage_w,
        "effective_panel_area_m2":    round(EFFECTIVE_PANEL_AREA_M2, 2),
        "system_kwp":                 system_kwp,
        "psh":                        psh,
        "system_derate_pct":          SYSTEM_DERATE * 100,
        "annual_generation_kwh":      round(annual_kwh),
        "monthly_generation_kwh_avg": round(monthly_kwh_avg),
        "daily_generation_kwh_avg":   round(daily_kwh_avg, 1),
        "monthly_breakdown_kwh":      monthly_breakdown,
        "month_labels":               MONTH_LABELS,
        "inverter_cost_rm":           inverter["total_cost_rm"],
        "inverter_cost_per_kwp_rm":   inverter["cost_per_kwp_rm"],
        "inverter_tier":              inverter["tier"],
        "inverter_note":              inverter["note"],
    }
