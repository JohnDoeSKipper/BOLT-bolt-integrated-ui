"""
Sizing optimizer — sweeps a (solar_kwp × battery_kwh) grid and finds the
configuration with the highest 25-yr NPV (or the lowest payback, or the
best NPV under a budget cap).

Why this exists:
    The base PowerRECO sizing path takes one solar size (from roof area)
    and one battery size (from Manager output) and computes ROI for that
    single config. The spec calls for "optimum battery + solar capacity
    (best ROI)" — a search, not a single eval.

Approach:
    1. Build a small grid: ~5 solar sizes × ~5 battery sizes = 25 configs.
       Configurable, but 25 keeps the sweep fast (<2 s).
    2. For each config:
         - Recompute solar generation (linear in kWp at fixed PSH).
         - Estimate the MD reduction the Manager would achieve with that
           battery_kwh, by interpolating from the actually-run Manager
           output. (Re-running the full Manager per battery size is ~10x
           slower; for the demo this interpolation is good enough — the
           docstring on `estimate_md_reduction_for_battery` documents the
           approximation.)
         - Call calculate_roi with the active tariff schedule.
    3. Return the full grid + the best config under each criterion.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional

from powerreco.roi_engine   import calculate_roi
from powerreco.solar_sizing import calculate_solar_sizing


DEFAULT_SOLAR_GRID_KWP   = [50, 100, 200, 400, 800]
DEFAULT_BATTERY_GRID_KWH = [50, 100, 200, 400, 600]


@dataclass
class SizingPoint:
    solar_kwp:    float
    battery_kwh:  float
    npv_25yr_rm:  float
    payback_yrs:  float
    irr_pct:      Optional[float]
    total_capex_rm:           float
    annual_md_savings_rm:     float
    annual_energy_savings_rm: float
    annual_nem_credit_rm:     float
    monthly_net_benefit_rm:   float
    feasible_under_budget:    bool


def estimate_md_reduction_for_battery(
    measured_md_reduction_kw: float,
    measured_battery_kwh: float,
    target_battery_kwh: float,
) -> float:
    """Estimate MD reduction the Manager would deliver at `target_battery_kwh`,
    using a single Manager-output measurement as the anchor.

    Approximation: MD reduction scales roughly with min(target/measured, 1)
    until the battery is "large enough" to absorb the daily peak, then
    saturates. We model that with a soft-cap:

        ratio = target / measured
        if ratio <= 1: returns measured * ratio
        else:          returns measured * (1 + 0.4 * log2(ratio))

    The diminishing-returns coefficient (0.4) is tuned to typical Manager
    behaviour on the case-study sites — larger batteries help further
    flatten the peak but with sharply diminishing returns past 2× the
    minimum size. The user can re-run the Manager per battery_kwh if they
    want exact numbers; this approximation is documented in the UI.
    """
    if measured_battery_kwh <= 0 or measured_md_reduction_kw <= 0:
        # Fall back to a crude linear: 1 kW of MD reduction per 5 kWh of battery
        return max(0.0, target_battery_kwh / 5.0)
    ratio = target_battery_kwh / measured_battery_kwh
    if ratio <= 1.0:
        return measured_md_reduction_kw * ratio
    import math
    return measured_md_reduction_kw * (1.0 + 0.4 * math.log2(ratio))


def sweep_sizing_grid(
    monthly_gen_kwh_per_kwp: float,
    measured_md_reduction_kw: float,
    measured_battery_kwh: float,
    solar_grid_kwp:   list[float] | None = None,
    battery_grid_kwh: list[float] | None = None,
    self_consumption_pct: float = 0.65,
    schedule_key: str | None = None,
    tariff: str = "C1",
    solar_cost_per_kwp: float = 3500.0,
    battery_cost_per_kwh: float = 2500.0,
    budget_rm: float | None = None,
) -> dict:
    """
    Sweep the (solar, battery) grid. Returns:

        {
          'grid_solar_kwp':   [...],
          'grid_battery_kwh': [...],
          'points':           list[SizingPoint]   — one per (s, b),
          'npv_matrix':       2D list  [solar_idx][battery_idx] → npv,
          'payback_matrix':   same shape  → payback yrs,
          'best_npv':         SizingPoint    — global best NPV (any cost),
          'best_payback':     SizingPoint    — global best payback,
          'best_under_budget': SizingPoint  | None    — best NPV with capex ≤ budget,
        }

    `monthly_gen_kwh_per_kwp` lets the caller pre-compute generation per kWp
    once (depends only on PSH + system_derate + seasonal factors averaged
    over the year). Solar generation at `solar_kwp` = `solar_kwp *
    monthly_gen_kwh_per_kwp`.
    """
    solar_grid   = list(solar_grid_kwp)   if solar_grid_kwp   else DEFAULT_SOLAR_GRID_KWP
    battery_grid = list(battery_grid_kwh) if battery_grid_kwh else DEFAULT_BATTERY_GRID_KWH

    npv_matrix:     list[list[float]] = []
    payback_matrix: list[list[float]] = []
    points:         list[SizingPoint] = []

    best_npv:     SizingPoint | None = None
    best_pb:      SizingPoint | None = None
    best_budget:  SizingPoint | None = None

    for s_kwp in solar_grid:
        row_npv:     list[float] = []
        row_payback: list[float] = []
        for b_kwh in battery_grid:
            md_red = estimate_md_reduction_for_battery(
                measured_md_reduction_kw, measured_battery_kwh, b_kwh,
            )
            roi = calculate_roi(
                solar_kwp=s_kwp,
                battery_kwh=b_kwh,
                monthly_generation_kwh=s_kwp * monthly_gen_kwh_per_kwp,
                md_reduction_kw=md_red,
                self_consumption_pct=self_consumption_pct,
                schedule_key=schedule_key,
                tariff=tariff,
                solar_cost_per_kwp=solar_cost_per_kwp,
                battery_cost_per_kwh=battery_cost_per_kwh,
            )
            capex = float(roi['total_capex_rm'])
            under_budget = budget_rm is None or capex <= budget_rm
            pt = SizingPoint(
                solar_kwp=float(s_kwp), battery_kwh=float(b_kwh),
                npv_25yr_rm=float(roi['npv_25yr_rm']),
                payback_yrs=float(roi['simple_payback_years']),
                irr_pct=roi.get('irr_pct'),
                total_capex_rm=capex,
                annual_md_savings_rm=float(roi['annual_md_savings_rm']),
                annual_energy_savings_rm=float(roi['annual_energy_savings_rm']),
                annual_nem_credit_rm=float(roi['annual_nem_credit_rm']),
                monthly_net_benefit_rm=float(roi['monthly_net_benefit_rm']),
                feasible_under_budget=under_budget,
            )
            points.append(pt)
            row_npv.append(pt.npv_25yr_rm)
            row_payback.append(pt.payback_yrs)

            if best_npv is None or pt.npv_25yr_rm > best_npv.npv_25yr_rm:
                best_npv = pt
            if pt.payback_yrs > 0 and (best_pb is None or pt.payback_yrs < best_pb.payback_yrs):
                best_pb = pt
            if under_budget and (
                best_budget is None or pt.npv_25yr_rm > best_budget.npv_25yr_rm
            ):
                best_budget = pt

        npv_matrix.append(row_npv)
        payback_matrix.append(row_payback)

    return {
        'grid_solar_kwp':   solar_grid,
        'grid_battery_kwh': battery_grid,
        'points':           [_pt_to_dict(p) for p in points],
        'npv_matrix':       npv_matrix,
        'payback_matrix':   payback_matrix,
        'best_npv':         _pt_to_dict(best_npv)    if best_npv    else None,
        'best_payback':     _pt_to_dict(best_pb)     if best_pb     else None,
        'best_under_budget':_pt_to_dict(best_budget) if best_budget else None,
        'budget_rm':        budget_rm,
        'schedule_key':     schedule_key,
        'tariff':           tariff,
    }


def _pt_to_dict(p: SizingPoint) -> dict:
    return {
        'solar_kwp':                p.solar_kwp,
        'battery_kwh':              p.battery_kwh,
        'npv_25yr_rm':              round(p.npv_25yr_rm),
        'payback_yrs':              round(p.payback_yrs, 1),
        'irr_pct':                  p.irr_pct,
        'total_capex_rm':           round(p.total_capex_rm),
        'annual_md_savings_rm':     round(p.annual_md_savings_rm),
        'annual_energy_savings_rm': round(p.annual_energy_savings_rm),
        'annual_nem_credit_rm':     round(p.annual_nem_credit_rm),
        'monthly_net_benefit_rm':   round(p.monthly_net_benefit_rm),
        'feasible_under_budget':    p.feasible_under_budget,
    }


def find_optimum_from_existing_run(
    solar_sizing: dict,
    battery_sizing: dict,
    schedule_key: str | None = None,
    tariff: str = "C1",
    self_consumption_pct: float = 0.65,
    solar_cost_per_kwp: float = 3500.0,
    battery_cost_per_kwh: float = 2500.0,
    budget_rm: float | None = None,
    solar_grid_kwp: list[float] | None = None,
    battery_grid_kwh: list[float] | None = None,
) -> dict:
    """Convenience: extracts the measured MD reduction + per-kWp generation
    from existing PowerRECO outputs and calls sweep_sizing_grid()."""
    # solar_sizing comes from calculate_solar_sizing — it has monthly avg.
    sys_kwp = float(solar_sizing.get('system_kwp', 0) or 0)
    monthly = float(solar_sizing.get('monthly_generation_kwh_avg', 0) or 0)
    monthly_per_kwp = monthly / sys_kwp if sys_kwp > 0 else 0.0

    measured_md = float(battery_sizing.get('md_reduction_kw', 0) or 0)
    measured_kwh = float(battery_sizing.get('min_capacity_kwh_commercial', 0) or 0)

    return sweep_sizing_grid(
        monthly_gen_kwh_per_kwp=monthly_per_kwp,
        measured_md_reduction_kw=measured_md,
        measured_battery_kwh=measured_kwh,
        solar_grid_kwp=solar_grid_kwp,
        battery_grid_kwh=battery_grid_kwh,
        self_consumption_pct=self_consumption_pct,
        schedule_key=schedule_key,
        tariff=tariff,
        solar_cost_per_kwp=solar_cost_per_kwp,
        battery_cost_per_kwh=battery_cost_per_kwh,
        budget_rm=budget_rm,
    )
