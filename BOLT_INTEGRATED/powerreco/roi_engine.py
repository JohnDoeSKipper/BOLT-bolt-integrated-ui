"""
ROI & financial analysis engine for PowerRECO.
25-year DCF model with IRR (Newton-Raphson), NPV at 6% WACC,
battery replacement at year 10.

Pulls MD + energy rates from calculator.rates so billing and payback stay
in sync. Pass `schedule_key='legacy_2014'` to compute against pre-July-2025
rates for sensitivity analysis; default is the post-July-2025 schedule.

Key design decisions
---------------------
- Tax multiplier is tariff-aware: Tariff A (domestic) is exempt from KWTBB
  and service tax; applying TAX_MULTIPLIER to domestic savings over-states
  them by ~7.7 %.
- Battery degradation resets after the year-10 replacement. The original code
  used min(yr, 10) which left the replacement battery starting at ~81.7 %
  capacity instead of 100 %.
- Inverter cost is included in CAPEX. It is priced per kWp via a tiered
  benchmark (see solar_sizing.estimate_inverter_cost) and can be overridden
  by passing inverter_cost_rm directly.
"""
from __future__ import annotations
import math

from calculator.rates import get_schedule

KWTBB = 0.016
SERVICE_TAX = 0.06
# Non-domestic tax multiplier: (1 + KWTBB)(1 + ST) ≈ 1.077
# Domestic (Tariff A): multiplier = 1.0 — neither levy applies.
_TAX_MULTIPLIER_COMMERCIAL = (1 + KWTBB) * (1 + SERVICE_TAX)
_DOMESTIC_TARIFFS = {"A"}

DEFAULT_SOLAR_COST_PER_KWP    = 3_500.0
DEFAULT_BATT_COST_PER_KWH     = 2_500.0
DEFAULT_INVERTER_COST_PER_KWP = 640.0   # mid-tier commercial (10–100 kWp)
DEFAULT_SELF_CONSUMPTION       = 0.65

SOLAR_LIFE_YEARS      = 25
BATTERY_LIFE_YEARS    = 10
BATT_REPLACE_DISCOUNT = 0.20
SOLAR_DEGRADATION     = 0.005
BATT_DEGRADATION      = 0.02
DISCOUNT_RATE         = 0.06

GRID_EMISSION_FACTOR = 0.000585  # tonnes CO₂ per kWh

# Feasibility thresholds — used by assess_feasibility()
FEASIBILITY_PAYBACK_GOOD      = 8.0   # years — clearly viable
FEASIBILITY_PAYBACK_MARGINAL  = 12.0  # years — borderline; sensitive to assumptions
FEASIBILITY_MIN_SOLAR_KWP     = 10.0  # below this, economies of scale are poor
FEASIBILITY_MIN_MD_SAVINGS_RM = 5_000.0  # annual MD savings floor for battery ROI

_VALID_TARIFFS = {"A", "B", "C1", "C2", "D", "E1", "E2"}


def _tax_multiplier(tariff: str) -> float:
    """Returns 1.0 for domestic tariff A (no KWTBB / ST), full multiplier otherwise."""
    return 1.0 if tariff in _DOMESTIC_TARIFFS else _TAX_MULTIPLIER_COMMERCIAL


def calculate_roi(
    solar_kwp: float,
    battery_kwh: float,
    monthly_generation_kwh: float,
    md_reduction_kw: float,
    self_consumption_pct: float = DEFAULT_SELF_CONSUMPTION,
    schedule_key: str | None = None,
    tariff: str = "C1",
    use_new_tariff: bool | None = None,  # deprecated; use schedule_key instead
    solar_cost_per_kwp: float = DEFAULT_SOLAR_COST_PER_KWP,
    battery_cost_per_kwh: float = DEFAULT_BATT_COST_PER_KWH,
    inverter_cost_rm: float | None = None,
    inverter_cost_per_kwp: float = DEFAULT_INVERTER_COST_PER_KWP,
    nem_buyback_rate: float | None = None,
) -> dict:
    """
    Full 25-year financial model for a solar + battery system.

    `schedule_key`: 'new_2025' (default) or 'legacy_2014'. Same key passed
    to calculator.tnb_tariffs.calculate_bill keeps Before/After bill and
    payback projections consistent.

    `tariff`: which TNB tariff this site is on. MD rate is read for that
    tariff from the active schedule. Energy rate likewise — so a Tariff D
    site no longer pays for savings at C1's energy rate.

    `use_new_tariff` is the legacy boolean kept for backwards compatibility
    with existing callers — `True` → new_2025, `False` → legacy_2014.
    Explicit `schedule_key` wins if both are passed.

    `inverter_cost_rm`: total inverter supply-and-install cost in RM.
    If None (default), computed as `solar_kwp * inverter_cost_per_kwp`.
    Pass 0 to exclude inverter from CAPEX (e.g., if already included in
    the solar_cost_per_kwp quote).
    """
    # ---- input guard -------------------------------------------------------
    if tariff not in _VALID_TARIFFS:
        raise ValueError(f"Unknown tariff '{tariff}'. Valid: {sorted(_VALID_TARIFFS)}")
    if solar_kwp < 0:
        raise ValueError(f"solar_kwp must be ≥ 0, got {solar_kwp}")
    if battery_kwh < 0:
        raise ValueError(f"battery_kwh must be ≥ 0, got {battery_kwh}")
    if monthly_generation_kwh < 0:
        raise ValueError(f"monthly_generation_kwh must be ≥ 0")
    if not 0.0 <= self_consumption_pct <= 1.0:
        raise ValueError(f"self_consumption_pct must be 0–1, got {self_consumption_pct}")

    # ---- resolve schedule --------------------------------------------------
    if schedule_key is None and use_new_tariff is not None:
        schedule_key = "new_2025" if use_new_tariff else "legacy_2014"
    sched = get_schedule(schedule_key)

    md_rate     = sched.md_rates.get(tariff, sched.md_rates.get("C1", 0.0))
    energy_rate = sched.energy_rates.get(tariff, sched.energy_rates.get("C1", 0.365))
    nem_rate    = nem_buyback_rate if nem_buyback_rate is not None else sched.nem_default_rate
    tax_mult    = _tax_multiplier(tariff)

    # ---- CAPEX -------------------------------------------------------------
    solar_capex    = solar_kwp * solar_cost_per_kwp
    battery_capex  = battery_kwh * battery_cost_per_kwh
    if inverter_cost_rm is None:
        inverter_capex = solar_kwp * inverter_cost_per_kwp
    else:
        inverter_capex = float(inverter_cost_rm)
    total_capex = solar_capex + battery_capex + inverter_capex

    # ---- Year-1 benefits ---------------------------------------------------
    annual_gen_kwh    = monthly_generation_kwh * 12
    self_consumed_kwh = annual_gen_kwh * self_consumption_pct
    exported_kwh      = annual_gen_kwh * (1.0 - self_consumption_pct)

    annual_energy_save = self_consumed_kwh * energy_rate * tax_mult
    annual_nem_credit  = exported_kwh * nem_rate
    annual_md_save     = md_reduction_kw * md_rate * 12 * tax_mult

    total_annual_benefit_yr1 = annual_energy_save + annual_nem_credit + annual_md_save

    simple_payback = (
        total_capex / total_annual_benefit_yr1
        if total_annual_benefit_yr1 > 0 else float("inf")
    )

    # ---- 25-year DCF -------------------------------------------------------
    npv = -total_capex
    cashflows = []
    cumulative_npv = []

    for yr in range(1, SOLAR_LIFE_YEARS + 1):
        solar_f = (1.0 - SOLAR_DEGRADATION) ** yr

        # Battery degradation resets for the replacement unit at year 10.
        # Original battery: years 1–10 (ages 1→10).
        # Replacement battery: years 11–25 (ages 1→15, no 2nd replacement modelled).
        if yr <= BATTERY_LIFE_YEARS:
            batt_f = (1.0 - BATT_DEGRADATION) ** yr
        else:
            batt_f = (1.0 - BATT_DEGRADATION) ** (yr - BATTERY_LIFE_YEARS)

        yr_benefit = (
            self_consumed_kwh * solar_f * energy_rate * tax_mult
            + exported_kwh    * solar_f * nem_rate
            + annual_md_save  * batt_f
        )

        yr_cost = 0.0
        if yr == BATTERY_LIFE_YEARS and battery_kwh > 0:
            yr_cost = battery_capex * (1.0 - BATT_REPLACE_DISCOUNT)

        net_cf = yr_benefit - yr_cost
        cashflows.append(net_cf)

        pv  = net_cf / ((1.0 + DISCOUNT_RATE) ** yr)
        npv += pv
        cumulative_npv.append(round(npv))

    irr = _irr(total_capex, cashflows)
    co2_offset = annual_gen_kwh * GRID_EMISSION_FACTOR

    return {
        "solar_capex_rm":           round(solar_capex),
        "battery_capex_rm":         round(battery_capex),
        "inverter_capex_rm":        round(inverter_capex),
        "total_capex_rm":           round(total_capex),
        "annual_energy_savings_rm": round(annual_energy_save),
        "annual_nem_credit_rm":     round(annual_nem_credit),
        "annual_md_savings_rm":     round(annual_md_save),
        "total_annual_benefit_rm":  round(total_annual_benefit_yr1),
        "monthly_net_benefit_rm":   round(total_annual_benefit_yr1 / 12),
        "simple_payback_years":     round(simple_payback, 1),
        "npv_25yr_rm":              round(npv),
        "irr_pct":                  round(irr * 100, 1) if irr is not None else None,
        "cumulative_npv":           cumulative_npv,
        "annual_gen_kwh":           round(annual_gen_kwh),
        "self_consumed_kwh_annual": round(self_consumed_kwh),
        "exported_kwh_annual":      round(exported_kwh),
        "co2_offset_tonnes_yr":     round(co2_offset, 1),
        "md_rate_used":             md_rate,
        "energy_rate_used":         energy_rate,
        "nem_rate_used":            nem_rate,
        "tax_multiplier_used":      round(tax_mult, 4),
        "schedule_key":             sched.key,
        "schedule_name":            sched.name,
        "tariff":                   tariff,
    }


def assess_feasibility(
    avg_monthly_kwh: float,
    peak_demand_kw: float,
    tariff: str,
    md_reduction_kw: float = 0.0,
    schedule_key: str | None = None,
    monthly_gen_kwh_per_kwp: float = 162.0,
    solar_cost_per_kwp: float = DEFAULT_SOLAR_COST_PER_KWP,
    battery_cost_per_kwh: float = DEFAULT_BATT_COST_PER_KWH,
    inverter_cost_per_kwp: float = DEFAULT_INVERTER_COST_PER_KWP,
    battery_kwh: float = 0.0,
    payback_threshold_yrs: float = 10.0,
) -> dict:
    """
    Determine whether investing in solar and/or battery is likely worthwhile
    BEFORE committing to a full system design.

    This is a pre-screening gate, not a detailed financial model. It runs
    quick heuristic calculations and flags the system as VIABLE, MARGINAL,
    or NOT_VIABLE for each component independently.

    Parameters
    ----------
    avg_monthly_kwh         Average monthly grid import (kWh).
    peak_demand_kw          Highest recorded demand (kW) — used for MD charge basis.
    tariff                  TNB tariff code (A–E2).
    md_reduction_kw         Achievable MD reduction from Manager optimisation (kW).
                            If 0, battery feasibility is assessed on energy shifting only.
    monthly_gen_kwh_per_kwp Expected monthly solar yield per kWp (kWh/kWp).
                            Default 162 = 4.5 PSH × 0.80 derate × 30 days, Malaysia avg.
    solar_cost_per_kwp      Installed solar panel cost (RM/kWp), excl. inverter.
    battery_cost_per_kwh    Battery cost (RM/kWh).
    inverter_cost_per_kwp   Inverter supply-and-install cost (RM/kWp).
    battery_kwh             Battery size for feasibility estimate; 0 = auto-estimate.
    payback_threshold_yrs   Max acceptable simple payback (years). Default 10.

    Returns
    -------
    dict with keys:
      solar_viable        bool
      battery_viable      bool
      solar_verdict       'VIABLE' | 'MARGINAL' | 'NOT_VIABLE' | 'INSUFFICIENT_DATA'
      battery_verdict     (same)
      solar_reasons       list[str]
      battery_reasons     list[str]
      estimated_solar_kwp float — representative system size used for the estimate
      estimated_payback_solar_yrs  float
      estimated_payback_battery_yrs float
      recommendation      str — plain-English summary
      caveats             list[str]
    """
    if tariff not in _VALID_TARIFFS:
        raise ValueError(f"Unknown tariff '{tariff}'. Valid: {sorted(_VALID_TARIFFS)}")

    sched       = get_schedule(schedule_key)
    md_rate     = sched.md_rates.get(tariff, 0.0)
    energy_rate = sched.energy_rates.get(tariff, sched.energy_rates.get("C1", 0.365))
    nem_rate    = sched.nem_default_rate
    tax_mult    = _tax_multiplier(tariff)
    has_md      = md_rate > 0.0

    solar_reasons   = []
    battery_reasons = []
    caveats         = []

    # ------------------------------------------------------------------ #
    # SOLAR FEASIBILITY                                                    #
    # ------------------------------------------------------------------ #
    # Use a representative system size: size solar to cover ~50% of the
    # average monthly demand. This is conservative and avoids over-sizing
    # for a first-pass check.
    avg_daily_kwh = avg_monthly_kwh / 30.0
    representative_kwp = max(
        FEASIBILITY_MIN_SOLAR_KWP,
        round((avg_daily_kwh * 0.50) / (monthly_gen_kwh_per_kwp / 30.0), 1),
    )

    solar_capex_rep = representative_kwp * (solar_cost_per_kwp + inverter_cost_per_kwp)
    annual_gen_rep  = representative_kwp * monthly_gen_kwh_per_kwp * 12

    # Conservative self-consumption (65%) for heuristic; export gets NEM rate.
    sc_pct = DEFAULT_SELF_CONSUMPTION
    annual_solar_saving = (
        annual_gen_rep * sc_pct * energy_rate * tax_mult
        + annual_gen_rep * (1 - sc_pct) * nem_rate
    )
    est_solar_payback = (
        solar_capex_rep / annual_solar_saving if annual_solar_saving > 0 else float("inf")
    )

    if representative_kwp < FEASIBILITY_MIN_SOLAR_KWP:
        solar_verdict = "INSUFFICIENT_DATA"
        solar_reasons.append(
            f"Estimated system size {representative_kwp:.0f} kWp is below the practical "
            f"minimum of {FEASIBILITY_MIN_SOLAR_KWP:.0f} kWp — economies of scale are poor."
        )
    elif avg_monthly_kwh < 500:
        solar_verdict = "NOT_VIABLE"
        solar_reasons.append(
            f"Average monthly consumption {avg_monthly_kwh:.0f} kWh is too low. "
            "A grid-tie solar system is generally not cost-effective below ~500 kWh/month."
        )
    elif est_solar_payback <= FEASIBILITY_PAYBACK_GOOD:
        solar_verdict = "VIABLE"
        solar_reasons.append(
            f"Estimated simple payback {est_solar_payback:.1f} years ≤ {FEASIBILITY_PAYBACK_GOOD:.0f} years "
            f"(threshold). Strong case for solar."
        )
    elif est_solar_payback <= FEASIBILITY_PAYBACK_MARGINAL:
        solar_verdict = "MARGINAL"
        solar_reasons.append(
            f"Estimated simple payback {est_solar_payback:.1f} years is borderline "
            f"({FEASIBILITY_PAYBACK_GOOD:.0f}–{FEASIBILITY_PAYBACK_MARGINAL:.0f} years). "
            "Viable but sensitive to tariff escalation and self-consumption rate."
        )
    elif est_solar_payback <= payback_threshold_yrs:
        solar_verdict = "MARGINAL"
        solar_reasons.append(
            f"Estimated simple payback {est_solar_payback:.1f} years is within user "
            f"threshold ({payback_threshold_yrs:.0f} years) but above the preferred "
            f"{FEASIBILITY_PAYBACK_MARGINAL:.0f}-year mark."
        )
    else:
        solar_verdict = "NOT_VIABLE"
        solar_reasons.append(
            f"Estimated simple payback {est_solar_payback:.1f} years exceeds the "
            f"{payback_threshold_yrs:.0f}-year threshold. Solar is unlikely to be "
            "economically attractive at current costs and tariff."
        )

    if tariff == "A":
        solar_reasons.append(
            "Tariff A (domestic): NEM rate is RM0.31/kWh. Residential solar can be "
            "viable but the analysis here assumes a commercial installation context."
        )
        caveats.append("Domestic solar schemes have separate SEDA NEM quotas and approval processes.")

    # ------------------------------------------------------------------ #
    # BATTERY FEASIBILITY                                                  #
    # ------------------------------------------------------------------ #
    if not has_md:
        battery_verdict = "NOT_VIABLE"
        battery_reasons.append(
            f"Tariff {tariff} has no maximum-demand (MD) charge. "
            "Battery ROI depends almost entirely on MD savings — "
            "without MD charges, the investment rarely pays back within 15 years."
        )
    else:
        # Estimate minimum viable battery size if not provided.
        bat_kwh = battery_kwh if battery_kwh > 0 else max(50.0, peak_demand_kw * 0.5)
        battery_capex_est = bat_kwh * battery_cost_per_kwh
        annual_md_saving_est = md_reduction_kw * md_rate * 12 * tax_mult if md_reduction_kw > 0 else 0.0

        if annual_md_saving_est <= 0:
            # Estimate MD reduction as 10% of peak demand (very conservative)
            est_md_red = peak_demand_kw * 0.10
            annual_md_saving_est = est_md_red * md_rate * 12 * tax_mult
            battery_reasons.append(
                f"No Manager MD-reduction result available. Estimated {est_md_red:.0f} kW "
                "reduction (10% of peak demand) used for heuristic — run Manager first for accuracy."
            )

        est_batt_payback = (
            battery_capex_est / annual_md_saving_est
            if annual_md_saving_est > 0 else float("inf")
        )

        if annual_md_saving_est < FEASIBILITY_MIN_MD_SAVINGS_RM:
            battery_verdict = "NOT_VIABLE"
            battery_reasons.append(
                f"Estimated annual MD savings RM{annual_md_saving_est:,.0f} is below the "
                f"minimum viable threshold of RM{FEASIBILITY_MIN_MD_SAVINGS_RM:,.0f}/year. "
                "The battery investment is unlikely to recover within system lifetime."
            )
        elif est_batt_payback <= FEASIBILITY_PAYBACK_GOOD:
            battery_verdict = "VIABLE"
            battery_reasons.append(
                f"Estimated battery simple payback {est_batt_payback:.1f} years "
                f"(RM{annual_md_saving_est:,.0f}/year MD savings on {bat_kwh:.0f} kWh, "
                f"MD rate RM{md_rate:.2f}/kW). Strong ROI case."
            )
        elif est_batt_payback <= payback_threshold_yrs:
            battery_verdict = "MARGINAL"
            battery_reasons.append(
                f"Estimated battery payback {est_batt_payback:.1f} years — within "
                f"threshold but sensitive to actual MD reduction and tariff changes."
            )
        else:
            battery_verdict = "NOT_VIABLE"
            battery_reasons.append(
                f"Estimated battery payback {est_batt_payback:.1f} years exceeds "
                f"the {payback_threshold_yrs:.0f}-year threshold. "
                "Battery alone is not financially attractive at current parameters."
            )

        caveats.append(
            f"MD rate used: RM{md_rate:.2f}/kW ({sched.name}). "
            "Verify your latest TNB bill to confirm the active schedule."
        )

    # ------------------------------------------------------------------ #
    # COMBINED RECOMMENDATION                                              #
    # ------------------------------------------------------------------ #
    solar_viable   = solar_verdict in ("VIABLE", "MARGINAL")
    battery_viable = battery_verdict in ("VIABLE", "MARGINAL")

    if solar_verdict == "VIABLE" and battery_verdict == "VIABLE":
        recommendation = (
            "Both solar and battery investments appear financially viable. "
            "Proceed to full sizing and ROI analysis."
        )
    elif solar_verdict == "VIABLE" and battery_verdict in ("MARGINAL", "NOT_VIABLE"):
        recommendation = (
            "Solar investment is viable. Battery-only ROI is "
            f"{'marginal' if battery_verdict == 'MARGINAL' else 'not viable'}. "
            "Consider solar first; add battery if MD reduction target justifies cost."
        )
    elif solar_verdict == "MARGINAL" and battery_verdict == "VIABLE":
        recommendation = (
            "Battery investment is clearly viable. Solar is marginal — "
            "acceptable if payback sensitivity to tariff escalation is tolerable."
        )
    elif solar_viable or battery_viable:
        recommendation = (
            "At least one component shows marginal viability. "
            "Conduct a detailed analysis and sensitivity test before committing capital."
        )
    else:
        recommendation = (
            "Neither solar nor battery investment appears financially viable under "
            "current costs, tariff rates, and load profile. "
            "Revisit if tariff escalation, capital cost reductions, or MD-intensive "
            "operations change the economics."
        )

    caveats.append(
        "All figures are rough heuristics for pre-screening only. "
        "Run full sizing + ROI analysis for investment-grade numbers."
    )

    return {
        "solar_viable":                   solar_viable,
        "battery_viable":                 battery_viable,
        "solar_verdict":                  solar_verdict,
        "battery_verdict":                battery_verdict,
        "solar_reasons":                  solar_reasons,
        "battery_reasons":                battery_reasons,
        "estimated_solar_kwp":            round(representative_kwp, 1),
        "estimated_payback_solar_yrs":    round(est_solar_payback, 1) if math.isfinite(est_solar_payback) else None,
        "estimated_payback_battery_yrs":  round(est_batt_payback, 1) if (has_md and math.isfinite(est_batt_payback)) else None,
        "annual_solar_savings_est_rm":    round(annual_solar_saving),
        "annual_md_savings_est_rm":       round(annual_md_saving_est) if has_md else 0,
        "md_rate_rm_per_kw":              md_rate,
        "energy_rate_rm_per_kwh":         energy_rate,
        "recommendation":                 recommendation,
        "caveats":                        caveats,
        "schedule_key":                   sched.key,
        "schedule_name":                  sched.name,
        "tariff":                         tariff,
        "payback_threshold_yrs":          payback_threshold_yrs,
    }


def _irr(capex: float, cashflows: list[float]) -> float | None:
    """Newton-Raphson IRR. Returns None if no solution found in (0%, 200%)."""
    r = 0.10
    for _ in range(200):
        npv  = -capex + sum(cf / (1 + r) ** (i + 1) for i, cf in enumerate(cashflows))
        dnpv = sum(-(i + 1) * cf / (1 + r) ** (i + 2) for i, cf in enumerate(cashflows))
        if abs(dnpv) < 1e-10:
            break
        r_new = r - npv / dnpv
        if abs(r_new - r) < 1e-8:
            r = r_new
            break
        r = max(-0.99, min(r_new, 9.99))
    return r if 0.0 < r < 2.0 else None
