"""
Energy flow decomposition — turns a Manager run + a solar capacity into
the five annual energy flows the Sankey diagram and the sustainability
KPIs need:

    grid    → load
    solar   → load   (direct self-consumption)
    solar   → battery
    battery → load
    solar   → grid   (exported / NEM)

Today PowerRECO's `self_consumption_pct` is a single-slider INPUT. With
the Manager's interval-by-interval output and a known solar capacity we
can derive these flows directly, which lets us:

  • Replace the slider with a computed self-consumption ratio.
  • Compute grid-independence %.
  • Draw an annual Sankey.
  • Plot a stacked solar-vs-consumption curve for a representative day.

Per-interval algorithm (each row is 30 min, INTERVAL_H = 0.5 h):

    solar_gen      = est_solar_gen_kw(ts, capacity_kwp) * INTERVAL_H
    load           = kw_managed * INTERVAL_H               # net post-Manager
    batt_charge    = battery_charge_kw * INTERVAL_H        # always ≥ 0
    batt_discharge = battery_discharge_kw * INTERVAL_H     # always ≥ 0

    # Solar is consumed in this priority order:
    #   1. Direct to load (up to the load's instantaneous demand)
    #   2. To battery (whatever's left, up to the charge happening this interval)
    #   3. To grid (whatever's left after both)
    solar_to_load    = min(solar_gen, load)
    solar_to_battery = min(solar_gen - solar_to_load, batt_charge)
    solar_to_grid    = max(solar_gen - solar_to_load - solar_to_battery, 0)

    # Battery discharge always goes to load by Manager construction.
    battery_to_load = batt_discharge

    # Remaining load is covered from grid.
    grid_to_load    = max(load - solar_to_load - battery_to_load, 0)

Note: this is an attribution model, not a power-flow simulation. The
Manager doesn't separately track where each kW came from; we infer it
from the Merit-Order rule above (solar self-consumed first, then stored,
then exported), which is how real net-metered Malaysian solar systems
behave. The model is accurate to within rounding for the demo.
"""
from __future__ import annotations
import math
import pandas as pd

from predictor.features import estimated_solar_gen_kw

INTERVAL_H = 0.5  # 30-minute intervals


def decompose_flows(
    manager_results: list[dict],
    capacity_kwp: float = 0.0,
) -> dict:
    """Return per-interval flow breakdown + period totals.

    `manager_results` is the list of dicts returned by `run_ai_manager` —
    each row has `timestamp`, `kw_managed`, `battery_charge_kw`,
    `battery_discharge_kw`. If `capacity_kwp` is 0, all solar flows are
    zero and only grid/battery flows are populated.

    Returned dict shape:
      {
        'df':       pd.DataFrame[timestamp, solar_gen_kwh, load_kwh,
                                 solar_to_load, solar_to_battery,
                                 solar_to_grid, battery_to_load,
                                 grid_to_load],
        'totals':   {flow → kWh},
        'annual':   {flow → kWh, scaled to 365 days based on run length},
        'metrics':  {
            'self_consumption_pct':  derived (0–1),
            'grid_independence_pct': derived (0–1),
            'solar_share_of_load_pct': derived,
            'battery_throughput_kwh':
            'period_days':
        },
      }
    """
    if not manager_results:
        return _empty_result()

    rows = []
    for r in manager_results:
        ts = pd.to_datetime(r["timestamp"])
        load_kwh   = float(r.get("kw_managed", 0.0))    * INTERVAL_H
        bc_kwh     = float(r.get("battery_charge_kw", 0.0))    * INTERVAL_H
        bd_kwh     = float(r.get("battery_discharge_kw", 0.0)) * INTERVAL_H
        rows.append({
            "timestamp":       ts,
            "load_kwh":        max(load_kwh, 0.0),
            "batt_charge_kwh": max(bc_kwh, 0.0),
            "batt_discharge_kwh": max(bd_kwh, 0.0),
        })
    df = pd.DataFrame(rows).sort_values("timestamp").reset_index(drop=True)

    # Solar generation curve at each timestamp
    if capacity_kwp > 0:
        sg_kw = estimated_solar_gen_kw(df["timestamp"], capacity_kwp)
        df["solar_gen_kwh"] = sg_kw.values * INTERVAL_H
    else:
        df["solar_gen_kwh"] = 0.0

    # Per-interval attribution (vectorised — same as the docstring algo)
    sg   = df["solar_gen_kwh"].clip(lower=0)
    load = df["load_kwh"].clip(lower=0)
    bc   = df["batt_charge_kwh"].clip(lower=0)
    bd   = df["batt_discharge_kwh"].clip(lower=0)

    s_to_load    = pd.concat([sg, load], axis=1).min(axis=1)
    s_left_after_load = (sg - s_to_load).clip(lower=0)
    s_to_battery = pd.concat([s_left_after_load, bc], axis=1).min(axis=1)
    s_to_grid    = (sg - s_to_load - s_to_battery).clip(lower=0)

    # Battery → load by Manager construction
    b_to_load = bd

    grid_to_load = (load - s_to_load - b_to_load).clip(lower=0)

    df["solar_to_load"]    = s_to_load
    df["solar_to_battery"] = s_to_battery
    df["solar_to_grid"]    = s_to_grid
    df["battery_to_load"]  = b_to_load
    df["grid_to_load"]     = grid_to_load

    # Period totals
    tot_solar_gen   = float(df["solar_gen_kwh"].sum())
    tot_load        = float(df["load_kwh"].sum())
    tot_s_load      = float(df["solar_to_load"].sum())
    tot_s_batt      = float(df["solar_to_battery"].sum())
    tot_s_grid      = float(df["solar_to_grid"].sum())
    tot_b_load      = float(df["battery_to_load"].sum())
    tot_g_load      = float(df["grid_to_load"].sum())

    totals = {
        "solar_gen_kwh":      round(tot_solar_gen, 1),
        "load_kwh":           round(tot_load, 1),
        "solar_to_load":      round(tot_s_load, 1),
        "solar_to_battery":   round(tot_s_batt, 1),
        "solar_to_grid":      round(tot_s_grid, 1),
        "battery_to_load":    round(tot_b_load, 1),
        "grid_to_load":       round(tot_g_load, 1),
    }

    # Annualise — scale by (365 / period_days). Period covers any whole or
    # partial month-block; we use the actual hours covered.
    n_intervals = len(df)
    period_hours = n_intervals * INTERVAL_H
    period_days  = period_hours / 24.0 if period_hours > 0 else 1.0
    scale = 365.0 / period_days if period_days > 0 else 1.0
    annual = {k: round(v * scale, 1) for k, v in totals.items()
              if k not in ("load_kwh", "solar_gen_kwh")}
    annual["load_kwh"]      = round(tot_load * scale, 1)
    annual["solar_gen_kwh"] = round(tot_solar_gen * scale, 1)

    # Metrics
    self_cons_pct = (
        (tot_s_load + tot_s_batt) / tot_solar_gen if tot_solar_gen > 0 else 0.0
    )
    grid_indep_pct = 1.0 - (tot_g_load / tot_load) if tot_load > 0 else 0.0
    solar_share_pct = (
        (tot_s_load + tot_b_load) / tot_load if tot_load > 0 else 0.0
    )
    batt_throughput = float(df["batt_charge_kwh"].sum() + df["batt_discharge_kwh"].sum()) / 2

    return {
        "df":      df,
        "totals":  totals,
        "annual":  annual,
        "metrics": {
            "self_consumption_pct":     round(self_cons_pct,     4),
            "grid_independence_pct":    round(grid_indep_pct,    4),
            "solar_share_of_load_pct":  round(solar_share_pct,   4),
            "battery_throughput_kwh":   round(batt_throughput,   1),
            "period_days":              round(period_days,       2),
            "capacity_kwp":             capacity_kwp,
        },
    }


def _empty_result() -> dict:
    return {
        "df": pd.DataFrame(columns=[
            "timestamp", "solar_gen_kwh", "load_kwh",
            "solar_to_load", "solar_to_battery", "solar_to_grid",
            "battery_to_load", "grid_to_load",
        ]),
        "totals": {},
        "annual": {},
        "metrics": {
            "self_consumption_pct": 0.0,
            "grid_independence_pct": 0.0,
            "solar_share_of_load_pct": 0.0,
            "battery_throughput_kwh": 0.0,
            "period_days": 0,
            "capacity_kwp": 0,
        },
    }


def representative_day(df: pd.DataFrame) -> pd.DataFrame:
    """Pick a single representative day to plot the stacked area chart.

    Strategy: the day closest to the median total daily load. Avoids
    cherry-picking the peak or the lull. Falls back to the first full
    day if median can't be computed.
    """
    if df.empty:
        return df
    df = df.copy()
    df["date"] = df["timestamp"].dt.date
    daily = df.groupby("date")["load_kwh"].sum()
    if len(daily) == 0:
        return df.iloc[:48]
    median_load = daily.median()
    chosen_date = (daily - median_load).abs().idxmin()
    sub = df[df["date"] == chosen_date].sort_values("timestamp").reset_index(drop=True)
    return sub.drop(columns=["date"])
