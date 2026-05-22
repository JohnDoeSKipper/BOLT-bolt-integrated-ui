"""
TNB Tariff Engine – Peninsular Malaysia.

Rate schedules live in calculator/rates.py — both this module and the ROI
engine read from there, so a rate change updates billing AND payback
projections consistently. The default schedule is the July 2025 revision
('new_2025'); pass `schedule_key='legacy_2014'` to compute against the
pre-July rates for sensitivity analysis.
"""
from __future__ import annotations
import pandas as pd
import numpy as np
from typing import Literal

from calculator.rates import (
    TariffSchedule, SCHEDULES, DEFAULT_SCHEDULE_KEY, get_schedule,
)

TariffCode = Literal["A", "B", "C1", "C2", "D", "E1", "E2"]
ScheduleKey = Literal["legacy_2014", "new_2025"]


# =========================================================================
# Tariff A is tiered — block structure is universal across schedules so it
# lives here, not in rates.py. If a future TNB revision re-tiers Tariff A,
# move this into rates.TariffSchedule as a dict[tariff -> block list].
# =========================================================================
TARIFF_A_BLOCKS: list[tuple] = [
    (200,  0.218, "First 200 kWh"),
    (100,  0.334, "Next 100 kWh  (201 – 300)"),
    (300,  0.516, "Next 300 kWh  (301 – 600)"),
    (300,  0.546, "Next 300 kWh  (601 – 900)"),
    (None, 0.571, "Remaining kWh (> 900)"),
]


# =========================================================================
# Deprecated module-level constants — kept for backwards compatibility with
# any external script that imported them directly. New code should call
# calculator.rates.get_md_rate() / get_energy_rate() / get_min_charge().
# These reflect the LEGACY (pre-July-2025) schedule for historical reasons.
# =========================================================================
_LEGACY = SCHEDULES["legacy_2014"]
TARIFF_A_MIN     = _LEGACY.min_charges["A"]
TARIFF_B_ENERGY  = _LEGACY.energy_rates["B"]
TARIFF_B_MIN     = _LEGACY.min_charges["B"]
TARIFF_C1_ENERGY = _LEGACY.energy_rates["C1"]
TARIFF_C1_MD     = _LEGACY.md_rates["C1"]
TARIFF_C1_MIN    = _LEGACY.min_charges["C1"]
TARIFF_C2_PEAK    = _LEGACY.energy_rates["C2"]
TARIFF_C2_OFFPEAK = _LEGACY.offpeak_rates["C2"]
TARIFF_C2_MD      = _LEGACY.md_rates["C2"]
TARIFF_C2_MIN     = _LEGACY.min_charges["C2"]
TARIFF_D_ENERGY = _LEGACY.energy_rates["D"]
TARIFF_D_MD     = _LEGACY.md_rates["D"]
TARIFF_D_MIN    = _LEGACY.min_charges["D"]
TARIFF_E1_PEAK    = _LEGACY.energy_rates["E1"]
TARIFF_E1_OFFPEAK = _LEGACY.offpeak_rates["E1"]
TARIFF_E1_MD      = _LEGACY.md_rates["E1"]
TARIFF_E1_MIN     = _LEGACY.min_charges["E1"]
TARIFF_E2_PEAK    = _LEGACY.energy_rates["E2"]
TARIFF_E2_OFFPEAK = _LEGACY.offpeak_rates["E2"]
TARIFF_E2_MD      = _LEGACY.md_rates["E2"]
TARIFF_E2_MIN     = _LEGACY.min_charges["E2"]

SERVICE_TAX_RATE = _LEGACY.service_tax_rate
KWTBB_RATE       = _LEGACY.kwtbb_rate
NEM_DEFAULT_RATE = _LEGACY.nem_default_rate


# =========================================================================
# Tariff metadata — descriptions for the UI dropdown.
# =========================================================================
TARIFF_META: dict[str, dict] = {
    "A":  {
        "name": "Tariff A – Domestic",
        "voltage": "Low Voltage (240 V / 415 V)",
        "category": "Residential",
        "tou": False, "has_md": False,
        "typical_demand": "< 12 kW",
        "description": "Single- or three-phase residential supply.",
    },
    "B":  {
        "name": "Tariff B – Low Voltage Commercial",
        "voltage": "Low Voltage (240 V / 415 V)",
        "category": "Commercial (small)",
        "tou": False, "has_md": False,
        "typical_demand": "12 – 25 kW",
        "description": "Shops, small offices, clinics on LV supply.",
    },
    "C1": {
        "name": "Tariff C1 – Medium Voltage General",
        "voltage": "Medium Voltage (6.6 kV / 11 kV / 22 kV)",
        "category": "Industrial / Commercial (medium)",
        "tou": False, "has_md": True,
        "typical_demand": "25 – 999 kW",
        "description": "Factories, malls, hotels with MV metered supply.",
    },
    "C2": {
        "name": "Tariff C2 – Medium Voltage Time-of-Use",
        "voltage": "Medium Voltage (6.6 kV / 11 kV / 22 kV)",
        "category": "Industrial / Commercial (medium, TOU)",
        "tou": True, "has_md": True,
        "typical_demand": "25 – 999 kW",
        "description": "Same as C1 but with peak/off-peak pricing.",
    },
    "D":  {
        "name": "Tariff D – High Voltage General",
        "voltage": "High Voltage (33 kV / 132 kV)",
        "category": "Industrial (large)",
        "tou": False, "has_md": True,
        "typical_demand": "1,000 – 4,999 kW",
        "description": "Large factories and industrial parks on HV supply.",
    },
    "E1": {
        "name": "Tariff E1 – High Voltage Peak / Off-Peak",
        "voltage": "High Voltage (33 kV / 132 kV)",
        "category": "Industrial (large, TOU)",
        "tou": True, "has_md": True,
        "typical_demand": "≥ 1,000 kW",
        "description": "Large HV industrial supply with TOU pricing.",
    },
    "E2": {
        "name": "Tariff E2 – High Voltage TOU (MD ≥ 1,500 kW)",
        "voltage": "High Voltage (33 kV / 132 kV)",
        "category": "Industrial (extra-large, TOU)",
        "tou": True, "has_md": True,
        "typical_demand": "≥ 1,500 kW",
        "description": "Extra-large HV industrial supply, mandatory TOU.",
    },
}


def is_peak_hour(ts: pd.Timestamp, schedule_key: str | None = None) -> bool:
    """Peak window per the active schedule. Default = 08:00-22:00 Mon-Sat."""
    sched = get_schedule(schedule_key)
    if ts.dayofweek == 6:  # Sunday
        return False
    if ts.dayofweek == 5 and not sched.saturday_is_peak:  # Saturday excluded?
        return False
    return sched.peak_start_hour <= ts.hour < sched.peak_end_hour


def get_md_rate(tariff: TariffCode, schedule_key: str | None = None) -> float:
    return get_schedule(schedule_key).md_rates.get(tariff, 0.0)


def get_min_charge(tariff: TariffCode, schedule_key: str | None = None) -> float:
    return get_schedule(schedule_key).min_charges.get(tariff, 0.0)


def calculate_bill(
    tariff: TariffCode,
    monthly_kwh: float,
    peak_kwh: float = 0.0,
    offpeak_kwh: float = 0.0,
    max_demand_kw: float = 0.0,
    icpt_sen_per_kwh: float = 0.0,
    apply_service_tax: bool = True,
    apply_kwtbb: bool = True,
    schedule_key: str | None = None,
) -> dict:
    """
    Calculate a full TNB bill for one billing month.

    Returns every component and the total. NEM credit is NOT included —
    deduct separately with compute_nem_credit().

    `schedule_key`: 'new_2025' (default) or 'legacy_2014'. Same parameter
    accepted by powerreco.roi_engine.calculate_roi(), so passing the same
    key to both keeps Before/After bill consistent with payback projections.
    """
    sched = get_schedule(schedule_key)
    is_domestic = tariff == "A"

    result: dict = {
        "tariff": tariff,
        "tariff_name": TARIFF_META[tariff]["name"],
        "schedule_key":  sched.key,
        "schedule_name": sched.name,
        "total_kwh": monthly_kwh,
        "peak_kwh": peak_kwh,
        "offpeak_kwh": offpeak_kwh,
        "max_demand_kw": max_demand_kw,
        "energy_charge": 0.0,
        "md_charge": 0.0,
        "icpt_charge": 0.0,
        "kwtbb_charge": 0.0,
        "service_tax": 0.0,
        "subtotal_pretax": 0.0,
        "minimum_charge_applied": False,
        "total_bill": 0.0,
        "blocks": [],
        "notes": [],
    }

    if tariff == "A":
        remaining, ec = monthly_kwh, 0.0
        for limit, rate, label in TARIFF_A_BLOCKS:
            if remaining <= 0:
                break
            used = remaining if limit is None else min(remaining, limit)
            cost = used * rate
            result["blocks"].append({
                "Block": label,
                "kWh": round(used, 3),
                "Rate (RM/kWh)": rate,
                "Charge (RM)": round(cost, 2),
            })
            ec += cost
            remaining -= used
        result["energy_charge"] = round(ec, 2)

    elif tariff == "B":
        rate = sched.energy_rates["B"]
        ec = round(monthly_kwh * rate, 2)
        result["energy_charge"] = ec
        result["blocks"] = [{"Block": "All kWh (flat rate)",
                              "kWh": monthly_kwh, "Rate (RM/kWh)": rate, "Charge (RM)": ec}]

    elif tariff == "C1":
        rate = sched.energy_rates["C1"]
        md   = sched.md_rates["C1"]
        ec = round(monthly_kwh * rate, 2)
        result["energy_charge"] = ec
        result["md_charge"]     = round(max_demand_kw * md, 2)
        result["blocks"] = [{"Block": "All kWh (flat rate)",
                              "kWh": monthly_kwh, "Rate (RM/kWh)": rate, "Charge (RM)": ec}]

    elif tariff == "C2":
        pk_rate = sched.energy_rates["C2"]
        op_rate = sched.offpeak_rates["C2"]
        md      = sched.md_rates["C2"]
        ep = round(peak_kwh    * pk_rate, 2)
        eo = round(offpeak_kwh * op_rate, 2)
        result["energy_charge"] = ep + eo
        result["md_charge"]     = round(max_demand_kw * md, 2)
        result["blocks"] = [
            {"Block": f"Peak kWh  ({sched.peak_start_hour:02d}:00–{sched.peak_end_hour:02d}:00, Mon–Sat)",
             "kWh": peak_kwh,    "Rate (RM/kWh)": pk_rate, "Charge (RM)": ep},
            {"Block": "Off-Peak kWh  (all other times)",
             "kWh": offpeak_kwh, "Rate (RM/kWh)": op_rate, "Charge (RM)": eo},
        ]

    elif tariff == "D":
        rate = sched.energy_rates["D"]
        md   = sched.md_rates["D"]
        ec = round(monthly_kwh * rate, 2)
        result["energy_charge"] = ec
        result["md_charge"]     = round(max_demand_kw * md, 2)
        result["blocks"] = [{"Block": "All kWh (flat rate)",
                              "kWh": monthly_kwh, "Rate (RM/kWh)": rate, "Charge (RM)": ec}]

    elif tariff in ("E1", "E2"):
        pk_rate = sched.energy_rates[tariff]
        op_rate = sched.offpeak_rates[tariff]
        md      = sched.md_rates[tariff]
        ep = round(peak_kwh    * pk_rate, 2)
        eo = round(offpeak_kwh * op_rate, 2)
        result["energy_charge"] = ep + eo
        result["md_charge"]     = round(max_demand_kw * md, 2)
        result["blocks"] = [
            {"Block": f"Peak kWh  ({sched.peak_start_hour:02d}:00–{sched.peak_end_hour:02d}:00, Mon–Sat)",
             "kWh": peak_kwh,    "Rate (RM/kWh)": pk_rate, "Charge (RM)": ep},
            {"Block": "Off-Peak kWh  (all other times)",
             "kWh": offpeak_kwh, "Rate (RM/kWh)": op_rate, "Charge (RM)": eo},
        ]

    kwh_for_icpt = monthly_kwh if not TARIFF_META[tariff]["tou"] else (peak_kwh + offpeak_kwh)
    result["icpt_charge"] = round(kwh_for_icpt * icpt_sen_per_kwh / 100, 2)

    if apply_kwtbb and not is_domestic:
        result["kwtbb_charge"] = round(result["energy_charge"] * sched.kwtbb_rate, 2)

    if apply_service_tax and not is_domestic:
        taxable = result["energy_charge"] + result["md_charge"] + result["kwtbb_charge"]
        result["service_tax"] = round(taxable * sched.service_tax_rate, 2)

    raw = (result["energy_charge"] + result["md_charge"] + result["icpt_charge"] +
           result["kwtbb_charge"]  + result["service_tax"])
    result["subtotal_pretax"] = round(
        result["energy_charge"] + result["md_charge"] + result["icpt_charge"], 2)

    min_c = sched.min_charges.get(tariff, 0.0)
    if raw < min_c:
        result["total_bill"] = min_c
        result["minimum_charge_applied"] = True
        result["notes"].append(f"Minimum monthly charge of RM {min_c:.2f} applied.")
    else:
        result["total_bill"] = round(raw, 2)

    if is_domestic:
        result["notes"].append("Tariff A: Service Tax and KWTBB are not applicable.")

    return result


def compute_nem_credit(export_kwh: float, nem_rate_rm: float = NEM_DEFAULT_RATE) -> dict:
    """NEM credit for solar export. Applied AFTER the gross bill."""
    credit = round(export_kwh * nem_rate_rm, 2)
    return {
        "export_kwh": round(export_kwh, 3),
        "nem_rate_rm": nem_rate_rm,
        "nem_credit_rm": credit,
    }


def auto_detect_tariff(df: pd.DataFrame, interval_minutes: int = 30) -> tuple[TariffCode, str, dict]:
    """
    Infer the most likely TNB tariff from a load profile.
    Returns (tariff_code, reason_string, stats_dict).
    """
    kfac = interval_minutes / 60.0
    df = df.copy()
    df["month"] = df["timestamp"].dt.to_period("M")
    df["hour"]  = df["timestamp"].dt.hour
    df["dow"]   = df["timestamp"].dt.dayofweek

    monthly = (df.groupby("month")["kw_import"]
               .agg(total_kwh=lambda x: (x * kfac).sum(), peak_kw="max")
               .reset_index())

    avg_monthly_kwh = float(monthly["total_kwh"].mean())
    abs_peak_kw     = float(df["kw_import"].max())
    avg_kw          = float(df["kw_import"].mean())
    load_factor     = avg_kw / abs_peak_kw if abs_peak_kw > 0 else 0.0

    biz_mask  = (df["hour"] >= 8) & (df["hour"] < 18) & (df["dow"] < 5)
    biz_avg   = float(df.loc[biz_mask,  "kw_import"].mean()) if biz_mask.any() else 0.0
    off_avg   = float(df.loc[~biz_mask, "kw_import"].mean()) if (~biz_mask).any() else 0.0
    biz_ratio = biz_avg / off_avg if off_avg > 0 else 1.0

    wkday_avg   = float(df.loc[df["dow"] < 5,  "kw_import"].mean())
    wkend_avg   = float(df.loc[df["dow"] >= 5, "kw_import"].mean())

    eve_mask  = (df["hour"] >= 18) & (df["hour"] < 23)
    noon_mask = (df["hour"] >= 9)  & (df["hour"] < 17)
    eve_avg   = float(df.loc[eve_mask,  "kw_import"].mean()) if eve_mask.any() else 0.0
    noon_avg  = float(df.loc[noon_mask, "kw_import"].mean()) if noon_mask.any() else 0.0
    res_shape = eve_avg > noon_avg

    stats = {
        "avg_monthly_kwh": round(avg_monthly_kwh, 1),
        "abs_peak_kw":     round(abs_peak_kw, 1),
        "avg_kw":          round(avg_kw, 1),
        "load_factor":     round(load_factor, 3),
        "biz_ratio":       round(biz_ratio, 2),
    }

    if abs_peak_kw >= 1000:
        # High-Voltage supply (33 kV / 132 kV). D, E1, E2 all live here.
        # TNB mandates TOU (E1/E2) once MD ≥ 1,500 kW; below that threshold
        # D vs E1 is the customer's elected metering choice.  We infer it from
        # the load profile: significant off-peak usage (≥ 30 % overnight or
        # load factor ≥ 0.65) signals a 24/7 operation that benefits from TOU.
        night_mask_hv = (df["hour"] < 8) | (df["hour"] >= 22)
        night_avg_hv  = float(df.loc[night_mask_hv, "kw_import"].mean()) if night_mask_hv.any() else 0.0
        night_frac_hv = night_avg_hv / avg_kw if avg_kw > 0 else 0.0
        tou_profile   = night_frac_hv >= 0.30 or load_factor >= 0.65

        if abs_peak_kw >= 1500 and tou_profile:
            code = "E2"
            why  = (f"Peak demand {abs_peak_kw:,.0f} kW ≥ 1,500 kW with strong off-peak usage "
                    f"({night_frac_hv:.0%} overnight, LF {load_factor:.0%}) → "
                    f"mandatory TOU, Tariff E2.")
        elif abs_peak_kw >= 1500:
            # MD ≥ 1,500 kW but daytime-heavy — technically E2 mandatory;
            # flag D as a likely-wrong election and nudge E2.
            code = "E2"
            why  = (f"Peak demand {abs_peak_kw:,.0f} kW ≥ 1,500 kW → Tariff E2 mandatory "
                    f"by TNB regulation regardless of load shape. "
                    f"(Daytime-heavy profile; load factor {load_factor:.0%}.)")
        elif tou_profile:
            code = "E1"
            why  = (f"Peak demand {abs_peak_kw:,.0f} kW → High Voltage supply. "
                    f"Significant off-peak usage ({night_frac_hv:.0%} overnight, "
                    f"LF {load_factor:.0%}) indicates TOU election → Tariff E1.")
        else:
            code = "D"
            why  = (f"Peak demand {abs_peak_kw:,.0f} kW → High Voltage supply. "
                    f"Daytime-heavy profile (off-peak {night_frac_hv:.0%}, "
                    f"LF {load_factor:.0%}) → Tariff D (non-TOU).")

    elif abs_peak_kw >= 25:
        night_mask = (df["hour"] < 8) | (df["hour"] >= 22)
        night_avg  = float(df.loc[night_mask, "kw_import"].mean()) if night_mask.any() else 0.0
        night_frac = night_avg / avg_kw if avg_kw > 0 else 0.0
        if night_frac >= 0.35 or load_factor >= 0.70:
            code = "C2"
            why  = (f"Peak demand {abs_peak_kw:,.0f} kW → Medium Voltage supply. "
                    f"High off-peak usage ({night_frac:.0%} overnight) – Tariff C2 (TOU).")
        else:
            code = "C1"
            why  = (f"Peak demand {abs_peak_kw:,.0f} kW → Medium Voltage supply, Tariff C1. "
                    f"Load factor {load_factor:.0%}, daytime-heavy profile.")

    elif avg_monthly_kwh > 2000 or (biz_ratio > 2.5 and not res_shape):
        code = "B"
        why  = (f"Peak demand {abs_peak_kw:.1f} kW < 25 kW (Low Voltage), "
                f"avg monthly {avg_monthly_kwh:,.0f} kWh with commercial daytime pattern → Tariff B.")

    else:
        code = "A"
        why  = (f"Peak demand {abs_peak_kw:.1f} kW, avg monthly {avg_monthly_kwh:,.0f} kWh → "
                f"Domestic Tariff A.")

    return code, why, stats


def compute_monthly_stats(df: pd.DataFrame, interval_minutes: int = 30,
                          schedule_key: str | None = None) -> pd.DataFrame:
    """Aggregate load profile into per-month billing stats.

    Peak / off-peak split follows the active schedule's TOU window. Most
    sites use the same 08:00-22:00 Mon-Sat convention so the default is
    fine; pass schedule_key='legacy_2014' if comparing against an older
    bill where the window differed.
    """
    kfac = interval_minutes / 60.0
    df = df.copy()
    df["month"]         = df["timestamp"].dt.to_period("M")
    df["is_peak"]       = df["timestamp"].apply(lambda t: is_peak_hour(t, schedule_key))
    df["kwh"]           = df["kw_import"] * kfac
    df["export_kwh_pt"] = df["kw_export"] * kfac

    stats = df.groupby("month", group_keys=False).apply(
        lambda g: pd.Series({
            "total_kwh":     round(float(g["kwh"].sum()), 3),
            "peak_kwh":      round(float(g.loc[g["is_peak"],  "kwh"].sum()), 3),
            "offpeak_kwh":   round(float(g.loc[~g["is_peak"], "kwh"].sum()), 3),
            "max_demand_kw": round(float(g["kw_import"].max()), 3),
            "export_kwh":    round(float(g["export_kwh_pt"].sum()), 3),
        })
    ).reset_index()

    return stats
