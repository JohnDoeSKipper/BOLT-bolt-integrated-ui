"""
TNB tariff rate schedules — single source of truth.

Both `calculator.tnb_tariffs.calculate_bill` and `powerreco.roi_engine.calculate_roi`
import from here, so a rate change propagates to billing AND ROI in one place.

Two schedules:
  - 'legacy_2014': pre-July-2025 rates (revised 2014, updated 2022).
  - 'new_2025':    July 2025 revision — MD jumped ~3× for medium-voltage
                   commercial. C1/C2 confirmed at RM 97.06/kW from the
                   official TNB announcement.

For D / E1 / E2 in 2025 the public schedule has not been published in the same
form as C1/C2 in some sources; the values below are scaled by the same C1
multiplier (97.06 / 30.30 ≈ 3.20×) as a defensible estimate. Override per
site if the operator has actual post-July-2025 bill stubs.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Literal

TariffCode = Literal["A", "B", "C1", "C2", "D", "E1", "E2"]
ScheduleKey = Literal["legacy_2014", "new_2025"]


@dataclass(frozen=True)
class TariffSchedule:
    """Immutable rate snapshot. New schedule = new instance; never mutate."""

    key: ScheduleKey
    name: str
    effective_from: str

    # MD charge in RM/kW/month. 0 for non-MD tariffs (A, B).
    md_rates: dict[str, float]

    # Energy charge in RM/kWh. For TOU tariffs (C2, E1, E2) this is the PEAK rate.
    energy_rates: dict[str, float]

    # Off-peak energy charge in RM/kWh for TOU tariffs only.
    offpeak_rates: dict[str, float]

    # Minimum monthly charge per tariff in RM.
    min_charges: dict[str, float]

    # Universal charges
    kwtbb_rate: float = 0.016
    service_tax_rate: float = 0.06
    nem_default_rate: float = 0.31

    # Peak window — 08:00-22:00 Mon-Sat is the legacy convention. The
    # July 2025 revision retains it for the C/E tariffs; revise here if
    # TNB publishes new TOU windows.
    peak_start_hour: int = 8
    peak_end_hour: int = 22
    saturday_is_peak: bool = True

    # Optional source / notes for transparency in the UI
    notes: str = ""


# =========================================================================
#                          LEGACY (PRE-JULY 2025)
# =========================================================================
LEGACY_2014 = TariffSchedule(
    key="legacy_2014",
    name="Pre-July 2025 (revised 2014, updated 2022)",
    effective_from="2014-01-01",
    md_rates={
        "A": 0.0, "B": 0.0,
        "C1": 30.30, "C2": 30.30,
        "D": 29.60, "E1": 29.60, "E2": 29.60,
    },
    energy_rates={
        # Tariff A is tiered — see TARIFF_A_BLOCKS in tnb_tariffs.py; this
        # entry is unused for A but kept for shape consistency.
        "A": 0.218,
        "B": 0.435,
        "C1": 0.365, "C2": 0.365,
        "D": 0.337, "E1": 0.337, "E2": 0.337,
    },
    offpeak_rates={
        "C2": 0.219,
        "E1": 0.202, "E2": 0.202,
    },
    min_charges={
        "A": 3.00, "B": 7.20,
        "C1": 600.00, "C2": 600.00,
        "D": 600.00, "E1": 600.00, "E2": 600.00,
    },
    notes="Official TNB schedule revised 2014, updated 2022.",
)


# =========================================================================
#                          NEW (POST-JULY 2025)
# =========================================================================
# MD multiplier applied to legacy non-C rates as a defensible estimate
# until TNB publishes the full schedule. C1/C2 is the confirmed value.
_MD_MULTIPLIER_2025 = 97.06 / 30.30  # ≈ 3.203

NEW_2025 = TariffSchedule(
    key="new_2025",
    name="Post-July 2025 (Commercial MD revision)",
    effective_from="2025-07-01",
    md_rates={
        "A": 0.0, "B": 0.0,
        "C1": 97.06, "C2": 97.06,                       # confirmed
        "D":  round(29.60 * _MD_MULTIPLIER_2025, 2),    # estimate ≈ 94.81
        "E1": round(29.60 * _MD_MULTIPLIER_2025, 2),    # estimate ≈ 94.81
        "E2": round(29.60 * _MD_MULTIPLIER_2025, 2),    # estimate ≈ 94.81
    },
    energy_rates={
        # July 2025 revision was primarily MD-side; energy rates held.
        "A": 0.218,
        "B": 0.435,
        "C1": 0.365, "C2": 0.365,
        "D": 0.337, "E1": 0.337, "E2": 0.337,
    },
    offpeak_rates={
        "C2": 0.219,
        "E1": 0.202, "E2": 0.202,
    },
    min_charges={
        "A": 3.00, "B": 7.20,
        "C1": 600.00, "C2": 600.00,
        "D": 600.00, "E1": 600.00, "E2": 600.00,
    },
    notes=(
        "C1/C2 MD rate confirmed at RM 97.06/kW from official TNB July 2025 "
        "announcement. D/E1/E2 figures scaled by the same C1 multiplier "
        "(~3.20×) as an estimate; override with the operator's actual bill "
        "stubs once published. Energy rates unchanged from legacy schedule."
    ),
)


# =========================================================================
#                          REGISTRY + LOOKUPS
# =========================================================================
SCHEDULES: dict[str, TariffSchedule] = {
    "legacy_2014": LEGACY_2014,
    "new_2025":    NEW_2025,
}

DEFAULT_SCHEDULE_KEY: ScheduleKey = "new_2025"


def get_schedule(key: str | None = None) -> TariffSchedule:
    """Return the named schedule, or the default if None.

    Unknown keys fall back to the default (with a comment in the code)
    so the system never crashes on a stale frontend value.
    """
    if key is None:
        return SCHEDULES[DEFAULT_SCHEDULE_KEY]
    return SCHEDULES.get(key, SCHEDULES[DEFAULT_SCHEDULE_KEY])


def get_md_rate(tariff: str, schedule_key: str | None = None) -> float:
    return get_schedule(schedule_key).md_rates.get(tariff, 0.0)


def get_energy_rate(tariff: str, schedule_key: str | None = None) -> float:
    return get_schedule(schedule_key).energy_rates.get(tariff, 0.0)


def get_offpeak_rate(tariff: str, schedule_key: str | None = None) -> float:
    return get_schedule(schedule_key).offpeak_rates.get(tariff, 0.0)


def get_min_charge(tariff: str, schedule_key: str | None = None) -> float:
    return get_schedule(schedule_key).min_charges.get(tariff, 0.0)
