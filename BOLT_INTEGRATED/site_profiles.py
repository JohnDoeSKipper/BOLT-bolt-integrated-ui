"""
Site Profiles — preset configurations for the four ESUM × RExharge case-study
buildings plus a generic 'Custom' fallback.

Each profile bundles:
  - Forecaster tuning knobs (per-site, derived from master's evaluation runs)
  - Suggested lat/lon for real-weather lookup (defaults to KL)
  - Default Manager loads with EV chargers as a first-class load type
  - Default battery + dispatch parameters
  - Hints about expected tariff and solar capacity (still auto-detected from
    the actual uploaded data — these are starting suggestions only)

Selecting a profile in the UI populates every dependent form. The user can
override any field before training. 'Custom' uses generic defaults so anyone
can upload a non-case-study site without crashing.

Values for SoL / E / SuN / Mi2 come from Manager/sites/<id>/config.json in
the upstream BOLT-master repo, which were tuned via the
scripts/eval_all_sites.py harness.
"""
from __future__ import annotations
from dataclasses import dataclass, field, asdict
from typing import Literal, Optional


SiteId = Literal["sol", "e", "sun", "mi2", "custom"]


@dataclass
class EVChargerConfig:
    """A single EV charger or homogeneous bank of chargers."""
    count: int = 0
    kw_each: float = 22.0                     # 7 / 11 / 22 (AC) or 50 / 150 (DC)
    kind: Literal["AC", "DC"] = "AC"

    @property
    def total_kw(self) -> float:
        return self.count * self.kw_each


@dataclass
class LoadDefinition:
    """A controllable load block the AI Manager can shift or shed."""
    name: str
    proportion: float                          # share of total kVA (sums to 1.0)
    max_cut_pct: float                         # 0.0–1.0, how aggressively cuttable
    kind: Literal["ev", "hvac", "process", "misc"] = "misc"
    # Time-of-day windows (hour 0-23). If set, the Manager interprets them:
    #   - allowed_window: load only present in this window (outside = freely shed)
    #   - protected_window: load cannot be cut inside this window
    allowed_window: Optional[tuple[int, int]] = None
    protected_window: Optional[tuple[int, int]] = None
    # Free-form metadata for EV-style loads
    ev_chargers: list[EVChargerConfig] = field(default_factory=list)


@dataclass
class PredictorParams:
    """Forecaster tuning knobs — per-site values from master's eval runs."""
    n_estimators: int = 300            # bumped from 200 (early stopping caps actual rounds)
    learning_rate: float = 0.05
    h_cap_short: float = 1.6
    h_cap_mid: float = 1.0
    h_cap_long_short: float = 0.8
    h_cap_long: float = 0.6
    tail_min_child: int = 5
    body_min_child: int = 15
    bias_window: int = 12


@dataclass
class SiteProfile:
    """Everything the system needs to bootstrap a site."""
    id: SiteId
    name: str
    description: str

    # Geographic — for Open-Meteo weather lookup. Default to KL.
    lat: float = 3.139
    lon: float = 101.6869
    timezone: str = "auto"

    # Defaults the UI pre-fills (all overridable; auto-detect still runs)
    expected_tariff: str = "C1"
    expected_solar_kwp: Optional[float] = None       # None = auto-detect from data
    expected_roof_area_m2: float = 800.0

    # Forecaster tuning
    predictor: PredictorParams = field(default_factory=PredictorParams)

    # Manager defaults
    battery_kwh: float = 200.0
    c_rate: float = 0.5
    md_target_pct: float = 0.85
    initial_soc_pct: float = 0.50
    charge_upper_pct: float = 0.70
    lookahead_intervals: int = 16

    # Loads — order = priority for cutting (first = most expendable)
    loads: dict[str, LoadDefinition] = field(default_factory=dict)

    # Hint to UI about expected uploaded file
    data_hint: str = ""


# =========================================================================
#                          DEFAULT EV PROFILE
# =========================================================================
# Reasonable starting point for a Malaysian commercial site: 4× 22 kW AC
# chargers, charging permitted 18:00 — 08:00 (cheap-hours window).
def _default_ev_load() -> LoadDefinition:
    return LoadDefinition(
        name="EV Chargers",
        proportion=0.30,
        max_cut_pct=0.40,                  # highly flexible
        kind="ev",
        allowed_window=(18, 8),            # 18:00 → 08:00 next morning
        ev_chargers=[EVChargerConfig(count=4, kw_each=22.0, kind="AC")],
    )


def _default_hvac_load() -> LoadDefinition:
    return LoadDefinition(
        name="HVAC",
        proportion=0.40,
        max_cut_pct=0.15,
        kind="hvac",
        protected_window=(9, 17),          # don't cut during business hours
    )


def _default_misc_load() -> LoadDefinition:
    return LoadDefinition(
        name="Misc Loads",
        proportion=0.30,
        max_cut_pct=0.20,
        kind="misc",
    )


def _default_loads() -> dict[str, LoadDefinition]:
    """Order matters — first key is cut first when the Manager needs to shed."""
    return {
        "ev":   _default_ev_load(),
        "hvac": _default_hvac_load(),
        "misc": _default_misc_load(),
    }


# =========================================================================
#                              PROFILES
# =========================================================================
SOL = SiteProfile(
    id="sol",
    name="SoL — Solar-Equipped Commercial",
    description=(
        "Commercial site with installed solar PV (~944 kWp from the original "
        "case study). Volatile net-load shape from cloud cover, so the "
        "forecaster uses a tight bias window (8) to track regime shifts.  "
        "tail_min_child bumped to 10 for smoother quantile bands under "
        "high cloud-cover variance."
    ),
    expected_tariff="C1",
    expected_solar_kwp=944.0,
    expected_roof_area_m2=1500.0,
    predictor=PredictorParams(n_estimators=300, bias_window=8,
                                tail_min_child=10, body_min_child=18),
    loads=_default_loads(),
    data_hint="1. Load Profile (With Solar Installed) SoL.xlsx",
)

E = SiteProfile(
    id="e",
    name="E — Industrial / No Solar",
    description=(
        "Large industrial site without solar. Stable daily shape with slow "
        "long-horizon drift, so the forecaster uses a wide bias window (24) "
        "to track systematic offsets over the full 24-h horizon."
    ),
    expected_tariff="C1",
    expected_solar_kwp=0.0,
    expected_roof_area_m2=2000.0,
    predictor=PredictorParams(n_estimators=300, bias_window=24),
    battery_kwh=300.0,
    loads=_default_loads(),
    data_hint="2. Load Profile (No Solar) E.xlsx",
)

SUN = SiteProfile(
    id="sun",
    name="SuN — No Solar / Variable",
    description=(
        "Commercial site without solar. Lower long-horizon model capacity "
        "(h_cap_long_short=0.6, h_cap_long=0.4) and slightly larger "
        "min_child_samples (8 / 20) — keeps the model from over-fitting to "
        "weekly noise. Bias window 12."
    ),
    expected_tariff="C1",
    expected_solar_kwp=0.0,
    expected_roof_area_m2=600.0,
    predictor=PredictorParams(
        n_estimators=240,
        h_cap_long_short=0.6,
        h_cap_long=0.4,
        tail_min_child=8,
        body_min_child=20,
        bias_window=12,
    ),
    loads=_default_loads(),
    data_hint="3. Load Profile (No Solar) SuN.xlsx",
)

MI2 = SiteProfile(
    id="mi2",
    name="Mi2 — Mixed Use w/ Solar",
    description=(
        "Mixed-use commercial site with solar installed. Bias window 18 — "
        "between SoL's volatile (8) and E's stable (24) — to match the "
        "site's moderate regime variability."
    ),
    expected_tariff="C1",
    expected_solar_kwp=780.0,
    expected_roof_area_m2=1200.0,
    predictor=PredictorParams(n_estimators=300, bias_window=18,
                                tail_min_child=10, body_min_child=18),
    battery_kwh=250.0,
    loads=_default_loads(),
    data_hint="4. Load Profile (With Solar) Mi2.xlsx",
)

CUSTOM = SiteProfile(
    id="custom",
    name="Custom — Upload your own site",
    description=(
        "Generic defaults — works for any Malaysian commercial building. "
        "Auto-detection fills in tariff, solar capacity, and PF from the "
        "uploaded data. Forecaster uses balanced tuning (bias_window=12)."
    ),
    expected_tariff="C1",
    expected_solar_kwp=None,
    expected_roof_area_m2=800.0,
    predictor=PredictorParams(n_estimators=300),  # bumped from 200
    loads=_default_loads(),
    data_hint="Any TNB-format Excel or CSV load profile.",
)


PROFILES: dict[str, SiteProfile] = {
    "sol":    SOL,
    "e":      E,
    "sun":    SUN,
    "mi2":    MI2,
    "custom": CUSTOM,
}
DEFAULT_PROFILE_ID: SiteId = "custom"


def list_profiles() -> list[dict]:
    """Lightweight metadata for the UI dropdown."""
    return [
        {
            "id":           p.id,
            "name":         p.name,
            "description":  p.description,
            "data_hint":    p.data_hint,
            "expected_tariff":   p.expected_tariff,
            "expected_solar_kwp": p.expected_solar_kwp,
        }
        for p in PROFILES.values()
    ]


def get_profile(profile_id: str | None) -> SiteProfile:
    """Return the named profile, falling back to 'custom' if not found."""
    if profile_id is None:
        return PROFILES[DEFAULT_PROFILE_ID]
    return PROFILES.get(profile_id, PROFILES[DEFAULT_PROFILE_ID])


def profile_to_dict(profile: SiteProfile) -> dict:
    """Serialize a profile (and its nested loads / EV configs) for JSON APIs."""
    out = asdict(profile)
    # asdict turns LoadDefinition.allowed_window tuples into lists — preserve
    # them so the Manager can compare against an int hour.
    for k, ld in out["loads"].items():
        if ld.get("allowed_window") is not None:
            ld["allowed_window"] = tuple(ld["allowed_window"])
        if ld.get("protected_window") is not None:
            ld["protected_window"] = tuple(ld["protected_window"])
    return out


def profile_loads_for_manager(profile: SiteProfile) -> dict:
    """
    Convert profile.loads → the dict shape run_ai_manager() expects.

    The Manager's per-load dict supports: name, proportion, max_cut_pct.
    We pass through the additional metadata (kind, allowed_window,
    protected_window, ev_chargers) for the window-aware cut logic added
    in manager/optimizer.py — older Manager versions ignore unknown keys.
    """
    out = {}
    for k, ld in profile.loads.items():
        entry = {
            "name":        ld.name,
            "proportion":  ld.proportion,
            "max_cut_pct": ld.max_cut_pct,
            "kind":        ld.kind,
        }
        if ld.allowed_window is not None:
            entry["allowed_window"] = list(ld.allowed_window)
        if ld.protected_window is not None:
            entry["protected_window"] = list(ld.protected_window)
        if ld.ev_chargers:
            entry["ev_chargers"] = [
                {"count": e.count, "kw_each": e.kw_each, "kind": e.kind}
                for e in ld.ev_chargers
            ]
            entry["ev_total_kw"] = sum(e.total_kw for e in ld.ev_chargers)
        out[k] = entry
    return out


def profile_predictor_kwargs(profile: SiteProfile) -> dict:
    """Forecaster constructor kwargs derived from the profile."""
    p = profile.predictor
    return {
        "n_estimators":     p.n_estimators,
        "learning_rate":    p.learning_rate,
        "h_cap_short":      p.h_cap_short,
        "h_cap_mid":        p.h_cap_mid,
        "h_cap_long_short": p.h_cap_long_short,
        "h_cap_long":       p.h_cap_long,
        "tail_min_child":   p.tail_min_child,
        "body_min_child":   p.body_min_child,
        "lat":              profile.lat,
        "lon":              profile.lon,
        "timezone":         profile.timezone,
    }
