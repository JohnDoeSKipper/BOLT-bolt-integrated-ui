"""
Generate per-site "live" sample data files for the Live Simulation tab.

Each generated CSV is a 14-day forward extension of the original case-study
Excel — last 14 days of real readings, with:
  • timestamps shifted forward by 14 days (so they do NOT overlap the
    training window — the forecaster genuinely hasn't seen these times),
  • small Gaussian noise (~3 %) on each power column (so the model can't
    pattern-match exactly to its training rows),
keeping every other shape (daily curve, weekly rhythm, solar dip if present)
authentic.

Files land at:
  BOLT_INTEGRATED/data/live_samples/<site_id>_live.csv

Use them via the Streamlit **🔴 Live Simulation** tab →
*"📥 Replace future stream from a CSV (advanced)"* expander.

Re-run any time with different params:
  python scripts/generate_live_samples.py
"""
from __future__ import annotations
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Make `from predictor.data_loader import auto_load` resolve when run as a script
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from predictor.data_loader import auto_load   # noqa: E402


# ----------- Source location of the original Excel files ----------------------
# The 4 case-study files live in the upstream BOLT-master clone:
SRC_DIR = Path("/Users/jaredang/Downloads/BOLT-master/Predictor/data")

SOURCES = {
    "sol": SRC_DIR / "1. Load Profile (With Solar Installed) SoL.xlsx",
    "e":   SRC_DIR / "2. Load Profile (No Solar) E.xlsx",
    "sun": SRC_DIR / "3. Load Profile (No Solar) SuN.xlsx",
    "mi2": SRC_DIR / "4. Load Profile (With Solar) Mi2.xlsx",
}

# ----------- Generation parameters -------------------------------------------
TAIL_DAYS = 14            # take this many days from the END of each source
FORWARD_SHIFT_DAYS = 14   # shift those timestamps this far into the future
NOISE_PCT = 0.03          # Gaussian σ in fractional units (≈ ±3 % on each row)

OUT_DIR = ROOT / "data" / "live_samples"


def generate_for_site(site_id: str, src_path: Path) -> dict | None:
    if not src_path.is_file():
        print(f"  [skip] {site_id:>4s} — source not found: {src_path.name}")
        return None

    df = auto_load(src_path).sort_values("timestamp").reset_index(drop=True)
    if df.empty:
        print(f"  [skip] {site_id:>4s} — auto_load returned empty frame")
        return None

    # Take the last TAIL_DAYS of real data
    last_ts = df["timestamp"].max()
    cutoff = last_ts - pd.Timedelta(days=TAIL_DAYS)
    tail = df[df["timestamp"] > cutoff].copy().reset_index(drop=True)
    if len(tail) == 0:
        print(f"  [skip] {site_id:>4s} — empty tail after cutoff {cutoff}")
        return None

    original_start, original_end = tail["timestamp"].min(), tail["timestamp"].max()

    # Shift timestamps forward so they're past any training timestamp.
    # The model genuinely hasn't seen this timeline.
    tail["timestamp"] = tail["timestamp"] + pd.Timedelta(days=FORWARD_SHIFT_DAYS)

    # Add small noise to every power column so it isn't an exact replay.
    # Seed per-site so re-runs are reproducible.
    rng = np.random.default_rng(seed=abs(hash(site_id)) % (2**32))
    for col in ("kw_import", "kw_export", "kvar_import", "kvar_export"):
        if col in tail.columns:
            noise = rng.normal(0, NOISE_PCT, size=len(tail))
            tail[col] = (tail[col].astype(float).values * (1.0 + noise)).clip(min=0.0)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / f"{site_id}_live.csv"
    tail.to_csv(out_path, index=False)

    summary = {
        "site_id":         site_id,
        "rows":            len(tail),
        "source_file":     src_path.name,
        "original_range":  f"{original_start} → {original_end}",
        "shifted_range":   f"{tail['timestamp'].min()} → {tail['timestamp'].max()}",
        "size_kb":         round(out_path.stat().st_size / 1024, 1),
        "out_path":        str(out_path.relative_to(ROOT)),
        "peak_kw":         round(float(tail["kw_import"].max()), 1),
        "has_export":      bool((tail.get("kw_export", pd.Series([0])) > 0).any()),
    }
    print(
        f"  [ok]   {site_id:>4s} · {summary['rows']:>4d} rows · "
        f"peak {summary['peak_kw']:>7.1f} kW · "
        f"export={'Y' if summary['has_export'] else 'N'} · "
        f"→ {summary['out_path']}"
    )
    return summary


def main():
    print(f"Generating live samples from {SRC_DIR}/  →  {OUT_DIR.relative_to(ROOT)}/")
    print(
        f"  tail={TAIL_DAYS}d  forward_shift={FORWARD_SHIFT_DAYS}d  "
        f"noise_pct={NOISE_PCT*100:.1f}%"
    )
    print()
    results = []
    for sid, src in SOURCES.items():
        r = generate_for_site(sid, src)
        if r:
            results.append(r)

    print()
    print(f"Generated {len(results)}/{len(SOURCES)} live-sample CSVs in {OUT_DIR}/")
    print("Use them via Streamlit  →  🔴 Live Simulation  →  📥 Replace future stream.")


if __name__ == "__main__":
    main()
