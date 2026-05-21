"""
Disk persistence for per-site overrides.

The override dict — what the user typed into Site Setup — lives in memory
(Streamlit's session_state, Flask's `S` dict).  Without persistence those
edits vanish on browser refresh or server restart, forcing the user to
re-enter location / capacity / EV config every time.

This module saves the whole `{site_id: {field: value}}` dict to a JSON
file under the project's `data/` directory and reloads it on app start.

Why JSON, not pickle: humans can open the file to inspect or hand-edit
the saved parameters between sessions, which matters during a demo when
you want to confirm "yes, the Penang lat/lon I typed yesterday is still
in effect".

Failure mode: any read/write error is silently swallowed — the app
falls back to the in-memory empty dict and the user just has to re-enter
their overrides. We never let a corrupt JSON crash the app.
"""
from __future__ import annotations
import json
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any

import joblib


# Single source of truth for where files live.
# Resolves to <repo>/BOLT_INTEGRATED/data/... at runtime.
_DATA_DIR = Path(__file__).resolve().parent / "data"
OVERRIDES_PATH = _DATA_DIR / "site_overrides.json"
SITES_DIR      = _DATA_DIR / "sites"


def _site_dir(site_id: str) -> Path:
    """Per-site state directory: data/sites/<site_id>/"""
    return SITES_DIR / site_id


def _data_path(site_id: str) -> Path:
    return _site_dir(site_id) / "load_profile.joblib"


def _forecaster_path(site_id: str) -> Path:
    return _site_dir(site_id) / "forecaster.joblib"


def load_overrides() -> dict[str, dict[str, Any]]:
    """Return the persisted overrides, or {} if the file is missing/corrupt.

    The dict shape is `{site_id: {field: value}}` — same shape used by
    `st.session_state.site_overrides` and `S['site_overrides']`.
    """
    if not OVERRIDES_PATH.is_file():
        return {}
    try:
        raw = OVERRIDES_PATH.read_text()
        data = json.loads(raw)
        if not isinstance(data, dict):
            return {}
        # Strip any keys whose values aren't themselves dicts (defensive)
        return {k: v for k, v in data.items() if isinstance(v, dict)}
    except (json.JSONDecodeError, OSError, ValueError):
        return {}


def save_overrides(overrides: dict[str, dict[str, Any]]) -> bool:
    """Atomically write the override dict to disk. Returns success bool.

    Uses a temp-file + rename so a half-written file from a crashed
    process can't corrupt the canonical file.
    """
    if overrides is None:
        return False
    try:
        _DATA_DIR.mkdir(parents=True, exist_ok=True)
        tmp = OVERRIDES_PATH.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(overrides, indent=2, sort_keys=True, default=str))
        tmp.replace(OVERRIDES_PATH)
        return True
    except (OSError, TypeError):
        return False


def clear_overrides() -> bool:
    """Delete the saved file (for reset/debugging). True if removed."""
    try:
        if OVERRIDES_PATH.is_file():
            OVERRIDES_PATH.unlink()
        return True
    except OSError:
        return False


# =========================================================================
#                       PER-SITE DATA + MODEL STATE
# =========================================================================
# Saved separately per site so SoL/E/SuN/Mi2/Custom each keep their own
# uploaded load profile + trained forecaster.  Files are joblib pickles
# (binary, fast, faithful for arbitrary Python objects).  Loading code is
# defensive: any unpickle / version mismatch silently returns None, and
# the caller falls back to "no saved state" (forces a fresh upload or
# retrain).
# =========================================================================


def save_load_profile(site_id: str, df) -> bool:
    """Persist the uploaded + normalized load DataFrame for `site_id`.
    Overwrites any previous save for that site.
    """
    try:
        p = _data_path(site_id)
        p.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(df, p)
        return True
    except Exception:
        return False


def load_load_profile(site_id: str):
    """Return the saved load DataFrame for `site_id`, or None if missing
    / corrupt.  Same canonical schema produced by predictor.data_loader."""
    p = _data_path(site_id)
    if not p.is_file():
        return None
    try:
        return joblib.load(p)
    except Exception:
        return None


def save_forecaster(site_id: str, fc) -> bool:
    """Persist a trained DirectMultiStepForecaster for `site_id`.
    Uses the forecaster's own .save() so calibration profile + history
    are preserved.
    """
    try:
        p = _forecaster_path(site_id)
        p.parent.mkdir(parents=True, exist_ok=True)
        fc.save(p)
        return True
    except Exception:
        return False


def load_forecaster(site_id: str):
    """Return the saved DirectMultiStepForecaster for `site_id`, or None.

    Returns None on:
      - File missing
      - Pickle corruption
      - Feature-shape mismatch from a codebase upgrade (caller must retrain)
    """
    p = _forecaster_path(site_id)
    if not p.is_file():
        return None
    try:
        from predictor.forecaster import DirectMultiStepForecaster
        return DirectMultiStepForecaster.load(p)
    except Exception:
        return None


def site_state_info(site_id: str) -> dict:
    """Snapshot of what's saved for a site, formatted for UI display."""
    info: dict[str, Any] = {
        "site_id":     site_id,
        "has_data":    False,
        "has_model":   False,
        "data_rows":   None,
        "data_kb":     None,
        "model_mb":    None,
        "data_saved":  None,
        "model_saved": None,
    }
    p = _data_path(site_id)
    if p.is_file():
        info["has_data"]   = True
        info["data_kb"]    = round(p.stat().st_size / 1024, 1)
        info["data_saved"] = datetime.fromtimestamp(p.stat().st_mtime).isoformat(timespec="seconds")
        try:
            df = joblib.load(p)
            info["data_rows"] = int(len(df))
        except Exception:
            pass
    pm = _forecaster_path(site_id)
    if pm.is_file():
        info["has_model"]    = True
        info["model_mb"]     = round(pm.stat().st_size / 1_000_000, 2)
        info["model_saved"]  = datetime.fromtimestamp(pm.stat().st_mtime).isoformat(timespec="seconds")
    return info


def all_sites_state_info(site_ids: list[str]) -> list[dict]:
    """Bulk version for the Site Setup overview table."""
    return [site_state_info(s) for s in site_ids]


def clear_site_state(site_id: str, *, also_clear_overrides: bool = False) -> bool:
    """Delete saved data + forecaster for a site. If `also_clear_overrides`
    is True, the Site Setup edits for that site are wiped too.
    """
    try:
        sd = _site_dir(site_id)
        if sd.is_dir():
            shutil.rmtree(sd)
        if also_clear_overrides:
            ov = load_overrides()
            if site_id in ov:
                del ov[site_id]
                save_overrides(ov)
        return True
    except Exception:
        return False
