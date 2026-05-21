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
from pathlib import Path
from typing import Any


# Single source of truth for where the file lives.
# Resolves to <repo>/BOLT_INTEGRATED/data/site_overrides.json at runtime.
_DATA_DIR = Path(__file__).resolve().parent / "data"
OVERRIDES_PATH = _DATA_DIR / "site_overrides.json"


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
