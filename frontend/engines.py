"""
frontend/engines.py
===================
The ONLY place where the frontend touches analysis logic.

Every engine exposes the same small contract, so the UI never cares where the
numbers come from:

    process_sequence(events: list[str])               -> dict   (DFA run result)
    detect_failures(events_df, max_response_min, max_gap_min) -> DataFrame
    calculate_response_times(events_df)               -> DataFrame
    get_dfa_definition()                              -> dict
    prepare_events(raw_df)                            -> DataFrame
    load_sample()                                     -> DataFrame (raw)

`MockEngine`    canned data from frontend.mock_data (no DFA logic at all).
`BackendEngine` thin wrapper over the real backend (automata.dfa, analysis.*).
                No DFA is re-implemented here: it only calls the backend.

Contract details
----------------
process_sequence -> {start_state, final_state, is_valid, is_complete, accepted,
                     halted, trace[], invalid_transitions[]}
    trace item: {index, from_state, symbol, to_state, valid, reason}
detect_failures  -> columns: incident_id, ambulance_id, failure_type, severity,
                     event_index, event, timestamp, description
calculate_response_times -> columns: incident_id, ambulance_id, call_time,
                     arrived_scene_time, dispatch_min, travel_min, response_min,
                     scene_min, transport_min, total_min   (NaN = not computable)
"""

from __future__ import annotations

from typing import Any, Dict, List, Sequence

import pandas as pd

from frontend import mock_data

REQUIRED_COLUMNS = ["incident_id", "ambulance_id", "event", "timestamp"]
FAILURE_COLUMNS = ["incident_id", "ambulance_id", "failure_type", "severity",
                   "event_index", "event", "timestamp", "description"]
METRIC_COLUMNS = ["incident_id", "ambulance_id", "call_time", "arrived_scene_time", "dispatch_min",
                  "travel_min", "response_min", "scene_min", "transport_min", "total_min"]
_SEVERITY = {"MISSING_EVENT": "HIGH", "OUT_OF_ORDER": "HIGH", "TIMESTAMP_REGRESSION": "HIGH",
             "UNKNOWN_EVENT": "MEDIUM", "INCOMPLETE_SEQUENCE": "MEDIUM", "EVENT_GAP_EXCEEDED": "MEDIUM",
             "SLOW_RESPONSE": "MEDIUM", "DUPLICATE_EVENT": "LOW"}


def _norm_event(e: Any) -> str:
    return str(e).strip().upper().replace(" ", "_")


# --------------------------------------------------------------------------- #
# Mock engine
# --------------------------------------------------------------------------- #
class MockEngine:
    name = "mock"
    label = "Mock data"
    is_mock = True

    def __init__(self) -> None:
        self._df, self._results, self._seq_failures = mock_data.build_mock()
        for _, (seq, res) in mock_data.build_presets().items():
            self._results.setdefault(tuple(seq), res)

    # -- contract ----------------------------------------------------------
    def load_sample(self) -> pd.DataFrame:
        return self._df.copy()

    def prepare_events(self, raw: pd.DataFrame) -> pd.DataFrame:
        missing = [c for c in REQUIRED_COLUMNS if c not in raw.columns]
        if missing:
            raise ValueError(f"Missing required column(s): {missing}")
        out = raw[REQUIRED_COLUMNS].copy()
        out["incident_id"] = out["incident_id"].astype(str).str.strip()
        out["ambulance_id"] = out["ambulance_id"].astype(str).str.strip()
        out["event"] = out["event"].map(_norm_event)
        out["timestamp"] = pd.to_datetime(out["timestamp"], errors="coerce")
        return out.reset_index(drop=True)

    def get_dfa_definition(self) -> Dict[str, Any]:
        return mock_data.MOCK_DFA_DEFINITION

    def process_sequence(self, events: Sequence[str]) -> Dict[str, Any]:
        key = tuple(_norm_event(e) for e in events)
        hit = self._results.get(key)
        if hit is not None:
            return dict(hit, mock_placeholder=False)
        # Unknown sequence (e.g. an uploaded CSV): NOT analysed. Placeholder only.
        n = min(len(key), len(mock_data.EVENTS))
        placeholder = mock_data._result(list(key), n)
        return dict(placeholder, mock_placeholder=True)

    def calculate_response_times(self, events: pd.DataFrame) -> pd.DataFrame:
        first = events.groupby(["incident_id", "event"], sort=False)["timestamp"].first().unstack("event")
        amb = events.groupby("incident_id", sort=False)["ambulance_id"].first()
        first = first.reindex(index=amb.index, columns=list(mock_data.EVENTS))

        def mins(a: str, b: str) -> pd.Series:
            d = (first[b] - first[a]).dt.total_seconds() / 60.0
            return d.where(d >= 0).round(2)

        out = pd.DataFrame({
            "incident_id": amb.index, "ambulance_id": amb.values,
            "call_time": first["CALL_RECEIVED"].values, "arrived_scene_time": first["ARRIVED_SCENE"].values,
            "dispatch_min": mins("CALL_RECEIVED", "AMBULANCE_DISPATCHED").values,
            "travel_min": mins("EN_ROUTE", "ARRIVED_SCENE").values,
            "response_min": mins("CALL_RECEIVED", "ARRIVED_SCENE").values,
            "scene_min": mins("ARRIVED_SCENE", "DEPARTED_SCENE").values,
            "transport_min": mins("DEPARTED_SCENE", "ARRIVED_HOSPITAL").values,
            "total_min": mins("CALL_RECEIVED", "HANDOVER_COMPLETE").values,
        })
        return out[METRIC_COLUMNS]

    def detect_failures(self, events: pd.DataFrame, max_response_min: float = 10.0,
                        max_gap_min: float = 30.0) -> pd.DataFrame:
        rows: List[Dict[str, Any]] = []

        def add(iid, amb, ftype, idx, ev, ts, desc):
            rows.append({"incident_id": iid, "ambulance_id": amb, "failure_type": ftype,
                         "severity": _SEVERITY[ftype], "event_index": idx, "event": ev,
                         "timestamp": ts, "description": desc})

        metrics = self.calculate_response_times(events).set_index("incident_id")
        for iid, g in events.groupby("incident_id", sort=False):
            amb = g["ambulance_id"].iloc[0]
            ev, ts = g["event"].tolist(), g["timestamp"].tolist()
            canned = self._seq_failures.get(tuple(ev))
            if canned:
                ftype, idx, desc = canned
                add(iid, amb, ftype, idx, ev[idx], ts[idx], desc)
            for i in range(1, len(ev)):                              # timing rules
                gap = (ts[i] - ts[i - 1]).total_seconds() / 60.0
                if gap < 0:
                    add(iid, amb, "TIMESTAMP_REGRESSION", i, ev[i], ts[i],
                        f"'{ev[i]}' is {abs(gap):.1f} min earlier than the preceding event '{ev[i - 1]}'.")
                elif gap > max_gap_min:
                    add(iid, amb, "EVENT_GAP_EXCEEDED", i, ev[i], ts[i],
                        f"{gap:.1f} min between '{ev[i - 1]}' and '{ev[i]}' (limit {max_gap_min:g}).")
            r = metrics.loc[iid, "response_min"]
            if pd.notna(r) and r > max_response_min:
                add(iid, amb, "SLOW_RESPONSE", None, "ARRIVED_SCENE", metrics.loc[iid, "arrived_scene_time"],
                    f"Response time {r:.1f} min exceeds the {max_response_min:g} min target.")
        return pd.DataFrame(rows, columns=FAILURE_COLUMNS)


# --------------------------------------------------------------------------- #
# Real backend engine (thin adapter - no DFA logic here)
# --------------------------------------------------------------------------- #
class BackendEngine:
    name = "backend"
    label = "Live backend"
    is_mock = False

    def __init__(self) -> None:
        from analysis import analytics, failure_detection        # noqa: WPS433 (lazy on purpose)
        from automata import dfa

        self._analytics, self._fd, self._dfa = analytics, failure_detection, dfa

    def load_sample(self) -> pd.DataFrame:
        return self._analytics.load_events()

    def prepare_events(self, raw: pd.DataFrame) -> pd.DataFrame:
        return self._analytics.prepare_events(raw)

    def get_dfa_definition(self) -> Dict[str, Any]:
        return self._dfa.get_dfa_definition()

    def process_sequence(self, events: Sequence[str]) -> Dict[str, Any]:
        return self._dfa.validate_event_sequence(list(events)).to_dict()

    def detect_failures(self, events: pd.DataFrame, max_response_min: float = 10.0,
                        max_gap_min: float = 30.0) -> pd.DataFrame:
        return self._fd.detect_failures(events, max_response_min, max_gap_min)

    def calculate_response_times(self, events: pd.DataFrame) -> pd.DataFrame:
        return self._analytics.compute_incident_metrics(events)[METRIC_COLUMNS]


# --------------------------------------------------------------------------- #
_CACHE: Dict[str, Any] = {}


def backend_available() -> bool:
    try:
        get_engine("backend")
        return True
    except Exception:                                            # ImportError or broken backend
        return False


def get_engine(name: str):
    """'mock' or 'backend'.  Engines are stateless, so they are shared."""
    if name not in _CACHE:
        _CACHE[name] = BackendEngine() if name == "backend" else MockEngine()
    return _CACHE[name]
