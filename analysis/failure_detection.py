"""
analysis/failure_detection.py
=============================
Detects failures in ambulance event logs using the ambulance DFA plus
timestamp and response-time rules.

Failure types
-------------
Sequence (from the DFA, first violation per incident):
    MISSING_EVENT        a required event was skipped
    DUPLICATE_EVENT      the previous event was repeated
    OUT_OF_ORDER         an earlier event reappeared / event after completion
    UNKNOWN_EVENT        event name is not in the alphabet
    INCOMPLETE_SEQUENCE  valid so far but never reached CLOSED
Timing:
    TIMESTAMP_REGRESSION a timestamp is earlier than the previous event's
    EVENT_GAP_EXCEEDED   gap between consecutive events > max_gap_min
    SLOW_RESPONSE        call -> arrival at scene > max_response_min

Events are checked in LOG ORDER (row order), not re-sorted by timestamp, so
that timestamp problems stay visible.
"""

from __future__ import annotations

import sys
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

from analysis import analytics
from automata.dfa import (
    EVENT_INDEX,
    EVENTS,
    STATE_INDEX,
    get_expected_event,
    validate_event_sequence,
)


class FailureType(str, Enum):
    MISSING_EVENT = "MISSING_EVENT"
    DUPLICATE_EVENT = "DUPLICATE_EVENT"
    OUT_OF_ORDER = "OUT_OF_ORDER"
    UNKNOWN_EVENT = "UNKNOWN_EVENT"
    INCOMPLETE_SEQUENCE = "INCOMPLETE_SEQUENCE"
    TIMESTAMP_REGRESSION = "TIMESTAMP_REGRESSION"
    EVENT_GAP_EXCEEDED = "EVENT_GAP_EXCEEDED"
    SLOW_RESPONSE = "SLOW_RESPONSE"


SEVERITY: Dict[FailureType, str] = {
    FailureType.MISSING_EVENT: "HIGH",
    FailureType.OUT_OF_ORDER: "HIGH",
    FailureType.TIMESTAMP_REGRESSION: "HIGH",
    FailureType.UNKNOWN_EVENT: "MEDIUM",
    FailureType.INCOMPLETE_SEQUENCE: "MEDIUM",
    FailureType.EVENT_GAP_EXCEEDED: "MEDIUM",
    FailureType.SLOW_RESPONSE: "MEDIUM",
    FailureType.DUPLICATE_EVENT: "LOW",
}

FAILURE_COLUMNS = [
    "incident_id", "ambulance_id", "failure_type", "severity",
    "event_index", "event", "timestamp", "description",
]
DEFAULT_MAX_RESPONSE_MIN = analytics.DEFAULT_RESPONSE_THRESHOLD_MIN
DEFAULT_MAX_GAP_MIN = 30.0


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _row(incident_id, ambulance_id, ftype: FailureType, event_index, event, timestamp, description) -> Dict[str, Any]:
    return {
        "incident_id": incident_id,
        "ambulance_id": ambulance_id,
        "failure_type": ftype.value,
        "severity": SEVERITY[ftype],
        "event_index": event_index,
        "event": event,
        "timestamp": timestamp,
        "description": description,
    }


def _to_frame(rows: List[Dict[str, Any]]) -> pd.DataFrame:
    return pd.DataFrame(rows, columns=FAILURE_COLUMNS)


def classify_invalid_transition(state: str, event: str) -> Tuple[FailureType, str]:
    """Explain WHY `event` is invalid when the DFA is in `state`."""
    if event not in EVENT_INDEX:
        return FailureType.UNKNOWN_EVENT, f"Unrecognised event '{event}'."

    expected_pos = STATE_INDEX[state]
    pos = EVENT_INDEX[event]
    expected = get_expected_event(state)

    if pos == expected_pos - 1:
        return FailureType.DUPLICATE_EVENT, f"'{event}' was logged twice in a row."
    if pos > expected_pos:
        skipped = ", ".join(EVENTS[expected_pos:pos])
        return FailureType.MISSING_EVENT, f"'{event}' logged before required event(s): {skipped}."
    if expected is None:
        return FailureType.OUT_OF_ORDER, f"'{event}' logged after the incident was already closed."
    return FailureType.OUT_OF_ORDER, f"'{event}' logged out of order; expected '{expected}'."


# --------------------------------------------------------------------------- #
# Per-incident validation summary
# --------------------------------------------------------------------------- #
def validate_incidents(df: pd.DataFrame) -> pd.DataFrame:
    """One row per incident with its DFA validation outcome."""
    rows = []
    for incident_id, g in df.groupby("incident_id", sort=False):
        res = validate_event_sequence(g["event"].tolist())
        bad = res.first_invalid
        rows.append({
            "incident_id": incident_id,
            "ambulance_id": g["ambulance_id"].iloc[0],
            "num_events": len(g),
            "is_valid": res.is_valid,
            "is_complete": res.is_complete,
            "accepted": res.accepted,
            "final_state": res.final_state,
            "first_invalid_index": bad.index if bad else None,
            "first_invalid_event": bad.symbol if bad else None,
            "expected_event": get_expected_event(res.final_state),
        })
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# Detectors
# --------------------------------------------------------------------------- #
def detect_sequence_failures(df: pd.DataFrame) -> pd.DataFrame:
    """DFA-based failures: first invalid transition, or incomplete lifecycle."""
    rows: List[Dict[str, Any]] = []
    for incident_id, g in df.groupby("incident_id", sort=False):
        amb = g["ambulance_id"].iloc[0]
        events = g["event"].tolist()
        res = validate_event_sequence(events)
        bad = res.first_invalid
        if bad is not None:
            ftype, desc = classify_invalid_transition(bad.from_state, bad.symbol)
            rows.append(_row(incident_id, amb, ftype, bad.index, bad.symbol, g["timestamp"].iloc[bad.index], desc))
        elif not res.is_complete:
            expected = get_expected_event(res.final_state)
            rows.append(_row(
                incident_id, amb, FailureType.INCOMPLETE_SEQUENCE, len(events) - 1, events[-1],
                g["timestamp"].iloc[-1],
                f"Sequence stopped in state {res.final_state}; expected '{expected}' next.",
            ))
    return _to_frame(rows)


def detect_timestamp_failures(df: pd.DataFrame, max_gap_min: float = DEFAULT_MAX_GAP_MIN) -> pd.DataFrame:
    """Timestamp regressions and abnormally long gaps between consecutive events."""
    rows: List[Dict[str, Any]] = []
    for incident_id, g in df.groupby("incident_id", sort=False):
        amb = g["ambulance_id"].iloc[0]
        ts = g["timestamp"].reset_index(drop=True)
        ev = g["event"].reset_index(drop=True)
        gaps = ts.diff().dt.total_seconds() / 60.0
        for i in range(1, len(g)):
            d = gaps.iloc[i]
            if pd.isna(d):
                continue
            if d < 0:
                rows.append(_row(
                    incident_id, amb, FailureType.TIMESTAMP_REGRESSION, i, ev.iloc[i], ts.iloc[i],
                    f"'{ev.iloc[i]}' is {abs(d):.1f} min earlier than the preceding event '{ev.iloc[i - 1]}'.",
                ))
            elif d > max_gap_min:
                rows.append(_row(
                    incident_id, amb, FailureType.EVENT_GAP_EXCEEDED, i, ev.iloc[i], ts.iloc[i],
                    f"{d:.1f} min between '{ev.iloc[i - 1]}' and '{ev.iloc[i]}' (limit {max_gap_min:g}).",
                ))
    return _to_frame(rows)


def detect_slow_responses(df: pd.DataFrame, max_response_min: float = DEFAULT_MAX_RESPONSE_MIN) -> pd.DataFrame:
    """Incidents whose call -> arrival time exceeds the threshold."""
    metrics = analytics.compute_incident_metrics(df)
    slow = analytics.flag_slow_responses(metrics, threshold_min=max_response_min)
    rows = [
        _row(
            r.incident_id, r.ambulance_id, FailureType.SLOW_RESPONSE, None, "ARRIVED_SCENE",
            r.arrived_scene_time,
            f"Response time {r.response_min:.1f} min exceeds the {max_response_min:g} min target.",
        )
        for r in slow.itertuples(index=False)
    ]
    return _to_frame(rows)


def detect_failures(
    df: pd.DataFrame,
    max_response_min: float = DEFAULT_MAX_RESPONSE_MIN,
    max_gap_min: float = DEFAULT_MAX_GAP_MIN,
) -> pd.DataFrame:
    """All failures, grouped by incident in original order."""
    parts = [
        detect_sequence_failures(df),
        detect_timestamp_failures(df, max_gap_min),
        detect_slow_responses(df, max_response_min),
    ]
    parts = [p for p in parts if not p.empty]
    if not parts:
        return _to_frame([])
    out = pd.concat(parts, ignore_index=True)
    order = {iid: i for i, iid in enumerate(df["incident_id"].drop_duplicates())}
    out["_o"] = out["incident_id"].map(order)
    return out.sort_values("_o", kind="stable").drop(columns="_o").reset_index(drop=True)


# --------------------------------------------------------------------------- #
# Summaries / report
# --------------------------------------------------------------------------- #
def summarize_failures(failures: pd.DataFrame) -> pd.DataFrame:
    """Count of failures by type and severity, most frequent first."""
    if failures.empty:
        return pd.DataFrame(columns=["failure_type", "severity", "count"])
    return (
        failures.groupby(["failure_type", "severity"]).size().reset_index(name="count")
        .sort_values("count", ascending=False, kind="stable").reset_index(drop=True)
    )


def failure_rate(df: pd.DataFrame, failures: pd.DataFrame) -> float:
    """Fraction of incidents with at least one failure (0.0 – 1.0)."""
    total = df["incident_id"].nunique()
    return round(failures["incident_id"].nunique() / total, 4) if total else 0.0


def generate_report(
    df: pd.DataFrame,
    max_response_min: float = DEFAULT_MAX_RESPONSE_MIN,
    max_gap_min: float = DEFAULT_MAX_GAP_MIN,
) -> Dict[str, Any]:
    """Single entry point for the frontend: everything in one dict."""
    failures = detect_failures(df, max_response_min, max_gap_min)
    bundle = analytics.compute_all_analytics(df, max_response_min)
    return {
        "total_incidents": int(df["incident_id"].nunique()),
        "failure_rate": failure_rate(df, failures),
        "validation": validate_incidents(df),
        "failures": failures,
        "failure_summary": summarize_failures(failures),
        **bundle,
    }


# --------------------------------------------------------------------------- #
# CLI demo:  python -m analysis.failure_detection [path/to/events.csv]
# --------------------------------------------------------------------------- #
def main(argv: Optional[List[str]] = None) -> None:
    argv = sys.argv[1:] if argv is None else argv
    df = analytics.load_events(argv[0]) if argv else analytics.load_events()
    report = generate_report(df)
    pd.set_option("display.width", 200, "display.max_columns", 20, "display.max_colwidth", 70)
    print(f"Incidents: {report['total_incidents']}   Failure rate: {report['failure_rate']:.0%}\n")
    print("== DFA validation ==");        print(report["validation"].to_string(index=False))
    print("\n== Failures ==");             print(report["failures"][["incident_id", "failure_type", "severity", "description"]].to_string(index=False))
    print("\n== Failure summary ==");      print(report["failure_summary"].to_string(index=False))
    print("\n== Response-time summary (min) =="); print(report["response_summary"])
    print("\n== By ambulance ==");         print(report["ambulance_performance"].to_string(index=False))


if __name__ == "__main__":
    main()
