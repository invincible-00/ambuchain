"""
analysis/analytics.py
=====================
Data loading and response-time calculations for AmbuChain.

Event table schema (one row per logged event, in LOG ORDER):

    incident_id | ambulance_id | event | timestamp

All durations are returned in MINUTES.  A duration that cannot be computed
(missing event, bad timestamp) or is negative (timestamp regression) is NaN,
so corrupt records never skew the statistics.  Regressions themselves are
reported by `analysis.failure_detection`.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional, Union

import pandas as pd

from automata.dfa import EVENTS, normalize_event

REQUIRED_COLUMNS = ["incident_id", "ambulance_id", "event", "timestamp"]
DEFAULT_DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "sample_data.csv"
DEFAULT_RESPONSE_THRESHOLD_MIN = 10.0

METRIC_COLUMNS = ["dispatch_min", "travel_min", "response_min", "scene_min", "transport_min", "total_min"]


# --------------------------------------------------------------------------- #
# Loading
# --------------------------------------------------------------------------- #
def prepare_events(df: pd.DataFrame) -> pd.DataFrame:
    """Validate columns and normalise types.  Row (log) order is preserved."""
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required column(s): {missing}")
    out = df[REQUIRED_COLUMNS].copy()
    out["incident_id"] = out["incident_id"].astype(str).str.strip()
    out["ambulance_id"] = out["ambulance_id"].astype(str).str.strip()
    out["event"] = out["event"].map(normalize_event)
    out["timestamp"] = pd.to_datetime(out["timestamp"], errors="coerce")
    return out.reset_index(drop=True)


def load_events(path: Union[str, Path] = DEFAULT_DATA_PATH) -> pd.DataFrame:
    """Read an events CSV and return a clean events DataFrame."""
    return prepare_events(pd.read_csv(path))


# --------------------------------------------------------------------------- #
# Metrics
# --------------------------------------------------------------------------- #
def pivot_event_times(df: pd.DataFrame) -> pd.DataFrame:
    """One row per incident, one timestamp column per lifecycle event.

    If an event repeats, its first logged occurrence is used.  Unknown events
    are ignored here.  Index = incident_id (original order kept).
    """
    amb = df.groupby("incident_id", sort=False)["ambulance_id"].first()
    wide = (
        df.groupby(["incident_id", "event"], sort=False)["timestamp"]
        .first()
        .unstack("event")
        .reindex(index=amb.index, columns=list(EVENTS))
    )
    for e in EVENTS:
        wide[e] = pd.to_datetime(wide[e])
    wide.insert(0, "ambulance_id", amb)
    return wide


def _minutes_between(wide: pd.DataFrame, start: str, end: str) -> pd.Series:
    delta = (wide[end] - wide[start]).dt.total_seconds() / 60.0
    return delta.where(delta >= 0).round(2)  # negatives -> NaN


def compute_incident_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """Per-incident timing metrics (minutes).

    dispatch_min   CALL_RECEIVED      -> AMBULANCE_DISPATCHED
    travel_min     EN_ROUTE           -> ARRIVED_SCENE
    response_min   CALL_RECEIVED      -> ARRIVED_SCENE   (headline response time)
    scene_min      ARRIVED_SCENE      -> DEPARTED_SCENE
    transport_min  DEPARTED_SCENE     -> ARRIVED_HOSPITAL
    total_min      CALL_RECEIVED      -> HANDOVER_COMPLETE
    """
    wide = pivot_event_times(df)
    metrics = pd.DataFrame(index=wide.index)
    metrics["ambulance_id"] = wide["ambulance_id"]
    metrics["call_time"] = wide["CALL_RECEIVED"]
    metrics["arrived_scene_time"] = wide["ARRIVED_SCENE"]
    metrics["dispatch_min"] = _minutes_between(wide, "CALL_RECEIVED", "AMBULANCE_DISPATCHED")
    metrics["travel_min"] = _minutes_between(wide, "EN_ROUTE", "ARRIVED_SCENE")
    metrics["response_min"] = _minutes_between(wide, "CALL_RECEIVED", "ARRIVED_SCENE")
    metrics["scene_min"] = _minutes_between(wide, "ARRIVED_SCENE", "DEPARTED_SCENE")
    metrics["transport_min"] = _minutes_between(wide, "DEPARTED_SCENE", "ARRIVED_HOSPITAL")
    metrics["total_min"] = _minutes_between(wide, "CALL_RECEIVED", "HANDOVER_COMPLETE")
    return metrics.reset_index()


def summarize_response_times(metrics: pd.DataFrame, column: str = "response_min") -> Dict[str, Optional[float]]:
    """Aggregate stats for one metric column (NaN values excluded)."""
    s = metrics[column].dropna()
    if s.empty:
        return {k: None for k in ("mean", "median", "p90", "min", "max", "std")} | {"count": 0}
    return {
        "count": int(s.count()),
        "mean": round(float(s.mean()), 2),
        "median": round(float(s.median()), 2),
        "p90": round(float(s.quantile(0.9)), 2),
        "min": round(float(s.min()), 2),
        "max": round(float(s.max()), 2),
        "std": round(float(s.std(ddof=0)), 2),
    }


def response_time_by_ambulance(metrics: pd.DataFrame, column: str = "response_min") -> pd.DataFrame:
    """Per-ambulance performance table, slowest average first."""
    out = (
        metrics.groupby("ambulance_id")[column]
        .agg(measured_incidents="count", mean="mean", median="median", min="min", max="max")
        .round(2)
        .reset_index()
        .sort_values("mean", ascending=False, na_position="last")
        .reset_index(drop=True)
    )
    incidents = metrics.groupby("ambulance_id").size().rename("total_incidents").reset_index()
    return out.merge(incidents, on="ambulance_id")


def flag_slow_responses(
    metrics: pd.DataFrame,
    threshold_min: float = DEFAULT_RESPONSE_THRESHOLD_MIN,
    column: str = "response_min",
) -> pd.DataFrame:
    """Incidents whose response time is strictly above the threshold."""
    slow = metrics[metrics[column] > threshold_min].copy()
    slow["threshold_min"] = threshold_min
    slow["excess_min"] = (slow[column] - threshold_min).round(2)
    return slow.reset_index(drop=True)


def compute_all_analytics(df: pd.DataFrame, threshold_min: float = DEFAULT_RESPONSE_THRESHOLD_MIN) -> Dict[str, Any]:
    """Convenience bundle: metrics + summary + ambulance table + slow cases."""
    metrics = compute_incident_metrics(df)
    return {
        "metrics": metrics,
        "response_summary": summarize_response_times(metrics),
        "ambulance_performance": response_time_by_ambulance(metrics),
        "slow_responses": flag_slow_responses(metrics, threshold_min),
    }
