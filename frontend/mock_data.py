"""
frontend/mock_data.py
=====================
Mock data so the UI can be built and demoed without the backend.

IMPORTANT: this is NOT a DFA.  The events are generated from known scenarios
(clean, skipped event, duplicate ...), so the "expected" validation result of
each generated sequence is known *by construction* and stored in a lookup table
keyed by the event sequence.  Nothing here walks transitions.
"""

from __future__ import annotations

import random
from datetime import datetime, timedelta
from typing import Any, Dict, List, Tuple

import pandas as pd

STATES: Tuple[str, ...] = (
    "IDLE", "CALL_LOGGED", "UNIT_ASSIGNED", "UNIT_MOVING", "ON_SCENE",
    "PATIENT_ABOARD", "TRANSPORTING", "AT_HOSPITAL", "CLOSED",
)
EVENTS: Tuple[str, ...] = (
    "CALL_RECEIVED", "AMBULANCE_DISPATCHED", "EN_ROUTE", "ARRIVED_SCENE",
    "PATIENT_LOADED", "DEPARTED_SCENE", "ARRIVED_HOSPITAL", "HANDOVER_COMPLETE",
)

# Static, display-only copy of the DFA definition (same shape as the backend's
# `get_dfa_definition()`), used when the backend is not connected.
MOCK_DFA_DEFINITION: Dict[str, Any] = {
    "Q": list(STATES),
    "Sigma": list(EVENTS),
    "q0": STATES[0],
    "F": [STATES[-1]],
    "delta": [{"from": STATES[i], "event": EVENTS[i], "to": STATES[i + 1]} for i in range(len(EVENTS))],
    "note": "Any (state, event) pair not in delta is an invalid transition (implicit trap state).",
}

UNDEFINED_TRANSITION = "UNDEFINED_TRANSITION"
UNKNOWN_SYMBOL = "UNKNOWN_SYMBOL"


# --------------------------------------------------------------------------- #
# Canned results, built from the known shape of each scenario
# --------------------------------------------------------------------------- #
def _result(seq: List[str], prefix: int, invalid: Dict[str, Any] | None = None) -> Dict[str, Any]:
    """Result dict (same keys as backend `ValidationResult.to_dict()`).

    `prefix` = how many leading events are known to be valid for this scenario.
    """
    trace = [
        {"index": j, "from_state": STATES[j], "symbol": seq[j], "to_state": STATES[j + 1], "valid": True, "reason": None}
        for j in range(prefix)
    ]
    if invalid:
        trace.append(invalid)
    return {
        "start_state": STATES[0],
        "final_state": STATES[prefix],
        "is_valid": invalid is None,
        "is_complete": prefix == len(EVENTS) and invalid is None,
        "accepted": invalid is None and prefix == len(EVENTS),
        "halted": invalid is not None,
        "trace": trace,
        "invalid_transitions": [invalid] if invalid else [],
    }


def _bad(index: int, state: int, symbol: str, reason: str = UNDEFINED_TRANSITION) -> Dict[str, Any]:
    return {"index": index, "from_state": STATES[state], "symbol": symbol, "to_state": None, "valid": False, "reason": reason}


def _pretty(names) -> str:
    return ", ".join(names)


# --------------------------------------------------------------------------- #
# Scenario builders: each returns (sequence of [event, ts], result, failure|None)
# failure = (failure_type, event_index, description)
# --------------------------------------------------------------------------- #
def _ideal(rng: random.Random, start: datetime, slow: bool = False) -> List[list]:
    gaps = [0, rng.randint(1, 3), 1, rng.randint(3, 6), rng.randint(6, 12), 1, rng.randint(8, 16), rng.randint(5, 9)]
    if slow:
        gaps[1], gaps[3] = rng.randint(4, 8), rng.randint(7, 10)
    t, out = start, []
    for name, g in zip(EVENTS, gaps):
        t += timedelta(minutes=g)
        out.append([name, t])
    return out


def _shift(seq: List[list], from_idx: int, minutes: int) -> None:
    for item in seq[from_idx:]:
        item[1] += timedelta(minutes=minutes)


def _scenario(kind: str, rng: random.Random, start: datetime):
    seq = _ideal(rng, start, slow=(kind == "slow"))
    failure = None

    if kind == "missing":
        m = rng.randint(2, 6)                                   # index of the dropped event
        dropped = seq.pop(m)[0]
        names = [e for e, _ in seq]
        res = _result(names, m, _bad(m, m, names[m]))
        failure = ("MISSING_EVENT", m, f"'{names[m]}' logged before required event(s): {dropped}.")
    elif kind == "duplicate":
        d = rng.randint(1, 6)
        seq.insert(d + 1, [seq[d][0], seq[d][1] + timedelta(minutes=1)])
        _shift(seq, d + 2, 1)
        names = [e for e, _ in seq]
        res = _result(names, d + 1, _bad(d + 1, d + 1, names[d + 1]))
        failure = ("DUPLICATE_EVENT", d + 1, f"'{names[d + 1]}' was logged twice in a row.")
    elif kind == "out_of_order":
        seq.insert(6, [EVENTS[4], seq[5][1] + timedelta(minutes=1)])        # PATIENT_LOADED after DEPARTED_SCENE
        _shift(seq, 7, 1)
        names = [e for e, _ in seq]
        res = _result(names, 6, _bad(6, 6, EVENTS[4]))
        failure = ("OUT_OF_ORDER", 6, f"'{EVENTS[4]}' logged out of order; expected '{EVENTS[6]}'.")
    elif kind == "unknown":
        u = rng.randint(2, 5)
        seq.insert(u, ["SIREN_FAULT", seq[u - 1][1] + timedelta(minutes=1)])
        _shift(seq, u + 1, 1)
        names = [e for e, _ in seq]
        res = _result(names, u, _bad(u, u, "SIREN_FAULT", UNKNOWN_SYMBOL))
        failure = ("UNKNOWN_EVENT", u, "Unrecognised event 'SIREN_FAULT'.")
    elif kind == "incomplete":
        c = rng.randint(3, 7)
        seq = seq[:c]
        names = [e for e, _ in seq]
        res = _result(names, c)
        failure = ("INCOMPLETE_SEQUENCE", c - 1, f"Sequence stopped in state {STATES[c]}; expected '{EVENTS[c]}' next.")
    else:                                                         # clean / slow / regression / gap
        if kind == "regression":
            seq[3][1] = seq[2][1] - timedelta(minutes=rng.randint(2, 6))
        elif kind == "gap":
            _shift(seq, 4, rng.randint(35, 50))
        names = [e for e, _ in seq]
        res = _result(names, len(EVENTS))
    return seq, res, failure


_MIX = (["clean"] * 22 + ["slow"] * 5 + ["missing"] * 3 + ["duplicate"] * 2 + ["out_of_order"] * 2
        + ["unknown"] * 2 + ["incomplete"] * 2 + ["regression"] + ["gap"])


def build_mock(seed: int = 42):
    """Returns (events_df, results_by_sequence, seq_failures_by_sequence)."""
    rng = random.Random(seed)
    kinds = _MIX[:]
    rng.shuffle(kinds)
    base = datetime(2026, 11, 17, 6, 0)
    rows, results, seq_failures = [], {}, {}
    for i, kind in enumerate(kinds, start=1):
        start = base + timedelta(minutes=i * 28 + rng.randint(0, 12))
        seq, res, failure = _scenario(kind, rng, start)
        iid, amb = f"INC{i:03d}", f"AMB{rng.randint(1, 6):02d}"
        for name, ts in seq:
            rows.append({"incident_id": iid, "ambulance_id": amb, "event": name, "timestamp": ts})
        key = tuple(e for e, _ in seq)
        results[key] = res
        if failure:
            seq_failures[key] = failure
    return pd.DataFrame(rows), results, seq_failures


# Presets for the DFA playground (registered so the mock engine can answer them)
def build_presets() -> Dict[str, Tuple[List[str], Dict[str, Any]]]:
    full = list(EVENTS)
    skip = full[:2] + full[3:]
    dup = full[:3] + [full[2]] + full[3:]
    unk = full[:2] + ["SIREN_FAULT"] + full[2:]
    return {
        "Valid full lifecycle": (full, _result(full, 8)),
        "Stops after ARRIVED_SCENE (incomplete)": (full[:4], _result(full[:4], 4)),
        "Skipped EN_ROUTE": (skip, _result(skip, 2, _bad(2, 2, skip[2]))),
        "Duplicate EN_ROUTE": (dup, _result(dup, 3, _bad(3, 3, "EN_ROUTE"))),
        "Unknown event SIREN_FAULT": (unk, _result(unk, 2, _bad(2, 2, "SIREN_FAULT", UNKNOWN_SYMBOL))),
    }
