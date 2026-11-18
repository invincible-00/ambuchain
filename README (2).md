# AmbuChain Backend

Custom DFA validation, failure detection and response-time analytics for ambulance event logs. Python 3.9+ and Pandas only; no UI.

## Structure

```
ambuchain/
├── automata/dfa.py                 # DFA engine + ambulance DFA
├── analysis/failure_detection.py   # failure detectors + report
├── analysis/analytics.py           # loading + response-time metrics
├── data/sample_data.csv            # 12 incidents (95 events), mixed good/bad
└── tests/test_backend.py           # 28 unit/integration tests
```

## Run

```bash
pip install pandas                                   # only dependency
python -m unittest discover -s tests -v              # run tests (from project root)
python -m analysis.failure_detection                 # demo report on sample data
python -m analysis.failure_detection my_events.csv   # demo on your own CSV
```

Always run from the project root so `automata` and `analysis` resolve as packages.

## Mathematical definition

`M = (Q, Σ, δ, q0, F)`

- **Q** = {IDLE, CALL_LOGGED, UNIT_ASSIGNED, UNIT_MOVING, ON_SCENE, PATIENT_ABOARD, TRANSPORTING, AT_HOSPITAL, CLOSED}
- **Σ** = {CALL_RECEIVED, AMBULANCE_DISPATCHED, EN_ROUTE, ARRIVED_SCENE, PATIENT_LOADED, DEPARTED_SCENE, ARRIVED_HOSPITAL, HANDOVER_COMPLETE}
- **q0** = IDLE, **F** = {CLOSED}
- **δ**:

| State | Event | Next state |
|---|---|---|
| IDLE | CALL_RECEIVED | CALL_LOGGED |
| CALL_LOGGED | AMBULANCE_DISPATCHED | UNIT_ASSIGNED |
| UNIT_ASSIGNED | EN_ROUTE | UNIT_MOVING |
| UNIT_MOVING | ARRIVED_SCENE | ON_SCENE |
| ON_SCENE | PATIENT_LOADED | PATIENT_ABOARD |
| PATIENT_ABOARD | DEPARTED_SCENE | TRANSPORTING |
| TRANSPORTING | ARRIVED_HOSPITAL | AT_HOSPITAL |
| AT_HOSPITAL | HANDOVER_COMPLETE | CLOSED |

Every other (state, event) pair is an **invalid transition** (implicit trap state; `DFA.complete()` makes it explicit).
A sequence is *valid* (no invalid transition), *complete* (ends in F), and *accepted* (both).

## Data format

CSV columns: `incident_id, ambulance_id, event, timestamp`. Rows are checked in **log order** (not re-sorted by timestamp) so timestamp errors stay visible.

## Interface for the frontend

```python
from automata.dfa import (validate_event_sequence, is_valid_sequence,
                          find_invalid_transitions, get_dfa_definition)
from analysis.analytics import (load_events, compute_incident_metrics,
                                summarize_response_times, response_time_by_ambulance,
                                flag_slow_responses)
from analysis.failure_detection import (validate_incidents, detect_failures,
                                        summarize_failures, failure_rate, generate_report)
```

| Function | Returns |
|---|---|
| `get_dfa_definition()` | dict with `Q, Sigma, q0, F, delta` for display |
| `validate_event_sequence(events, stop_on_error=True)` | `ValidationResult` (`.accepted`, `.is_valid`, `.is_complete`, `.final_state`, `.trace`, `.first_invalid`, `.to_dict()`) |
| `is_valid_sequence(events)` | `bool` |
| `find_invalid_transitions(events)` | list of dicts (all violations, recovery mode) |
| `load_events(path=sample)` | clean events DataFrame |
| `validate_incidents(df)` | one row per incident: validity, final state, first invalid event, expected event |
| `detect_failures(df, max_response_min=10, max_gap_min=30)` | failures DataFrame: `incident_id, ambulance_id, failure_type, severity, event_index, event, timestamp, description` |
| `compute_incident_metrics(df)` | per-incident minutes: `dispatch_min, travel_min, response_min, scene_min, transport_min, total_min` |
| `summarize_response_times(metrics, column)` | dict: count, mean, median, p90, min, max, std |
| `response_time_by_ambulance(metrics)` | per-ambulance table |
| `flag_slow_responses(metrics, threshold_min)` | incidents above threshold |
| `generate_report(df)` | **one-call dict** with all of the above (`validation`, `failures`, `failure_summary`, `failure_rate`, `metrics`, `response_summary`, `ambulance_performance`, `slow_responses`) |

Minimal usage:

```python
df = load_events("data/sample_data.csv")
report = generate_report(df)
report["failures"]          # DataFrame
report["response_summary"]  # dict
```

## Failure types

`MISSING_EVENT`, `DUPLICATE_EVENT`, `OUT_OF_ORDER`, `UNKNOWN_EVENT`, `INCOMPLETE_SEQUENCE`, `TIMESTAMP_REGRESSION`, `EVENT_GAP_EXCEEDED`, `SLOW_RESPONSE`, each with a `HIGH/MEDIUM/LOW` severity.

## Notes

- Response time = `CALL_RECEIVED → ARRIVED_SCENE`; slow means strictly above the threshold.
- Negative or uncomputable durations are `NaN` and excluded from statistics.
- Sequence failures report the **first** violation per incident (standard DFA behaviour). Use `find_invalid_transitions` for all violations; one skipped event may cascade into several.
- Repeated events use the first logged occurrence for metrics.
