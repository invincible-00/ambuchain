# AmbuChain – Frontend (Streamlit)

Dark-navy / cyan dashboard for ambulance workflow analysis.
Works **today on mock data**, and switches to the real backend with one sidebar toggle.

```
app.py                     # entry point (page config + dashboard.run())
frontend/
  dashboard.py             # sidebar, upload, and the 5 pages
  visualizations.py        # pure Plotly figure builders (no Streamlit, no analysis logic)
  engines.py               # the ONLY bridge to analysis logic: MockEngine / BackendEngine
  mock_data.py             # mock events + canned results (no DFA logic)
  style.css                # theme / responsive styling
.streamlit/config.toml     # dark theme
```

## Run
```bash
pip install -r requirements.txt
streamlit run app.py          # from the project root
```
Pages: Overview · Incident Explorer · DFA Visualisation · Failure Detection · Response Analytics.
CSV upload (columns `incident_id, ambulance_id, event, timestamp`) and the detection thresholds live in the sidebar.

## Connecting the backend
Put `automata/` and `analysis/` next to `app.py` (this repo layout already does). The sidebar then offers
**Live backend**. `BackendEngine` in `frontend/engines.py` only *calls* the backend – no second DFA exists:

| Frontend contract | Backend call |
|---|---|
| `process_sequence(events)` | `automata.dfa.validate_event_sequence(events).to_dict()` |
| `detect_failures(df, max_response_min, max_gap_min)` | `analysis.failure_detection.detect_failures` |
| `calculate_response_times(df)` | `analysis.analytics.compute_incident_metrics` |
| `get_dfa_definition()` | `automata.dfa.get_dfa_definition()` |

If the backend's function names/shapes change, edit **only** `BackendEngine`. To make the backend the default,
change `options` order in `render_sidebar()` or call `get_engine("backend")` directly.

## Mock mode notes
* The mock sample (40 incidents) was checked against the real backend: sequence results, failures and metrics match.
* Uploaded CSVs in mock mode get placeholder sequence results (the mock can't analyse unseen data). The UI says so.
