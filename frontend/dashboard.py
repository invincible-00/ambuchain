"""
frontend/dashboard.py
=====================
Streamlit pages for AmbuChain.  All analysis comes from `frontend.engines`
(mock now, real backend later); all charts come from `frontend.visualizations`.

Entry point: `run()` (called by app.py).
"""

from __future__ import annotations

import html
import io
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
import streamlit as st

from frontend import engines, mock_data
from frontend import visualizations as viz

CSS_PATH = Path(__file__).with_name("style.css")
PAGES = {
    "Overview": "◎",
    "Incident Explorer": "⌕",
    "DFA Visualisation": "⬡",
    "Failure Detection": "⚠",
    "Response Analytics": "⏱",
}
SEV_COLOR = viz.SEVERITY_COLORS
PLOT_CFG = {"displayModeBar": False}


# =========================================================================== #
# Small HTML helpers
# =========================================================================== #
def esc(x: Any) -> str:
    return html.escape(str(x))


def inject_css() -> None:
    try:
        st.markdown(f"<style>{CSS_PATH.read_text(encoding='utf-8')}</style>", unsafe_allow_html=True)
    except OSError:
        pass  # styling is optional


def hero(title_plain: str, title_accent: str, subtitle: str, ctx: Dict[str, Any]) -> None:
    eng = ctx["engine"]
    chip = (f"<span class='chip {'mock' if eng.is_mock else 'live'}'><span class='dot'></span>{esc(eng.label)}</span>")
    src = f"<span class='chip'>Source <b>{esc(ctx['source_label'])}</b></span>"
    st.markdown(
        f"<div class='hero'><div><h1>{esc(title_plain)} <span>{esc(title_accent)}</span></h1><p>{esc(subtitle)}</p></div>"
        f"<div style='display:flex;gap:.5rem;flex-wrap:wrap'>{src}{chip}</div></div>", unsafe_allow_html=True)


def section(title: str) -> None:
    st.markdown(f"<div class='section-title'>{esc(title)}</div>", unsafe_allow_html=True)


def kpi(label: str, value: Any, sub: str = "", tone: str = "", unit: str = "", text: bool = False) -> str:
    u = f"<small>{esc(unit)}</small>" if unit else ""
    return (f"<div class='kpi {tone}{' text' if text else ''}'><div class='label'>{esc(label)}</div>"
            f"<div class='value'>{esc(value)}{u}</div><div class='sub'>{esc(sub)}&nbsp;</div></div>")


def kpi_row(cards: List[str]) -> None:
    for col, card in zip(st.columns(len(cards)), cards):
        col.markdown(card, unsafe_allow_html=True)


def badge(text: str, kind: str) -> str:
    return f"<span class='badge {kind}'>{esc(text)}</span>"


def empty_state(msg: str) -> None:
    st.markdown(f"<div class='empty'>{esc(msg)}</div>", unsafe_allow_html=True)


def fmt(v: Any, nd: int = 1) -> str:
    return "–" if v is None or (isinstance(v, float) and pd.isna(v)) else f"{v:.{nd}f}"


def stepper(states: List[str], result: Dict[str, Any], q0: str) -> str:
    path = [q0] + [t["to_state"] for t in result["trace"] if t["valid"]]
    final = result["final_state"]
    cls_final = "final" if result["accepted"] else ("stop" if result["halted"] else "pause")
    parts = []
    for s in states:
        cls = cls_final if s == final else ("done" if s in path else "")
        parts.append(f"<span class='step {cls}'>{esc(s.replace('_', ' '))}</span>")
    return "<div class='stepper'>" + "<span class='arrow'>›</span>".join(parts) + "</div>"


# =========================================================================== #
# Data loading & analysis
# =========================================================================== #
@st.cache_data(show_spinner=False)
def parse_csv(raw: bytes) -> tuple:
    """Parse an uploaded CSV. Returns (DataFrame | None, error | None)."""
    try:
        df = pd.read_csv(io.BytesIO(raw))
    except Exception as exc:  # malformed file
        return None, f"Could not read the file as CSV ({exc})."
    df.columns = [str(c).strip().lower() for c in df.columns]
    missing = [c for c in engines.REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        return None, f"Missing required column(s): {', '.join(missing)}. Expected: {', '.join(engines.REQUIRED_COLUMNS)}."
    if df.empty:
        return None, "The file has a header but no rows."
    return df, None


@st.cache_data(show_spinner="Analysing incidents…")
def run_analysis(events: pd.DataFrame, engine_name: str, max_response: float, max_gap: float) -> Dict[str, Any]:
    """Call the engine contract once and assemble everything the pages need."""
    eng = engines.get_engine(engine_name)
    failures = eng.detect_failures(events, max_response, max_gap)
    metrics = eng.calculate_response_times(events)

    results: Dict[str, Dict[str, Any]] = {}
    rows = []
    for iid, g in events.groupby("incident_id", sort=False):
        res = eng.process_sequence(g["event"].tolist())
        results[iid] = res
        bad = res["invalid_transitions"][0] if res["invalid_transitions"] else None
        rows.append({
            "incident_id": iid, "ambulance_id": g["ambulance_id"].iloc[0], "num_events": len(g),
            "accepted": res["accepted"], "is_valid": res["is_valid"], "is_complete": res["is_complete"],
            "final_state": res["final_state"],
            "first_invalid_event": bad["symbol"] if bad else None,
            "start_time": g["timestamp"].min(),
        })
    validation = pd.DataFrame(rows)

    per_inc = failures.groupby("incident_id")["failure_type"].agg(lambda s: ", ".join(dict.fromkeys(s)))
    counts = failures.groupby("incident_id").size()
    validation["failure_count"] = validation["incident_id"].map(counts).fillna(0).astype(int)
    validation["failure_types"] = validation["incident_id"].map(per_inc).fillna("")
    validation["status"] = validation["failure_count"].map(lambda n: "Failed" if n else "Successful")
    validation = validation.merge(metrics[["incident_id", "response_min", "total_min"]], on="incident_id", how="left")

    summary = (failures.groupby(["failure_type", "severity"]).size().reset_index(name="count")
               .sort_values("count", ascending=False) if not failures.empty
               else pd.DataFrame(columns=["failure_type", "severity", "count"]))

    amb = (metrics.groupby("ambulance_id")["response_min"]
           .agg(measured="count", mean="mean", median="median", max="max").round(2).reset_index())
    amb["incidents"] = amb["ambulance_id"].map(metrics.groupby("ambulance_id").size())
    fail_by_amb = validation.groupby("ambulance_id")["failure_count"].apply(lambda s: int((s > 0).sum()))
    amb["failed_incidents"] = amb["ambulance_id"].map(fail_by_amb).fillna(0).astype(int)

    return {"events": events, "validation": validation, "failures": failures, "metrics": metrics,
            "summary": summary, "ambulances": amb, "results": results,
            "dfa": eng.get_dfa_definition()}


def response_stats(metrics: pd.DataFrame, threshold: float) -> Dict[str, Optional[float]]:
    s = metrics["response_min"].dropna()
    if s.empty:
        return dict(count=0, mean=None, median=None, p90=None, max=None, within=None)
    return dict(count=int(s.size), mean=s.mean(), median=s.median(), p90=s.quantile(0.9), max=s.max(),
                within=float((s <= threshold).mean() * 100))


# =========================================================================== #
# Sidebar
# =========================================================================== #
def render_sidebar() -> Dict[str, Any]:
    with st.sidebar:
        st.markdown("<div class='brand'><div class='logo'>🚑</div><div><div class='name'>Ambu<span>Chain</span></div>"
                    "<div class='tag'>Emergency workflow analysis</div></div></div>", unsafe_allow_html=True)

        st.markdown("<div class='side-label'>Navigate</div>", unsafe_allow_html=True)
        page = st.radio("Page", list(PAGES), format_func=lambda p: f"{PAGES[p]}   {p}", label_visibility="collapsed",
                        key="page")

        st.markdown("<div class='side-label'>Data engine</div>", unsafe_allow_html=True)
        backend_ok = engines.backend_available()
        options = ["mock", "backend"] if backend_ok else ["mock"]
        engine_name = st.radio("Engine", options, label_visibility="collapsed", key="engine_choice",
                               format_func=lambda n: engines.get_engine(n).label)
        if not backend_ok:
            st.caption("Backend not found. Add `automata/` and `analysis/` next to `app.py` to enable it.")

        st.markdown("<div class='side-label'>Upload events (CSV)</div>", unsafe_allow_html=True)
        up = st.file_uploader("CSV", type=["csv"], label_visibility="collapsed", key="upload")
        template = "incident_id,ambulance_id,event,timestamp\nINC001,AMB01,CALL_RECEIVED,2026-11-10 08:00:00\n"
        st.download_button("Download CSV template", template, "events_template.csv", "text/csv", use_container_width=True)

        with st.expander("Detection thresholds"):
            max_response = st.slider("Response target (min)", 3.0, 30.0, 10.0, 0.5, key="max_response",
                                     help="Call → arrival on scene. Strictly above this is flagged SLOW_RESPONSE.")
            max_gap = st.slider("Max gap between events (min)", 5.0, 90.0, 30.0, 1.0, key="max_gap")

    eng = engines.get_engine(engine_name)
    raw, source = None, "sample data"
    if up is not None:
        raw, err = parse_csv(up.getvalue())
        if err:
            st.sidebar.error(err)
            raw = None
        else:
            source = up.name
    if raw is None:
        raw = eng.load_sample()
    return dict(page=page, engine=eng, engine_name=engine_name, raw=raw, source_label=source,
                uploaded=up is not None and source == up.name, max_response=max_response, max_gap=max_gap)


def load_analysis(ctx: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    eng = ctx["engine"]
    try:
        events = eng.prepare_events(ctx["raw"])
    except ValueError as exc:
        st.error(f"Could not prepare the data: {exc}")
        return None
    bad_ts = int(events["timestamp"].isna().sum())
    if bad_ts:
        st.warning(f"{bad_ts} row(s) have an unreadable timestamp; they are kept but excluded from timing metrics.")
    if eng.is_mock and ctx["uploaded"]:
        st.info("**Mock engine:** uploaded data is shown with placeholder sequence results and no failure "
                "detection for sequences it doesn't recognise. Switch *Data engine* to **Live backend** for real analysis.")
    return run_analysis(events, ctx["engine_name"], ctx["max_response"], ctx["max_gap"])


# =========================================================================== #
# Pages
# =========================================================================== #
def page_overview(ctx: Dict[str, Any], a: Dict[str, Any]) -> None:
    v, f, m = a["validation"], a["failures"], a["metrics"]
    total = len(v)
    ok, bad = int((v["status"] == "Successful").sum()), int((v["status"] == "Failed").sum())
    accepted = int(v["accepted"].sum())
    high = int((f["severity"] == "HIGH").sum()) if not f.empty else 0
    st_ = response_stats(m, ctx["max_response"])

    hero("Operations", "overview", "Is every ambulance incident following the correct emergency-response lifecycle?", ctx)
    kpi_row([
        kpi("Total incidents", total, f"{len(a['events'])} events logged"),
        kpi("Successful", ok, f"{ok / total:.0%} · no failures" if total else "", "good"),
        kpi("Failed", bad, f"{high} high-severity", "bad" if bad else "good"),
        kpi("DFA-accepted", accepted, "valid & complete sequence", ""),
        kpi("Median response", fmt(st_["median"]), f"p90 {fmt(st_['p90'])} min · target {ctx['max_response']:g}", "warn", "min"),
    ])

    section("At a glance")
    c1, c2, c3 = st.columns([1, 1.15, 1.25])
    c1.plotly_chart(viz.workflow_donut(ok, bad), config=PLOT_CFG, theme=None)
    c2.plotly_chart(viz.failure_type_bar(a["summary"]), config=PLOT_CFG, theme=None)
    c3.plotly_chart(viz.response_histogram(m, ctx["max_response"]), config=PLOT_CFG, theme=None)

    section("Workflow progression & incidents needing attention")
    c1, c2 = st.columns([1.5, 1])
    c1.plotly_chart(viz.final_state_bar(v, a["dfa"]["Q"]), config=PLOT_CFG, theme=None)
    with c2:
        if f.empty:
            empty_state("No failures detected. Every workflow looks healthy.")
        else:
            order = {"HIGH": 0, "MEDIUM": 1, "LOW": 2}
            top = f.assign(_o=f["severity"].map(order)).sort_values("_o", kind="stable").head(6)
            items = "".join(
                f"<div class='fail-item' style='--sev:{SEV_COLOR[r.severity]}'>"
                f"<div class='t'>{esc(r.incident_id)} · {esc(r.failure_type.replace('_', ' ').title())} "
                f"{badge(r.severity.title(), 'fail' if r.severity == 'HIGH' else ('warn' if r.severity == 'MEDIUM' else 'info'))}</div>"
                f"<div class='d'>{esc(r.description)}</div></div>" for r in top.itertuples())
            st.markdown(items, unsafe_allow_html=True)
            st.caption("Open **Failure Detection** for the full list.")


def page_explorer(ctx: Dict[str, Any], a: Dict[str, Any]) -> None:
    hero("Incident", "explorer", "Search incidents, then drill into one to see its event timeline and DFA path.", ctx)
    v = a["validation"]

    c1, c2, c3, c4 = st.columns([1.2, 1, 1, 1.3])
    q = c1.text_input("Search", placeholder="Incident or ambulance ID…", label_visibility="collapsed")
    status = c2.multiselect("Status", ["Successful", "Failed"], placeholder="Status", label_visibility="collapsed")
    ambs = c3.multiselect("Ambulance", sorted(v["ambulance_id"].unique()), placeholder="Ambulance", label_visibility="collapsed")
    ftypes = sorted({t for s in v["failure_types"] for t in s.split(", ") if t})
    ftype = c4.multiselect("Failure type", ftypes, placeholder="Failure type", label_visibility="collapsed")

    mask = pd.Series(True, index=v.index)
    if q:
        mask &= v["incident_id"].str.contains(q, case=False) | v["ambulance_id"].str.contains(q, case=False)
    if status:
        mask &= v["status"].isin(status)
    if ambs:
        mask &= v["ambulance_id"].isin(ambs)
    if ftype:
        mask &= v["failure_types"].map(lambda s: any(t in s.split(", ") for t in ftype))
    view = v[mask].reset_index(drop=True)

    st.caption(f"{len(view)} of {len(v)} incidents · click a row to inspect it")
    if view.empty:
        empty_state("No incidents match these filters.")
        return

    table = pd.DataFrame({
        "Incident": view["incident_id"], "Ambulance": view["ambulance_id"],
        "Status": view["status"].map({"Successful": "✅ Successful", "Failed": "❌ Failed"}),
        "Final state": view["final_state"].str.replace("_", " "), "Events": view["num_events"],
        "Response (min)": view["response_min"], "Total (min)": view["total_min"],
        "Failures": view["failure_types"].str.replace("_", " ").str.title(),
    })
    sel = st.dataframe(table, hide_index=True, height=min(420, 38 + 35 * len(table)), on_select="rerun",
                       selection_mode="single-row", key="explorer_table",
                       column_config={"Response (min)": st.column_config.NumberColumn(format="%.1f"),
                                      "Total (min)": st.column_config.NumberColumn(format="%.1f"),
                                      "Failures": st.column_config.TextColumn(width="large")})
    rows = sel.selection.rows if sel and sel.selection else []
    if rows:
        chosen = view.loc[rows[0], "incident_id"]
    else:
        chosen = st.selectbox("…or choose an incident", view["incident_id"], key="explorer_pick")
    incident_detail(ctx, a, chosen)


def incident_detail(ctx: Dict[str, Any], a: Dict[str, Any], iid: str) -> None:
    row = a["validation"].set_index("incident_id").loc[iid]
    res = a["results"][iid]
    dfa = a["dfa"]
    g = a["events"][a["events"]["incident_id"] == iid].reset_index(drop=True)
    fails = a["failures"][a["failures"]["incident_id"] == iid]
    m = a["metrics"].set_index("incident_id").loc[iid]
    fail_idx = [int(i) for i in fails["event_index"].dropna()]

    section(f"Incident {iid}")
    ok = row["status"] == "Successful"
    st.markdown(
        f"{badge('Successful' if ok else 'Failed', 'ok' if ok else 'fail')} &nbsp;"
        f"{badge('DFA accepted' if res['accepted'] else ('DFA halted' if res['halted'] else 'DFA incomplete'), 'ok' if res['accepted'] else ('fail' if res['halted'] else 'warn'))} &nbsp;"
        f"<span class='chip'>Ambulance <b>{esc(row['ambulance_id'])}</b></span> "
        f"<span class='chip'>Events <b>{int(row['num_events'])}</b></span> "
        f"<span class='chip'>Response <b>{fmt(m['response_min'])} min</b></span> "
        f"<span class='chip'>Total <b>{fmt(m['total_min'])} min</b></span>", unsafe_allow_html=True)
    st.markdown("<div style='height:.7rem'></div>" + stepper(dfa["Q"], res, dfa["q0"]), unsafe_allow_html=True)
    st.markdown("<div style='height:.7rem'></div>", unsafe_allow_html=True)

    c1, c2 = st.columns([1.5, 1])
    c1.plotly_chart(viz.incident_timeline(g, dfa["Sigma"], fail_idx), config=PLOT_CFG, theme=None)
    with c2:
        st.plotly_chart(viz.phase_breakdown(m), config=PLOT_CFG, theme=None)
        if fails.empty:
            empty_state("No failures for this incident.")
        else:
            st.markdown("".join(
                f"<div class='fail-item' style='--sev:{SEV_COLOR[r.severity]}'><div class='t'>"
                f"{esc(r.failure_type.replace('_', ' ').title())} · {esc(r.severity.title())}</div>"
                f"<div class='d'>{esc(r.description)}</div></div>" for r in fails.itertuples()), unsafe_allow_html=True)

    section("Event log with DFA transitions")
    trace = {t["index"]: t for t in res["trace"]}
    log = []
    for i, r in g.iterrows():
        t = trace.get(i)
        if t is None:
            verdict, frm, to = "not evaluated (run stopped)", "", ""
        elif t["valid"]:
            verdict, frm, to = "✅ valid", t["from_state"], t["to_state"]
        else:
            verdict = "❌ " + ("unknown event" if t["reason"] == "UNKNOWN_SYMBOL" else "no transition")
            frm, to = t["from_state"], "TRAP"
        log.append({"#": i + 1, "Timestamp": r["timestamp"].strftime("%Y-%m-%d %H:%M:%S") if pd.notna(r["timestamp"]) else "invalid",
                    "Event": r["event"], "From state": frm, "To state": to, "DFA verdict": verdict,
                    "Timing flag": "⚠ timestamp / gap" if i in fail_idx and (t is None or t["valid"]) else ""})
    st.dataframe(pd.DataFrame(log), hide_index=True, height=min(420, 38 + 35 * len(log)))


def page_dfa(ctx: Dict[str, Any], a: Dict[str, Any]) -> None:
    dfa = a["dfa"]
    hero("DFA", "visualisation", "The ambulance lifecycle as a deterministic finite automaton  M = (Q, Σ, δ, q0, F).", ctx)
    kpi_row([kpi("States |Q|", len(dfa["Q"])), kpi("Alphabet |Σ|", len(dfa["Sigma"])),
             kpi("Start state q0", dfa["q0"].replace("_", " "), text=True), kpi("Accepting F", ", ".join(dfa["F"]).replace("_", " "), tone="good", text=True)])

    tab_machine, tab_run = st.tabs(["State machine", "Run a sequence"])
    with tab_machine:
        st.plotly_chart(viz.dfa_diagram(dfa), config=PLOT_CFG, theme=None)
        st.caption("Double ring = accepting state. Any (state, event) pair without an arrow leads to the implicit trap state.")
        c1, c2 = st.columns([1.4, 1])
        c1.dataframe(pd.DataFrame(dfa["delta"]).rename(columns={"from": "From state", "event": "Event (symbol)", "to": "To state"}),
                     hide_index=True)
        with c2:
            st.markdown("<div class='panel'><b>Formal definition</b><br><br>"
                        f"<b>Q</b> = {{{esc(', '.join(dfa['Q']))}}}<br><br><b>Σ</b> = {{{esc(', '.join(dfa['Sigma']))}}}<br><br>"
                        f"<b>q0</b> = {esc(dfa['q0'])} &nbsp; <b>F</b> = {{{esc(', '.join(dfa['F']))}}}<br><br>"
                        "<span style='color:#8DA2BE;font-size:.85rem'>valid = no invalid transition · complete = ends in F · "
                        "accepted = valid &amp; complete</span></div>", unsafe_allow_html=True)

    with tab_run:
        presets = mock_data.build_presets()
        mode = st.radio("Sequence source", ["Existing incident", "Custom sequence"], horizontal=True, key="dfa_mode")
        v = a["validation"]
        if mode == "Existing incident":
            labels = {r.incident_id: f"{r.incident_id} · {r.ambulance_id} · {r.status}" for r in v.itertuples()}
            iid = st.selectbox("Incident", list(labels), format_func=labels.get, key="dfa_incident")
            seq = a["events"].loc[a["events"]["incident_id"] == iid, "event"].tolist()
            res = a["results"][iid]
        else:
            def load_preset():
                p = st.session_state.get("dfa_preset")
                if p in presets:
                    st.session_state["dfa_text"] = ", ".join(presets[p][0])
            st.session_state.setdefault("dfa_text", ", ".join(presets["Valid full lifecycle"][0]))
            st.selectbox("Load a preset", list(presets), key="dfa_preset", on_change=load_preset)
            text = st.text_area("Events (comma or newline separated)", key="dfa_text", height=110)
            seq = [e.strip() for e in text.replace("\n", ",").split(",") if e.strip()]
            res = ctx["engine"].process_sequence(seq) if seq else None
            if res is None:
                empty_state("Enter at least one event to run it through the DFA.")
            elif res.get("mock_placeholder"):
                st.warning("Mock engine: this exact sequence isn't one it knows, so the result below is a placeholder. "
                           "Use the presets, or switch to the Live backend.")
        if res is not None:
            kind = "ok" if res["accepted"] else ("fail" if res["halted"] else "warn")
            verdict = "Accepted" if res["accepted"] else ("Rejected – invalid transition" if res["halted"] else "Valid but incomplete")
            st.markdown(f"{badge(verdict, kind)} &nbsp; <span class='chip'>Final state <b>{esc(res['final_state'])}</b></span> "
                        f"<span class='chip'>Events <b>{len(seq)}</b></span>", unsafe_allow_html=True)
            st.plotly_chart(viz.dfa_diagram(dfa, res), config=PLOT_CFG, theme=None)
            bad = res["invalid_transitions"][0] if res["invalid_transitions"] else None
            if bad:
                why = "is not a known event" if bad["reason"] == "UNKNOWN_SYMBOL" else f"has no transition from {bad['from_state']}"
                st.error(f"Event #{bad['index'] + 1} `{bad['symbol']}` {why}.")
            trace = pd.DataFrame([{"#": t["index"] + 1, "Event": t["symbol"], "From": t["from_state"],
                                   "To": t["to_state"] or "TRAP", "Valid": "✅" if t["valid"] else "❌"} for t in res["trace"]])
            st.dataframe(trace, hide_index=True)


def page_failures(ctx: Dict[str, Any], a: Dict[str, Any]) -> None:
    hero("Failure", "detection", "Sequence violations from the DFA plus timing and response-time rule breaches.", ctx)
    f, v = a["failures"], a["validation"]
    if f.empty:
        kpi_row([kpi("Failures", 0, "", "good"), kpi("Failure rate", "0%", "", "good")])
        empty_state("No failures detected in this dataset.")
        return

    c1, c2, c3 = st.columns(3)
    types = c1.multiselect("Failure type", sorted(f["failure_type"].unique()), placeholder="All failure types",
                           format_func=lambda s: s.replace("_", " ").title())
    sevs = c2.multiselect("Severity", viz.SEVERITY_ORDER, placeholder="All severities", format_func=str.title)
    ambs = c3.multiselect("Ambulance", sorted(f["ambulance_id"].unique()), placeholder="All ambulances")
    fv = f
    if types:
        fv = fv[fv["failure_type"].isin(types)]
    if sevs:
        fv = fv[fv["severity"].isin(sevs)]
    if ambs:
        fv = fv[fv["ambulance_id"].isin(ambs)]

    n_inc = fv["incident_id"].nunique()
    kpi_row([
        kpi("Failures found", len(fv), f"across {n_inc} incident(s)", "bad"),
        kpi("Failure rate", f"{(f['incident_id'].nunique() / len(v)):.0%}", f"{f['incident_id'].nunique()} of {len(v)} incidents", "warn"),
        kpi("High severity", int((fv["severity"] == "HIGH").sum()), "missing · out-of-order · regression", "bad"),
        kpi("Most common", fv["failure_type"].value_counts().idxmax().replace("_", " ").title() if len(fv) else "–", "in current filter", text=True),
    ])

    summ = (fv.groupby(["failure_type", "severity"]).size().reset_index(name="count"))
    section("Where failures come from")
    c1, c2, c3 = st.columns([1.4, 1, 1.2])
    c1.plotly_chart(viz.failure_type_bar(summ), config=PLOT_CFG, theme=None)
    c2.plotly_chart(viz.severity_donut(summ), config=PLOT_CFG, theme=None)
    c3.plotly_chart(viz.failure_heatmap(fv), config=PLOT_CFG, theme=None)

    section("All detected failures")
    t = fv.copy()
    t["failure_type"] = t["failure_type"].str.replace("_", " ").str.title()
    t["event"] = t["event"].astype(str).str.replace("_", " ")
    t["timestamp"] = pd.to_datetime(t["timestamp"]).dt.strftime("%H:%M:%S").fillna("")
    t["event_index"] = t["event_index"].map(lambda x: "" if pd.isna(x) else str(int(x) + 1))
    t = t.rename(columns={"incident_id": "Incident", "ambulance_id": "Ambulance", "failure_type": "Failure",
                          "severity": "Severity", "event_index": "Event #", "event": "Event",
                          "timestamp": "Time", "description": "Description"})
    styler = t.style.map(lambda s: f"color:{SEV_COLOR.get(s, '#E6F1FF')};font-weight:600", subset=["Severity"])
    st.dataframe(styler, hide_index=True, height=min(520, 38 + 35 * len(t)),
                 column_config={"Description": st.column_config.TextColumn(width="large")})
    st.download_button("Download failures (CSV)", fv.to_csv(index=False), "ambuchain_failures.csv", "text/csv")


def page_response(ctx: Dict[str, Any], a: Dict[str, Any]) -> None:
    hero("Response-time", "analytics", "How quickly ambulances reach the scene, and where the time goes.", ctx)
    m, thr = a["metrics"], ctx["max_response"]
    s = response_stats(m, thr)
    if not s["count"]:
        empty_state("No response times could be computed (missing CALL_RECEIVED / ARRIVED_SCENE or bad timestamps).")
        return
    unmeasured = len(m) - s["count"]

    kpi_row([
        kpi("Mean response", fmt(s["mean"]), f"{s['count']} measured incident(s)", "", "min"),
        kpi("Median", fmt(s["median"]), "", "", "min"),
        kpi("90th percentile", fmt(s["p90"]), "", "warn", "min"),
        kpi("Slowest", fmt(s["max"]), f"{unmeasured} not measurable" if unmeasured else "", "bad" if s["max"] > thr else "", "min"),
    ])
    st.markdown("<div style='height:.6rem'></div>", unsafe_allow_html=True)

    c1, c2 = st.columns([1, 2.2])
    c1.plotly_chart(viz.sla_gauge(s["within"]), config=PLOT_CFG, theme=None)
    c2.plotly_chart(viz.response_trend(m, thr, height=230), config=PLOT_CFG, theme=None)

    c1, c2 = st.columns(2)
    c1.plotly_chart(viz.response_histogram(m, thr), config=PLOT_CFG, theme=None)
    c2.plotly_chart(viz.phase_box(m), config=PLOT_CFG, theme=None)

    section("Ambulance performance")
    amb = a["ambulances"]
    c1, c2 = st.columns([1.2, 1])
    c1.plotly_chart(viz.ambulance_bar(amb, thr), config=PLOT_CFG, theme=None)
    c2.dataframe(amb.sort_values("mean", ascending=False, na_position="last").rename(columns={
        "ambulance_id": "Ambulance", "incidents": "Incidents", "measured": "Measured", "mean": "Mean", "median": "Median",
        "max": "Max", "failed_incidents": "Failed"})[["Ambulance", "Incidents", "Measured", "Mean", "Median", "Max", "Failed"]],
        hide_index=True, height=38 + 35 * len(amb))

    section(f"Slow responses (> {thr:g} min)")
    slow = m[m["response_min"] > thr].sort_values("response_min", ascending=False)
    if slow.empty:
        empty_state("No incident exceeded the response target.")
    else:
        out = slow.assign(excess=(slow["response_min"] - thr).round(2))[
            ["incident_id", "ambulance_id", "dispatch_min", "travel_min", "response_min", "excess"]]
        st.dataframe(out.rename(columns={"incident_id": "Incident", "ambulance_id": "Ambulance", "dispatch_min": "Dispatch (min)",
                                         "travel_min": "Travel (min)", "response_min": "Response (min)", "excess": "Over target (min)"}),
                     hide_index=True)
    st.download_button("Download metrics (CSV)", m.to_csv(index=False), "ambuchain_metrics.csv", "text/csv")


# =========================================================================== #
def run() -> None:
    inject_css()
    ctx = render_sidebar()
    a = load_analysis(ctx)
    if a is None:
        return
    {"Overview": page_overview, "Incident Explorer": page_explorer, "DFA Visualisation": page_dfa,
     "Failure Detection": page_failures, "Response Analytics": page_response}[ctx["page"]](ctx, a)
