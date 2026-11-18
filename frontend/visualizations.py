"""
frontend/visualizations.py
==========================
Pure Plotly figure builders (no Streamlit, no analysis logic).
Every function takes plain DataFrames / dicts and returns a `go.Figure`.
"""

from __future__ import annotations

import math
from typing import Any, Dict, Iterable, List, Optional, Sequence

import pandas as pd
import plotly.graph_objects as go

P = dict(
    bg="#0A1628", panel="#0F2238", panel2="#152F4F", line="#24415F", text="#E6F1FF", muted="#8DA2BE",
    cyan="#22D3EE", teal="#2DD4BF", blue="#38BDF8", green="#34D399", amber="#FBBF24", red="#F87171", violet="#A78BFA",
)
SEVERITY_COLORS = {"HIGH": P["red"], "MEDIUM": P["amber"], "LOW": P["blue"]}
SEVERITY_ORDER = ["HIGH", "MEDIUM", "LOW"]
FONT = "Inter, 'Segoe UI', system-ui, sans-serif"


def _pretty(s: Any) -> str:
    return str(s).replace("_", " ")


def _style(fig: go.Figure, height: int = 320, title: Optional[str] = None, legend: Any = False,
           margin: Optional[dict] = None) -> go.Figure:
    fig.update_layout(
        height=height,
        title=dict(text=title, x=0.0, xanchor="left", font=dict(size=15, color=P["text"])) if title else None,
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family=FONT, color=P["text"], size=12),
        margin=margin or dict(l=8, r=8, t=48 if title else 12, b=48 if legend is True else 28),
        showlegend=bool(legend),
        legend=(dict(orientation="h", yanchor="bottom", y=1.0, xanchor="right", x=1.0, font=dict(size=11),
                     bgcolor="rgba(0,0,0,0)") if legend == "top" else
                dict(orientation="h", yref="container", yanchor="bottom", y=0, xanchor="center", x=0.5,
                     font=dict(size=11), bgcolor="rgba(0,0,0,0)")),
        hoverlabel=dict(bgcolor=P["panel2"], bordercolor=P["cyan"], font=dict(family=FONT, color=P["text"])),
    )
    fig.update_xaxes(automargin=True, gridcolor=P["line"], zerolinecolor=P["line"], linecolor=P["line"], tickfont=dict(color=P["muted"]))
    fig.update_yaxes(automargin=True, gridcolor=P["line"], zerolinecolor=P["line"], linecolor=P["line"], tickfont=dict(color=P["muted"]))
    return fig


def empty_figure(message: str = "No data to display", height: int = 260) -> go.Figure:
    fig = go.Figure()
    fig.add_annotation(text=message, x=0.5, y=0.5, xref="paper", yref="paper", showarrow=False,
                       font=dict(size=14, color=P["muted"]))
    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False)
    return _style(fig, height)


# --------------------------------------------------------------------------- #
# Overview
# --------------------------------------------------------------------------- #
def workflow_donut(successful: int, failed: int, height: int = 300) -> go.Figure:
    total = successful + failed
    if total == 0:
        return empty_figure(height=height)
    fig = go.Figure(go.Pie(
        labels=["Successful", "Failed"], values=[successful, failed], hole=0.72, sort=False,
        marker=dict(colors=[P["green"], P["red"]], line=dict(color=P["bg"], width=3)),
        textinfo="none", hovertemplate="%{label}: %{value} (%{percent})<extra></extra>",
    ))
    fig.add_annotation(text=f"<b>{successful / total:.0%}</b><br><span style='font-size:11px;color:{P['muted']}'>success rate</span>",
                       x=0.5, y=0.5, showarrow=False, font=dict(size=26, color=P["text"]))
    return _style(fig, height, "Workflow outcome", legend=True)


def final_state_bar(validation: pd.DataFrame, state_order: Sequence[str], height: int = 300) -> go.Figure:
    """How far each incident progressed through the DFA."""
    if validation.empty:
        return empty_figure(height=height)
    counts = validation["final_state"].value_counts().reindex(list(state_order), fill_value=0)
    colors = [P["green"] if s == state_order[-1] else P["cyan"] for s in counts.index]
    fig = go.Figure(go.Bar(x=[_pretty(s) for s in counts.index], y=counts.values, marker_color=colors,
                           text=counts.values, textposition="outside", cliponaxis=False,
                           hovertemplate="%{x}: %{y} incident(s)<extra></extra>"))
    fig.update_yaxes(showgrid=True, rangemode="tozero")
    fig.update_xaxes(tickangle=-35)
    return _style(fig, height, "Last DFA state reached per incident", margin=dict(l=8, r=8, t=48, b=8))


# --------------------------------------------------------------------------- #
# Failures
# --------------------------------------------------------------------------- #
def failure_type_bar(summary: pd.DataFrame, height: int = 320) -> go.Figure:
    if summary.empty:
        return empty_figure("No failures detected", height)
    fig = go.Figure()
    ordered = summary.sort_values("count", ascending=True)
    for sev in SEVERITY_ORDER:
        part = ordered[ordered["severity"] == sev]
        if part.empty:
            continue
        fig.add_trace(go.Bar(y=[_pretty(t) for t in part["failure_type"]], x=part["count"], orientation="h",
                             name=sev.title(), marker_color=SEVERITY_COLORS[sev], text=part["count"],
                             textposition="outside", cliponaxis=False,
                             hovertemplate="%{y}: %{x}<extra>" + sev.title() + " severity</extra>"))
    fig.update_layout(barmode="stack")
    fig.update_xaxes(range=[0, float(ordered.groupby("failure_type")["count"].sum().max()) * 1.25], visible=False)
    return _style(fig, height, "Failures by type", legend=True)


def severity_donut(summary: pd.DataFrame, height: int = 320) -> go.Figure:
    if summary.empty:
        return empty_figure("No failures detected", height)
    by_sev = summary.groupby("severity")["count"].sum().reindex(SEVERITY_ORDER, fill_value=0)
    by_sev = by_sev[by_sev > 0]
    fig = go.Figure(go.Pie(labels=[s.title() for s in by_sev.index], values=by_sev.values, hole=0.62, sort=False,
                           marker=dict(colors=[SEVERITY_COLORS[s] for s in by_sev.index], line=dict(color=P["bg"], width=3)),
                           textinfo="label+value", insidetextorientation="horizontal",
                           textfont=dict(color=P["bg"], size=12)))
    return _style(fig, height, "Severity mix")


def failure_heatmap(failures: pd.DataFrame, height: int = 320) -> go.Figure:
    if failures.empty:
        return empty_figure("No failures detected", height)
    ct = pd.crosstab(failures["failure_type"], failures["ambulance_id"])
    fig = go.Figure(go.Heatmap(
        z=ct.values, x=list(ct.columns), y=[_pretty(c) for c in ct.index], xgap=3, ygap=3,
        colorscale=[[0, P["panel"]], [1, P["cyan"]]], showscale=False,
        text=ct.values, texttemplate="%{text}", textfont=dict(color=P["text"]),
        hovertemplate="%{y} · %{x}: %{z}<extra></extra>",
    ))
    fig.update_xaxes(showgrid=False, side="bottom")
    fig.update_yaxes(showgrid=False, autorange="reversed")
    return _style(fig, height, "Failures by ambulance")


# --------------------------------------------------------------------------- #
# Response-time analytics
# --------------------------------------------------------------------------- #
def response_histogram(metrics: pd.DataFrame, threshold: float, height: int = 320) -> go.Figure:
    s = metrics["response_min"].dropna()
    if s.empty:
        return empty_figure(height=height)
    fig = go.Figure(go.Histogram(x=s, xbins=dict(size=1), marker=dict(color=P["cyan"], line=dict(color=P["bg"], width=1)),
                                 hovertemplate="%{x} min: %{y} incident(s)<extra></extra>"))
    fig.add_vline(x=threshold, line=dict(color=P["amber"], dash="dash", width=2),
                  annotation_text=f"Target {threshold:g} min", annotation_font_color=P["amber"])
    fig.add_vline(x=float(s.median()), line=dict(color=P["green"], dash="dot", width=2),
                  annotation_text=f"Median {s.median():.1f}", annotation_position="top left",
                  annotation_font_color=P["green"])
    fig.update_xaxes(title="Response time (min)")
    fig.update_yaxes(title="Incidents")
    fig.update_layout(bargap=0.08)
    return _style(fig, height, "Response-time distribution", margin=dict(l=8, r=8, t=48, b=44))


def response_trend(metrics: pd.DataFrame, threshold: float, height: int = 320) -> go.Figure:
    m = metrics.dropna(subset=["response_min", "call_time"]).sort_values("call_time")
    if m.empty:
        return empty_figure(height=height)
    over = m["response_min"] > threshold
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=m["call_time"], y=m["response_min"].rolling(5, min_periods=1).median(), mode="lines",
                             name="Rolling median (5)", line=dict(color=P["muted"], width=1.5, dash="dot"), hoverinfo="skip"))
    for flag, name, color in ((False, "Within target", P["cyan"]), (True, "Over target", P["amber"])):
        part = m[over == flag]
        fig.add_trace(go.Scatter(
            x=part["call_time"], y=part["response_min"], mode="markers", name=name,
            marker=dict(color=color, size=10, line=dict(color=P["bg"], width=1.5)),
            customdata=part[["incident_id", "ambulance_id"]].values,
            hovertemplate="%{customdata[0]} · %{customdata[1]}<br>%{y:.1f} min at %{x|%H:%M}<extra></extra>"))
    fig.add_hline(y=threshold, line=dict(color=P["amber"], dash="dash", width=1.5))
    fig.update_yaxes(title="Response (min)", rangemode="tozero")
    return _style(fig, height, "Response time through the day", legend="top")


def ambulance_bar(table: pd.DataFrame, threshold: float, height: int = 320) -> go.Figure:
    """`table` needs: ambulance_id, mean."""
    t = table.dropna(subset=["mean"]).sort_values("mean", ascending=False)
    if t.empty:
        return empty_figure(height=height)
    colors = [P["amber"] if v > threshold else P["cyan"] for v in t["mean"]]
    fig = go.Figure(go.Bar(x=t["ambulance_id"], y=t["mean"], marker_color=colors, text=t["mean"].round(1),
                           textposition="outside", cliponaxis=False,
                           hovertemplate="%{x}: %{y:.1f} min average<extra></extra>"))
    fig.add_hline(y=threshold, line=dict(color=P["amber"], dash="dash", width=1.5))
    fig.update_yaxes(title="Mean response (min)", rangemode="tozero")
    return _style(fig, height, "Average response by ambulance")


PHASES = [("dispatch_min", "Dispatch", P["violet"]), ("travel_min", "Travel", P["cyan"]),
          ("scene_min", "On scene", P["teal"]), ("transport_min", "Transport", P["blue"])]


def phase_box(metrics: pd.DataFrame, height: int = 320) -> go.Figure:
    fig = go.Figure()
    for col, label, color in PHASES:
        s = metrics[col].dropna()
        if not s.empty:
            fig.add_trace(go.Box(y=s, name=label, marker_color=color, line=dict(color=color), boxpoints="outliers"))
    if not fig.data:
        return empty_figure(height=height)
    fig.update_yaxes(title="Minutes", rangemode="tozero")
    return _style(fig, height, "Time spent per phase")


def phase_breakdown(metrics_row: pd.Series, height: int = 170) -> go.Figure:
    """Stacked horizontal bar of one incident's phases."""
    fig = go.Figure()
    for col, label, color in PHASES:
        v = metrics_row.get(col)
        if pd.notna(v) and v > 0:
            fig.add_trace(go.Bar(y=[""], x=[v], orientation="h", name=label, marker_color=color, text=f"{v:g}",
                                 textposition="inside", insidetextanchor="middle",
                                 textfont=dict(color=P["bg"]),
                                 hovertemplate=f"{label}: {v:g} min<extra></extra>"))
    if not fig.data:
        return empty_figure("Phase timings not computable for this incident", height)
    fig.update_layout(barmode="stack")
    return _style(fig, height, "Phase breakdown (min)", legend=True, margin=dict(l=8, r=8, t=48, b=52))


def sla_gauge(value: float, target: float = 90.0, height: int = 230) -> go.Figure:
    """value & target are percentages (share of responses within the time target)."""
    fig = go.Figure(go.Indicator(
        mode="gauge+number", value=value, number=dict(suffix="%", font=dict(color=P["text"], size=34)),
        gauge=dict(axis=dict(range=[0, 100], tickvals=[0, 50, 100], tickcolor=P["muted"]), bar=dict(color=P["cyan"], thickness=0.28),
                   bgcolor=P["panel"], borderwidth=0,
                   steps=[dict(range=[0, target], color=P["panel2"])],
                   threshold=dict(line=dict(color=P["amber"], width=3), thickness=0.8, value=target)),
    ))
    return _style(fig, height, "Within response target", margin=dict(l=24, r=24, t=56, b=8))


# --------------------------------------------------------------------------- #
# Incident explorer
# --------------------------------------------------------------------------- #
def incident_timeline(group: pd.DataFrame, lifecycle: Sequence[str], failing_idx: Iterable[int] = (),
                      height: int = 380) -> go.Figure:
    """Events in LOG ORDER (x = timestamp, y = lifecycle stage).  A line that
    runs backwards in time is a timestamp regression; red markers = failures."""
    g = group.reset_index(drop=True)
    if g.empty or g["timestamp"].isna().all():
        return empty_figure("No timestamps available", height)
    bad = set(failing_idx)
    cats = list(lifecycle) + [e for e in g["event"].unique() if e not in lifecycle]
    colors = [P["red"] if i in bad else P["cyan"] for i in range(len(g))]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=g["timestamp"], y=g["event"], mode="lines", line=dict(color=P["line"], width=2),
                             hoverinfo="skip", showlegend=False))
    fig.add_trace(go.Scatter(
        x=g["timestamp"], y=g["event"], mode="markers+text", text=[str(i + 1) for i in range(len(g))],
        textposition="middle center", textfont=dict(size=9, color=P["bg"]),
        marker=dict(size=20, color=colors, line=dict(color=P["bg"], width=2)), showlegend=False,
        hovertemplate="#%{text} %{y}<br>%{x|%H:%M:%S}<extra></extra>"))
    fig.update_yaxes(categoryorder="array", categoryarray=cats[::-1], tickmode="array",
                     tickvals=cats, ticktext=[_pretty(c) for c in cats])
    fig.update_xaxes(tickformat="%H:%M")
    return _style(fig, height, "Event timeline (log order · red = failure)", margin=dict(l=8, r=16, t=48, b=8))


# --------------------------------------------------------------------------- #
# DFA diagram
# --------------------------------------------------------------------------- #
def _layout(states: Sequence[str]) -> Dict[str, tuple]:
    """Snake layout: first half left->right on top, rest right->left below."""
    top = math.ceil(len(states) / 2)
    pos = {}
    for i, s in enumerate(states):
        pos[s] = (i, 1.0) if i < top else ((top - 1) - (i - top), 0.0)
    return pos


def dfa_diagram(definition: Dict[str, Any], result: Optional[Dict[str, Any]] = None, height: int = 400) -> go.Figure:
    """Draw the DFA from its definition (Q, delta, q0, F).  If a run `result`
    is given, the traversed path is highlighted and the stop point marked.
    This only DRAWS what the backend reports; it never evaluates transitions."""
    states, delta = definition["Q"], definition["delta"]
    pos = _layout(states)
    top = math.ceil(len(states) / 2)
    trace = (result or {}).get("trace", [])
    taken = {(t["from_state"], t["symbol"]) for t in trace if t["valid"]}
    visited = {definition["q0"]} | {t["to_state"] for t in trace if t["valid"]}
    invalid = next((t for t in trace if not t["valid"]), None)
    has_run = result is not None

    fig = go.Figure()
    R = 0.2  # arrow inset from node centre

    def arrow(p1, p2, color, width, label=None, label_color=None, dash=None):
        dx, dy = p2[0] - p1[0], p2[1] - p1[1]
        L = math.hypot(dx, dy) or 1
        s = (p1[0] + dx / L * R, p1[1] + dy / L * R)
        e = (p2[0] - dx / L * (R + 0.02), p2[1] - dy / L * (R + 0.02))
        if dash:  # annotations can't dash: draw as line + head
            fig.add_shape(type="line", x0=s[0], y0=s[1], x1=e[0], y1=e[1], line=dict(color=color, width=width, dash="dash"))
            fig.add_annotation(x=e[0], y=e[1], ax=e[0] - dx / L * 0.1, ay=e[1] - dy / L * 0.1, xref="x", yref="y",
                               axref="x", ayref="y", showarrow=True, arrowhead=3, arrowsize=0.9, arrowwidth=width, arrowcolor=color)
        else:
            fig.add_annotation(x=e[0], y=e[1], ax=s[0], ay=s[1], xref="x", yref="y", axref="x", ayref="y",
                               showarrow=True, arrowhead=3, arrowsize=0.9, arrowwidth=width, arrowcolor=color)
        if label:
            mx, my = (p1[0] + p2[0]) / 2, (p1[1] + p2[1]) / 2
            vertical = abs(dx) < 1e-9
            fig.add_annotation(x=mx + (0.1 if vertical else 0), y=my + (0 if vertical else 0.13),
                               text=_pretty(label), showarrow=False, xanchor="left" if vertical else "center",
                               font=dict(size=10, color=label_color or P["muted"]))

    for tr in delta:
        a, b, ev = pos[tr["from"]], pos[tr["to"]], tr["event"]
        used = (tr["from"], ev) in taken
        if has_run and not used:
            arrow(a, b, P["line"], 1.5, ev, "#4B6483")
        elif used:
            arrow(a, b, P["cyan"], 3.5, ev, P["cyan"])
        else:
            arrow(a, b, "#3B6A91", 2, ev)

    # start marker
    q0 = pos[definition["q0"]]
    fig.add_annotation(x=q0[0] - R, y=q0[1], ax=q0[0] - 0.62, ay=q0[1], xref="x", yref="y", axref="x", ayref="y",
                       showarrow=True, arrowhead=3, arrowwidth=2, arrowcolor=P["muted"])
    fig.add_annotation(x=q0[0] - 0.62, y=q0[1] + 0.12, text="start", showarrow=False, font=dict(size=10, color=P["muted"]))

    # trap state + the invalid transition
    ymin, ymax = -0.75, 1.75
    if invalid:
        fx, fy = pos[invalid["from_state"]]
        ty = 2.15 if fy > 0.5 else -1.15
        ymax, ymin = (2.6, ymin) if fy > 0.5 else (ymax, -1.6)
        arrow((fx, fy), (fx, ty), P["red"], 3, None, dash=True)
        sym = _pretty(invalid["symbol"]) + (" (unknown)" if invalid.get("reason") == "UNKNOWN_SYMBOL" else "")
        fig.add_annotation(x=fx + 0.1, y=(fy + ty) / 2, text=sym, showarrow=False, xanchor="left",
                           font=dict(size=10, color=P["red"]))
        fig.add_trace(go.Scatter(x=[fx], y=[ty], mode="markers+text", text=["TRAP"], textposition="middle center",
                                 textfont=dict(size=10, color=P["red"]), hoverinfo="skip", showlegend=False,
                                 marker=dict(size=44, color=P["bg"], line=dict(color=P["red"], width=3, dash="dot"))))

    # nodes
    final = (result or {}).get("final_state")
    xs, ys, fills, borders, texts, hover, tcolors = [], [], [], [], [], [], []
    for i, s in enumerate(states):
        x, y = pos[s]
        xs.append(x); ys.append(y)
        if has_run and s in visited:
            fills.append(P["cyan"]); tcolors.append(P["bg"])
        else:
            fills.append(P["panel2"]); tcolors.append(P["text"])
        borders.append(P["cyan"] if (not has_run or s in visited) else P["line"])
        texts.append(f"q{i}")
        hover.append(f"{s}" + (" (start)" if s == definition["q0"] else "") + (" (accepting)" if s in definition["F"] else ""))
    fig.add_trace(go.Scatter(x=xs, y=ys, mode="markers+text", text=texts, textfont=dict(size=12, color=tcolors),
                             marker=dict(size=44, color=fills, line=dict(color=borders, width=2.5)),
                             hovertext=hover, hoverinfo="text", showlegend=False))
    # accepting ring (double circle)
    acc = [pos[s] for s in definition["F"]]
    fig.add_trace(go.Scatter(x=[a[0] for a in acc], y=[a[1] for a in acc], mode="markers", hoverinfo="skip", showlegend=False,
                             marker=dict(size=56, color="rgba(0,0,0,0)", line=dict(color=P["green"], width=2.5))))
    # result ring around where the run ended
    if has_run and final in pos:
        ring = P["green"] if result.get("accepted") else (P["red"] if result.get("halted") else P["amber"])
        fig.add_trace(go.Scatter(x=[pos[final][0]], y=[pos[final][1]], mode="markers", hoverinfo="skip", showlegend=False,
                                 marker=dict(size=66, color="rgba(0,0,0,0)", line=dict(color=ring, width=3))))
    for i, s in enumerate(states):
        x, y = pos[s]
        fig.add_annotation(x=x, y=y - 0.27, text=f"<b>{_pretty(s)}</b>", showarrow=False, yanchor="top",
                           font=dict(size=11, color=P["text"] if (not has_run or s in visited) else P["muted"]))

    fig.update_xaxes(visible=False, range=[-0.95, top - 0.1], fixedrange=True)
    fig.update_yaxes(visible=False, range=[ymin, ymax], fixedrange=True)
    return _style(fig, height, margin=dict(l=0, r=0, t=8, b=0))
