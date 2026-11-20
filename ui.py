"""Design system: styles, header, cards, Plotly theme and interactive image viewers.

Colour semantics (used everywhere)
    green  #22c55e  normal / success / ground truth
    red    #ef4444  anomaly / error / prediction
    amber  #f59e0b  warning / unavailable
    cyan   #22d3ee  information / selected state
"""

from __future__ import annotations

import base64
import html
import io

import cv2
import numpy as np
import plotly.graph_objects as go
import streamlit as st
import streamlit.components.v1 as components
from PIL import Image
from plotly.subplots import make_subplots

GREEN, RED, AMBER, CYAN, BLUE, VIOLET = "#22c55e", "#ef4444", "#f59e0b", "#22d3ee", "#3b82f6", "#a78bfa"
MODEL_COLORS = {"patchcore_r18": CYAN, "patchcore_wrn50": BLUE, "efficientad_s": VIOLET, "efficientad_m": "#f472b6"}

CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');
html, body, [class*="css"], .stMarkdown, .stText, button, input, select, textarea {
  font-family: 'Inter', -apple-system, 'Segoe UI', Roboto, sans-serif !important;
}
.stApp { background: radial-gradient(1200px 600px at 85% -10%, rgba(34,211,238,0.07), transparent 60%),
                     radial-gradient(900px 500px at -10% 10%, rgba(59,130,246,0.06), transparent 60%), #0a0f1c; }
.block-container { padding-top: 1.1rem; padding-bottom: 3rem; max-width: 1480px; }
header[data-testid="stHeader"] { background: transparent; height: 0; }
[data-testid="stToolbar"] { display: none; }
h1, h2, h3 { letter-spacing: -0.01em; }
h3 { font-size: 1.05rem !important; font-weight: 600 !important; color: #cbd5e1; margin-top: 0.6rem; }

/* sidebar navigation */
section[data-testid="stSidebar"] { background: linear-gradient(180deg, #0d1527 0%, #0a1020 100%);
  border-right: 1px solid rgba(148,163,184,0.12); }
section[data-testid="stSidebar"] .stRadio > div { gap: 2px; }
section[data-testid="stSidebar"] .stRadio label { padding: 8px 12px; border-radius: 8px; width: 100%;
  transition: background .15s ease, color .15s ease; color: #94a3b8; border: 1px solid transparent; }
section[data-testid="stSidebar"] .stRadio label:hover { background: rgba(34,211,238,0.07); color: #e2e8f0; }
section[data-testid="stSidebar"] .stRadio label:has(input:checked) { background: rgba(34,211,238,0.12);
  color: #e0f7fb; border-color: rgba(34,211,238,0.35); }
section[data-testid="stSidebar"] .stRadio label > div:first-child { display: none; }
section[data-testid="stSidebar"] .stRadio label p { color: inherit !important; font-size: 0.9rem; }
section[data-testid="stSidebar"] label[data-testid="stWidgetLabel"] p { color: #94a3b8 !important; }
[data-testid="stMain"] div[role="radiogroup"] label p { color: inherit !important; }
.nav-caption { font-size: 0.68rem; letter-spacing: .12em; text-transform: uppercase; color: #64748b; margin: 14px 0 6px 4px; }

/* top header */
.topbar { display: flex; align-items: center; justify-content: space-between; gap: 16px; flex-wrap: wrap;
  padding: 12px 18px; margin-bottom: 18px; border-radius: 14px; border: 1px solid rgba(148,163,184,0.14);
  background: linear-gradient(90deg, rgba(17,26,46,0.92), rgba(15,23,42,0.75)); }
.brand { display: flex; align-items: center; gap: 12px; }
.brand-title { font-weight: 700; font-size: 1.02rem; color: #f1f5f9; }
.brand-sub { font-size: 0.74rem; color: #64748b; }
.chips { display: flex; gap: 8px; flex-wrap: wrap; }
.chip { font-size: 0.74rem; padding: 5px 10px; border-radius: 999px; border: 1px solid rgba(148,163,184,0.2);
  color: #cbd5e1; background: rgba(15,23,42,0.6); white-space: nowrap; }
.chip b { color: #f1f5f9; font-weight: 600; }
.dot { display: inline-block; width: 7px; height: 7px; border-radius: 50%; margin-right: 6px; vertical-align: middle; }

/* page header */
.page-head { margin: 4px 0 16px 0; }
.page-head .eyebrow { font-size: 0.7rem; letter-spacing: .14em; text-transform: uppercase; color: #22d3ee; font-weight: 600; }
.page-head h1 { font-size: 1.65rem; font-weight: 700; color: #f8fafc; margin: 2px 0 4px 0; padding: 0; }
.page-head p { color: #94a3b8; font-size: 0.92rem; margin: 0; max-width: 880px; }

/* cards */
.card { border: 1px solid rgba(148,163,184,0.14); border-radius: 14px; padding: 16px 18px;
  background: linear-gradient(180deg, rgba(17,26,46,0.85), rgba(13,20,36,0.85)); height: 100%;
  transition: border-color .15s ease, transform .15s ease; }
.card:hover { border-color: rgba(34,211,238,0.35); }
.kpi-label { font-size: 0.74rem; color: #94a3b8; text-transform: uppercase; letter-spacing: .08em; }
.kpi-value { font-size: 1.7rem; font-weight: 700; color: #f8fafc; margin-top: 4px; font-variant-numeric: tabular-nums; }
.kpi-sub { font-size: 0.78rem; color: #64748b; margin-top: 2px; }
.badge { display: inline-block; font-size: 0.7rem; font-weight: 600; padding: 3px 9px; border-radius: 999px; margin-right: 4px; }
.b-green { background: rgba(34,197,94,0.14); color: #4ade80; border: 1px solid rgba(34,197,94,0.35); }
.b-red { background: rgba(239,68,68,0.14); color: #f87171; border: 1px solid rgba(239,68,68,0.35); }
.b-amber { background: rgba(245,158,11,0.14); color: #fbbf24; border: 1px solid rgba(245,158,11,0.35); }
.b-cyan { background: rgba(34,211,238,0.12); color: #67e8f9; border: 1px solid rgba(34,211,238,0.35); }
.model-card h4 { margin: 0 0 2px 0; font-size: 1.02rem; color: #f1f5f9; }
.model-card .arch { font-size: 0.78rem; color: #94a3b8; min-height: 3.4em; margin: 6px 0 10px 0; }
.metric-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 6px 14px; }
.metric-grid .k { font-size: 0.72rem; color: #64748b; }
.metric-grid .v { font-size: 0.98rem; font-weight: 600; color: #e2e8f0; font-variant-numeric: tabular-nums; }
.unavailable { border: 1px dashed rgba(245,158,11,0.45); background: rgba(245,158,11,0.05); border-radius: 12px;
  padding: 14px 16px; color: #fcd34d; font-size: 0.84rem; }
.disabled-panel { border: 1px dashed rgba(148,163,184,0.3); border-radius: 12px; padding: 26px 16px; text-align: center;
  color: #94a3b8; font-size: 0.84rem; background: rgba(15,23,42,0.5); }
.verdict { border-radius: 12px; padding: 14px 18px; font-weight: 700; font-size: 1.15rem; display: flex;
  align-items: center; justify-content: space-between; }
.v-ok { background: rgba(34,197,94,0.12); border: 1px solid rgba(34,197,94,0.45); color: #4ade80; }
.v-bad { background: rgba(239,68,68,0.12); border: 1px solid rgba(239,68,68,0.45); color: #f87171; }
.legend { display: flex; gap: 16px; flex-wrap: wrap; font-size: 0.8rem; color: #cbd5e1; margin: 6px 0; }
.sw { display: inline-block; width: 12px; height: 12px; border-radius: 3px; margin-right: 6px; vertical-align: -1px; }
.gradient-bar { height: 10px; border-radius: 5px; background: linear-gradient(90deg,#00007f,#0000ff,#00ffff,#ffff00,#ff0000); }
.mono { font-family: 'JetBrains Mono', monospace; font-size: 0.8rem; color: #94a3b8; }
div[data-testid="stMetric"] { border: 1px solid rgba(148,163,184,0.14); border-radius: 12px; padding: 10px 14px;
  background: rgba(17,26,46,0.7); }
div[data-testid="stExpander"] { border-radius: 12px; border-color: rgba(148,163,184,0.16); }
.stButton > button, .stDownloadButton > button { border-radius: 10px; font-weight: 600; transition: all .15s ease; }
.stButton > button:hover { border-color: #22d3ee; color: #22d3ee; }
div[data-testid="stDataFrame"] { border-radius: 12px; overflow: hidden; }
/* horizontal radios in the main area are rendered as segmented controls */
section.main div[role="radiogroup"], [data-testid="stMain"] div[role="radiogroup"] { gap: 0 !important;
  display: inline-flex; flex-wrap: wrap; border: 1px solid rgba(148,163,184,0.22); border-radius: 10px; padding: 3px;
  background: rgba(15,23,42,0.6); }
[data-testid="stMain"] div[role="radiogroup"] label { margin: 0 !important; padding: 6px 13px; border-radius: 7px;
  color: #94a3b8; transition: background .15s ease, color .15s ease; }
[data-testid="stMain"] div[role="radiogroup"] label:hover { color: #e2e8f0; background: rgba(148,163,184,0.08); }
[data-testid="stMain"] div[role="radiogroup"] label:has(input:checked) { background: rgba(34,211,238,0.16);
  color: #e0f7fb; box-shadow: inset 0 0 0 1px rgba(34,211,238,0.45); }
[data-testid="stMain"] div[role="radiogroup"] label > div:first-child { display: none; }
.thumb-cap { font-size: 0.75rem; color: #94a3b8; margin-top: -6px; }
</style>
"""

LOGO = """<svg width="34" height="34" viewBox="0 0 34 34" fill="none" xmlns="http://www.w3.org/2000/svg">
<rect x="1" y="1" width="32" height="32" rx="9" fill="url(#g)" stroke="rgba(34,211,238,0.5)"/>
<circle cx="17" cy="17" r="7.5" stroke="#e0f7fb" stroke-width="2"/><circle cx="17" cy="17" r="2.6" fill="#22d3ee"/>
<path d="M17 4.5v4M17 25.5v4M4.5 17h4M25.5 17h4" stroke="#67e8f9" stroke-width="2" stroke-linecap="round"/>
<defs><linearGradient id="g" x1="0" y1="0" x2="34" y2="34"><stop stop-color="#0e7490"/><stop offset="1" stop-color="#1e3a8a"/></linearGradient></defs></svg>"""


def inject_css() -> None:
    st.markdown(CSS, unsafe_allow_html=True)


def topbar(dataset: str, model: str, status: str, status_ok: bool) -> None:
    color = GREEN if status_ok else AMBER
    st.markdown(f"""<div class="topbar"><div class="brand">{LOGO}<div>
      <div class="brand-title">Industrial AI Inspection Platform</div>
      <div class="brand-sub">EfficientAD (with PatchCore baseline) · anomaly detection &amp; localisation</div></div></div>
      <div class="chips"><span class="chip">Dataset <b>{html.escape(dataset)}</b></span>
      <span class="chip">Model <b>{html.escape(model)}</b></span>
      <span class="chip"><span class="dot" style="background:{color}"></span>{html.escape(status)}</span></div></div>""",
                unsafe_allow_html=True)


def page_header(eyebrow: str, title: str, text: str = "") -> None:
    st.markdown(f'<div class="page-head"><div class="eyebrow">{eyebrow}</div><h1>{title}</h1>'
                f'<p>{text}</p></div>', unsafe_allow_html=True)


def kpi(label: str, value: str, sub: str = "", accent: str | None = None) -> str:
    style = f' style="border-top: 2px solid {accent}"' if accent else ""
    return (f'<div class="card"{style}><div class="kpi-label">{label}</div><div class="kpi-value">{value}</div>'
            f'<div class="kpi-sub">{sub}</div></div>')


def badge(text: str, kind: str) -> str:
    return f'<span class="badge b-{kind}">{html.escape(text)}</span>'


def html_block(markup: str) -> None:
    st.markdown(markup, unsafe_allow_html=True)


def fmt(value, digits: int = 4) -> str:
    return "—" if value is None or (isinstance(value, float) and np.isnan(value)) else f"{value:.{digits}f}"


# ------------------------------------------------------------------ plotly
SERIES = [CYAN, BLUE, VIOLET, "#94a3b8"]  # multi-series charts: neutral, never green/red


def style_fig(fig: go.Figure, height: int = 320, title: str | None = None, legend: bool = True) -> go.Figure:
    fig.update_layout(
        template="plotly_dark", height=height, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(15,23,42,0.55)",
        font=dict(family="Inter, sans-serif", size=12, color="#cbd5e1"), colorway=SERIES,
        margin=dict(l=48, r=18, t=46 if title else 12, b=96 if legend else 44),
        title=dict(text=title or "", font=dict(size=14, color="#e2e8f0"), x=0.01, y=0.97),
        showlegend=legend, legend=dict(orientation="h", yanchor="top", y=-0.3, x=0, bgcolor="rgba(0,0,0,0)",
                                       font=dict(size=11)),
        hoverlabel=dict(bgcolor="#0f172a", bordercolor="#22d3ee", font=dict(color="#e2e8f0")),
    )
    fig.update_xaxes(gridcolor="rgba(148,163,184,0.12)", zerolinecolor="rgba(148,163,184,0.2)")
    fig.update_yaxes(gridcolor="rgba(148,163,184,0.12)", zerolinecolor="rgba(148,163,184,0.2)")
    return fig


def plotly_config() -> dict:
    """Image viewers: toolbar for zoom / pan / reset, wheel zoom."""
    return {"displaylogo": False, "scrollZoom": True,
            "modeBarButtonsToRemove": ["select2d", "lasso2d", "autoScale2d", "toImage"]}


def chart_config() -> dict:
    """Plain charts: tooltips only, no toolbar clutter."""
    return {"displaylogo": False, "displayModeBar": False}


def display_size(image: np.ndarray, max_side: int = 720) -> np.ndarray:
    h, w = image.shape[:2]
    scale = min(1.0, max_side / max(h, w))
    if scale == 1.0:
        return image
    interp = cv2.INTER_NEAREST if image.dtype == np.uint8 and image.ndim == 2 else cv2.INTER_AREA
    return cv2.resize(image, (round(w * scale), round(h * scale)), interpolation=interp)


def image_figure(image: np.ndarray, height: int = 520, boxes: list[dict] | None = None, scale: float = 1.0,
                 heat: np.ndarray | None = None, heat_range: tuple[float, float] | None = None,
                 threshold: float | None = None) -> go.Figure:
    """Zoomable / pannable image (aspect ratio preserved) with optional heatmap and boxes."""
    fig = go.Figure()
    h, w = image.shape[:2]
    if heat is not None:
        lo, hi = heat_range
        # <=320 px grid keeps the figure light; x/y are mapped back to image pixels.
        small = display_size(heat.astype(np.float32), 320)
        dx, dy = w / small.shape[1], h / small.shape[0]
        fig.add_trace(go.Heatmap(
            z=small, x0=dx / 2, dx=dx, y0=dy / 2, dy=dy, zsmooth="best", colorscale="Jet", zmin=lo, zmax=hi,
            colorbar=dict(title=dict(text="anomaly<br>score", side="right"), thickness=12, len=0.9,
                          tickfont=dict(size=10)),
            hovertemplate="x %{x:.0f}, y %{y:.0f}<br>score %{z:.3f}<extra></extra>"))
        if threshold is not None:
            fig.add_annotation(xref="paper", yref="paper", x=1.0, y=-0.06, showarrow=False, xanchor="right",
                               text=f"pixel threshold = {threshold:.3f}", font=dict(size=11, color="#fbbf24"))
    else:
        fig.add_trace(go.Image(z=image, hovertemplate="x %{x}, y %{y}<extra></extra>"))
    for index, box in enumerate(boxes or []):
        x0, y0, x1, y1 = (v * scale for v in (box["x0"], box["y0"], box["x1"], box["y1"]))
        fig.add_shape(type="rect", x0=x0, y0=y0, x1=x1, y1=y1, line=dict(color="#fbbf24", width=2))
        fig.add_annotation(x=x0, y=y0, text=f"R{index + 1}", showarrow=False, yshift=9, xanchor="left",
                           font=dict(size=11, color="#0a0f1c"), bgcolor="#fbbf24")
    fig.update_xaxes(visible=False, range=[0, w], constrain="domain")
    fig.update_yaxes(visible=False, range=[h, 0], scaleanchor="x", scaleratio=1, constrain="domain")
    fig.update_layout(height=height, margin=dict(l=0, r=0, t=0, b=24 if heat is not None else 0),
                      paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", dragmode="pan",
                      font=dict(family="Inter, sans-serif", color="#cbd5e1"))
    return fig


def synced_figure(left: np.ndarray, right: np.ndarray, titles: tuple[str, str], height: int = 470) -> go.Figure:
    """Two images whose zoom and pan are linked (shared axes)."""
    fig = make_subplots(rows=1, cols=2, shared_xaxes=True, shared_yaxes=True, horizontal_spacing=0.02,
                        subplot_titles=titles)
    fig.add_trace(go.Image(z=left, hovertemplate="x %{x}, y %{y}<extra></extra>"), 1, 1)
    fig.add_trace(go.Image(z=right, hovertemplate="x %{x}, y %{y}<extra></extra>"), 1, 2)
    h, w = left.shape[:2]
    fig.update_xaxes(visible=False, range=[0, w], matches="x", constrain="domain")
    fig.update_yaxes(visible=False, range=[h, 0], matches="y", scaleanchor="x", scaleratio=1, constrain="domain")
    fig.update_layout(height=height, margin=dict(l=0, r=0, t=30, b=0), paper_bgcolor="rgba(0,0,0,0)",
                      plot_bgcolor="rgba(0,0,0,0)", dragmode="pan",
                      font=dict(family="Inter, sans-serif", color="#cbd5e1", size=12))
    return fig


# ---------------------------------------------------------- compare slider
def _data_uri(image: np.ndarray, max_side: int = 900) -> str:
    image = display_size(image, max_side)
    buffer = io.BytesIO()
    Image.fromarray(image).save(buffer, format="JPEG", quality=90)
    return "data:image/jpeg;base64," + base64.b64encode(buffer.getvalue()).decode()


def compare_slider(before: np.ndarray, after: np.ndarray, before_label: str, after_label: str,
                   height: int = 560, key: str = "cmp") -> None:
    """Before/after slider with synchronized mouse-wheel zoom and drag-to-pan (plain JS)."""
    h, w = before.shape[:2]
    ratio = w / h
    markup = f"""
<div id="{key}" class="wrap" style="aspect-ratio:{ratio};">
  <div class="layer"><img src="{_data_uri(before)}" draggable="false"></div>
  <div class="layer top"><img src="{_data_uri(after)}" draggable="false"></div>
  <div class="handle"><span>&#8596;</span></div>
  <div class="tag l">{html.escape(before_label)}</div><div class="tag r">{html.escape(after_label)}</div>
  <div class="hint">scroll = zoom · drag = pan · drag handle = compare · double-click = reset</div>
</div>
<style>
 body {{ margin:0; background:transparent; font-family: Inter, -apple-system, sans-serif; }}
 .wrap {{ position:relative; max-height:{height - 10}px; max-width:100%; margin:0 auto; overflow:hidden; border-radius:12px;
   border:1px solid rgba(148,163,184,0.2); background:#05080f; user-select:none; cursor:grab; }}
 .layer {{ position:absolute; inset:0; overflow:hidden; }}
 .layer img {{ position:absolute; left:0; top:0; width:100%; height:100%; object-fit:contain; transform-origin:0 0; }}
 .top {{ clip-path: inset(0 0 0 50%); }}
 .handle {{ position:absolute; top:0; bottom:0; left:50%; width:2px; background:#22d3ee; cursor:ew-resize; }}
 .handle span {{ position:absolute; top:50%; left:-15px; width:30px; height:30px; margin-top:-15px; border-radius:50%;
   background:#0f172a; border:2px solid #22d3ee; color:#22d3ee; display:flex; align-items:center; justify-content:center; font-size:15px; }}
 .tag {{ position:absolute; top:10px; font-size:12px; padding:4px 9px; border-radius:6px; background:rgba(15,23,42,0.82);
   color:#e2e8f0; border:1px solid rgba(148,163,184,0.25); pointer-events:none; }}
 .tag.l {{ left:10px; }} .tag.r {{ right:10px; }}
 .hint {{ position:absolute; bottom:8px; left:50%; transform:translateX(-50%); font-size:11px; color:#94a3b8;
   background:rgba(15,23,42,0.75); padding:3px 9px; border-radius:6px; pointer-events:none; white-space:nowrap; }}
</style>
<script>
(function() {{
  const root = document.getElementById("{key}");
  const top = root.querySelector(".top"), handle = root.querySelector(".handle");
  const imgs = root.querySelectorAll("img");
  let split = 0.5, scale = 1, tx = 0, ty = 0, mode = null, sx = 0, sy = 0, stx = 0, sty = 0;
  function apply() {{
    top.style.clipPath = "inset(0 0 0 " + (split * 100) + "%)";
    handle.style.left = (split * 100) + "%";
    imgs.forEach(i => i.style.transform = "translate(" + tx + "px," + ty + "px) scale(" + scale + ")");
  }}
  function clampPan() {{
    const r = root.getBoundingClientRect();
    tx = Math.min(0, Math.max(r.width - r.width * scale, tx));
    ty = Math.min(0, Math.max(r.height - r.height * scale, ty));
  }}
  root.addEventListener("wheel", e => {{
    e.preventDefault();
    const r = root.getBoundingClientRect(), mx = e.clientX - r.left, my = e.clientY - r.top;
    const ns = Math.min(8, Math.max(1, scale * (e.deltaY < 0 ? 1.15 : 1 / 1.15)));
    tx = mx - (mx - tx) * ns / scale; ty = my - (my - ty) * ns / scale; scale = ns; clampPan(); apply();
  }}, {{ passive: false }});
  root.addEventListener("mousedown", e => {{
    mode = (e.target === handle || handle.contains(e.target)) ? "split" : "pan";
    sx = e.clientX; sy = e.clientY; stx = tx; sty = ty; root.style.cursor = mode === "pan" ? "grabbing" : "ew-resize";
  }});
  window.addEventListener("mousemove", e => {{
    if (!mode) return;
    const r = root.getBoundingClientRect();
    if (mode === "split") {{ split = Math.min(1, Math.max(0, (e.clientX - r.left) / r.width)); }}
    else {{ tx = stx + e.clientX - sx; ty = sty + e.clientY - sy; clampPan(); }}
    apply();
  }});
  window.addEventListener("mouseup", () => {{ mode = null; root.style.cursor = "grab"; }});
  root.addEventListener("dblclick", () => {{ scale = 1; tx = 0; ty = 0; split = 0.5; apply(); }});
  apply();
}})();
</script>"""
    components.html(markup, height=height)


def segmented(label: str, options: list, key: str, default=None, format_func=str, container=st):
    """Segmented control (a horizontal radio styled by CSS) - works in the browser and in AppTest."""
    if key not in st.session_state or st.session_state[key] not in options:
        st.session_state[key] = default if default in options else options[0]
    return container.radio(label, options, key=key, horizontal=True, format_func=format_func)


def mermaid(source: str, height: int = 460) -> None:
    """Render a Mermaid diagram (dark theme); the source text is shown if the library cannot load."""
    escaped = html.escape(source)
    components.html(f"""<div class="mermaid" style="display:flex;justify-content:center">{escaped}</div>
<pre id="fallback" style="display:none;color:#94a3b8;font-size:12px">{escaped}</pre>
<script src="https://cdn.jsdelivr.net/npm/mermaid@10.9.1/dist/mermaid.min.js"></script>
<script>
  if (window.mermaid) {{
    mermaid.initialize({{ startOnLoad: true, theme: "dark",
      themeVariables: {{ primaryColor: "#0e7490", primaryTextColor: "#e2e8f0", lineColor: "#22d3ee",
                         background: "transparent", fontFamily: "Inter, sans-serif" }} }});
  }} else {{ document.getElementById("fallback").style.display = "block"; }}
</script>""", height=height, scrolling=True)


def legend(items: list[tuple[str, str]]) -> None:
    html_block('<div class="legend">' + "".join(
        f'<span><span class="sw" style="background:{c}"></span>{html.escape(t)}</span>' for c, t in items) + "</div>")


def heat_legend(threshold: float, top: float) -> None:
    html_block(f"""<div style="margin:4px 0 10px 0"><div class="legend" style="justify-content:space-between;margin:0">
      <span>0 · normal</span><span style="color:#fbbf24">threshold {threshold:.3f} (middle)</span>
      <span>{top:.2f}+ · strongly anomalous</span></div><div class="gradient-bar"></div></div>""")
