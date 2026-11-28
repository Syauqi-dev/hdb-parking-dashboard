"""
HDB Carpark Forecast - replay dashboard (FYP2), v3: user-facing redesign.
Tabs: About | Find a slot (drivers) | Forecast & daily pattern (analysts) | Model comparison (analysts)

Reads dashboard_assets/replay_test.csv written by export_dashboard_assets.py. No model code runs here.
Needs: streamlit (>=1.51), pandas, numpy, plotly, and layout_utils.py next to this file.

HONESTY NOTES (also shown in the app):
- Free-slot COUNTS (now) are real data; the 2-hour numbers are model forecasts.
- The data has NO per-bay information, so which bays are free/taken on the map is SIMULATED (consistent with the counts).
- Everything is a replay of May 2024, not live.
"""
import json
from datetime import timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from layout_utils import BLOCKS, H, W, YC, build_layout, fill_rank, slot_from_point, taken_mask

st.set_page_config(page_title="HDB Carpark Forecast (Replay)", page_icon="🅿️", layout="wide",
                   initial_sidebar_state="collapsed")
HERE = Path(__file__).parent
ASSETS, RESULTS = HERE / "dashboard_assets", HERE / "results"
HORIZON = pd.Timedelta(hours=2)

# ---- Settings the supervisor can change -------------------------------------------------------------
USER_CARPARK = "TMBT"                 # carpark shown in "Find a slot" (smallest carpark). Falls back to the smallest in the data.
AUTHOR_LINE = "FYP2 · Electrical & Electronic Engineering · UiTM Shah Alam"  # set to "" to hide
PLENTY_FREE, FILLING_FREE = 0.30, 0.10   # share of slots free: >=30% "Plenty of space", 10-30% "Filling up", <10% "Almost full"
POSSIBLE_FROM = 0.15                     # chance of "busier than usual": <15% Unlikely, 15%..cut-off Possible, >=cut-off Likely
# -----------------------------------------------------------------------------------------------------

# Colours. Red/green = taken/free bays (Find a slot). Purple = busy level line (analyst charts), so red keeps ONE meaning.
BLUE, AMBER, BUSY, GREY, TEAL = "#3B82F6", "#F59E0B", "#A855F7", "#94A3B8", "#14B8A6"
FREE_C, TAKEN_C, ORANGE = "#22C55E", "#EF4444", "#F97316"

CSS = """
<style>
@keyframes fadein {from {opacity:0; transform:translateY(6px);} to {opacity:1; transform:none;}}
.hero {background:linear-gradient(120deg,#0f766e 0%,#1d4ed8 100%); color:#fff; padding:1.1rem 1.4rem;
       border-radius:14px; margin-bottom:.9rem; animation:fadein .5s ease both;}
.hero .t {font-size:1.65rem; font-weight:700; line-height:1.25;}
.hero .s {opacity:.92; font-size:.9rem; margin-top:.2rem;}
.badge {display:inline-block; padding:.15rem .65rem; border-radius:999px; font-size:.74rem; font-weight:600;
        background:rgba(255,255,255,.22); margin:.5rem .4rem 0 0;}
.card {border:1px solid rgba(128,128,128,.30); background:rgba(128,128,128,.09); border-radius:12px;
       padding:.8rem 1rem; margin-bottom:.6rem; animation:fadein .45s ease both;
       transition:transform .18s ease, box-shadow .18s ease;}
.card:hover {transform:translateY(-2px); box-shadow:0 6px 18px rgba(0,0,0,.14);}
.card .lbl {font-size:.74rem; opacity:.72; text-transform:uppercase; letter-spacing:.05em;}
.card .val {font-size:2.1rem; font-weight:800; line-height:1.2;}
.card .sub {font-size:.85rem; opacity:.8;}
.pill {display:inline-block; padding:.22rem .95rem; border-radius:999px; font-weight:700; font-size:1.05rem; margin:.2rem 0;}
.pill.high {background:#F59E0B; color:#1f1300;}
.pill.normal {background:#3B82F6; color:#fff;}
.pill.ok {background:#14B8A6; color:#04201c;}
.pill.bad {background:#64748B; color:#fff;}
.pill.full {background:#EF4444; color:#fff;}
.pill.likely {background:#F97316; color:#1f0a00;}
.track {position:relative; height:16px; border-radius:999px; background:rgba(128,128,128,.28); margin:.9rem 0 .5rem;}
.fill {position:absolute; left:0; top:0; bottom:0; border-radius:999px; background:#3B82F6; transition:width .5s ease;}
.mk {position:absolute; top:-5px; width:4px; height:26px; border-radius:2px; transition:left .5s ease;}
.legend {font-size:.8rem; opacity:.85;}
.legend span {margin-right:.9rem; white-space:nowrap;}
.step {display:inline-block; width:1.7rem; height:1.7rem; line-height:1.7rem; text-align:center; border-radius:50%;
       background:#0f766e; color:#fff; font-weight:700; margin-right:.5rem;}
@media (prefers-reduced-motion: reduce) {
  .hero,.card,.fill,.mk {animation:none !important; transition:none !important;}
}
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)

# Numbers below were copied from YOUR runs (test = May 2024). Update them if you re-run.
CLASSIFICATION = pd.DataFrame([
    ["ANN",          .921, .928, .884, .724, .939, .818, .981],
    ["Random Forest", .930, .939, .896, .746, .952, .837, .985],
    ["Bagging",      .936, .942, .904, .767, .950, .849, .987],
    ["Profile rule", .947, .932, .916, .827, .908, .866, .977],
    ["Persistence",  .869, .785, .786, .653, .651, .652, .912],
    ["Majority",     .811, .500, np.nan, np.nan, np.nan, np.nan, .500],
], columns=["Model", "Accuracy", "Balanced acc.", "Macro-F1", "High precision",
            "High recall", "High F1", "ROC-AUC"]).set_index("Model")

REGRESSION = pd.DataFrame([
    ["ANN",                          .0172, .0288, .972],
    ["Bagging",                      .0177, .0332, .963],
    ["Random Forest",                .0184, .0333, .962],
    ["Rule: persistence + profile change", .0178, .0309, .968],
    ["Rule: profile only",           .0277, np.nan, np.nan],
    ["Rule: same time last week",    .0361, np.nan, np.nan],
    ["Rule: same time yesterday",    .0454, np.nan, np.nan],
    ["Persistence",                  .0469, np.nan, np.nan],
], columns=["Model", "MAE", "RMSE", "R2"]).set_index("Model")

CLS_VS_REG = pd.DataFrame([
    ["ANN",           .928, .926, .981, .978],
    ["Random Forest", .939, .909, .985, .972],
    ["Bagging",       .942, .928, .987, .977],
], columns=["Model", "Bal. acc. direct", "Bal. acc. via regression",
            "AUC direct", "AUC via regression"]).set_index("Model")


@st.cache_data
def load_replay():
    f = ASSETS / "replay_test.csv"
    if not f.exists():
        return None, None
    df = pd.read_csv(f, parse_dates=["t0", "target_time"])
    df = df.sort_values(["carpark", "t0"]).reset_index(drop=True)
    meta = json.loads((ASSETS / "meta.json").read_text()) if (ASSETS / "meta.json").exists() else {}
    return df, meta


@st.cache_data
def load_controls():
    rows = []
    for name, f in [("ANN", "ann"), ("Random Forest", "rf"), ("Bagging", "bag")]:
        p = RESULTS / f"{f}_regression_control.csv"
        if p.exists():
            r = pd.read_csv(p).iloc[0]
            rows.append([name, r.real_mae, r.shuffled_mae_mean, r.shuffled_mae_mean / r.real_mae,
                         r.real_r2, r.shuffled_r2_mean, int(r.n_shuffles)])
    return pd.DataFrame(rows, columns=["Model", "Real MAE", "Shuffled-target MAE", "Shuffled / real",
                                       "Real R2", "Shuffled-target R2", "Shuffles"]).set_index("Model")


def auc(y, s):  # rank-based ROC-AUC, no sklearn needed
    y, s = np.asarray(y), np.asarray(s)
    r = pd.Series(s).rank().values
    n1, n0 = (y == 1).sum(), (y == 0).sum()
    return np.nan if n1 == 0 or n0 == 0 else (r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)


def cls_metrics(y, pred, score=None):
    y, pred = np.asarray(y), np.asarray(pred)
    tp, fn = ((pred == 1) & (y == 1)).sum(), ((pred == 0) & (y == 1)).sum()
    tn, fp = ((pred == 0) & (y == 0)).sum(), ((pred == 1) & (y == 0)).sum()
    rec, spec = tp / max(tp + fn, 1), tn / max(tn + fp, 1)
    return {"Balanced acc.": (rec + spec) / 2, "High recall": rec,
            "High precision": tp / max(tp + fp, 1), "ROC-AUC": auc(y, score) if score is not None else np.nan,
            "Busy periods found": int(tp), "Busy periods missed": int(fn), "False alarms": int(fp)}


pct = lambda x: f"{100 * x:.0f}%"


def is_dark():
    try:
        return st.context.theme.type == "dark"
    except Exception:
        return False


def card(label, value, sub=""):
    return f'<div class="card"><div class="lbl">{label}</div><div class="val">{value}</div><div class="sub">{sub}</div></div>'


def pill_card(label, text, kind, sub=""):
    return (f'<div class="card"><div class="lbl">{label}</div>'
            f'<div><span class="pill {kind}">{text}</span></div><div class="sub">{sub}</div></div>')


def occupancy_bar(now, fc, thr):
    c = lambda x: max(0, min(100, 100 * x))
    return (f'<div class="track"><div class="fill" style="width:{c(now):.1f}%"></div>'
            f'<div class="mk" style="left:{c(thr):.1f}%; background:{BUSY}"></div>'
            f'<div class="mk" style="left:{c(fc):.1f}%; background:{AMBER}"></div></div>'
            f'<div class="legend"><span><b style="color:{BLUE}">■</b> now {pct(now)}</span>'
            f'<span><b style="color:{AMBER}">▮</b> forecast in 2 h {pct(fc)}</span>'
            f'<span><b style="color:{BUSY}">▮</b> busy level {pct(thr)}</span></div>')


def busy_gauge(prob, cutoff):
    """Plain-language gauge: chance of being among the carpark's busiest times in 2 hours."""
    col = ORANGE if prob >= cutoff else (AMBER if prob >= POSSIBLE_FROM else TEAL)
    fig = go.Figure(go.Indicator(
        mode="gauge+number", value=prob * 100, number={"suffix": "%", "font": {"size": 32}},
        gauge={"axis": {"range": [0, 100], "ticksuffix": "%"}, "bar": {"color": col, "thickness": 0.3},
               "steps": [{"range": [0, POSSIBLE_FROM * 100], "color": "rgba(20,184,166,.22)"},
                         {"range": [POSSIBLE_FROM * 100, cutoff * 100], "color": "rgba(245,158,11,.24)"},
                         {"range": [cutoff * 100, 100], "color": "rgba(249,115,22,.30)"}]}))
    fig.update_layout(height=190, margin=dict(l=20, r=20, t=15, b=0))
    return fig


def status_of(free_share):
    if free_share >= PLENTY_FREE:
        return "Plenty of space", "ok"
    if free_share >= FILLING_FREE:
        return "Filling up", "high"
    return "Almost full", "full"


def likelihood_of(prob, cutoff):
    if prob >= cutoff:
        return "Likely", "likely"
    if prob >= POSSIBLE_FROM:
        return "Possible", "high"
    return "Unlikely", "ok"


def day_is_weekend(ts):
    return ts.dayofweek >= 5


def typical_free(d, weekend, total):
    """Average free slots by half-hour of day (May 2024), for weekdays or weekends/holidays."""
    p = d.copy()
    p["wk"] = (p.t0.dt.dayofweek >= 5) | (p.is_holiday == 1)
    p["s"] = p.t0.dt.hour * 2 + p.t0.dt.minute // 30
    g = p[p.wk == weekend].groupby("s").occ_now.mean().reindex(range(48)).interpolate(limit_direction="both")
    return total * (1 - g)


def layout_figure(labels, xs, ys, taken, selected):
    fig = go.Figure()
    for x0, y0, x1, y1, txt in BLOCKS:
        fig.add_shape(type="rect", x0=x0, y0=y0, x1=x1, y1=y1, fillcolor="rgba(128,128,128,.28)", line_width=0, layer="below")
        if txt:
            fig.add_annotation(x=(x0 + x1) / 2, y=(y0 + y1) / 2, text=txt, showarrow=False, font=dict(size=10))
    fig.add_shape(type="line", x0=24, x1=76, y0=YC, y1=YC, line=dict(color="rgba(128,128,128,.6)", width=2), layer="below")
    for (x, y, t) in [(50, H - 11.5, "◀  ◀  ◀  ◀  ◀"), (50, 11.5, "▶  ▶  ▶  ▶  ▶"), (14, YC, "▲\n▼"), (86, YC, "▲\n▼")]:
        fig.add_annotation(x=x, y=y, text=t, showarrow=False, font=dict(size=11, color="rgba(128,128,128,.9)"))
    free_i, taken_i = np.where(~taken)[0], np.where(taken)[0]
    for idx, name, colr in [(free_i, "Free bay", FREE_C), (taken_i, "Taken bay", TAKEN_C)]:
        fig.add_trace(go.Scatter(
            x=xs[idx], y=ys[idx], mode="markers", name=f"{name} ({len(idx)})", customdata=labels[idx],
            marker=dict(symbol="square", size=12, color=colr, line=dict(width=0)),
            hovertemplate="<b>Bay %{customdata}</b><br>" + name + "<extra></extra>",
            selected=dict(marker=dict(opacity=1)), unselected=dict(marker=dict(opacity=1))))
    # second cue for colour-blind users: taken bays carry a small white cross
    fig.add_trace(go.Scatter(x=xs[taken_i], y=ys[taken_i], mode="markers", showlegend=False, hoverinfo="skip",
                             marker=dict(symbol="x-thin", size=7, color="white", line=dict(width=1.5, color="white"))))
    if selected is not None:
        fig.add_trace(go.Scatter(x=[xs[selected]], y=[ys[selected]], mode="markers", name="Chosen bay", hoverinfo="skip",
                                 marker=dict(symbol="square-open", size=21, color=BLUE, line=dict(width=3, color=BLUE))))
    fig.update_xaxes(range=[-1, W + 1], visible=False, fixedrange=True)
    fig.update_yaxes(range=[-1, H + 1], visible=False, fixedrange=True)
    fig.update_layout(height=470, margin=dict(l=5, r=5, t=5, b=5), legend=dict(orientation="h", y=-0.02),
                      dragmode="select", clickmode="event+select")
    return fig


df, meta = load_replay()
if df is None:
    st.error("dashboard_assets/replay_test.csv not found. Run `python export_dashboard_assets.py --data <dataset.csv>` first.")
    st.stop()

cutoff = meta.get("cutoff", 0.5)
carparks = sorted(df["carpark"].unique())

st.markdown(
    f'<div class="hero"><div class="t">🅿️ Find a parking slot · HDB carpark forecast</div>'
    f'<div class="s">See how many slots are free now and how many should be free in 2 hours. {AUTHOR_LINE}</div>'
    f'<span class="badge">REPLAY · May 2024 data</span><span class="badge">Prototype, not live</span></div>',
    unsafe_allow_html=True)

tab_about, tab_user, tab_an, tab_cmp = st.tabs(["About", "Find a slot", "Forecast & daily pattern", "Model comparison"])

# ============================== Tab 1: About / how to use ==============================
with tab_about:
    st.subheader("How to use this app")
    st.markdown(
        '<span class="step">1</span> Open the **Find a slot** tab.<br>'
        '<span class="step">2</span> Pick a date and time. This is a **replay of May 2024**, so you choose the "now" moment.<br>'
        '<span class="step">3</span> Read the big numbers: **slots free now** and **slots expected free in 2 hours**.<br>'
        '<span class="step">4</span> Tap any bay on the map to see if it is free now and likely free in 2 hours.',
        unsafe_allow_html=True)
    st.markdown("")
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("#### What you will see")
        st.markdown("- **Free slots now:** from the carpark's real counts.\n"
                    "- **Free slots in 2 hours:** a machine-learning forecast, with how far off it usually is.\n"
                    "- **Chance of being busier than usual:** how likely the carpark is to be among its busiest times.\n"
                    "- **Quieter times:** when this carpark is usually emptier (a typical pattern, not a forecast).")
    with c2:
        st.markdown("#### Good to know")
        st.markdown("- **Busier than usual** means among this carpark's busiest 1 in 5 times. It does not mean full.\n"
                    "- **The bay map is a demo.** The data says how many slots are free, not which ones. "
                    "Bay colours are simulated to match the real count.\n"
                    "- **Replay only.** It uses past data (1 Mar - 31 May 2024), not live carpark data.")
    st.info("The other two tabs (Forecast & daily pattern, Model comparison) are for supervisors and examiners.")
    with st.expander("For examiners: question, data, method and limits"):
        st.markdown(f"""
**Question answered:** at a forecast time, will a carpark be **HIGH** (above its own 80th-percentile occupancy, learned from the training period) **2 hours later**? The same models also forecast the occupancy number.

**Data:** Singapore HDB Carpark Availability (data.gov.sg), 18 carparks, 30-minute snapshots, 1 Mar-31 May 2024.
**Split (by target time):** train 8 Mar-15 Apr, validation 16-30 Apr (cut-off tuned here), test 1-31 May (never used for tuning).
**Models shown here:** Bagging classifier + Bagging regressor, settings fixed in advance. Cut-off used: {cutoff:.2f}.

**Limits**
- Replay of past data, not a live forecast. One test month, one run per model, no confidence intervals.
- A strong rule that only knows the usual daily/weekly pattern is about as good; the models find more busy periods at the cost of more false alarms.
- HIGH is relative to each carpark's own history. All 18 carparks are HDB (residential-type), so results may not carry over to malls or offices.
- Validated at a 2-hour horizon only.
- Per-bay status is not in the data: the map in Find a slot is simulated from the counts.

Export created: {meta.get('created', 'unknown')} (scikit-learn {meta.get('sklearn', '?')}).
""")

# ============================== Tab 2: Find a slot (drivers) ==============================
with tab_user:
    totals = df.groupby("carpark").total_lots.first()
    ucp = USER_CARPARK if USER_CARPARK in totals.index else totals.idxmin()
    if ucp != USER_CARPARK:
        st.warning(f"Carpark {USER_CARPARK} was not found in the data, so the smallest carpark ({ucp}) is shown.")
    ud = df[df.carpark == ucp].reset_index(drop=True)
    total = int(ud.total_lots.iloc[0])
    udmin, udmax = ud.t0.dt.date.min(), ud.t0.dt.date.max()

    st.subheader(f"Carpark {ucp} · {total} slots")
    st.markdown('<span class="step">1</span> **Pick when you want to check** (replay of May 2024)', unsafe_allow_html=True)

    def u_jump(kind):
        q = ud.occ_now.quantile(.9 if kind == "busy" else .1)
        pool = ud[ud.occ_now >= q] if kind == "busy" else ud[ud.occ_now <= q]
        r = pool.sample(1).iloc[0]
        st.session_state["u_day"], st.session_state["u_time"] = r.t0.date(), r.t0.strftime("%H:%M")

    if "u_day" not in st.session_state or not (udmin <= st.session_state["u_day"] <= udmax):
        st.session_state["u_day"] = udmin + timedelta(days=10)
    k1, k2, k3, k4 = st.columns([1.2, 1, 1, 1])
    uday = k1.date_input("Date", min_value=udmin, max_value=udmax, key="u_day")
    uslots = ud[ud.t0.dt.date == uday].t0.dt.strftime("%H:%M").tolist()
    if not uslots:
        st.warning("No data for that date."); st.stop()
    if st.session_state.get("u_time") not in uslots:
        st.session_state["u_time"] = "18:00" if "18:00" in uslots else uslots[0]
    utime = k2.selectbox("Time", uslots, key="u_time")
    k3.markdown("<div style='height:1.7rem'></div>", unsafe_allow_html=True)
    k3.button("Show a busy time", on_click=u_jump, args=("busy",))
    k4.markdown("<div style='height:1.7rem'></div>", unsafe_allow_html=True)
    k4.button("Show a quiet time", on_click=u_jump, args=("quiet",))

    ur = ud[ud.t0 == pd.Timestamp(f"{uday} {utime}")].iloc[0]
    free_now = int(np.clip(round(total * (1 - ur.occ_now)), 0, total))
    free_fc = int(np.clip(round(total * (1 - ur.pred_occ)), 0, total))
    err = (ud.pred_occ - ud.actual_occ).abs() * total
    e_typ, e_90 = max(1, int(round(err.median()))), max(1, int(round(err.quantile(.9))))
    s_now, k_now = status_of(free_now / total)
    s_fc, k_fc = status_of(free_fc / total)
    lik, k_lik = likelihood_of(ur.prob_high, cutoff)
    diff = free_fc - free_now

    st.markdown('<span class="step">2</span> **Read the numbers**', unsafe_allow_html=True)
    c1, c2, c3 = st.columns([1, 1, 1.15])
    c1.markdown(card("Free slots right now", f"{free_now}",
                     f"of {total} slots · {100 * free_now / total:.0f}% free") +
                f'<span class="pill {k_now}">{s_now}</span>', unsafe_allow_html=True)
    c2.markdown(card("Free slots in 2 hours (forecast)", f"about {free_fc}",
                     (f"▲ {diff} more than now" if diff > 0 else f"▼ {-diff} fewer than now" if diff < 0 else "about the same as now") +
                     f" · usually within ±{e_typ} slots") +
                f'<span class="pill {k_fc}">{s_fc}</span>', unsafe_allow_html=True)
    with c3:
        st.markdown("**Will it be one of its busiest times in 2 hours?**")
        st.plotly_chart(busy_gauge(ur.prob_high, cutoff), width="stretch", key="u_gauge")
        st.markdown(f'<span class="pill {k_lik}">{lik}</span> that it is among its busiest 1 in 5 times (not the same as full).',
                    unsafe_allow_html=True)
    st.caption(f"How reliable? In May 2024 the 2-hour forecast for {ucp} was usually within about ±{e_typ} slots, "
               f"and 9 times in 10 within ±{e_90} slots. Replay data, not live.")

    st.markdown('<span class="step">3</span> **When is it usually quieter?** (typical pattern, not a forecast)', unsafe_allow_html=True)
    weekend = bool(day_is_weekend(ur.t0) or ur.is_holiday == 1)
    tf = typical_free(ud, weekend, total)
    s0 = ur.t0.hour * 2 + ur.t0.minute // 30
    ahead = [(s0 + k) % 48 for k in range(1, 25)]
    best = max(ahead, key=lambda s: tf[s])
    q1, q2 = st.columns([1, 1.5])
    q1.markdown(card("Usually most free around", f"{best // 2:02d}:{(best % 2) * 30:02d}",
                     f"in the next 12 hours · typically about {int(round(tf[best]))} free slots"), unsafe_allow_html=True)
    q1.caption(f"Based on the average {'weekend / holiday' if weekend else 'weekday'} in May 2024.")
    figt = go.Figure()
    figt.add_trace(go.Scatter(x=[s / 2 for s in range(48)], y=tf.values, mode="lines", name="Typical free slots",
                              line=dict(width=3, color=TEAL)))
    figt.add_trace(go.Scatter(x=[s0 / 2], y=[tf[s0]], mode="markers", name="Now",
                              marker=dict(size=12, color=BLUE, line=dict(width=2, color="white"))))
    figt.add_trace(go.Scatter(x=[((s0 + 4) % 48) / 2], y=[tf[(s0 + 4) % 48]], mode="markers", name="In 2 hours",
                              marker=dict(size=13, symbol="diamond", color=AMBER, line=dict(width=2, color="white"))))
    figt.update_layout(height=230, margin=dict(l=10, r=10, t=10, b=10), xaxis_title="Hour of day", yaxis_title="Typical free slots",
                       legend=dict(orientation="h", y=-0.35))
    q2.plotly_chart(figt, width="stretch", key="u_typical")

    st.markdown('<span class="step">4</span> **Check a bay on the map**', unsafe_allow_html=True)
    view = st.radio("Show the map for:", ["Now", "In 2 hours (forecast)"], horizontal=True, key="u_view")
    labels, xs, ys = build_layout(total)
    rank = fill_rank(total, ucp)
    taken_now, taken_fc = taken_mask(rank, total - free_now), taken_mask(rank, total - free_fc)
    taken = taken_now if view == "Now" else taken_fc
    labels_list = list(labels)
    sel_label = st.session_state.get("u_slot")
    sel_idx = labels_list.index(sel_label) if sel_label in labels_list else None
    ev = st.plotly_chart(layout_figure(labels, xs, ys, taken, sel_idx), width="stretch", key="u_map",
                         on_select="rerun", selection_mode="points")
    # a click on the map selects that bay (guarded so it only fires once per new click)
    try:
        pts = ev["selection"]["points"]
    except Exception:
        pts = []
    if pts:
        pt = pts[-1]
        sig = (round(float(pt.get("x", 0)), 2), round(float(pt.get("y", 0)), 2))
        if sig != st.session_state.get("_last_click"):
            j = slot_from_point(float(pt["x"]), float(pt["y"]), xs, ys)
            st.session_state["_last_click"] = sig
            if j is not None and labels_list[j] != sel_label:
                st.session_state["u_slot"] = labels_list[j]
                st.rerun()
    st.selectbox("Or pick a bay from the list", labels_list, index=None, placeholder="Tap a bay on the map, or choose here",
                 key="u_slot")
    sel_label = st.session_state.get("u_slot")
    if sel_label in labels_list:
        j = labels_list.index(sel_label)
        n_txt = "Taken" if taken_now[j] else "Free"
        f_txt = "likely Taken" if taken_fc[j] else "likely Free"
        b1, b2, b3 = st.columns(3)
        b1.markdown(card("Chosen bay", sel_label), unsafe_allow_html=True)
        b2.markdown(pill_card("Right now", ("❌ " if taken_now[j] else "✅ ") + n_txt, "full" if taken_now[j] else "ok"),
                    unsafe_allow_html=True)
        b3.markdown(pill_card("In 2 hours", ("❌ " if taken_fc[j] else "✅ ") + f_txt, "full" if taken_fc[j] else "ok"),
                    unsafe_allow_html=True)
    else:
        st.info("Tap any bay on the map (or choose one from the list) to see if it is free now and likely free in 2 hours.")
    st.caption("🟢 Free bay · 🔴 Taken bay (marked with a white cross). **Demo layout:** the data only tells us how many slots are free, "
               "not which ones, so bay positions and colours are simulated to match the real count now and the forecast count in 2 hours.")

    with st.expander("Replay check: what actually happened 2 hours later (not available in a live app)"):
        free_act = int(np.clip(round(total * (1 - ur.actual_occ)), 0, total))
        x1, x2, x3 = st.columns(3)
        x1.markdown(card("Forecast", f"about {free_fc}", "free slots"), unsafe_allow_html=True)
        x2.markdown(card("Actual", f"{free_act}", "free slots"), unsafe_allow_html=True)
        x3.markdown(card("Difference", f"{abs(free_fc - free_act)}", "slots"), unsafe_allow_html=True)

# ============================== Tab 3: Forecast & daily pattern (analysts) ==============================
with tab_an:
    st.caption("For supervisors and examiners. Pick any of the 18 carparks.")
    cp = st.selectbox("Carpark", carparks, key="cp")
    d = df[df["carpark"] == cp].reset_index(drop=True)
    dmin, dmax = d["t0"].dt.date.min(), d["t0"].dt.date.max()

    def jump(kind):
        masks = {"Correct HIGH call": (d.pred_high == 1) & (d.actual_high == 1),
                 "False alarm": (d.pred_high == 1) & (d.actual_high == 0),
                 "Missed busy period": (d.pred_high == 0) & (d.actual_high == 1)}
        pool = d[masks[kind]]
        if len(pool):
            r = pool.sample(1).iloc[0]
            st.session_state["day"], st.session_state["time"] = r.t0.date(), r.t0.strftime("%H:%M")

    if "day" not in st.session_state or not (dmin <= st.session_state["day"] <= dmax):
        st.session_state["day"] = dmin + timedelta(days=10)
    a1, a2, a3, a4 = st.columns([1.2, 1, 1.2, 1])
    day = a1.date_input("Date (forecast made at)", min_value=dmin, max_value=dmax, key="day")
    slots = d[d["t0"].dt.date == day]["t0"].dt.strftime("%H:%M").tolist()
    if not slots:
        st.warning("No data for this carpark on that date."); st.stop()
    if st.session_state.get("time") not in slots:
        st.session_state["time"] = "18:00" if "18:00" in slots else slots[0]
    tstr = a2.selectbox("Time (forecast made at)", slots, key="time")
    kind = a3.selectbox("Jump to an example", ["Correct HIGH call", "False alarm", "Missed busy period"], key="kind")
    a4.markdown("<div style='height:1.7rem'></div>", unsafe_allow_html=True)
    a4.button("Jump (random example)", on_click=jump, args=(kind,))
    row = d[d["t0"] == pd.Timestamp(f"{day} {tstr}")].iloc[0]

    st.subheader("Forecast")
    high_pred, high_true = row.pred_high == 1, row.actual_high == 1
    ok = row.pred_high == row.actual_high
    c1, c2, c3 = st.columns(3)
    c1.markdown(card("Occupancy now", pct(row.occ_now), f"≈ {int(round(row.total_lots * (1 - row.occ_now)))} of {int(row.total_lots)} lots free"),
                unsafe_allow_html=True)
    c2.markdown(card("Forecast in 2 h", pct(row.pred_occ), f"{100 * (row.pred_occ - row.occ_now):+.0f} points vs now"), unsafe_allow_html=True)
    c3.markdown(pill_card("Model verdict", "HIGH" if high_pred else "NOT HIGH", "high" if high_pred else "normal",
                          f"Chance of HIGH: {100 * row.prob_high:.1f}% (cut-off {cutoff:.0%}) · HIGH = above this carpark's own busy level ({pct(row.thr)})"),
                unsafe_allow_html=True)
    st.markdown(f"**{cp}** · {row.t0:%a %d %b %Y, %H:%M}")
    st.markdown(occupancy_bar(row.occ_now, row.pred_occ, row.thr), unsafe_allow_html=True)
    st.caption("HIGH means busier than this carpark's own 80th percentile in the training period. It does not mean full.")

    st.markdown("##### What actually happened (revealed because this is a replay)")
    e1, e2, e3, e4 = st.columns(4)
    e1.markdown(card("Actual occupancy at +2 h", pct(row.actual_occ)), unsafe_allow_html=True)
    e2.markdown(pill_card("Actual outcome", "HIGH" if high_true else "NOT HIGH", "high" if high_true else "normal"), unsafe_allow_html=True)
    e3.markdown(pill_card("Model call", "✅ Correct" if ok else "❌ Wrong", "ok" if ok else "bad"), unsafe_allow_html=True)
    rule_ok = row.rule_high == row.actual_high
    e4.markdown(pill_card("Simple rule's call", ("✅ " if rule_ok else "❌ ") + ("HIGH" if row.rule_high == 1 else "NOT HIGH"),
                          "ok" if rule_ok else "bad", "occupancy now + usual change > busy level"), unsafe_allow_html=True)

    win = d[(d.t0 >= row.t0 - pd.Timedelta(hours=12)) & (d.t0 <= row.t0 + pd.Timedelta(hours=12))]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=win.t0, y=win.occ_now * 100, mode="lines", name="Actual occupancy", line=dict(width=3, color=BLUE)))
    fig.add_hline(y=row.thr * 100, line_dash="dash", line_color=BUSY, opacity=.9)
    fig.add_vrect(x0=row.t0, x1=row.t0 + HORIZON, fillcolor=GREY, opacity=.18, line_width=0)
    fig.add_trace(go.Scatter(x=[row.t0], y=[row.occ_now * 100], mode="markers", name="Forecast made here",
                             marker=dict(size=12, color=BLUE, line=dict(width=2, color="white"))))
    fig.add_trace(go.Scatter(x=[row.t0 + HORIZON], y=[row.pred_occ * 100], mode="markers", name="Model forecast (+2 h)",
                             marker=dict(size=15, symbol="diamond", color=AMBER, line=dict(width=2, color="white"))))
    fig.add_trace(go.Scatter(x=[None], y=[None], mode="lines", name=f"Busy level ({pct(row.thr)})", line=dict(color=BUSY, dash="dash")))
    fig.update_layout(height=370, margin=dict(l=10, r=10, t=10, b=10), yaxis_title="Occupancy (%)",
                      yaxis_range=[0, 100], legend=dict(orientation="h", y=-0.2))
    st.plotly_chart(fig, width="stretch")
    st.caption("Shaded band = the 2-hour forecast horizon. The line to the right of the blue dot is the real future, shown only because this is a replay.")

    st.subheader(f"Daily pattern of {cp} (May 2024)")
    p = d.copy()
    p["hour"] = p.t0.dt.hour + p.t0.dt.minute / 60
    p["kind"] = np.where((p.t0.dt.dayofweek >= 5) | (p.is_holiday == 1), "Weekend / holiday", "Weekday")
    g = p.groupby(["kind", "hour"]).occ_now.mean().reset_index()
    fig2 = go.Figure()
    for k, colr in [("Weekday", BLUE), ("Weekend / holiday", AMBER)]:
        gg = g[g.kind == k]
        if len(gg):
            fig2.add_trace(go.Scatter(x=gg.hour, y=gg.occ_now * 100, mode="lines", name=k, line=dict(width=3, color=colr)))
    fig2.add_hline(y=row.thr * 100, line_dash="dash", line_color=BUSY, opacity=.9)
    fig2.update_layout(height=320, xaxis_title="Hour of day", yaxis_title="Mean occupancy (%)",
                       margin=dict(l=10, r=10, t=10, b=10), legend=dict(orientation="h", y=-0.25))
    st.plotly_chart(fig2, width="stretch")
    hm = p.assign(h=p.t0.dt.hour, dow=p.t0.dt.day_name().str[:3]).pivot_table(index="dow", columns="h", values="occ_now", aggfunc="mean")
    hm = hm.reindex(["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"])
    fig3 = go.Figure(go.Heatmap(z=hm.values * 100, x=hm.columns, y=hm.index, colorbar=dict(title="%"),
                                colorscale="Inferno" if is_dark() else "YlOrRd"))
    fig3.update_layout(height=290, xaxis_title="Hour of day", margin=dict(l=10, r=10, t=10, b=10))
    st.plotly_chart(fig3, width="stretch")
    st.caption("Why this matters: the daily and weekly rhythm is strong, so a rule that knows the usual pattern is a tough baseline.")

# ============================== Tab 4: Model comparison (analysts, last) ==============================
with tab_cmp:
    st.subheader("Busy periods in May: found, missed, false alarms")
    ma, mb = cls_metrics(df.actual_high, df.pred_high, df.prob_high), cls_metrics(df.actual_high, df.rule_high)
    for title, m in [("Bagging classifier", ma), ("Simple profile rule", mb)]:
        st.markdown(f"**{title}**")
        k1, k2, k3, k4 = st.columns(4)
        k1.markdown(card("Found", f"{m['Busy periods found']:,}", f"recall {m['High recall']:.2f}"), unsafe_allow_html=True)
        k2.markdown(card("Missed", f"{m['Busy periods missed']:,}", "busy, but called NOT HIGH"), unsafe_allow_html=True)
        k3.markdown(card("False alarms", f"{m['False alarms']:,}", f"precision {m['High precision']:.2f}"), unsafe_allow_html=True)
        k4.markdown(card("Balanced accuracy", f"{m['Balanced acc.']:.3f}"), unsafe_allow_html=True)
    st.caption("The model finds more busy periods than the rule but raises more false alarms. Which is better depends on "
               "whether a miss or a false alarm costs the driver more. Recomputed balanced accuracy can differ by about 0.002 "
               "from the notebook table below (cut-off search detail).")

    st.subheader("Classification: will the carpark be HIGH in 2 hours? (test = May 2024)")
    st.dataframe(CLASSIFICATION.style.format("{:.3f}", na_rep="-"), width="stretch")
    st.info("The three models roughly tie a strong lookup rule: they find more busy periods (higher recall) but raise more false alarms "
            "(lower precision). Differences are about the size of week-to-week variation, so there is no firm ranking.")

    st.subheader("Regression: how full will it be in 2 hours?")
    st.dataframe(REGRESSION.style.format("{:.4f}", na_rep="-"), width="stretch")
    st.warning("The ANN's MAE (0.0172) vs the best rule (0.0178) is a gap of 0.06 occupancy points from one seed and one test month. "
               "Treat it as a tie until a multi-seed check says otherwise.")

    st.subheader("Same yes/no question: direct classification vs regression-then-threshold")
    st.dataframe(CLS_VS_REG.style.format("{:.3f}"), width="stretch")
    st.caption("Predicting the number first did not help the yes/no decision. Single test month, one run; regressors had no class weights.")

    st.subheader("Control: regression with shuffled targets")
    ctl = load_controls()
    if len(ctl):
        st.dataframe(ctl.style.format({"Real MAE": "{:.4f}", "Shuffled-target MAE": "{:.4f}", "Shuffled / real": "{:.1f}x",
                                       "Real R2": "{:.3f}", "Shuffled-target R2": "{:.3f}"}), width="stretch")
        st.caption("Shuffling the targets destroys any real link between features and outcome. Errors rise roughly 7-8x and R2 falls to about 0 "
                   "(trees), so the real scores are not an artefact. The ANN control used only 3 shuffles and its mean R2 is +0.09, not about 0; "
                   "inspect the per-shuffle values before relying on it. This control shows the models beat chance, not that they beat the rule.")

    st.subheader("Balanced accuracy by carpark (Bagging classifier)")
    per = (df.groupby("carpark").apply(lambda g: cls_metrics(g.actual_high, g.pred_high)["Balanced acc."], include_groups=False)
           .rename("Balanced accuracy").reset_index())
    figb = go.Figure(go.Bar(x=per.carpark, y=per["Balanced accuracy"], marker_color=TEAL))
    figb.update_layout(height=280, yaxis_range=[0.5, 1], margin=dict(l=10, r=10, t=10, b=10), yaxis_title="Balanced accuracy")
    st.plotly_chart(figb, width="stretch")
