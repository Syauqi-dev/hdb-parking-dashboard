"""
HDB Carpark High-Occupancy Forecast - replay dashboard (FYP2).
Reads dashboard_assets/replay_test.csv written by export_dashboard_assets.py.
Needs only streamlit, pandas, numpy, plotly (no TensorFlow, no scikit-learn).
"""
import json
from datetime import timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(page_title="HDB Carpark Forecast (Replay)", page_icon="🅿️", layout="wide")
HERE = Path(__file__).parent
ASSETS, RESULTS = HERE / "dashboard_assets", HERE / "results"
HORIZON = pd.Timedelta(hours=2)

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
df, meta = load_replay()
if df is None:
    st.error("dashboard_assets/replay_test.csv not found. Run `python export_dashboard_assets.py --data <dataset.csv>` first.")
    st.stop()

cutoff = meta.get("cutoff", np.nan)
lots_ok = df["total_lots"].median() >= 100  # only show lot counts if the column looks like real lots

# ------------------------------ sidebar: pick a moment ------------------------------
st.sidebar.header("Pick a moment to replay")
carparks = sorted(df["carpark"].unique())
cp = st.sidebar.selectbox("Carpark", carparks, key="cp")
d = df[df["carpark"] == cp].reset_index(drop=True)
dmin, dmax = d["t0"].dt.date.min(), d["t0"].dt.date.max()


def jump(kind):
    """Callback: move the date/time pickers to an example of the chosen outcome."""
    masks = {"Correct HIGH call": (d.pred_high == 1) & (d.actual_high == 1),
             "False alarm": (d.pred_high == 1) & (d.actual_high == 0),
             "Missed busy period": (d.pred_high == 0) & (d.actual_high == 1)}
    pool = d[masks[kind]]
    if len(pool):
        row = pool.sample(1).iloc[0]
        st.session_state["day"] = row.t0.date()
        st.session_state["time"] = row.t0.strftime("%H:%M")


if "day" not in st.session_state or not (dmin <= st.session_state["day"] <= dmax):
    st.session_state["day"] = dmin + timedelta(days=10)
day = st.sidebar.date_input("Date (forecast made at)", min_value=dmin, max_value=dmax, key="day")
slots = d[d["t0"].dt.date == day]["t0"].dt.strftime("%H:%M").tolist()
if not slots:
    st.sidebar.warning("No data for this carpark on that date."); st.stop()
if st.session_state.get("time") not in slots:
    st.session_state["time"] = "18:00" if "18:00" in slots else slots[0]
tstr = st.sidebar.selectbox("Time (forecast made at)", slots, key="time")

st.sidebar.markdown("**Jump to an example**")
kind = st.sidebar.selectbox("Outcome type", ["Correct HIGH call", "False alarm", "Missed busy period"], key="kind")
st.sidebar.button("Jump (random example)", on_click=jump, args=(kind,))

row = d[d["t0"] == pd.Timestamp(f"{day} {tstr}")].iloc[0]

st.title("🅿️ HDB Carpark High-Occupancy Forecast")
st.caption("REPLAY of May 2024 test data. Not live. The model sees only information available at the forecast time.")

tab1, tab2, tab3, tab4 = st.tabs(["Forecast", "Daily pattern", "Model comparison", "About"])

# ------------------------------ Tab 1: forecast ------------------------------
with tab1:
    verdict = "HIGH" if row.pred_high == 1 else "NOT HIGH"
    truth = "HIGH" if row.actual_high == 1 else "NOT HIGH"
    ok = row.pred_high == row.actual_high
    c = st.columns(4)
    c[0].metric("Occupancy now", pct(row.occ_now))
    c[1].metric("Forecast in 2 h", pct(row.pred_occ), f"{100 * (row.pred_occ - row.occ_now):+.0f} pts")
    c[2].metric("Chance of HIGH in 2 h", pct(row.prob_high), help=f"Model says HIGH when this is at least {cutoff:.0%} (cut-off tuned on April validation data).")
    c[3].metric("Model verdict", verdict)

    st.markdown(f"**{cp}** on **{row.t0:%a %d %b %Y, %H:%M}**: the carpark is {pct(row.occ_now)} full. "
                f"HIGH here means above this carpark's own busy level of **{pct(row.thr)}** "
                f"(its 80th percentile in the training period), not that it is full.")
    if lots_ok:
        st.markdown(f"About **{int(round(row.total_lots * (1 - row.occ_now)))}** of {int(row.total_lots)} lots free now.")

    st.subheader("What actually happened (revealed because this is a replay)")
    a = st.columns(4)
    a[0].metric("Actual occupancy at +2 h", pct(row.actual_occ))
    a[1].metric("Actual outcome", truth)
    a[2].metric("Model call", "✅ correct" if ok else "❌ wrong")
    a[3].metric("Simple rule's call", ("HIGH" if row.rule_high == 1 else "NOT HIGH") + (" ✅" if row.rule_high == row.actual_high else " ❌"),
                help="Rule: occupancy now + typical change for this time of week > the busy level.")

    win = d[(d.t0 >= row.t0 - pd.Timedelta(hours=12)) & (d.t0 <= row.t0 + pd.Timedelta(hours=12))]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=win.t0, y=win.occ_now * 100, mode="lines", name="Actual occupancy", line=dict(width=2)))
    fig.add_hline(y=row.thr * 100, line_dash="dash", line_color="crimson", opacity=.7)
    fig.add_vrect(x0=row.t0, x1=row.t0 + HORIZON, fillcolor="grey", opacity=.12, line_width=0)
    fig.add_trace(go.Scatter(x=[row.t0], y=[row.occ_now * 100], mode="markers", name="Forecast made here",
                             marker=dict(size=12, symbol="circle", color="black")))
    fig.add_trace(go.Scatter(x=[row.t0 + HORIZON], y=[row.pred_occ * 100], mode="markers", name="Model forecast (+2 h)",
                             marker=dict(size=14, symbol="diamond", color="orange")))
    fig.add_trace(go.Scatter(x=[row.t0, row.t0 + HORIZON], y=[None, None], mode="lines", name=f"Busy level ({pct(row.thr)}, dashed)",
                             line=dict(color="crimson", dash="dash")))
    fig.update_layout(height=380, margin=dict(l=10, r=10, t=10, b=10), yaxis_title="Occupancy (%)",
                      yaxis_range=[0, 100], legend=dict(orientation="h", y=-0.2))
    st.plotly_chart(fig, width="stretch")
    st.caption("Grey band = the 2-hour forecast horizon. The line to the right of the black dot is the real future, shown only because this is a replay.")

# ------------------------------ Tab 2: daily pattern ------------------------------
with tab2:
    st.subheader(f"Typical rhythm of {cp} (May 2024)")
    p = d.copy()
    p["hour"] = p.t0.dt.hour + p.t0.dt.minute / 60
    p["kind"] = np.where((p.t0.dt.dayofweek >= 5) | (p.is_holiday == 1), "Weekend / holiday", "Weekday")
    g = p.groupby(["kind", "hour"]).occ_now.mean().reset_index()
    fig2 = go.Figure()
    for k, gg in g.groupby("kind"):
        fig2.add_trace(go.Scatter(x=gg.hour, y=gg.occ_now * 100, mode="lines", name=k))
    fig2.add_hline(y=row.thr * 100, line_dash="dash", line_color="crimson", opacity=.7)
    fig2.update_layout(height=340, xaxis_title="Hour of day", yaxis_title="Mean occupancy (%)",
                       margin=dict(l=10, r=10, t=10, b=10), legend=dict(orientation="h", y=-0.25))
    st.plotly_chart(fig2, width="stretch")
    hm = p.assign(h=p.t0.dt.hour, dow=p.t0.dt.day_name().str[:3]).pivot_table(index="dow", columns="h", values="occ_now", aggfunc="mean")
    hm = hm.reindex(["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"])
    fig3 = go.Figure(go.Heatmap(z=hm.values * 100, x=hm.columns, y=hm.index, colorbar=dict(title="%"), colorscale="YlOrRd"))
    fig3.update_layout(height=300, xaxis_title="Hour of day", margin=dict(l=10, r=10, t=10, b=10))
    st.plotly_chart(fig3, width="stretch")
    st.caption("Why this matters for forecasting: the daily and weekly rhythm is strong, so a rule that knows the usual pattern is a tough baseline.")

# ------------------------------ Tab 3: model comparison ------------------------------
with tab3:
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
                   "(trees) - so the real scores are not an artefact. The ANN control used only 3 shuffles and its mean R2 is +0.09, not about 0; "
                   "inspect the per-shuffle values before relying on it. This control shows the models beat chance, not that they beat the rule.")

    st.subheader("Recomputed from this replay file (Bagging classifier vs simple rule)")
    rep = pd.DataFrame({"Bagging classifier": cls_metrics(df.actual_high, df.pred_high, df.prob_high),
                        "Simple rule": cls_metrics(df.actual_high, df.rule_high)}).T
    st.dataframe(rep.style.format({"Balanced acc.": "{:.3f}", "High recall": "{:.3f}", "High precision": "{:.3f}", "ROC-AUC": "{:.3f}"}, na_rep="-"),
                 width="stretch")
    per = (df.groupby("carpark").apply(lambda g: cls_metrics(g.actual_high, g.pred_high)["Balanced acc."], include_groups=False)
           .rename("Balanced accuracy").reset_index())
    figb = go.Figure(go.Bar(x=per.carpark, y=per["Balanced accuracy"]))
    figb.update_layout(height=280, yaxis_range=[0.5, 1], margin=dict(l=10, r=10, t=10, b=10), yaxis_title="Balanced accuracy")
    st.plotly_chart(figb, width="stretch")

# ------------------------------ Tab 4: about ------------------------------
with tab4:
    st.markdown(f"""
**Question answered:** at a forecast time, will this carpark be **HIGH** (above its own 80th-percentile occupancy, learned from the training period) **2 hours later**?

**Data:** Singapore HDB Carpark Availability (data.gov.sg), 18 carparks, 30-minute snapshots, 1 Mar-31 May 2024.
**Split (by target time):** train 8 Mar-15 Apr, validation 16-30 Apr (cut-off tuned here), test 1-31 May (never used for tuning).
**Model shown:** Bagging classifier + Bagging regressor, settings fixed in advance. Cut-off used: {cutoff:.2f}.

**Limits - read before presenting**
- This is a **replay** of past data, not a live forecast.
- One test month, one run per model. No confidence intervals.
- A strong rule that only knows the usual daily/weekly pattern is about as good; the models' advantage is more busy periods found, at the cost of more false alarms.
- HIGH is relative to each carpark's own history, so 70% can be HIGH for one carpark and normal for another.
- Not validated on unseen carparks or other months.

Export created: {meta.get('created', 'unknown')} (scikit-learn {meta.get('sklearn', '?')}).
""")
