import os
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(page_title="Student Withdrawal | Lecturer Review", page_icon="🎓", layout="wide", initial_sidebar_state="expanded")
BASE = Path(__file__).resolve().parent
DATA_PATH = Path(os.getenv("DASHBOARD_CSV_PATH", str(BASE / "demo_predictions.csv")))
STAGE = "DEMONSTRATION DATA" if DATA_PATH.name == "demo_predictions.csv" else "RESTRICTED DATA"

st.markdown("""<style>
.block-container {padding-top: 1.4rem; max-width: 1450px;}
[data-testid="stMetric"] {background:#f4f7fc; border:1px solid #dfe7f1; border-radius:12px; padding:14px 18px;}
[data-testid="stSidebar"] {background:#f5f8fc;}
h1,h2,h3 {color:#17365c;}
small {color:#566579;}
</style>""", unsafe_allow_html=True)

REQUIRED = {"code_module","code_presentation","id_student","cutoff_day","probability","alert","review_priority","risk_trend","risk_change","top_3_risk_drivers","drivers_of_risk_increase","lecturer_action","follow_up_action","effectiveness_measure","research_reference"}
@st.cache_data(show_spinner="Loading dashboard data...")
def load_data(path):
    d = pd.read_csv(path, low_memory=False)
    missing = REQUIRED.difference(d.columns)
    if missing:
        raise ValueError("Missing columns: " + ", ".join(sorted(missing)))
    d["id_student"] = d["id_student"].astype(str).str.replace(r"\.0$", "", regex=True)
    for col in ["cutoff_day", "probability", "risk_change", "previous_risk"]:
        if col in d:
            d[col] = pd.to_numeric(d[col], errors="coerce")
    if d["alert"].dtype != bool:
        d["alert"] = d["alert"].astype(str).str.lower().eq("true")
    d["enrolment_id"] = d["code_module"].astype(str) + " / " + d["code_presentation"].astype(str) + " / " + d["id_student"]
    return d

def split_bars(value):
    if pd.isna(value) or not str(value).strip():
        return []
    return [s.strip() for s in str(value).split("|") if s.strip()]

def draw_list(items, placeholder):
    if not items:
        st.info(placeholder)
    else:
        for i, item in enumerate(items, 1):
            st.markdown(f"**{i}.** {item}")

if not DATA_PATH.is_file():
    st.error("Dashboard data not found. Provide demo_predictions.csv or configure DASHBOARD_CSV_PATH.")
    st.stop()
try:
    data = load_data(str(DATA_PATH))
except Exception as exc:
    st.error(f"Unable to load dashboard data. {exc}")
    st.stop()

st.title("Student Withdrawal Early Warning System")
st.caption("Lecturer decision support  ·  XGBoost model outputs  ·  14-day scoring intervals")
if STAGE == "DEMONSTRATION DATA":
    st.info("Synthetic demonstration dataset. Student IDs, scores, and intervention records are fictional and must not be reported as study results.")
else:
    st.warning("Restricted student-level information. Use only within an appropriately access-controlled deployment.")

with st.sidebar:
    st.header("View settings")
    page = st.radio("Dashboard", ["Lecturer review", "Model monitoring"], label_visibility="collapsed")
    module = st.selectbox("Module", sorted(data.code_module.dropna().unique()))
    presentations = sorted(data.loc[data.code_module.eq(module), "code_presentation"].dropna().unique())
    presentation = st.selectbox("Presentation", presentations)
    scope = data[data.code_module.eq(module) & data.code_presentation.eq(presentation)].copy()
    available_days = sorted(int(n) for n in scope.cutoff_day.dropna().unique())
    day = st.selectbox("Scoring day", available_days, index=0)
    threshold = st.slider("Review threshold", 0.30, 0.90, 0.65, step=0.01, help="Exploratory threshold for review only. The CSV's alert/priority classifications remain calculated at their original threshold.")
    st.caption("Threshold changes adjust the current review list, not the original model's calibrated operational policy.")
    st.divider()
    st.caption("All records are prediction observations. A student can appear on multiple days.")

current = scope[scope.cutoff_day.eq(day)].copy()
current["threshold_flag"] = current.probability.ge(threshold)
current["enrolment_id"] = current["enrolment_id"].astype(str)

if page == "Lecturer review":
    st.subheader("Current review queue")
    flagged = current[current.threshold_flag].copy()
    flagged = flagged.sort_values(["probability", "risk_change"], ascending=[False,False])
    c1,c2,c3,c4 = st.columns(4)
    c1.metric("Students to review", flagged.enrolment_id.nunique())
    c2.metric("High priority", flagged.loc[flagged.review_priority.eq("High"),"enrolment_id"].nunique())
    c3.metric("Increasing risk", flagged.loc[flagged.risk_trend.eq("Increasing"),"enrolment_id"].nunique())
    c4.metric("Highest predicted risk", f"{flagged.probability.max():.1%}" if not flagged.empty else "—")

    st.caption("Select a record from the table to inspect its full risk history and the explanation at the current scoring day.")
    display = flagged[["enrolment_id","probability","risk_change","risk_trend","review_priority","top_3_risk_drivers"]].copy()
    display.rename(columns={"enrolment_id":"Student enrolment","probability":"Risk","risk_change":"Change (pp)","risk_trend":"Trend","review_priority":"Priority","top_3_risk_drivers":"Top risk drivers"}, inplace=True)
    display["Change (pp)"] = display["Change (pp)"].mul(100).round(1)
    display["Risk"] = display["Risk"].map(lambda v: f"{v:.1%}")
    selected = []
    try:
        ev = st.dataframe(display, use_container_width=True, hide_index=True, on_select="rerun", selection_mode="single-row", height=310, key="review_queue")
        selected = ev.selection.rows
    except TypeError:
        st.dataframe(display, use_container_width=True, hide_index=True, height=300)
    ids = flagged.enrolment_id.tolist()
    if not ids:
        st.warning("No predictions exceed the selected review threshold at this scoring point. Select another day or test a different threshold.")
        st.stop()
    fallback = st.selectbox("Selected student enrolment (fallback selector)", ids, help="A selected row in the review queue takes precedence over this selector.")
    chosen = ids[selected[0]] if selected and selected[0] < len(ids) else fallback
    record = flagged.loc[flagged.enrolment_id.eq(chosen)].iloc[0]
    history = scope.loc[scope.enrolment_id.eq(chosen)].sort_values("cutoff_day")

    st.divider()
    st.subheader("Student risk review")
    h1,h2,h3 = st.columns(3)
    h1.metric("Current risk", f"{record.probability:.1%}")
    h2.metric("Previous risk", f"{record.previous_risk:.1%}" if pd.notna(record.get("previous_risk")) else "First observation")
    h3.metric("Risk change", f"{record.risk_change*100:+.1f} pp" if pd.notna(record.risk_change) else "—")
    left,right = st.columns([1.15,1], gap="large")
    with left:
        st.markdown("#### Predicted risk over time")
        fig = px.line(history, x="cutoff_day", y="probability", markers=True, labels={"cutoff_day":"Day of module","probability":"Predicted withdrawal risk"})
        fig.update_traces(line_color="#356aab", line_width=3, marker_size=8)
        fig.add_hline(y=threshold, line_dash="dash", line_color="#b44e54", annotation_text=f"Review threshold {threshold:.0%}")
        fig.add_vline(x=day, line_dash="dot", line_color="#64748b")
        fig.update_yaxes(tickformat=".0%", range=[0,1]);fig.update_layout(height=350, margin=dict(l=5,r=5,t=15,b=5))
        st.plotly_chart(fig, use_container_width=True)
        st.caption("A score is a prediction, not a confirmed withdrawal. The complete historical trajectory is shown for the selected enrolment.")
    with right:
        st.markdown("#### Current positive SHAP drivers")
        draw_list(split_bars(record.top_3_risk_drivers), "No positive actionable drivers recorded.")
        st.markdown("#### Contributions that increased")
        draw_list(split_bars(record.drivers_of_risk_increase), "No increased contribution recorded.")
        st.caption("SHAP values describe model contributions, not causal effects or probability-point changes.")
    st.divider()
    st.subheader("Recommended lecturer intervention")
    a,b = st.columns([1.4,1], gap="large")
    with a:
        st.markdown("#### Recommended actions")
        draw_list(split_bars(record.lecturer_action), "Review the academic and engagement record.")
    with b:
        st.markdown("#### Follow-up at next review")
        draw_list(split_bars(record.follow_up_action), "No follow-up specified.")
        st.markdown("#### Measures to monitor")
        draw_list(split_bars(record.effectiveness_measure), "No effectiveness measure specified.")
        refs = list(dict.fromkeys(split_bars(record.research_reference)))
        if refs:
            st.caption("Research informing suggested actions  ·  " + ", ".join(refs))
    st.warning("Recommended actions have not been completed or evaluated. The source CSV contains no lecturer intervention log.")
    st.download_button("Export filtered review queue", flagged.drop(columns=["threshold_flag"]).to_csv(index=False).encode(), file_name=f"review_{module}_{presentation}_day{day}.csv", mime="text/csv")

else:
    st.subheader("Module monitoring")
    st.caption("Exploratory summaries use the current loaded dataset and do not establish clinical, academic, or fairness validity.")
    module_day = data[data.cutoff_day.eq(day)].copy()
    g = module_day.groupby("code_module",as_index=False).agg(observations=("probability","size"), mean_risk=("probability","mean"), median_risk=("probability","median"), max_risk=("probability","max"))
    g["alerts_at_threshold"] = module_day.groupby("code_module")["probability"].apply(lambda x: int((x>=threshold).sum())).reindex(g.code_module).values
    l,r = st.columns(2)
    with l:
        st.markdown("#### Alerts across modules")
        fig=px.bar(g, x="code_module",y="alerts_at_threshold",labels={"code_module":"Module","alerts_at_threshold":"Alert observations"})
        fig.update_traces(marker_color="#376ca5")
        st.plotly_chart(fig,use_container_width=True)
    with r:
        st.markdown("#### Risk-score distribution")
        fig=px.box(module_day,x="code_module",y="probability",points=False,labels={"code_module":"Module","probability":"Predicted risk"})
        fig.add_hline(y=threshold,line_dash="dash",line_color="#b44e54")
        fig.update_yaxes(range=[0,1],tickformat=".0%")
        st.plotly_chart(fig,use_container_width=True)
    st.markdown("#### Alert volume by scoring interval")
    sub=data[data.code_presentation.eq(presentation)].copy()
    vol=sub.groupby(["code_module","cutoff_day"])["probability"].apply(lambda x: int((x>=threshold).sum())).reset_index(name="alerts")
    line=px.line(vol, x="cutoff_day",y="alerts",color="code_module",markers=True,labels={"cutoff_day":"Scoring day","alerts":"Alert observations","code_module":"Module"})
    st.plotly_chart(line,use_container_width=True)
    st.markdown("#### Module summary at selected scoring day")
    st.dataframe(g.style.format({"mean_risk":"{:.1%}","median_risk":"{:.1%}","max_risk":"{:.1%}"}),use_container_width=True,hide_index=True)
    st.caption("Comparisons depend on the number and composition of available observations per module and interval.")
