
import streamlit as st
import os
import joblib
import pandas as pd
import plotly.graph_objects as go
 
# ============================================================
# PAGE CONFIGURATION
# ============================================================
 
st.set_page_config(
    page_title="FraudShield AI — Transaction Security",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)
 
# ============================================================
# DESIGN TOKENS
# ------------------------------------------------------------
# Palette   : Ink #0A0E1A · Panel #121826 · Panel-2 #1A2333
#             Vault Gold #C9A227 (signature) · Signal Red #E5484D (fraud)
#             Trust Teal #2DD4BF (normal) · Text #EDF1F7 · Muted #7C8798
# Type      : Space Grotesk (display/eyebrows) · Inter (body)
#             IBM Plex Mono (data, amounts, ids)
# Signature : radial "clearance seal" badge marking system status
# ============================================================
 
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@500;600;700&family=Inter:wght@400;500;600&family=IBM+Plex+Mono:wght@400;500;600&display=swap');
 
:root{
    --ink:#0A0E1A;
    --panel:#121826;
    --panel-2:#1A2333;
    --border:#26324A;
    --gold:#C9A227;
    --gold-soft:rgba(201,162,39,0.14);
    --red:#E5484D;
    --red-soft:rgba(229,72,77,0.14);
    --teal:#2DD4BF;
    --teal-soft:rgba(45,212,191,0.12);
    --text:#EDF1F7;
    --muted:#7C8798;
}
 
#MainMenu {visibility:hidden;}
footer {visibility:hidden;}
header {visibility:hidden;}
 
html, body, [class*="css"]{
    font-family:'Inter', sans-serif;
    color:var(--text);
}
 
.stApp{
    background:
        radial-gradient(circle at 12% 0%, rgba(201,162,39,0.06) 0%, transparent 45%),
        radial-gradient(circle at 90% 10%, rgba(45,212,191,0.05) 0%, transparent 40%),
        var(--ink);
}
 
.block-container{
    padding-top:1.5rem;
    padding-bottom:3rem;
    max-width:1200px;
}
 
section[data-testid="stSidebar"]{
    background:var(--panel);
    border-right:1px solid var(--border);
}
section[data-testid="stSidebar"] .block-container{
    padding-top:2rem;
}
 
/* ---------- Eyebrow / structural labels ---------- */
.eyebrow{
    font-family:'Space Grotesk', sans-serif;
    font-size:11px;
    font-weight:600;
    letter-spacing:0.16em;
    text-transform:uppercase;
    color:var(--gold);
    margin-bottom:6px;
    display:block;
}
 
/* ---------- Hero ---------- */
.hero-wrap{
    display:flex;
    align-items:center;
    justify-content:space-between;
    gap:32px;
    padding:8px 4px 28px 4px;
    border-bottom:1px solid var(--border);
    margin-bottom:28px;
}
.hero-title{
    font-family:'Space Grotesk', sans-serif;
    font-size:38px;
    font-weight:700;
    color:var(--text);
    line-height:1.15;
    margin:0;
}
.hero-title span{ color:var(--gold); }
.hero-sub{
    color:var(--muted);
    font-size:15.5px;
    margin-top:8px;
    max-width:520px;
}
 
/* ---------- Seal badge (signature element) ---------- */
.seal-wrap{ display:flex; flex-direction:column; align-items:center; gap:8px; }
.seal{ width:96px; height:96px; }
.seal-ring{
    fill:none; stroke:var(--border); stroke-width:1;
}
.seal-ring-active{
    fill:none; stroke:var(--gold); stroke-width:1.5;
    stroke-dasharray:4 6;
    animation:spin 22s linear infinite;
    transform-origin:50% 50%;
}
.seal-core{ fill:var(--panel-2); stroke:var(--gold); stroke-width:1.5; }
@keyframes spin{ from{transform:rotate(0deg);} to{transform:rotate(360deg);} }
.seal-label{
    font-family:'IBM Plex Mono', monospace;
    font-size:10px;
    letter-spacing:0.1em;
    color:var(--teal);
    text-transform:uppercase;
}
 
/* ---------- KPI cards ---------- */
.kpi-card{
    background:var(--panel);
    border:1px solid var(--border);
    border-radius:10px;
    padding:18px 20px;
    height:100%;
}
.kpi-label{
    font-family:'Space Grotesk', sans-serif;
    font-size:11px;
    font-weight:600;
    letter-spacing:0.12em;
    text-transform:uppercase;
    color:var(--muted);
}
.kpi-value{
    font-family:'IBM Plex Mono', monospace;
    font-size:28px;
    font-weight:600;
    color:var(--text);
    margin-top:6px;
}
.kpi-value.gold{ color:var(--gold); }
.kpi-value.teal{ color:var(--teal); }
.kpi-value.red{ color:var(--red); }
 
/* ---------- Section card ---------- */
.section-card{
    background:var(--panel);
    border:1px solid var(--border);
    border-radius:12px;
    padding:24px 26px;
    margin-bottom:22px;
}
.section-title{
    font-family:'Space Grotesk', sans-serif;
    font-size:18px;
    font-weight:600;
    color:var(--text);
    margin:0 0 4px 0;
}
.section-desc{
    color:var(--muted);
    font-size:13.5px;
    margin-bottom:16px;
}
 
/* ---------- Status pills ---------- */
.pill{
    display:inline-flex;
    align-items:center;
    gap:6px;
    font-family:'IBM Plex Mono', monospace;
    font-size:12px;
    padding:5px 10px;
    border-radius:999px;
    margin:3px 4px 3px 0;
}
.pill-ok{ background:var(--teal-soft); color:var(--teal); border:1px solid rgba(45,212,191,0.3); }
.pill-fail{ background:var(--red-soft); color:var(--red); border:1px solid rgba(229,72,77,0.3); }
.pill-dot{ width:6px; height:6px; border-radius:50%; background:currentColor; }
 
/* ---------- Tag chips (stack list) ---------- */
.chip-row{ display:flex; flex-wrap:wrap; gap:6px; margin-top:6px; }
.chip{
    font-family:'IBM Plex Mono', monospace;
    font-size:11px;
    color:var(--muted);
    background:var(--panel-2);
    border:1px solid var(--border);
    padding:4px 9px;
    border-radius:6px;
}
 
/* ---------- Uploader ---------- */
[data-testid="stFileUploaderDropzone"]{
    background:var(--panel-2) !important;
    border:1.5px dashed var(--border) !important;
    border-radius:10px !important;
}
 
/* ---------- Buttons ---------- */
.stButton>button, .stDownloadButton>button{
    background:var(--gold) !important;
    color:#1A1300 !important;
    border:none !important;
    font-family:'Space Grotesk', sans-serif !important;
    font-weight:600 !important;
    letter-spacing:0.02em;
    border-radius:8px !important;
    padding:0.55rem 1.2rem !important;
}
.stButton>button:hover, .stDownloadButton>button:hover{
    background:#DFB431 !important;
}
 
/* ---------- Selectbox / slider labels ---------- */
.stSelectbox label, .stSlider label{
    font-family:'Space Grotesk', sans-serif;
    font-size:12px;
    letter-spacing:0.06em;
    text-transform:uppercase;
    color:var(--muted) !important;
}
 
/* ---------- Dataframe container ---------- */
[data-testid="stDataFrame"]{
    border:1px solid var(--border);
    border-radius:8px;
    overflow:hidden;
}
 
hr{ border-color:var(--border) !important; }
</style>
""", unsafe_allow_html=True)
 
# ============================================================
# LOAD MODEL & SCALER
# ============================================================
 
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(BASE_DIR, "models")
MODEL_PATH = os.path.join(MODELS_DIR, "random_forest.pkl")
SCALER_PATH = os.path.join(MODELS_DIR, "robust_scaler.pkl")
 
model = None
scaler = None
 
if os.path.exists(MODEL_PATH):
    model = joblib.load(MODEL_PATH)
 
if os.path.exists(SCALER_PATH):
    scaler = joblib.load(SCALER_PATH)
 
system_armed = model is not None and scaler is not None
 
# ============================================================
# SIDEBAR
# ============================================================
 
with st.sidebar:
 
    st.markdown("""
    <div style="display:flex;align-items:center;gap:10px;margin-bottom:2px;">
        <span style="font-size:26px;">🛡️</span>
        <span style="font-family:'Space Grotesk',sans-serif;font-size:20px;font-weight:700;color:#EDF1F7;">
            FraudShield <span style="color:#C9A227;">AI</span>
        </span>
    </div>
    <div style="color:#7C8798;font-size:12.5px;margin:6px 0 18px 0;line-height:1.5;">
        Intelligent credit card fraud detection platform. Score transactions
        and flag anomalies before they clear.
    </div>
    """, unsafe_allow_html=True)
 
    st.markdown('<span class="eyebrow">System Status</span>', unsafe_allow_html=True)
 
    model_pill = (
        '<span class="pill pill-ok"><span class="pill-dot"></span>RANDOM FOREST — ARMED</span>'
        if model is not None else
        '<span class="pill pill-fail"><span class="pill-dot"></span>MODEL NOT FOUND</span>'
    )
    scaler_pill = (
        '<span class="pill pill-ok"><span class="pill-dot"></span>ROBUST SCALER — ARMED</span>'
        if scaler is not None else
        '<span class="pill pill-fail"><span class="pill-dot"></span>SCALER NOT FOUND</span>'
    )
    st.markdown(f"<div>{model_pill}<br>{scaler_pill}</div>", unsafe_allow_html=True)
 
    st.markdown("<hr>", unsafe_allow_html=True)
 
    st.markdown('<span class="eyebrow">Stack</span>', unsafe_allow_html=True)
    st.markdown("""
    <div class="chip-row">
        <span class="chip">Python</span>
        <span class="chip">Scikit-learn</span>
        <span class="chip">Random Forest</span>
        <span class="chip">SMOTE</span>
        <span class="chip">Streamlit</span>
    </div>
    """, unsafe_allow_html=True)
 
    st.markdown("<hr>", unsafe_allow_html=True)
 
    st.markdown('<span class="eyebrow">Developer</span>', unsafe_allow_html=True)
    st.markdown('<div style="font-family:\'IBM Plex Mono\',monospace;font-size:13px;color:#EDF1F7;">Mazen Mohamed</div>', unsafe_allow_html=True)
 
# ============================================================
# HERO
# ============================================================
 
seal_state_color = "#2DD4BF" if system_armed else "#E5484D"
seal_state_text = "ARMED" if system_armed else "OFFLINE"
 
st.markdown(f"""
<div class="hero-wrap">
    <div>
        <p class="hero-title">Transaction <span>Security</span><br>Review Console</p>
        <p class="hero-sub">Upload a transaction batch and FraudShield AI will score every row,
        rank the highest-risk cases, and hand you a clean case file to act on.</p>
    </div>
    <div class="seal-wrap">
        <svg class="seal" viewBox="0 0 100 100">
            <circle class="seal-ring" cx="50" cy="50" r="46"/>
            <circle class="seal-ring-active" cx="50" cy="50" r="38" style="stroke:{seal_state_color};"/>
            <circle class="seal-core" cx="50" cy="50" r="27" style="stroke:{seal_state_color};"/>
            <text x="50" y="47" text-anchor="middle" font-family="Space Grotesk" font-size="15" font-weight="700" fill="{seal_state_color}">AI</text>
            <text x="50" y="61" text-anchor="middle" font-family="IBM Plex Mono" font-size="7" fill="{seal_state_color}">GUARD</text>
        </svg>
        <span class="seal-label" style="color:{seal_state_color};">{seal_state_text}</span>
    </div>
</div>
""", unsafe_allow_html=True)
 
# ============================================================
# MODEL METRICS (reference performance)
# ============================================================
 
st.markdown('<span class="eyebrow">Model Performance — Held-out Test Set</span>', unsafe_allow_html=True)
 
k1, k2, k3, k4 = st.columns(4)
kpi_data = [
    (k1, "Accuracy", "99.95%", ""),
    (k2, "Precision", "89.02%", "gold"),
    (k3, "Recall", "76.84%", "teal"),
    (k4, "F1 Score", "82.49%", "gold"),
]
for col, label, value, tone in kpi_data:
    with col:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-label">{label}</div>
            <div class="kpi-value {tone}">{value}</div>
        </div>
        """, unsafe_allow_html=True)
 
st.markdown("<div style='height:26px;'></div>", unsafe_allow_html=True)
 
# ============================================================
# UPLOAD
# ============================================================
 
st.markdown('<div class="section-card">', unsafe_allow_html=True)
st.markdown('<span class="eyebrow">Step 01</span>', unsafe_allow_html=True)
st.markdown('<p class="section-title">Upload Transaction Batch</p>', unsafe_allow_html=True)
st.markdown('<p class="section-desc">CSV with the standard feature columns, including <code>Time</code> and <code>Amount</code>.</p>', unsafe_allow_html=True)
 
uploaded_file = st.file_uploader("Choose CSV file", type=["csv"], label_visibility="collapsed")
 
df = None
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.markdown(f"""
    <div class="pill pill-ok" style="margin-top:12px;">
        <span class="pill-dot"></span>{len(df):,} ROWS LOADED
    </div>
    """, unsafe_allow_html=True)
    st.dataframe(df.head(), use_container_width=True)
    run = st.button("Run Fraud Scan →")
else:
    run = False
 
st.markdown('</div>', unsafe_allow_html=True)
 
# ============================================================
# PREDICTION + RESULTS
# ============================================================
 
if run and df is not None:
 
    prediction_data = df.copy()
 
    if "Time" in prediction_data.columns and "Amount" in prediction_data.columns and scaler is not None:
        prediction_data[["Time", "Amount"]] = scaler.transform(prediction_data[["Time", "Amount"]])
 
    predictions = model.predict(prediction_data)
    probabilities = model.predict_proba(prediction_data)[:, 1]
 
    df["Prediction"] = predictions
    df["Fraud Probability"] = (probabilities * 100).round(2)
    df["Prediction"] = df["Prediction"].map({0: "Normal", 1: "Fraud"})
    df.rename(columns={"Prediction": "Transaction Status"}, inplace=True)
 
    total = len(df)
    fraud = int((predictions == 1).sum())
    normal = int((predictions == 0).sum())
    fraud_rate = fraud / total * 100 if total else 0
 
    # ---------------- Results summary ----------------
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.markdown('<span class="eyebrow">Step 02 — Scan Complete</span>', unsafe_allow_html=True)
    st.markdown('<p class="section-title">Case Summary</p>', unsafe_allow_html=True)
 
    r1, r2, r3, r4 = st.columns(4)
    result_data = [
        (r1, "Total Transactions", f"{total:,}", ""),
        (r2, "Flagged Fraud", f"{fraud:,}", "red"),
        (r3, "Cleared Normal", f"{normal:,}", "teal"),
        (r4, "Fraud Rate", f"{fraud_rate:.2f}%", "gold"),
    ]
    for col, label, value, tone in result_data:
        with col:
            st.markdown(f"""
            <div class="kpi-card">
                <div class="kpi-label">{label}</div>
                <div class="kpi-value {tone}">{value}</div>
            </div>
            """, unsafe_allow_html=True)
 
    st.markdown("<div style='height:20px;'></div>", unsafe_allow_html=True)
 
    chart = go.Figure(data=[go.Pie(
        labels=["Normal", "Fraud"],
        values=[normal, fraud],
        hole=0.62,
        marker=dict(colors=["#2DD4BF", "#E5484D"], line=dict(color="#0A0E1A", width=3)),
        textfont=dict(family="IBM Plex Mono", color="#EDF1F7", size=13),
    )])
    chart.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter", color="#EDF1F7"),
        legend=dict(orientation="h", y=-0.1),
        margin=dict(t=10, b=10, l=10, r=10),
        height=320,
        annotations=[dict(text=f"{fraud_rate:.1f}%<br>flagged", x=0.5, y=0.5,
                           font=dict(family="IBM Plex Mono", size=14, color="#C9A227"),
                           showarrow=False)]
    )
    st.plotly_chart(chart, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)
 
    # ---------------- Top 10 suspicious ----------------
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.markdown('<span class="eyebrow">Priority Review</span>', unsafe_allow_html=True)
    st.markdown('<p class="section-title">🚨 Top 10 Suspicious Transactions</p>', unsafe_allow_html=True)
    st.markdown('<p class="section-desc">Ranked by predicted fraud probability — start here.</p>', unsafe_allow_html=True)
 
    top_fraud = df.sort_values(by="Fraud Probability", ascending=False).head(10)
    st.dataframe(top_fraud, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)
 
    # ---------------- Full case file ----------------
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.markdown('<span class="eyebrow">Full Ledger</span>', unsafe_allow_html=True)
    st.markdown('<p class="section-title">Filter &amp; Review Transactions</p>', unsafe_allow_html=True)
 
    fc1, fc2 = st.columns([1, 2])
    with fc1:
        filter_option = st.selectbox("Show", ["All Transactions", "Fraud Only", "Normal Only"])
    with fc2:
        rows = st.slider("Rows to display", 10, 1000, 100)
 
    if filter_option == "Fraud Only":
        filtered_df = df[df["Transaction Status"] == "Fraud"]
    elif filter_option == "Normal Only":
        filtered_df = df[df["Transaction Status"] == "Normal"]
    else:
        filtered_df = df
 
    filtered_df = filtered_df.sort_values(by="Fraud Probability", ascending=False)
    display_df = filtered_df.head(rows)
 
    def highlight_prediction(row):
        if row["Transaction Status"] == "Fraud":
            return ["background-color:#2A1418;color:#F5B7B9"] * len(row)
        return ["background-color:#122A26;color:#A9EDE3"] * len(row)
 
    st.dataframe(
        display_df.style.apply(highlight_prediction, axis=1),
        use_container_width=True
    )
 
    csv = df.to_csv(index=False).encode("utf-8")
    st.download_button("⬇ Download Full Results (CSV)", csv, "prediction_results.csv", "text/csv")
 
    st.markdown('</div>', unsafe_allow_html=True)