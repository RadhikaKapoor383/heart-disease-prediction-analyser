import streamlit as st
import pandas as pd
import numpy as np
import pickle
import plotly.graph_objects as go

# ─── Page Config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Heart Disease Predictor",
    page_icon="assets/heart.svg",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─── Custom CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:ital,wght@0,400;0,600;1,400&family=Lato:wght@300;400;700&display=swap');

:root {
    --bg:       #f5f0eb;
    --card:     #ffffff;
    --border:   #e0d8cf;
    --accent:   #c0392b;
    --accent2:  #1a6b3c;
    --text:     #1c1c1c;
    --muted:    #7a736b;
    --warning:  #b8860b;
    --soft:     #fdf8f3;
}

html, body, [class*="css"] {
    font-family: 'Lato', sans-serif;
    background-color: var(--bg);
    color: var(--text);
}

#MainMenu, footer, header { visibility: hidden; }
.block-container { padding-top: 2rem; padding-bottom: 2rem; }

/* Hero */
.hero {
    background: linear-gradient(135deg, #fff8f5 0%, #f5f0eb 60%, #f0f5f2 100%);
    border: 1px solid var(--border);
    border-top: 4px solid var(--accent);
    border-radius: 12px;
    padding: 2.5rem 3rem;
    margin-bottom: 2rem;
    position: relative;
    overflow: hidden;
}
.hero::after {
    content: '';
    position: absolute;
    bottom: -60px;
    right: -60px;
    width: 200px;
    height: 200px;
    border-radius: 50%;
    background: radial-gradient(circle, rgba(192,57,43,0.05) 0%, transparent 70%);
}
.hero-eyebrow {
    font-family: 'Lato', sans-serif;
    font-size: 0.72rem;
    font-weight: 700;
    letter-spacing: 0.18em;
    text-transform: uppercase;
    color: var(--accent);
    margin-bottom: 0.6rem;
}
.hero-title {
    font-family: 'Playfair Display', serif;
    font-size: 2.6rem;
    font-weight: 600;
    color: var(--text);
    margin: 0 0 0.4rem 0;
    line-height: 1.15;
}
.hero-title em { font-style: italic; color: var(--accent); }
.hero-sub {
    color: var(--muted);
    font-size: 0.92rem;
    font-weight: 300;
    margin: 0;
    letter-spacing: 0.015em;
}

/* Cards */
.metric-row { display: flex; gap: 1rem; margin-bottom: 1.5rem; }
.metric-card {
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 1.2rem 1.5rem;
    flex: 1;
    box-shadow: 0 1px 4px rgba(0,0,0,0.05);
}
.metric-label {
    font-size: 0.68rem;
    color: var(--muted);
    text-transform: uppercase;
    letter-spacing: 0.12em;
    font-weight: 700;
    margin-bottom: 0.4rem;
}
.metric-value {
    font-family: 'Playfair Display', serif;
    font-size: 1.8rem;
    color: var(--text);
}
.metric-value.green { color: var(--accent2); }
.metric-value.red   { color: var(--accent);  }

/* Section headers */
.section-header {
    font-family: 'Playfair Display', serif;
    font-size: 1.2rem;
    font-weight: 600;
    color: var(--text);
    margin: 1.5rem 0 1rem 0;
    padding-bottom: 0.5rem;
    border-bottom: 1px solid var(--border);
    display: flex;
    align-items: center;
    gap: 0.5rem;
}

/* Result boxes */
.result-positive {
    background: linear-gradient(135deg, rgba(192,57,43,0.06), rgba(192,57,43,0.02));
    border: 1px solid var(--accent);
    border-left: 4px solid var(--accent);
    border-radius: 10px;
    padding: 1.8rem 2rem;
    text-align: center;
    margin: 1rem 0;
}
.result-negative {
    background: linear-gradient(135deg, rgba(26,107,60,0.06), rgba(26,107,60,0.02));
    border: 1px solid var(--accent2);
    border-left: 4px solid var(--accent2);
    border-radius: 10px;
    padding: 1.8rem 2rem;
    text-align: center;
    margin: 1rem 0;
}
.result-icon {
    font-size: 2rem;
    margin-bottom: 0.5rem;
    font-family: 'Segoe UI Symbol', sans-serif;
}
.result-label {
    font-family: 'Playfair Display', serif;
    font-size: 1.8rem;
    font-weight: 600;
    margin: 0.3rem 0;
}
.result-sub { color: var(--muted); font-size: 0.88rem; margin-top: 0.4rem; }

/* Sidebar */
[data-testid="stSidebar"] {
    background: var(--soft);
    border-right: 1px solid var(--border);
}

/* Button */
.stButton > button {
    background: var(--accent) !important;
    color: #ffffff !important;
    border: none !important;
    border-radius: 8px !important;
    padding: 0.7rem 2rem !important;
    font-family: 'Lato', sans-serif !important;
    font-size: 0.95rem !important;
    font-weight: 700 !important;
    width: 100% !important;
    letter-spacing: 0.06em !important;
    text-transform: uppercase !important;
    transition: opacity 0.2s !important;
}
.stButton > button:hover { opacity: 0.88 !important; }

hr { border-color: var(--border) !important; }

.info-box {
    background: #fffdf0;
    border: 1px solid #e8d87a;
    border-left: 3px solid var(--warning);
    border-radius: 8px;
    padding: 0.9rem 1.1rem;
    font-size: 0.82rem;
    color: var(--muted);
    margin-top: 1rem;
    line-height: 1.6;
}

.sidebar-detail {
    display: flex;
    justify-content: space-between;
    padding: 0.45rem 0;
    border-bottom: 1px solid var(--border);
    font-size: 0.83rem;
}
.sidebar-detail span:first-child { color: var(--muted); }
.sidebar-detail span:last-child  { color: var(--text); font-weight: 600; }

.improvement-item {
    display: flex;
    align-items: flex-start;
    gap: 0.5rem;
    padding: 0.3rem 0;
    font-size: 0.83rem;
    color: var(--muted);
    line-height: 1.4;
}
.improvement-check { color: var(--accent2); font-weight: 700; flex-shrink: 0; }

.placeholder-box {
    background: var(--card);
    border: 1.5px dashed var(--border);
    border-radius: 12px;
    padding: 3rem 2rem;
    text-align: center;
    color: var(--muted);
}
.placeholder-title {
    font-family: 'Playfair Display', serif;
    font-size: 1.1rem;
    color: var(--text);
    margin-bottom: 0.4rem;
}
</style>
""", unsafe_allow_html=True)

# ─── Load Model ───────────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    with open('models/hgb_model.pkl', 'rb') as f:
        return pickle.load(f)

model = load_model()

# ─── Hero ─────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
    <div class="hero-eyebrow">Clinical Decision Support &nbsp;·&nbsp; AI-Powered</div>
    <p class="hero-title">Heart Disease <em>Risk Predictor</em></p>
    <p class="hero-sub">UCI Multi-Site Dataset &nbsp;·&nbsp; HistGradientBoosting &nbsp;·&nbsp; ~84% Accuracy &nbsp;·&nbsp; Binary Classification</p>
</div>
""", unsafe_allow_html=True)

# ─── Layout ───────────────────────────────────────────────────────────────────
left, right = st.columns([1.1, 1], gap="large")

# ─── LEFT: Inputs ─────────────────────────────────────────────────────────────
with left:
    st.markdown('<p class="section-header">Patient Information</p>', unsafe_allow_html=True)

    c1, c2 = st.columns(2)
    with c1:
        age      = st.slider("Age", 20, 80, 52)
        sex      = st.selectbox("Sex", ["Male", "Female"])
        cp       = st.selectbox("Chest Pain Type", [
            "typical angina", "atypical angina", "non-anginal", "asymptomatic"
        ])
        fbs      = st.selectbox("Fasting Blood Sugar > 120 mg/dl", [True, False])
        trestbps = st.slider("Resting Blood Pressure (mm Hg)", 80, 200, 130)

    with c2:
        chol    = st.slider("Cholesterol (mg/dl)", 100, 600, 240)
        thalch  = st.slider("Max Heart Rate", 60, 220, 150)
        oldpeak = st.slider("ST Depression (oldpeak)", 0.0, 6.0, 1.0, 0.1)
        exang   = st.selectbox("Exercise-Induced Angina", [True, False])
        restecg = st.selectbox("Resting ECG", [
            "normal", "lv hypertrophy", "st-t abnormality"
        ])

    c3, c4 = st.columns(2)
    with c3:
        slope = st.selectbox("ST Slope", ["flat", "upsloping", "downsloping"])
        ca    = st.slider("Major Vessels (ca)", 0, 3, 0)
    with c4:
        thal = st.selectbox("Thalassemia (thal)", [
            "normal", "fixed defect", "reversable defect"
        ])
        ca_missing   = 0
        thal_missing = 0

    st.markdown("<br>", unsafe_allow_html=True)
    predict_btn = st.button("Predict Risk")

# ─── RIGHT: Results ───────────────────────────────────────────────────────────
with right:
    st.markdown('<p class="section-header">Prediction Result</p>', unsafe_allow_html=True)

    if predict_btn:
        input_df = pd.DataFrame([{
            'age': age, 'trestbps': trestbps, 'chol': chol,
            'thalch': thalch, 'oldpeak': oldpeak, 'ca': ca,
            'ca_missing': ca_missing, 'thal_missing': thal_missing,
            'sex': sex, 'cp': cp, 'fbs': fbs, 'restecg': restecg,
            'exang': exang, 'slope': slope, 'thal': thal,
        }])

        prediction = model.predict(input_df)[0]
        proba      = model.predict_proba(input_df)[0]
        risk_pct   = proba[1] * 100

        if prediction == 1:
            st.markdown(f"""
            <div class="result-positive">
                <div class="result-icon" style="color:#c0392b">&#9888;</div>
                <div class="result-label" style="color:#c0392b">Heart Disease Detected</div>
                <div class="result-sub">Model predicts presence of heart disease</div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="result-negative">
                <div class="result-icon" style="color:#1a6b3c">&#10003;</div>
                <div class="result-label" style="color:#1a6b3c">No Heart Disease</div>
                <div class="result-sub">Model predicts no heart disease detected</div>
            </div>
            """, unsafe_allow_html=True)

        # ── Gauge ─────────────────────────────────────────────────────────
        st.markdown("<br>", unsafe_allow_html=True)
        fig = go.Figure(go.Indicator(
            mode  = "gauge+number",
            value = risk_pct,
            title = {"text": "Disease Risk %", "font": {"color": "#faf0e5", "size": 13, "family": "Lato"}},
            number = {"suffix": "%", "font": {"color": "#ffe9e9", "size": 36, "family": "Playfair Display"}},
            gauge = {
                "axis":        {"range": [0, 100], "tickcolor": "#b0a898", "tickfont": {"color": "#7a736b"}},
                "bar":         {"color": "#c0392b" if risk_pct > 50 else "#1a6b3c"},
                "bgcolor":     "#f5f0eb",
                "bordercolor": "#e0d8cf",
                "steps": [
                    {"range": [0,  40], "color": "rgba(26,107,60,0.08)"},
                    {"range": [40, 70], "color": "rgba(184,134,11,0.08)"},
                    {"range": [70,100], "color": "rgba(192,57,43,0.08)"},
                ],
                "threshold": {
                    "line":      {"color": "#b8860b", "width": 2},
                    "thickness": 0.75,
                    "value":     50
                }
            }
        ))
        fig.update_layout(
            height=220,
            margin=dict(t=30, b=10, l=20, r=20),
            paper_bgcolor="rgba(0,0,0,0)",
            font_color="#fff4f4"
        )
        st.plotly_chart(fig, use_container_width=True)

        # ── Confidence bars ────────────────────────────────────────────────
        fig2 = go.Figure()
        fig2.add_trace(go.Bar(
            x            = ["No Disease", "Disease"],
            y            = [proba[0]*100, proba[1]*100],
            marker_color = ["#1a6b3c", "#c0392b"],
            text         = [f"{proba[0]*100:.1f}%", f"{proba[1]*100:.1f}%"],
            textposition = "outside",
            textfont     = {"color": "#fdf7e1", "family": "Lato"},
        ))
        fig2.update_layout(
            title         = {"text": "Confidence Breakdown", "font": {"color": "#faf1e7", "size": 13, "family": "Lato"}},
            height        = 220,
            margin        = dict(t=40, b=20, l=10, r=10),
            yaxis         = dict(range=[0, 115], showgrid=False, visible=False),
            xaxis         = dict(tickfont={"color": "#fff3f3", "family": "Lato"}),
            paper_bgcolor = "rgba(0,0,0,0)",
            plot_bgcolor  = "rgba(0,0,0,0)",
            showlegend    = False,
        )
        st.plotly_chart(fig2, use_container_width=True)

        # ── Metric cards ───────────────────────────────────────────────────
        st.markdown(f"""
        <div class="metric-row">
            <div class="metric-card">
                <div class="metric-label">Risk Score</div>
                <div class="metric-value {'red' if risk_pct > 50 else 'green'}">{risk_pct:.1f}%</div>
            </div>
            <div class="metric-card">
                <div class="metric-label">Prediction</div>
                <div class="metric-value {'red' if prediction == 1 else 'green'}">
                    {'Disease' if prediction == 1 else 'Healthy'}
                </div>
            </div>
            <div class="metric-card">
                <div class="metric-label">Model</div>
                <div class="metric-value {'red' if prediction == 1 else 'green'}">
                    HistGradBoost
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    else:
        st.markdown("""
        <div class="placeholder-box">
            <svg width="40" height="40" viewBox="0 0 24 24" fill="none"
                 stroke="#c0392b" stroke-width="1.5" stroke-linecap="round"
                 stroke-linejoin="round" style="margin-bottom:1rem">
                <path d="M20.84 4.61a5.5 5.5 0 0 0-7.78 0L12 5.67l-1.06-1.06
                         a5.5 5.5 0 0 0-7.78 7.78l1.06 1.06L12 21.23l7.78-7.78
                         1.06-1.06a5.5 5.5 0 0 0 0-7.78z"/>
            </svg>
            <div class="placeholder-title">Fill in patient details</div>
            <div style="font-size:0.88rem; margin-top:0.4rem;">
                and click <strong style="color:#c0392b">Predict Risk</strong> to see results
            </div>
        </div>
        """, unsafe_allow_html=True)

# ─── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style="display:flex; align-items:center; gap:0.6rem; margin-bottom:0.5rem;">
        <svg width="20" height="20" viewBox="0 0 24 24" fill="none"
             stroke="#c0392b" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
            <path d="M20.84 4.61a5.5 5.5 0 0 0-7.78 0L12 5.67l-1.06-1.06
                     a5.5 5.5 0 0 0-7.78 7.78l1.06 1.06L12 21.23l7.78-7.78
                     1.06-1.06a5.5 5.5 0 0 0 0-7.78z"/>
        </svg>
        <span style="font-family:'Playfair Display',serif; font-size:1.1rem; font-weight:600; color:#1c1c1c;">About</span>
    </div>
    <div style="color:#7a736b; font-size:0.85rem; line-height:1.75;">
        This app predicts the <strong style="color:#1c1c1c">presence or absence</strong>
        of heart disease using clinical measurements from the
        <strong style="color:#1c1c1c">UCI Heart Disease Dataset</strong>
        (920 patients, 4 collection sites).
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    st.markdown("""
    <div style="display:flex; align-items:center; gap:0.5rem; margin-bottom:0.8rem;">
        <svg width="16" height="16" viewBox="0 0 24 24" fill="none"
             stroke="#c0392b" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
            <line x1="18" y1="20" x2="18" y2="10"/><line x1="12" y1="20" x2="12" y2="4"/>
            <line x1="6" y1="20" x2="6" y2="14"/>
        </svg>
        <span style="font-family:'Playfair Display',serif; font-size:1rem; font-weight:600; color:#1c1c1c;">Model Details</span>
    </div>
    """, unsafe_allow_html=True)

    details = {
        "Algorithm":      "HistGradientBoosting",
        "Test Accuracy":  "~84.8%",
        "CV Accuracy":    "~79.2%",
        "Dataset":        "920 patients",
        "Target":         "Binary (0 / 1)",
        "Features":       "15 clinical inputs",
    }
    for k, v in details.items():
        st.markdown(f"""
        <div class="sidebar-detail">
            <span>{k}</span><span>{v}</span>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")

    st.markdown("""
    <div style="display:flex; align-items:center; gap:0.5rem; margin-bottom:0.8rem;">
        <svg width="16" height="16" viewBox="0 0 24 24" fill="none"
             stroke="#c0392b" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
            <circle cx="12" cy="12" r="3"/><path d="M19.07 4.93a10 10 0 0 1 0 14.14
            M4.93 4.93a10 10 0 0 0 0 14.14"/>
        </svg>
        <span style="font-family:'Playfair Display',serif; font-size:1rem; font-weight:600; color:#1c1c1c;">Improvements Applied</span>
    </div>
    """, unsafe_allow_html=True)

    improvements = [
        "Binary target (5 → 2 classes)",
        "Per-site imputation (ca, thal)",
        "class_weight = 'balanced'",
        "5-Fold Cross Validation",
        "RandomizedSearchCV (RF)",
        "XGBoost + HistGradBoost",
    ]
    for imp in improvements:
        st.markdown(f"""
        <div class="improvement-item">
            <span class="improvement-check">&#10003;</span>
            <span>{imp}</span>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("""
    <div class="info-box">
        <strong style="color:#b8860b;">&#9888; Disclaimer</strong><br>
        This tool is for <strong>educational purposes only</strong>.
        It is not a substitute for professional medical advice,
        diagnosis, or treatment.
    </div>
    """, unsafe_allow_html=True)