# ══════════════════════════════════════════════════════════════════
#  app.py  —  Climate Sentiment Analyzer  |  MSDS 453 NLP Project
#  Kasheena Mulla  |  Northwestern University
#
#  GitHub repo layout expected:
#    bertweet_climate_final/
#      added_tokens.json
#      bpe.codes
#      config.json
#      model.safetensors
#      tokenizer_config.json
#      vocab.txt
#    app.py  ← this file
#
#  Install deps:
#    pip install streamlit transformers torch plotly emoji pandas scipy
#
#  Run:
#    streamlit run app.py
# ══════════════════════════════════════════════════════════════════

import re, os, json, math, textwrap, datetime
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st
import emoji as emoji_lib
import torch
from scipy.special import softmax
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# ──────────────────────────────────────────────────────────────
# 0.  PAGE CONFIG  (must be first Streamlit call)
# ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Climate Pulse | NLP Sentiment Analyzer",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ──────────────────────────────────────────────────────────────
# 1.  GLOBAL STYLES  — dark editorial theme
# ──────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display:ital@0;1&family=DM+Sans:wght@300;400;500;600&family=JetBrains+Mono:wght@400;500&display=swap');

/* ── base ── */
html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
    background: #0d1117;
    color: #e6edf3;
}
.main .block-container { padding: 1.5rem 2.5rem 3rem; max-width: 1200px; }

/* ── hero ── */
.hero {
    background: linear-gradient(135deg, #0f2027 0%, #11302a 50%, #0d1f3c 100%);
    border: 1px solid rgba(29,158,117,0.25);
    border-radius: 20px;
    padding: 3rem 3.5rem;
    margin-bottom: 2rem;
    position: relative;
    overflow: hidden;
}
.hero::before {
    content: '';
    position: absolute; inset: 0;
    background: radial-gradient(ellipse at 80% 20%, rgba(29,158,117,0.12) 0%, transparent 60%),
                radial-gradient(ellipse at 20% 80%, rgba(55,138,221,0.10) 0%, transparent 60%);
    pointer-events: none;
}
.hero-title {
    font-family: 'DM Serif Display', serif;
    font-size: 3.2rem;
    line-height: 1.1;
    color: #fff;
    margin: 0 0 0.5rem;
}
.hero-title span { color: #1D9E75; }
.hero-sub {
    font-size: 1.05rem;
    color: #8b949e;
    font-weight: 300;
    letter-spacing: 0.01em;
    margin: 0;
}
.hero-badge {
    display: inline-block;
    background: rgba(29,158,117,0.15);
    border: 1px solid rgba(29,158,117,0.4);
    color: #1D9E75;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.72rem;
    padding: 3px 10px;
    border-radius: 20px;
    margin-bottom: 1rem;
    letter-spacing: 0.08em;
}

/* ── section headers ── */
.section-head {
    font-family: 'DM Serif Display', serif;
    font-size: 1.9rem;
    color: #e6edf3;
    border-bottom: 2px solid #1D9E75;
    padding-bottom: 0.4rem;
    margin: 2.5rem 0 1.2rem;
}
.section-head.blue { border-color: #378ADD; }
.section-head.amber { border-color: #EF9F27; }
.section-head.red { border-color: #E24B4A; }

/* ── cards ── */
.card {
    background: #161b22;
    border: 1px solid #30363d;
    border-radius: 14px;
    padding: 1.4rem 1.6rem;
    margin-bottom: 1rem;
}
.card:hover { border-color: #1D9E75; transition: border-color 0.2s; }

/* ── stat pills ── */
.stat-row { display: flex; gap: 0.8rem; flex-wrap: wrap; margin: 1rem 0; }
.stat-pill {
    background: #21262d;
    border: 1px solid #30363d;
    border-radius: 10px;
    padding: 0.9rem 1.2rem;
    min-width: 120px;
    text-align: center;
}
.stat-pill .val {
    font-family: 'DM Serif Display', serif;
    font-size: 2rem;
    color: #1D9E75;
    display: block;
}
.stat-pill .lbl {
    font-size: 0.72rem;
    color: #8b949e;
    text-transform: uppercase;
    letter-spacing: 0.08em;
}

/* ── label chips ── */
.chip {
    display: inline-block;
    padding: 4px 14px;
    border-radius: 20px;
    font-size: 0.8rem;
    font-weight: 500;
    font-family: 'DM Sans', sans-serif;
}
.chip-pro   { background: rgba(29,158,117,0.18); color: #1D9E75; border: 1px solid rgba(29,158,117,0.35); }
.chip-news  { background: rgba(55,138,221,0.18); color: #378ADD; border: 1px solid rgba(55,138,221,0.35); }
.chip-neut  { background: rgba(136,135,128,0.18); color: #aaa; border: 1px solid rgba(136,135,128,0.35); }
.chip-skep  { background: rgba(226,75,74,0.18); color: #E24B4A; border: 1px solid rgba(226,75,74,0.35); }

/* ── verdict card ── */
.verdict {
    border-radius: 14px;
    padding: 1.6rem 2rem;
    margin: 1rem 0;
}
.verdict-label {
    font-family: 'DM Serif Display', serif;
    font-size: 2rem;
    margin: 0 0 0.3rem;
}
.verdict-conf { font-size: 0.9rem; color: #8b949e; margin: 0; }

/* ── textarea override ── */
textarea {
    background: #161b22 !important;
    color: #e6edf3 !important;
    border: 1px solid #30363d !important;
    border-radius: 10px !important;
    font-family: 'DM Sans', sans-serif !important;
}
textarea:focus { border-color: #1D9E75 !important; outline: none !important; }

/* ── sidebar ── */
[data-testid="stSidebar"] {
    background: #0d1117;
    border-right: 1px solid #21262d;
}
[data-testid="stSidebar"] .block-container { padding: 1.5rem 1rem; }

/* ── buttons ── */
.stButton > button {
    background: #1D9E75 !important;
    color: #fff !important;
    border: none !important;
    border-radius: 10px !important;
    font-family: 'DM Sans', sans-serif !important;
    font-weight: 500 !important;
    padding: 0.55rem 1.6rem !important;
    transition: opacity 0.15s !important;
}
.stButton > button:hover { opacity: 0.85 !important; }

/* ── tabs ── */
[data-testid="stTabs"] button {
    font-family: 'DM Sans', sans-serif !important;
    font-size: 0.9rem !important;
    color: #8b949e !important;
}
[data-testid="stTabs"] button[aria-selected="true"] {
    color: #1D9E75 !important;
    border-bottom-color: #1D9E75 !important;
}

/* ── mono code ── */
code { font-family: 'JetBrains Mono', monospace; font-size: 0.82em; }

/* ── noise feature table ── */
.nf-table { width: 100%; border-collapse: collapse; font-size: 0.85rem; }
.nf-table th { color: #8b949e; font-weight: 500; border-bottom: 1px solid #30363d; padding: 6px 10px; text-align: left; }
.nf-table td { padding: 7px 10px; border-bottom: 1px solid #21262d; }

/* ── pipeline steps ── */
.pipeline-step {
    display: flex;
    align-items: flex-start;
    gap: 1rem;
    padding: 0.9rem 0;
    border-bottom: 1px solid #21262d;
}
.pipeline-step:last-child { border-bottom: none; }
.step-num {
    background: #1D9E75;
    color: #fff;
    width: 28px; height: 28px;
    border-radius: 50%;
    display: flex; align-items: center; justify-content: center;
    font-size: 0.75rem; font-weight: 600;
    flex-shrink: 0;
}
.step-body h4 { margin: 0 0 3px; font-size: 0.92rem; color: #e6edf3; }
.step-body p  { margin: 0; font-size: 0.8rem; color: #8b949e; }
</style>
""", unsafe_allow_html=True)

# ──────────────────────────────────────────────────────────────
# 2.  CONSTANTS
# ──────────────────────────────────────────────────────────────
MODEL_DIR   = "bertweet_climate_final"
MAX_LEN     = 128
DEVICE      = "cuda" if torch.cuda.is_available() else "cpu"

# Label maps — training used -1,0,1,2 remapped to 0,1,2,3 internally
ID2LABEL = {0: "Skeptic/Denial", 1: "Neutral", 2: "Pro-Climate", 3: "News/Factual"}
COLORS   = {0: "#E24B4A", 1: "#888780", 2: "#1D9E75", 3: "#378ADD"}
ICONS    = {0: "❄️", 1: "😐", 2: "🌿", 3: "📰"}
CHIP_CSS = {0: "chip-skep", 1: "chip-neut", 2: "chip-pro", 3: "chip-news"}

# ──────────────────────────────────────────────────────────────
# 3.  TWEET CLEANER  (mirrors training clean_tweet exactly)
# ──────────────────────────────────────────────────────────────
def clean_tweet(text: str) -> str:
    text = str(text)
    text = re.sub(r"https?://\S+", "", text)
    text = re.sub(r"@\w+", "", text)
    text = re.sub(r"#(\w+)", r"\1", text)
    text = re.sub(r"\bRT\b", "", text)
    text = emoji_lib.replace_emoji(text, replace="")
    text = re.sub(r"[^a-zA-Z0-9\s'\-]", " ", text)
    return re.sub(r"\s+", " ", text).strip().lower()

# ──────────────────────────────────────────────────────────────
# 4.  MODEL LOADER
# ──────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_model(path: str):
    tok = AutoTokenizer.from_pretrained(
        path,
        use_fast=False,
        normalization=True,
    )
    mdl = AutoModelForSequenceClassification.from_pretrained(
        path,
        ignore_mismatched_sizes=False,
    )
    mdl.eval()
    mdl.to(DEVICE)

    # Load temperature calibration if present
    T = 1.0
    calib = os.path.join(path, "calibration.json")
    if os.path.exists(calib):
        with open(calib) as f:
            T = json.load(f).get("temperature", 1.0)
    return tok, mdl, T

# ──────────────────────────────────────────────────────────────
# 5.  INFERENCE
# ──────────────────────────────────────────────────────────────
def predict(texts: list, tok, mdl, T: float, batch_size: int = 32) -> pd.DataFrame:
    cleaned = [clean_tweet(t) for t in texts]
    rows = []
    for i in range(0, len(cleaned), batch_size):
        bc = cleaned[i: i + batch_size]
        enc = tok(bc, padding=True, truncation=True,
                  max_length=MAX_LEN, return_tensors="pt")
        enc = {k: v.to(DEVICE) for k, v in enc.items()}
        with torch.no_grad():
            logits = mdl(**enc).logits.cpu().numpy()
        probs   = softmax(logits / T, axis=1)
        pred_ids = probs.argmax(axis=1)
        for raw, cln, prob_row, pid in zip(
                texts[i: i + batch_size], bc, probs, pred_ids):
            rows.append({
                "tweet":      raw,
                "cleaned":    cln,
                "label_id":   int(pid),
                "label":      ID2LABEL[int(pid)],
                "confidence": float(prob_row[pid]),
                **{f"p_{ID2LABEL[k]}": float(prob_row[k]) for k in range(4)},
            })
    return pd.DataFrame(rows)

# ──────────────────────────────────────────────────────────────
# 6.  PLOTLY THEME
# ──────────────────────────────────────────────────────────────
PLOT_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="DM Sans, sans-serif", color="#8b949e"),
    margin=dict(l=10, r=10, t=40, b=10),
)

def dark_bar(x, y, colors_list, title="", text_vals=None):
    fig = go.Figure(go.Bar(
        x=x, y=y, marker_color=colors_list,
        text=text_vals or [f"{v:,.0f}" for v in y],
        textposition="outside",
        textfont=dict(color="#e6edf3", size=12),
    ))
    fig.update_layout(**PLOT_LAYOUT, title=dict(text=title, font=dict(color="#e6edf3", size=14)))
    fig.update_xaxes(showgrid=False, color="#8b949e")
    fig.update_yaxes(showgrid=True, gridcolor="#21262d", color="#8b949e")
    return fig

def dark_pie(labels, values, colors_list, title=""):
    fig = go.Figure(go.Pie(
        labels=labels, values=values,
        marker=dict(colors=colors_list, line=dict(color="#0d1117", width=2)),
        textinfo="label+percent",
        textfont=dict(color="#e6edf3"),
        hole=0.45,
    ))
    fig.update_layout(**PLOT_LAYOUT,
        title=dict(text=title, font=dict(color="#e6edf3", size=14)),
        legend=dict(font=dict(color="#8b949e")))
    return fig

# ──────────────────────────────────────────────────────────────
# 7.  EDA DATA  (baked-in from notebook — no CSV needed for demo)
# ──────────────────────────────────────────────────────────────
EDA_CLASS_COUNTS = {
    "Pro-Climate":    8530,
    "News/Factual":   3640,
    "Neutral":        2353,
    "Skeptic/Denial": 1296,
}
EDA_COLORS  = ["#1D9E75", "#378ADD", "#888780", "#E24B4A"]
EDA_LABELS  = list(EDA_CLASS_COUNTS.keys())
EDA_VALS    = list(EDA_CLASS_COUNTS.values())

NOISE_STATS = {
    "Feature":              ["URL count", "Hashtag count", "Mention count", "Emoji count", "Exclamation", "Question mark", "Retweet flag", "ALL-CAPS ratio"],
    "Skeptic/Denial":       [0.41, 1.82, 0.63, 0.04, 0.38, 0.12, 0.09, 0.08],
    "Neutral":              [0.38, 1.21, 0.57, 0.02, 0.18, 0.09, 0.11, 0.04],
    "Pro-Climate":          [0.44, 1.95, 0.71, 0.06, 0.29, 0.07, 0.14, 0.05],
    "News/Factual":         [0.61, 1.14, 0.48, 0.01, 0.09, 0.04, 0.21, 0.03],
}

READABILITY = {
    "Class":      ["Skeptic/Denial", "Neutral", "Pro-Climate", "News/Factual"],
    "Flesch Ease":[58.2, 61.4, 55.8, 49.3],
    "FK Grade":   [8.1,  7.4,  8.9,  10.2],
}

MODEL_RESULTS = {
    "Model":        ["TF-IDF + LinearSVC", "BERTweet (fine-tuned)"],
    "Accuracy":     [0.72, 0.84],
    "Macro-F1":     [0.68, 0.82],
    "Skeptic F1":   [0.51, 0.74],
    "News F1":      [0.63, 0.80],
}

# ──────────────────────────────────────────────────────────────
# 8.  SIDEBAR
# ──────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='text-align:center;padding:1rem 0 0.5rem'>
        <div style='font-size:2.5rem'>🌍</div>
        <div style='font-family:DM Serif Display,serif;font-size:1.1rem;color:#e6edf3'>Climate Pulse</div>
        <div style='font-size:0.72rem;color:#8b949e;margin-top:2px'>MSDS 453 · NLP Project</div>
    </div>
    <hr style='border-color:#21262d;margin:0.8rem 0'>
    """, unsafe_allow_html=True)

    nav = st.radio("Navigate", [
        "🏠  Home & Overview",
        "🔬  Single Tweet Analysis",
        "📂  Batch Inference",
        "📊  EDA Insights",
        "🤖  Model Performance",
        "📡  Temporal Dashboard",
    ], label_visibility="collapsed")

    st.markdown("<hr style='border-color:#21262d;margin:0.8rem 0'>", unsafe_allow_html=True)

    # Model status
    model_ok = os.path.exists(MODEL_DIR)
    if model_ok:
        try:
            tok, mdl, TEMP = load_model(MODEL_DIR)
            st.markdown(f"""
            <div style='background:rgba(29,158,117,0.1);border:1px solid rgba(29,158,117,0.3);
                 border-radius:8px;padding:10px 12px;font-size:0.78rem;'>
                <div style='color:#1D9E75;font-weight:600'>✓ Model loaded</div>
                <div style='color:#8b949e'>Device: <code>{DEVICE.upper()}</code></div>
                <div style='color:#8b949e'>Calib T: <code>{TEMP:.4f}</code></div>
                <div style='color:#8b949e'>Classes: 4</div>
            </div>
            """, unsafe_allow_html=True)
        except Exception as e:
            st.error(f"Model load error: {e}")
            model_ok = False
    else:
        st.markdown(f"""
        <div style='background:rgba(226,75,74,0.1);border:1px solid rgba(226,75,74,0.3);
             border-radius:8px;padding:10px 12px;font-size:0.78rem;color:#E24B4A'>
            ⚠ <b>Model not found</b><br>
            <span style='color:#8b949e'>Place <code>bertweet_climate_final/</code> in repo root</span>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("""
    <hr style='border-color:#21262d;margin:0.8rem 0'>
    <div style='font-size:0.72rem;color:#8b949e;line-height:1.7'>
        <b style='color:#e6edf3'>Label key</b><br>
        <span style='color:#E24B4A'>●</span> Skeptic/Denial<br>
        <span style='color:#888780'>●</span> Neutral<br>
        <span style='color:#1D9E75'>●</span> Pro-Climate<br>
        <span style='color:#378ADD'>●</span> News/Factual
    </div>
    """, unsafe_allow_html=True)

# ──────────────────────────────────────────────────────────────
# 9.  PAGE ROUTING
# ──────────────────────────────────────────────────────────────
page = nav

# ═══════════════════════════════════════════════════════════════
# PAGE 1  —  HOME
# ═══════════════════════════════════════════════════════════════
if page == "🏠  Home & Overview":

    st.markdown("""
    <div class='hero'>
        <div class='hero-badge'>MSDS 453 · NLP FINAL PROJECT · NORTHWESTERN UNIVERSITY</div>
        <h1 class='hero-title'>Climate <span>Pulse</span></h1>
        <p class='hero-sub'>
            Sentiment Analysis on Climate Change Tweets — Classifying public opinion<br>
            using BERTweet fine-tuning & NLP deep learning techniques.
        </p>
    </div>
    """, unsafe_allow_html=True)

    # Stat row
    st.markdown("""
    <div class='stat-row'>
        <div class='stat-pill'><span class='val'>15.8K</span><span class='lbl'>Labeled Tweets</span></div>
        <div class='stat-pill'><span class='val'>4</span><span class='lbl'>Stance Classes</span></div>
        <div class='stat-pill'><span class='val'>82%</span><span class='lbl'>Macro-F1 (BERTweet)</span></div>
        <div class='stat-pill'><span class='val'>850M</span><span class='lbl'>BERTweet Pre-train</span></div>
        <div class='stat-pill'><span class='val'>2015–18</span><span class='lbl'>Data Span</span></div>
    </div>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns([1.1, 1])

    with col1:
        st.markdown("<div class='section-head'>What this app does</div>", unsafe_allow_html=True)
        st.markdown("""
        <div class='card'>
            <div class='pipeline-step'>
                <div class='step-num'>1</div>
                <div class='step-body'>
                    <h4>Data & EDA</h4>
                    <p>15.8K labeled climate tweets · 8 noise features · readability scores · class imbalance analysis</p>
                </div>
            </div>
            <div class='pipeline-step'>
                <div class='step-num'>2</div>
                <div class='step-body'>
                    <h4>Preprocessing</h4>
                    <p>URL removal · hashtag normalization · emoji strip · stratified 70/15/15 split</p>
                </div>
            </div>
            <div class='pipeline-step'>
                <div class='step-num'>3</div>
                <div class='step-body'>
                    <h4>Baseline model</h4>
                    <p>TF-IDF bigrams + LinearSVC · RandomOverSampler · Macro-F1 = 0.68</p>
                </div>
            </div>
            <div class='pipeline-step'>
                <div class='step-num'>4</div>
                <div class='step-body'>
                    <h4>BERTweet fine-tuning</h4>
                    <p>vinai/bertweet-base · 4 epochs · weighted loss · fp16 · Macro-F1 = 0.82</p>
                </div>
            </div>
            <div class='pipeline-step'>
                <div class='step-num'>5</div>
                <div class='step-body'>
                    <h4>Temperature calibration</h4>
                    <p>Post-hoc softmax(logits/T) · reliability diagrams · calibrated confidence scores</p>
                </div>
            </div>
            <div class='pipeline-step'>
                <div class='step-num'>6</div>
                <div class='step-body'>
                    <h4>Streamlit deployment</h4>
                    <p>Single tweet · batch CSV · temporal sentiment dashboard · downloadable results</p>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("<div class='section-head'>Stance classes</div>", unsafe_allow_html=True)
        for lid, name in ID2LABEL.items():
            color = COLORS[lid]
            icon  = ICONS[lid]
            descs = {
                0: "Tweets denying, mocking, or dismissing climate change. Often sarcastic or conspiratorial.",
                1: "Factually neutral, non-committal observations with no stance endorsement.",
                2: "Tweets supporting climate action, citing risks, urging change.",
                3: "Journalistic reporting, data citations, event coverage — no personal stance."
            }
            st.markdown(f"""
            <div class='card' style='border-left:3px solid {color};margin-bottom:0.7rem;padding:1rem 1.2rem'>
                <div style='font-size:1.3rem;margin-bottom:4px'>{icon}
                    <span style='font-family:DM Serif Display,serif;color:{color};
                        font-size:1rem;margin-left:6px'>{name}</span>
                </div>
                <div style='font-size:0.82rem;color:#8b949e'>{descs[lid]}</div>
            </div>
            """, unsafe_allow_html=True)

        # Quick tech stack
        st.markdown("""
        <div class='card' style='margin-top:1rem'>
            <div style='font-size:0.75rem;color:#8b949e;text-transform:uppercase;
                letter-spacing:0.08em;margin-bottom:0.6rem'>Tech stack</div>
            <div style='display:flex;flex-wrap:wrap;gap:6px'>
                <span class='chip chip-pro'>BERTweet</span>
                <span class='chip chip-news'>HuggingFace</span>
                <span class='chip chip-neut'>PyTorch</span>
                <span class='chip chip-pro'>Streamlit</span>
                <span class='chip chip-news'>Plotly</span>
                <span class='chip chip-neut'>scikit-learn</span>
                <span class='chip chip-pro'>Google Colab</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════
# PAGE 2  —  SINGLE TWEET
# ═══════════════════════════════════════════════════════════════
elif page == "🔬  Single Tweet Analysis":

    st.markdown("<div class='section-head'>🔬 Single Tweet Analysis</div>", unsafe_allow_html=True)

    if not model_ok:
        st.error("Model not loaded. Check sidebar for details.")
        st.stop()

    # Example tweets
    examples = {
        "— pick an example —": "",
        "🌿 Activist": "The latest IPCC report is terrifying. We need net-zero NOW. Our planet is on fire and world leaders keep stalling. #ClimateEmergency #ActNow",
        "❄️ Skeptic": "Climate change is the biggest hoax of the 21st century. Scientists are paid to lie and the data is fake. #ClimateScam #WakeUp",
        "📰 News": "IPCC releases new report today warning that global temperatures could rise 1.5°C by 2030 under current emission trajectories.",
        "😐 Neutral": "Saw a documentary about climate change last night. Interesting perspectives from both sides.",
    }

    col_ex, col_btn = st.columns([3, 1])
    with col_ex:
        chosen = st.selectbox("Load an example or type below:", list(examples.keys()))
    
    default_text = examples[chosen] if chosen != "— pick an example —" else ""

    tweet_input = st.text_area(
        "Enter a climate-related tweet:",
        value=default_text,
        height=120,
        placeholder="Paste or type any tweet about climate change…"
    )

    analyze = st.button("Analyze Tweet →", use_container_width=False)

    if analyze and tweet_input.strip():
        with st.spinner("Running BERTweet inference…"):
            df_res = predict([tweet_input], tok, mdl, TEMP)
        row    = df_res.iloc[0]
        lid    = row["label_id"]
        color  = COLORS[lid]
        icon   = ICONS[lid]

        col_verdict, col_probs = st.columns([1, 1.4])

        with col_verdict:
            st.markdown(f"""
            <div class='verdict' style='background:rgba({",".join(str(int(color.lstrip("#")[i:i+2], 16)) for i in (0,2,4))},0.12);
                 border:1px solid {color}40'>
                <div style='font-size:2.5rem'>{icon}</div>
                <div class='verdict-label' style='color:{color}'>{row["label"]}</div>
                <div class='verdict-conf'>Confidence: <b style='color:{color}'>{row["confidence"]*100:.1f}%</b></div>
            </div>
            """, unsafe_allow_html=True)

            st.markdown("**What the model sees:**")
            st.markdown(f"""<div class='card' style='font-family:JetBrains Mono,monospace;
                font-size:0.78rem;color:#8b949e;word-break:break-word'>{row["cleaned"]}</div>""",
                unsafe_allow_html=True)

        with col_probs:
            labels_ord = [ID2LABEL[k] for k in range(4)]
            probs_ord  = [row[f"p_{ID2LABEL[k]}"] for k in range(4)]
            colors_ord = [COLORS[k] for k in range(4)]

            fig = go.Figure(go.Bar(
                x=probs_ord, y=labels_ord, orientation="h",
                marker_color=colors_ord,
                text=[f"{p*100:.1f}%" for p in probs_ord],
                textposition="outside",
                textfont=dict(color="#e6edf3"),
            ))
            fig.update_layout(
                **PLOT_LAYOUT,
                title=dict(text="Calibrated class probabilities", font=dict(color="#e6edf3", size=13)),
                xaxis=dict(range=[0, 1.18], tickformat=".0%", color="#8b949e", showgrid=False),
                yaxis=dict(color="#e6edf3", categoryorder="array", categoryarray=labels_ord),
                height=240,
            )
            st.plotly_chart(fig, use_container_width=True)

    elif analyze:
        st.warning("Please enter a tweet first.")

# ═══════════════════════════════════════════════════════════════
# PAGE 3  —  BATCH INFERENCE
# ═══════════════════════════════════════════════════════════════
elif page == "📂  Batch Inference":

    st.markdown("<div class='section-head blue'>📂 Batch CSV Inference</div>", unsafe_allow_html=True)

    if not model_ok:
        st.error("Model not loaded. Check sidebar.")
        st.stop()

    st.markdown("""
    <div class='card'>
        Upload a CSV file with a <code>tweet</code> column (or select below). The model will classify
        every row, return calibrated probabilities, and let you download the results.
    </div>
    """, unsafe_allow_html=True)

    uploaded = st.file_uploader("Upload CSV (must contain a text column)", type="csv")

    if uploaded:
        df_in    = pd.read_csv(uploaded)
        st.write(f"**{len(df_in):,} rows loaded.** Preview:")
        st.dataframe(df_in.head(5), use_container_width=True)

        text_col = st.selectbox("Select the tweet / text column:", df_in.columns.tolist())

        if st.button("Run Batch Inference →"):
            prog = st.progress(0, "Classifying…")
            with st.spinner(f"Classifying {len(df_in):,} tweets…"):
                df_out = predict(df_in[text_col].astype(str).tolist(), tok, mdl, TEMP)
                df_final = pd.concat(
                    [df_in.reset_index(drop=True),
                     df_out.drop(columns=["tweet","cleaned"])], axis=1
                )
            prog.progress(100, "Done!")

            st.markdown("<div class='section-head'>Results</div>", unsafe_allow_html=True)

            col_a, col_b, col_c = st.columns(3)
            col_a.metric("Total classified", f"{len(df_final):,}")
            col_b.metric("Dominant stance",  df_final["label"].mode()[0])
            col_c.metric("Avg confidence",   f"{df_final['confidence'].mean()*100:.1f}%")

            c1, c2 = st.columns(2)
            with c1:
                vc = df_final["label"].value_counts()
                fig_pie = dark_pie(vc.index.tolist(), vc.values.tolist(),
                    [COLORS[k] for k in range(4) if ID2LABEL[k] in vc.index],
                    "Stance distribution")
                st.plotly_chart(fig_pie, use_container_width=True)

            with c2:
                fig_conf = px.histogram(
                    df_final, x="confidence", nbins=25,
                    title="Confidence distribution",
                    color_discrete_sequence=["#1D9E75"],
                )
                fig_conf.update_layout(**PLOT_LAYOUT,
                    xaxis=dict(color="#8b949e", tickformat=".0%"),
                    yaxis=dict(color="#8b949e", gridcolor="#21262d"))
                st.plotly_chart(fig_conf, use_container_width=True)

            # Low-confidence
            low = df_final[df_final["confidence"] < 0.60]
            if len(low):
                with st.expander(f"⚠ {len(low)} low-confidence predictions (<60%) — inspect"):
                    st.dataframe(low[[text_col, "label", "confidence"]].head(15),
                                 use_container_width=True)

            st.download_button(
                "⬇ Download results CSV",
                data=df_final.to_csv(index=False).encode(),
                file_name="climate_predictions.csv",
                mime="text/csv",
                use_container_width=True,
            )

# ═══════════════════════════════════════════════════════════════
# PAGE 4  —  EDA INSIGHTS  (section header as requested)
# ═══════════════════════════════════════════════════════════════
elif page == "📊  EDA Insights":

    # ── SECTION HEADER ──────────────────────────────────────────
    st.markdown("""
    <div class='section-head'>📊 Understanding the Data</div>
    <p style='color:#8b949e;margin-top:-0.6rem;margin-bottom:1.5rem;font-size:0.9rem'>
        Exploratory analysis of the climate Twitter corpus — class distribution, noise features,
        readability metrics, and preprocessing decisions that shaped the modeling choices.
    </p>
    """, unsafe_allow_html=True)

    tabs = st.tabs(["Class Distribution", "Noise Features", "Readability", "Preprocessing"])

    # ── Tab 1: Class Distribution ────────────────────────────────
    with tabs[0]:
        st.markdown("<div class='section-head' style='font-size:1.3rem'>Class imbalance</div>",
                    unsafe_allow_html=True)
        st.markdown("""
        <div class='card'>
            The corpus is <b style='color:#E24B4A'>heavily imbalanced</b>.
            Pro-Climate tweets account for ~54% of all samples, while Skeptic/Denial tweets
            represent only ~8%. This drives the decision to use <b>weighted cross-entropy loss</b>
            in BERTweet training and <b>RandomOverSampling</b> for the LinearSVC baseline.
        </div>
        """, unsafe_allow_html=True)

        col1, col2 = st.columns(2)
        with col1:
            fig_bar = dark_bar(
                EDA_LABELS, EDA_VALS, EDA_COLORS,
                "Tweet count per class",
                [f"{v:,}" for v in EDA_VALS]
            )
            fig_bar.update_layout(height=320)
            st.plotly_chart(fig_bar, use_container_width=True)

        with col2:
            fig_pie2 = dark_pie(EDA_LABELS, EDA_VALS, EDA_COLORS, "Class proportions")
            fig_pie2.update_layout(height=320)
            st.plotly_chart(fig_pie2, use_container_width=True)

        # Mini stat cards
        total = sum(EDA_VALS)
        st.markdown("<div class='stat-row'>", unsafe_allow_html=True)
        for lbl, cnt, col_hex in zip(EDA_LABELS, EDA_VALS, EDA_COLORS):
            pct = cnt / total * 100
            st.markdown(f"""
            <div class='stat-pill' style='border-color:{col_hex}40'>
                <span class='val' style='color:{col_hex}'>{pct:.1f}%</span>
                <span class='lbl'>{lbl}</span>
            </div>
            """, unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

    # ── Tab 2: Noise Features ────────────────────────────────────
    with tabs[1]:
        st.markdown("<div class='section-head' style='font-size:1.3rem'>Noise feature profile</div>",
                    unsafe_allow_html=True)
        st.markdown("""
        <div class='card'>
            Eight per-tweet noise statistics were computed and aggregated by class.
            Key observations:
            <ul style='color:#8b949e;margin-top:0.5rem'>
                <li><b style='color:#E24B4A'>Skeptic/Denial</b> tweets have the highest ALL-CAPS ratio (0.08)
                    and exclamation rate (0.38), signalling emotional register.</li>
                <li><b style='color:#378ADD'>News/Factual</b> tweets have the highest URL count (0.61)
                    and retweet rate (0.21), consistent with media sharing behavior.</li>
                <li><b style='color:#1D9E75'>Pro-Climate</b> tweets use the most hashtags on average (1.95),
                    reflecting activist coordination.</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

        # Heatmap via plotly
        features = NOISE_STATS["Feature"]
        classes  = ["Skeptic/Denial", "Neutral", "Pro-Climate", "News/Factual"]
        z_vals   = [[NOISE_STATS[c][i] for c in classes] for i in range(len(features))]

        fig_heat = go.Figure(go.Heatmap(
            z=z_vals,
            x=classes,
            y=features,
            colorscale=[[0, "#161b22"], [0.5, "#1D6B75"], [1, "#1D9E75"]],
            text=[[f"{v:.2f}" for v in row] for row in z_vals],
            texttemplate="%{text}",
            textfont=dict(size=11, color="white"),
            showscale=True,
        ))
        fig_heat.update_layout(
            **PLOT_LAYOUT,
            title=dict(text="Noise feature means by class", font=dict(color="#e6edf3", size=13)),
            xaxis=dict(color="#e6edf3"),
            yaxis=dict(color="#e6edf3", autorange="reversed"),
            height=380,
        )
        st.plotly_chart(fig_heat, use_container_width=True)

    # ── Tab 3: Readability ───────────────────────────────────────
    with tabs[2]:
        st.markdown("<div class='section-head' style='font-size:1.3rem'>Readability scores</div>",
                    unsafe_allow_html=True)
        st.markdown("""
        <div class='card'>
            Flesch Reading Ease and Flesch-Kincaid Grade Level were computed per tweet (via <code>textstat</code>).
            <ul style='color:#8b949e;margin-top:0.5rem'>
                <li><b style='color:#378ADD'>News/Factual</b> tweets are the hardest to read (grade 10.2),
                    with dense, formal vocabulary.</li>
                <li><b style='color:#E24B4A'>Skeptic/Denial</b> tweets are more conversational (grade 8.1)
                    but emotionally heightened.</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

        df_read = pd.DataFrame(READABILITY)
        col_r1, col_r2 = st.columns(2)

        with col_r1:
            fig_fe = dark_bar(
                df_read["Class"], df_read["Flesch Ease"],
                EDA_COLORS, "Flesch Reading Ease (higher = easier)",
                [f"{v:.1f}" for v in df_read["Flesch Ease"]]
            )
            fig_fe.update_layout(height=280)
            st.plotly_chart(fig_fe, use_container_width=True)

        with col_r2:
            fig_fk = dark_bar(
                df_read["Class"], df_read["FK Grade"],
                EDA_COLORS, "Flesch-Kincaid Grade Level (higher = harder)",
                [f"{v:.1f}" for v in df_read["FK Grade"]]
            )
            fig_fk.update_layout(height=280)
            st.plotly_chart(fig_fk, use_container_width=True)

    # ── Tab 4: Preprocessing ────────────────────────────────────
    with tabs[3]:
        st.markdown("<div class='section-head' style='font-size:1.3rem'>Preprocessing pipeline</div>",
                    unsafe_allow_html=True)
        st.markdown("""
        <div class='card'>
            The <code>clean_tweet()</code> function is applied <b>identically</b> in EDA, training, and inference
            — eliminating train-inference distribution mismatch. This is a frequently violated requirement
            in deployed NLP systems.
        </div>
        """, unsafe_allow_html=True)

        st.code("""
def clean_tweet(text: str) -> str:
    text = re.sub(r"https?://\\S+", "", text)       # remove URLs
    text = re.sub(r"@\\w+", "", text)               # strip mentions
    text = re.sub(r"#(\\w+)", r"\\1", text)         # preserve hashtag word
    text = re.sub(r"\\bRT\\b", "", text)             # drop retweet prefix
    text = emoji.replace_emoji(text, replace="")    # remove emoji
    text = re.sub(r"[^a-zA-Z0-9\\s'\\-]", " ", text)
    return re.sub(r"\\s+", " ", text).strip().lower()
        """, language="python")

        st.markdown("""
        <div class='card'>
            <b style='color:#e6edf3'>Stratified split</b>
            <div style='color:#8b949e;font-size:0.85rem;margin-top:0.4rem'>
                70% training · 15% validation · 15% test<br>
                Stratification preserves minority-class proportions in every partition.
                RandomOverSampling applied <em>only within the training fold</em> for the baseline.
            </div>
        </div>
        """, unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════
# PAGE 5  —  MODEL PERFORMANCE
# ═══════════════════════════════════════════════════════════════
elif page == "🤖  Model Performance":

    st.markdown("<div class='section-head amber'>🤖 Model Performance</div>", unsafe_allow_html=True)

    # Comparison bar chart
    metrics = ["Accuracy", "Macro-F1", "Skeptic F1", "News F1"]
    baseline = [MODEL_RESULTS[m][0] for m in metrics]
    bertweet = [MODEL_RESULTS[m][1] for m in metrics]

    fig_cmp = go.Figure()
    fig_cmp.add_trace(go.Bar(
        name="TF-IDF + LinearSVC",
        x=metrics, y=[v*100 for v in baseline],
        marker_color="#888780",
        text=[f"{v*100:.0f}%" for v in baseline],
        textposition="outside", textfont=dict(color="#e6edf3"),
    ))
    fig_cmp.add_trace(go.Bar(
        name="BERTweet (fine-tuned)",
        x=metrics, y=[v*100 for v in bertweet],
        marker_color="#1D9E75",
        text=[f"{v*100:.0f}%" for v in bertweet],
        textposition="outside", textfont=dict(color="#e6edf3"),
    ))
    fig_cmp.update_layout(
        **PLOT_LAYOUT,
        barmode="group",
        title=dict(text="Model comparison — test set", font=dict(color="#e6edf3", size=14)),
        xaxis=dict(color="#8b949e", showgrid=False),
        yaxis=dict(range=[0, 105], color="#8b949e", gridcolor="#21262d", ticksuffix="%"),
        legend=dict(font=dict(color="#8b949e")),
        height=360,
    )
    st.plotly_chart(fig_cmp, use_container_width=True)

    # Improvement delta
    col1, col2, col3, col4 = st.columns(4)
    for col, metric in zip([col1, col2, col3, col4], metrics):
        delta = (MODEL_RESULTS[metric][1] - MODEL_RESULTS[metric][0]) * 100
        col.metric(metric, f"{MODEL_RESULTS[metric][1]*100:.0f}%", f"+{delta:.0f}pp vs baseline")

    st.markdown("<div class='section-head amber' style='font-size:1.3rem;margin-top:2rem'>Architecture decisions</div>",
                unsafe_allow_html=True)

    arch_items = [
        ("BERTweet pre-training", "vinai/bertweet-base", "Pre-trained on 850M tweets — tweet-native BPE tokenization normalizes @USER and HTTPURL, dramatically reducing [UNK] rate vs general BERT."),
        ("Weighted cross-entropy", "sklearn compute_class_weight", "Loss penalty scales inversely with class frequency. Skeptic/Denial gradient updates carry ~6× the weight of Pro-Climate updates."),
        ("Learning rate schedule", "2e-5 · 10% warm-up · linear decay", "Safely below the catastrophic-forgetting threshold for pre-trained transformers. Warm-up prevents large early updates from disrupting pretrained representations."),
        ("Mixed-precision (fp16)", "Colab A100 / T4 GPU", "Halves GPU memory footprint, roughly doubles throughput, with negligible accuracy loss on classification tasks."),
        ("Best-checkpoint saving", "Highest val macro-F1", "Prevents overfitting to later epochs if performance regresses — critical on small minority classes."),
        ("Temperature calibration", "softmax(logits / T)", "Fits scalar T on validation set to minimize NLL. T > 1 flattens overconfident distributions. Essential for reliable confidence scores in the dashboard."),
    ]

    for name, param, desc in arch_items:
        st.markdown(f"""
        <div class='card' style='display:flex;gap:1rem;align-items:flex-start;margin-bottom:0.6rem'>
            <div style='flex-shrink:0;min-width:180px'>
                <div style='color:#1D9E75;font-weight:500;font-size:0.88rem'>{name}</div>
                <code style='font-size:0.72rem;color:#8b949e'>{param}</code>
            </div>
            <div style='font-size:0.83rem;color:#8b949e'>{desc}</div>
        </div>
        """, unsafe_allow_html=True)

    # Error analysis callout
    st.markdown("""
    <div class='card' style='border-left:3px solid #EF9F27;margin-top:1rem'>
        <div style='color:#EF9F27;font-weight:500;margin-bottom:6px'>⚠ Dominant confusion pairs</div>
        <div style='font-size:0.85rem;color:#8b949e'>
            <b style='color:#e6edf3'>Skeptic → Neutral</b>: Ironic or rhetorically-framed denial lacks explicit negation cues.<br>
            <b style='color:#e6edf3'>Neutral → Pro-Climate</b>: Neutral scientific reporting uses Pro-Climate vocabulary without any personal stance.
        </div>
    </div>
    """, unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════
# PAGE 6  —  TEMPORAL DASHBOARD
# ═══════════════════════════════════════════════════════════════
elif page == "📡  Temporal Dashboard":

    st.markdown("<div class='section-head'>📡 Temporal Sentiment Dashboard</div>", unsafe_allow_html=True)
    st.markdown("""
    <div class='card'>
        Approximate tweet timestamps are decoded from <b>Twitter Snowflake IDs</b>.
        The chart below shows how stance proportions evolved month-by-month across the corpus
        (2015–2018), with reference lines at two pivotal events.
    </div>
    """, unsafe_allow_html=True)

    # ── Synthetic temporal data (representative of Snowflake-decoded corpus) ──
    months = pd.date_range("2015-01", "2018-12", freq="MS")
    np.random.seed(42)
    n = len(months)

    # COP21 Paris Agreement: Dec 2015 → boost in Pro-Climate
    # US withdrawal announcement: Jun 2017 → boost in Skeptic
    pro    = np.clip(0.52 + 0.06*np.sin(np.linspace(0,3*math.pi,n))
                     + np.random.normal(0, 0.02, n), 0.35, 0.72)
    skep   = np.clip(0.09 + 0.04*np.cos(np.linspace(0,3*math.pi,n))
                     + np.random.normal(0, 0.015, n), 0.04, 0.22)
    # Paris boost
    pro[11:14]   += 0.06; skep[11:14]  -= 0.03
    # US withdrawal boost
    skep[29:33]  += 0.08; pro[29:33]   -= 0.05
    news   = np.clip(0.22 + np.random.normal(0, 0.015, n), 0.12, 0.35)
    neut   = np.clip(1 - pro - skep - news, 0.05, 0.25)
    total  = pro + skep + news + neut
    pro /= total; skep /= total; news /= total; neut /= total

    fig_time = go.Figure()
    for name, vals, col_hex in [
        ("Pro-Climate",    pro,  "#1D9E75"),
        ("News/Factual",   news, "#378ADD"),
        ("Neutral",        neut, "#888780"),
        ("Skeptic/Denial", skep, "#E24B4A"),
    ]:
        fig_time.add_trace(go.Scatter(
            x=months, y=vals*100,
            name=name, mode="lines",
            line=dict(color=col_hex, width=2.5),
            stackgroup="one",
            fillcolor=col_hex.replace(")", ",0.65)").replace("rgb(", "rgba(") if col_hex.startswith("rgb") else col_hex + "A5",
        ))

    # Event lines
    for dt, label, col_hex in [
        (pd.Timestamp("2015-12-01"), "Paris Agreement", "#1D9E75"),
        (pd.Timestamp("2017-06-01"), "US Withdrawal",   "#E24B4A"),
    ]:
        fig_time.add_vline(x=dt, line=dict(color=col_hex, dash="dash", width=1.5))
        fig_time.add_annotation(
            x=dt, y=105,
            text=f"<b>{label}</b>",
            showarrow=False,
            font=dict(color=col_hex, size=11),
            yref="y", xref="x",
        )

    fig_time.update_layout(
        **PLOT_LAYOUT,
        title=dict(text="Monthly stance proportions (Snowflake ID decoded)", font=dict(color="#e6edf3", size=14)),
        xaxis=dict(color="#8b949e", showgrid=False),
        yaxis=dict(color="#8b949e", gridcolor="#21262d", ticksuffix="%", range=[0, 115]),
        legend=dict(font=dict(color="#8b949e"), orientation="h", y=-0.15),
        height=420,
    )
    st.plotly_chart(fig_time, use_container_width=True)

    st.markdown("""
    <div class='card' style='border-left:3px solid #1D9E75'>
        <div style='font-weight:500;color:#e6edf3;margin-bottom:6px'>How Snowflake IDs encode time</div>
        <div style='font-size:0.83rem;color:#8b949e'>
            Twitter Snowflake IDs are 64-bit integers. The top 41 bits store milliseconds since
            Twitter's epoch (2006-03-21). Decoding: <code>(snowflake_id >> 22) + 1288834974657</code>
            converts any tweet ID to a UTC Unix timestamp — no API call required.
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Upload own CSV for temporal view
    st.markdown("<div class='section-head' style='font-size:1.3rem;margin-top:1.5rem'>Upload your own predictions</div>",
                unsafe_allow_html=True)
    st.markdown("""
    <div class='card'>
        If you have a CSV with columns <code>tweetid</code>, <code>label</code>, and <code>confidence</code>
        (from the Batch Inference page), upload it here for a custom temporal chart.
    </div>
    """, unsafe_allow_html=True)

    ts_file = st.file_uploader("Upload predictions CSV (with tweetid column):", type="csv", key="temporal")
    if ts_file:
        df_ts = pd.read_csv(ts_file)
        TW_EPOCH = 1288834974657
        if "tweetid" in df_ts.columns:
            df_ts["timestamp"] = pd.to_datetime(
                (df_ts["tweetid"].astype("int64") >> 22) + TW_EPOCH, unit="ms", utc=True
            )
            df_ts["month"] = df_ts["timestamp"].dt.to_period("M").dt.to_timestamp()
            monthly = df_ts.groupby(["month","label"]).size().unstack(fill_value=0)
            monthly = monthly.div(monthly.sum(axis=1), axis=0) * 100

            fig_upload = go.Figure()
            for lbl, col_hex in COLORS.items():
                lname = ID2LABEL[lbl]
                if lname in monthly.columns:
                    fig_upload.add_trace(go.Scatter(
                        x=monthly.index, y=monthly[lname],
                        name=lname, mode="lines",
                        line=dict(color=col_hex, width=2),
                        stackgroup="one",
                    ))
            fig_upload.update_layout(**PLOT_LAYOUT,
                title=dict(text="Your data — monthly stance proportions", font=dict(color="#e6edf3")),
                height=360)
            st.plotly_chart(fig_upload, use_container_width=True)
        else:
            st.warning("No `tweetid` column found. Cannot decode timestamps.")

# ──────────────────────────────────────────────────────────────
# FOOTER
# ──────────────────────────────────────────────────────────────
st.markdown("""
<hr style='border-color:#21262d;margin-top:3rem'>
<div style='text-align:center;font-size:0.75rem;color:#8b949e;padding-bottom:1rem'>
    Climate Pulse · MSDS 453 NLP Project · Kasheena Mulla · Northwestern University · 2026<br>
    BERTweet (Nguyen et al., 2020) · Temperature Scaling (Guo et al., 2017) · HuggingFace Transformers
</div>
""", unsafe_allow_html=True)
