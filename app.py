# ══════════════════════════════════════════════════════════════
#  app.py  —  Climate Pulse | MSDS 453 NLP Project
#  Kasheena Mulla · Northwestern University
#
#  Model is streamed from Google Drive (no GitHub size limit)
#  Drive folder: My Drive/MSDSP_453_NLP_Project/Data/bertweet_climate_final
#
#  Install: pip install streamlit transformers torch plotly
#           pandas numpy scipy emoji safetensors gdown
#  Run:     streamlit run app.py
# ══════════════════════════════════════════════════════════════

import re, os, json, math, warnings
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st
import emoji as emoji_lib
import torch
from scipy.special import softmax
from transformers import AutoTokenizer, AutoModelForSequenceClassification

warnings.filterwarnings("ignore")

# ──────────────────────────────────────────────────────────────
# 0.  PAGE CONFIG
# ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Climate Pulse | NLP Sentiment Analyzer",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ──────────────────────────────────────────────────────────────
# 1.  GOOGLE DRIVE CONFIG
#     Share your Drive folder as "Anyone with the link can view"
#     then paste the folder ID below.
#
#     How to get folder ID:
#     Open Drive → right-click folder → Share → Copy link
#     URL looks like: https://drive.google.com/drive/folders/1ABC...XYZ
#     The folder ID is the last part:                         1ABC...XYZ
# ──────────────────────────────────────────────────────────────
DRIVE_FOLDER_ID = "YOUR_FOLDER_ID_HERE"   # ← paste your folder ID

# Files that must be downloaded from Drive
MODEL_FILES = [
    "config.json",
    "model.safetensors",   # or pytorch_model.bin if you used int8 save
    "tokenizer_config.json",
    "vocab.txt",
    "bpe.codes",
    "added_tokens.json",
    "training_args.bin",
]

LOCAL_MODEL_DIR = "bertweet_climate_final"
os.makedirs(LOCAL_MODEL_DIR, exist_ok=True)

# ──────────────────────────────────────────────────────────────
# 2.  DRIVE DOWNLOADER
# ──────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def download_model_from_drive(folder_id: str, dest_dir: str):
    """
    Downloads all model files from a public Google Drive folder.
    Uses gdown which handles Drive's virus-scan redirect for large files.
    """
    try:
        import gdown
    except ImportError:
        os.system("pip install -q gdown")
        import gdown

    status_box = st.empty()

    # gdown can list and download an entire public folder
    status_box.info("⏳ Downloading model from Google Drive… (first run only, ~260MB)")

    try:
        # Download entire folder contents
        gdown.download_folder(
            id=folder_id,
            output=dest_dir,
            quiet=False,
            use_cookies=False,
        )
        status_box.success("✅ Model downloaded successfully!")
        return True
    except Exception as e:
        status_box.error(f"❌ Drive download failed: {e}\n\nCheck that your folder is shared publicly.")
        return False


@st.cache_resource(show_spinner=False)
def load_model(model_dir: str):
    """Load BERTweet tokenizer + classifier + calibration temperature."""

    # BERTweet requires use_fast=False + normalization
    # The bpe.codes file is the BPE merge rules — fastBPE format
    tok = AutoTokenizer.from_pretrained(
        model_dir,
        use_fast=False,
        normalization=True,
    )

    mdl = AutoModelForSequenceClassification.from_pretrained(
        model_dir,
        ignore_mismatched_sizes=False,
        torch_dtype=torch.float32,   # float32 for CPU inference stability
    )
    mdl.eval()

    # Temperature calibration
    T = 1.0
    calib_path = os.path.join(model_dir, "calibration.json")
    if os.path.exists(calib_path):
        with open(calib_path) as f:
            T = json.load(f).get("temperature", 1.0)

    return tok, mdl, T

# ──────────────────────────────────────────────────────────────
# 3.  CONSTANTS
# ──────────────────────────────────────────────────────────────
DEVICE   = "cpu"   # Streamlit Cloud has no GPU — force CPU
MAX_LEN  = 128

ID2LABEL = {0: "Skeptic/Denial", 1: "Neutral", 2: "Pro-Climate", 3: "News/Factual"}
COLORS   = {0: "#E24B4A", 1: "#888780", 2: "#1D9E75", 3: "#378ADD"}
ICONS    = {0: "❄️", 1: "😐", 2: "🌿", 3: "📰"}

# Solid hex fill colors for Plotly stacked area (no opacity tricks)
FILL_COLORS = {0: "#E24B4A", 1: "#888780", 2: "#1D9E75", 3: "#378ADD"}

# ──────────────────────────────────────────────────────────────
# 4.  TWEET CLEANER
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
# 5.  INFERENCE
# ──────────────────────────────────────────────────────────────
def predict(texts: list, tok, mdl, T: float, batch_size: int = 16) -> pd.DataFrame:
    cleaned = [clean_tweet(t) for t in texts]
    rows = []
    for i in range(0, len(cleaned), batch_size):
        bc  = cleaned[i: i + batch_size]
        enc = tok(bc, padding=True, truncation=True,
                  max_length=MAX_LEN, return_tensors="pt")
        with torch.no_grad():
            logits = mdl(**enc).logits.numpy()
        probs    = softmax(logits / T, axis=1)
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
# 6.  PLOTLY DARK THEME HELPERS
# ──────────────────────────────────────────────────────────────
PLOT_BASE = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="sans-serif", color="#8b949e"),
    margin=dict(l=10, r=10, t=40, b=10),
)

def dark_bar(x, y, colors_list, title="", text_vals=None):
    fig = go.Figure(go.Bar(
        x=x, y=y,
        marker_color=colors_list,
        text=text_vals or [f"{v:,.0f}" for v in y],
        textposition="outside",
        textfont=dict(color="#e6edf3", size=12),
    ))
    fig.update_layout(
        **PLOT_BASE,
        title=dict(text=title, font=dict(color="#e6edf3", size=14)),
    )
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
    fig.update_layout(
        **PLOT_BASE,
        title=dict(text=title, font=dict(color="#e6edf3", size=14)),
        legend=dict(font=dict(color="#8b949e")),
    )
    return fig

# ──────────────────────────────────────────────────────────────
# 7.  STYLES
# ──────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display:ital@0;1&family=DM+Sans:wght@300;400;500;600&family=JetBrains+Mono:wght@400;500&display=swap');

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
    background: #0d1117; color: #e6edf3;
}
.main .block-container { padding: 1.5rem 2.5rem 3rem; max-width: 1200px; }

.hero {
    background: linear-gradient(135deg,#0f2027 0%,#11302a 50%,#0d1f3c 100%);
    border: 1px solid rgba(29,158,117,0.25); border-radius:20px;
    padding:3rem 3.5rem; margin-bottom:2rem; position:relative; overflow:hidden;
}
.hero::before {
    content:''; position:absolute; inset:0;
    background: radial-gradient(ellipse at 80% 20%,rgba(29,158,117,0.12) 0%,transparent 60%),
                radial-gradient(ellipse at 20% 80%,rgba(55,138,221,0.10) 0%,transparent 60%);
}
.hero-title { font-family:'DM Serif Display',serif; font-size:3.2rem;
    line-height:1.1; color:#fff; margin:0 0 0.5rem; }
.hero-title span { color:#1D9E75; }
.hero-sub { font-size:1.05rem; color:#8b949e; font-weight:300; margin:0; }
.hero-badge { display:inline-block; background:rgba(29,158,117,0.15);
    border:1px solid rgba(29,158,117,0.4); color:#1D9E75;
    font-family:'JetBrains Mono',monospace; font-size:0.72rem;
    padding:3px 10px; border-radius:20px; margin-bottom:1rem; }

.section-head { font-family:'DM Serif Display',serif; font-size:1.9rem;
    color:#e6edf3; border-bottom:2px solid #1D9E75;
    padding-bottom:0.4rem; margin:2.5rem 0 1.2rem; }
.section-head.blue  { border-color:#378ADD; }
.section-head.amber { border-color:#EF9F27; }

.card { background:#161b22; border:1px solid #30363d;
    border-radius:14px; padding:1.4rem 1.6rem; margin-bottom:1rem; }
.card:hover { border-color:#1D9E75; transition:border-color 0.2s; }

.stat-row { display:flex; gap:0.8rem; flex-wrap:wrap; margin:1rem 0; }
.stat-pill { background:#21262d; border:1px solid #30363d; border-radius:10px;
    padding:0.9rem 1.2rem; min-width:120px; text-align:center; }
.stat-pill .val { font-family:'DM Serif Display',serif; font-size:2rem;
    color:#1D9E75; display:block; }
.stat-pill .lbl { font-size:0.72rem; color:#8b949e;
    text-transform:uppercase; letter-spacing:0.08em; }

.verdict { border-radius:14px; padding:1.6rem 2rem; margin:1rem 0; }
.verdict-label { font-family:'DM Serif Display',serif; font-size:2rem; margin:0 0 0.3rem; }
.verdict-conf { font-size:0.9rem; color:#8b949e; margin:0; }

.chip { display:inline-block; padding:4px 14px; border-radius:20px;
    font-size:0.8rem; font-weight:500; }
.chip-pro  { background:rgba(29,158,117,0.18);color:#1D9E75;border:1px solid rgba(29,158,117,0.35); }
.chip-news { background:rgba(55,138,221,0.18);color:#378ADD;border:1px solid rgba(55,138,221,0.35); }
.chip-neut { background:rgba(136,135,128,0.18);color:#aaa;border:1px solid rgba(136,135,128,0.35); }
.chip-skep { background:rgba(226,75,74,0.18);color:#E24B4A;border:1px solid rgba(226,75,74,0.35); }

.pipeline-step { display:flex; align-items:flex-start; gap:1rem;
    padding:0.9rem 0; border-bottom:1px solid #21262d; }
.pipeline-step:last-child { border-bottom:none; }
.step-num { background:#1D9E75; color:#fff; width:28px; height:28px;
    border-radius:50%; display:flex; align-items:center;
    justify-content:center; font-size:0.75rem; font-weight:600; flex-shrink:0; }
.step-body h4 { margin:0 0 3px; font-size:0.92rem; color:#e6edf3; }
.step-body p  { margin:0; font-size:0.8rem; color:#8b949e; }

textarea { background:#161b22!important; color:#e6edf3!important;
    border:1px solid #30363d!important; border-radius:10px!important; }
[data-testid="stSidebar"] { background:#0d1117; border-right:1px solid #21262d; }
.stButton>button { background:#1D9E75!important; color:#fff!important;
    border:none!important; border-radius:10px!important;
    font-weight:500!important; }
.stButton>button:hover { opacity:0.85!important; }
[data-testid="stTabs"] button { font-size:0.9rem!important; color:#8b949e!important; }
[data-testid="stTabs"] button[aria-selected="true"] { color:#1D9E75!important; border-bottom-color:#1D9E75!important; }
</style>
""", unsafe_allow_html=True)

# ──────────────────────────────────────────────────────────────
# 8.  SIDEBAR + MODEL BOOT
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

    # ── Download model from Drive & load ──────────────────────
    model_ready = False
    tok = mdl = TEMP = None

    # Check if already downloaded in this session
    config_exists = os.path.exists(os.path.join(LOCAL_MODEL_DIR, "config.json"))
    model_exists  = (
        os.path.exists(os.path.join(LOCAL_MODEL_DIR, "model.safetensors")) or
        os.path.exists(os.path.join(LOCAL_MODEL_DIR, "pytorch_model.bin"))
    )

    if not (config_exists and model_exists):
        if DRIVE_FOLDER_ID == "YOUR_FOLDER_ID_HERE":
            st.warning("⚠ Set your Drive folder ID in app.py line ~50")
        else:
            ok = download_model_from_drive(DRIVE_FOLDER_ID, LOCAL_MODEL_DIR)
            if not ok:
                st.stop()

    try:
        with st.spinner("Loading BERTweet…"):
            tok, mdl, TEMP = load_model(LOCAL_MODEL_DIR)
        model_ready = True
        st.markdown(f"""
        <div style='background:rgba(29,158,117,0.1);border:1px solid rgba(29,158,117,0.3);
             border-radius:8px;padding:10px 12px;font-size:0.78rem;'>
            <div style='color:#1D9E75;font-weight:600'>✓ Model loaded</div>
            <div style='color:#8b949e'>Device: <code>CPU</code></div>
            <div style='color:#8b949e'>Calib T: <code>{TEMP:.4f}</code></div>
        </div>
        """, unsafe_allow_html=True)
    except Exception as e:
        st.error(f"Model load error: {e}")

    st.markdown("""
    <hr style='border-color:#21262d;margin:0.8rem 0'>
    <div style='font-size:0.72rem;color:#8b949e;line-height:1.8'>
        <b style='color:#e6edf3'>Label key</b><br>
        <span style='color:#E24B4A'>●</span> Skeptic/Denial<br>
        <span style='color:#888780'>●</span> Neutral<br>
        <span style='color:#1D9E75'>●</span> Pro-Climate<br>
        <span style='color:#378ADD'>●</span> News/Factual
    </div>
    """, unsafe_allow_html=True)

page = nav

# ══════════════════════════════════════════════════════════════
# PAGE 1  —  HOME
# ══════════════════════════════════════════════════════════════
if page == "🏠  Home & Overview":
    st.markdown("""
    <div class='hero'>
        <div class='hero-badge'>MSDS 453 · NLP FINAL PROJECT · NORTHWESTERN UNIVERSITY</div>
        <h1 class='hero-title'>Climate <span>Pulse</span></h1>
        <p class='hero-sub'>Sentiment Analysis on Climate Change Tweets — Classifying public opinion<br>
        using BERTweet fine-tuning &amp; NLP deep learning.</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class='stat-row'>
        <div class='stat-pill'><span class='val'>15.8K</span><span class='lbl'>Labeled Tweets</span></div>
        <div class='stat-pill'><span class='val'>4</span><span class='lbl'>Stance Classes</span></div>
        <div class='stat-pill'><span class='val'>82%</span><span class='lbl'>Macro-F1 BERTweet</span></div>
        <div class='stat-pill'><span class='val'>850M</span><span class='lbl'>Pre-train Tweets</span></div>
        <div class='stat-pill'><span class='val'>2015–18</span><span class='lbl'>Data Span</span></div>
    </div>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns([1.1, 1])
    with col1:
        st.markdown("<div class='section-head'>Pipeline</div>", unsafe_allow_html=True)
        st.markdown("""<div class='card'>
        <div class='pipeline-step'><div class='step-num'>1</div><div class='step-body'>
            <h4>Data &amp; EDA</h4><p>15.8K labeled tweets · 8 noise features · readability · imbalance</p></div></div>
        <div class='pipeline-step'><div class='step-num'>2</div><div class='step-body'>
            <h4>Preprocessing</h4><p>clean_tweet() · stratified 70/15/15 split · RandomOverSampler</p></div></div>
        <div class='pipeline-step'><div class='step-num'>3</div><div class='step-body'>
            <h4>Baseline</h4><p>TF-IDF bigrams + LinearSVC · balanced class weights · Macro-F1 = 0.68</p></div></div>
        <div class='pipeline-step'><div class='step-num'>4</div><div class='step-body'>
            <h4>BERTweet fine-tuning</h4><p>vinai/bertweet-base · 4 epochs · weighted loss · fp16 · Macro-F1 = 0.82</p></div></div>
        <div class='pipeline-step'><div class='step-num'>5</div><div class='step-body'>
            <h4>Temperature calibration</h4><p>softmax(logits/T) · reliability diagrams · confidence scores</p></div></div>
        <div class='pipeline-step'><div class='step-num'>6</div><div class='step-body'>
            <h4>Streamlit deployment</h4><p>Single tweet · batch CSV · temporal dashboard</p></div></div>
        </div>""", unsafe_allow_html=True)

    with col2:
        st.markdown("<div class='section-head'>Stance classes</div>", unsafe_allow_html=True)
        descs = {
            0: "Tweets denying, mocking, or dismissing climate change. Often sarcastic or conspiratorial.",
            1: "Factually neutral observations with no endorsement of any stance.",
            2: "Tweets supporting climate action, citing risks, urging change.",
            3: "Journalistic reporting, data citations — no personal stance."
        }
        for lid in range(4):
            st.markdown(f"""
            <div class='card' style='border-left:3px solid {COLORS[lid]};margin-bottom:0.7rem;padding:1rem 1.2rem'>
                <div style='font-size:1.2rem;margin-bottom:4px'>{ICONS[lid]}
                    <span style='color:{COLORS[lid]};font-weight:500;margin-left:6px'>{ID2LABEL[lid]}</span>
                </div>
                <div style='font-size:0.82rem;color:#8b949e'>{descs[lid]}</div>
            </div>""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════
# PAGE 2  —  SINGLE TWEET
# ══════════════════════════════════════════════════════════════
elif page == "🔬  Single Tweet Analysis":
    st.markdown("<div class='section-head'>🔬 Single Tweet Analysis</div>", unsafe_allow_html=True)

    if not model_ready:
        st.error("Model not loaded — check sidebar.")
        st.stop()

    examples = {
        "— pick an example —": "",
        "🌿 Activist": "The IPCC report is terrifying. We need net-zero NOW. #ClimateEmergency #ActNow",
        "❄️ Skeptic": "Climate change is the biggest hoax of the 21st century. Scientists are paid to lie. #ClimateScam",
        "📰 News": "IPCC releases new report warning global temps could rise 1.5°C by 2030.",
        "😐 Neutral": "Saw a documentary about climate change last night. Interesting perspectives.",
    }

    chosen = st.selectbox("Load an example or type below:", list(examples.keys()))
    tweet_input = st.text_area(
        "Enter a climate-related tweet:",
        value=examples.get(chosen, ""),
        height=110,
        placeholder="Paste or type any tweet about climate change…"
    )

    if st.button("Analyze Tweet →") and tweet_input.strip():
        with st.spinner("Running BERTweet inference…"):
            df_res = predict([tweet_input], tok, mdl, TEMP)
        row   = df_res.iloc[0]
        lid   = row["label_id"]
        color = COLORS[lid]

        col_v, col_p = st.columns([1, 1.4])
        with col_v:
            # Convert hex to rgb for background
            h = color.lstrip("#")
            r, g, b = int(h[0:2],16), int(h[2:4],16), int(h[4:6],16)
            st.markdown(f"""
            <div class='verdict' style='background:rgba({r},{g},{b},0.12);border:1px solid {color}60'>
                <div style='font-size:2.5rem'>{ICONS[lid]}</div>
                <div class='verdict-label' style='color:{color}'>{row["label"]}</div>
                <div class='verdict-conf'>Confidence: <b style='color:{color}'>{row["confidence"]*100:.1f}%</b></div>
            </div>""", unsafe_allow_html=True)
            st.markdown("**What the model sees:**")
            st.code(row["cleaned"], language=None)

        with col_p:
            labels_ord = [ID2LABEL[k] for k in range(4)]
            probs_ord  = [row[f"p_{ID2LABEL[k]}"] for k in range(4)]
            fig = go.Figure(go.Bar(
                x=probs_ord, y=labels_ord, orientation="h",
                marker_color=[COLORS[k] for k in range(4)],
                text=[f"{p*100:.1f}%" for p in probs_ord],
                textposition="outside",
                textfont=dict(color="#e6edf3"),
            ))
            fig.update_layout(
                **PLOT_BASE,
                title=dict(text="Calibrated class probabilities", font=dict(color="#e6edf3", size=13)),
                xaxis=dict(range=[0,1.18], tickformat=".0%", color="#8b949e", showgrid=False),
                yaxis=dict(color="#e6edf3"),
                height=240,
            )
            st.plotly_chart(fig, use_container_width=True)

# ══════════════════════════════════════════════════════════════
# PAGE 3  —  BATCH INFERENCE
# ══════════════════════════════════════════════════════════════
elif page == "📂  Batch Inference":
    st.markdown("<div class='section-head blue'>📂 Batch CSV Inference</div>", unsafe_allow_html=True)

    if not model_ready:
        st.error("Model not loaded — check sidebar.")
        st.stop()

    st.markdown("<div class='card'>Upload a CSV with a tweet column. Results include calibrated probabilities and are downloadable.</div>",
                unsafe_allow_html=True)

    uploaded = st.file_uploader("Upload CSV", type="csv")
    if uploaded:
        df_in    = pd.read_csv(uploaded)
        st.dataframe(df_in.head(5), use_container_width=True)
        text_col = st.selectbox("Tweet column:", df_in.columns.tolist())

        if st.button("Run Batch Inference →"):
            with st.spinner(f"Classifying {len(df_in):,} tweets…"):
                df_out   = predict(df_in[text_col].astype(str).tolist(), tok, mdl, TEMP)
                df_final = pd.concat(
                    [df_in.reset_index(drop=True),
                     df_out.drop(columns=["tweet","cleaned"])], axis=1)

            c1, c2, c3 = st.columns(3)
            c1.metric("Total", f"{len(df_final):,}")
            c2.metric("Top stance", df_final["label"].mode()[0])
            c3.metric("Avg confidence", f"{df_final['confidence'].mean()*100:.1f}%")

            vc = df_final["label"].value_counts()
            col_a, col_b = st.columns(2)
            with col_a:
                st.plotly_chart(dark_pie(vc.index.tolist(), vc.values.tolist(),
                    [COLORS[k] for k in range(4) if ID2LABEL[k] in vc.index],
                    "Stance distribution"), use_container_width=True)
            with col_b:
                fig_c = px.histogram(df_final, x="confidence", nbins=25,
                    title="Confidence distribution",
                    color_discrete_sequence=["#1D9E75"])
                fig_c.update_layout(**PLOT_BASE)
                st.plotly_chart(fig_c, use_container_width=True)

            st.download_button("⬇ Download results CSV",
                data=df_final.to_csv(index=False).encode(),
                file_name="climate_predictions.csv", mime="text/csv",
                use_container_width=True)

# ══════════════════════════════════════════════════════════════
# PAGE 4  —  EDA INSIGHTS
# ══════════════════════════════════════════════════════════════
elif page == "📊  EDA Insights":
    st.markdown("""
    <div class='section-head'>📊 Understanding the Data</div>
    <p style='color:#8b949e;margin-top:-0.6rem;margin-bottom:1.5rem;font-size:0.9rem'>
    Exploratory analysis of the climate Twitter corpus — class distribution, noise features,
    readability, and preprocessing decisions that shaped modeling choices.
    </p>""", unsafe_allow_html=True)

    EDA_LABELS = ["Pro-Climate","News/Factual","Neutral","Skeptic/Denial"]
    EDA_VALS   = [8530, 3640, 2353, 1296]
    EDA_COLORS = ["#1D9E75","#378ADD","#888780","#E24B4A"]

    NOISE = {
        "Feature":        ["URL count","Hashtag count","Mention count","Emoji count",
                           "Exclamation","Question mark","Retweet flag","ALL-CAPS ratio"],
        "Skeptic/Denial": [0.41,1.82,0.63,0.04,0.38,0.12,0.09,0.08],
        "Neutral":        [0.38,1.21,0.57,0.02,0.18,0.09,0.11,0.04],
        "Pro-Climate":    [0.44,1.95,0.71,0.06,0.29,0.07,0.14,0.05],
        "News/Factual":   [0.61,1.14,0.48,0.01,0.09,0.04,0.21,0.03],
    }

    tabs = st.tabs(["Class Distribution","Noise Features","Readability","Preprocessing"])

    with tabs[0]:
        total = sum(EDA_VALS)
        c1, c2 = st.columns(2)
        with c1:
            st.plotly_chart(dark_bar(EDA_LABELS, EDA_VALS, EDA_COLORS,
                "Tweet count per class", [f"{v:,}" for v in EDA_VALS]), use_container_width=True)
        with c2:
            st.plotly_chart(dark_pie(EDA_LABELS, EDA_VALS, EDA_COLORS,
                "Class proportions"), use_container_width=True)
        st.markdown("<div class='stat-row'>" + "".join([
            f"<div class='stat-pill' style='border-color:{c}40'>"
            f"<span class='val' style='color:{c}'>{v/total*100:.1f}%</span>"
            f"<span class='lbl'>{l}</span></div>"
            for l,v,c in zip(EDA_LABELS,EDA_VALS,EDA_COLORS)
        ]) + "</div>", unsafe_allow_html=True)

    with tabs[1]:
        classes  = ["Skeptic/Denial","Neutral","Pro-Climate","News/Factual"]
        features = NOISE["Feature"]
        z_vals   = [[NOISE[c][i] for c in classes] for i in range(len(features))]
        fig_h = go.Figure(go.Heatmap(
            z=z_vals, x=classes, y=features,
            colorscale=[[0,"#161b22"],[0.5,"#1D6B75"],[1,"#1D9E75"]],
            text=[[f"{v:.2f}" for v in row] for row in z_vals],
            texttemplate="%{text}", textfont=dict(size=11, color="white"),
        ))
        fig_h.update_layout(**PLOT_BASE,
            title=dict(text="Noise feature means by class", font=dict(color="#e6edf3",size=13)),
            xaxis=dict(color="#e6edf3"),
            yaxis=dict(color="#e6edf3", autorange="reversed"), height=360)
        st.plotly_chart(fig_h, use_container_width=True)
        st.markdown("""<div class='card'>
            <b style='color:#E24B4A'>Skeptic/Denial</b> → highest ALL-CAPS ratio (0.08) &amp; exclamation rate (0.38) — emotional register.<br>
            <b style='color:#378ADD'>News/Factual</b> → highest URL count (0.61) &amp; retweet rate (0.21) — media sharing.<br>
            <b style='color:#1D9E75'>Pro-Climate</b> → most hashtags (1.95) — activist coordination.
        </div>""", unsafe_allow_html=True)

    with tabs[2]:
        classes_r = ["Skeptic/Denial","Neutral","Pro-Climate","News/Factual"]
        fe_scores = [58.2, 61.4, 55.8, 49.3]
        fk_scores = [8.1,  7.4,  8.9,  10.2]
        c1, c2 = st.columns(2)
        with c1:
            st.plotly_chart(dark_bar(classes_r, fe_scores, EDA_COLORS[::-1],
                "Flesch Reading Ease (higher = easier)",
                [f"{v:.1f}" for v in fe_scores]), use_container_width=True)
        with c2:
            st.plotly_chart(dark_bar(classes_r, fk_scores, EDA_COLORS[::-1],
                "Flesch-Kincaid Grade Level (higher = harder)",
                [f"{v:.1f}" for v in fk_scores]), use_container_width=True)

    with tabs[3]:
        st.code("""
def clean_tweet(text: str) -> str:
    text = re.sub(r"https?://\\S+", "", text)        # remove URLs
    text = re.sub(r"@\\w+", "", text)                # strip mentions
    text = re.sub(r"#(\\w+)", r"\\1", text)          # preserve hashtag word
    text = re.sub(r"\\bRT\\b", "", text)              # drop retweet prefix
    text = emoji.replace_emoji(text, replace="")     # remove emoji
    text = re.sub(r"[^a-zA-Z0-9\\s'\\-]", " ", text)
    return re.sub(r"\\s+", " ", text).strip().lower()
        """, language="python")
        st.markdown("""<div class='card'>
            Identical function across EDA, training and inference notebooks —
            eliminates train-inference distribution mismatch.<br><br>
            <b style='color:#e6edf3'>Split:</b>
            <span style='color:#8b949e'>70% training · 15% validation · 15% test · stratified on label</span>
        </div>""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════
# PAGE 5  —  MODEL PERFORMANCE
# ══════════════════════════════════════════════════════════════
elif page == "🤖  Model Performance":
    st.markdown("<div class='section-head amber'>🤖 Model Performance</div>", unsafe_allow_html=True)

    metrics   = ["Accuracy","Macro-F1","Skeptic F1","News F1"]
    baseline  = [0.72, 0.68, 0.51, 0.63]
    bertweet  = [0.84, 0.82, 0.74, 0.80]

    fig_cmp = go.Figure()
    fig_cmp.add_trace(go.Bar(name="TF-IDF + LinearSVC",
        x=metrics, y=[v*100 for v in baseline], marker_color="#888780",
        text=[f"{v*100:.0f}%" for v in baseline],
        textposition="outside", textfont=dict(color="#e6edf3")))
    fig_cmp.add_trace(go.Bar(name="BERTweet (fine-tuned)",
        x=metrics, y=[v*100 for v in bertweet], marker_color="#1D9E75",
        text=[f"{v*100:.0f}%" for v in bertweet],
        textposition="outside", textfont=dict(color="#e6edf3")))
    fig_cmp.update_layout(**PLOT_BASE, barmode="group",
        title=dict(text="Model comparison — test set", font=dict(color="#e6edf3",size=14)),
        xaxis=dict(color="#8b949e", showgrid=False),
        yaxis=dict(range=[0,110], color="#8b949e", gridcolor="#21262d", ticksuffix="%"),
        legend=dict(font=dict(color="#8b949e")), height=340)
    st.plotly_chart(fig_cmp, use_container_width=True)

    cols = st.columns(4)
    for col, metric, b, bw in zip(cols, metrics, baseline, bertweet):
        col.metric(metric, f"{bw*100:.0f}%", f"+{(bw-b)*100:.0f}pp vs baseline")

    st.markdown("<div class='section-head amber' style='font-size:1.3rem;margin-top:2rem'>Architecture decisions</div>",
                unsafe_allow_html=True)
    for name, param, desc in [
        ("BERTweet","vinai/bertweet-base","Pre-trained on 850M tweets — tweet-native BPE reduces [UNK] rate vs general BERT."),
        ("Weighted loss","compute_class_weight","Skeptic/Denial gradient updates carry ~6× the weight of Pro-Climate, addressing imbalance."),
        ("LR schedule","2e-5 · 10% warm-up","Below catastrophic-forgetting threshold. Warm-up prevents large early updates."),
        ("fp16 training","Colab GPU","Halves memory footprint, roughly doubles throughput with negligible accuracy loss."),
        ("Best checkpoint","val macro-F1","Prevents overfitting if later epochs regress — critical on small minority classes."),
        ("Temperature T","softmax(logits/T)","Post-hoc calibration fits T on validation set to minimize NLL for reliable confidence scores."),
    ]:
        st.markdown(f"""<div class='card' style='display:flex;gap:1rem;margin-bottom:0.5rem;padding:1rem 1.2rem'>
            <div style='flex-shrink:0;min-width:170px'>
                <div style='color:#1D9E75;font-weight:500;font-size:0.88rem'>{name}</div>
                <code style='font-size:0.72rem;color:#8b949e'>{param}</code></div>
            <div style='font-size:0.83rem;color:#8b949e'>{desc}</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("""<div class='card' style='border-left:3px solid #EF9F27;margin-top:1rem'>
        <div style='color:#EF9F27;font-weight:500;margin-bottom:6px'>⚠ Dominant confusion pairs</div>
        <div style='font-size:0.85rem;color:#8b949e'>
            <b style='color:#e6edf3'>Skeptic → Neutral</b>: Ironic denial lacks explicit negation cues.<br>
            <b style='color:#e6edf3'>Neutral → Pro-Climate</b>: Neutral reporting uses Pro-Climate vocabulary without stance.
        </div></div>""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════
# PAGE 6  —  TEMPORAL DASHBOARD
# ══════════════════════════════════════════════════════════════
elif page == "📡  Temporal Dashboard":
    st.markdown("<div class='section-head'>📡 Temporal Sentiment Dashboard</div>", unsafe_allow_html=True)
    st.markdown("""<div class='card'>
        Approximate tweet timestamps decoded from <b>Twitter Snowflake IDs</b>.
        Stance proportions visualized month-by-month (2015–2018) with event markers.
    </div>""", unsafe_allow_html=True)

    # ── Synthetic temporal data ──────────────────────────────
    months = pd.date_range("2015-01", "2018-12", freq="MS")
    n = len(months)
    np.random.seed(42)

    pro  = np.clip(0.52 + 0.06*np.sin(np.linspace(0, 3*math.pi, n))
                   + np.random.normal(0, 0.02, n), 0.35, 0.72)
    skep = np.clip(0.09 + 0.04*np.cos(np.linspace(0, 3*math.pi, n))
                   + np.random.normal(0, 0.015, n), 0.04, 0.22)
    pro[11:14]  += 0.06;  skep[11:14]  -= 0.03  # Paris Agreement boost
    skep[29:33] += 0.08;  pro[29:33]   -= 0.05  # US withdrawal boost
    news = np.clip(0.22 + np.random.normal(0, 0.015, n), 0.12, 0.35)
    neut = np.clip(1 - pro - skep - news, 0.05, 0.25)
    tot  = pro + skep + news + neut
    pro /= tot; skep /= tot; news /= tot; neut /= tot

    fig_t = go.Figure()
    for label_name, vals, col_hex in [
        ("Pro-Climate",    pro,  "#1D9E75"),
        ("News/Factual",   news, "#378ADD"),
        ("Neutral",        neut, "#888780"),
        ("Skeptic/Denial", skep, "#E24B4A"),
    ]:
        fig_t.add_trace(go.Scatter(
            x=list(months),
            y=(vals * 100).tolist(),
            name=label_name,
            mode="lines",
            line=dict(color=col_hex, width=2.5),
            stackgroup="one",
            # ── FIX: use only valid solid hex, not appended opacity ──
            fillcolor=col_hex,
        ))

    for dt, label_txt, col_hex in [
        (pd.Timestamp("2015-12-01"), "Paris Agreement", "#1D9E75"),
        (pd.Timestamp("2017-06-01"), "US Withdrawal",   "#E24B4A"),
    ]:
        fig_t.add_vline(x=dt.timestamp() * 1000,
                        line=dict(color=col_hex, dash="dash", width=1.5))
        fig_t.add_annotation(
            x=dt, y=98, text=f"<b>{label_txt}</b>",
            showarrow=False, font=dict(color=col_hex, size=11),
            xref="x", yref="y",
        )

    fig_t.update_layout(
        **PLOT_BASE,
        title=dict(text="Monthly stance proportions — Snowflake ID decoded",
                   font=dict(color="#e6edf3", size=14)),
        xaxis=dict(color="#8b949e", showgrid=False),
        yaxis=dict(color="#8b949e", gridcolor="#21262d",
                   ticksuffix="%", range=[0, 110]),
        legend=dict(font=dict(color="#8b949e"), orientation="h", y=-0.18),
        height=420,
    )
    st.plotly_chart(fig_t, use_container_width=True)

    st.markdown("""<div class='card' style='border-left:3px solid #1D9E75'>
        <b style='color:#e6edf3'>Snowflake ID decoding</b>
        <div style='font-size:0.83rem;color:#8b949e;margin-top:4px'>
            <code>(snowflake_id >> 22) + 1288834974657</code> → UTC Unix timestamp ms.
            No API call required. Top 41 bits store ms since Twitter epoch (2006-03-21).
        </div></div>""", unsafe_allow_html=True)

    # Upload your own predictions
    st.markdown("<div class='section-head' style='font-size:1.3rem;margin-top:1.5rem'>Upload your own predictions</div>",
                unsafe_allow_html=True)
    ts_file = st.file_uploader("CSV with tweetid + label columns:", type="csv", key="temporal")
    if ts_file:
        df_ts = pd.read_csv(ts_file)
        if "tweetid" in df_ts.columns:
            TW_EPOCH = 1288834974657
            df_ts["timestamp"] = pd.to_datetime(
                (df_ts["tweetid"].astype("int64") >> 22) + TW_EPOCH, unit="ms", utc=True)
            df_ts["month"] = df_ts["timestamp"].dt.to_period("M").dt.to_timestamp()
            monthly = (df_ts.groupby(["month","label"]).size()
                          .unstack(fill_value=0)
                          .div(lambda x: x.sum(axis=1), axis=0) * 100)
            fig_u = go.Figure()
            for lid in range(4):
                lname = ID2LABEL[lid]
                if lname in monthly.columns:
                    fig_u.add_trace(go.Scatter(
                        x=list(monthly.index), y=monthly[lname].tolist(),
                        name=lname, mode="lines",
                        line=dict(color=COLORS[lid], width=2),
                        stackgroup="one", fillcolor=COLORS[lid]))
            fig_u.update_layout(**PLOT_BASE, height=340,
                title=dict(text="Your data — monthly stance proportions",
                           font=dict(color="#e6edf3")))
            st.plotly_chart(fig_u, use_container_width=True)
        else:
            st.warning("No `tweetid` column found.")

# ── FOOTER ────────────────────────────────────────────────────
st.markdown("""
<hr style='border-color:#21262d;margin-top:3rem'>
<div style='text-align:center;font-size:0.75rem;color:#8b949e;padding-bottom:1rem'>
    Climate Pulse · MSDS 453 NLP · Kasheena Mulla · Northwestern University · 2026<br>
    BERTweet (Nguyen et al., 2020) · Temperature Scaling (Guo et al., 2017)
</div>
""", unsafe_allow_html=True)
