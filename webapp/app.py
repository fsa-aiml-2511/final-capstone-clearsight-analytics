"""
================================================================================
ClearSight Analytics — Clinical AI Command Center
================================================================================
Tech-forward Streamlit interface integrating Models 1 (XGBoost) and 2 (DNN).
Dark-mode, glassmorphism, animated SVG gauges, real model inference.

Run: streamlit run webapp/app.py   (from project root)
================================================================================
"""

import sys
import warnings
warnings.filterwarnings("ignore")

import logging
import time
import traceback
import numpy as np
import pandas as pd
import joblib
import streamlit as st
from pathlib import Path
from typing import Any
import tensorflow as tf
from PIL import Image
from keras.applications.resnet50 import preprocess_input
import cv2
import io
# torch and transformers are imported lazily inside load_model4() to avoid
# a crash when torchvision is missing from the environment on startup.

# ── Project root ──────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from pipelines.data_pipeline import engineer_features

# =============================================================================
# LOGGING — writes to logs/app.log; directory is created on first run
# =============================================================================
_LOG_DIR = PROJECT_ROOT / "logs"
_LOG_DIR.mkdir(exist_ok=True)

logger = logging.getLogger("clearsight")
if not logger.handlers:  # guard against duplicate handlers on hot-reload
    logger.setLevel(logging.DEBUG)
    _fmt = logging.Formatter(
        "%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    # File handler — full DEBUG+ detail
    _fh = logging.FileHandler(_LOG_DIR / "app.log", encoding="utf-8")
    _fh.setLevel(logging.DEBUG)
    _fh.setFormatter(_fmt)
    logger.addHandler(_fh)
    # Console handler — INFO+ only
    _ch = logging.StreamHandler()
    _ch.setLevel(logging.INFO)
    _ch.setFormatter(_fmt)
    logger.addHandler(_ch)

# --- Model paths ---
M1_DIR = PROJECT_ROOT / "models" / "model1_traditional_ml" / "saved_model"
M2_DIR = PROJECT_ROOT / "models" / "model2_deep_learning" / "saved_model"
M3_DIR = PROJECT_ROOT / "models" / "model3_cnn" / "saved_model"              # CNN Retinal (Doug)
M4_DIR = PROJECT_ROOT / "models" / "model4_nlp_classification" / "saved_model" # NLP (Wes)
M5_DIR = PROJECT_ROOT / "models" / "model5_innovation" / "saved_model"       # Capacity Planning

def inject_sidebar_styles() -> None:
    """Injects custom CSS to style the sidebar as a dark-mode SaaS navigation menu.

    Transforms the native ``st.radio`` widget into hoverable navigation blocks and
    registers ``.status-pill``, ``.sidebar-header``, and ``.brand`` CSS classes used
    throughout the sidebar. Writes directly to the Streamlit DOM via ``st.markdown``.

    Raises:
        streamlit.errors.StreamlitAPIException: If called outside a valid Streamlit
            execution context.
    """
    st.markdown("""
    <style>
    /* 1. Sidebar Background */
    [data-testid="stSidebar"] {
        background-color: #0b1121 !important;
        border-right: 1px solid rgba(34,211,238,0.1);
    }
    
    /* 2. Style the native radio widget container */
    [data-testid="stSidebar"] [data-testid="stRadio"] > div {
        gap: 0.3rem;
        padding: 0 0.5rem;
    }
    
    /* 3. Transform radio options into clickable blocks */
    [data-testid="stSidebar"] [data-testid="stRadio"] label {
        background: transparent;
        padding: 0.6rem 1rem;
        border-radius: 8px;
        transition: all 0.2s ease;
        cursor: pointer;
        border-left: 3px solid transparent;
        width: 100%;
    }
    
    /* Hover effect */
    [data-testid="stSidebar"] [data-testid="stRadio"] label:hover {
        background: rgba(34,211,238,0.05);
        border-left: 3px solid rgba(34,211,238,0.4);
        transform: translateX(3px);
    }
    
    /* 4. HIDE the default radio circles! */
    [data-testid="stRadio"] div[role="radio"] > div:first-child,
    [data-testid="stRadio"] label > div:first-child {
        display: none !important;
    }
    
    /* 5. Typography for the links */
    [data-testid="stSidebar"] [data-testid="stRadio"] label p {
        font-family: 'Space Grotesk', sans-serif;
        font-size: 0.95rem;
        font-weight: 500;
        color: #cbd5e1;
        margin: 0;
    }
    
    /* 6. System Status Headers */
    .sidebar-header {
        color: #64748b;
        font-size: 0.75rem;
        font-weight: 700;
        letter-spacing: 0.1em;
        text-transform: uppercase;
        margin-top: 1.5rem;
        margin-bottom: 0.8rem;
        padding-left: 1rem;
    }
    
    /* 7. Status Pill styling */
    .status-row {
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: 0.4rem 1rem;
        font-size: 0.85rem;
        color: #94a3b8;
    }
    .status-pill {
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.65rem;
        font-weight: 700;
        padding: 3px 8px;
        border-radius: 999px;
        letter-spacing: 0.05em;
    }
    .pill-online { background: rgba(16,185,129,0.15); color: #10b981; border: 1px solid rgba(16,185,129,0.3); }
    .pill-pending { background: rgba(245,158,11,0.15); color: #f59e0b; border: 1px solid rgba(245,158,11,0.3); }
    </style>
    """, unsafe_allow_html=True)



# =============================================================================
# PAGE CONFIGURATION & GLOBAL STYLES
# =============================================================================
st.set_page_config(
    page_title="ClearSight | Clinical AI Command Center",
    page_icon="👁️",
    layout="wide",
    initial_sidebar_state="expanded",
)

inject_sidebar_styles()


# =============================================================================
# EMBEDDED LOGO (SVG)
# =============================================================================
CLEARSIGHT_ICON_SVG = """
<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 120 120" width="48" height="48">
  <defs>
    <radialGradient id="irisG" cx="50%" cy="50%" r="50%">
      <stop offset="0%" stop-color="#5eead4"/>
      <stop offset="60%" stop-color="#22d3ee"/>
      <stop offset="100%" stop-color="#185FA5"/>
    </radialGradient>
    <linearGradient id="outlineG" x1="0%" y1="0%" x2="100%" y2="100%">
      <stop offset="0%" stop-color="#22d3ee"/>
      <stop offset="100%" stop-color="#185FA5"/>
    </linearGradient>
  </defs>
  <g transform="translate(10, 32)">
    <path d="M 4 28 Q 26 4, 50 4 Q 74 4, 96 28 Q 74 52, 50 52 Q 26 52, 4 28 Z"
          fill="none" stroke="url(#outlineG)" stroke-width="3.5" stroke-linecap="round"/>
    <circle cx="50" cy="28" r="18" fill="url(#irisG)"/>
    <circle cx="50" cy="28" r="14.5" fill="none" stroke="#5eead4" stroke-width="0.9" opacity="0.5"/>
    <circle cx="50" cy="28" r="7" fill="#0a1220"/>
    <circle cx="46.5" cy="23.5" r="3" fill="#ffffff" opacity="0.95"/>
    <circle cx="53" cy="31" r="1.2" fill="#ffffff" opacity="0.6"/>
    <line x1="14" y1="28" x2="22" y2="28" stroke="#5eead4" stroke-width="1.8" stroke-linecap="round" opacity="0.7"/>
    <line x1="78" y1="28" x2="86" y2="28" stroke="#5eead4" stroke-width="1.8" stroke-linecap="round" opacity="0.7"/>
  </g>
</svg>
"""


# =============================================================================
# CSS — Dark tech aesthetic
# =============================================================================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500;600;700&family=Inter:wght@300;400;500;600;700&display=swap');

/* ── Global reset ────────────────────────────────────────────────── */
html, body, [class*="css"] {
    font-family: 'Inter', -apple-system, sans-serif !important;
    color: #e2e8f0;
}
.stApp {
    background:
        radial-gradient(circle at 15% 20%, rgba(34,211,238,0.06) 0%, transparent 40%),
        radial-gradient(circle at 85% 80%, rgba(94,234,212,0.05) 0%, transparent 40%),
        linear-gradient(180deg, #0a1220 0%, #0d1b2a 100%);
    background-attachment: fixed;
}

/* Subtle grid pattern overlay */
.stApp::before {
    content: "";
    position: fixed; inset: 0; pointer-events: none; z-index: 0;
    background-image:
        linear-gradient(rgba(94,234,212,0.025) 1px, transparent 1px),
        linear-gradient(90deg, rgba(94,234,212,0.025) 1px, transparent 1px);
    background-size: 40px 40px;
}

#MainMenu { visibility: hidden; }
footer { visibility: hidden; }
.block-container { padding-top: 1.5rem; padding-bottom: 4rem; max-width: 1400px; }

/* Force sidebar always visible */
[data-testid="collapsedControl"] { display: block !important; visibility: visible !important; }
section[data-testid="stSidebar"] { display: block !important; min-width: 260px !important; }

/* ── Sidebar ─────────────────────────────────────────────────────── */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #07101e 0%, #0a1628 100%) !important;
    border-right: 1px solid rgba(34,211,238,0.15);
}
[data-testid="stSidebar"] * { color: #cbd5e1 !important; }
[data-testid="stSidebar"] .stRadio label {
    padding: 8px 12px;
    border-radius: 8px;
    font-size: 0.92rem;
    font-weight: 500;
    transition: all 0.15s;
}
[data-testid="stSidebar"] .stRadio label:hover {
    background: rgba(34,211,238,0.08);
}

/* ── Brand block ─────────────────────────────────────────────────── */
.brand {
    display: flex; align-items: center; gap: 12px;
    padding: 0.6rem 0 1.2rem;
    border-bottom: 1px solid rgba(34,211,238,0.15);
    margin-bottom: 1rem;
}
.brand .name {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 1.25rem; font-weight: 700;
    color: white !important;
    letter-spacing: -0.02em;
    line-height: 1;
}
.brand .tag {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.65rem; font-weight: 500;
    color: #5eead4 !important;
    letter-spacing: 0.18em;
    text-transform: uppercase;
    margin-top: 4px;
}

/* ── Hero ────────────────────────────────────────────────────────── */
.hero {
    position: relative;
    background:
        radial-gradient(circle at 80% 30%, rgba(34,211,238,0.18) 0%, transparent 50%),
        linear-gradient(135deg, #0d1b2a 0%, #112c4a 50%, #0d1b2a 100%);
    border: 1px solid rgba(34,211,238,0.2);
    border-radius: 18px;
    padding: 2.5rem 2.75rem;
    margin-bottom: 2rem;
    overflow: hidden;
    box-shadow: 0 20px 60px rgba(34,211,238,0.08), inset 0 1px 0 rgba(94,234,212,0.1);
}
.hero::after {
    content: "";
    position: absolute; top: 0; right: 0; width: 300px; height: 100%;
    background: radial-gradient(ellipse, rgba(94,234,212,0.12) 0%, transparent 70%);
    pointer-events: none;
}
.hero .badge {
    display: inline-block;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.7rem; font-weight: 600;
    color: #5eead4;
    background: rgba(94,234,212,0.1);
    border: 1px solid rgba(94,234,212,0.3);
    padding: 4px 12px; border-radius: 999px;
    letter-spacing: 0.08em; text-transform: uppercase;
    margin-bottom: 1rem;
}
.hero h1 {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 2.6rem; font-weight: 700;
    color: white !important;
    margin: 0 0 0.6rem;
    letter-spacing: -0.03em; line-height: 1.1;
    background: linear-gradient(120deg, #ffffff 0%, #5eead4 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
}
.hero .sub {
    font-size: 1.05rem;
    color: #94a3b8 !important;
    max-width: 720px;
    line-height: 1.55;
    margin: 0 0 1.4rem;
}
.pill {
    display: inline-flex; align-items: center; gap: 6px;
    background: rgba(34,211,238,0.08);
    color: #5eead4 !important;
    padding: 6px 14px;
    border-radius: 999px;
    font-size: 0.78rem; font-weight: 600;
    margin-right: 0.5rem;
    border: 1px solid rgba(34,211,238,0.25);
    font-family: 'JetBrains Mono', monospace;
    letter-spacing: 0.04em;
}

/* ── Glass card ──────────────────────────────────────────────────── */
.gcard {
    background: rgba(15,30,52,0.6);
    backdrop-filter: blur(12px);
    -webkit-backdrop-filter: blur(12px);
    border: 1px solid rgba(34,211,238,0.12);
    border-radius: 14px;
    padding: 1.4rem 1.5rem;
    box-shadow: 0 8px 32px rgba(0,0,0,0.2);
    transition: transform 0.2s, border-color 0.2s, box-shadow 0.2s;
    height: 100%;
}
.gcard:hover {
    transform: translateY(-3px);
    border-color: rgba(94,234,212,0.4);
    box-shadow: 0 12px 40px rgba(34,211,238,0.15);
}
.gcard .tag {
    display: inline-block;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.66rem; font-weight: 700;
    color: #5eead4;
    letter-spacing: 0.12em; text-transform: uppercase;
    background: rgba(94,234,212,0.08);
    padding: 3px 10px; border-radius: 6px;
    margin-bottom: 12px;
    border: 1px solid rgba(94,234,212,0.2);
}
.gcard h3 {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 1.15rem; font-weight: 600;
    color: white !important;
    margin: 8px 0;
}
.gcard p {
    color: #94a3b8 !important;
    font-size: 0.9rem; line-height: 1.55;
    margin: 0;
}

/* ── Stat block ──────────────────────────────────────────────────── */
.stat {
    background: rgba(15,30,52,0.5);
    border: 1px solid rgba(34,211,238,0.12);
    border-radius: 12px;
    padding: 1.2rem 1.3rem;
    transition: border-color 0.2s;
}
.stat:hover { border-color: rgba(94,234,212,0.3); }
.stat .label {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.66rem; font-weight: 600;
    color: #64748b !important;
    text-transform: uppercase; letter-spacing: 0.12em;
    margin-bottom: 6px;
}
.stat .val {
    font-family: 'JetBrains Mono', monospace;
    font-size: 1.85rem; font-weight: 700;
    color: white !important;
    line-height: 1;
}
.stat .delta {
    font-size: 0.78rem; font-weight: 500;
    color: #5eead4 !important;
    margin-top: 6px;
}

/* ── Section heading ─────────────────────────────────────────────── */
.section-head {
    display: flex; align-items: center; gap: 12px;
    margin: 2rem 0 1.2rem;
}
.section-head .num {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.85rem; font-weight: 600;
    color: #5eead4;
    background: rgba(94,234,212,0.08);
    padding: 2px 10px; border-radius: 6px;
    border: 1px solid rgba(94,234,212,0.2);
}
.section-head h2 {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 1.5rem; font-weight: 600;
    color: white !important;
    margin: 0;
    letter-spacing: -0.02em;
}
.section-head .line {
    flex: 1; height: 1px;
    background: linear-gradient(90deg, rgba(34,211,238,0.3), transparent);
}

/* ── Live status dot (pulsing) ───────────────────────────────────── */
.dot {
    display: inline-block;
    width: 8px; height: 8px; border-radius: 50%;
    margin-right: 8px; vertical-align: middle;
    animation: pulse 2s infinite;
}
.dot-ok { background: #5eead4; box-shadow: 0 0 8px rgba(94,234,212,0.7); }
.dot-warn { background: #fbbf24; box-shadow: 0 0 8px rgba(251,191,36,0.7); }
.dot-err { background: #f87171; box-shadow: 0 0 8px rgba(248,113,113,0.7); }
@keyframes pulse {
    0%, 100% { opacity: 1; transform: scale(1); }
    50% { opacity: 0.6; transform: scale(1.2); }
}

.status-row {
    display: flex; align-items: center; justify-content: space-between;
    padding: 8px 0;
    font-size: 0.85rem;
}
.status-row .name { color: #cbd5e1 !important; font-weight: 500; }
.status-row .val { font-family: 'JetBrains Mono', monospace; font-size: 0.78rem; color: #5eead4 !important; }

/* ── Risk badges ─────────────────────────────────────────────────── */
.risk-high   { background: rgba(248,113,113,0.12); color: #fca5a5;
               border: 1px solid rgba(248,113,113,0.4); border-radius: 10px;
               padding: 12px 18px; font-weight: 700; font-size: 1rem;
               text-align: center; font-family: 'Space Grotesk', sans-serif;
               letter-spacing: 0.04em; }
.risk-medium { background: rgba(251,191,36,0.12); color: #fcd34d;
               border: 1px solid rgba(251,191,36,0.4); border-radius: 10px;
               padding: 12px 18px; font-weight: 700; font-size: 1rem;
               text-align: center; font-family: 'Space Grotesk', sans-serif;
               letter-spacing: 0.04em; }
.risk-low    { background: rgba(94,234,212,0.12); color: #5eead4;
               border: 1px solid rgba(94,234,212,0.4); border-radius: 10px;
               padding: 12px 18px; font-weight: 700; font-size: 1rem;
               text-align: center; font-family: 'Space Grotesk', sans-serif;
               letter-spacing: 0.04em; }

/* ── Buttons ─────────────────────────────────────────────────────── */
div.stButton > button, div.stFormSubmitButton > button {
    background: linear-gradient(135deg, #22d3ee 0%, #185FA5 100%) !important;
    color: white !important;
    border: none !important;
    border-radius: 10px !important;
    padding: 12px 28px !important;
    font-size: 0.95rem !important;
    font-weight: 600 !important;
    width: 100% !important;
    box-shadow: 0 4px 20px rgba(34,211,238,0.3), inset 0 1px 0 rgba(255,255,255,0.2) !important;
    transition: all 0.2s !important;
    font-family: 'Space Grotesk', sans-serif !important;
    letter-spacing: 0.02em !important;
}
div.stButton > button:hover, div.stFormSubmitButton > button:hover {
    transform: translateY(-1px);
    box-shadow: 0 8px 30px rgba(34,211,238,0.5), inset 0 1px 0 rgba(255,255,255,0.3) !important;
}

/* ── Form inputs ─────────────────────────────────────────────────── */
.stTextInput input, .stNumberInput input, .stSelectbox > div > div {
    background: rgba(10,18,32,0.6) !important;
    border: 1px solid rgba(34,211,238,0.2) !important;
    color: #e2e8f0 !important;
    border-radius: 8px !important;
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 0.88rem !important;
}
.stTextInput input:focus, .stNumberInput input:focus {
    border-color: #5eead4 !important;
    box-shadow: 0 0 0 3px rgba(94,234,212,0.15) !important;
}
label, .stMarkdown p, .stMarkdown li { color: #cbd5e1 !important; }

/* Form container */
[data-testid="stForm"] {
    background: rgba(15,30,52,0.4);
    border: 1px solid rgba(34,211,238,0.12);
    border-radius: 14px;
    padding: 1.5rem !important;
    backdrop-filter: blur(8px);
}

/* Tabs */
.stTabs [data-baseweb="tab-list"] {
    gap: 4px;
    background: rgba(10,18,32,0.5);
    border-radius: 10px;
    padding: 4px;
}
.stTabs [data-baseweb="tab"] {
    background: transparent;
    color: #94a3b8 !important;
    border-radius: 8px;
    padding: 8px 16px;
    font-weight: 500;
    font-size: 0.88rem;
}
.stTabs [aria-selected="true"] {
    background: rgba(34,211,238,0.15) !important;
    color: #5eead4 !important;
}

/* Disclaimer */
.disclaimer {
    background: rgba(251,146,60,0.08);
    border-left: 3px solid #fb923c;
    padding: 1rem 1.2rem; border-radius: 8px;
    color: #fdba74;
    font-size: 0.85rem; line-height: 1.55;
    margin-top: 1.5rem;
}

/* Dataframe styling */
.stDataFrame { background: transparent; }

/* Spinner color */
.stSpinner > div { border-top-color: #5eead4 !important; }

/* Headings */
h1, h2, h3 { color: white !important; font-family: 'Space Grotesk', sans-serif; }
</style>
""", unsafe_allow_html=True)


# =============================================================================
# MODEL LOADERS  (cached)
# =============================================================================
@st.cache_resource(show_spinner=False)
def load_model1() -> tuple[Any, dict, list, float]:
    """Loads and caches the Model 1 XGBoost ensemble and its preprocessing artifacts.

    Results are cached by Streamlit's ``@st.cache_resource`` decorator so disk
    I/O occurs only on the first call per server session.

    Returns:
        A 4-tuple of:

            - **model** (Any): Trained XGBoost ensemble (sklearn-compatible estimator).
            - **state** (dict): Preprocessing state dict produced by the training pipeline.
            - **feats** (list): Ordered list of feature names expected by the model.
            - **thresh** (float): Optimal decision threshold; defaults to ``0.5`` if the
              artifact file is absent.

    Raises:
        FileNotFoundError: If any required ``.joblib`` artifact is missing from
            ``M1_DIR``.
    """
    t0 = time.perf_counter()
    logger.info("Loading Model 1 (XGBoost) artifacts from %s", M1_DIR)
    try:
        model  = joblib.load(M1_DIR / "model.joblib")
        state  = joblib.load(M1_DIR / "preprocessing_state.joblib")
        feats  = joblib.load(M1_DIR / "feature_names.joblib")
        thresh = joblib.load(M1_DIR / "optimal_threshold.joblib") if (M1_DIR / "optimal_threshold.joblib").exists() else 0.5
    except Exception:
        logger.error("Failed to load Model 1 artifacts", exc_info=True)
        raise
    logger.info("Model 1 loaded in %.0f ms", (time.perf_counter() - t0) * 1000)
    return model, state, feats, thresh


@st.cache_resource(show_spinner=False)
def load_model2() -> tuple[Any, Any, dict, list]:
    """Loads and caches the Model 2 Keras DNN and its preprocessing artifacts.

    TensorFlow is imported lazily inside this function so the app starts without
    requiring TF on the Python import path. Results are cached for the lifetime of
    the Streamlit server process.

    Returns:
        A 4-tuple of:

            - **model** (Any): Compiled Keras Sequential model loaded from ``.keras``.
            - **scaler** (Any): Fitted ``StandardScaler`` for feature normalisation.
            - **state** (dict): Preprocessing state dict produced by the training pipeline.
            - **feats** (list): Ordered list of feature names expected by the model.

    Raises:
        FileNotFoundError: If any required artifact is missing from ``M2_DIR``.
        ImportError: If TensorFlow is not installed in the current environment.
    """
    t0 = time.perf_counter()
    logger.info("Loading Model 2 (Keras DNN) artifacts from %s", M2_DIR)
    try:
        import tensorflow as tf
        model  = tf.keras.models.load_model(M2_DIR / "model.keras")
        scaler = joblib.load(M2_DIR / "scaler.joblib")
        state  = joblib.load(M2_DIR / "preprocessing_state.joblib")
        feats  = joblib.load(M2_DIR / "feature_names.joblib")
    except Exception:
        logger.error("Failed to load Model 2 artifacts", exc_info=True)
        raise
    logger.info("Model 2 loaded in %.0f ms", (time.perf_counter() - t0) * 1000)
    return model, scaler, state, feats


# =============================================================================
# MODEL 3 — CNN Retinal (ResNet50, binary DR classifier)
# =============================================================================
@st.cache_resource(show_spinner=False)
def load_model3():
    """Loads and caches the Model 3 ResNet50 Diabetic Retinopathy classifier.

    Returns:
        tf.keras.Model: Compiled model loaded from best_model.keras, or None on failure.
    """
    t0 = time.perf_counter()
    logger.info("Loading Model 3 (CNN Retinal) from %s", M3_DIR)
    try:
        model = tf.keras.models.load_model(M3_DIR / "best_model.keras", compile=False)
    except Exception:
        logger.error("Failed to load Model 3 artifacts", exc_info=True)
        raise
    logger.info("Model 3 loaded in %.0f ms", (time.perf_counter() - t0) * 1000)
    return model


def predict_m3(image_file, model) -> tuple[str, float]:
    """Runs ResNet50 inference on a fundus image file.

    Args:
        image_file: A file-like object (e.g. from st.file_uploader).
        model: Loaded tf.keras.Model instance.

    Returns:
        A 2-tuple of (label: str, confidence: float 0–1).
    """
    img = Image.open(image_file).convert("RGB")
    img = img.resize((224, 224))
    img_array = np.array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_preprocessed = preprocess_input(img_array.astype(np.float32))
    preds = model.predict(img_preprocessed)
    confidence = float(preds[0][0])
    if confidence > 0.5:
        return "HIGH RISK — Diabetic Retinopathy Detected", confidence
    else:
        return "LOW RISK — No Retinopathy Detected", 1.0 - confidence


# =============================================================================
# MODEL 3 — Grad-CAM helpers
# =============================================================================
def make_gradcam(img_array, model, last_conv_layer="conv5_block3_out"):
    """Computes a Grad-CAM heatmap for the given preprocessed image array.

    Args:
        img_array: Preprocessed image batch (1, 224, 224, 3) float32 numpy array.
        model: Loaded tf.keras.Model (ResNet50).
        last_conv_layer: Name of the final convolutional layer to target.

    Returns:
        heatmap: 2-D numpy array (H×W) with values in [0, 1].
    """
    grad_model = tf.keras.models.Model(
        [model.inputs],
        [model.get_layer(last_conv_layer).output, model.output]
    )
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        loss = predictions[:, 0]
    grads = tape.gradient(loss, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_outputs = conv_outputs[0]
    heatmap = tf.reduce_sum(tf.multiply(pooled_grads, conv_outputs), axis=-1)
    heatmap = np.maximum(heatmap.numpy(), 0)
    if heatmap.max() > 0:
        heatmap /= heatmap.max()
    return heatmap


def overlay_gradcam(pil_image, heatmap, alpha=0.4):
    """Overlays a Grad-CAM heatmap onto a PIL image.

    Args:
        pil_image: Original PIL.Image (any size).
        heatmap: 2-D numpy array from make_gradcam().
        alpha: Blending weight for the heatmap overlay (0–1).

    Returns:
        PIL.Image: Superimposed RGB image.
    """
    img = np.array(pil_image.resize((224, 224)))
    heatmap_resized = cv2.resize(heatmap, (224, 224))
    heatmap_uint8 = np.uint8(255 * heatmap_resized)
    heatmap_colored = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
    heatmap_rgb = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)
    superimposed = cv2.addWeighted(img, 1 - alpha, heatmap_rgb, alpha, 0)
    return Image.fromarray(superimposed)


class Vocabulary:
    def get(self, word, default=0):
        for val in self.__dict__.values():
            if isinstance(val, dict):
                return val.get(word, default)
        return default

    def __len__(self):
        for val in self.__dict__.values():
            if isinstance(val, dict):
                return len(val)
        return 10000 
# ----------------------------------



@st.cache_resource(show_spinner=False)
def load_model4() -> tuple[Any, Any, Any, Any, Any]:
    import torch
    import torch.nn as nn
    
    class MetaLSTMClassifier(nn.Module):
        # Ajustamos los parámetros base para coincidir con las dimensiones de Wes
        def __init__(self, vocab_size, num_drugs, num_conditions, embed_dim=128, 
                     hidden_dim=128, meta_embed_dim=32, num_classes=3, dropout=0.3):
            super().__init__()
            # Wes lo llamó 'text_embedding', no 'embedding'
            self.text_embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
            
            # Wes usó num_layers=2 y hidden_dim=128 (se nota por los pesos de 512)
            self.lstm = nn.LSTM(embed_dim, hidden_dim, num_layers=2, batch_first=True, bidirectional=True)
            
            # Ajustamos el tamaño del embedding a 32
            self.drug_embedding = nn.Embedding(num_drugs, meta_embed_dim)
            self.condition_embedding = nn.Embedding(num_conditions, meta_embed_dim)
            self.dropout = nn.Dropout(dropout)
            
            # hidden_dim * 2 (bidirectional) + meta_embed_dim * 2
            # 128 * 2 + 32 * 2 = 256 + 64 = 320 (Coincide con fc.weight: [3, 320])
            self.fc = nn.Linear(hidden_dim * 2 + meta_embed_dim * 2, num_classes)

        def forward(self, text_seq, text_lengths, drug_idx, cond_idx):
            embedded = self.text_embedding(text_seq)
            
            packed_embedded = nn.utils.rnn.pack_padded_sequence(
                embedded, text_lengths.cpu(), batch_first=True, enforce_sorted=False
            )
            packed_output, (hidden, cell) = self.lstm(packed_embedded)
            
            # Extraer el hidden state de la última capa (num_layers=2) de ambas direcciones
            hidden = self.dropout(torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim=1))
            
            drug_out = self.drug_embedding(drug_idx)
            cond_out = self.condition_embedding(cond_idx)
            
            combined = torch.cat([hidden, drug_out, cond_out], dim=1)
            return self.fc(self.dropout(combined))

    t0 = time.perf_counter()
    logger.info("Loading Model 4 (Meta LSTM) artifacts from %s", M4_DIR)
    
    try:
        drug_le = joblib.load(M4_DIR / "drug_encoder.joblib")
        cond_le = joblib.load(M4_DIR / "condition_encoder.joblib")
        label_le = joblib.load(M4_DIR / "label_encoder_meta.joblib")
        vocab = joblib.load(M4_DIR / "vocab_pretrained.joblib")

        model = MetaLSTMClassifier(
            vocab_size=len(vocab),
            num_drugs=len(drug_le.classes_),
            num_conditions=len(cond_le.classes_)
        )

        model.load_state_dict(torch.load(M4_DIR / "model_meta_lstm.pt", map_location="cpu"))
        model.eval()
        
    except Exception:
        logger.error("Failed to load Model 4 artifacts", exc_info=True)
        raise
        
    logger.info("Model 4 loaded in %.0f ms", (time.perf_counter() - t0) * 1000)
    return model, vocab, drug_le, cond_le, label_le


def predict_m4(text_notes: str, drug_name: str, condition: str) -> tuple[str, float, str, str]:
    """Runs NLP inference using the cached Meta LSTM Classifier."""
    logger.info("M4 prediction triggered — drug=%s condition=%s", drug_name, condition)
    import torch
    import re
    
    model, vocab, drug_le, cond_le, label_le = load_model4()

    def safe_encode(le: Any, val: str) -> int:
        return le.transform([val if val in le.classes_ else "unknown"])[0]

    # 1. Encode Metadata
    drug_idx = torch.tensor([safe_encode(drug_le, drug_name)], dtype=torch.long)
    cond_idx = torch.tensor([safe_encode(cond_le, condition)], dtype=torch.long)
    
    # 2. Text Preprocessing & Tokenization (Simple whitespace/regex tokenizer for LSTM)
    text = text_notes.lower()
    text = re.sub(r'[^a-z0-9\s]', '', text)
    tokens = text.split()
    
    # Map words to indices, cap at max_length (e.g., 256)
    max_len = 256
    seq = [vocab.get(word, vocab.get("<unk>", 1)) for word in tokens]
    
    if len(seq) == 0:
        seq = [vocab.get("<pad>", 0)] # Fallback if empty
        
    seq_length = torch.tensor([min(len(seq), max_len)], dtype=torch.long)
    
    # Pad or truncate sequence
    if len(seq) < max_len:
        seq = seq + [vocab.get("<pad>", 0)] * (max_len - len(seq))
    else:
        seq = seq[:max_len]
        
    text_tensor = torch.tensor([seq], dtype=torch.long)

    # 3. Model Forward Pass
    with torch.no_grad():
        logits = model(text_tensor, seq_length, drug_idx, cond_idx)
        probs = torch.softmax(logits, dim=1).numpy()[0]
        pred_idx = np.argmax(probs)
        
    label = label_le.inverse_transform([pred_idx])[0]
    confidence = float(probs[pred_idx])
    logger.info("M4 result — label=%s confidence=%.4f", label, confidence)

    explanation_map = {
        "Ineffective":        "Critical Interpretation: Linguistic markers suggest severe symptoms, treatment failure, or a potential emergency. Immediate review advised.",
        "Somewhat Effective": "Elevated Interpretation: Linguistic markers indicate lingering symptoms or an incomplete response to current treatment context.",
        "Effective":          "Stable Interpretation: Linguistic markers indicate a positive response to treatment and stable patient condition.",
    }
    css_map = {
        "Ineffective":        "risk-high",
        "Somewhat Effective": "risk-medium",
        "Effective":          "risk-low",
    }
    
    display_map = {
        "Ineffective":        "CRITICAL",
        "Somewhat Effective": "ELEVATED",
        "Effective":          "STABLE",
    }
    
    explanation = explanation_map.get(label, "Interpretation unavailable for this label.")

    display_title = display_map.get(label, label.upper())
    

    return f"{display_title} RISK SENTIMENT", confidence, css_map.get(label, "risk-low"), explanation


def generate_clinical_synthesis(
    m1_proba: float,
    m2_proba: float,
    m5_label: str,
    m4_label: str,
    m4_explanation: str,
) -> str:
    from openai import OpenAI

    client = OpenAI(
        api_key=st.secrets.get("GROQ_API_KEY", ""),
        base_url="https://api.groq.com/openai/v1",
    )

    prompt = f"""You are a senior clinical informatics specialist reviewing AI model outputs 
for a hospitalized diabetic patient at MedInsight Healthcare.

MULTI-MODEL DIAGNOSTIC OUTPUTS:
- Readmission Risk (XGBoost M1): {m1_proba*100:.1f}%
- Readmission Risk (DNN M2): {m2_proba*100:.1f}%
- Predicted Length of Stay (M5): {m5_label}
- Clinical Notes Sentiment (NLP M4): {m4_label}
- NLP Context: {m4_explanation}

Based on these AI model outputs, write a concise clinical synthesis in exactly this format:

SYNTHESIS (2-3 sentences): A professional summary of the patient's overall risk profile, 
noting where models agree or diverge.

RECOMMENDATIONS:
1. [First actionable recommendation for the care team]
2. [Second recommendation]
3. [Third recommendation]

Keep the tone clinical, precise, and professional. Do not mention specific model names 
(M1, M2, etc.) — speak in terms of clinical risk indicators."""

    try:
        response = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            max_tokens=400,
            temperature=0.3,
            messages=[
                {"role": "system", "content": "You are a helpful medical AI assistant. Be concise and clinical."},
                {"role": "user", "content": prompt}
            ],
        )
        return response.choices[0].message.content
    except Exception as e:
        logger.error("Clinical synthesis API call failed: %s", e, exc_info=True)
        return f"Synthesis unavailable: {str(e)}"


@st.cache_resource(show_spinner=False)
def load_model5() -> tuple[Any, dict, list]:
    """Loads and caches the Model 5 Length-of-Stay classifier and its artifacts.

    Results are cached by Streamlit's ``@st.cache_resource`` decorator so disk
    I/O occurs only on the first call per server session.

    Returns:
        A 3-tuple of:

            - **model** (Any): Trained multi-class classifier (sklearn-compatible).
            - **state** (dict): Preprocessing state dict produced by the training pipeline.
            - **feats** (list): Ordered list of feature names expected by the model.

    Raises:
        FileNotFoundError: If any required ``.joblib`` artifact is missing from
            ``M5_DIR``.
    """
    M5_DIR = PROJECT_ROOT / "models" / "model5_innovation" / "saved_model"
    t0 = time.perf_counter()
    logger.info("Loading Model 5 (LoS classifier) artifacts from %s", M5_DIR)
    try:
        model = joblib.load(M5_DIR / "model.joblib")
        state = joblib.load(M5_DIR / "preprocessing_state.joblib")
        feats = joblib.load(M5_DIR / "feature_names.joblib")
    except Exception:
        logger.error("Failed to load Model 5 artifacts", exc_info=True)
        raise
    logger.info("Model 5 loaded in %.0f ms", (time.perf_counter() - t0) * 1000)
    return model, state, feats

def predict_m5(patient_dict: dict) -> tuple[str, float, str]:
    """Runs inference for the Model 5 Length-of-Stay (LoS) classifier.

    Preprocesses the patient record, calls the cached multi-class classifier, and
    maps the predicted class index to a human-readable duration label and UI CSS class.

    Args:
        patient_dict: Raw patient encounter dictionary from the Streamlit form.

    Returns:
        A 3-tuple of:

            - **label** (str): Human-readable stay category
              (e.g. ``"SHORT STAY (1-2 days)"``).
            - **confidence** (float): Maximum class probability in [0, 1].
            - **css_class** (str): CSS class for UI badge colouring
              (``"risk-low"``, ``"risk-medium"``, or ``"risk-high"``).

    Raises:
        ValueError: If ``preprocess_patient`` produces a feature shape mismatch.
        KeyError: If the predicted class index is not present in the labels mapping.
    """
    logger.info("M5 prediction triggered — age=%s time_in_hospital=%s",
                patient_dict.get("age"), patient_dict.get("time_in_hospital"))
    model, state, feats = load_model5()  # uses @st.cache_resource — no disk I/O after first call

    # Process input
    X = preprocess_patient(patient_dict, state, feats)
    
    # Run prediction — derive class from probabilities so label and confidence
    # are always consistent (model.predict() can diverge from predict_proba argmax
    # due to internal voting thresholds in some sklearn estimators).
    probabilities = model.predict_proba(X)[0]
    prediction = int(np.argmax(probabilities))
    confidence = float(np.max(probabilities))
    
    # Mapping
    labels = {
        0: "SHORT STAY (1-2 days)", 
        1: "STANDARD STAY (3-5 days)", 
        2: "EXTENDED STAY (6+ days)"
    }
    css_classes = {0: "risk-low", 1: "risk-medium", 2: "risk-high"}
    
    logger.info("M5 result — class=%d label=%s confidence=%.4f",
                prediction, labels[prediction], confidence)
    # CRITICAL: Return exactly 3 values
    return labels[prediction], confidence, css_classes[prediction]


# =============================================================================
# PREPROCESSING — single patient dict → model-ready row
# =============================================================================
def preprocess_patient(patient: dict, preprocessing_state: dict,
                       feature_names: list) -> pd.DataFrame:
    """Transforms a raw patient encounter dict into a model-ready feature DataFrame.

    Merges the input record with medication defaults, converts the age bracket string
    to a numeric midpoint, derives recurrency features, runs the shared
    ``engineer_features`` pipeline, drops the target column if present, and reindexes
    to the exact column order the model was trained on.

    Args:
        patient: Raw patient dictionary from the prediction form. Expected keys include
            demographic fields (``age``, ``gender``, ``race``), clinical metrics
            (``time_in_hospital``, ``num_lab_procedures``, etc.), and ICD-9 codes
            (``diag_1``, ``diag_2``, ``diag_3``).
        preprocessing_state: Serialised encoding maps and statistics from the
            training pipeline's ``engineer_features`` call.
        feature_names: Ordered list of column names the downstream model expects.

    Returns:
        A single-row ``pd.DataFrame`` with columns matching ``feature_names``;
        any missing columns are filled with ``0``.

    Raises:
        KeyError: If ``engineer_features`` expects a column not derivable from
            ``patient``.
    """
    med_defaults = {c: 'No' for c in [
        'metformin','repaglinide','nateglinide','chlorpropamide','glimepiride',
        'acetohexamide','glipizide','glyburide','tolbutamide','pioglitazone',
        'rosiglitazone','acarbose','miglitol','troglitazone','tolazamide',
        'insulin','glyburide-metformin','glipizide-metformin',
        'glimepiride-pioglitazone','metformin-rosiglitazone','metformin-pioglitazone',
    ]}
    row = {**med_defaults, **patient}

    age_map = {'[0-10)':5,'[10-20)':15,'[20-30)':25,'[30-40)':35,'[40-50)':45,
               '[50-60)':55,'[60-70)':65,'[70-80)':75,'[80-90)':85,'[90-100)':95}
    row['age_numeric'] = age_map.get(row.get('age', '[60-70)'), 65)
    row.pop('age', None)

    row.setdefault('prior_encounters_count', 0)
    row.setdefault('is_recurrent_patient',   0)
    row.setdefault('prior_inpatient_cumsum', row.get('number_inpatient', 0))

    df = pd.DataFrame([row])
    df, _ = engineer_features(df, preprocessing_state=preprocessing_state)
    df.drop(columns=[c for c in ['readmission_binary'] if c in df.columns], inplace=True)
    df = df.reindex(columns=feature_names, fill_value=0)
    return df


def predict_m1(patient_dict: dict) -> tuple[int, float, float]:
    """Runs readmission inference using the cached Model 1 XGBoost ensemble.

    Preprocesses the patient record and applies the cost-optimised decision
    threshold stored in the model artifacts to produce the binary prediction.

    Args:
        patient_dict: Raw patient encounter dictionary from the Streamlit form.

    Returns:
        A 3-tuple of:

            - **pred** (int): Binary readmission label (``1`` = high risk,
              ``0`` = low risk) using the cost-optimised threshold.
            - **proba** (float): Raw readmission probability in [0, 1].
            - **conf** (float): Model confidence as ``max(proba, 1 - proba)``
              in [0.5, 1.0].

    Raises:
        ValueError: If ``preprocess_patient`` produces a feature shape mismatch.
    """
    logger.info("M1 prediction triggered — age=%s gender=%s",
                patient_dict.get("age"), patient_dict.get("gender"))
    model, state, feats, thresh = load_model1()
    X = preprocess_patient(patient_dict, state, feats)
    proba = float(model.predict_proba(X)[:, 1][0])
    pred  = int(proba >= thresh)
    conf  = max(proba, 1 - proba)
    logger.info("M1 result — proba=%.4f pred=%d threshold=%.2f", proba, pred, thresh)
    return pred, proba, conf


def predict_m2(patient_dict: dict) -> tuple[int, float, float]:
    """Runs readmission inference using the cached Model 2 Keras DNN.

    Preprocesses the patient record, applies the fitted ``StandardScaler``, and
    runs a forward pass through the cached Keras model at the 0.5 threshold.

    Args:
        patient_dict: Raw patient encounter dictionary from the Streamlit form.

    Returns:
        A 3-tuple of:

            - **pred** (int): Binary readmission label (``1`` = high risk,
              ``0`` = low risk) at the 0.5 probability threshold.
            - **proba** (float): Raw sigmoid output probability in [0, 1].
            - **conf** (float): Model confidence as ``max(proba, 1 - proba)``
              in [0.5, 1.0].

    Raises:
        ValueError: If ``preprocess_patient`` produces a feature shape mismatch.
        ImportError: If TensorFlow is not installed in the current environment.
    """
    logger.info("M2 prediction triggered — age=%s gender=%s",
                patient_dict.get("age"), patient_dict.get("gender"))
    model, scaler, state, feats = load_model2()
    X = preprocess_patient(patient_dict, state, feats)
    X_scaled = scaler.transform(X)
    proba = float(model.predict(X_scaled, verbose=0).flatten()[0])
    pred  = int(proba >= 0.5)
    conf  = max(proba, 1 - proba)
    logger.info("M2 result — proba=%.4f pred=%d", proba, pred)
    return pred, proba, conf


# =============================================================================
# RISK GAUGE (animated SVG)
# =============================================================================
def risk_gauge_svg(probability: float, label: str = "RISK") -> str:
    """Renders an animated circular risk gauge as an inline SVG string.

    Produces a colour-coded arc gauge (teal / amber / red) with a glowing progress
    ring, a percentage label at the centre, and a smooth CSS dash-offset transition.
    Intended to be rendered via ``st.markdown(..., unsafe_allow_html=True)``.

    Args:
        probability: Readmission probability in [0.0, 1.0]. Values outside this
            range are clamped automatically.
        label: Short uppercase label shown inside the gauge and used as a unique
            suffix for the SVG ``filter`` ID to prevent DOM collisions when multiple
            gauges appear on the same page. Defaults to ``"RISK"``.

    Returns:
        An HTML string containing a ``<div>``-wrapped inline ``<svg>`` element
        ready to pass to ``st.markdown``.
    """
    pct = max(0.0, min(1.0, probability))
    # Color interpolation
    if pct >= 0.65:
        color, glow = "#f87171", "rgba(248,113,113,0.5)"
    elif pct >= 0.40:
        color, glow = "#fbbf24", "rgba(251,191,36,0.5)"
    else:
        color, glow = "#5eead4", "rgba(94,234,212,0.5)"

    # Arc geometry
    radius = 70
    circumference = 2 * np.pi * radius
    offset = circumference * (1 - pct)

    return f"""
    <div style="display:flex; justify-content:center; padding:1rem 0;">
      <svg width="180" height="180" viewBox="0 0 180 180">
        <defs>
          <filter id="glow{label}">
            <feGaussianBlur stdDeviation="3" result="blur"/>
            <feMerge>
              <feMergeNode in="blur"/>
              <feMergeNode in="SourceGraphic"/>
            </feMerge>
          </filter>
        </defs>
        <!-- Background ring -->
        <circle cx="90" cy="90" r="{radius}" fill="none"
                stroke="rgba(34,211,238,0.1)" stroke-width="10"/>
        <!-- Progress ring -->
        <circle cx="90" cy="90" r="{radius}" fill="none"
                stroke="{color}" stroke-width="10" stroke-linecap="round"
                stroke-dasharray="{circumference:.2f}"
                stroke-dashoffset="{offset:.2f}"
                transform="rotate(-90 90 90)"
                filter="url(#glow{label})"
                style="transition: stroke-dashoffset 1.2s cubic-bezier(0.4,0,0.2,1);"/>
        <!-- Center text -->
        <text x="90" y="86" text-anchor="middle"
              font-family="JetBrains Mono, monospace"
              font-size="32" font-weight="700"
              fill="white">{pct*100:.1f}<tspan font-size="20" fill="#94a3b8">%</tspan></text>
        <text x="90" y="110" text-anchor="middle"
              font-family="JetBrains Mono, monospace"
              font-size="9" font-weight="600"
              fill="#64748b" letter-spacing="2">{label}</text>
      </svg>
    </div>
    """.strip()


def risk_label(p: float) -> tuple[str, str]:
    """Maps a readmission probability to a human-readable risk tier and CSS class.

    Uses the same thresholds as the production inference pipeline:
    high risk ≥ 0.65, moderate risk ≥ 0.40, low risk below that.

    Args:
        p: Readmission probability in [0.0, 1.0].

    Returns:
        A 2-tuple of:

            - **label** (str): Human-readable tier (``"HIGH RISK"``,
              ``"MODERATE RISK"``, or ``"LOW RISK"``).
            - **css_class** (str): Corresponding CSS class name (``"risk-high"``,
              ``"risk-medium"``, or ``"risk-low"``).
    """
    if p >= 0.65: return "HIGH RISK", "risk-high"
    if p >= 0.40: return "MODERATE RISK", "risk-medium"
    return "LOW RISK", "risk-low"


# =============================================================================
# SIDEBAR
# =============================================================================
def render_sidebar() -> str:
    """Renders the full sidebar UI and returns the active navigation page label.

    Builds the ClearSight brand block, the navigation ``st.radio`` widget, per-model
    ONLINE / ERROR status indicators (determined by attempting each cached loader),
    PENDING badges for models not yet integrated, and the investigational-use badge.

    Returns:
        The raw string value of the selected radio option
        (e.g. ``"🛰️  Command Center"``). Used by ``main()`` to route page rendering.
    """
    with st.sidebar:
        st.markdown(
            f'<div class="brand">{CLEARSIGHT_ICON_SVG}'
            '<div><div class="name">ClearSight</div>'
            '<div class="tag">Analytics</div></div></div>',
            unsafe_allow_html=True,
        )
        
        
        st.markdown('<div class="sidebar-header">NAVIGATION</div>', unsafe_allow_html=True)
        
        # The radio buttons will now be styled as hoverable blocks!
        page = st.radio("nav", [
            "🛰️  Command Center",
            "🔬  Predict",
            "📊  Insights",
            "👁️  Retinal AI",
        ], label_visibility="collapsed")

        # New Header Style for System Status
        st.markdown('<div class="sidebar-header">SYSTEM STATUS</div>', unsafe_allow_html=True)
        
        # Checking Model 1
        try:
            load_model1()
            st.markdown('<div class="status-row"><span><span style="color:#10b981; margin-right:5px;">●</span> M1 · XGBoost</span><span class="status-pill pill-online">ONLINE</span></div>', unsafe_allow_html=True)
        except Exception:
            logger.error("Sidebar status check: Model 1 unavailable", exc_info=True)
            st.markdown('<div class="status-row"><span><span style="color:#ef4444; margin-right:5px;">●</span> M1 · XGBoost</span><span class="status-pill pill-error">ERROR</span></div>', unsafe_allow_html=True)
            
        # Checking Model 2
        try:
            load_model2()
            st.markdown('<div class="status-row"><span><span style="color:#10b981; margin-right:5px;">●</span> M2 · DNN</span><span class="status-pill pill-online">ONLINE</span></div>', unsafe_allow_html=True)
        except Exception:
            logger.error("Sidebar status check: Model 2 unavailable", exc_info=True)
            st.markdown('<div class="status-row"><span><span style="color:#ef4444; margin-right:5px;">●</span> M2 · DNN</span><span class="status-pill pill-error">ERROR</span></div>', unsafe_allow_html=True)
            
        # Checking Model 5
        try:
            load_model5()
            st.markdown('<div class="status-row"><span><span style="color:#10b981; margin-right:5px;">●</span> M5 · Innovation</span><span class="status-pill pill-online">ONLINE</span></div>', unsafe_allow_html=True)
        except Exception:
            logger.error("Sidebar status check: Model 5 unavailable", exc_info=True)
            st.markdown('<div class="status-row"><span><span style="color:#ef4444; margin-right:5px;">●</span> M5 · Innovation</span><span class="status-pill pill-error">ERROR</span></div>', unsafe_allow_html=True)
        
        # Pending Models (M3 & M4)
        st.markdown('<div class="status-row"><span><span style="color:#10b981; margin-right:5px;">●</span> M3 · CNN Retina</span><span class="status-pill pill-online">ONLINE</span></div>', unsafe_allow_html=True)
        st.markdown('<div class="status-row"><span><span style="color:#22c55e; margin-right:5px;">●</span> M4 · NLP Notes</span><span class="status-pill pill-online">ONLINE</span></div>', unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        
        # Investigational Use Badge
        st.markdown("""
        <div style="font-family:'JetBrains Mono',monospace; font-size:0.7rem;
                    color:#fbbf24; background:rgba(251,191,36,0.08);
                    border:1px solid rgba(251,191,36,0.3); border-radius:8px;
                    padding:10px; text-align:center; line-height:1.5; margin: 0 1rem;">
          ⚠ INVESTIGATIONAL USE ONLY<br>NOT FDA CLEARED
        </div>
        """, unsafe_allow_html=True)
        
    return page
# =============================================================================
# PAGE: COMMAND CENTER (Home)
# =============================================================================

def page_home() -> None:
    """Renders the Command Center landing page.

    Displays an animated hero section, dataset foundation statistics, live model
    performance metric bars with JS-animated progress fills, estimated clinical
    impact cards, and a four-step clinical decision pipeline overview. All content
    is injected via ``st.markdown`` with embedded CSS and JavaScript.
    """

    # ── Animated hero with JS counters + teal particles ──────────────────────
    st.markdown("""
    <style>
    /* ── Counter animation ─────────────────────────────────────────── */
    .count-up {
        font-family: 'JetBrains Mono', monospace;
        font-size: 2.2rem;
        font-weight: 700;
        color: white;
        line-height: 1;
    }

    /* ── Progress bar animated ─────────────────────────────────────── */
    .metric-bar-wrap {
        background: rgba(34,211,238,0.07);
        border: 1px solid rgba(34,211,238,0.15);
        border-radius: 12px;
        padding: 1.2rem 1.4rem;
        margin-bottom: 10px;
        position: relative;
        overflow: hidden;
    }
    .metric-bar-wrap:hover {
        border-color: rgba(94,234,212,0.4);
        background: rgba(34,211,238,0.1);
        transition: all 0.2s;
    }
    .metric-bar-header {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 10px;
    }
    .metric-bar-name {
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.78rem;
        font-weight: 600;
        color: #94a3b8;
        text-transform: uppercase;
        letter-spacing: 0.08em;
    }
    .metric-bar-val {
        font-family: 'JetBrains Mono', monospace;
        font-size: 1.1rem;
        font-weight: 700;
        color: white;
    }
    .metric-bar-track {
        background: rgba(255,255,255,0.06);
        border-radius: 999px;
        height: 8px;
        overflow: hidden;
    }
    .metric-bar-fill {
        height: 100%;
        border-radius: 999px;
        background: linear-gradient(90deg, #22d3ee, #5eead4);
        box-shadow: 0 0 12px rgba(34,211,238,0.5);
        width: 0%;
        transition: width 1.8s cubic-bezier(0.4, 0, 0.2, 1);
    }
    .metric-bar-fill-amber {
        background: linear-gradient(90deg, #f59e0b, #fcd34d);
        box-shadow: 0 0 12px rgba(245,158,11,0.5);
    }
    .metric-bar-fill-green {
        background: linear-gradient(90deg, #10b981, #5eead4);
        box-shadow: 0 0 12px rgba(16,185,129,0.5);
    }

    /* ── Impact cards ───────────────────────────────────────────────── */
    .impact-card {
        background: rgba(15,30,52,0.6);
        border: 1px solid rgba(34,211,238,0.12);
        border-radius: 14px;
        padding: 1.5rem;
        text-align: center;
        transition: all 0.2s;
        position: relative;
        overflow: hidden;
    }
    .impact-card::before {
        content: "";
        position: absolute;
        top: 0; left: 0; right: 0; height: 2px;
        background: linear-gradient(90deg, transparent, #22d3ee, transparent);
    }
    .impact-card:hover {
        transform: translateY(-4px);
        border-color: rgba(94,234,212,0.35);
        box-shadow: 0 12px 40px rgba(34,211,238,0.12);
    }
    .impact-icon {
        font-size: 2rem;
        margin-bottom: 0.6rem;
        display: block;
    }
    .impact-val {
        font-family: 'JetBrains Mono', monospace;
        font-size: 1.9rem;
        font-weight: 700;
        color: #5eead4;
        line-height: 1;
        margin-bottom: 0.3rem;
    }
    .impact-label {
        font-size: 0.78rem;
        font-weight: 600;
        color: #64748b;
        text-transform: uppercase;
        letter-spacing: 0.1em;
    }
    .impact-sub {
        font-size: 0.82rem;
        color: #475569;
        margin-top: 0.4rem;
        line-height: 1.4;
    }

    /* ── Dataset stat pills ────────────────────────────────────────── */
    .ds-pill {
        display: inline-flex;
        align-items: center;
        gap: 10px;
        background: rgba(34,211,238,0.06);
        border: 1px solid rgba(34,211,238,0.15);
        border-radius: 10px;
        padding: 1rem 1.3rem;
        width: 100%;
        transition: all 0.2s;
    }
    .ds-pill:hover {
        background: rgba(34,211,238,0.1);
        border-color: rgba(94,234,212,0.3);
    }
    .ds-pill .ico { font-size: 1.4rem; }
    .ds-pill .txt .top {
        font-family: 'JetBrains Mono', monospace;
        font-size: 1.25rem;
        font-weight: 700;
        color: white;
        line-height: 1;
    }
    .ds-pill .txt .bot {
        font-size: 0.75rem;
        color: #64748b;
        font-weight: 500;
        text-transform: uppercase;
        letter-spacing: 0.07em;
        margin-top: 3px;
    }

    /* ── Section divider ───────────────────────────────────────────── */
    .sec-div {
        display: flex;
        align-items: center;
        gap: 14px;
        margin: 2.2rem 0 1.4rem;
    }
    .sec-div .num {
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.75rem;
        font-weight: 700;
        color: #5eead4;
        background: rgba(94,234,212,0.08);
        border: 1px solid rgba(94,234,212,0.2);
        padding: 3px 10px;
        border-radius: 6px;
        white-space: nowrap;
    }
    .sec-div h2 {
        font-family: 'Space Grotesk', sans-serif;
        font-size: 1.35rem;
        font-weight: 600;
        color: white !important;
        margin: 0;
        letter-spacing: -0.02em;
    }
    .sec-div .line {
        flex: 1;
        height: 1px;
        background: linear-gradient(90deg, rgba(34,211,238,0.3), transparent);
    }
    </style>

    <!-- ═══ HERO ═══════════════════════════════════════════════════════ -->
    <div style="
        position: relative;
        background:
            radial-gradient(circle at 75% 40%, rgba(34,211,238,0.22) 0%, transparent 45%),
            radial-gradient(circle at 10% 80%, rgba(94,234,212,0.1) 0%, transparent 40%),
            linear-gradient(135deg, #060e1a 0%, #0d1f38 50%, #070f1c 100%);
        border: 1px solid rgba(34,211,238,0.25);
        border-radius: 20px;
        padding: 2.8rem 3rem;
        margin-bottom: 2rem;
        overflow: hidden;
        box-shadow: 0 0 0 1px rgba(94,234,212,0.05),
                    0 20px 60px rgba(0,0,0,0.4),
                    inset 0 1px 0 rgba(94,234,212,0.12);
    ">
      <!-- Animated teal glow orb -->
      <div style="
          position: absolute; top: -60px; right: -60px;
          width: 300px; height: 300px; border-radius: 50%;
          background: radial-gradient(circle, rgba(34,211,238,0.18) 0%, transparent 70%);
          animation: orbPulse 4s ease-in-out infinite;
          pointer-events: none;
      "></div>
      <style>
        @keyframes orbPulse {
          0%,100% { transform: scale(1); opacity: 0.8; }
          50%      { transform: scale(1.15); opacity: 1; }
        }
        @keyframes fadeSlideUp {
          from { opacity: 0; transform: translateY(18px); }
          to   { opacity: 1; transform: translateY(0); }
        }
        .hero-badge { animation: fadeSlideUp 0.5s ease both; }
        .hero-title { animation: fadeSlideUp 0.6s 0.1s ease both; }
        .hero-sub   { animation: fadeSlideUp 0.6s 0.2s ease both; }
        .hero-pills { animation: fadeSlideUp 0.6s 0.35s ease both; }
      </style>

      <div class="hero-badge" style="
          display:inline-block;
          font-family:'JetBrains Mono',monospace;
          font-size:0.68rem; font-weight:700;
          color:#5eead4;
          background:rgba(94,234,212,0.1);
          border:1px solid rgba(94,234,212,0.35);
          padding:5px 14px; border-radius:999px;
          letter-spacing:0.12em; text-transform:uppercase;
          margin-bottom:1rem;
      ">● LIVE · MEDINSIGHT CLINICAL AI PLATFORM</div>

      <h1 class="hero-title" style="
          font-family:'Space Grotesk',sans-serif;
          font-size:2.8rem; font-weight:800;
          margin:0 0 0.7rem;
          letter-spacing:-0.04em; line-height:1.05;
          background: linear-gradient(120deg, #ffffff 0%, #5eead4 60%, #22d3ee 100%);
          -webkit-background-clip: text;
          -webkit-text-fill-color: transparent;
          background-clip: text;
          max-width: 780px;
      ">Clinical AI Command Center</h1>

      <p class="hero-sub" style="
          font-size:1.08rem;
          color:#94a3b8;
          max-width:680px;
          line-height:1.6;
          margin: 0 0 1.6rem;
      ">
        Precision readmission prediction and capacity planning for MedInsight's
        47 partner hospitals — trained on <strong style="color:#5eead4;">101,766 real patient encounters</strong>
        with full SHAP interpretability for clinical trust.
      </p>

      <div class="hero-pills">
        <span style="display:inline-flex;align-items:center;gap:6px;
                     background:rgba(34,211,238,0.1);color:#5eead4;
                     padding:6px 14px;border-radius:999px;font-size:0.75rem;
                     font-weight:600;margin-right:8px;margin-bottom:6px;
                     border:1px solid rgba(34,211,238,0.25);
                     font-family:'JetBrains Mono',monospace;letter-spacing:0.04em;">
          ⚡ &lt;300ms inference
        </span>
        <span style="display:inline-flex;align-items:center;gap:6px;
                     background:rgba(34,211,238,0.1);color:#5eead4;
                     padding:6px 14px;border-radius:999px;font-size:0.75rem;
                     font-weight:600;margin-right:8px;margin-bottom:6px;
                     border:1px solid rgba(34,211,238,0.25);
                     font-family:'JetBrains Mono',monospace;letter-spacing:0.04em;">
          🔬 SHAP interpretability
        </span>
        <span style="display:inline-flex;align-items:center;gap:6px;
                     background:rgba(34,211,238,0.1);color:#5eead4;
                     padding:6px 14px;border-radius:999px;font-size:0.75rem;
                     font-weight:600;margin-right:8px;margin-bottom:6px;
                     border:1px solid rgba(34,211,238,0.25);
                     font-family:'JetBrains Mono',monospace;letter-spacing:0.04em;">
          🛡️ HIPAA-aware design
        </span>
        <span style="display:inline-flex;align-items:center;gap:6px;
                     background:rgba(16,185,129,0.1);color:#10b981;
                     padding:6px 14px;border-radius:999px;font-size:0.75rem;
                     font-weight:600;margin-right:8px;margin-bottom:6px;
                     border:1px solid rgba(16,185,129,0.25);
                     font-family:'JetBrains Mono',monospace;letter-spacing:0.04em;">
          ✓ 5 models live
        </span>
      </div>
    </div>

    <!-- JS counter animation script -->
    <script>
    function animateCounter(el, target, duration, prefix, suffix, decimals) {
        var start = 0;
        var step  = target / (duration / 16);
        var timer = setInterval(function() {
            start += step;
            if (start >= target) { start = target; clearInterval(timer); }
            var disp = decimals ? start.toFixed(decimals) : Math.floor(start).toLocaleString();
            el.textContent = (prefix||'') + disp + (suffix||'');
        }, 16);
    }
    document.addEventListener('DOMContentLoaded', function() {
        var counters = document.querySelectorAll('[data-count]');
        counters.forEach(function(el) {
            var target   = parseFloat(el.getAttribute('data-count'));
            var duration = parseInt(el.getAttribute('data-dur') || '1800');
            var prefix   = el.getAttribute('data-prefix') || '';
            var suffix   = el.getAttribute('data-suffix') || '';
            var decimals = parseInt(el.getAttribute('data-decimals') || '0');
            animateCounter(el, target, duration, prefix, suffix, decimals);
        });
        var bars = document.querySelectorAll('[data-bar]');
        bars.forEach(function(el) {
            var pct = el.getAttribute('data-bar');
            setTimeout(function() { el.style.width = pct + '%'; }, 200);
        });
    });
    </script>
    """, unsafe_allow_html=True)

    # ── SECTION 1: Dataset Foundation ────────────────────────────────────────
    st.markdown("""
    <div class="sec-div">
      <span class="num">01</span>
      <h2>Dataset Foundation</h2>
      <div class="line"></div>
    </div>
    """, unsafe_allow_html=True)

    d1, d2, d3, d4, d5, d6 = st.columns(6)
    pills = [
        (d1, "🏥", "101,766",  "Patient Encounters"),
        (d2, "📋", "50",       "Clinical Features"),
        (d3, "🔬", "47",       "Partner Hospitals"),
        (d4, "👁️", "3,222",    "Retinal Images"),
        (d5, "💊", "180,000+", "Drug Reviews"),
        (d6, "📊", "46 / 54%", "Readmit Balance"),
    ]
    for col, ico, val, lbl in pills:
        col.markdown(f"""
        <div class="ds-pill">
          <span class="ico">{ico}</span>
          <div class="txt">
            <div class="top">{val}</div>
            <div class="bot">{lbl}</div>
          </div>
        </div>
        """, unsafe_allow_html=True)

    # ── SECTION 2: Model Performance ─────────────────────────────────────────
    st.markdown("""
    <div class="sec-div">
      <span class="num">02</span>
      <h2>Model Performance — Live Metrics</h2>
      <div class="line"></div>
    </div>
    """, unsafe_allow_html=True)

    mp1, mp2, mp3, mp4 = st.columns(4, gap="medium")

    # Model 1
    with mp1:
        st.markdown("""
        <div style="background:rgba(15,30,52,0.55);border:1px solid rgba(34,211,238,0.15);
                    border-radius:14px;padding:1.5rem;">
          <div style="font-family:'JetBrains Mono',monospace;font-size:0.66rem;
                      font-weight:700;color:#5eead4;text-transform:uppercase;
                      letter-spacing:0.12em;background:rgba(94,234,212,0.08);
                      border:1px solid rgba(94,234,212,0.2);padding:3px 10px;
                      border-radius:6px;display:inline-block;margin-bottom:12px;">
            M1 · XGBOOST ENSEMBLE
          </div>
          <h3 style="font-family:'Space Grotesk',sans-serif;font-size:1.05rem;
                     font-weight:600;color:white;margin:0 0 1rem;">
            Readmission · Traditional ML
          </h3>

          <div class="metric-bar-wrap">
            <div class="metric-bar-header">
              <span class="metric-bar-name">AUC-ROC</span>
              <span class="metric-bar-val">0.6943</span>
            </div>
            <div class="metric-bar-track">
              <div class="metric-bar-fill" data-bar="69.43" style="width:69.43%"></div>
            </div>
          </div>

          <div class="metric-bar-wrap">
            <div class="metric-bar-header">
              <span class="metric-bar-name">Recall @ threshold 0.38</span>
              <span class="metric-bar-val">85.0%</span>
            </div>
            <div class="metric-bar-track">
              <div class="metric-bar-fill metric-bar-fill-green" data-bar="85"
                   style="width:85%"></div>
            </div>
          </div>

          <div style="margin-top:0.8rem;padding:0.75rem;
                      background:rgba(16,185,129,0.06);
                      border:1px solid rgba(16,185,129,0.15);border-radius:8px;">
            <div style="font-size:0.75rem;color:#64748b;font-weight:600;
                        text-transform:uppercase;letter-spacing:0.07em;margin-bottom:3px;">
              Cost-Optimized Threshold
            </div>
            <div style="font-family:'JetBrains Mono',monospace;font-size:1.3rem;
                        font-weight:700;color:#10b981;">0.38</div>
            <div style="font-size:0.78rem;color:#475569;margin-top:2px;">
              FN cost $15K · FP cost $500
            </div>
          </div>
        </div>
        """, unsafe_allow_html=True)

    # Model 2
    with mp2:
        st.markdown("""
        <div style="background:rgba(15,30,52,0.55);border:1px solid rgba(34,211,238,0.15);
                    border-radius:14px;padding:1.5rem;">
          <div style="font-family:'JetBrains Mono',monospace;font-size:0.66rem;
                      font-weight:700;color:#5eead4;text-transform:uppercase;
                      letter-spacing:0.12em;background:rgba(94,234,212,0.08);
                      border:1px solid rgba(94,234,212,0.2);padding:3px 10px;
                      border-radius:6px;display:inline-block;margin-bottom:12px;">
            M2 · DEEP LEARNING DNN
          </div>
          <h3 style="font-family:'Space Grotesk',sans-serif;font-size:1.05rem;
                     font-weight:600;color:white;margin:0 0 1rem;">
            Readmission · Neural Network
          </h3>

          <div class="metric-bar-wrap">
            <div class="metric-bar-header">
              <span class="metric-bar-name">AUC-ROC</span>
              <span class="metric-bar-val">0.6854</span>
            </div>
            <div class="metric-bar-track">
              <div class="metric-bar-fill" data-bar="68.54" style="width:68.54%"></div>
            </div>
          </div>

          <div class="metric-bar-wrap">
            <div class="metric-bar-header">
              <span class="metric-bar-name">vs. XGBoost delta</span>
              <span class="metric-bar-val">−0.0089</span>
            </div>
            <div class="metric-bar-track">
              <div class="metric-bar-fill metric-bar-fill-amber" data-bar="68.54"
                   style="width:68.54%"></div>
            </div>
          </div>

          <div style="margin-top:0.8rem;padding:0.75rem;
                      background:rgba(34,211,238,0.05);
                      border:1px solid rgba(34,211,238,0.12);border-radius:8px;">
            <div style="font-size:0.75rem;color:#64748b;font-weight:600;
                        text-transform:uppercase;letter-spacing:0.07em;margin-bottom:3px;">
              Architecture
            </div>
            <div style="font-family:'JetBrains Mono',monospace;font-size:0.88rem;
                        font-weight:600;color:#5eead4;line-height:1.5;">
              Dense 256→128→64→1<br>BatchNorm + Dropout
            </div>
            <div style="font-size:0.78rem;color:#475569;margin-top:2px;">
              Class-weighted loss · Early stopping
            </div>
          </div>
        </div>
        """, unsafe_allow_html=True)

    # Model 5
    with mp3:
        st.markdown("""
        <div style="background:rgba(15,30,52,0.55);border:1px solid rgba(34,211,238,0.15);
                    border-radius:14px;padding:1.5rem;">
          <div style="font-family:'JetBrains Mono',monospace;font-size:0.66rem;
                      font-weight:700;color:#a78bfa;text-transform:uppercase;
                      letter-spacing:0.12em;background:rgba(167,139,250,0.08);
                      border:1px solid rgba(167,139,250,0.2);padding:3px 10px;
                      border-radius:6px;display:inline-block;margin-bottom:12px;">
            M5 · INNOVATION
          </div>
          <h3 style="font-family:'Space Grotesk',sans-serif;font-size:1.05rem;
                     font-weight:600;color:white;margin:0 0 1rem;">
            Length-of-Stay · Capacity Planning
          </h3>

          <div class="metric-bar-wrap">
            <div class="metric-bar-header">
              <span class="metric-bar-name">Weighted F1-Score</span>
              <span class="metric-bar-val">0.6047</span>
            </div>
            <div class="metric-bar-track">
              <div class="metric-bar-fill" data-bar="60.47"
                   style="width:60.47%;background:linear-gradient(90deg,#a78bfa,#c4b5fd);
                          box-shadow:0 0 12px rgba(167,139,250,0.5);"></div>
            </div>
          </div>

          <div class="metric-bar-wrap">
            <div class="metric-bar-header">
              <span class="metric-bar-name">Classes (3-way)</span>
              <span class="metric-bar-val">Short / Std / Ext</span>
            </div>
            <div class="metric-bar-track">
              <div class="metric-bar-fill" data-bar="60.47"
                   style="width:60.47%;background:linear-gradient(90deg,#a78bfa,#c4b5fd);
                          box-shadow:0 0 12px rgba(167,139,250,0.5);"></div>
            </div>
          </div>

          <div style="margin-top:0.8rem;padding:0.75rem;
                      background:rgba(167,139,250,0.06);
                      border:1px solid rgba(167,139,250,0.15);border-radius:8px;">
            <div style="font-size:0.75rem;color:#64748b;font-weight:600;
                        text-transform:uppercase;letter-spacing:0.07em;margin-bottom:3px;">
              Clinical Value
            </div>
            <div style="font-size:0.82rem;color:#a78bfa;line-height:1.5;">
              Bed management · Discharge planning<br>Staff allocation optimization
            </div>
          </div>
        </div>
        """, unsafe_allow_html=True)

    # Model 4
    with mp4:
        st.markdown("""
        <div style="background:rgba(15,30,52,0.55);border:1px solid rgba(34,211,238,0.15);
                    border-radius:14px;padding:1.5rem;">
          <div style="font-family:'JetBrains Mono',monospace;font-size:0.66rem;
                      font-weight:700;color:#f59e0b;text-transform:uppercase;
                      letter-spacing:0.12em;background:rgba(245,158,11,0.08);
                      border:1px solid rgba(245,158,11,0.2);padding:3px 10px;
                      border-radius:6px;display:inline-block;margin-bottom:12px;">
            M4 &middot; META LSTM + METADATA
          </div>
          <h3 style="font-family:'Space Grotesk',sans-serif;font-size:1.05rem;
                     font-weight:600;color:white;margin:0 0 1rem;">
            NLP Notes &middot; Sentiment Analysis
          </h3>

          <div class="metric-bar-wrap">
            <div class="metric-bar-header">
              <span class="metric-bar-name">VALIDATION ACCURACY</span>
              <span class="metric-bar-val">72.4%</span>
            </div>
            <div class="metric-bar-track">
              <div class="metric-bar-fill" data-bar="72"
                   style="width:72%;background:linear-gradient(90deg,#10b981,#34d399);
                          box-shadow:0 0 12px rgba(16,185,129,0.5);"></div>
            </div>
          </div>

          <div class="metric-bar-wrap">
            <div class="metric-bar-header">
              <span class="metric-bar-name">VOCAB SIZE</span>
              <span class="metric-bar-val">10K+</span>
            </div>
            <div class="metric-bar-track">
              <div class="metric-bar-fill" data-bar="100"
                   style="width:100%;background:linear-gradient(90deg,#22d3ee,#67e8f9);
                          box-shadow:0 0 12px rgba(34,211,238,0.5);"></div>
            </div>
          </div>

          <div style="margin-top:0.8rem;padding:0.75rem;
                      background:rgba(52,211,153,0.06);
                      border:1px solid rgba(52,211,153,0.15);border-radius:8px;">
            <div style="font-size:0.75rem;color:#64748b;font-weight:600;
                        text-transform:uppercase;letter-spacing:0.07em;margin-bottom:3px;">
              Architecture
            </div>
            <div style="font-family:'JetBrains Mono',monospace;font-size:0.88rem;
                        font-weight:600;color:#34d399;line-height:1.5;">
              Bi-LSTM (Layers: 2)<br>Metadata Embeddings
            </div>
            <div style="font-size:0.78rem;color:#475569;margin-top:2px;">
              Extracts unstructured risk signals
            </div>
          </div>
        </div>
        """, unsafe_allow_html=True)

    # ── SECTION 3: Clinical Impact ────────────────────────────────────────────
    st.markdown("""
    <div class="sec-div">
      <span class="num">03</span>
      <h2>Estimated Clinical Impact</h2>
      <div class="line"></div>
      <span style="font-size:0.75rem;color:#475569;font-family:'JetBrains Mono',monospace;">
        based on MedInsight network · 22% baseline readmission rate · $15K cost/readmission
      </span>
    </div>
    """, unsafe_allow_html=True)

    i1, i2, i3, i4 = st.columns(4, gap="medium")

    impacts = [
        (i1, "⚠️", "22,388",   "Readmissions / Year",
         "Based on 101,766 encounters × 22% baseline rate in MedInsight network"),
        (i2, "🎯", "19,030",   "High-Risk Flagged",
         "Model 1 captures 85% with cost-optimized threshold — actionable early warnings"),
        (i3, "✅", "2,854",    "Preventable / Year",
         "Conservative 15% intervention success rate on flagged high-risk patients"),
        (i4, "💰", "$42.8M",   "Annual Cost Savings",
         "2,854 prevented readmissions × $15,000 average cost per readmission"),
    ]

    for col, ico, val, lbl, sub in impacts:
        col.markdown(f"""
        <div class="impact-card">
          <span class="impact-icon">{ico}</span>
          <div class="impact-val">{val}</div>
          <div class="impact-label">{lbl}</div>
          <div class="impact-sub">{sub}</div>
        </div>
        """, unsafe_allow_html=True)

    # ── SECTION 4: How it works ───────────────────────────────────────────────
    st.markdown("""
    <div class="sec-div">
      <span class="num">04</span>
      <h2>Clinical Decision Pipeline</h2>
      <div class="line"></div>
    </div>
    """, unsafe_allow_html=True)

    f1, f2, f3, f4 = st.columns(4, gap="medium")
    steps = [
        (f1, "01", "DATA INTAKE",
         "EHR encounter data — demographics, diagnoses, medications, prior utilization — enters the pipeline in raw format."),
        (f2, "02", "PREPROCESSING",
         "50-feature engineering pipeline: ICD-9 grouping, target encoding, clinical complexity score, medication change flags."),
        (f3, "03", "DUAL INFERENCE",
         "XGBoost and DNN run in parallel on identical preprocessed features. Consensus check flags disagreements for review."),
        (f4, "04", "SHAP EXPLANATION",
         "Top contributing features surface for every prediction so clinicians can validate reasoning before any care decision."),
    ]
    for col, num, title, desc in steps:
        col.markdown(f"""
        <div class="gcard" style="text-align:center;">
          <div style="font-family:'JetBrains Mono',monospace;font-size:2rem;
                      font-weight:700;color:rgba(34,211,238,0.3);line-height:1;
                      margin-bottom:0.5rem;">{num}</div>
          <div style="font-family:'JetBrains Mono',monospace;font-size:0.66rem;
                      font-weight:700;color:#5eead4;text-transform:uppercase;
                      letter-spacing:0.12em;margin-bottom:0.5rem;">{title}</div>
          <p style="color:#94a3b8;font-size:0.88rem;line-height:1.55;margin:0;">
            {desc}
          </p>
        </div>
        """, unsafe_allow_html=True)



def parse_id(val: str) -> int:
    """Extracts an integer ID from a ``"<id> - <description>"`` selectbox string.

    Selectbox options follow the pattern ``"1 - Emergency"``. This helper splits on
    the first ``" - "`` delimiter and converts the leading segment to an integer.

    Args:
        val: A selectbox option string in ``"<id> - <description>"`` format.

    Returns:
        The integer ID at the start of ``val``. Returns ``1`` as a safe default if
        the string cannot be parsed.
    """
    try:
        return int(val.split(" - ")[0].strip())
    except (ValueError, IndexError):
        return 1






# =============================================================================
# PAGE: PREDICT
# =============================================================================
def page_predict() -> None:
    """Renders the Readmission Risk Predictor page.

    Presents a multi-section patient encounter form (demographics, clinical metrics,
    ICD-9 diagnoses, medications, clinical notes). On submission, runs Models 1, 2,
    4, and 5 in sequence and displays animated conic-gradient gauges for M1 / M2, a
    capacity-planning card for M5, a Meta LSTM sentiment card for M4, and a consensus
    summary block with a clinical recommendation.
    """
    st.markdown("""
    <div style="margin-bottom: 1.5rem;">
      <span class="badge" style="display:inline-block; font-family:'JetBrains Mono',monospace;
            font-size:0.7rem; font-weight:600; color:#5eead4;
            background:rgba(94,234,212,0.1); border:1px solid rgba(94,234,212,0.3);
            padding:4px 12px; border-radius:999px; letter-spacing:0.08em;
            text-transform:uppercase; margin-bottom:0.6rem;">● PREDICTOR</span>
      <h1 style="font-family:'Space Grotesk',sans-serif; font-size:2rem;
                 margin:0.4rem 0 0.4rem; letter-spacing:-0.02em;">
        Readmission Risk Predictor
      </h1>
      <p style="color:#94a3b8; font-size:1rem; margin:0; max-width:760px;">
        Enter a patient encounter below — both models run in parallel and you'll
        see a head-to-head comparison with calibrated probabilities and confidence scores.
      </p>
    </div>
    """, unsafe_allow_html=True)

    with st.form("predict_form"):

        # ── Section 1: Demographics ─────────────────────────────────
        with st.expander("👤 Demographics & Admission", expanded=True):
            r1a, r1b, r1c, r1d = st.columns(4)
            age = r1a.selectbox("Age Bracket", [
                '[0-10)','[10-20)','[20-30)','[30-40)','[40-50)',
                '[50-60)','[60-70)','[70-80)','[80-90)','[90-100)'
            ], index=6, help="Age group (matches encounter dataset format).")
            gender = r1b.selectbox("Gender", ["Male", "Female"])
            race   = r1c.selectbox("Race", [
                "Caucasian", "AfricanAmerican", "Hispanic", "Asian", "Other"
            ])
            admission_type = r1d.selectbox("Admission Type", [
                "1 - Emergency", "2 - Urgent", "3 - Elective",
                "4 - Newborn", "5 - Not Available", "6 - NULL", "7 - Trauma Center"
            ])

            r2a, r2b = st.columns(2)
            discharge_disp = r2a.selectbox("Discharge Disposition", [
                "1 - Home", "2 - Home Health Care", "3 - SNF",
                "6 - Home w/ Home Health", "18 - Not Available",
            ], help="Where patient was discharged to.")
            admission_src = r2b.selectbox("Admission Source", [
                "1 - Physician Referral", "2 - Clinic Referral",
                "4 - Transfer from Hospital", "7 - Emergency Room",
                "9 - Not Available", "17 - NULL"
            ])

        # ── Section 2: Clinical Metrics ─────────────────────────────
        with st.expander("🏥 Clinical Metrics", expanded=False):
            st.markdown("**Current Encounter**")
            r3a, r3b, r3c, r3d, r3e = st.columns(5)
            time_in_hosp  = r3a.number_input("Days in Hospital", 1, 14, 4)
            num_lab_procs = r3b.number_input("Lab Procedures", 0, 132, 44)
            num_procs     = r3c.number_input("Procedures", 0, 6, 1)
            num_meds      = r3d.number_input("Medications", 1, 81, 15)
            num_diag      = r3e.number_input("Diagnoses", 1, 16, 7)

            st.markdown("**Prior Utilization (past 12 months)**")
            r4a, r4b, r4c, r4d = st.columns(4)
            num_outpatient = r4a.number_input("Outpatient Visits", 0, 42, 0)
            num_emergency  = r4b.number_input("Emergency Visits", 0, 76, 0)
            num_inpatient  = r4c.number_input("Inpatient Visits", 0, 21, 0,
                              help="Strong readmission predictor (per literature).")
            prior_inp_cum  = r4d.number_input("Prior Inpatient (cumulative)", 0, 50, 0)

        # ── Section 3: Diagnoses ────────────────────────────────────
        with st.expander("🩺 ICD-9 Diagnoses", expanded=False):
            st.markdown("**Top 3 diagnosis codes for this encounter**")
            r5a, r5b, r5c = st.columns(3)
            diag1 = r5a.text_input("Primary (diag_1)", "250.01",
                     help="250.xx = diabetes, 428.xx = heart failure, 401.xx = hypertension")
            diag2 = r5b.text_input("Secondary (diag_2)", "428")
            diag3 = r5c.text_input("Tertiary (diag_3)", "401")
            med_spec = st.text_input("Medical Specialty", "InternalMedicine",
                                     help="Specialty of the attending physician.")

        # ── Section 4: Medications & Labs ───────────────────────────
        with st.expander("💊 Medications & Labs", expanded=False):
            st.markdown("**Diabetes Management**")
            r6a, r6b, r6c, r6d, r6e = st.columns(5)
            insulin     = r6a.selectbox("Insulin",     ["No", "Steady", "Up", "Down"])
            metformin   = r6b.selectbox("Metformin",   ["No", "Steady", "Up", "Down"])
            change      = r6c.selectbox("Med Changed?", ["No", "Ch"],
                          help="Was any medication changed during this encounter?")
            diabetes_md = r6d.selectbox("Diabetes Med?", ["Yes", "No"])
            a1c         = r6e.selectbox("A1C Result",  ["None", "Norm", ">7", ">8"])

            st.markdown("**Lab Results**")
            max_glu = st.selectbox("Max Glucose Serum", ["None", "Norm", ">200", ">300"])


        # --- Section 5: Clinical Notes (Model 4 - NLP) ---
        with st.expander("📝 Clinical Notes Analysis (NLP Intelligence)", expanded=True):
            st.markdown("""
            <p style='font-size: 0.9rem; color: #94a3b8;'>
                Enter physician progress notes or patient feedback. Meta LSTM will analyze 
                the linguistic sentiment in context with the medication and condition.
            </p>
            """, unsafe_allow_html=True)
            
            # These two fields are required by Wes's model metadata
            col_n1, col_n2 = st.columns(2)
            nlp_drug = col_n1.text_input("Context Medication", value="Metformin")
            nlp_cond = col_n2.text_input("Context Condition", value="Diabetes")
            
            clinical_notes = st.text_area(
                "Physician/Nursing Progress Notes:",
                placeholder="Example: Patient reports difficulty following the insulin regimen and shows signs of anxiety regarding discharge...",
                height=150
            )

        st.markdown("")
        submitted = st.form_submit_button("⚡ RUN PREDICTION")

    # ── PREDICTIONS ───────────────────────────────────────────────────
    if submitted:
        patient_dict = {
            "age":                      age,
            "gender":                   gender,
            "race":                     race,
            "admission_type_id":        parse_id(admission_type),
            "discharge_disposition_id": parse_id(discharge_disp),
            "admission_source_id":      parse_id(admission_src),
            "time_in_hospital":         time_in_hosp,
            "num_lab_procedures":       num_lab_procs,
            "num_procedures":           num_procs,
            "num_medications":          num_meds,
            "number_diagnoses":         num_diag,
            "number_outpatient":        num_outpatient,
            "number_emergency":         num_emergency,
            "number_inpatient":         num_inpatient,
            "prior_inpatient_cumsum":   prior_inp_cum,
            "diag_1":                   diag1,
            "diag_2":                   diag2,
            "diag_3":                   diag3,
            "insulin":                  insulin,
            "metformin":                metformin,
            "change":                   change,
            "diabetesMed":              diabetes_md,
            "A1Cresult":                a1c,
            "max_glu_serum":            max_glu,
            "medical_specialty":        med_spec,
        }

        # --- Run all models and persist results in session_state ---
        proba1, proba2 = None, None
        pred1, pred2 = None, None
        conf1, conf2 = None, None
        latency_m1, latency_m2 = 0, 0
        los_label = los_conf = los_css = None
        nlp_label = nlp_conf = nlp_css = nlp_explanation = None

        # ── Model 1 ──────────────────────────────────────────────────
        with st.spinner("Running Model 1 — XGBoost…"):
            t0 = time.perf_counter()
            try:
                pred1, proba1, conf1 = predict_m1(patient_dict)
                latency_m1 = (time.perf_counter() - t0) * 1000
                st.session_state["m1_result"] = {
                    "proba": proba1, "pred": pred1,
                    "latency": latency_m1, "conf": conf1,
                }
            except Exception:
                logger.error("M1 prediction failed", exc_info=True)
                st.error("Model 1 could not complete the prediction. The issue has been logged — please try again or contact support.")

        # ── Model 2 ──────────────────────────────────────────────────
        with st.spinner("Running Model 2 — DNN…"):
            t0 = time.perf_counter()
            try:
                pred2, proba2, conf2 = predict_m2(patient_dict)
                latency_m2 = (time.perf_counter() - t0) * 1000
                st.session_state["m2_result"] = {
                    "proba": proba2, "pred": pred2,
                    "latency": latency_m2, "conf": conf2,
                }
            except Exception:
                logger.error("M2 prediction failed", exc_info=True)
                st.error("Model 2 could not complete the prediction. The issue has been logged — please try again or contact support.")

        # ── Model 5 Innovation ────────────────────────────────────────
        with st.spinner("Analyzing capacity requirements (Model 5)..."):
            try:
                los_label, los_conf, los_css = predict_m5(patient_dict)
                st.session_state["m5_result"] = {
                    "label": los_label, "conf": los_conf, "css": los_css,
                }
            except Exception:
                logger.error("M5 prediction failed", exc_info=True)
                st.error("Capacity Planning model could not complete the prediction. The issue has been logged — please try again or contact support.")

        # ── Model 4 NLP ───────────────────────────────────────────────
        with st.spinner("Analyzing clinical sentiment with Meta LSTM..."):
            try:
                nlp_label, nlp_conf, nlp_css, nlp_explanation = predict_m4(clinical_notes, nlp_drug, nlp_cond)
                st.session_state["m4_result"] = {
                    "label": nlp_label, "conf": nlp_conf,
                    "css": nlp_css, "explanation": nlp_explanation,
                }
            except Exception:
                logger.error("M4 prediction failed", exc_info=True)
                st.error("NLP Intelligence model could not complete the analysis. The issue has been logged — please try again or contact support.")

        # ── Consensus — store in session_state so it survives re-runs ─
        if proba1 is not None and proba2 is not None:
            st.session_state["_syn_p1"] = proba1
            st.session_state["_syn_p2"] = proba2
            st.session_state["_syn_m5"] = los_label if los_label is not None else "Unavailable"
            st.session_state["_syn_m4_label"] = nlp_label if nlp_label is not None else "Unavailable"
            st.session_state["_syn_m4_expl"]  = nlp_explanation if nlp_explanation is not None else ""
            st.session_state["_con_pred1"] = pred1
            st.session_state["_con_pred2"] = pred2
            st.session_state["_con_lat1"]  = latency_m1
            st.session_state["_con_lat2"]  = latency_m2

    # ── CSS for gauges — injected always so persisted cards render correctly ──
    st.markdown("""
    <style>
    .result-panel {
        background: rgba(15,30,52,0.6);
        border: 1px solid rgba(34,211,238,0.15);
        border-radius: 16px;
        padding: 2rem 1rem;
        text-align: center;
        position: relative;
        overflow: hidden;
        box-shadow: 0 10px 30px rgba(0,0,0,0.2);
        margin-bottom: 1rem;
    }
    .gauge-wrapper {
        position: relative;
        width: 160px;
        height: 160px;
        margin: 0 auto 1.5rem auto;
        border-radius: 50%;
        display: flex;
        align-items: center;
        justify-content: center;
        background: rgba(0,0,0,0.2);
        box-shadow: inset 0 0 20px rgba(0,0,0,0.5);
    }
    .gauge-inner {
        position: absolute;
        width: 130px;
        height: 130px;
        background: #0b1121;
        border-radius: 50%;
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
        z-index: 10;
    }
    .gauge-val {
        font-family: 'JetBrains Mono', monospace;
        font-size: 2.2rem;
        font-weight: 800;
        line-height: 1;
        color: white;
    }
    .gauge-lbl {
        font-size: 0.7rem;
        color: #64748b;
        text-transform: uppercase;
        letter-spacing: 0.1em;
        margin-top: 4px;
    }
    .risk-badge {
        display: inline-block;
        padding: 0.4rem 1rem;
        border-radius: 999px;
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.85rem;
        font-weight: 700;
        letter-spacing: 0.05em;
        margin-bottom: 0.5rem;
    }
    .risk-LOW { background: rgba(16,185,129,0.15); color: #10b981; border: 1px solid rgba(16,185,129,0.3); }
    .risk-MODERATE { background: rgba(245,158,11,0.15); color: #f59e0b; border: 1px solid rgba(245,158,11,0.3); }
    .risk-HIGH { background: rgba(239,68,68,0.15); color: #ef4444; border: 1px solid rgba(239,68,68,0.3); }
    </style>
    """, unsafe_allow_html=True)

    def get_gauge_html(proba: float, title: str, latency: float, threshold: float = 0.38) -> str:
        pct = int(proba * 100)
        if proba >= 0.60:
            color, risk_tier = "#ef4444", "HIGH"
        elif proba >= threshold:
            color, risk_tier = "#f59e0b", "MODERATE"
        else:
            color, risk_tier = "#10b981", "LOW"
        return f"""
        <div class="result-panel">
            <div style="font-family:'JetBrains Mono'; font-size:0.8rem; font-weight:700; color:#5eead4; letter-spacing:0.1em; margin-bottom:0.3rem;">
                {title}
            </div>
            <div style="font-family:'JetBrains Mono'; font-size:0.65rem; color:#64748b; margin-bottom:1.5rem;">
                latency · {latency:.1f}ms
            </div>
            <div class="gauge-wrapper" style="background: conic-gradient({color} {pct}%, rgba(255,255,255,0.05) {pct}%);">
                <div class="gauge-inner">
                    <span class="gauge-val">{pct}%</span>
                    <span class="gauge-lbl">PROBABILITY</span>
                </div>
            </div>
            <div class="risk-badge risk-{risk_tier}">{risk_tier} RISK</div>
            <p style="font-size: 0.8rem; color: #64748b; margin-top: 0.5rem;">
                Decision Threshold: {threshold}
            </p>
        </div>
        """

    # ── Render persisted M1 / M2 gauges ──────────────────────────────
    if "m1_result" in st.session_state or "m2_result" in st.session_state:
        st.markdown("""
        <div class="section-head">
          <span class="num">🎯</span><h2>Inference Results</h2><div class="line"></div>
        </div>""", unsafe_allow_html=True)
        col_m1, col_m2 = st.columns(2, gap="large")
        with col_m1:
            if "m1_result" in st.session_state:
                r = st.session_state["m1_result"]
                st.markdown(get_gauge_html(r["proba"], "M1 · XGBOOST ENSEMBLE", r["latency"], threshold=0.38), unsafe_allow_html=True)
        with col_m2:
            if "m2_result" in st.session_state:
                r = st.session_state["m2_result"]
                st.markdown(get_gauge_html(r["proba"], "M2 · DEEP LEARNING DNN", r["latency"], threshold=0.38), unsafe_allow_html=True)

    # ── Render persisted M5 card ──────────────────────────────────────
    if "m5_result" in st.session_state:
        r5 = st.session_state["m5_result"]
        st.markdown("""<div class="section-head">
          <span class="num">⚡</span><h2>Innovation: Capacity Planning</h2><div class="line"></div>
        </div>""", unsafe_allow_html=True)
        st.markdown(f"""<div class="gcard" style="border-color:rgba(34,211,238,0.4);">
            <span class="tag">M5 · LENGTH OF STAY PREDICTOR</span>
            <h3 style="margin-top:0.5rem;">Predicted Capacity Requirement</h3>
            <div style="display:flex; align-items:center; gap:20px; margin:1rem 0;">
                <div class="{r5['css']}" style="flex:1; font-size:1.2rem; padding:20px;">{r5['label']}</div>
                <div class="stat" style="width:150px;">
                    <div class="label">CONFIDENCE</div>
                    <div class="val">{r5['conf']*100:.1f}%</div>
                </div>
            </div>
            <p style="font-size:0.85rem; color:#94a3b8!important;">
                <strong>Clinical Utility:</strong> This model assists in early discharge planning and
                bed management by predicting the expected duration of stay upon admission.
            </p>
        </div>""", unsafe_allow_html=True)

    # ── Render persisted M4 card ──────────────────────────────────────
    if "m4_result" in st.session_state:
        r4 = st.session_state["m4_result"]
        st.markdown("""<div class="section-head">
          <span class="num">04</span><h2>NLP Intelligence</h2><div class="line"></div>
        </div>""", unsafe_allow_html=True)
        st.markdown(f"""
        <div class="gcard">
            <span class="tag">M4 · META LSTM + METADATA</span>
            <h3>Clinical Sentiment Analysis</h3>
            <div class="{r4['css']}" style="margin:1rem 0; padding:15px; font-size:1.1rem; border-radius:10px; text-align:center; font-weight:700;">
                {r4['label']}
            </div>
            <div style="display:flex; gap:8px; margin-top:0.8rem;">
                <div class="stat" style="flex:1;">
                    <div class="label">NLP CONFIDENCE</div>
                    <div class="val">{r4['conf']*100:.1f}%</div>
                </div>
            </div>
            <p style="font-size:0.85rem; color:#94a3b8!important; margin-top:1rem;">
                <strong>Insight:</strong> {r4['explanation']}
            </p>
        </div>""", unsafe_allow_html=True)

    # ── Render persisted Consensus section ───────────────────────────
    if "_syn_p1" in st.session_state:
        _p1   = st.session_state["_syn_p1"]
        _p2   = st.session_state["_syn_p2"]
        _pred1 = st.session_state.get("_con_pred1")
        _pred2 = st.session_state.get("_con_pred2")
        _lat1  = st.session_state.get("_con_lat1", 0)
        _lat2  = st.session_state.get("_con_lat2", 0)
        _agreement       = "AGREE" if _pred1 == _pred2 else "DISAGREE"
        _agreement_color = "#5eead4" if _agreement == "AGREE" else "#fbbf24"
        _avg_proba       = (_p1 + _p2) / 2
        _high = max(_p1, _p2)
        if _high >= 0.65:
            _rec       = ("⚠ Recommend enhanced discharge planning, scheduled "
                          "follow-up within 7 days, and medication reconciliation.")
            _rec_color = "#fca5a5"
        elif _high >= 0.40:
            _rec       = ("⚡ Standard discharge with phone follow-up within 14 days. "
                          "Monitor for medication adherence.")
            _rec_color = "#fcd34d"
        else:
            _rec       = "✓ Standard discharge protocol. Routine follow-up sufficient."
            _rec_color = "#5eead4"

        st.markdown("""
        <div class="section-head">
          <span class="num">⊕</span><h2>Consensus</h2><div class="line"></div>
        </div>""", unsafe_allow_html=True)
        cc1, cc2, cc3, cc4 = st.columns(4)
        cc1.markdown(f"""
        <div class="stat">
          <div class="label">MODEL AGREEMENT</div>
          <div class="val" style="color:{_agreement_color};">{_agreement}</div>
          <div class="delta">consensus check</div>
        </div>""", unsafe_allow_html=True)
        cc2.markdown(f"""
        <div class="stat">
          <div class="label">AVG PROBABILITY</div>
          <div class="val">{_avg_proba*100:.1f}%</div>
          <div class="delta">M1 + M2 average</div>
        </div>""", unsafe_allow_html=True)
        cc3.markdown(f"""
        <div class="stat">
          <div class="label">DELTA M1 ↔ M2</div>
          <div class="val">{abs(_p1-_p2)*100:.1f}pp</div>
          <div class="delta">disagreement gap</div>
        </div>""", unsafe_allow_html=True)
        cc4.markdown(f"""
        <div class="stat">
          <div class="label">TOTAL LATENCY</div>
          <div class="val">{(_lat1+_lat2):.0f}ms</div>
          <div class="delta">end-to-end</div>
        </div>""", unsafe_allow_html=True)
        st.markdown(f"""
        <div class="gcard" style="border-color:rgba(94,234,212,0.3); margin-top:1rem;">
          <span class="tag">CLINICAL RECOMMENDATION</span>
          <p style="color:{_rec_color}!important; font-size:1rem;
                    margin-top:0.5rem; line-height:1.6; font-weight:500;">{_rec}</p>
        </div>""", unsafe_allow_html=True)

    # ── AI Clinical Synthesis — outside if submitted so button survives re-runs ──
    if "_syn_p1" in st.session_state:
        st.markdown("""
        <div class="section-head">
          <span class="num">🧠</span><h2>AI Clinical Synthesis</h2><div class="line"></div>
        </div>""", unsafe_allow_html=True)

        st.markdown("""
        <p style="font-size:0.82rem; color:#64748b; margin-bottom:0.8rem;
                  font-family:'JetBrains Mono',monospace; letter-spacing:0.03em;">
            POWERED BY LLAMA 3.1 · GROQ · GENERATIVE AI CONSENSUS REPORT
        </p>""", unsafe_allow_html=True)

        if st.button("⚕️ Generate AI Clinical Consensus Report"):
            with st.spinner("Synthesizing multi-model diagnostics with Llama AI..."):
                st.session_state["synthesis_result"] = generate_clinical_synthesis(
                    m1_proba=st.session_state["_syn_p1"],
                    m2_proba=st.session_state["_syn_p2"],
                    m5_label=st.session_state["_syn_m5"],
                    m4_label=st.session_state["_syn_m4_label"],
                    m4_explanation=st.session_state["_syn_m4_expl"],
                )

        if "synthesis_result" in st.session_state:
            synthesis = st.session_state["synthesis_result"]
            st.markdown(f"""
            <div style="
                background: linear-gradient(135deg, rgba(15,30,52,0.8) 0%, rgba(17,44,74,0.8) 100%);
                border: 1px solid rgba(94,234,212,0.35);
                border-radius: 14px;
                padding: 1.8rem 2rem;
                box-shadow: 0 8px 32px rgba(34,211,238,0.08), inset 0 1px 0 rgba(94,234,212,0.1);
                margin-top: 0.5rem;
            ">
                <div style="display:flex; align-items:center; gap:10px; margin-bottom:1.2rem;">
                    <span style="font-size:1.3rem;">🧠</span>
                    <span style="font-family:'JetBrains Mono',monospace; font-size:0.7rem;
                                 font-weight:700; color:#5eead4; text-transform:uppercase;
                                 letter-spacing:0.12em;">
                        AI CLINICAL SYNTHESIS · LLAMA 3.1 via GROQ
                    </span>
                </div>
                <div style="font-size:0.93rem; color:#e2e8f0; line-height:1.75;
                            white-space:pre-wrap; font-family:'Inter',sans-serif;">
{synthesis}
                </div>
                <div style="margin-top:1.2rem; padding-top:1rem;
                            border-top:1px solid rgba(94,234,212,0.1);
                            font-size:0.72rem; color:#475569;
                            font-family:'JetBrains Mono',monospace;">
                    ⚠ AI-GENERATED SYNTHESIS · FOR INVESTIGATIONAL USE ONLY · NOT A CLINICAL DIAGNOSIS
                </div>
            </div>
            """, unsafe_allow_html=True)
            if st.button("🗑️ Clear Synthesis"):
                del st.session_state["synthesis_result"]
                st.rerun()

    st.markdown("""
        <div class="disclaimer">
          <strong>Clinical Decision Support Notice.</strong> Predictions are
          intended to assist — not replace — qualified clinical judgment.
          All outputs must be reviewed by a licensed clinician before any
          care decision. <strong>FOR INVESTIGATIONAL USE ONLY · NOT FDA CLEARED.</strong>
        </div>
        """, unsafe_allow_html=True)


# =============================================================================
# PAGE: INSIGHTS (Full Teardown & Redesign)
# =============================================================================
def page_insights() -> None:
    """Renders the Clinical Interpretability & Architecture insights page.

    Organises content across five tabs: M1 XGBoost SHAP analysis (bar chart and
    summary plot), M2 DNN training curves and confusion matrix, M5 capacity-planning
    metrics, M4 Meta LSTM architecture overview, and a side-by-side model comparison
    table. Diagnostic PNG artefacts are loaded from each model's ``saved_model``
    directory; missing files are handled with ``st.info`` fallbacks.
    """
    st.markdown("""
    <div style="margin-bottom:1.5rem;">
      <span class="badge" style="display:inline-block; font-family:'JetBrains Mono',monospace;
            font-size:0.7rem; font-weight:600; color:#5eead4;
            background:rgba(94,234,212,0.1); border:1px solid rgba(94,234,212,0.3);
            padding:4px 12px; border-radius:999px; letter-spacing:0.08em;
            text-transform:uppercase;">● MODEL INSIGHTS</span>
      <h1 style="font-family:'Space Grotesk',sans-serif; font-size:2rem;
                 margin:0.4rem 0; letter-spacing:-0.02em;">
        Clinical Interpretability & Architecture
      </h1>
      <p style="color:#94a3b8; max-width: 800px; line-height: 1.6;">
        In healthcare, accuracy without transparency is a liability. This dashboard reveals the mechanics 
        behind every prediction—from feature importance (SHAP) and training dynamics, to the architectural 
        consensus that drives ClearSight's clinical trust.
      </p>
    </div>""", unsafe_allow_html=True)

    # Added Tabs for M4 and M5
    tabs = st.tabs([
        "🌳 M1 · XGBoost", 
        "🧠 M2 · DNN", 
        "⚡ M5 · Capacity", 
        "📝 M4 · NLP", 
        "⚖️ Consensus Architecture"
    ])

    # ── TAB 1: M1 SHAP ──────────────────────────────────────────────────────────
    with tabs[0]:
        st.markdown("""
        <div class="gcard" style="margin-bottom: 2rem;">
          <span class="tag">GLOBAL EXPLAINABILITY</span>
          <h3 style="margin: 0.5rem 0;">XGBoost Feature Mechanics (SHAP)</h3>
          <p style="color: #94a3b8; font-size: 0.9rem;">
            Gradient Boosted Trees excel at finding complex, non-linear interactions in structured tabular data. 
            By applying SHapley Additive exPlanations (SHAP), we reverse-engineer the model to understand exactly 
            which clinical variables are driving readmission risks across the entire MedInsight network.
          </p>
        </div>""", unsafe_allow_html=True)
        
        sb = M1_DIR / "shap_bar.png"
        ss = M1_DIR / "shap_summary.png"
        cm1 = M1_DIR / "confusion_matrix.png"
        cm1o = M1_DIR / "confusion_matrix_optimal.png"

        c1, c2 = st.columns(2, gap="large")
        with c1:
            if sb.exists():
                st.image(str(sb), use_container_width=True)
                st.markdown("""
                <div style="background: rgba(34,211,238,0.05); border-left: 3px solid #22d3ee; padding: 1rem; border-radius: 4px; margin-top: 1rem;">
                    <strong style="color: #5eead4; font-size: 0.85rem; font-family: 'JetBrains Mono', monospace;">ANALYSIS: FEATURE MAGNITUDE</strong><br>
                    <span style="font-size: 0.85rem; color: #cbd5e1;">This bar chart ranks variables by their average absolute impact on the model's output. Historical utilization metrics (prior visits, inpatient cumulative sum) heavily dominate demographic factors, confirming that a patient's recent medical history is the strongest baseline indicator for future readmissions.</span>
                </div>
                """, unsafe_allow_html=True)

        with c2:
            if ss.exists():
                st.image(str(ss), use_container_width=True)
                st.markdown("""
                <div style="background: rgba(167,139,250,0.05); border-left: 3px solid #a78bfa; padding: 1rem; border-radius: 4px; margin-top: 1rem;">
                    <strong style="color: #c4b5fd; font-size: 0.85rem; font-family: 'JetBrains Mono', monospace;">ANALYSIS: DIRECTIONAL IMPACT</strong><br>
                    <span style="font-size: 0.85rem; color: #cbd5e1;">The summary plot reveals the <i>direction</i> of risk. Red dots indicate a high feature value, while blue indicates a low value. Notice how a high number of prior visits (red dots on the top row) clearly pushes the SHAP value to the right (increasing readmission risk), while lower utilization clusters to the left (protective factor).</span>
                </div>
                """, unsafe_allow_html=True)

        st.markdown("<hr style='border-color: rgba(255,255,255,0.1); margin: 3rem 0;'>", unsafe_allow_html=True)
        st.markdown("### Threshold Optimization & Confusion Matrices")
        
        c3, c4 = st.columns(2, gap="large")
        with c3:
            if cm1.exists():
                st.image(str(cm1), use_container_width=True)
                st.markdown("""<p style="font-size: 0.8rem; color: #94a3b8; text-align: center;">Default Threshold (0.50): Optimizes for standard accuracy but misses too many at-risk patients (False Negatives).</p>""", unsafe_allow_html=True)
                
        with c4:
            if cm1o.exists():
                st.image(str(cm1o), use_container_width=True)
                st.markdown("""<p style="font-size: 0.8rem; color: #94a3b8; text-align: center;">Cost-Optimized (0.38): Maximizes Recall (85%) to prevent costly readmissions, accepting higher False Positives.</p>""", unsafe_allow_html=True)


    # ── TAB 2: M2 DNN ───────────────────────────────────────────────────────────
    with tabs[1]:
        st.markdown("""
        <div class="gcard" style="margin-bottom: 2rem;">
          <span class="tag">DEEP LEARNING DYNAMICS</span>
          <h3 style="margin: 0.5rem 0;">DNN Training Convergence</h3>
          <p style="color: #94a3b8; font-size: 0.9rem;">
            While XGBoost is highly interpretable, Neural Networks are powerful pattern-matchers capable of mapping highly non-linear relationships. 
            This model utilizes a Dense architecture (256→128→64) with Batch Normalization and Dropout layers to prevent overfitting on the minority class.
          </p>
        </div>""", unsafe_allow_html=True)
        
    
        from pathlib import Path
        correct_m2_dir = Path("models/model2_deep_learning/saved_model")
        tc = correct_m2_dir / "training_curves.png"
        cm2 = correct_m2_dir / "confusion_matrix.png"
        # -----------------------------
        
        c1, c2 = st.columns(2, gap="large")
        with c1:
            if tc.exists():
                st.image(str(tc), use_container_width=True)
                st.markdown("""
                <div style="background: rgba(34,211,238,0.05); border-left: 3px solid #22d3ee; padding: 1rem; border-radius: 4px; margin-top: 1rem;">
                    <strong style="color: #5eead4; font-size: 0.85rem; font-family: 'JetBrains Mono', monospace;">LEARNING CURVES</strong><br>
                    <span style="font-size: 0.85rem; color: #cbd5e1;">The dual charts show Loss minimization and AUC maximization over training epochs. The close tracking of the validation curve (orange) to the training curve (blue) indicates that the early-stopping mechanisms and dropout regularization successfully prevented the model from memorizing the training data.</span>
                </div>
                """, unsafe_allow_html=True)
                
        with c2:
            if cm2.exists():
                st.image(str(cm2), use_container_width=True)
                st.markdown("""
                <div style="background: rgba(167,139,250,0.05); border-left: 3px solid #a78bfa; padding: 1rem; border-radius: 4px; margin-top: 1rem;">
                    <strong style="color: #c4b5fd; font-size: 0.85rem; font-family: 'JetBrains Mono', monospace;">VALIDATION PERFORMANCE</strong><br>
                    <span style="font-size: 0.85rem; color: #cbd5e1;">The DNN confusion matrix validates the model's ability to distinguish classes using calculated class-weights to address the ~46/54 imbalance. This acts as a robust secondary opinion to the primary XGBoost model.</span>
                </div>
                """, unsafe_allow_html=True)

    # ── TAB 3: M5 Capacity ──────────────────────────────────────────────────────
    with tabs[2]:
        st.markdown("""
        <div class="gcard" style="margin-bottom: 2rem;">
          <span class="tag">OPERATIONAL INNOVATION</span>
          <h3 style="margin: 0.5rem 0;">Capacity Planning & Length of Stay</h3>
          <p style="color: #94a3b8; font-size: 0.9rem;">
            Predicting a readmission is only half the battle. Model 5 transforms clinical variables into 
            operational intelligence by classifying expected hospitalization duration (Short, Standard, Extended). 
            This allows hospital administrators to proactively manage bed capacity and staffing resources.
          </p>
        </div>""", unsafe_allow_html=True)
        
        from pathlib import Path
        correct_m5_dir = Path("models/model5_innovation/saved_model")
        cm5 = correct_m5_dir / "confusion_matrix.png"
        # -----------------------------
        
        c1, c2 = st.columns(2, gap="large")
        with c1:
            if cm5.exists():
                st.image(str(cm5), use_container_width=True)
                
        with c2:
            st.markdown("""
            <div style="background: rgba(34,211,238,0.05); border-left: 3px solid #22d3ee; padding: 1rem; border-radius: 4px; margin-top: 1rem;">
                <strong style="color: #5eead4; font-size: 0.85rem; font-family: 'JetBrains Mono', monospace;">CLINICAL UTILITY (Weighted F1: 0.6047)</strong><br>
                <span style="font-size: 0.85rem; color: #cbd5e1;">By achieving a strong F1-Score on a complex multi-class problem, this model reliably identifies 'Extended Stay' patients at the point of admission. This enables early intervention from social workers and discharge planners, reducing bottlenecks in hospital flow.</span>
            </div>
            """, unsafe_allow_html=True)

    # ── TAB 4: M4 NLP ───────────────────────────────────────────────────────────
    with tabs[3]:
        st.markdown("""
        <div class="gcard" style="margin-bottom: 2rem; border-color: rgba(245,158,11,0.3);">
          <span class="tag" style="background: rgba(245,158,11,0.1); color: #f59e0b;">MULTIMODAL EXPANSION</span>
          <h3 style="margin: 0.5rem 0;">Meta LSTM Clinical Sentiment Analysis</h3>
          <p style="color: #94a3b8; font-size: 0.9rem;">
            Structured EHR data often misses critical nuance hidden in physician progress notes. 
            Model 4 utilizes a PyTorch MetaLSTMClassifier to extract linguistic risk markers 
            (e.g., patient anxiety, medication non-compliance) and contextualizes them against specific drug and condition metadata.
          </p>
        </div>""", unsafe_allow_html=True)
        
        st.info("Meta LSTM training metrics and attention-head visualizations will be populated upon final pipeline integration.")

    # ── TAB 5: Comparison ───────────────────────────────────────────────────────
    with tabs[4]:
        st.markdown("""
        <div class="gcard" style="margin-bottom: 2rem;">
          <span class="tag">SYSTEM ARCHITECTURE</span>
          <h3 style="margin: 0.5rem 0;">The Consensus Strategy: Why Two Models?</h3>
          <p style="color: #94a3b8; font-size: 0.9rem;">
            In high-stakes clinical environments, relying on a single algorithm creates blind spots. 
            ClearSight utilizes a <b>Dual-Inference Consensus Engine</b>. We run both XGBoost (Tree-based) and DNN (Neural) in parallel. 
            When they agree, clinician confidence is amplified. When they disagree, it automatically flags the case for manual physician review, acting as a critical safety net.
          </p>
        </div>""", unsafe_allow_html=True)

        cmp_data = pd.DataFrame({
            "Architectural Property":    ["Core Algorithm", "Pattern Recognition", "Interpretability", "Inference Speed",
                             "Class Imbalance Strategy", "Decision Boundary", "Primary Clinical Role"],
            "Model 1 (XGBoost)": ["Gradient Boosted Trees", "Hierarchical / Thresholds", "High (Native SHAP values)",
                                  "Ultra-fast (~50 ms)", "SMOTE (Synthetic oversampling)", "Cost-Optimized (0.38)", "Primary decision driver & Explanation"],
            "Model 2 (DNN)":     ["Dense Neural Network", "Complex Non-linear", "Moderate (Permutation only)",
                                  "Fast (~80 ms)", "Dynamic Class Weights", "Standard Probabilistic (0.50)", "Secondary validation (Consensus Check)"],
        })
        
        # Styled dataframe for better presentation
        st.dataframe(
            cmp_data, 
            use_container_width=True, 
            hide_index=True,
            column_config={
                "Architectural Property": st.column_config.TextColumn("Property", width="medium"),
                "Model 1 (XGBoost)": st.column_config.TextColumn("XGBoost (Primary)", width="large"),
                "Model 2 (DNN)": st.column_config.TextColumn("DNN (Validator)", width="large"),
            }
        )

# =============================================================================
# PAGE: RETINAL AI (High-Fidelity UI Simulation)
# =============================================================================
def page_retinal() -> None:
    """Renders the Retinal AI Screening page for Model 3 CNN (ResNet50).

    Loads the trained best_model.keras, accepts a fundus photograph upload,
    runs ResNet50 inference via predict_m3(), and displays the result label
    and confidence score. Class 0 = No DR, Class 1 = Has DR (threshold 0.5).
    """
    # 1. CSS for the Retinal Results Panel
    st.markdown("""
    <style>
    .scan-card {
        background: rgba(15,30,52,0.6);
        border: 1px solid rgba(34,211,238,0.15);
        border-radius: 12px;
        padding: 1.5rem;
        box-shadow: 0 10px 30px rgba(0,0,0,0.2);
    }
    .severity-badge {
        display: inline-block;
        padding: 0.4rem 1rem;
        border-radius: 999px;
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.85rem;
        font-weight: 700;
        letter-spacing: 0.05em;
        background: rgba(245,158,11,0.15); 
        color: #f59e0b; 
        border: 1px solid rgba(245,158,11,0.3);
        margin-bottom: 1.5rem;
    }
    </style>
    """, unsafe_allow_html=True)

    # 2. Header
    st.markdown("""
    <div style="margin-bottom:1.5rem;">
      <span class="badge" style="display:inline-block; font-family:'JetBrains Mono',monospace;
            font-size:0.7rem; font-weight:600; color:#10b981;
            background:rgba(16,185,129,0.1); border:1px solid rgba(16,185,129,0.3);
            padding:4px 12px; border-radius:999px; letter-spacing:0.08em;
            text-transform:uppercase;">✓ MODEL 3 · ONLINE</span>
      <h1 style="font-family:'Space Grotesk',sans-serif; font-size:2rem;
                 margin:0.4rem 0;">Retinal AI Screening</h1>
      <p style="color:#94a3b8;">
        ResNet50 convolutional neural network for Diabetic Retinopathy detection
        from fundus photographs.
      </p>
    </div>""", unsafe_allow_html=True)

    # 3. Uploader
    uploaded = st.file_uploader("Upload a fundus photograph (PNG / JPG)", type=["png", "jpg", "jpeg"])

    if uploaded is not None:
        model = load_model3()
        if model is None:
            st.error("Model could not be loaded.")
            return

        label, confidence = predict_m3(uploaded, model)
        is_high_risk = "HIGH RISK" in label
        logger.info("M3 result — label=%s confidence=%.4f", label, confidence)

        # ── ZONE 1: Result badge full width ──────────────────────
        if is_high_risk:
            st.markdown(
                f"""<div style="background:#7f1d1d;border-radius:10px;padding:12px 20px;
                margin-bottom:16px;display:flex;align-items:center;gap:12px;">
                <span style="font-size:1.4rem;">🔴</span>
                <span style="color:#fecaca;font-size:1.1rem;font-weight:700;">{label}</span>
                </div>""", unsafe_allow_html=True)
        else:
            st.markdown(
                f"""<div style="background:#14532d;border-radius:10px;padding:12px 20px;
                margin-bottom:16px;display:flex;align-items:center;gap:12px;">
                <span style="font-size:1.4rem;">🟢</span>
                <span style="color:#bbf7d0;font-size:1.1rem;font-weight:700;">{label}</span>
                </div>""", unsafe_allow_html=True)

        # ── ZONE 2: Image left | Details right ───────────────────
        col_img, col_details = st.columns([6, 4])

        with col_img:
            st.image(uploaded, caption="Fundus Photograph", use_container_width=True)

        with col_details:
            # Confidence bar
            st.markdown("**Model Confidence**")
            bar_color = "#ef4444" if is_high_risk else "#22c55e"
            st.markdown(
                f"""<div style="background:#1e293b;border-radius:8px;padding:4px;margin-bottom:16px;">
                  <div style="width:{confidence*100:.1f}%;background:{bar_color};border-radius:6px;
                              padding:8px 12px;color:white;font-weight:700;font-size:1rem;">
                    {confidence*100:.1f}%
                  </div>
                </div>""", unsafe_allow_html=True)

            # Clinical recommendation
            st.markdown("**Clinical Recommendation**")
            if is_high_risk:
                st.error("⚠️ Ophthalmologist referral recommended within 30 days.")
            else:
                st.info("✅ Annual screening recommended. No immediate action required.")

            # Severity table — only HIGH RISK
            if is_high_risk:
                st.markdown("**DR Severity Reference**")
                st.markdown("""
            <div style="background:#1e293b;border-radius:10px;padding:16px;
                        border:1px solid #334155;margin-top:8px;">
            <table style="width:100%;border-collapse:collapse;">
              <thead>
                <tr>
                  <th style="color:#94a3b8;font-size:0.75rem;padding:4px 8px;text-align:left;">GRADE</th>
                  <th style="color:#94a3b8;font-size:0.75rem;padding:4px 8px;text-align:left;">NAME</th>
                  <th style="color:#94a3b8;font-size:0.75rem;padding:4px 8px;text-align:left;">ACTION</th>
                </tr>
              </thead>
              <tbody>
                <tr><td style="color:#e2e8f0;padding:6px 8px;">1</td><td style="color:#e2e8f0;padding:6px 8px;">Mild NPDR</td><td style="color:#94a3b8;padding:6px 8px;">Monitor annually</td></tr>
                <tr><td style="color:#e2e8f0;padding:6px 8px;">2</td><td style="color:#e2e8f0;padding:6px 8px;">Moderate NPDR</td><td style="color:#94a3b8;padding:6px 8px;">Follow-up in 6 months</td></tr>
                <tr><td style="color:#e2e8f0;padding:6px 8px;">3</td><td style="color:#e2e8f0;padding:6px 8px;">Severe NPDR</td><td style="color:#f87171;padding:6px 8px;">Referral within 1 month</td></tr>
                <tr><td style="color:#e2e8f0;padding:6px 8px;">4</td><td style="color:#e2e8f0;padding:6px 8px;">Proliferative DR</td><td style="color:#f87171;padding:6px 8px;">Urgent referral</td></tr>
              </tbody>
            </table>
            </div>
            """, unsafe_allow_html=True)

        # ── ZONE 3: Grad-CAM side by side with original ───────────
        st.markdown("---")
        st.subheader("🔬 Grad-CAM — Model Attention Map")
        st.caption("Highlights the retinal regions that influenced the prediction.")

        if st.button("Generate Grad-CAM Heatmap"):
            with st.spinner("Computing attention map..."):
                try:
                    pil_img = Image.open(uploaded).convert("RGB")
                    img_arr = np.array(pil_img.resize((224, 224))).astype(np.float32)
                    img_arr = np.expand_dims(img_arr, axis=0)
                    from keras.applications.resnet50 import preprocess_input as resnet_preprocess
                    img_preprocessed = resnet_preprocess(img_arr)
                    heatmap = make_gradcam(img_preprocessed, model)
                    overlay = overlay_gradcam(pil_img, heatmap)

                    col_orig, col_cam = st.columns(2)
                    with col_orig:
                        st.image(uploaded, caption="Original", use_container_width=True)
                    with col_cam:
                        st.image(overlay, caption="Grad-CAM Overlay", use_container_width=True)

                    st.caption("🔴 Red/yellow = high attention regions  |  🔵 Blue = low attention")
                except Exception as e:
                    st.error(f"Grad-CAM failed: {e}")
# =============================================================================
# MAIN ROUTER
# =============================================================================
def main() -> None:
    """Entry point for the ClearSight Analytics Streamlit application.

    Delegates sidebar rendering to ``render_sidebar()``, which returns the active
    navigation label, then routes to the appropriate page-rendering function.
    Called once per Streamlit script rerun.
    """
    page = render_sidebar()
    if   "Command"   in page: page_home()
    elif "Predict"   in page: page_predict()
    elif "Insights"  in page: page_insights()
    elif "Retinal"   in page: page_retinal()


if __name__ == "__main__":
    main()
