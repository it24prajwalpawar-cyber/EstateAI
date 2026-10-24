import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="EstateAI | Intelligent Property Valuation",
    page_icon="◇",
    layout="wide",
    initial_sidebar_state="collapsed"
)


# ============================================================
# PORTABLE FIRST-RUN SETUP
# Downloads the original public dataset if missing and trains the
# same Multiple Linear Regression model if model files are missing.
# The frontend styling/layout below is intentionally unchanged.
# ============================================================

from pathlib import Path
from urllib.request import urlretrieve

PROJECT_DIR = Path(__file__).resolve().parent
RESEARCH_DATASET_PATH = PROJECT_DIR / "dataset" / "train.csv"
MODEL_DIR = PROJECT_DIR / "model"
MODEL_PATH = MODEL_DIR / "estateai_indian_house_model.pkl"
FEATURES_PATH = MODEL_DIR / "feature_names.pkl"
METRICS_PATH = MODEL_DIR / "estateai_model_metrics.json"

def ensure_project_assets():
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    (PROJECT_DIR / "graphs").mkdir(parents=True, exist_ok=True)
    if not RESEARCH_DATASET_PATH.exists():
        st.error("The research dataset dataset/train.csv is missing. Keep it inside the project's dataset folder.")
        st.stop()
    if not MODEL_PATH.exists() or not METRICS_PATH.exists():
        try:
            from train_estateai_model import main as train_estateai_model
            train_estateai_model()
        except Exception as exc:
            st.error(f"EstateAI model training failed: {exc}")
            st.stop()

ensure_project_assets()

# Load the trained pipeline. It includes imputation, scaling, categorical encoding, and target transformation.
model = joblib.load(MODEL_PATH)
with METRICS_PATH.open("r", encoding="utf-8") as metrics_file:
    estateai_metrics = json.load(metrics_file)
feature_names = []

# ============================================================
# ML ANALYTICS DATA
# ============================================================

@st.cache_data
def load_ml_analysis():
    """Build analytics from the Indian housing dataset using its real columns."""
    raw = pd.read_csv(RESEARCH_DATASET_PATH)

    # The Indian dataset does not contain BATHROOM, FLOOR, or PARKING.
    # Use fields that actually exist in train.csv instead.
    numeric_source_columns = [
        "TARGET(PRICE_IN_LACS)", "SQUARE_FT", "BHK_NO.",
        "UNDER_CONSTRUCTION", "RERA", "READY_TO_MOVE", "RESALE"
    ]
    for column in numeric_source_columns:
        raw[column] = pd.to_numeric(raw[column], errors="coerce")

    raw = raw[
        raw["TARGET(PRICE_IN_LACS)"].between(0.01, 1000)
        & raw["SQUARE_FT"].between(100, 10000)
        & raw["BHK_NO."].between(1, 10)
    ].copy()

    # Price is converted from lakh rupees to rupees for the existing analytics.
    df = pd.DataFrame({
        "price": raw["TARGET(PRICE_IN_LACS)"] * 100000,
        "area": raw["SQUARE_FT"],
        "bedrooms": raw["BHK_NO."],
        "under_construction": raw["UNDER_CONSTRUCTION"],
        "rera": raw["RERA"],
        "ready_to_move": raw["READY_TO_MOVE"],
        "resale": raw["RESALE"],
    })
    df = df.dropna()
    X = df.drop("price", axis=1)
    y = df["price"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    analysis_model = LinearRegression()
    analysis_model.fit(X_train, y_train)
    y_pred = analysis_model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    return df, X, y_test, y_pred, mae, rmse, r2

(
    analysis_df,
    analysis_X,
    analysis_y_test,
    analysis_y_pred,
    analysis_mae,
    analysis_rmse,
    analysis_r2
) = load_ml_analysis()

# ============================================================
# HTML RENDERING HELPER
#
# IMPORTANT:
# This removes indentation from HTML before giving it to
# Streamlit. This prevents HTML from appearing as plain code.
# ============================================================

def render_html(content):

    cleaned_lines = []

    for line in content.splitlines():
        cleaned_lines.append(line.lstrip())

    cleaned_html = "\n".join(cleaned_lines)

    st.markdown(
        cleaned_html,
        unsafe_allow_html=True
    )


# ============================================================
# PREMIUM LUXURY CSS
# ============================================================

st.markdown(
"""
<style>

/* =========================================================
   IMPORT FONT
   ========================================================= */

@import url(
'https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Playfair+Display:wght@500;600;700&display=swap'
);


/* =========================================================
   COLOR SYSTEM

   Obsidian       #08090C
   Surface        #101216
   Surface 2      #15171D
   Gold           #C9A86A
   Gold Light     #E5C98B
   Ivory          #F4F0E8
   Slate          #9297A3
   Emerald        #5FC39A
   ========================================================= */


/* =========================================================
   GLOBAL
   ========================================================= */

html,
body,
[class*="css"] {

    font-family: "DM Sans", sans-serif;

}


.stApp {

    background:

        radial-gradient(
            ellipse at 15% 0%,
            rgba(201, 168, 106, 0.065),
            transparent 32%
        ),

        radial-gradient(
            ellipse at 90% 20%,
            rgba(94, 195, 154, 0.035),
            transparent 30%
        ),

        #08090C;

    color: #F4F0E8;

}


.block-container {

    max-width: 1320px;

    padding-top: 28px;

    padding-bottom: 80px;

}


/* =========================================================
   HIDE STREAMLIT DEFAULT ELEMENTS
   ========================================================= */

#MainMenu {
    visibility: hidden;
}

footer {
    visibility: hidden;
}

header[data-testid="stHeader"] {

    background:
        rgba(8, 9, 12, 0.88);

    backdrop-filter:
        blur(18px);

}


/* =========================================================
   TOP NAVIGATION
   ========================================================= */

.top-nav {

    display: flex;

    align-items: center;

    justify-content: space-between;

    padding:
        13px 18px;

    margin-bottom: 30px;

    background:
        rgba(16, 18, 22, 0.78);

    border:
        1px solid rgba(255,255,255,0.065);

    border-radius:
        14px;

    backdrop-filter:
        blur(20px);

}


.brand {

    display: flex;

    align-items: center;

    gap: 11px;

}


.brand-mark {

    width: 33px;

    height: 33px;

    display: flex;

    align-items: center;

    justify-content: center;

    border-radius: 9px;

    background:

        linear-gradient(
            135deg,
            #DCC18A,
            #A98248
        );

    color:
        #111;

    font-family:
        "Playfair Display",
        serif;

    font-size: 18px;

    font-weight: 700;

    box-shadow:
        0 4px 15px rgba(201,168,106,0.18);

}


.brand-name {

    color:
        #F4F0E8;

    font-size:
        14px;

    font-weight:
        700;

    letter-spacing:
        0.3px;

}


.brand-tag {

    color:
        #727782;

    font-size:
        10px;

    letter-spacing:
        1.5px;

    text-transform:
        uppercase;

}


.nav-status {

    display: flex;

    align-items: center;

    gap: 8px;

    color:
        #777C87;

    font-size:
        10px;

    letter-spacing:
        1.2px;

    text-transform:
        uppercase;

}


.status-dot {

    width:
        6px;

    height:
        6px;

    border-radius:
        50%;

    background:
        #5FC39A;

    box-shadow:
        0 0 10px rgba(95,195,154,0.65);

}


/* =========================================================
   HERO
   ========================================================= */

.hero {

    position: relative;

    overflow: hidden;

    min-height:
        515px;

    padding:
        72px 72px 68px;

    border-radius:
        28px;

    margin-bottom:
        70px;

    background:

        linear-gradient(
            110deg,
            #101216 0%,
            #0D0F13 52%,
            #11120F 100%
        );

    border:
        1px solid rgba(255,255,255,0.075);

    box-shadow:

        0 35px 90px rgba(0,0,0,0.42),

        inset 0 1px 0
        rgba(255,255,255,0.035);

}


/* Fine luxury grid */

.hero::before {

    content: "";

    position: absolute;

    inset: 0;

    background-image:

        linear-gradient(
            rgba(201,168,106,0.025) 1px,
            transparent 1px
        ),

        linear-gradient(
            90deg,
            rgba(201,168,106,0.025) 1px,
            transparent 1px
        );

    background-size:
        55px 55px;

    mask-image:
        linear-gradient(
            to bottom right,
            black,
            transparent 70%
        );

}


/* Gold atmospheric light */

.hero::after {

    content: "";

    position: absolute;

    width:
        480px;

    height:
        480px;

    right:
        -170px;

    top:
        -180px;

    border-radius:
        50%;

    background:
        rgba(201,168,106,0.075);

    filter:
        blur(5px);

}


.hero-content {

    position:
        relative;

    z-index:
        2;

    max-width:
        850px;

}


.eyebrow {

    display:
        inline-flex;

    align-items:
        center;

    gap:
        9px;

    color:
        #C9A86A;

    font-size:
        10px;

    font-weight:
        700;

    letter-spacing:
        2.5px;

    text-transform:
        uppercase;

}


.eyebrow-line {

    width:
        25px;

    height:
        1px;

    background:
        #C9A86A;

}


.hero-title {

    margin-top:
        22px;

    color:
        #F4F0E8;

    font-family:
        "Playfair Display",
        serif;

    font-size:
        clamp(48px, 6vw, 78px);

    line-height:
        1.02;

    font-weight:
        600;

    letter-spacing:
        -3px;

}


.hero-title em {

    color:
        #C9A86A;

    font-style:
        normal;

}


.hero-description {

    max-width:
        670px;

    margin-top:
        24px;

    color:
        #898E99;

    font-size:
        15px;

    line-height:
        1.9;

}


.hero-bottom {

    display:
        flex;

    flex-wrap:
        wrap;

    align-items:
        center;

    gap:
        10px;

    margin-top:
        42px;

}


.hero-chip {

    padding:
        9px 14px;

    color:
        #AEB2BA;

    background:
        rgba(255,255,255,0.035);

    border:
        1px solid rgba(255,255,255,0.07);

    border-radius:
        999px;

    font-size:
        11px;

}


/* =========================================================
   SECTION HEADER
   ========================================================= */

.section-eyebrow {

    color:
        #C9A86A;

    font-size:
        10px;

    font-weight:
        700;

    letter-spacing:
        2.3px;

    text-transform:
        uppercase;

    margin-bottom:
        10px;

}


.section-title {

    color:
        #F4F0E8;

    font-family:
        "Playfair Display",
        serif;

    font-size:
        34px;

    font-weight:
        600;

    letter-spacing:
        -0.8px;

    margin-bottom:
        7px;

}


.section-description {

    color:
        #747984;

    font-size:
        13px;

    line-height:
        1.7;

    margin-bottom:
        30px;

}


/* =========================================================
   INPUT PANEL
   ========================================================= */

.panel {

    background:
        linear-gradient(
            145deg,
            rgba(18,20,25,0.96),
            rgba(12,14,18,0.96)
        );

    border:
        1px solid rgba(255,255,255,0.07);

    border-radius:
        21px;

    padding:
        28px;

    margin-bottom:
        22px;

    box-shadow:
        0 20px 50px rgba(0,0,0,0.22);

}


.panel-heading {

    display:
        flex;

    align-items:
        center;

    gap:
        13px;

    margin-bottom:
        25px;

}


.panel-icon {

    width:
        38px;

    height:
        38px;

    display:
        flex;

    align-items:
        center;

    justify-content:
        center;

    border-radius:
        10px;

    background:
        rgba(201,168,106,0.08);

    border:
        1px solid rgba(201,168,106,0.15);

    color:
        #D7B878;

    font-size:
        16px;

}


.panel-title {

    color:
        #E9E5DD;

    font-size:
        15px;

    font-weight:
        700;

}


.panel-subtitle {

    color:
        #676C76;

    font-size:
        11px;

    margin-top:
        3px;

}


/* =========================================================
   STREAMLIT INPUTS
   ========================================================= */

label {

    color:
        #888D97 !important;

    font-size:
        11px !important;

    font-weight:
        600 !important;

}


div[data-baseweb="input"] {

    background:
        #0B0D11 !important;

    border:
        1px solid #252932 !important;

    border-radius:
        10px !important;

}


div[data-baseweb="input"]:focus-within {

    border-color:
        rgba(201,168,106,0.55) !important;

    box-shadow:
        0 0 0 1px rgba(201,168,106,0.15) !important;

}


div[data-baseweb="input"] input {

    color:
        #E8E4DC !important;

}


div[data-baseweb="select"] > div {

    background:
        #0B0D11 !important;

    border:
        1px solid #252932 !important;

    border-radius:
        10px !important;

}


div[data-baseweb="select"] > div:focus-within {

    border-color:
        rgba(201,168,106,0.55) !important;

}


div[data-baseweb="select"] span {

    color:
        #E8E4DC !important;

}


/* =========================================================
   PREDICTION BUTTON
   ========================================================= */

.stButton > button {

    height:
        61px;

    border-radius:
        12px;

    border:
        1px solid rgba(226,197,137,0.35);

    background:

        linear-gradient(
            135deg,
            #C8A76A,
            #A98248
        );

    color:
        #11120F;

    font-size:
        13px;

    font-weight:
        800;

    letter-spacing:
        1.1px;

    box-shadow:

        0 12px 30px
        rgba(201,168,106,0.16);

    transition:
        all 0.25s ease;

}


.stButton > button:hover {

    transform:
        translateY(-2px);

    background:

        linear-gradient(
            135deg,
            #E1C687,
            #BC9659
        );

    border-color:
        rgba(255,232,176,0.65);

    box-shadow:

        0 16px 35px
        rgba(201,168,106,0.24);

}


/* =========================================================
   RESULT
   ========================================================= */

.result {

    position:
        relative;

    overflow:
        hidden;

    padding:
        48px;

    margin-top:
        30px;

    margin-bottom:
        25px;

    border-radius:
        23px;

    background:

        linear-gradient(
            135deg,
            #11130F 0%,
            #161711 55%,
            #101410 100%
        );

    border:
        1px solid rgba(201,168,106,0.20);

    box-shadow:

        0 25px 70px rgba(0,0,0,0.40),

        inset 0 1px 0
        rgba(255,255,255,0.035);

}


.result::after {

    content:
        "";

    position:
        absolute;

    width:
        330px;

    height:
        330px;

    right:
        -150px;

    top:
        -160px;

    border-radius:
        50%;

    background:
        rgba(201,168,106,0.08);

    filter:
        blur(4px);

}


.result-content {

    position:
        relative;

    z-index:
        2;

}


.result-eyebrow {

    color:
        #C9A86A;

    font-size:
        10px;

    font-weight:
        800;

    letter-spacing:
        2px;

}


.result-price {

    margin-top:
        12px;

    color:
        #F6F0E4;

    font-family:
        "Playfair Display",
        serif;

    font-size:
        clamp(42px, 5vw, 65px);

    line-height:
        1;

    font-weight:
        600;

    letter-spacing:
        -1.8px;

}


.result-lakh {

    margin-top:
        12px;

    color:
        #9A9D96;

    font-size:
        14px;

}


.result-note {

    margin-top:
        20px;

    color:
        #62675F;

    font-size:
        11px;

}


/* =========================================================
   SUMMARY CARDS
   ========================================================= */

.summary-card {

    min-height:
        125px;

    padding:
        20px;

    background:
        #101216;

    border:
        1px solid rgba(255,255,255,0.065);

    border-radius:
        16px;

    transition:
        all 0.25s ease;

}


.summary-card:hover {

    transform:
        translateY(-3px);

    border-color:
        rgba(201,168,106,0.22);

}


.summary-icon {

    color:
        #C9A86A;

    font-size:
        18px;

}


.summary-label {

    color:
        #646973;

    font-size:
        9px;

    font-weight:
        700;

    letter-spacing:
        1.4px;

    text-transform:
        uppercase;

    margin-top:
        14px;

}


.summary-value {

    color:
        #E9E5DD;

    font-size:
        21px;

    font-weight:
        700;

    margin-top:
        4px;

}


/* =========================================================
   WORKFLOW
   ========================================================= */

.workflow {

    min-height:
        205px;

    padding:
        25px;

    background:
        #0E1014;

    border:
        1px solid rgba(255,255,255,0.065);

    border-radius:
        17px;

    transition:
        all 0.3s ease;

}


.workflow:hover {

    transform:
        translateY(-5px);

    border-color:
        rgba(201,168,106,0.23);

    box-shadow:
        0 18px 40px rgba(0,0,0,0.25);

}


.workflow-number {

    color:
        #C9A86A;

    font-family:
        "Playfair Display",
        serif;

    font-size:
        28px;

}


.workflow-title {

    color:
        #E7E3DB;

    font-size:
        14px;

    font-weight:
        700;

    margin-top:
        17px;

}


.workflow-text {

    color:
        #666B75;

    font-size:
        11px;

    line-height:
        1.75;

    margin-top:
        8px;

}


/* =========================================================
   MODEL SECTION
   ========================================================= */

.model-card {

    min-height:
        315px;

    padding:
        32px;

    background:
        #101216;

    border:
        1px solid rgba(255,255,255,0.065);

    border-radius:
        20px;

}


.model-name {

    color:
        #F0ECE4;

    font-family:
        "Playfair Display",
        serif;

    font-size:
        27px;

    font-weight:
        600;

}


.model-line {

    width:
        42px;

    height:
        1px;

    margin:
        17px 0;

    background:
        #C9A86A;

}


.model-description {

    color:
        #7E838D;

    font-size:
        13px;

    line-height:
        1.9;

}


.model-description b {

    color:
        #C9A86A;

}


.model-info {

    min-height:
        315px;

    padding:
        30px;

    background:

        linear-gradient(
            145deg,
            #12151A,
            #0D0F13
        );

    border:
        1px solid rgba(201,168,106,0.13);

    border-radius:
        20px;

}


.info-row {

    padding:
        15px 0;

    border-bottom:
        1px solid rgba(255,255,255,0.055);

}


.info-row:last-child {

    border-bottom:
        none;

}


.info-label {

    color:
        #5F646D;

    font-size:
        9px;

    font-weight:
        700;

    letter-spacing:
        1.4px;

    text-transform:
        uppercase;

}


.info-value {

    color:
        #D9D5CD;

    font-size:
        13px;

    margin-top:
        4px;

}


/* =========================================================
   PERFORMANCE CARDS
   ========================================================= */

.performance-card {

    padding:
        27px;

    background:
        #101216;

    border:
        1px solid rgba(255,255,255,0.065);

    border-radius:
        18px;

}


.performance-icon {

    color:
        #C9A86A;

    font-size:
        17px;

}


.performance-label {

    color:
        #626771;

    font-size:
        9px;

    font-weight:
        700;

    letter-spacing:
        1.3px;

    text-transform:
        uppercase;

    margin-top:
        15px;

}


.performance-value {

    color:
        #F0ECE4;

    font-family:
        "Playfair Display",
        serif;

    font-size:
        29px;

    margin-top:
        4px;

}


/* =========================================================
   NATIVE METRIC OVERRIDE
   ========================================================= */

[data-testid="stMetric"] {

    background:
        #101216 !important;

    border:
        1px solid rgba(255,255,255,0.065) !important;

    border-radius:
        16px !important;

    padding:
        18px !important;

}


[data-testid="stMetricLabel"] {

    color:
        #686D76 !important;

}


[data-testid="stMetricValue"] {

    color:
        #E9E5DD !important;

}


/* =========================================================
   EXPANDER
   ========================================================= */

[data-testid="stExpander"] {

    background:
        #0F1115;

    border:
        1px solid rgba(255,255,255,0.065);

    border-radius:
        14px;

}


[data-testid="stExpander"] summary {

    color:
        #AEB1B7 !important;

}


/* =========================================================
   LATEX / CODE AREAS
   ========================================================= */

[data-testid="stExpander"] code {

    color:
        #D5BA80;

}


/* =========================================================
   FOOTER
   ========================================================= */

.footer {

    margin-top:
        80px;

    padding-top:
        28px;

    border-top:
        1px solid rgba(255,255,255,0.055);

    text-align:
        center;

    color:
        #464A52;

    font-size:
        10px;

    line-height:
        2;

    letter-spacing:
        0.4px;

}


.footer-brand {

    color:
        #777B84;

    font-weight:
        700;

    letter-spacing:
        2px;

}


/* =========================================================
   MOBILE
   ========================================================= */

@media (max-width: 800px) {

    .hero {

        padding:
            48px 30px;

        min-height:
            auto;

    }

    .hero-title {

        font-size:
            45px;

        letter-spacing:
            -2px;

    }

    .section-title {

        font-size:
            28px;

    }

    .result {

        padding:
            35px 25px;

    }

}

</style>
""",
unsafe_allow_html=True
)


# ============================================================
# TOP NAVIGATION
# ============================================================

render_html("""
<div class="top-nav">

<div class="brand">

<div class="brand-mark">
E
</div>

<div>

<div class="brand-name">
EstateAI
</div>

<div class="brand-tag">
Intelligent Property Valuation
</div>

</div>

</div>

<div class="nav-status">

<div class="status-dot"></div>

MODEL ONLINE

</div>

</div>
""")


# ============================================================
# HERO
# ============================================================

render_html("""
<div class="hero">

<div class="hero-content">

<div class="eyebrow">

<div class="eyebrow-line"></div>

AI-POWERED PROPERTY VALUATION

</div>

<div class="hero-title">

Know the value.<br>

<span>Before you decide.</span>

</div>

<div class="hero-description">

EstateAI combines historical housing data with
Multiple Linear Regression to generate an intelligent
property valuation based on the characteristics of
your home.

</div>

<div class="hero-bottom">

<div class="hero-chip">
◇ Machine Learning
</div>

<div class="hero-chip">
◇ Data Driven
</div>

<div class="hero-chip">
◇ Instant Valuation
</div>

<div class="hero-chip">
◇ Private Local Model
</div>

</div>

</div>

</div>
""")


# ============================================================
# PROPERTY INPUT HEADER
# ============================================================

render_html("""
<div class="section-eyebrow">
PROPERTY DETAILS
</div>

<div class="section-title">
Tell us about the property
</div>

<div class="section-description">
Provide the characteristics below. The trained model will
use these variables to estimate the property's market value.
</div>
""")


# ============================================================
# SPECIFICATIONS PANEL
# ============================================================

render_html("""
<div class="panel">

<div class="panel-heading">

<div class="panel-icon">
⌂
</div>

<div>

<div class="panel-title">
Property Specifications
</div>

<div class="panel-subtitle">
Physical characteristics of the property
</div>

</div>

</div>

</div>
""")


# ============================================================
# RESEARCH DATASET-ALIGNED PREDICTION FORM
# ============================================================

st.caption("Prediction model trained on the Indian Housing dataset (prices in lakh rupees).")

a1, a2, a3 = st.columns(3, gap="large")
with a1:
    area = st.number_input("Property Area (sq. ft.)", min_value=100.0, max_value=10000.0, value=1000.0, step=50.0)
    bedrooms = st.number_input("BHK / Rooms", min_value=1, max_value=10, value=2, step=1)
    posted_by = st.selectbox("Posted By", ["Owner", "Dealer", "Builder"])
with a2:
    under_construction = st.selectbox("Under Construction?", ["No", "Yes"])
    rera = st.selectbox("RERA Registered?", ["No", "Yes"])
    bhk_or_rk = st.selectbox("Property Type", ["BHK", "RK"])
with a3:
    ready_to_move = st.selectbox("Ready to Move?", ["Yes", "No"])
    resale = st.selectbox("Resale Property?", ["Yes", "No"])
    address = st.text_input("Locality / City", value="Bangalore", help="Enter the locality and city, e.g. Andheri,Mumbai")

predict = st.button("◇   ESTIMATE PROPERTY VALUE", use_container_width=True)

if predict:
    address_clean = address.strip() or "Unknown"
    address_parts = [part.strip() for part in address_clean.split(",") if part.strip()]
    locality = address_parts[0] if address_parts else "Unknown"
    location_field = address_parts[-1] if address_parts else "Unknown"
    bhk_safe = max(int(bedrooms), 1)
    new_house = pd.DataFrame([{
        "UNDER_CONSTRUCTION": 1 if under_construction == "Yes" else 0,
        "RERA": 1 if rera == "Yes" else 0,
        "BHK_NO.": int(bedrooms),
        "SQUARE_FT": float(area),
        "READY_TO_MOVE": 1 if ready_to_move == "Yes" else 0,
        "RESALE": 1 if resale == "Yes" else 0,
        "SQFT_PER_BHK": float(area) / bhk_safe,
        "LOG_SQUARE_FT": float(np.log1p(area)),
        "POSTED_BY": posted_by,
        "BHK_OR_RK": bhk_or_rk,
        "LOCALITY": locality,
        "LOCATION_FIELD": location_field
    }])
    predicted_price_lakh = max(0.0, float(model.predict(new_house)[0]))
    predicted_price_rupees = predicted_price_lakh * 100000

    render_html(f"""
    <div class="result">
      <div class="result-content">
        <div class="result-eyebrow">ESTIMATED PROPERTY VALUE</div>
        <div class="result-price">₹ {predicted_price_rupees:,.0f}</div>
        <div class="result-lakh">Approximately ₹ {predicted_price_lakh:,.2f} Lakh</div>
        <div class="result-note">Estimated using EstateAI's Multiple Linear Regression model trained on the Indian housing dataset.</div>
      </div>
    </div>
    """)
    render_html('<div class="section-eyebrow">PROPERTY SNAPSHOT</div>')
    s1, s2, s3, s4 = st.columns(4, gap="medium")
    summary_items = [
        ("⌂", "Area", f"{area:,.0f} sq. ft."),
        ("□", "BHK / Rooms", str(bedrooms)),
        ("◇", "Posted By", posted_by),
        ("▣", "Location", address_clean)
    ]
    for column, item in zip([s1, s2, s3, s4], summary_items):
        icon, label, value = item
        with column:
            render_html(f'<div class="summary-card"><div class="summary-icon">{icon}</div><div class="summary-label">{label}</div><div class="summary-value">{value}</div></div>')
    with st.expander("◇   View complete property details"):
        st.dataframe(pd.DataFrame({
            "Property Feature": ["Area (sq. ft.)", "BHK / Rooms", "Posted By", "Under Construction", "RERA Registered", "Property Type", "Ready to Move", "Resale", "Locality / City"],
            "Value": [area, bedrooms, posted_by, under_construction, rera, bhk_or_rk, ready_to_move, resale, address_clean]
        }), use_container_width=True, hide_index=True)


# HOW IT WORKS
# ============================================================

st.write("")
st.write("")


render_html("""
<div class="section-eyebrow">
THE PROCESS
</div>

<div class="section-title">
How EstateAI works
</div>

<div class="section-description">
A complete machine learning pipeline transforms property
characteristics into an estimated valuation.
</div>
""")


w1, w2, w3, w4, w5 = st.columns(
    5,
    gap="medium"
)


workflow_data = [

    (
        "01",
        "Property Input",
        "The user enters the physical and categorical characteristics of the property."
    ),

    (
        "02",
        "Preprocessing",
        "Categorical values are transformed into numerical features."
    ),

    (
        "03",
        "Regression Model",
        "The trained model processes the prepared feature vector."
    ),

    (
        "04",
        "Prediction",
        "Learned coefficients are applied to calculate the estimated price."
    ),

    (
        "05",
        "Final Value",
        "The predicted property value is presented instantly."
    )

]


for column, item in zip(
    [w1, w2, w3, w4, w5],
    workflow_data
):

    number, title, description = item

    with column:

        render_html(
            f"""
<div class="workflow">

<div class="workflow-number">
{number}
</div>

<div class="workflow-title">
{title}
</div>

<div class="workflow-text">
{description}
</div>

</div>
"""
        )


# ============================================================
# MODEL SECTION
# ============================================================

st.write("")
st.write("")


render_html("""
<div class="section-eyebrow">
THE INTELLIGENCE
</div>

<div class="section-title">
The model behind the valuation
</div>

<div class="section-description">
A supervised learning model trained on historical housing data.
</div>
""")


model_left, model_right = st.columns(
    [1.7, 1],
    gap="large"
)


with model_left:

    render_html("""
<div class="model-card">

<div class="model-name">
Multiple Linear Regression
</div>

<div class="model-line"></div>

<div class="model-description">

Multiple Linear Regression is a supervised
machine learning algorithm used to predict
a continuous numerical value using multiple
independent variables.

<br><br>

For EstateAI, the target variable is
<b>house price</b>.

<br><br>

During training, the algorithm learns
coefficients from historical property data.
These learned coefficients are then used
to estimate the price of a new property.

</div>

</div>
""")


with model_right:

    render_html("""
<div class="model-info">

<div class="info-row">

<div class="info-label">
Model Type
</div>

<div class="info-value">
Multiple Linear Regression
</div>

</div>

<div class="info-row">

<div class="info-label">
Learning Type
</div>

<div class="info-value">
Supervised Learning
</div>

</div>

<div class="info-row">

<div class="info-label">
Problem Type
</div>

<div class="info-value">
Regression
</div>

</div>

<div class="info-row">

<div class="info-label">
Target Variable
</div>

<div class="info-value">
House Price
</div>

</div>

</div>
""")


# ============================================================
# MATHEMATICAL FORMULA
# ============================================================

st.write("")

with st.expander(
    "◇   View the mathematical foundation"
):

    st.write(
        "The Multiple Linear Regression model can be represented as:"
    )

    st.latex(
        r"""
        \hat{y}
        =
        \beta_0
        +
        \beta_1X_1
        +
        \beta_2X_2
        +
        \cdots
        +
        \beta_nX_n
        """
    )

    st.write(
        "Here, the model learns the coefficients β from "
        "the training dataset and uses them to estimate "
        "the house price."
    )


# ============================================================
# PERFORMANCE
# ============================================================

st.write("")
st.write("")


render_html("""
<div class="section-eyebrow">
MODEL PERFORMANCE
</div>

<div class="section-title">
Measured on unseen data
</div>

<div class="section-description">
The following metrics were calculated using the held-out
test split from the Indian housing dataset. The prediction algorithm remains Multiple Linear Regression, trained on log-transformed prices.
</div>
""")


p1, p2, p3 = st.columns(
    3,
    gap="medium"
)


performance_data = [

    (
        p1,
        "◎",
        "R² SCORE",
        f"{estateai_metrics['test_r2'] * 100:.2f}%"
    ),

    (
        p2,
        "◇",
        "MEAN ABSOLUTE ERROR",
        f"₹{estateai_metrics['test_mae_lakh']:.2f} L"
    ),

    (
        p3,
        "△",
        "ROOT MEAN SQUARE ERROR",
        f"₹{estateai_metrics['test_rmse_lakh']:.2f} L"
    )

]


for column, icon, label, value in performance_data:

    with column:

        render_html(
            f"""
<div class="performance-card">

<div class="performance-icon">
{icon}
</div>

<div class="performance-label">
{label}
</div>

<div class="performance-value">
{value}
</div>

</div>
"""
        )




# ============================================================
# ML VISUAL ANALYTICS
# ============================================================

st.write("")
st.write("")

render_html("""
<div class="section-eyebrow">
MODEL VISUALS
</div>

<div class="section-title">
See how the model behaves
</div>

<div class="section-description">
These visualizations use the same held-out test data and
Multiple Linear Regression pipeline used for model evaluation.
</div>
""")

# ------------------------------------------------------------
# GRAPH 1 — ACTUAL VS PREDICTED
# ------------------------------------------------------------

render_html("""
<div class="panel">
<div class="panel-heading">
<div class="panel-icon">↗</div>
<div>
<div class="panel-title">Actual vs Predicted Prices</div>
<div class="panel-subtitle">
Points closer to the diagonal line represent more accurate predictions.
</div>
</div>
</div>
</div>
""")

fig1, ax1 = plt.subplots(figsize=(10, 5.5))
fig1.patch.set_facecolor("#101216")
ax1.set_facecolor("#101216")

ax1.scatter(
    analysis_y_test,
    analysis_y_pred,
    s=48,
    alpha=0.70,
    color="#C9A86A",
    edgecolors="none"
)

min_value = min(
    analysis_y_test.min(),
    analysis_y_pred.min()
)

max_value = max(
    analysis_y_test.max(),
    analysis_y_pred.max()
)

ax1.plot(
    [min_value, max_value],
    [min_value, max_value],
    color="#F4F0E8",
    linestyle="--",
    linewidth=1.5,
    label="Perfect Prediction"
)

ax1.set_xlabel(
    "Actual Price",
    color="#A8ADB8",
    fontsize=11
)

ax1.set_ylabel(
    "Predicted Price",
    color="#A8ADB8",
    fontsize=11
)

ax1.tick_params(
    colors="#A8ADB8"
)

ax1.grid(
    alpha=0.12
)

for spine in ax1.spines.values():
    spine.set_color("#252830")

legend = ax1.legend(
    facecolor="#101216",
    edgecolor="#252830"
)

for text in legend.get_texts():
    text.set_color("#F4F0E8")

st.pyplot(fig1, use_container_width=True)
plt.close(fig1)

# ------------------------------------------------------------
# GRAPH 2 — RESIDUAL ANALYSIS
# ------------------------------------------------------------

render_html("""
<div class="panel">
<div class="panel-heading">
<div class="panel-icon">△</div>
<div>
<div class="panel-title">Residual Analysis</div>
<div class="panel-subtitle">
Residual = Actual Price − Predicted Price.
</div>
</div>
</div>
</div>
""")

residuals = analysis_y_test - analysis_y_pred

fig2, ax2 = plt.subplots(figsize=(10, 5.5))
fig2.patch.set_facecolor("#101216")
ax2.set_facecolor("#101216")

ax2.scatter(
    analysis_y_pred,
    residuals,
    s=48,
    alpha=0.70,
    color="#C9A86A",
    edgecolors="none"
)

ax2.axhline(
    y=0,
    color="#F4F0E8",
    linestyle="--",
    linewidth=1.5
)

ax2.set_xlabel(
    "Predicted Price",
    color="#A8ADB8",
    fontsize=11
)

ax2.set_ylabel(
    "Residual (Actual − Predicted)",
    color="#A8ADB8",
    fontsize=11
)

ax2.tick_params(
    colors="#A8ADB8"
)

ax2.grid(
    alpha=0.12
)

for spine in ax2.spines.values():
    spine.set_color("#252830")

st.pyplot(fig2, use_container_width=True)
plt.close(fig2)

# ------------------------------------------------------------
# GRAPH 3 — FEATURE RELATIONSHIP WITH PRICE
# ------------------------------------------------------------

render_html("""
<div class="panel">
<div class="panel-heading">
<div class="panel-icon">◇</div>
<div>
<div class="panel-title">Feature Relationship with Price</div>
<div class="panel-subtitle">
Correlation shows the strength of a linear relationship with price.
It does not by itself prove causation.
</div>
</div>
</div>
</div>
""")

numeric_columns = [
    "price",
    "area",
    "bedrooms",
    "under_construction",
    "rera",
    "ready_to_move",
    "resale"
]

correlation = (
    analysis_df[numeric_columns]
    .corr()["price"]
    .drop("price")
    .sort_values()
)

fig3, ax3 = plt.subplots(figsize=(10, 5.2))
fig3.patch.set_facecolor("#101216")
ax3.set_facecolor("#101216")

bars = ax3.barh(
    correlation.index,
    correlation.values,
    color="#C9A86A"
)

ax3.set_xlabel(
    "Correlation with Price",
    color="#A8ADB8",
    fontsize=11
)

ax3.tick_params(
    colors="#A8ADB8"
)

ax3.grid(
    axis="x",
    alpha=0.12
)

for spine in ax3.spines.values():
    spine.set_color("#252830")

for bar, value in zip(bars, correlation.values):
    ax3.text(
        value,
        bar.get_y() + bar.get_height() / 2,
        f" {value:.2f}",
        va="center",
        color="#F4F0E8",
        fontsize=10
    )

st.pyplot(fig3, use_container_width=True)
plt.close(fig3)

# ------------------------------------------------------------
# GRAPH 4 — ACTUAL / PREDICTED / ERROR SUMMARY
# ------------------------------------------------------------

render_html("""
<div class="section-description">
<strong>How to read these visuals:</strong>
Actual vs Predicted shows prediction alignment.
Residual Analysis shows the distribution of errors.
Feature Relationship provides a simple numerical view of
relationships between selected numerical features and price.
</div>
""")


# ============================================================
# DATASET
# ============================================================

st.write("")
st.write("")


render_html("""
<div class="section-eyebrow">
DATASET
</div>

<div class="section-title">
The data behind EstateAI
</div>

<div class="section-description">
EstateAI uses the Indian housing research dataset. Analytics use the available filtered records from train.csv.
</div>
""")


d1, d2, d3 = st.columns(
    3,
    gap="medium"
)


with d1:

    st.metric(
        "Total Properties",
        f"{len(analysis_df):,}"
    )


with d2:

    st.metric(
        "Training Properties",
        f"{len(analysis_df) - len(analysis_y_test):,}"
    )


with d3:

    st.metric(
        "Testing Properties",
        f"{len(analysis_y_test):,}"
    )


# ============================================================
# ABOUT
# ============================================================

st.write("")


with st.expander(
    "◇   About this project"
):

    st.markdown(
"""
### EstateAI — Intelligent House Price Prediction

EstateAI is an AIML semester project demonstrating
a complete supervised machine learning workflow.

**Algorithm**

Multiple Linear Regression

**Learning Type**

Supervised Learning

**Problem Type**

Regression

**Target**

House Price

**Dataset**

Indian housing research dataset (filtered records used for analytics).

**Technology**

Python • Pandas • NumPy • Scikit-learn •
Matplotlib • Seaborn • Streamlit

**Machine Learning Pipeline**

Data Collection → Exploratory Data Analysis →
Preprocessing → Train/Test Split →
Model Training → Evaluation → Prediction →
Interactive Web Application
"""
    )


# ============================================================
# FOOTER
# ============================================================

render_html("""
<div class="footer">

<div class="footer-brand">
ESTATEAI
</div>

Intelligent Property Valuation

<br>

Multiple Linear Regression • AIML Semester Project

<br><br>

Built with Python • Pandas • Scikit-learn • Streamlit

</div>
""")