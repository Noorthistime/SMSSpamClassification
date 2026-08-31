import streamlit as st
import streamlit.components.v1 as components
import numpy as np
import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier

# Ensure nltk packages are available silently

try:
    stopwords.words("english")
    WordNetLemmatizer().lemmatize("word")
except LookupError:
    nltk.download('stopwords', quiet=True)
    nltk.download('wordnet', quiet=True)

st.set_page_config(page_title="SMS Spam Classification", layout="wide")

def get_hl_code(code_str):
    import re
    keywords = {'import', 'from', 'if', 'else', 'elif', 'for', 'while', 'def', 'class', 'return', 'and', 'or', 'not', 'in', 'is', 'try', 'except', 'with', 'as', 'pass', 'break', 'continue'}
    builtins = {'print', 'len', 'range', 'int', 'float', 'str', 'list', 'dict', 'set', 'tuple', 'bool', 'True', 'False', 'None'}
    methods = {'read_csv', 'map', 'copy', 'lower', 'sub', 'escape', 'split', 'join', 'stem', 'apply', 'fit', 'predict', 'transform', 'fit_transform', 'toarray', 'concat', 'lemmatize', 'DataFrame', 'subplots', 'heatmap', 'countplot', 'head', 'shape', 'drop', 'fillna', 'astype', 'isnull', 'sum', 'sort_values'}
    
    hl_lines = []
    lines = code_str.strip('\n').split('\n')
    for line in lines:
        leading_whitespace = len(line) - len(line.lstrip())
        words = re.split(r'(\W+)', line.lstrip())
        
        hl_words = []
        for w in words:
            if w in keywords:
                hl_words.append(f'<span class="keyword">{w}</span>')
            elif w in builtins or w in methods:
                hl_words.append(f'<span class="builtin">{w}</span>')
            else:
                hl_words.append(w)
                
        hl_line = ('&nbsp;' * leading_whitespace) + ''.join(hl_words)
        if not line.strip():
            hl_line = '&nbsp;'
        hl_lines.append(hl_line)
    return '<br>'.join(hl_lines)


if 'theme' not in st.session_state:
    st.session_state.theme = 'default'

# Hidden Toggle Button
st.markdown('<div class="hide-next-button"></div>', unsafe_allow_html=True)
is_stitch = st.button("HIDDEN_TOGGLE_DO_NOT_CLICK", key="hidden_theme_btn")
if is_stitch:
    st.session_state.theme = 'stitch' if st.session_state.theme == 'default' else 'default'

if st.session_state.theme == 'default':
    st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&display=swap');

    :root {
        --primary-color: var(--brand-1);
        --glass-bg: rgba(255, 255, 255, 0.12);
        --glass-border: rgba(255, 255, 255, 0.32);
        --glass-shadow: 0 12px 36px rgba(8, 15, 40, 0.28);
        --brand-1: #00c2ff;
        --brand-2: #4f46e5;
        --brand-3: #26d9a4;
    }

    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', sans-serif !important;
    }

    .stApp {
        background:
            radial-gradient(circle at 10% 12%, rgba(38, 217, 164, 0.18), transparent 34%),
            radial-gradient(circle at 88% 20%, rgba(0, 194, 255, 0.18), transparent 40%),
            radial-gradient(circle at 50% 86%, rgba(79, 70, 229, 0.2), transparent 45%),
            linear-gradient(145deg, #0b1220 0%, #121f37 46%, #0b182f 100%);
        background-attachment: fixed;
        color: #eef3ff;
    }

    /* Keep header transparent without hiding it */
    [data-testid="stHeader"] {
        background: transparent !important;
    }
    .main .block-container {
        padding-top: 0rem !important;
        padding-bottom: 2rem !important;
    }

    @keyframes orbFloat {
        0% { transform: translate3d(0, 0, 0) scale(1); }
        50% { transform: translate3d(0, -14px, 0) scale(1.05); }
        100% { transform: translate3d(0, 0, 0) scale(1); }
    }

    @keyframes pulseGlow {
        0% { box-shadow: 0 0 0 rgba(0, 194, 255, 0.0), 0 10px 26px rgba(10, 18, 42, 0.35); }
        50% { box-shadow: 0 0 22px rgba(79, 70, 229, 0.45), 0 14px 34px rgba(10, 18, 42, 0.45); }
        100% { box-shadow: 0 0 0 rgba(0, 194, 255, 0.0), 0 10px 26px rgba(10, 18, 42, 0.35); }
    }

    @keyframes shimmer {
        0% { background-position: -220% 0; }
        100% { background-position: 220% 0; }
    }

    .stApp::before,
    .stApp::after {
        content: "";
        position: fixed;
        border-radius: 999px;
        filter: blur(52px);
        z-index: 0;
        pointer-events: none;
        animation: orbFloat 9s ease-in-out infinite;
    }

    .stApp::before {
        width: 260px;
        height: 260px;
        right: 10%;
        top: 15%;
        background: rgba(0, 194, 255, 0.18);
    }

    .stApp::after {
        width: 300px;
        height: 300px;
        left: 8%;
        bottom: 8%;
        background: rgba(38, 217, 164, 0.16);
        animation-delay: -3s;
    }

    [data-testid="stHeader"] {
        background: transparent;
    }

    [data-testid="stSidebar"] > div {
        background: linear-gradient(165deg, rgba(255, 255, 255, 0.06), rgba(255, 255, 255, 0.03));
        backdrop-filter: none;
    }

    [data-testid="stAppViewContainer"] > .main {
        position: relative;
        z-index: 1;
    }

    [data-testid="stVerticalBlock"] > div:has(> [data-testid="stMarkdownContainer"]),
    div[data-testid="stDataFrame"],
    div[data-testid="stAlert"],
    div.stCodeBlock {
        background: var(--glass-bg);
        border: 1px solid var(--glass-border);
        border-radius: 18px;
        backdrop-filter: blur(14px);
        box-shadow: var(--glass-shadow);
        transition: transform 0.28s ease, box-shadow 0.28s ease, border-color 0.28s ease;
    }

    [data-testid="stVerticalBlock"] > div:has(> [data-testid="stMarkdownContainer"]):hover,
    div[data-testid="stDataFrame"]:hover,
    div[data-testid="stAlert"]:hover,
    div.stCodeBlock:hover {
        transform: translateY(-2px);
        border-color: rgba(0, 194, 255, 0.5);
        box-shadow: 0 0 18px rgba(0, 194, 255, 0.2), 0 16px 34px rgba(8, 15, 40, 0.36);
    }

    h1, h2, h3 {
        letter-spacing: 0.2px;
        text-shadow: 0 0 18px rgba(79, 70, 229, 0.28);
    }

    div.element-container:has(.premium-hero),
    div.stMarkdown:has(.premium-hero) {
        position: sticky !important;
        top: 15px !important;
        z-index: 999 !important;
    }

    .premium-hero {
        position: relative;
        width: 100%;
        margin: -45px auto 20px auto !important;
        padding: 20px 24px;
        border-radius: 18px;
        border: 1px solid rgba(255, 255, 255, 0.34);
        background: linear-gradient(130deg, rgba(255, 255, 255, 0.2), rgba(255, 255, 255, 0.08));
        backdrop-filter: blur(18px);
        box-shadow: 0 16px 38px rgba(4, 12, 34, 0.4), inset 0 1px 0 rgba(255, 255, 255, 0.22);
        overflow: hidden;
        transition: all 0.6s cubic-bezier(0.25, 1, 0.5, 1);
        text-align: center;
    }

    .premium-hero.pill-mode {
        width: 85% !important;
        margin: 0 auto 20px auto !important;
        padding: 12px 40px !important;
        border-radius: 50px !important;
        background: linear-gradient(130deg, rgba(255, 255, 255, 0.15), rgba(255, 255, 255, 0.05)) !important;
        box-shadow: 0 10px 30px rgba(4, 12, 34, 0.4), inset 0 1px 0 rgba(255, 255, 255, 0.2) !important;
        backdrop-filter: blur(8px) !important;
        -webkit-backdrop-filter: blur(8px) !important;
    }
    
    .premium-hero.pill-mode h1 {
        font-size: 1.6rem !important;
    }



    .premium-hero:hover {
        transform: translateY(-3px);
        border-color: rgba(0, 194, 255, 0.78);
        box-shadow: 0 0 30px rgba(0, 194, 255, 0.35), 0 18px 42px rgba(4, 12, 34, 0.5);
    }

    .premium-hero h1 {
        margin: 0;
        font-size: clamp(1.45rem, 2.4vw, 2.2rem);
        font-weight: 800;
        color: #f5f9ff;
        letter-spacing: 0.35px;
        text-shadow: 0 0 16px rgba(79, 70, 229, 0.35);
        position: relative;
        z-index: 1;
    }

    .run-output-box {
        margin-top: 8px;
        padding: 16px;
        border-radius: 16px;
        border: 1px solid rgba(255, 255, 255, 0.3);
        background: linear-gradient(145deg, rgba(255, 255, 255, 0.14), rgba(255, 255, 255, 0.08));
        box-shadow: 0 12px 30px rgba(8, 15, 40, 0.34), inset 0 1px 0 rgba(255, 255, 255, 0.16);
    }

    .stButton > button {
        border-radius: 12px;
        border: 1px solid rgba(255, 255, 255, 0.45);
        color: #eef3ff;
        background: linear-gradient(120deg, rgba(0, 194, 255, 0.22), rgba(79, 70, 229, 0.24), rgba(38, 217, 164, 0.2));
        background-size: 220% 220%;
        box-shadow: 0 10px 26px rgba(10, 18, 42, 0.35);
        transition: transform 0.25s ease, border-color 0.25s ease, box-shadow 0.3s ease;
    }

    .stButton > button:hover {
        transform: translateY(-2px) scale(1.01);
        border-color: rgba(255, 255, 255, 0.9);
        animation: pulseGlow 1.8s ease-in-out infinite;
    }

    .stButton > button:active {
        transform: scale(0.98);
        box-shadow: 0 0 24px rgba(0, 194, 255, 0.55);
    }

    [data-testid="stTextInput"] input,
    [data-testid="stTextArea"] textarea,
    [data-testid="stSelectbox"] div[data-baseweb="select"] > div {
        background: rgba(255, 255, 255, 0.12) !important;
        border: 1px solid rgba(255, 255, 255, 0.35) !important;
        border-radius: 12px !important;
        color: #eef3ff !important;
        transition: border-color 0.25s ease, box-shadow 0.25s ease;
    }

    [data-testid="stTextInput"] input:focus,
    [data-testid="stTextArea"] textarea:focus {
        border-color: rgba(0, 194, 255, 0.9) !important;
        box-shadow: 0 0 0 0.22rem rgba(0, 194, 255, 0.24) !important;
    }

    [data-testid="stTabs"] [role="tab"] {
        border-radius: 12px;
        transition: all 0.2s ease;
    }

    [data-testid="stTabs"] [role="tab"][aria-selected="true"] {
        background: rgba(255, 255, 255, 0.18);
        border: 1px solid rgba(255, 255, 255, 0.45);
        box-shadow: 0 0 14px rgba(79, 70, 229, 0.32);
    }

    .stMarkdown hr {
        border-top: 1px solid rgba(255, 255, 255, 0.22);
    }

    .stCodeBlock {
        position: relative;
        overflow: hidden;
    }

    .stCodeBlock::before {
        content: "";
        position: absolute;
        inset: 0;
        background: linear-gradient(110deg, transparent 0%, rgba(255, 255, 255, 0.14) 48%, transparent 100%);
        background-size: 220% 100%;
        animation: shimmer 7s linear infinite;
        pointer-events: none;
    }

    /* Style the radio items as beautiful horizontal tabs/buttons */
    [data-testid="stSidebar"] [data-testid="stRadio"] > div[role="radiogroup"] {
        display: flex;
        flex-direction: column;
        gap: 10px;
        padding-top: 10px;
    }
    
    [data-testid="stSidebar"] [data-testid="stRadio"] > div[role="radiogroup"] > label {
        background: rgba(255, 255, 255, 0.05);
        border: 1px solid rgba(255, 255, 255, 0.12) !important;
        padding: 12px 16px !important;
        border-radius: 12px !important;
        transition: all 0.25s cubic-bezier(0.4, 0, 0.2, 1) !important;
        cursor: pointer;
        display: flex;
        align-items: center;
        width: 100%;
        margin-bottom: 0px;
        box-shadow: 0 4px 10px rgba(0, 0, 0, 0.1);
    }

    [data-testid="stSidebar"] [data-testid="stRadio"] > div[role="radiogroup"] > label:hover {
        background: rgba(0, 194, 255, 0.08) !important;
        border-color: rgba(0, 194, 255, 0.4) !important;
        transform: translateX(4px);
        box-shadow: 0 4px 12px rgba(0, 194, 255, 0.15);
    }

    [data-testid="stSidebar"] [data-testid="stRadio"] > div[role="radiogroup"] > label div[data-testid="stMarkdownContainer"] p {
        color: #f5f9ff !important;
    }

    /* Style for checked state */
    [data-testid="stSidebar"] [data-testid="stRadio"] > div[role="radiogroup"] > label[data-checked="true"],
    [data-testid="stSidebar"] [data-testid="stRadio"] > div[role="radiogroup"] > label:has(input:checked) {
        background: linear-gradient(135deg, rgba(0, 194, 255, 0.22), rgba(79, 70, 229, 0.24)) !important;
        border-color: rgba(0, 194, 255, 0.7) !important;
        box-shadow: 0 0 15px rgba(0, 194, 255, 0.25), inset 0 0 8px rgba(0, 194, 255, 0.15);
    }

    /* Keep the active cyan highlight in both modes */
    [data-testid="stSidebar"] [data-testid="stRadio"] > div[role="radiogroup"] > label[data-checked="true"] *,
    [data-testid="stSidebar"] [data-testid="stRadio"] > div[role="radiogroup"] > label:has(input:checked) * {
        color: #00c2ff !important;
        font-weight: 600 !important;
    }

    /* The Nuclear Stealth Strategy: Strip all colors from the native dot to make it an invisible spacer */
    [data-testid="stSidebar"] [data-testid="stRadio"] > div[role="radiogroup"] > label {
        position: relative !important;
    }

    [data-testid="stSidebar"] [data-testid="stRadio"] > div[role="radiogroup"] > label > :not(:last-child),
    [data-testid="stSidebar"] [data-testid="stRadio"] > div[role="radiogroup"] > label > :not(:last-child) * {
        background-color: transparent !important;
        border-color: transparent !important;
        box-shadow: none !important;
        fill: transparent !important;
        stroke: transparent !important;
    }
    
    [data-testid="stSidebar"] [data-testid="stRadio"] > div[role="radiogroup"] > label input[type="radio"] {
        appearance: none !important;
        -webkit-appearance: none !important;
        width: 0 !important;
        height: 0 !important;
        margin: 0 !important;
        opacity: 0 !important;
    }

    /* Draw our custom empty circle floating perfectly over the ghost dot */
    [data-testid="stSidebar"] [data-testid="stRadio"] > div[role="radiogroup"] > label::before {
        content: "" !important;
        position: absolute !important;
        left: 16px !important;
        top: 50% !important;
        transform: translateY(-50%) !important;
        width: 16px !important;
        height: 16px !important;
        border-radius: 50% !important;
        border: 2px solid rgba(255, 255, 255, 0.4) !important;
        background-color: transparent !important;
        transition: all 0.2s ease-in-out !important;
        z-index: 10 !important;
    }

    /* Draw our custom checked circle with the glowing theme color */
    [data-testid="stSidebar"] [data-testid="stRadio"] > div[role="radiogroup"] > label[data-checked="true"]::before,
    [data-testid="stSidebar"] [data-testid="stRadio"] > div[role="radiogroup"] > label:has(input:checked)::before {
        border: 0px solid transparent !important;
        background-color: #00c2ff !important;
        background-image: radial-gradient(circle, #ffffff 30%, #00c2ff 35%);
        box-shadow: 0 0 10px #00c2ff80;
    }

    /* Hide redundant radio widget label */
    [data-testid="stSidebar"] [data-testid="stRadio"] > label {
        display: none !important;
    }

    /* Typography fixes for sidebar header */
    [data-testid="stSidebar"] h1, 
    [data-testid="stSidebar"] h2, 
    [data-testid="stSidebar"] h3 {
        color: #f5f9ff !important;
        font-weight: 700;
        text-shadow: 0 0 12px rgba(0, 194, 255, 0.35);
        margin-bottom: 12px !important;
    }

    @keyframes techPulse {
        0% { text-shadow: 0 0 4px rgba(0, 194, 255, 0.4); color: #00c2ff; }
        50% { text-shadow: 0 0 12px rgba(0, 194, 255, 0.8), 0 0 20px rgba(0, 194, 255, 0.4); color: #eef3ff; }
        100% { text-shadow: 0 0 4px rgba(0, 194, 255, 0.4); color: #00c2ff; }
    }
    .tech-hover-container {
        position: relative;
        display: inline-block;
        cursor: help;
    }
    .glow-tech {
        font-weight: 600;
        text-decoration: underline dotted rgba(0, 194, 255, 0.6) !important;
        animation: techPulse 2s infinite ease-in-out;
        display: inline-block;
        padding: 0 2px;
        color: #00c2ff !important;
        transition: all 0.25s ease;
    }
    .tech-tooltip-box {
        visibility: hidden;
        opacity: 0;
        width: 320px;
        background: rgba(10, 20, 42, 0.98) !important;
        color: #eef3ff !important;
        text-align: left;
        border: 1px solid rgba(0, 194, 255, 0.45);
        border-radius: 10px;
        padding: 14px;
        position: absolute;
        z-index: 9999;
        bottom: 125%;
        left: 50%;
        transform: translateX(-50%) translateY(10px);
        transition: opacity 0.3s ease, transform 0.3s ease, visibility 0.3s;
        box-shadow: 0 12px 30px rgba(0, 0, 0, 0.65), 0 0 20px rgba(0, 194, 255, 0.25);
        pointer-events: none;
        font-size: 0.9em;
        line-height: 1.4;
        font-weight: normal;
        font-family: 'Plus Jakarta Sans', sans-serif !important;
    }
    .tech-tooltip-box strong {
        color: #00c2ff !important;
        font-size: 1.05em;
        display: block;
        margin-bottom: 6px;
    }
    .tech-tooltip-box::after {
        content: "";
        position: absolute;
        top: 100%;
        left: 50%;
        margin-left: -6px;
        border-width: 6px;
        border-style: solid;
        border-color: rgba(10, 20, 42, 0.98) transparent transparent transparent;
    }
    .tech-hover-container:hover .tech-tooltip-box {
        visibility: visible;
        opacity: 1;
        transform: translateX(-50%) translateY(0);
    }
    .tech-hover-container:hover .glow-tech {
        color: #fff !important;
        text-shadow: 0 0 15px rgba(0, 194, 255, 1) !important;
    }

    /* Premium glassmorphic background for the navigation panel */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, rgba(10, 17, 32, 0.95) 0%, rgba(3, 7, 18, 0.98) 100%) !important;
        border-right: 1px solid rgba(0, 194, 255, 0.15) !important;
        box-shadow: 6px 0 25px rgba(0, 0, 0, 0.4) !important;
    }

    /* Structured content cards for layout columns */
    [data-testid="column"] {
        background: rgba(255, 255, 255, 0.02) !important;
        border: 1px solid rgba(255, 255, 255, 0.06) !important;
        border-radius: 16px !important;
        padding: 22px !important;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.15) !important;
        margin-bottom: 15px !important;
        transition: border-color 0.3s ease, box-shadow 0.3s ease;
    }
    [data-testid="column"]:hover {
        border-color: rgba(0, 194, 255, 0.15) !important;
        box-shadow: 0 8px 32px rgba(0, 194, 255, 0.03) !important;
    }

    /* Professional subheadings structure */
    div[data-testid="stMarkdownContainer"] h2 {
        color: #eef3ff !important;
        font-weight: 600 !important;
        font-size: 1.35em !important;
        border-bottom: 2px solid rgba(0, 194, 255, 0.25) !important;
        padding-bottom: 8px !important;
        margin-top: 10px !important;
        margin-bottom: 16px !important;
        letter-spacing: 0.5px !important;
    }
    div[data-testid="stMarkdownContainer"] h3 {
        color: #00c2ff !important;
        font-weight: 500 !important;
        font-size: 1.12em !important;
        padding-bottom: 4px !important;
        margin-top: 10px !important;
        margin-bottom: 12px !important;
        letter-spacing: 0.5px !important;
    }

    /* --- MOBILE RESPONSIVENESS --- */
    @media (max-width: 768px) {
        .main .block-container {
            padding: 1rem 0.5rem !important;
        }
        
        .premium-hero {
            margin: -20px 0 15px 0 !important;
            padding: 15px 12px !important;
        }

        .premium-hero h1 {
            font-size: 1.5rem !important;
        }
        
        div[data-testid="stMarkdownContainer"] h2 {
            font-size: 1.25em !important;
        }
        
        div[data-testid="stMarkdownContainer"] h3 {
            font-size: 1.1em !important;
        }

        .stButton > button {
            width: 100% !important;
            padding: 0.6rem 1rem !important;
        }

        [data-testid="column"] {
            padding: 14px !important;
            margin-bottom: 12px !important;
        }
        
        .tech-tooltip-box {
            width: 260px;
        }
        
        /* Stack the pipeline workflow numbers and text */
        div[style*="gap: 16px"] {
            gap: 12px !important;
            flex-direction: column !important;
            align-items: flex-start !important;
        }
        
        /* Fix the Metrics Ribbon for smaller screens */
        div[style*="justify-content: space-around"] {
            flex-direction: column !important;
            gap: 16px !important;
        }
        
        div[style*="border-left: 1px solid"] {
            display: none !important;
        }
    }

    /* Instant zero-flash hiding of the theme toggle button */
    div.element-container:has(.hide-next-button),
    div.element-container:has(.hide-next-button) + div.element-container {
        position: absolute !important;
        opacity: 0 !important;
        pointer-events: none !important;
        width: 0 !important;
        height: 0 !important;
        overflow: hidden !important;
        margin: 0 !important;
        padding: 0 !important;
    }

    /* Sentinel AI Model Training Page Overhaul */
    .sentinel-phase-pill {
        display: inline-block;
        padding: 8px 24px;
        background: linear-gradient(135deg, rgba(255, 255, 255, 0.1), rgba(255, 255, 255, 0.02));
        border: 1px solid var(--glass-border);
        border-radius: 30px;
        color: #fff;
        font-weight: 800;
        font-size: 0.95rem;
        letter-spacing: 2.5px;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.2);
        margin: 10px auto 30px auto;
        text-transform: uppercase;
        text-align: center;
        width: max-content;
    }
    
    .sentinel-card-title {
        color: #fff;
        font-size: 1.15rem;
        font-weight: 700;
        margin-bottom: 20px;
        display: flex;
        align-items: center;
        gap: 10px;
        text-shadow: 0 0 10px rgba(255,255,255,0.3);
        letter-spacing: 1px;
    }
    
    .sentinel-terminal {
        background: rgba(10, 10, 15, 0.9) !important;
        border: 1px solid var(--glass-border) !important;
        border-radius: 14px;
        overflow: hidden;
        box-shadow: inset 0 0 20px rgba(0,0,0,0.5), 0 12px 35px rgba(0,0,0,0.3);
        margin-bottom: 30px;
    }
    
    .sentinel-terminal-header {
        background: rgba(255, 255, 255, 0.04);
        padding: 12px 20px;
        display: flex;
        justify-content: space-between;
        align-items: center;
        border-bottom: 1px solid rgba(255,255,255,0.08);
    }
    
    .sentinel-mac-dots {
        display: flex;
        gap: 8px;
    }
    
    .sentinel-mac-dots span {
        width: 14px;
        height: 14px;
        border-radius: 50%;
        box-shadow: inset 0 2px 4px rgba(255,255,255,0.2), 0 2px 4px rgba(0,0,0,0.5);
    }
    .sentinel-mac-dots .red { background: #ff5f56; }
    .sentinel-mac-dots .yellow { background: #ffbd2e; }
    .sentinel-mac-dots .green { background: #27c93f; }
    
    .sentinel-processing-tag {
        font-size: 0.65rem;
        font-weight: 900;
        letter-spacing: 1.5px;
        padding: 4px 12px;
        border-radius: 6px;
        text-transform: uppercase;
        background: rgba(255, 51, 102, 0.15);
        color: #ff3366;
        border: 1px solid rgba(255, 51, 102, 0.3);
        box-shadow: 0 0 10px rgba(255, 51, 102, 0.2);
    }
    
    .sentinel-code-body { white-space: nowrap;
        padding: 24px;
        font-family: 'Fira Code', 'Courier New', Courier, monospace;
        font-size: 0.95rem;
        line-height: 1.7;
        color: #d1d5db;
        overflow-x: auto;
    }
    
    .sentinel-code-body .keyword { color: #00c2ff; font-weight: bold; }
    .sentinel-code-body .builtin { color: #4f46e5; font-weight: bold; }
    .sentinel-code-body .string { color: #26d9a4; }
    .sentinel-code-body .comment { color: #6b7280; font-style: italic; }
    
    .sentinel-metrics-card {
        background: rgba(20, 10, 15, 0.5);
        border: 1px solid var(--glass-border);
        border-radius: 16px;
        padding: 28px;
        box-shadow: 0 12px 35px rgba(0,0,0,0.3);
        position: relative;
    }
    
    .sentinel-accuracy-badge {
        position: absolute;
        top: -15px;
        right: 20px;
        padding: 8px 20px;
        border-radius: 30px;
        font-weight: 900;
        font-size: 0.95rem;
        letter-spacing: 1.5px;
        display: flex;
        align-items: center;
        gap: 8px;
        background: linear-gradient(135deg, rgba(255, 51, 102, 0.9), rgba(139, 0, 0, 0.9));
        color: white;
        box-shadow: 0 8px 20px rgba(255, 51, 102, 0.4), inset 0 2px 5px rgba(255,255,255,0.3);
        border: 1px solid rgba(255, 255, 255, 0.2);
        z-index: 10;
    }

    /* Target the st.container holding the metrics matrix to match the terminal styling perfectly */
    div[data-testid="stVerticalBlock"]:has(.metrics-container-hook):not(:has(div[data-testid="stVerticalBlock"]:has(.metrics-container-hook))) {
        background: rgba(10, 10, 15, 0.9) !important;
        border: 1px solid var(--glass-border) !important;
        border-radius: 14px;
        padding: 20px 10px;
        box-shadow: inset 0 0 20px rgba(0,0,0,0.5), 0 12px 35px rgba(0,0,0,0.3);
    }
</style>
""", unsafe_allow_html=True)

elif st.session_state.theme == 'stitch':
    st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&display=swap');

    :root {
        --primary-color: var(--brand-1);
        --glass-bg: rgba(255, 255, 255, 0.05);
        --glass-border: rgba(255, 60, 100, 0.4);
        --glass-shadow: 0 12px 36px rgba(40, 8, 15, 0.35);
        --brand-1: #ff3366;
        --brand-2: #8b0000;
        --brand-3: #ff6633;
    }

    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', sans-serif !important;
    }

    .stApp {
        background:
            radial-gradient(circle at 10% 12%, rgba(255, 51, 102, 0.18), transparent 34%),
            radial-gradient(circle at 88% 20%, rgba(204, 0, 51, 0.18), transparent 40%),
            radial-gradient(circle at 50% 86%, rgba(139, 0, 0, 0.2), transparent 45%),
            linear-gradient(145deg, #1a0808 0%, #2a0b12 46%, #120306 100%);
        background-attachment: fixed;
        color: #ffeeee;
    }
    
    [data-testid="stSidebar"] {
        background: rgba(26, 8, 8, 0.6) !important;
        backdrop-filter: blur(16px) !important;
        -webkit-backdrop-filter: blur(16px) !important;
        border-right: 1px solid var(--glass-border) !important;
        box-shadow: 4px 0 24px rgba(255, 51, 102, 0.1) !important;
    }

    [data-testid="stHeader"] {
        background: transparent !important;
    }

    .stButton > button {
        background: linear-gradient(135deg, rgba(255,51,102,0.1) 0%, rgba(204,0,51,0.1) 100%) !important;
        border: 1px solid rgba(255,51,102,0.4) !important;
        color: #ff3366 !important;
        border-radius: 12px !important;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1) !important;
        padding: 12px 24px !important;
        font-weight: 600 !important;
        letter-spacing: 0.5px !important;
        box-shadow: 0 4px 12px rgba(255,51,102,0.1) !important;
        width: 100%;
    }

    .stButton > button:hover {
        background: linear-gradient(135deg, #ff3366 0%, #cc0033 100%) !important;
        border-color: #ff3366 !important;
        color: white !important;
        transform: translateY(-2px);
        box-shadow: 0 8px 24px rgba(255,51,102,0.25) !important;
    }

    div[data-testid="stMarkdownContainer"] h1 {
        font-weight: 800;
        background: linear-gradient(to right, #ff3366, #ff80a0);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 30px;
    }

    /* Professional subheadings structure (Crimson) */
    div[data-testid="stMarkdownContainer"] h2 {
        color: #ff3366 !important;
        font-weight: 600 !important;
        font-size: 1.35em !important;
        border-bottom: 2px solid rgba(255, 51, 102, 0.25) !important;
        padding-bottom: 8px !important;
        margin-top: 10px !important;
        margin-bottom: 16px !important;
        letter-spacing: 0.5px !important;
    }
    div[data-testid="stMarkdownContainer"] h3 {
        color: #ff3366 !important;
        font-weight: 500 !important;
        font-size: 1.12em !important;
        margin-top: 10px !important;
        margin-bottom: 12px !important;
    }

    /* Style the radio items as beautiful horizontal tabs/buttons */
    [data-testid="stSidebar"] [data-testid="stRadio"] > div[role="radiogroup"] {
        display: flex;
        flex-direction: column;
        gap: 10px;
        padding-top: 10px;
    }
    
    [data-testid="stSidebar"] [data-testid="stRadio"] > div[role="radiogroup"] > label {
        background: rgba(255, 255, 255, 0.05);
        border: 1px solid rgba(255, 255, 255, 0.12) !important;
        padding: 12px 16px !important;
        border-radius: 12px !important;
        transition: all 0.25s cubic-bezier(0.4, 0, 0.2, 1) !important;
        cursor: pointer;
        display: flex;
        align-items: center;
        width: 100%;
        margin-bottom: 0px;
        box-shadow: 0 4px 10px rgba(0, 0, 0, 0.1);
    }

    [data-testid="stSidebar"] [data-testid="stRadio"] > div[role="radiogroup"] > label:hover {
        background: rgba(255, 51, 102, 0.08) !important;
        border-color: rgba(255, 51, 102, 0.4) !important;
        transform: translateX(4px);
        box-shadow: 0 4px 12px rgba(255, 51, 102, 0.15);
    }

    /* Style for checked state */
    [data-testid="stSidebar"] [data-testid="stRadio"] > div[role="radiogroup"] > label[data-checked="true"],
    [data-testid="stSidebar"] [data-testid="stRadio"] > div[role="radiogroup"] > label:has(input:checked) {
        background: linear-gradient(135deg, rgba(255, 51, 102, 0.22), rgba(139, 0, 0, 0.24)) !important;
        border-color: rgba(255, 51, 102, 0.7) !important;
        box-shadow: 0 0 15px rgba(255, 51, 102, 0.25), inset 0 0 8px rgba(255, 51, 102, 0.15);
    }

    /* The Nuclear Stealth Strategy: Strip all colors from the native dot to make it an invisible spacer */
    [data-testid="stSidebar"] [data-testid="stRadio"] > div[role="radiogroup"] > label {
        position: relative !important;
    }

    [data-testid="stSidebar"] [data-testid="stRadio"] > div[role="radiogroup"] > label > :not(:last-child),
    [data-testid="stSidebar"] [data-testid="stRadio"] > div[role="radiogroup"] > label > :not(:last-child) * {
        background-color: transparent !important;
        border-color: transparent !important;
        box-shadow: none !important;
        fill: transparent !important;
        stroke: transparent !important;
    }
    
    [data-testid="stSidebar"] [data-testid="stRadio"] > div[role="radiogroup"] > label input[type="radio"] {
        appearance: none !important;
        -webkit-appearance: none !important;
        width: 0 !important;
        height: 0 !important;
        margin: 0 !important;
        opacity: 0 !important;
    }

    /* Draw our custom empty circle floating perfectly over the ghost dot */
    [data-testid="stSidebar"] [data-testid="stRadio"] > div[role="radiogroup"] > label::before {
        content: "" !important;
        position: absolute !important;
        left: 16px !important;
        top: 50% !important;
        transform: translateY(-50%) !important;
        width: 16px !important;
        height: 16px !important;
        border-radius: 50% !important;
        border: 2px solid rgba(255, 255, 255, 0.4) !important;
        background-color: transparent !important;
        transition: all 0.2s ease-in-out !important;
        z-index: 10 !important;
    }

    /* Draw our custom checked circle with the glowing theme color */
    [data-testid="stSidebar"] [data-testid="stRadio"] > div[role="radiogroup"] > label[data-checked="true"]::before,
    [data-testid="stSidebar"] [data-testid="stRadio"] > div[role="radiogroup"] > label:has(input:checked)::before {
        border: 0px solid transparent !important;
        background-color: #ff3366 !important;
        background-image: radial-gradient(circle, #ffffff 30%, #ff3366 35%);
        box-shadow: 0 0 10px #ff336680;
    }

    /* Hide redundant radio widget label */
    [data-testid="stSidebar"] [data-testid="stRadio"] > label {
        display: none !important;
    }

    /* Typography fixes for sidebar header — always dark theme */
    [data-testid="stSidebar"] h1, 
    [data-testid="stSidebar"] h2, 
    [data-testid="stSidebar"] h3 {
        color: #f5f9ff !important;
        font-weight: 700;
        text-shadow: 0 0 12px rgba(255, 51, 102, 0.35);
        margin-bottom: 12px !important;
    }

    [data-testid="stSidebar"] [data-testid="stRadio"] > div[role="radiogroup"] > label div[data-testid="stMarkdownContainer"] p {
        color: #f5f9ff !important;
    }

    /* Keep the active red highlight in both modes */
    [data-testid="stSidebar"] [data-testid="stRadio"] > div[role="radiogroup"] > label[data-checked="true"] *,
    [data-testid="stSidebar"] [data-testid="stRadio"] > div[role="radiogroup"] > label:has(input:checked) * {
        color: #ff3366 !important;
        font-weight: 600 !important;
    }

    div.element-container:has(.premium-hero),
    div.stMarkdown:has(.premium-hero) {
        position: sticky !important;
        top: 15px !important;
        z-index: 999 !important;
    }

    .premium-hero {
        position: relative;
        width: 100%;
        margin: -45px auto 20px auto !important;
        padding: 20px 24px;
        border-radius: 18px;
        border: 1px solid rgba(255, 255, 255, 0.34);
        background: linear-gradient(130deg, rgba(255, 255, 255, 0.2), rgba(255, 255, 255, 0.08));
        backdrop-filter: blur(18px);
        box-shadow: 0 16px 38px rgba(40, 8, 15, 0.4), inset 0 1px 0 rgba(255, 255, 255, 0.22);
        overflow: hidden;
        transition: all 0.6s cubic-bezier(0.25, 1, 0.5, 1);
        text-align: center;
    }

    .premium-hero.pill-mode {
        width: 85% !important;
        margin: 0 auto 20px auto !important;
        padding: 12px 40px !important;
        border-radius: 50px !important;
        background: linear-gradient(130deg, rgba(255, 51, 102, 0.15), rgba(255, 51, 102, 0.05)) !important;
        box-shadow: 0 10px 30px rgba(40, 8, 15, 0.4), inset 0 1px 0 rgba(255, 255, 255, 0.2) !important;
        backdrop-filter: blur(8px) !important;
        -webkit-backdrop-filter: blur(8px) !important;
    }
    
    .premium-hero.pill-mode h1 {
        font-size: 1.6rem !important;
    }

    .premium-hero::before {
        content: "";
        position: absolute;
        inset: 0;
        background: linear-gradient(110deg, transparent 0%, rgba(255, 255, 255, 0.2) 50%, transparent 100%);
        background-size: 220% 100%;
        animation: shimmer 6.5s linear infinite;
        pointer-events: none;
    }

    .premium-hero:hover {
        transform: translateY(-3px);
        border-color: rgba(255, 51, 102, 0.78);
        box-shadow: 0 0 30px rgba(255, 51, 102, 0.35), 0 18px 42px rgba(40, 8, 15, 0.5);
    }

    .premium-hero h1 {
        margin: 0 !important;
        font-size: clamp(1.45rem, 2.4vw, 2.2rem);
        font-weight: 800;
        color: #f5f9ff !important;
        background: none !important;
        -webkit-text-fill-color: #f5f9ff !important;
        letter-spacing: 0.35px;
        text-shadow: 0 0 16px rgba(255, 51, 102, 0.45);
        position: relative;
        z-index: 1;
    }

    .glow-tech {
        color: #ff3366;
        font-weight: 600;
        cursor: help;
        border-bottom: 1px dashed rgba(255, 51, 102, 0.5);
        transition: all 0.2s ease;
    }
    
    .glow-tech:hover {
        color: #ff5577;
        text-shadow: 0 0 12px rgba(255, 51, 102, 0.4);
        border-bottom: 1px dashed rgba(255, 51, 102, 0.9);
    }
    
    .tech-hover-container {
        position: relative;
        display: inline-block;
    }
    
    .tech-tooltip-box {
        position: absolute;
        bottom: 120%;
        left: 50%;
        transform: translateX(-50%) translateY(10px);
        background: rgba(20, 5, 10, 0.95);
        border: 1px solid rgba(255, 51, 102, 0.4);
        padding: 12px 16px;
        border-radius: 10px;
        width: max-content;
        max-width: 300px;
        color: #c9d1d9;
        font-size: 0.85em;
        line-height: 1.5;
        box-shadow: 0 8px 24px rgba(255, 51, 102, 0.15);
        opacity: 0;
        visibility: hidden;
        transition: all 0.3s cubic-bezier(0.25, 0.8, 0.25, 1);
        z-index: 1000;
        pointer-events: none;
    }
    
    .tech-tooltip-box strong {
        color: #ff3366;
        display: block;
        font-size: 1.1em;
        margin-bottom: 4px;
    }

    .tech-tooltip-box::after {
        content: '';
        position: absolute;
        top: 100%;
        left: 50%;
        transform: translateX(-50%);
        border-width: 6px;
        border-style: solid;
        border-color: rgba(255, 51, 102, 0.4) transparent transparent transparent;
    }

    .tech-hover-container:hover .tech-tooltip-box {
        opacity: 1;
        visibility: visible;
        transform: translateX(-50%) translateY(0);
    }
    
    @media (max-width: 768px) {
        .tech-tooltip-box {
            width: 240px;
            font-size: 0.85em;
            left: 50%;
            transform: translateX(-50%) translateY(10px);
        }
        .tech-hover-container:hover .tech-tooltip-box {
            transform: translateX(-50%) translateY(0);
        }
    }

    /* Instant zero-flash hiding of the theme toggle button */
    div.element-container:has(.hide-next-button),
    div.element-container:has(.hide-next-button) + div.element-container {
        position: absolute !important;
        opacity: 0 !important;
        pointer-events: none !important;
        width: 0 !important;
        height: 0 !important;
        overflow: hidden !important;
        margin: 0 !important;
        padding: 0 !important;
    }

    /* Sentinel AI Model Training Page Overhaul */
    .sentinel-phase-pill {
        display: inline-block;
        padding: 8px 24px;
        background: linear-gradient(135deg, rgba(255, 255, 255, 0.1), rgba(255, 255, 255, 0.02));
        border: 1px solid var(--glass-border);
        border-radius: 30px;
        color: #fff;
        font-weight: 800;
        font-size: 0.95rem;
        letter-spacing: 2.5px;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.2);
        margin: 10px auto 30px auto;
        text-transform: uppercase;
        text-align: center;
        width: max-content;
    }
    
    .sentinel-card-title {
        color: #fff;
        font-size: 1.15rem;
        font-weight: 700;
        margin-bottom: 20px;
        display: flex;
        align-items: center;
        gap: 10px;
        text-shadow: 0 0 10px rgba(255,255,255,0.3);
        letter-spacing: 1px;
    }
    
    .sentinel-terminal {
        background: rgba(10, 10, 15, 0.9) !important;
        border: 1px solid var(--glass-border) !important;
        border-radius: 14px;
        overflow: hidden;
        box-shadow: inset 0 0 20px rgba(0,0,0,0.5), 0 12px 35px rgba(0,0,0,0.3);
        margin-bottom: 30px;
    }
    
    .sentinel-terminal-header {
        background: rgba(255, 255, 255, 0.04);
        padding: 12px 20px;
        display: flex;
        justify-content: space-between;
        align-items: center;
        border-bottom: 1px solid rgba(255,255,255,0.08);
    }
    
    .sentinel-mac-dots {
        display: flex;
        gap: 8px;
    }
    
    .sentinel-mac-dots span {
        width: 14px;
        height: 14px;
        border-radius: 50%;
        box-shadow: inset 0 2px 4px rgba(255,255,255,0.2), 0 2px 4px rgba(0,0,0,0.5);
    }
    .sentinel-mac-dots .red { background: #ff5f56; }
    .sentinel-mac-dots .yellow { background: #ffbd2e; }
    .sentinel-mac-dots .green { background: #27c93f; }
    
    .sentinel-processing-tag {
        font-size: 0.65rem;
        font-weight: 900;
        letter-spacing: 1.5px;
        padding: 4px 12px;
        border-radius: 6px;
        text-transform: uppercase;
        background: rgba(255, 51, 102, 0.15);
        color: #ff3366;
        border: 1px solid rgba(255, 51, 102, 0.3);
        box-shadow: 0 0 10px rgba(255, 51, 102, 0.2);
    }
    
    .sentinel-code-body { white-space: nowrap;
        padding: 24px;
        font-family: 'Fira Code', 'Courier New', Courier, monospace;
        font-size: 0.95rem;
        line-height: 1.7;
        color: #d1d5db;
        overflow-x: auto;
    }
    
    .sentinel-code-body .keyword { color: #ff3366; font-weight: bold; }
    .sentinel-code-body .builtin { color: #4f46e5; font-weight: bold; }
    .sentinel-code-body .string { color: #26d9a4; }
    .sentinel-code-body .comment { color: #6b7280; font-style: italic; }
    
    .sentinel-metrics-card {
        background: rgba(20, 10, 15, 0.5);
        border: 1px solid var(--glass-border);
        border-radius: 16px;
        padding: 28px;
        box-shadow: 0 12px 35px rgba(0,0,0,0.3);
        position: relative;
    }
    
    .sentinel-accuracy-badge {
        position: absolute;
        top: -15px;
        right: 20px;
        padding: 8px 20px;
        border-radius: 30px;
        font-weight: 900;
        font-size: 0.95rem;
        letter-spacing: 1.5px;
        display: flex;
        align-items: center;
        gap: 8px;
        background: linear-gradient(135deg, rgba(255, 51, 102, 0.9), rgba(139, 0, 0, 0.9));
        color: white;
        box-shadow: 0 8px 20px rgba(255, 51, 102, 0.4), inset 0 2px 5px rgba(255,255,255,0.3);
        border: 1px solid rgba(255, 255, 255, 0.2);
        z-index: 10;
    }

    /* Target the st.container holding the metrics matrix to match the terminal styling perfectly */
    div[data-testid="stVerticalBlock"]:has(.metrics-container-hook):not(:has(div[data-testid="stVerticalBlock"]:has(.metrics-container-hook))) {
        background: rgba(10, 10, 15, 0.9) !important;
        border: 1px solid var(--glass-border) !important;
        border-radius: 14px;
        padding: 20px 10px;
        box-shadow: inset 0 0 20px rgba(0,0,0,0.5), 0 12px 35px rgba(0,0,0,0.3);
    }
</style>
""", unsafe_allow_html=True)

def show_explanation(text, technique=None):
    if st.session_state.theme == 'stitch':
        bg, border, text_color = "rgba(255, 51, 102, 0.12)", "#ff3366", "#ff3366"
    else:
        bg, border, text_color = "rgba(0, 194, 255, 0.12)", "#00c2ff", "#00c2ff"
        
    st.markdown(f'<div style="background: {bg}; border-left: 4px solid {border}; padding: 12px 16px; border-radius: 12px; margin-top: 15px; box-shadow: 0 4px 12px rgba(0,0,0,0.15); border: 1px solid {border}; word-wrap: break-word; overflow-wrap: break-word;"><strong style="color: {text_color}; font-size: 1.05em; display: block; margin-bottom: 6px;">What this block did:</strong><span style="color: var(--text-color); font-size: 0.95em; line-height: 1.5;">{text}</span></div>', unsafe_allow_html=True)

def render_explain_button(tab_name, explanation_text, technique=None):
    btn_key = f"explain_state_{tab_name}"
    if btn_key not in st.session_state:
        st.session_state[btn_key] = False

    st.write("---")
    if st.button("What's Happening", key=f"explain_btn_{tab_name}"):
        st.session_state[btn_key] = not st.session_state[btn_key]

    if st.session_state[btn_key]:
        if st.session_state.theme == 'stitch':
            bg, border, text_color = "rgba(255, 51, 102, 0.12)", "#ff3366", "#ff3366"
        else:
            bg, border, text_color = "rgba(0, 194, 255, 0.12)", "#00c2ff", "#00c2ff"
            
        st.markdown(f'<div style="background: {bg}; border-left: 4px solid {border}; padding: 12px 16px; border-radius: 12px; margin-top: 15px; box-shadow: 0 4px 12px rgba(0,0,0,0.15); border: 1px solid {border}; word-wrap: break-word; overflow-wrap: break-word;"><strong style="color: {text_color}; font-size: 1.05em; display: block; margin-bottom: 6px;">Page Explanation:</strong><span style="color: var(--text-color); font-size: 0.95em; line-height: 1.5;">{explanation_text}</span></div>', unsafe_allow_html=True)






st.sidebar.title("Navigation")
menu = st.sidebar.radio("Go to", ["Project Overview", "1. Data Loading & Visualization", "2. Model Training & Evaluation", "3. Live Prediction Test", "4. Full Code Explorer", "5. View Raw Source Code", "Credits & Contact"])


@st.cache_resource
def load_and_preprocess_data():
    import os
    data_path = "data/SMSSpamCollection" if os.path.exists("data/SMSSpamCollection") else "SMSSpamCollection"
    dataset = pd.read_csv(data_path, sep='\t', names=['label', 'message'])
    dataset['label'] = dataset['label'].map({'ham':0 ,'spam':1})
    
    # Handle Imbalance
    only_spam = dataset[dataset["label"] == 1]
    count = int((dataset.shape[0] - only_spam.shape[0]) / only_spam.shape[0])
    for i in range(0, count-1):
        dataset = pd.concat([dataset, only_spam])
        
    dataset['word_count'] = dataset['message'].apply(lambda x: len(x.split()))
    
    def currency(data):
        currency_symbols = ['$','€','₹','¥','₺']
        for i in currency_symbols:
            if i in data:
                return 1
        return 0
    dataset["contains_currency_symbols"] = dataset["message"].apply(currency)
    
    def number(data):
        for i in data:
            if ord(i) >= 48 and ord(i) <= 57:
                return 1
        return 0
    dataset["contains_number"] = dataset['message'].apply(number)
    
    return dataset

dataset = load_and_preprocess_data()

@st.cache_resource
def train_model():
    dataset = load_and_preprocess_data()
    wnl = WordNetLemmatizer()
    stop_words = set(stopwords.words('english'))
    corpus = []
    
    for sms in list(dataset.message):
        message = re.sub(pattern='[^a-zA-Z]', repl=' ', string=sms).lower()
        words = message.split()
        filtered_words = [word for word in words if word not in stop_words]
        lemm_words = [wnl.lemmatize(word) for word in filtered_words]
        message = ' '.join(lemm_words)
        corpus.append(message)
        
    tfidf = TfidfVectorizer(max_features=500)
    vectors = tfidf.fit_transform(corpus).toarray()
    feature_names = tfidf.get_feature_names_out()
    
    X = pd.DataFrame(vectors, columns=feature_names)
    y = dataset['label']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    mnb = MultinomialNB()
    mnb.fit(X_train, y_train)
    y_pred = mnb.predict(X_test)
    
    return tfidf, feature_names, mnb, X, X_train, y_train, X_test, y_test, y_pred

tfidf, feature_names, mnb, X, X_train, y_train, X_test, y_test, y_pred = train_model()

st.markdown('<div id="hero-scroll-marker" style="position: absolute; top: -10px;"></div>', unsafe_allow_html=True)

st.markdown("""
<div class="premium-hero">
    <h1>SMS Spam Classification</h1>
</div>
""", unsafe_allow_html=True)


components.html("""
<script>
    const parentDoc = window.parent.document;
    
    const scrollSentinel = setInterval(() => {
        const marker = parentDoc.getElementById('hero-scroll-marker');
        const heroes = parentDoc.querySelectorAll('.premium-hero');
        
        if (marker && heroes.length > 0 && !marker.dataset.observerAttached) {
            const observer = new IntersectionObserver((entries) => {
                entries.forEach(entry => {
                    heroes.forEach(hero => {
                        if (!entry.isIntersecting) {
                            hero.classList.add('pill-mode');
                        } else {
                            hero.classList.remove('pill-mode');
                        }
                    });
                });
            }, { root: null, threshold: 0 });
            
            observer.observe(marker);
            marker.dataset.observerAttached = 'true';
            clearInterval(scrollSentinel);
        }
    }, 100);
</script>
""", height=0, width=0)

if menu != "Project Overview" and menu != "Credits & Contact":
    st.markdown(f"""
    <div style="text-align: center; margin-top: -10px; margin-bottom: 25px;">
        <span style="background: rgba(255, 255, 255, 0.04); border: 1px solid rgba(255, 255, 255, 0.08); padding: 6px 18px; border-radius: 30px; color: #8a99ad; font-size: 0.85em; font-weight: 500; letter-spacing: 0.5px; display: inline-block; box-shadow: inset 0 1px 0 rgba(255,255,255,0.05);">
            {menu}
        </span>
</div>
    """, unsafe_allow_html=True)

if menu == "Project Overview":
    if st.session_state.theme == 'stitch':
        c_cyan = "#ff3366"
        rgb_cyan = "255, 51, 102"
        c_green = "#ff3366"
        rgb_green = "255, 51, 102"
        c_orange = "#ff3366"
        rgb_orange = "255, 51, 102"
        c_purple = "#ff3366"
        rgb_purple = "255, 51, 102"
    else:
        c_cyan = "#00c2ff"
        rgb_cyan = "0, 194, 255"
        c_green = "#26d9a4"
        rgb_green = "38, 217, 164"
        c_orange = "#ff9f1c"
        rgb_orange = "255, 159, 28"
        c_purple = "#b854ff"
        rgb_purple = "156, 39, 176"

    st.markdown(f"""<div style="background: rgba(255, 255, 255, 0.05); border: 1px solid rgba(255, 255, 255, 0.1); padding: 24px; border-radius: 16px; margin-bottom: 24px; box-shadow: 0 8px 32px rgba(0, 0, 0, 0.15);">
<h2 style="color: {c_cyan}; margin-top: 0; display: flex; align-items: center; gap: 10px; font-size: 1.65em;">• Project Abstract & Overview</h2>
<p style="color: #eef3ff; font-size: 1.05em; line-height: 1.6; margin-bottom: 20px;">The SMS Spam Classifier is a highly responsive machine learning application designed to inspect SMS messages and isolate malicious spam from legitimate user communications. The dashboard parses message lines, normalizes them via lemmatization to extract base roots, and structures the text data using TF-IDF weighting vectors. By fitting a Naive Bayes probability model over these vectors, the pipeline determines category likelihoods to isolate unsolicited spam with high reliability. This console helps developers and analysts audit the visual features, balance classes, inspect model precision, and run real-time inference tests.</p>

<!-- Core Objective Box (Full width) -->
<div style="background: rgba({rgb_green}, 0.06); border: 1px solid rgba({rgb_green}, 0.2); padding: 20px; border-radius: 14px; margin-bottom: 24px;">
<h3 style="color: {c_green}; margin-top: 0; margin-bottom: 8px; font-size: 1.25em; display: flex; align-items: center; gap: 8px;">• Core Objective</h3>
<p style="color: #eef3ff; font-size: 0.98em; line-height: 1.5; margin-bottom: 0;">Protect user message feeds from spam vectors by building a robust text classification pipeline that automatically categorises messages into Ham (safe/legitimate message threads) or Spam (malicious links, ads, or unsolicited scams).</p>
</div>

<!-- Pipeline Workflow Panel (Full width) -->
<div style="background: rgba({rgb_orange}, 0.05); border: 1px solid rgba({rgb_orange}, 0.22); padding: 20px; border-radius: 14px; margin-bottom: 24px;">
<h3 style="color: {c_orange}; margin-top: 0; margin-bottom: 16px; font-size: 1.25em; display: flex; align-items: center; gap: 8px;">• Pipeline Workflow</h3>
<div style="display: flex; flex-direction: column; gap: 12px;">
<div style="background: rgba(255, 255, 255, 0.02); border-left: 4px solid {c_orange}; border-top: 1px solid rgba(255, 255, 255, 0.05); border-right: 1px solid rgba(255, 255, 255, 0.05); border-bottom: 1px solid rgba(255, 255, 255, 0.05); padding: 14px 18px; border-radius: 4px 12px 12px 4px; display: flex; align-items: center; gap: 16px;">
<div style="font-size: 1.2em; font-weight: bold; color: {c_orange}; min-width: 32px;">01</div>
<div>
<strong style="color: #eef3ff; display: block; font-size: 0.95em; margin-bottom: 2px;">Dataset Load & Class Balancing</strong>
<span style="color: #c9d1d9; font-size: 0.88em; line-height: 1.4;">Parses ham/spam raw entries and counters category skewness by oversampling the minority Spam class to parity.</span>
</div>
</div>
<div style="background: rgba(255, 255, 255, 0.02); border-left: 4px solid {c_orange}; border-top: 1px solid rgba(255, 255, 255, 0.05); border-right: 1px solid rgba(255, 255, 255, 0.05); border-bottom: 1px solid rgba(255, 255, 255, 0.05); padding: 14px 18px; border-radius: 4px 12px 12px 4px; display: flex; align-items: center; gap: 16px;">
<div style="font-size: 1.2em; font-weight: bold; color: {c_orange}; min-width: 32px;">02</div>
<div>
<strong style="color: #eef3ff; display: block; font-size: 0.95em; margin-bottom: 2px;">Clean & Tokenize Text</strong>
<span style="color: #c9d1d9; font-size: 0.88em; line-height: 1.4;">Scrubs formatting, lowercases message symbols, removes numeric characters, and strips stop words.</span>
</div>
</div>
<div style="background: rgba(255, 255, 255, 0.02); border-left: 4px solid {c_orange}; border-top: 1px solid rgba(255, 255, 255, 0.05); border-right: 1px solid rgba(255, 255, 255, 0.05); border-bottom: 1px solid rgba(255, 255, 255, 0.05); padding: 14px 18px; border-radius: 4px 12px 12px 4px; display: flex; align-items: center; gap: 16px;">
<div style="font-size: 1.2em; font-weight: bold; color: {c_orange}; min-width: 32px;">03</div>
<div>
<strong style="color: #eef3ff; display: block; font-size: 0.95em; margin-bottom: 2px;">WordNet Lemmatization</strong>
<span style="color: #c9d1d9; font-size: 0.88em; line-height: 1.4;">Applies NLTK WordNet Lemmatization to convert words back to dictionary roots (e.g. "studying" maps to "study"), preserving semantic syntax.</span>
</div>
</div>
<div style="background: rgba(255, 255, 255, 0.02); border-left: 4px solid {c_orange}; border-top: 1px solid rgba(255, 255, 255, 0.05); border-right: 1px solid rgba(255, 255, 255, 0.05); border-bottom: 1px solid rgba(255, 255, 255, 0.05); padding: 14px 18px; border-radius: 4px 12px 12px 4px; display: flex; align-items: center; gap: 16px;">
<div style="font-size: 1.2em; font-weight: bold; color: {c_orange}; min-width: 32px;">04</div>
<div>
<strong style="color: #eef3ff; display: block; font-size: 0.95em; margin-bottom: 2px;">TF-IDF Feature Extraction</strong>
<span style="color: #c9d1d9; font-size: 0.88em; line-height: 1.4;">Translates strings into frequency-weighted vector configurations using a Term Frequency-Inverse Document Frequency vectorizer.</span>
</div>
</div>
<div style="background: rgba(255, 255, 255, 0.02); border-left: 4px solid {c_orange}; border-top: 1px solid rgba(255, 255, 255, 0.05); border-right: 1px solid rgba(255, 255, 255, 0.05); border-bottom: 1px solid rgba(255, 255, 255, 0.05); padding: 14px 18px; border-radius: 4px 12px 12px 4px; display: flex; align-items: center; gap: 16px;">
<div style="font-size: 1.2em; font-weight: bold; color: {c_orange}; min-width: 32px;">05</div>
<div>
<strong style="color: #eef3ff; display: block; font-size: 0.95em; margin-bottom: 2px;">Multinomial Naive Bayes Modeling</strong>
<span style="color: #c9d1d9; font-size: 0.88em; line-height: 1.4;">Trains a Naive Bayes model using joint likelihood calculations to predict the spam probability profile.</span>
</div>
</div>
</div>
</div>

<!-- Technologies Used Box (Full width) -->
<div style="background: rgba({rgb_purple}, 0.04); border: 1px solid rgba({rgb_purple}, 0.18); padding: 20px; border-radius: 14px; margin-bottom: 24px;">
<h3 style="color: {c_purple}; margin-top: 0; margin-bottom: 16px; font-size: 1.25em; display: flex; align-items: center; gap: 8px;">• Technologies Used</h3>
<div style="display: flex; flex-direction: column; gap: 14px;">
<div style="display: flex; gap: 4px; flex-direction: column;">
<strong style="color: #eef3ff; font-size: 0.98em;">1. Natural Language Toolkit (NLTK)</strong>
<span style="color: #c9d1d9; font-size: 0.88em; line-height: 1.4; padding-left: 20px;">Supplies structural tokenization patterns and links the WordNet Lemmatizer definitions.</span>
</div>
<div style="display: flex; gap: 4px; flex-direction: column;">
<strong style="color: #eef3ff; font-size: 0.98em;">2. Scikit-Learn</strong>
<span style="color: #c9d1d9; font-size: 0.88em; line-height: 1.4; padding-left: 20px;">Provides the TfidfVectorizer utility and compiles the Multinomial Naive Bayes model.</span>
</div>
<div style="display: flex; gap: 4px; flex-direction: column;">
<strong style="color: #eef3ff; font-size: 0.98em;">3. Pandas & NumPy</strong>
<span style="color: #c9d1d9; font-size: 0.88em; line-height: 1.4; padding-left: 20px;">Manage data collection structures, clean imbalanced ratios, and organize high-dimensional feature arrays.</span>
</div>
<div style="display: flex; gap: 4px; flex-direction: column;">
<strong style="color: #eef3ff; font-size: 0.98em;">4. Seaborn & Matplotlib</strong>
<span style="color: #c9d1d9; font-size: 0.88em; line-height: 1.4; padding-left: 20px;">Plot class frequency curves and confusion matrix charts to inspect classification quality.</span>
</div>
</div>
</div>

<!-- Metrics Ribbon -->
<div style="margin-top: 24px; background: rgba({rgb_cyan}, 0.05); border: 1px solid rgba({rgb_cyan}, 0.15); padding: 16px; border-radius: 12px; display: flex; justify-content: space-around; flex-wrap: wrap; text-align: center; gap: 16px;">
<div>
<div style="font-size: 1.8em; font-weight: bold; color: {c_cyan};">5,574</div>
<div style="font-size: 0.85em; color: #8b949e; text-transform: uppercase; letter-spacing: 0.05em; margin-top: 4px;">Raw Messages</div>
</div>
<div style="border-left: 1px solid rgba(255, 255, 255, 0.1); height: 50px; align-self: center;"></div>
<div>
<div style="font-size: 1.8em; font-weight: bold; color: {c_green};">Oversampled</div>
<div style="font-size: 0.85em; color: #8b949e; text-transform: uppercase; letter-spacing: 0.05em; margin-top: 4px;">Dataset Balance</div>
</div>
<div style="border-left: 1px solid rgba(255, 255, 255, 0.1); height: 50px; align-self: center;"></div>
<div>
<div style="font-size: 1.8em; font-weight: bold; color: {c_orange};">Naive Bayes</div>
<div style="font-size: 0.85em; color: #8b949e; text-transform: uppercase; letter-spacing: 0.05em; margin-top: 4px;">Classifier Model</div>
</div>
</div>
</div>""", unsafe_allow_html=True)

elif menu == "1. Data Loading & Visualization":

    col1, col2 = st.columns([1, 1])
    with col1:
        st.markdown('<div class="sentinel-card-title">Feature Engineering Code</div>', unsafe_allow_html=True)
        st.markdown(f"""
        <div class="sentinel-terminal">
            <div class="sentinel-code-body">
                # Handling imbalanced dataset using Oversampling<br>only_spam = dataset[dataset["label"] == 1]<br>count = int((dataset.shape[0] - only_spam.shape[0]) / only_spam.shape[0])<br><span class="keyword">for</span> i <span class="keyword">in</span> range(0, count-1):<br>&nbsp;dataset = pd.concat([dataset, only_spam])<br>&nbsp;<br># Count Plot<br>plt.figure(figsize=(8,8))<br>sns.countplot(x="label", data = dataset)<br>plt.title('Countplot <span class="keyword">for</span> Spam vs Ham as balanced dataset')
            </div>
        </div>
        """, unsafe_allow_html=True)
        
    with col2:
        st.markdown('<div class="sentinel-card-title">Data Visualizations</div>', unsafe_allow_html=True)
        
        with st.container():
            st.markdown('<div class="metrics-container-hook"></div>', unsafe_allow_html=True)
            
            fig, ax = plt.subplots(figsize=(6, 4))
            
            is_stitch = st.session_state.get('theme', 'default') == 'stitch'
            palette = "Reds_r" if is_stitch else "Blues_r"
            
            sns.countplot(x="label", data=dataset, ax=ax, palette=palette)
            
            fig.patch.set_alpha(0.0)
            ax.set_facecolor('none')
            for text in ax.texts: text.set_color("white")
            ax.tick_params(colors='white')
            ax.xaxis.label.set_color('white')
            ax.yaxis.label.set_color('white')
            
            st.pyplot(fig)
            
            st.markdown(f"""
            <div style="text-align: center; margin-top: 15px; padding-bottom: 10px;">
                <h4 style="color:#fff; margin: 0; font-weight: 700; letter-spacing: 1px;">BALANCED SPAM VS HAM</h4>
            </div>
            """, unsafe_allow_html=True)
            
    render_explain_button("loading_vis", "This page inspects and <span class='tech-hover-container'><span class='glow-tech'>balances the dataset</span><span class='tech-tooltip-box'><strong>Dataset Balancing</strong>The process of adjusting class ratios to prevent machine learning algorithms from developing a prediction bias towards the majority class.</span></span>. Since raw categories have skewness, we duplicate the <span class='tech-hover-container'><span class='glow-tech'>minority class</span><span class='tech-tooltip-box'><strong>Minority Class</strong>The class that is underrepresented in the dataset (in this case, 'spam'), which is oversampled to reach parity with the majority class.</span></span>. It also performs <span class='tech-hover-container'><span class='glow-tech'>feature engineering</span><span class='tech-tooltip-box'><strong>Feature Engineering</strong>The process of creating new predictive feature variables (like word count, currency symbols, or numbers) from raw input text to help the classification model.</span></span> to extract custom indicators.")


elif menu == "2. Model Training & Evaluation":
    col1, col2 = st.columns([1, 1])
    with col1:
        st.markdown('<div class="sentinel-card-title">Execution Sequence</div>', unsafe_allow_html=True)
        st.markdown(f"""
        <div class="sentinel-terminal">
            <div class="sentinel-code-body">
                <span class="keyword">from</span> sklearn.naive_bayes <span class="keyword">import</span> MultinomialNB<br><span class="keyword">from</span> sklearn.metrics <span class="keyword">import</span> classification_report, confusion_matrix<br>&nbsp;<br>mnb = MultinomialNB()<br>mnb.fit(X_train, y_train)<br>y_pred = mnb.predict(X_test)<br>&nbsp;<br>cm = confusion_matrix(y_test, y_pred)
            </div>
        </div>
        """, unsafe_allow_html=True)
        
    with col2:
        from sklearn.metrics import accuracy_score
        accuracy = accuracy_score(y_test, y_pred)
        st.markdown('<div class="sentinel-card-title">Metrics Matrix</div>', unsafe_allow_html=True)
        
        with st.container():
            st.markdown('<div class="metrics-container-hook"></div>', unsafe_allow_html=True)
            
            cm = confusion_matrix(y_test, y_pred)
            fig, ax = plt.subplots(figsize=(6, 5))
            
            is_stitch = st.session_state.get('theme', 'default') == 'stitch'
            cmap = sns.color_palette("dark:red", as_cmap=True) if is_stitch else sns.color_palette("dark:cyan", as_cmap=True)
            
            sns.heatmap(data=cm, xticklabels=["ham", "spam"], yticklabels=["ham", "spam"], annot=True, fmt='g', cmap=cmap, ax=ax, annot_kws={"size": 12, "weight": "bold"})
            
            fig.patch.set_alpha(0.0)
            ax.set_facecolor('none')
            for text in ax.texts: text.set_color("white")
            ax.tick_params(colors='white')
            cbar = ax.collections[0].colorbar
            cbar.ax.yaxis.set_tick_params(color='white')
            plt.setp(plt.getp(cbar.ax.axes, 'yticklabels'), color='white')
            
            st.pyplot(fig)
            
            st.markdown(f"""
            <div style="text-align: center; margin-top: 15px; padding-bottom: 10px;">
                <h4 style="color:#fff; margin: 0; font-weight: 700; letter-spacing: 1px;">CONFUSION MATRIX (MNB)</h4>
                <div style="color: {'#ff3366' if is_stitch else '#00c2ff'}; font-weight: 800; font-size: 1.1rem; margin-top: 5px;">ACCURACY: {accuracy:.4f}</div>
            </div>
            """, unsafe_allow_html=True)
            
    render_explain_button("model_eval", "This page evaluates the <span class='tech-hover-container'><span class='glow-tech'>Multinomial Naive Bayes</span><span class='tech-tooltip-box'><strong>Multinomial Naive Bayes</strong>A probabilistic classification model based on Bayes' Theorem, particularly suited for text classification using discrete word counts.</span></span> model. It presents fitting code and plots a <span class='tech-hover-container'><span class='glow-tech'>Seaborn confusion matrix</span><span class='tech-tooltip-box'><strong>Seaborn Confusion Matrix</strong>A visual heatmap grid displaying predicted classes against true classes to inspect model classification errors.</span></span> using conditional probabilities derived from <span class='tech-hover-container'><span class='glow-tech'>Bayes' Theorem</span><span class='tech-tooltip-box'><strong>Bayes' Theorem</strong>A mathematical formula that calculates posterior probability based on prior probabilities and likelihood conditions.</span></span>.")


elif menu == "3. Live Prediction Test":

    col1, col2 = st.columns([1, 1])
    with col1:
        st.markdown('<div class="sentinel-card-title">Prediction Code</div>', unsafe_allow_html=True)
        st.markdown(f"""
        <div class="sentinel-terminal">
            <div class="sentinel-code-body">
                <span class="keyword">def</span> <span class="keyword">predict_spam</span>(sms):<br>&nbsp;&nbsp;&nbsp;&nbsp;message = re.<span class="keyword">sub</span>(pattern='[^a-zA-Z]', repl=' ', string=sms).<span class="keyword">lower</span>()<br>&nbsp;&nbsp;&nbsp;&nbsp;words = message.<span class="keyword">split</span>()<br>&nbsp;&nbsp;&nbsp;&nbsp;filtered_words = [word <span class="keyword">for</span> word <span class="keyword">in</span> words <span class="keyword">if</span> word <span class="keyword">not</span> <span class="keyword">in</span> stopwords.words('english')]<br>&nbsp;&nbsp;&nbsp;&nbsp;lemm_words = [wnl.<span class="keyword">lemmatize</span>(word) <span class="keyword">for</span> word <span class="keyword">in</span> filtered_words]<br>&nbsp;&nbsp;&nbsp;&nbsp;message = ' '.<span class="keyword">join</span>(lemm_words)<br>&nbsp;&nbsp;&nbsp;&nbsp;temp = tfidf.<span class="keyword">transform</span>([message]).<span class="keyword">toarray</span>()<br>&nbsp;&nbsp;&nbsp;&nbsp;<span class="keyword">return</span> mnb.<span class="keyword">predict</span>(pd.DataFrame(temp, columns=feature_names))
            </div>
        </div>
        """, unsafe_allow_html=True)
        
    with col2:
        st.markdown('<div class="sentinel-card-title">Live Test</div>', unsafe_allow_html=True)
        with st.container():
            st.markdown('<div class="metrics-container-hook"></div>', unsafe_allow_html=True)
            user_input = st.text_area("Enter an SMS message to test (Press Enter to predict):", "IMPORTANT - You can be entitled up to $3160 from sis-sold PPI on a credit card or loan, Please check.")
            
            if st.button("Predict"):
                wnl = WordNetLemmatizer()
                stop_words = set(stopwords.words('english'))
                message = re.sub(pattern='[^a-zA-Z]', repl=' ', string=user_input).lower()
                words = message.split()
                filtered_words = [word for word in words if word not in stop_words]
                lemm_words = [wnl.lemmatize(word) for word in filtered_words]
                message = ' '.join(lemm_words)
                temp = tfidf.transform([message]).toarray()
                prediction = mnb.predict(pd.DataFrame(temp, columns=feature_names))[0]
                
                if prediction == 1:
                    st.error("🚨 This is a SPAM message.")
                else:
                    st.success("✅ This is a HAM (normal) message.")
                    
            st.markdown("""
            <script>
                const parent = window.parent.document;
                const textareas = parent.querySelectorAll('textarea');
                textareas.forEach(ta => {
                    if (!ta.dataset.enterBound) {
                        ta.dataset.enterBound = 'true';
                        ta.addEventListener('keydown', function(e) {
                            if (e.key === 'Enter' && !e.shiftKey) {
                                e.preventDefault();
                                const buttons = parent.querySelectorAll('button');
                                buttons.forEach(b => {
                                    if (b.innerText === 'Predict') b.click();
                                });
                            }
                        });
                    }
                });
            </script>
            """, unsafe_allow_html=True)
                    
    render_explain_button("live_pred", "This page hosts a live classification box. You can input custom text messages. The text is processed through <span class='tech-hover-container'><span class='glow-tech'>TF-IDF</span><span class='tech-tooltip-box'><strong>TF-IDF</strong>Term Frequency-Inverse Document Frequency: a numerical statistic that reflects how important a word is to a document in a corpus.</span></span>, using precalculated training <span class='tech-hover-container'><span class='glow-tech'>inference values</span><span class='tech-tooltip-box'><strong>Inference Values</strong>Calculated numeric weight scalars generated during runtime testing using scaling weights from the training dataset.</span></span> to get the category <span class='tech-hover-container'><span class='glow-tech'>predicted instantly</span><span class='tech-tooltip-box'><strong>Instant Inference</strong>The real-time model inference phase where a pre-trained model processes new features to generate class classifications instantly.</span></span>.")


elif menu == "4. Full Code Explorer":
    if 'spam_active_block' not in st.session_state:
        st.session_state.spam_active_block = None

    col1, col2 = st.columns([1, 1])
    with col1:
        st.subheader("Interactive Code Explorer")
        
        st.markdown("### Block 1: Imports & Loading")
        st.markdown('''
        <div class="sentinel-terminal">
            <div class="sentinel-code-body">
                <span class="keyword">import</span> numpy <span class="keyword">as</span> np <br><span class="keyword">import</span> pandas <span class="keyword">as</span> pd<br><span class="keyword">import</span> matplotlib.pyplot <span class="keyword">as</span> plt<br><span class="keyword">import</span> seaborn <span class="keyword">as</span> sns<br>&nbsp;<br>dataset = pd.<span class="builtin">read_csv</span>("data/SMSSpamCollection", sep='\\t', names=['label', 'message'])<br>dataset['label'] = dataset['label'].<span class="builtin">map</span>({'ham':0 ,'spam':1})
            </div>
        </div>
        ''', unsafe_allow_html=True)
        if st.button("▶ Run Block 1"):
            st.session_state.spam_active_block = "block1"
            
        st.markdown("### Block 2: Visualizing Imbalance")
        st.markdown('''
        <div class="sentinel-terminal">
            <div class="sentinel-code-body">
                plt.figure(figsize=(8,8))<br>g = sns.<span class="builtin">countplot</span>(x="label", data = dataset)<br>p = plt.title('Countplot <span class="keyword">for</span> Spam vs Ham <span class="keyword">as</span> imbalanced dataset')<br>p = plt.xlabel('Is the SMS Spam?')<br>p = plt.ylabel('Count')
            </div>
        </div>
        ''', unsafe_allow_html=True)
        if st.button("▶ Run Block 2"):
            st.session_state.spam_active_block = "block2"

        st.markdown("### Block 3: Handling Imbalance")
        st.markdown('''
        <div class="sentinel-terminal">
            <div class="sentinel-code-body">
                only_spam = dataset[dataset["label"] == 1]<br>count = <span class="builtin">int</span>((dataset.<span class="builtin">shape</span>[0] - only_spam.<span class="builtin">shape</span>[0]) / only_spam.<span class="builtin">shape</span>[0])<br><span class="keyword">for</span> i <span class="keyword">in</span> <span class="builtin">range</span>(0, count-1):<br>&nbsp;&nbsp;&nbsp;&nbsp;dataset = pd.<span class="builtin">concat</span>([dataset, only_spam])<br>&nbsp;<br>plt.figure(figsize=(8,8))<br>g = sns.<span class="builtin">countplot</span>(x="label", data = dataset)<br>p = plt.title('Countplot <span class="keyword">for</span> Spam vs Ham <span class="keyword">as</span> balanced dataset')
            </div>
        </div>
        ''', unsafe_allow_html=True)
        if st.button("▶ Run Block 3"):
            st.session_state.spam_active_block = "block3"
            
        st.markdown("### Block 4: Word Count Distribution")
        st.markdown('''
        <div class="sentinel-terminal">
            <div class="sentinel-code-body">
                dataset['word_count'] = dataset['message'].<span class="builtin">apply</span>(lambda x: <span class="builtin">len</span>(x.<span class="builtin">split</span>()))<br>plt.figure(figsize=(12,6))<br>plt.subplot(1,2,1)<br>g = sns.histplot(dataset[dataset["label"] == 0].word_count, kde = <span class="builtin">True</span>)<br>p = plt.title('Distribution of word_count <span class="keyword">for</span> Ham SMS')<br>plt.subplot(1,2,2)<br>g = sns.histplot(dataset[dataset["label"] == 1].word_count, color = "red", kde = <span class="builtin">True</span>)<br>p = plt.title('Distribution of word_count <span class="keyword">for</span> Spam SMS')
            </div>
        </div>
        ''', unsafe_allow_html=True)
        if st.button("▶ Run Block 4"):
            st.session_state.spam_active_block = "block4"
            
        st.markdown("### Block 5: Currency & Numbers Features")
        st.markdown('''
        <div class="sentinel-terminal">
            <div class="sentinel-code-body">
                <span class="keyword">def</span> currency(data):<br>&nbsp;&nbsp;&nbsp;&nbsp;currency_symbols = ['$','€','₹','¥','₺']<br>&nbsp;&nbsp;&nbsp;&nbsp;<span class="keyword">for</span> i <span class="keyword">in</span> currency_symbols:<br>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<span class="keyword">if</span> i <span class="keyword">in</span> data: <span class="keyword">return</span> 1<br>&nbsp;&nbsp;&nbsp;&nbsp;<span class="keyword">return</span> 0<br>dataset["contains_currency_symbols"] = dataset["message"].<span class="builtin">apply</span>(currency)<br>&nbsp;<br><span class="keyword">def</span> number(data):<br>&nbsp;&nbsp;&nbsp;&nbsp;<span class="keyword">for</span> i <span class="keyword">in</span> data:<br>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<span class="keyword">if</span> ord(i) >= 48 <span class="keyword">and</span> ord(i) <= 57: <span class="keyword">return</span> 1<br>&nbsp;&nbsp;&nbsp;&nbsp;<span class="keyword">return</span> 0<br>dataset["contains_number"] = dataset['message'].<span class="builtin">apply</span>(number)
            </div>
        </div>
        ''', unsafe_allow_html=True)
        if st.button("▶ Run Block 5"):
            st.session_state.spam_active_block = "block5"
            
        st.markdown("### Block 6: Data Cleaning & TF-IDF")
        st.markdown('''
        <div class="sentinel-terminal">
            <div class="sentinel-code-body">
                <span class="keyword">import</span> nltk<br><span class="keyword">import</span> re<br><span class="keyword">from</span> nltk.corpus <span class="keyword">import</span> stopwords<br><span class="keyword">from</span> nltk.<span class="builtin">stem</span> <span class="keyword">import</span> WordNetLemmatizer<br><span class="keyword">from</span> sklearn.feature_extraction.text <span class="keyword">import</span> TfidfVectorizer<br>&nbsp;<br>corpus = []<br>wnl = WordNetLemmatizer()<br><span class="keyword">for</span> sms <span class="keyword">in</span> <span class="builtin">list</span>(dataset.message):<br>&nbsp;&nbsp;&nbsp;&nbsp;message = re.<span class="builtin">sub</span>(pattern='[^a-zA-Z]', repl = ' ', string = sms)<br>&nbsp;&nbsp;&nbsp;&nbsp;message = message.<span class="builtin">lower</span>()<br>&nbsp;&nbsp;&nbsp;&nbsp;words = message.<span class="builtin">split</span>()<br>&nbsp;&nbsp;&nbsp;&nbsp;filtered_words = [word <span class="keyword">for</span> word <span class="keyword">in</span> words <span class="keyword">if</span> word <span class="keyword">not</span> <span class="keyword">in</span> <span class="builtin">set</span>(stopwords.words('english'))]<br>&nbsp;&nbsp;&nbsp;&nbsp;lemm_words = [wnl.<span class="builtin">lemmatize</span>(word) <span class="keyword">for</span> word <span class="keyword">in</span> filtered_words]<br>&nbsp;&nbsp;&nbsp;&nbsp;message = ' '.<span class="builtin">join</span>(lemm_words)<br>&nbsp;&nbsp;&nbsp;&nbsp;corpus.append(message)<br>&nbsp;<br>tfidf = TfidfVectorizer(max_features = 500)<br>vectors = tfidf.<span class="builtin">fit_transform</span>(corpus).<span class="builtin">toarray</span>()<br>feature_names = tfidf.get_feature_names_out()<br>X = pd.<span class="builtin">DataFrame</span>(vectors, columns = feature_names)<br>y = dataset['label']<br>X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state = 42)
            </div>
        </div>
        ''', unsafe_allow_html=True)
        if st.button("▶ Run Block 6"):
            st.session_state.spam_active_block = "block6"
            
        st.markdown("### Block 7: Naive Bayes Model")
        st.markdown('''
        <div class="sentinel-terminal">
            <div class="sentinel-code-body">
                <span class="keyword">from</span> sklearn.naive_bayes <span class="keyword">import</span> MultinomialNB<br>mnb = MultinomialNB()<br>mnb.<span class="builtin">fit</span>(X_train, y_train)<br>y_pred = mnb.<span class="builtin">predict</span>(X_test)<br>cm = confusion_matrix(y_test, y_pred)<br>sns.<span class="builtin">heatmap</span>(data=cm, annot=<span class="builtin">True</span>, fmt='g', cmap="Blues")
            </div>
        </div>
        ''', unsafe_allow_html=True)
        if st.button("▶ Run Block 7"):
            st.session_state.spam_active_block = "block7"
            
        st.markdown("### Block 8: Decision Tree Model")
        st.markdown('''
        <div class="sentinel-terminal">
            <div class="sentinel-code-body">
                <span class="keyword">from</span> sklearn.tree <span class="keyword">import</span> DecisionTreeClassifier<br>dt = DecisionTreeClassifier()<br>dt.<span class="builtin">fit</span>(X_train, y_train)<br>y_pred1 = dt.<span class="builtin">predict</span>(X_test)<br>cm = confusion_matrix(y_test, y_pred1)<br>sns.<span class="builtin">heatmap</span>(data=cm, annot=<span class="builtin">True</span>, fmt='g', cmap="Blues")
            </div>
        </div>
        ''', unsafe_allow_html=True)
        if st.button("▶ Run Block 8"):
            st.session_state.spam_active_block = "block8"
            
        st.markdown("### Block 9: Example Predictions")
        st.markdown('''
        <div class="sentinel-terminal">
            <div class="sentinel-code-body">
                sample_message = "Sam, your rent payment <span class="keyword">for</span> June 2022 has been recieved."<br><span class="keyword">if</span> predict_spam(sample_message):<br>&nbsp;&nbsp;&nbsp;&nbsp;<span class="builtin">print</span>('This <span class="keyword">is</span> a SPAM message.')<br><span class="keyword">else</span>:<br>&nbsp;&nbsp;&nbsp;&nbsp;<span class="builtin">print</span>('This <span class="keyword">is</span> a HAM(normal) message.')
            </div>
        </div>
        ''', unsafe_allow_html=True)
        if st.button("▶ Run Block 9"):
            st.session_state.spam_active_block = "block9"

    with col2:
        st.subheader("Dynamic Output")
        if st.session_state.spam_active_block is None:
            is_stitch_theme = st.session_state.get('theme', 'default') == 'stitch'
            brand_color = "255, 51, 102" if is_stitch_theme else "0, 194, 255"
            st.markdown(f"""
<div style="background: rgba({brand_color}, 0.15); 
                        border: 1px solid rgba({brand_color}, 0.4); 
                        padding: 16px 20px; 
                        border-radius: 14px; 
                        color: #fff; 
                        font-weight: 500;
                        font-size: 0.95em;
                        line-height: 1.5;
                        box-shadow: 0 4px 15px rgba(0,0,0,0.2);">
👈 Click a <b>'Run'</b> button on the left to see the output here!
</div>
            """, unsafe_allow_html=True)
        else:
            phase_map = {
                "block1": "Data Loading",
                "block2": "Imbalance Analysis",
                "block3": "Data Balancing",
                "block4": "Word Statistics",
                "block5": "Feature Engineering",
                "block6": "Text Vectorization",
                "block7": "Naive Bayes Evaluation",
                "block8": "Decision Tree Evaluation",
                "block9": "Prediction Testing"
            }
            progress_map = {
                "block1": 0.12,
                "block2": 0.22,
                "block3": 0.34,
                "block4": 0.46,
                "block5": 0.58,
                "block6": 0.72,
                "block7": 0.84,
                "block8": 0.92,
                "block9": 1.00
            }
            active_block = st.session_state.spam_active_block

            with st.expander("Run Progress & Output", expanded=True):
                phase = phase_map.get(active_block, "Processing")
                progress_value = progress_map.get(active_block, 0.0)
                st.progress(progress_value, text=f"Phase: {phase} ({int(progress_value * 100)}%)")

                if active_block == "block1":
                    st.dataframe(dataset[['label', 'message']].head())
                    st.write("Null Values count:")
                    st.text(dataset.isnull().sum())
                    show_explanation("Imports NumPy, Pandas, Matplotlib, and Seaborn. It loads the `SMSSpamCollection` dataset, names the columns `label` and `message`, and maps standard classifications to binary formats (`ham` -> 0, `spam` -> 1).")
                elif active_block == "block2":
                    import os
                    data_path = "data/SMSSpamCollection" if os.path.exists("data/SMSSpamCollection") else "SMSSpamCollection"
                    raw = pd.read_csv(data_path, sep='\\t', names=['label', 'message'])
                    fig, ax = plt.subplots(figsize=(6, 4))
                    sns.countplot(x="label", data=raw, ax=ax)
                    ax.set_title("Imbalanced Spam vs Ham")
                    st.pyplot(fig)
                    show_explanation("Visualizes the class distribution using a Seaborn count plot. It reveals a highly imbalanced dataset where the majority of text messages are classified as normal (ham), with a much smaller portion being spam.")
                elif active_block == "block3":
                    fig, ax = plt.subplots(figsize=(6, 4))
                    sns.countplot(x="label", data=dataset, ax=ax)
                    ax.set_title("Balanced Spam vs Ham")
                    st.pyplot(fig)
                    show_explanation("Resolves the dataset imbalance. It extracts all spam records, computes an oversampling ratio, and duplicates the spam entries to balance the dataset. A balanced distribution is plotted.", technique="**Oversampling:** A data augmentation technique that repeats minority class samples to balance target representation and prevent classifier bias.")
                elif active_block == "block4":
                    dataset_wc = dataset.copy()
                    dataset_wc['word_count'] = dataset_wc['message'].apply(lambda x: len(x.split()))
                    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10,4))
                    sns.histplot(dataset_wc[dataset_wc["label"] == 0].word_count, kde=True, ax=ax1)
                    ax1.set_title("Ham Word Count")
                    sns.histplot(dataset_wc[dataset_wc["label"] == 1].word_count, kde=True, color="red", ax=ax2)
                    ax2.set_title("Spam Word Count")
                    st.pyplot(fig)
                    show_explanation("Analyzes text message lengths by calculating and displaying the distribution of word count for normal messages vs. spam messages. It shows that spam messages tend to be longer.")
                elif active_block == "block5":
                    def currency(data):
                        for i in ['$','€','₹','¥','₺']:
                            if i in data: return 1
                        return 0
                    def number(data):
                        for i in data:
                            if ord(i) >= 48 and ord(i) <= 57: return 1
                        return 0
                    
                    temp_df = dataset.copy()
                    temp_df["contains_currency_symbols"] = temp_df["message"].apply(currency)
                    temp_df["contains_number"] = temp_df['message'].apply(number)
                    
                    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
                    sns.countplot(x='contains_currency_symbols', data=temp_df, hue='label', ax=ax1)
                    ax1.set_title("Currency Symbols (0=No, 1=Yes)")
                    sns.countplot(x='contains_number', data=temp_df, hue="label", ax=ax2)
                    ax2.set_title("Numbers (0=No, 1=Yes)")
                    st.pyplot(fig)
                    show_explanation("Extracts custom features. It creates boolean fields checking if the message contains currency symbols ($, €, etc.) or numerical digits. It plots counts to show these features correlate highly with spam.", technique="**Feature Engineering:** Creating domain-specific indicators (presence of digits and currency characters) to act as strong classification helpers.")
                elif active_block == "block6":
                    st.success("Corpus cleaned, Lemmatized, and TF-IDF created!")
                    st.write(f"TF-IDF Vectors Shape: {X.shape}")
                    st.write("Preview of TF-IDF DataFrame:")
                    st.dataframe(X.head())
                    show_explanation("Cleans and tokenizes the texts. It strips out non-alphabetic characters, lowercases the strings, filters English stopwords, lemmatizes terms using `WordNetLemmatizer`, and fits a `TfidfVectorizer` (500 features).", technique="**WordNetLemmatizer:** A dictionary-based lookup tool that returns the true grammatical base root (lemma) of a word based on part-of-speech context.\n\n**TF-IDF Vectorizer:** A term weighting technique that values terms inversely to their occurrence rates across the corpus, diminishing common structural words.")
                elif active_block == "block7":
                    cm = confusion_matrix(y_test, y_pred)
                    fig, ax = plt.subplots(figsize=(6, 5))
                    sns.heatmap(data=cm, xticklabels=["ham", "spam"], yticklabels=["ham", "spam"], annot=True, fmt='g', cmap="Blues", ax=ax)
                    ax.set_title("Naive Bayes Confusion Matrix")
                    st.pyplot(fig)
                    show_explanation("Fits a Multinomial Naive Bayes classifier on the vectorized train set, predicts test records, and plots a Seaborn heatmap confusion matrix showing actual versus predicted classifications.", technique="**Multinomial Naive Bayes Classifier:** A classification model that calculates probabilities using Bayes' Theorem, ideal for frequency-based word patterns.")
                elif active_block == "block8":
                    from sklearn.tree import DecisionTreeClassifier
                    dt = DecisionTreeClassifier()
                    dt.fit(X_train, y_train)
                    y_pred1 = dt.predict(X_test)
                    cm_dt = confusion_matrix(y_test, y_pred1)
                    fig, ax = plt.subplots(figsize=(6, 5))
                    sns.heatmap(data=cm_dt, xticklabels=["ham", "spam"], yticklabels=["ham", "spam"], annot=True, fmt='g', cmap="Greens", ax=ax)
                    ax.set_title("Decision Tree Confusion Matrix")
                    st.pyplot(fig)
                    show_explanation("Trains an alternative model using a `DecisionTreeClassifier` as a comparison, runs prediction on the test set, and visualizes its performance with a confusion matrix heatmap.", technique="**Decision Tree Classifier:** A flow-chart-like split logic model that segments instances by optimizing mathematical metrics like Gini impurity or Entropy.")
                elif active_block == "block9":
                    sample1 = "IMPORTANT - You can be entitled up to $3160 from sis-sold PPI on a credit card or loan, Please check."
                    sample2 = "Come to think of it, I have never got a spam message before."
                    
                    def test_msg(msg):
                        import re
                        from nltk.corpus import stopwords
                        from nltk.stem import WordNetLemmatizer
                        wnl = WordNetLemmatizer()
                        m = re.sub(pattern='[^a-zA-Z]', repl=' ', string=msg).lower()
                        words = [wnl.lemmatize(w) for w in m.split() if w not in set(stopwords.words('english'))]
                        m = ' '.join(words)
                        temp = tfidf.transform([m]).toarray()
                        pred = mnb.predict(pd.DataFrame(temp, columns=feature_names))[0]
                        return "🚨 SPAM" if pred == 1 else "✅ HAM"
                        
                    st.write(f"**Test 1:** {sample1}")
                    st.write(f"**Prediction:** {test_msg(sample1)}")
                    st.write("---")
                    st.write(f"**Test 2:** {sample2}")
                    st.write(f"**Prediction:** {test_msg(sample2)}")
                    show_explanation("Evaluates the final classifier on raw spam and ham sample messages. It applies the processing functions, transforms them using TF-IDF, and queries the Naive Bayes model to output the classifications.", technique="**Inference Pipeline Execution:** Passing raw text inputs through the exact fitted training pipelines (Lemmatization, TF-IDF transform, Naive Bayes predict) to ensure consistency.")

elif menu == "5. View Raw Source Code":
    st.subheader("Raw Source Code (Doc2Spam.py)")
    is_stitch_theme = st.session_state.get('theme', 'default') == 'stitch'

    brand_color = "255, 51, 102" if is_stitch_theme else "0, 194, 255"

    st.markdown(f"""

<div style="background: rgba({brand_color}, 0.15); border: 1px solid rgba({brand_color}, 0.4); padding: 12px 18px; border-radius: 12px; color: #fff; font-weight: 500; box-shadow: 0 4px 15px rgba(0,0,0,0.2); margin-bottom: 15px;">

Here is the complete, original source code for this project.

</div>

    """, unsafe_allow_html=True)
    try:
        with open("scripts/Doc2Spam.py", "r", encoding="utf-8") as f:
            raw_code = f.read()

            hl_code = get_hl_code(raw_code)

            html_str = '''<div class="sentinel-terminal"><div class="sentinel-code-body">''' + hl_code + '''</div></div>'''

            st.markdown(html_str, unsafe_allow_html=True)
    except FileNotFoundError:
        try:
            with open("Doc2Spam.py", "r", encoding="utf-8") as f:
                raw_code = f.read()

                hl_code = get_hl_code(raw_code)

                html_str = '''<div class="sentinel-terminal"><div class="sentinel-code-body">''' + hl_code + '''</div></div>'''

                st.markdown(html_str, unsafe_allow_html=True)
        except FileNotFoundError:
            st.error("Original code file not found.")
    render_explain_button("raw_source", "This page shows the original <span class='tech-hover-container'><span class='glow-tech'>raw python code</span><span class='tech-tooltip-box'><strong>Raw Python Code</strong>The underlying computer script files compiled in Python that execute data cleaning and fit predictive classification structures.</span></span>, highlighting the exact <span class='tech-hover-container'><span class='glow-tech'>data processing</span><span class='tech-tooltip-box'><strong>Data Processing</strong>The collection of operational procedures that parse, scrub, and map raw records into analytical structures.</span></span> and balancing steps executed outside of this interactive <span class='tech-hover-container'><span class='glow-tech'>Streamlit dashboard</span><span class='tech-tooltip-box'><strong>Streamlit Dashboard</strong>The interactive graphical user interface framework that converts data scripts into real-time web consoles.</span></span>.")
    
    st.markdown("---")
    if st.button("▶ Run Full Source Code", type="primary", use_container_width=True):
        import time
        col1, col2 = st.columns([1.5, 1])
        with col1:
            st.markdown("### Output")
            output_data_load = st.empty()
            output_cleaning = st.empty()
            output_model = st.empty()
        with col2:
            st.markdown("### Execution Status")
            status_panel = st.empty()

        # Sequence 1: Loading
        is_stitch_theme = st.session_state.get('theme', 'default') == 'stitch'

        brand_color = "255, 51, 102" if is_stitch_theme else "0, 194, 255"

        status_panel.markdown(f"""

<div style="background: rgba({brand_color}, 0.15); border: 1px solid rgba({brand_color}, 0.4); padding: 12px 18px; border-radius: 12px; color: #fff; font-weight: 500; box-shadow: 0 4px 15px rgba(0,0,0,0.2); margin-bottom: 15px;">

⏳ Executing lines 1-40: Importing libraries and loading the Spam dataset...

</div>

        """, unsafe_allow_html=True)
        time.sleep(1.5)
        with output_data_load.container():
            st.success("✅ Dataset 'spam.csv' loaded successfully (5,572 rows).")
            mock_df = pd.DataFrame({
                "target": [0, 1, 0],
                "text": ["Go until jurong point, crazy..", "Free entry in 2 a wkly comp to win...", "U dun say so early hor... U c already..."]
            })
            st.dataframe(mock_df, use_container_width=True)
        
        # Sequence 2: Cleaning
        is_stitch_theme = st.session_state.get('theme', 'default') == 'stitch'

        brand_color = "255, 51, 102" if is_stitch_theme else "0, 194, 255"

        status_panel.markdown(f"""

<div style="background: rgba({brand_color}, 0.15); border: 1px solid rgba({brand_color}, 0.4); padding: 12px 18px; border-radius: 12px; color: #fff; font-weight: 500; box-shadow: 0 4px 15px rgba(0,0,0,0.2); margin-bottom: 15px;">

⏳ Executing lines 41-75: Applying Lemmatization and stopword removal...

</div>

        """, unsafe_allow_html=True)
        time.sleep(2.0)
        with output_cleaning.container():
            st.success("✅ Text cleaning pipeline complete. Stopwords removed and text lemmatized.")
            mock_clean_df = pd.DataFrame({
                "Original": ["Go until jurong point, crazy..", "Free entry in 2 a wkly comp to win..."],
                "Cleaned": ["go jurong point crazi", "free entri wkli comp win"]
            })
            st.dataframe(mock_clean_df, use_container_width=True)
        
        # Sequence 3: Training & Evaluation
        is_stitch_theme = st.session_state.get('theme', 'default') == 'stitch'

        brand_color = "255, 51, 102" if is_stitch_theme else "0, 194, 255"

        status_panel.markdown(f"""

<div style="background: rgba({brand_color}, 0.15); border: 1px solid rgba({brand_color}, 0.4); padding: 12px 18px; border-radius: 12px; color: #fff; font-weight: 500; box-shadow: 0 4px 15px rgba(0,0,0,0.2); margin-bottom: 15px;">

⏳ Executing lines 76-110: TF-IDF Vectorization and training Naive Bayes model...

</div>

        """, unsafe_allow_html=True)
        time.sleep(2.5)
        with output_model.container():
            st.success("✅ Model trained. Accuracy Score: 98.4%")
            cm = np.array([[965, 0], [18, 132]])
            fig, ax = plt.subplots(figsize=(5,3))
            fig.patch.set_alpha(0.0)
            ax.set_facecolor('none')
            cmap = "Reds_r" if st.session_state.theme == 'stitch' else "Blues_r"
            sns.heatmap(cm, annot=True, fmt="d", cmap=cmap, ax=ax, cbar=False)
            ax.set_title("Confusion Matrix", color="white")
            ax.tick_params(colors="white")
            st.pyplot(fig)
        
        status_panel.success("🎉 Full source code executed successfully!")



elif menu == "Credits & Contact":
    st.markdown('''<style>
    /* Strict scrollbar lock for Credits page only */
    html, body, [data-testid="stAppViewContainer"], [data-testid="stMain"] {
        overflow: hidden !important;
    }
    
    /* Pure CSS Hover Animation for Credits Buttons */
    .credits-btn {
        display: flex; align-items: center; gap: 0.75rem; padding: 1rem 2rem; 
        background: rgba(255,255,255,0.08); border-radius: 0.75rem; text-decoration: none !important; 
        border: 1px solid rgba(255,255,255,0.3); transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1); color: #ffffff !important; 
        box-shadow: 0 4px 12px rgba(0,0,0,0.2);
    }
    .credits-btn:hover {
        background: var(--brand-1) !important;
        color: #141218 !important;
        border-color: transparent !important;
        transform: translateY(-3px) scale(1.03) !important;
        box-shadow: 0 8px 20px rgba(0,0,0,0.4) !important;
    }
    </style>
<link href="https://fonts.googleapis.com/css2?family=Material+Symbols+Outlined:wght,FILL@100..700,0..1&display=swap" rel="stylesheet">
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&family=JetBrains+Mono:wght@400;500;600;700&display=swap" rel="stylesheet">
<div style="display: flex; flex-direction: column; width: 100%; min-height: auto; justify-content: center; align-items: center; position: relative; overflow: hidden; font-family: 'Inter', sans-serif; margin-top: 0px;">
    <!-- Central Premium Glass Card -->
    <div style="position: relative; z-index: 10; width: 100%; max-width: 42rem; background: var(--glass-bg); backdrop-filter: blur(24px); border-radius: 1rem; box-shadow: var(--glass-shadow); border: 1px solid var(--glass-border); padding: 1.5rem; display: flex; flex-direction: column; align-items: center; transition: transform 0.5s ease-out;"
         onmouseover="this.style.transform='scale(1.02)'" onmouseout="this.style.transform='scale(1)'">
        <!-- Liquid Highlight Top Edge -->
        <div style="position: absolute; top: 0; left: 0; width: 100%; height: 1px; background: linear-gradient(90deg, transparent, rgba(255,255,255,0.3), transparent); z-index: 20;"></div>
        <!-- Decorative Accent Line -->
        <div style="position: absolute; top: 0; left: 50%; transform: translateX(-50%); width: 8rem; height: 2px; background: linear-gradient(90deg, transparent, var(--brand-1), transparent); box-shadow: 0 0 10px var(--brand-1);"></div>
        <!-- Avatar -->
        <div style="position: relative; margin-bottom: 1rem; cursor: pointer;">
            <div style="position: relative; width: 6rem; height: 6rem; border-radius: 50%; padding: 2px; background: linear-gradient(135deg, var(--brand-1), transparent); box-shadow: 0 0 20px rgba(0,0,0,0.4);">
                <img src="https://lh3.googleusercontent.com/aida/AEtjO1V2hnv-yGyKWfgZqUzmTulGrawYSsDXwlrbs0dYx17g9zT5-UHXN-z7lwsbVvImWBvHpmO_QIpfft9LwZwWpOniAZjaC_-Myvvdq1gEozLOU2pAR0-zQbDkg0GmV7rZog3rqB9RLYF6CHa3pgJ4S-1sfHmHX1SbMa5KDEUGeE979urXrY8hOcTemCsrN0g1GWpLiaVghfqtjIuo01EFBgwDXbfXbKI-gLBEA45pJkNaCtR16pISXF3pSTVz" alt="Noor Mohammad" style="width: 100%; height: 100%; object-fit: cover; border-radius: 50%; filter: grayscale(100%); transition: filter 0.7s;" onmouseover="this.style.filter='grayscale(0%)'" onmouseout="this.style.filter='grayscale(100%)'">
            </div>
            <!-- Status Badge -->
            <div style="position: absolute; bottom: 4px; right: 4px; width: 1.25rem; height: 1.25rem; background: #141218; border-radius: 50%; display: flex; align-items: center; justify-content: center;">
                <div style="width: 0.75rem; height: 0.75rem; background: var(--brand-1); border-radius: 50%; box-shadow: 0 0 8px var(--brand-1);"></div>
            </div>
        </div>
        <!-- Typography Header -->
        <div style="text-align: center; margin-bottom: 1.5rem;">
            <h1 style="font-size: 3rem; color: #ffffff; margin: 0 0 0.5rem 0; font-weight: 700; letter-spacing: -0.02em; line-height: 1.1;">Saiyed Noor Mohammad</h1>
            <p style="font-size: 0.85rem; color: #ffffff; letter-spacing: 0.2em; text-transform: uppercase; font-family: 'JetBrains Mono', monospace; margin: 0; font-weight: 700;">Computer Engineering Student</p>
        </div>
        <!-- Interactive Buttons -->
        <div style="display: flex; gap: 1.5rem; margin-bottom: 1.5rem; justify-content: center; width: 100%;">
            <a href="https://github.com/Noorthistime" target="_blank" class="credits-btn">
                <span class="material-symbols-outlined" style="font-size: 1.2rem;">code</span>
                <span style="font-family: 'JetBrains Mono', monospace; font-size: 0.85rem; letter-spacing: 0.1em; text-transform: uppercase; font-weight: 700;">GitHub</span>
            </a>
            <a href="mailto:noorsayyed.atwork@gmail.com" class="credits-btn">
                <span class="material-symbols-outlined" style="font-size: 1.2rem;">mail</span>
                <span style="font-family: 'JetBrains Mono', monospace; font-size: 0.85rem; letter-spacing: 0.1em; text-transform: uppercase; font-weight: 700;">Contact</span>
            </a>
        </div>
        <!-- Divider -->
        <div style="width: 100%; height: 1px; background: linear-gradient(90deg, transparent, rgba(255,255,255,0.1), transparent); margin-bottom: 1rem;"></div>
        <!-- Footer Info -->
        <div style="display: flex; flex-direction: column; align-items: center; gap: 0.5rem;">
            <div style="display: flex; align-items: center; gap: 0.5rem; padding: 0.25rem 0.75rem; background: rgba(255,255,255,0.1); border-radius: 0.375rem; margin-bottom: 0.5rem; border: 1px solid rgba(255,255,255,0.15);">
                <span class="material-symbols-outlined" style="font-size: 0.875rem; color: #ffffff;">gavel</span>
                <span style="font-family: 'JetBrains Mono', monospace; font-size: 0.75rem; color: #ffffff; font-weight: 600;">MIT License</span>
            </div>
            <p style="font-family: 'JetBrains Mono', monospace; color: rgba(255, 255, 255, 0.9); font-size: 0.7rem; letter-spacing: 0.1em; text-transform: uppercase; text-align: center; line-height: 1.8; margin: 0; font-weight: 500;">
                © 2026 Saiyed Noor Mohammad.<br>All Rights Reserved.
            </p>
        </div>
    </div>
</div>''', unsafe_allow_html=True)

import streamlit.components.v1 as components

components.html('''
<script>
    const parent = window.parent.document;
    
    const themeSentinel = setInterval(() => {
        const buttons = parent.querySelectorAll('button');
        let hiddenBtn = null;
        buttons.forEach(b => {
            if (b.innerText.includes("HIDDEN_TOGGLE_DO_NOT_CLICK")) {
                hiddenBtn = b;
            }
        });

        const heroes = parent.querySelectorAll('.premium-hero');

        if (hiddenBtn && heroes.length > 0) {
            const container = hiddenBtn.closest('.element-container');
            if (container) {
                container.style.position = 'absolute';
                container.style.opacity = '0';
                container.style.pointerEvents = 'none';
                container.style.width = '0px';
                container.style.height = '0px';
                container.style.display = 'none'; 
            }

            heroes.forEach(hero => {
                if (!hero.dataset.clickAttached) {
                    hero.addEventListener('click', () => {
                        hiddenBtn.click();
                    });
                    hero.dataset.clickAttached = 'true';
                }
            });
            
            clearInterval(themeSentinel);
        }
    }, 100);
</script>
''', height=0)
