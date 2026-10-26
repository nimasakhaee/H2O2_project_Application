
# ══════════════════════════════════════════════════════════════════════════════
# Streamlite.py — Streamlit app for the XGBoost H2O2 model (Kim project 3)
# Run with :  streamlit run Streamlite.py
# Requires :  H2O2_XGB_package.pkl (made by make_gui_package.py) in the same folder
#             pip install streamlit plotly xgboost scikit-learn pandas numpy joblib
# ══════════════════════════════════════════════════════════════════════════════
import random
import warnings
from pathlib import Path
import numpy as np
import pandas as pd
import joblib
import streamlit as st
import plotly.graph_objects as go
warnings.filterwarnings("ignore")

PKG_PATH = r"G:\My Drive\NEU\research\research\Kim project 3\NEW SYSTEM\Results\18- GUI\XGB_package.pkl"
PI_LEVEL = 0.90
HR_MED, HR_LOW = 1.0, 3.0     # h/h* ≤ 1 → high; 1–3 → medium; > 3 → low confidence
O2_SAT_MG_L = 8.26            # dissolved O2, air-saturated at 25 °C (mg O2/L)
F_ANODE     = 0.5             # anodic O2 (1 O2 per 4 e⁻) covers half the 2 e⁻ demand

# >>> CORE (prediction logic, no Streamlit calls) ──────────────────────────────
def load_pkg(path):
    return joblib.load(path)

def apply_preprocessor(X_a, pkg):
    prep, NUM, CAT, BIN = pkg['prep'], pkg['numeric_cols'], pkg['categorical_cols'], pkg['binary_cols']
    Xo = X_a.copy().reset_index(drop=True)
    Xo[NUM] = prep['num_imputer'].transform(np.array(Xo[NUM], dtype=float, copy=True))
    Xo[pkg['integer_cols']] = Xo[pkg['integer_cols']].round()
    Xo[BIN] = prep['bin_imputer'].transform(Xo[BIN])
    Xo[CAT] = prep['cat_imputer'].transform(Xo[CAT])
    for c in CAT:
        Xo[c] = pd.Categorical(Xo[c], categories=prep['cat_categories'][c])
    return Xo[pkg['feature_cols']]

def leverage_ratio(Xp, pkg):
    L = pkg['leverage']
    NUM, CAT, BIN = pkg['numeric_cols'], pkg['categorical_cols'], pkg['binary_cols']
    dum = pd.get_dummies(Xp[CAT], dtype=float).reindex(columns=L['dummy_cols'], fill_value=0.0)
    nums = (Xp[NUM] - pd.Series(L['num_mean'])) / pd.Series(L['num_std'])
    d = np.r_[1.0, pd.concat([dum, Xp[BIN].astype(float), nums], axis=1).values.ravel()]
    return float(d @ L['XtX_inv'] @ d) / L['h_star']

def faraday_bound(I_mA, t_min, V_mL, Q_mLmin, is_flow, pkg):
    """Upper bound on C (mg/L) at 100 % Faradaic efficiency (2 e⁻ per H2O2)."""
    K = pkg['constants']
    mg_per_C = K['MW_H2O2'] * 1000.0 / (K['n_e'] * K['F'])             # mg H2O2 per coulomb
    out = {}
    if V_mL > 0:                                                         # closed batch / recirculation
        out['volume'] = I_mA / 1000.0 * t_min * 60.0 * mg_per_C / (V_mL / 1000.0)
    if is_flow and Q_mLmin > 0:                                          # single pass, steady state
        out['single_pass'] = I_mA / 1000.0 * mg_per_C / (Q_mLmin / 1000.0 / 60.0)
    return out

def range_for(feat, rtype, pkg):
    s = pkg['num_stats_by_type'][rtype].get(feat) or pkg['num_stats'].get(feat)
    return (s['min'], s['max'], s['median']) if s else (0.0, 1e6, 0.0)

def predict_all(inputs, pkg):
    """inputs: dict with categorical values, 'Binder' label and numeric values."""
    rtype = 'Flow-through' if pkg['is_flow_type'].get(inputs[pkg['rt_col']], False) else 'Batch'
    row = {c: inputs[c] for c in pkg['categorical_cols'] + pkg['numeric_cols']}
    row.update(pkg['binder_map'].get(inputs['Binder'], {b: 0 for b in pkg['binary_cols']}))
    X_raw = pd.DataFrame([row])[pkg['feature_cols']]
    for c in pkg['numeric_cols'] + pkg['binary_cols']:
        X_raw[c] = pd.to_numeric(X_raw[c])
    Xp = apply_preprocessor(X_raw, pkg)
    pred = max(float(pkg['model'].predict(Xp)[0]), 0.0)
    f = float(np.exp(pkg['pi_q'][PI_LEVEL]))
    lo, hi = max((1 + pred) / f - 1, 0.0), (1 + pred) * f - 1
    hr = leverage_ratio(Xp, pkg)

    # Check 1 — leverage only
    level = 'High' if hr <= HR_MED else ('Medium' if hr <= HR_LOW else 'Low')

    # Check 2 — Faraday limit
    fb = faraday_bound(float(inputs['Current (mA)']), float(inputs['Reaction time (min)']),
                       float(inputs['Reactor volume (mL)']), float(inputs[pkg['flow_col']]),
                       rtype == 'Flow-through', pkg)
    bound = max(fb.values()) if fb else np.nan          # flow-through: the more lenient bound
    exceeds = bool(fb) and pred > bound
    fe_implied = 100 * pred / bound if fb and bound > 0 else np.nan

    # Check 3 — category combination
    combo = tuple(str(inputs[c]) for c in pkg['categorical_cols'])
    combo_seen = combo in pkg['seen_combos']
    unseen_pairs = [] if combo_seen else [
        f"{a} × {b}" for (a, b), seen in pkg['seen_pairs'].items()
        if (str(inputs[a]), str(inputs[b])) not in seen]

    return dict(pred=pred, lo=lo, hi=hi, f=f, rtype=rtype,
                h_ratio=hr, level=level,
                faraday=fb, bound=bound, exceeds=exceeds, fe_implied=fe_implied,
                combo_seen=combo_seen, unseen_pairs=unseen_pairs)
def predict_curve(inputs, pkg, n_points=10):
    """Predictions at n_points times from 0 to the user's reaction time (C(0) = 0)."""
    T = float(inputs['Reaction time (min)'])
    t = np.linspace(0.0, T, n_points)
    rows = []
    for ti in t[1:]:
        d = {c: inputs[c] for c in pkg['categorical_cols'] + pkg['numeric_cols']}
        d.update(pkg['binder_map'].get(inputs['Binder'], {b: 0 for b in pkg['binary_cols']}))
        d['Reaction time (min)'] = ti
        rows.append(d)
    X_raw = pd.DataFrame(rows)[pkg['feature_cols']]
    for c in pkg['numeric_cols'] + pkg['binary_cols']:
        X_raw[c] = pd.to_numeric(X_raw[c])
    p = np.clip(pkg['model'].predict(apply_preprocessor(X_raw, pkg)), 0, None)
    f = float(np.exp(pkg['pi_q'][PI_LEVEL]))
    pred = np.r_[0.0, p]
    lo = np.r_[0.0, np.clip((1 + p) / f - 1, 0, None)]
    hi = np.r_[0.0, (1 + p) * f - 1]
    return t, pred, lo, hi
# <<< CORE ─────────────────────────────────────────────────────────────────────

# ══════════════════════════════════════════════════════════════════════════════
# PAGE + STYLE
# ══════════════════════════════════════════════════════════════════════════════
# Force the light theme from inside the app (no config file needed), so text is
# never white-on-white when the computer uses dark mode.
for _k, _v in {"theme.base": "light", "theme.textColor": "#1b1b19",
               "theme.backgroundColor": "#f7f6f2", "theme.secondaryBackgroundColor": "#ffffff",
               "theme.primaryColor": "#0f6e63"}.items():
    try:
        st._config.set_option(_k, _v)
    except Exception:
        pass

st.set_page_config(page_title="H₂O₂ Production Predictor", page_icon="⚗", layout="wide",
                   initial_sidebar_state="expanded")
INK, MUTED, LINE, PAPER, ACCENT = "#1b1b19", "#6f6d66", "#e4e2db", "#f7f6f2", "#0f6e63"
C_TRAIN, C_VAL, C_TEST = "#bdbbb2", "#4a7fb5", "#0f6e63"
LEVEL_COLORS = {'High': "#1a7f37", 'Medium': "#b7791f", 'Low': "#b3261e"}
# light, journal-style section tints: (background, accent)
SECTION_COLORS = [("#EAF1F8", "#3A6EA5", "#F5F9FC"),   # ① electrode — soft blue
                  ("#EEF5EC", "#4E8A4A", "#F7FBF6"),   # ② catalyst & binder — sage
                  ("#F8F2E6", "#A8742A", "#FCF9F3"),   # ③ reactor — sand
                  ("#F2EEF8", "#6B57A5", "#F9F7FC")]   # ④ electrolyte & operation — lavender
                 # (header tint, accent, panel tint)

st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Source+Serif+4:opsz,wght@8..60,400;8..60,700&family=Inter:wght@400;500;600&display=swap');
html, body, [class*="css"] {{ font-family: 'Inter', sans-serif; }}
[data-testid="stAppViewContainer"] {{ background: {PAPER}; }}
[data-testid="stHeader"] {{ background: transparent; }}
.block-container {{ padding-top: 2.4rem; max-width: 1180px; }}
[data-testid="stSidebar"] {{ background: #ffffff; border-right: 1px solid {LINE}; }}
h1, h2, h3, h4 {{ font-family: 'Source Serif 4', serif !important; color: {INK}; }}
.masthead {{ border-bottom: 2px solid {INK}; padding-bottom: 14px; margin-bottom: 6px; }}
.eyebrow {{ font-size: 11.5px; font-weight: 600; letter-spacing: 2.4px; text-transform: uppercase;
            color: {ACCENT}; margin-bottom: 8px; }}
.page-title {{ font-family:'Source Serif 4', serif; font-size: 34px; font-weight: 700; color:#05070a;
               line-height: 1.15; margin: 0; }}
.page-subtitle {{ font-size: 14.5px; color: {MUTED}; line-height: 1.6; margin-top: 10px; max-width: 800px; }}
.card-title {{ font-size: 12px; font-weight: 600; letter-spacing: 1.4px; text-transform: uppercase;
               color: {INK}; margin: 2px 0 12px 0; padding: 9px 12px; border-radius: 3px;
               border-left: 4px solid; }}
.card-title span {{ margin-right: 8px; }}
.masthead {{ background: linear-gradient(90deg, #EAF1F8 0%, #F2EEF8 55%, #F8F2E6 100%);
             padding: 18px 22px 14px 22px !important; border-radius: 6px 6px 0 0; }}
.result-box {{ background: linear-gradient(180deg, #F3F8F6 0%, #ffffff 60%) !important; }}
[data-testid="stVerticalBlockBorderWrapper"]:has(.sec-0):not(:has([data-testid="stVerticalBlockBorderWrapper"] .sec-0)) {{ background: #F5F9FC !important;
    border-color: #3A6EA533 !important; }}
[data-testid="stVerticalBlockBorderWrapper"]:has(.sec-1):not(:has([data-testid="stVerticalBlockBorderWrapper"] .sec-1)) {{ background: #F7FBF6 !important;
    border-color: #4E8A4A33 !important; }}
[data-testid="stVerticalBlockBorderWrapper"]:has(.sec-2):not(:has([data-testid="stVerticalBlockBorderWrapper"] .sec-2)) {{ background: #FCF9F3 !important;
    border-color: #A8742A33 !important; }}
[data-testid="stVerticalBlockBorderWrapper"]:has(.sec-3):not(:has([data-testid="stVerticalBlockBorderWrapper"] .sec-3)) {{ background: #F9F7FC !important;
    border-color: #6B57A533 !important; }}
.result-label {{ font-size: 15px !important; letter-spacing: 1.4px !important; }}
.result-sub {{ font-size: 16px !important; line-height: 1.55; }}
.badge {{ font-size: 14px !important; padding: 4px 14px !important; }}
.check {{ background:#ffffff; border:1px solid {LINE}; border-top: 4px solid; border-radius: 5px;
           padding: 14px 16px; height: 100%; }}
.check-title {{ font-size: 12.5px; font-weight: 600; letter-spacing: 1.2px; text-transform: uppercase;
                color: {MUTED} !important; margin-bottom: 6px; }}
.check-status {{ font-family:'Source Serif 4', serif; font-size: 19px; font-weight: 700; margin-bottom: 8px; }}
.check-body {{ font-size: 14.5px; line-height: 1.55; }}
.check-status, .check-status * {{ color: inherit; }}
.note {{ font-size: 14.5px; color: {MUTED}; line-height: 1.55; margin: 10px 2px 6px 2px; }}
[data-testid="stAlert"] p {{ font-size: 15.5px !important; }}
.result-box {{ background:#fff; border:1px solid {LINE}; border-top: 3px solid {ACCENT}; border-radius: 4px;
               padding: 22px 26px; text-align:center; }}
.result-label {{ font-size: 11.5px; letter-spacing: 1.8px; text-transform: uppercase; color:{MUTED}; margin-bottom: 8px; }}
.result-value {{ font-family:'Source Serif 4', serif; font-size: 52px; font-weight: 700; color:{INK}; line-height: 1; }}
.result-unit {{ font-size: 15px; color:{MUTED}; margin-top: 4px; }}
.result-sub {{ font-size: 13px; color:{INK}; margin-top: 12px; padding-top: 10px; border-top: 1px solid {LINE}; }}
.badge {{ display:inline-block; padding: 4px 12px; border-radius: 12px; color:#fff; font-size: 12.5px;
          font-weight: 600; letter-spacing: 0.4px; }}
.sb-k {{ font-size: 12px; color:{MUTED}; text-transform: uppercase; letter-spacing: 1px; }}
.sb-v {{ font-family:'Source Serif 4', serif; font-size: 15px; color:{INK}; margin-bottom: 12px; }}
div[data-testid="stButton"] > button {{ background: {INK} !important; color: #fff !important;
    font-weight: 600 !important; border-radius: 4px !important; padding: 12px 34px !important;
    border: none !important; width: 100%; }}
div[data-testid="stButton"] > button:hover {{ background: {ACCENT} !important; }}
section[data-testid="stSidebar"] div[data-testid="stButton"] > button {{
    background: #eef1ee !important; color: {INK} !important; border: 1px solid {LINE} !important; }}

/* ---- force black text on light backgrounds (also when Streamlit runs in dark theme) ---- */
[data-testid="stAppViewContainer"], [data-testid="stSidebar"] {{ color: {INK} !important; }}
[data-testid="stAppViewContainer"] p, [data-testid="stAppViewContainer"] span,
[data-testid="stAppViewContainer"] label, [data-testid="stAppViewContainer"] li,
[data-testid="stSidebar"] p, [data-testid="stSidebar"] span, [data-testid="stSidebar"] label,
div[data-testid="stWidgetLabel"] *, [data-testid="stCaptionContainer"] *,
[data-testid="stMarkdownContainer"] * {{ color: {INK} !important; }}
input, textarea, [data-baseweb="input"] input, [data-baseweb="select"] * {{
    color: {INK} !important; -webkit-text-fill-color: {INK} !important; }}
[data-baseweb="input"], [data-baseweb="input"] > div, [data-baseweb="select"] > div,
[data-testid="stNumberInput"] div {{ background-color: #ffffff !important; }}
[data-testid="stNumberInput"] button {{ background: #f0efea !important; color: {INK} !important; }}
input:disabled {{ color: #8a8880 !important; -webkit-text-fill-color: #8a8880 !important;
                  background: #f1f0ec !important; }}
ul[role="listbox"], li[role="option"] {{ background: #ffffff !important; color: {INK} !important; }}
[data-testid="stAlert"] * {{ color: {INK} !important; }}
/* keep white text only where the background is dark */
div[data-testid="stButton"] > button p {{ color: #ffffff !important; }}
section[data-testid="stSidebar"] div[data-testid="stButton"] > button p {{ color: {INK} !important; }}
.badge, .badge * {{ color: #ffffff !important; }}
</style>
""", unsafe_allow_html=True)

@st.cache_resource(show_spinner="Loading model…")
def get_pkg(path):
    return load_pkg(path)

try:
    pkg = get_pkg(PKG_PATH)
except FileNotFoundError:
    st.error(f"**Package not found:** `{PKG_PATH}` — run `make_gui_package.py` first "
             "and put the .pkl next to this script.")
    st.stop()

CAT, NUM = pkg['categorical_cols'], pkg['numeric_cols']
FLOW, STIR, RT = pkg['flow_col'], pkg['stir_col'], pkg['rt_col']
UNIT = pkg['unit']
INT_FEATS = set(pkg['integer_cols'])

GROUPS = [
    ("① Electrode configuration", ['Anode', 'Cathode', 'Effective size (cm2)', 'Spacing (cm)']),
    ("② Catalyst & binder",       ['Cathode Catalyst', 'Binder', 'Binder amount (mL)']),
    ("③ Reactor & hydraulics",    [RT, 'Reactor volume (mL)', FLOW, STIR]),
    ("④ Electrolyte & operation", ['Electrolyte', 'Electrolyte Conc.(mM)', 'Initial pH',
                                   'Current (mA)', 'Reaction time (min)', 'Reusability']),
]
LABELS = {'Effective size (cm2)': 'Effective size (cm²)'}

def rtype_of(v):
    return 'Flow-through' if pkg['is_flow_type'].get(v, False) else 'Batch'

# ── session defaults: training medians / most frequent category ───────────────
def set_defaults(force=False):
    for c in CAT:
        if force or f'inp_{c}' not in st.session_state:
            st.session_state[f'inp_{c}'] = pkg['cat_modes'][c]
    if force or 'inp_Binder' not in st.session_state:
        st.session_state['inp_Binder'] = pkg['binder_default']
    rt = rtype_of(st.session_state[f'inp_{RT}'])
    for f in NUM:
        if force or f'inp_{f}' not in st.session_state:
            st.session_state[f'inp_{f}'] = float(range_for(f, rt, pkg)[2])
set_defaults()

# ── "Random record": fill every input from one real training record ──────────
def load_random_record():
    ex = pkg.get('example_rows')
    if ex is None or len(ex) == 0:
        st.session_state['loaded_record'] = None
        return
    i = random.randrange(len(ex))
    r = ex.iloc[i]
    for c in CAT:
        v = str(r[c]) if pd.notna(r[c]) else pkg['cat_modes'][c]
        st.session_state[f'inp_{c}'] = v if v in pkg['cat_values'][c] else pkg['cat_modes'][c]
    st.session_state['inp_Binder'] = (r['Binder'] if r['Binder'] in pkg['binder_options']
                                      else pkg['binder_default'])
    rt = rtype_of(st.session_state[f'inp_{RT}'])
    for f in NUM:
        lo, hi, med = range_for(f, rt, pkg)
        v = float(r[f]) if pd.notna(r[f]) else med                 # missing → training median
        st.session_state[f'inp_{f}'] = float(min(max(v, lo), hi))
    st.session_state['loaded_record'] = {'row': i, 'title': str(r.get('_title', '')),
                                         'observed': float(r.get('_observed', np.nan))}

# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    if st.button("Reset to defaults", key="reset_btn"):
        set_defaults(force=True)
        st.session_state.pop('loaded_record', None)
        st.rerun()
    st.caption("Defaults are training medians; inputs are limited to the training range.")

# ══════════════════════════════════════════════════════════════════════════════
# MASTHEAD
# ══════════════════════════════════════════════════════════════════════════════
st.markdown(
    '<div class="masthead"><div class="eyebrow">Electrochemical hydrogen peroxide production</div>'
    '<h1 class="page-title">Data-driven prediction of H₂O₂ concentration</h1></div>'
    '<p class="page-subtitle">An XGBoost regression model relating reactor configuration, '
    'electrode chemistry and operating conditions to the H₂O₂ concentration reached at the '
    'specified reaction time. Specify the experimental parameters below to obtain the predicted '
    'concentration, a 90% prediction interval, the Faraday upper bound and a confidence rating '
    'for the input.</p>', unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# INPUTS
# ══════════════════════════════════════════════════════════════════════════════
inputs = {}
rt_now = rtype_of(st.session_state[f'inp_{RT}'])

def num_widget(feat, rtype):
    label = LABELS.get(feat, feat)
    if rtype == 'Batch' and feat == FLOW:
        st.number_input(label, value=0.0, disabled=True, key='dis_flow',
                        help="Batch reactor: flow rate is fixed at 0.")
        return 0.0
    if rtype == 'Flow-through' and feat == STIR:
        st.number_input(label, value=0.0, disabled=True, key='dis_stir',
                        help="Flow-through reactor: stirring speed is fixed at 0.")
        return 0.0
    if feat == 'Binder amount (mL)' and st.session_state.get('inp_Binder') == 'None':
        st.number_input(label, value=0.0, disabled=True, key='dis_binder',
                        help="No binder selected: binder amount is fixed at 0.")
        return 0.0
    lo, hi, med = range_for(feat, rtype, pkg)
    key = f'inp_{feat}'
    v = st.session_state.get(key, med)
    st.session_state[key] = float(min(max(v, lo), hi))        # keep inside the training range
    is_int = feat in INT_FEATS
    return st.number_input(f"{label}  [{lo:g} – {hi:g}]", min_value=float(lo), max_value=float(hi),
                           step=1.0 if is_int else float(max((hi - lo) / 100, 1e-3)),
                           format="%.0f" if is_int else "%.3f", key=key)

def render(container, title, feats):
    k = [g[0] for g in GROUPS].index(title)
    bg, acc, _panel = SECTION_COLORS[k % len(SECTION_COLORS)]
    with container:
        try:
            box = st.container(border=True)          # Streamlit ≥ 1.29
        except TypeError:
            box = st.container()
    with box:
        st.markdown(f'<div class="card-title sec-{k}" style="background:{bg}; border-left-color:{acc};">'
                    f'<span style="color:{acc}">▪</span>{title}</div>', unsafe_allow_html=True)
        for feat in feats:
            if feat == 'Binder':
                inputs['Binder'] = st.selectbox("Binder", pkg['binder_options'], key='inp_Binder')
            elif feat in CAT:
                inputs[feat] = st.selectbox(feat, pkg['cat_values'][feat], key=f'inp_{feat}')
            else:
                inputs[feat] = num_widget(feat, rtype_of(inputs.get(RT, st.session_state[f'inp_{RT}'])))

cols = st.columns(2)
render(cols[0], *GROUPS[0]); render(cols[1], *GROUPS[1])
cols = st.columns(2)
render(cols[0], *GROUPS[2]); render(cols[1], *GROUPS[3])

# ══════════════════════════════════════════════════════════════════════════════
# PREDICT
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("<br>", unsafe_allow_html=True)
_, rcol, bcol, _ = st.columns([0.7, 1.3, 1.6, 0.7])
rcol.button("Random record from the data", key="rand_btn", on_click=load_random_record)
clicked = bcol.button("Predict H₂O₂ concentration", key="predict_btn")
rec = st.session_state.get('loaded_record')
if rec:
    st.caption(f"Inputs loaded from training record #{rec['row'] + 1}"
               + (f" — {rec['title'][:90]}" if rec['title'] else "")
               + (f" · observed H₂O₂ = {rec['observed']:,.2f} {UNIT}" if np.isfinite(rec['observed']) else "")
               + ". Editing any input afterwards changes the case.")

if clicked:
    try:
        r = predict_all(inputs, pkg)
    except Exception as ex:
        st.error(f"Prediction failed: {ex}")
        st.stop()
    t_lab = f"{float(inputs['Reaction time (min)']):g} min"
    obs_html = ""
    if rec and np.isfinite(rec['observed']):
        obs_html = ('<div class="result-sub"><b>Observed in loaded record:</b> '
                    + f"{rec['observed']:,.2f} {UNIT}</div>")

    _, rc, _ = st.columns([0.6, 2.8, 0.6])
    with rc:
        st.markdown(f"""
        <div class="result-box">
          <div class="result-label">Predicted H₂O₂ concentration at t = {t_lab}</div>
          <div class="result-value">{r['pred']:,.2f}</div>
          <div class="result-unit">{UNIT}</div>
          <div class="result-sub"><b>{int(100*PI_LEVEL)}% prediction interval:</b>
               {r['lo']:,.2f} – {r['hi']:,.2f} {UNIT} &nbsp;(÷/× {r['f']:.2f})</div>
          {obs_html}
        </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)

    # ── Check 1: confidence from leverage ─────────────────────────────────────
    col1 = LEVEL_COLORS[r['level']]
    c1.markdown(f"""
    <div class="check" style="border-top-color:{col1}">
      <div class="check-title">1 · Confidence (leverage)</div>
      <div class="check-status" style="color:{col1} !important">{r['level']} confidence</div>
      <div class="check-body">h/h* = {r['h_ratio']:.2f}<br>
      How typical the input is compared with the training data.
      High h/h* ≤ {HR_MED:g} · Medium {HR_MED:g}–{HR_LOW:g} · Low &gt; {HR_LOW:g}.</div>
    </div>""", unsafe_allow_html=True)

    # ── Check 2: Faraday limit ────────────────────────────────────────────────
    fb = r['faraday']
    if not fb:
        st2, col2, body2 = "Not available", MUTED, "Current, time or volume missing."
    else:
        if 'volume' in fb and 'single_pass' in fb:
            bound_txt = (f"{fb['volume']:,.1f} {UNIT} (recirculation) · "
                         f"{fb['single_pass']:,.1f} {UNIT} (single pass)")
        else:
            bound_txt = f"{list(fb.values())[0]:,.1f} {UNIT}"
        if r['exceeds']:
            st2, col2 = "Warning: above the Faraday limit", LEVEL_COLORS['Low']
        else:
            st2, col2 = "No warning: within the Faraday limit", LEVEL_COLORS['High']
        body2 = (f"Upper bound at 100% FE: {bound_txt}<br>"
                 f"Implied Faradaic efficiency: {r['fe_implied']:.1f}%")
    c2.markdown(f"""
    <div class="check" style="border-top-color:{col2}">
      <div class="check-title">2 · Faraday limit</div>
      <div class="check-status" style="color:{col2} !important">{st2}</div>
      <div class="check-body">{body2}<br>
      The charge passed sets the maximum H₂O₂ that can be produced (2 e⁻ per H₂O₂).</div>
    </div>""", unsafe_allow_html=True)

    # ── Check 3: category combination ─────────────────────────────────────────
    if r['combo_seen']:
        st3, col3 = "Seen in training", LEVEL_COLORS['High']
        body3 = "This anode–cathode–catalyst–reactor–electrolyte combination occurs in the training data."
    else:
        st3, col3 = "Unseen: use with caution", LEVEL_COLORS['Medium']
        body3 = ("This exact combination of categories never occurs together in the training data."
                 + (f"<br>Unseen pairs: {', '.join(r['unseen_pairs'])}" if r['unseen_pairs'] else
                    "<br>Every pair occurs, but not all five together."))
    c3.markdown(f"""
    <div class="check" style="border-top-color:{col3}">
      <div class="check-title">3 · Category combination</div>
      <div class="check-status" style="color:{col3} !important">{st3}</div>
      <div class="check-body">{body3}</div>
    </div>""", unsafe_allow_html=True)

    st.markdown('<div class="note">The 90% interval is calibrated on cross-validation errors and '
                'assumes conditions similar to the training studies. The three checks are '
                'independent: each one answers a different question about this prediction.</div>',
                unsafe_allow_html=True)

    # ── Time course: 0 → user's reaction time ─────────────────────────────────
    T_end = float(inputs['Reaction time (min)'])
    if T_end > 0:
        t, cp, clo, chi = predict_curve(inputs, pkg, n_points=10)
        I_ = float(inputs['Current (mA)']); V_ = float(inputs['Reactor volume (mL)'])
        Q_ = float(inputs[FLOW])
        tl = np.linspace(0, T_end, 100)
        fb_vol = [faraday_bound(I_, ti, V_, Q_, r['rtype'] == 'Flow-through', pkg).get('volume', np.nan)
                  for ti in tl]
        fb_sp = faraday_bound(I_, T_end, V_, Q_, r['rtype'] == 'Flow-through', pkg).get('single_pass')

        BLUE = "#3A6EA5"
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=np.r_[t, t[::-1]], y=np.r_[chi, clo[::-1]], fill='toself',
                                 fillcolor='rgba(58,110,165,0.18)', line=dict(width=0),
                                 hoverinfo='skip', name=f'{int(100*PI_LEVEL)}% prediction interval'))
        fig.add_trace(go.Scatter(x=t, y=cp, mode='lines+markers', name='Predicted',
                                 line=dict(color='#9a9a9a', width=1.6),
                                 marker=dict(symbol='circle', size=9, color=BLUE,
                                             line=dict(color='white', width=1)),
                                 hovertemplate='t = %{x:.1f} min<br>C = %{y:.2f} mg/L<extra></extra>'))
        if np.isfinite(fb_vol).any():
            fig.add_trace(go.Scatter(x=tl, y=fb_vol, mode='lines', name='Faraday limit (100% FE)'
                                     + (', recirculation' if fb_sp else ''),
                                     line=dict(color='#D62728', width=2),
                                     hovertemplate='t = %{x:.1f} min<br>limit = %{y:.1f} mg/L<extra></extra>'))
        if r['rtype'] == 'Batch' and np.isfinite(fb_vol).any():
            o2 = O2_SAT_MG_L * 34.0147 / 31.998 + F_ANODE * np.asarray(fb_vol)
            fig.add_trace(go.Scatter(x=tl, y=o2, mode='lines', name='O₂ budget',
                                     line=dict(color='#7B3FA0', width=2),
                                     hovertemplate='t = %{x:.1f} min<br>O₂ budget = %{y:.1f} mg/L<extra></extra>'))
        if fb_sp:
            fig.add_trace(go.Scatter(x=[0, T_end], y=[fb_sp, fb_sp], mode='lines',
                                     name='Faraday limit (100% FE), single pass',
                                     line=dict(color='#D62728', width=2, dash='dash')))
        top = float(np.nanmax(chi)) * 1.25
        lim_end = np.nanmax([v for v in [fb_vol[-1], fb_sp] if v is not None])
        y_max = max(top, min(lim_end * 1.05, 3 * top)) if np.isfinite(lim_end) else top
        axis = dict(showline=True, linecolor=INK, mirror=True, ticks='outside', showgrid=True,
                    gridcolor='#ecebe6', zeroline=False, tickfont=dict(size=13, color=INK),
                    title_font=dict(size=15, color=INK))
        fig.update_layout(height=470, plot_bgcolor='white', paper_bgcolor='white',
                          font=dict(family='Inter', color=INK, size=13),
                          margin=dict(l=70, r=30, t=40, b=60),
                          xaxis={**axis, 'title': 'Reaction time (min)', 'range': [0, T_end * 1.02]},
                          yaxis={**axis, 'title': f'H₂O₂ concentration ({UNIT})', 'range': [0, y_max]},
                          legend=dict(orientation='h', yanchor='bottom', y=1.01, x=0,
                                      font=dict(size=13, color=INK)))
        st.markdown("#### Predicted time course")
        st.plotly_chart(fig, use_container_width=True)
        st.markdown(f'<div class="note">Model predictions at 10 equally spaced times from 0 to '
                    f'{T_end:g} min (C = 0 at t = 0), all other inputs fixed. The shaded band is the '
                    f'{int(100*PI_LEVEL)}% prediction interval; the red line is the maximum H₂O₂ the '
                    'charge passed can produce (100% Faradaic efficiency). For batch reactors the purple line '
                    f'is the O₂ budget: dissolved O₂ at the start ({O2_SAT_MG_L:g} mg/L) plus O₂ '
                    'produced at the anode; above it, extra O₂ (aeration or sparging) would be needed.</div>',
                    unsafe_allow_html=True)