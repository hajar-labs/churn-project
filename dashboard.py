import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import pickle
from pathlib import Path

# ── Fallback: train RF only if XGBoost pkl not found ──────────────────────────
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix
import warnings
warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Churn Analytics Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─────────────────────────────────────────────
# CUSTOM CSS
# ─────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');
    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
    .main { background: #0f1117; }
    .block-container { padding: 1.5rem 2rem; }
    .kpi-card {
        background: linear-gradient(135deg, #1e2130 0%, #252a3d 100%);
        border: 1px solid #2d3250; border-radius: 16px;
        padding: 1.4rem 1.6rem; text-align: center;
        box-shadow: 0 4px 20px rgba(0,0,0,0.3); transition: transform 0.2s;
    }
    .kpi-card:hover { transform: translateY(-3px); }
    .kpi-label { color: #8b9cc8; font-size: 0.75rem; font-weight: 600;
                 letter-spacing: 0.1em; text-transform: uppercase; margin-bottom: 0.5rem; }
    .kpi-value { color: #ffffff; font-size: 2rem; font-weight: 700; line-height: 1; }
    .kpi-delta-pos { color: #4ade80; font-size: 0.8rem; margin-top: 0.3rem; }
    .kpi-delta-neg { color: #f87171; font-size: 0.8rem; margin-top: 0.3rem; }
    .section-title {
        color: #c7d2fe; font-size: 1.1rem; font-weight: 600;
        margin: 1.5rem 0 0.8rem 0; padding-left: 0.5rem;
        border-left: 3px solid #6366f1;
    }
    [data-testid="stSidebar"] { background: #13151f; border-right: 1px solid #2d3250; }
    .js-plotly-plot { border-radius: 12px; }
    .stTabs [data-baseweb="tab-list"] { background: #1e2130; border-radius: 10px; padding: 4px; }
    .stTabs [data-baseweb="tab"] { background: transparent; color: #8b9cc8; border-radius: 8px; }
    .stTabs [aria-selected="true"] { background: #6366f1 !important; color: white !important; }
    .pred-high { background: linear-gradient(135deg, #7f1d1d, #991b1b);
                 border: 1px solid #ef4444; border-radius: 12px; padding: 1rem; }
    .pred-low  { background: linear-gradient(135deg, #14532d, #166534);
                 border: 1px solid #22c55e; border-radius: 12px; padding: 1rem; }
    .pred-text { color: white; font-size: 1.3rem; font-weight: 700; text-align: center; }
    .pred-sub  { color: rgba(255,255,255,0.7); font-size: 0.85rem; text-align: center; margin-top: 0.3rem; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# DATA LOADING
# ─────────────────────────────────────────────
@st.cache_data
def load_data():
    try:
        url = "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv"
        df = pd.read_csv(url)
    except Exception:
        np.random.seed(42)
        n = 7043
        genders    = np.random.choice(['Male', 'Female'], n)
        senior     = np.random.choice([0, 1], n, p=[0.84, 0.16])
        partner    = np.random.choice(['Yes', 'No'], n)
        dependents = np.random.choice(['Yes', 'No'], n, p=[0.3, 0.7])
        tenure     = np.random.randint(0, 73, n)
        phone      = np.random.choice(['Yes', 'No'], n, p=[0.9, 0.1])
        multilines = np.where(phone == 'No', 'No phone service',
                              np.random.choice(['Yes', 'No'], n))
        internet   = np.random.choice(['DSL', 'Fiber optic', 'No'], n, p=[0.34, 0.44, 0.22])
        contract   = np.random.choice(['Month-to-month', 'One year', 'Two year'], n, p=[0.55, 0.21, 0.24])
        paperless  = np.random.choice(['Yes', 'No'], n, p=[0.59, 0.41])
        payment    = np.random.choice(['Electronic check', 'Mailed check',
                                       'Bank transfer (automatic)', 'Credit card (automatic)'], n)
        monthly    = np.round(np.random.uniform(18, 118, n), 2)
        total      = np.clip(np.round(monthly * tenure + np.random.normal(0, 50, n), 2), 0, None)
        churn_prob = np.clip(0.1 + 0.3*(internet=='Fiber optic')
                             + 0.25*(contract=='Month-to-month') - 0.003*tenure, 0.02, 0.95)
        churn      = np.where(np.random.random(n) < churn_prob, 'Yes', 'No')
        df = pd.DataFrame({
            'customerID': [f'ID-{i:04d}' for i in range(n)],
            'gender': genders, 'SeniorCitizen': senior,
            'Partner': partner, 'Dependents': dependents, 'tenure': tenure,
            'PhoneService': phone, 'MultipleLines': multilines,
            'InternetService': internet, 'Contract': contract,
            'PaperlessBilling': paperless, 'PaymentMethod': payment,
            'MonthlyCharges': monthly, 'TotalCharges': total.astype(str), 'Churn': churn
        })
    df['TotalCharges']      = pd.to_numeric(df['TotalCharges'], errors='coerce').fillna(0)
    df['Churn_binary']      = (df['Churn'] == 'Yes').astype(int)
    df['SeniorCitizen_label'] = df['SeniorCitizen'].map({0: 'No', 1: 'Yes'})
    return df

# ─────────────────────────────────────────────
# BINARY ENCODING  (mirrors 01_cleaning.ipynb)
# ─────────────────────────────────────────────
CAT_FEATURES = ['InternetService', 'Contract', 'PaymentMethod']
NUM_FEATURES = ['SeniorCitizen', 'tenure', 'MonthlyCharges', 'TotalCharges']
BIN_COLS     = ['gender', 'Partner', 'Dependents',
                'PhoneService', 'MultipleLines', 'PaperlessBilling']

def encode_binary(df_raw):
    d = df_raw.copy()
    # Yes → 1, everything else (No / No phone service / Female) → 0
    for col in BIN_COLS:
        d[col + '_bin'] = (d[col] == 'Yes').astype(int)
    # gender: Male=1
    d['gender_bin'] = (d['gender'] == 'Male').astype(int)
    return d

BIN_FEATURES = [c + '_bin' for c in BIN_COLS]
ALL_FEATURES  = NUM_FEATURES + BIN_FEATURES + CAT_FEATURES

# ─────────────────────────────────────────────
# MODEL  — load XGBoost pkl, fallback to RF
# ─────────────────────────────────────────────
PKL_PATHS = [Path("data/model_xgb.pkl"), Path("model_xgb.pkl")]

@st.cache_resource
def load_or_train_model(_df):
    # ── Try XGBoost pkl ────────────────────────────────────────────────────
    for p in PKL_PATHS:
        if p.exists():
            try:
                with open(p, 'rb') as f:
                    b = pickle.load(f)
                if all(k in b for k in ('model', 'preprocessor', 'feature_names', 'threshold', 'metrics')):
                    return b, "XGBoost (tuned)"
            except Exception:
                pass

    # ── Fallback: sklearn Pipeline (RF + OHE inside Pipeline) ─────────────
    df_enc = encode_binary(_df)
    X = df_enc[ALL_FEATURES].copy()
    y = df_enc['Churn_binary']

    preprocessor = ColumnTransformer(
        [('ohe', OneHotEncoder(handle_unknown='ignore', sparse_output=False), CAT_FEATURES)],
        remainder='passthrough'
    )
    pipeline = Pipeline([
        ('pre', preprocessor),
        ('clf', RandomForestClassifier(n_estimators=200, max_depth=8,
                                        random_state=42, n_jobs=-1,
                                        class_weight='balanced'))
    ])
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42)
    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)
    y_prob = pipeline.predict_proba(X_test)[:, 1]
    acc    = accuracy_score(y_test, y_pred)
    auc    = roc_auc_score(y_test, y_prob)
    cm_val = confusion_matrix(y_test, y_pred)

    ohe_cols = preprocessor.named_transformers_['ohe'].get_feature_names_out(CAT_FEATURES).tolist()
    exp_feat = ohe_cols + NUM_FEATURES + BIN_FEATURES
    importances = pd.Series(
        pipeline.named_steps['clf'].feature_importances_, index=exp_feat
    ).sort_values(ascending=True).tail(15)

    return {
        'model': pipeline, 'preprocessor': None,
        'feature_names': ALL_FEATURES, 'threshold': 0.40,
        'metrics': {'roc_auc': round(auc, 4), 'accuracy': round(acc, 4),
                    'cm': cm_val, 'importances': importances},
        '_is_fallback': True,
    }, "Random Forest (fallback)"


def get_metrics(bundle, df):
    if bundle.get('_is_fallback'):
        m = bundle['metrics']
        return m['accuracy'], m['roc_auc'], m['cm'], m['importances']
    # XGBoost bundle ── recompute on full data for display
    df_enc = encode_binary(df)
    feats  = bundle['feature_names']
    avail  = [f for f in feats if f in df_enc.columns]
    X      = df_enc[avail] if len(avail) == len(feats) else df_enc[df.columns.intersection(feats)]
    try:
        X_pre  = bundle['preprocessor'].transform(X)
        y_prob = bundle['model'].predict_proba(X_pre)[:, 1]
    except Exception:
        y_prob = bundle['model'].predict_proba(X)[:, 1]
    thr    = bundle['threshold']
    y_pred = (y_prob >= thr).astype(int)
    y_true = df['Churn_binary'].values
    acc    = accuracy_score(y_true, y_pred)
    auc    = bundle['metrics']['roc_auc']
    cm_val = confusion_matrix(y_true, y_pred)
    try:
        imp = pd.Series(bundle['model'].feature_importances_,
                        index=bundle['feature_names']).sort_values(ascending=True).tail(15)
    except Exception:
        imp = pd.Series(dtype=float)
    return acc, auc, cm_val, imp


def predict_single(bundle, input_dict):
    raw_df = pd.DataFrame([input_dict])
    enc_df = encode_binary(raw_df)
    if bundle.get('_is_fallback'):
        X = enc_df[ALL_FEATURES]
        return bundle['model'].predict_proba(X)[0][1]
    # XGBoost bundle
    feats = bundle['feature_names']
    avail = [f for f in feats if f in enc_df.columns]
    X_raw = enc_df[avail] if len(avail) == len(feats) else enc_df
    try:
        X_pre = bundle['preprocessor'].transform(X_raw)
        return bundle['model'].predict_proba(X_pre)[0][1]
    except Exception:
        return bundle['model'].predict_proba(X_raw)[0][1]


# ─────────────────────────────────────────────
# INIT
# ─────────────────────────────────────────────
COLORS = {
    'churn': '#f87171', 'stay': '#4ade80', 'primary': '#6366f1',
    'secondary': '#8b5cf6', 'accent': '#06b6d4',
    'bg': '#1e2130', 'grid': '#2d3250', 'text': '#c7d2fe',
}

def chart_layout(fig, title="", height=380):
    fig.update_layout(
        title=dict(text=title, font=dict(size=14, color=COLORS['text']), x=0.02),
        plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color=COLORS['text'], family='Inter'),
        height=height, margin=dict(l=10, r=10, t=40, b=10),
        legend=dict(bgcolor='rgba(0,0,0,0)', font=dict(size=11)),
        xaxis=dict(gridcolor=COLORS['grid'], linecolor=COLORS['grid'], tickfont=dict(size=11)),
        yaxis=dict(gridcolor=COLORS['grid'], linecolor=COLORS['grid'], tickfont=dict(size=11)),
    )
    return fig

df = load_data()
bundle, model_name = load_or_train_model(df)
accuracy, auc, cm, importances = get_metrics(bundle, df)
threshold = bundle['threshold']

# ─────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🔮 Churn Analytics")
    st.markdown("<hr style='border-color:#2d3250;margin:0.5rem 0 1rem'>", unsafe_allow_html=True)
    st.markdown("### 🎛️ Filtres")

    selected_contract = st.multiselect("Type de contrat",
        df['Contract'].unique().tolist(), default=df['Contract'].unique().tolist())
    selected_internet = st.multiselect("Service Internet",
        df['InternetService'].unique().tolist(), default=df['InternetService'].unique().tolist())
    tenure_range  = st.slider("Ancienneté (mois)", 0, 72, (0, 72))
    monthly_range = st.slider("Charges mensuelles ($)",
        int(df['MonthlyCharges'].min()), int(df['MonthlyCharges'].max()),
        (int(df['MonthlyCharges'].min()), int(df['MonthlyCharges'].max())))

    st.markdown("<hr style='border-color:#2d3250;margin:1rem 0'>", unsafe_allow_html=True)
    badge = "#6366f1" if not bundle.get('_is_fallback') else "#f59e0b"
    st.markdown(
        f"**Modèle:** <span style='background:{badge};color:white;padding:2px 8px;"
        f"border-radius:999px;font-size:0.75rem;font-weight:600'>{model_name}</span>",
        unsafe_allow_html=True)
    st.markdown(f"**Accuracy:** `{accuracy:.1%}`")
    st.markdown(f"**AUC-ROC:** `{auc:.3f}`")
    st.markdown(f"**Seuil décision:** `{threshold:.2f}`")
    st.markdown(f"**Dataset:** `{len(df):,}` clients")
    if bundle.get('_is_fallback'):
        st.info("💡 Place `data/model_xgb.pkl` next to the app to load your tuned XGBoost model.", icon="ℹ️")

# ─────────────────────────────────────────────
# FILTER
# ─────────────────────────────────────────────
filtered = df[
    (df['Contract'].isin(selected_contract)) &
    (df['InternetService'].isin(selected_internet)) &
    (df['tenure'].between(*tenure_range)) &
    (df['MonthlyCharges'].between(*monthly_range))
]
churn_df    = filtered[filtered['Churn'] == 'Yes']
stay_df     = filtered[filtered['Churn'] == 'No']
churn_rate  = filtered['Churn_binary'].mean()
monthly_rev = filtered['MonthlyCharges'].sum()
lost_rev    = churn_df['MonthlyCharges'].sum()

# ─────────────────────────────────────────────
# HEADER + KPIs
# ─────────────────────────────────────────────
st.markdown("""
<div style='padding:1rem 0 0.5rem'>
  <h1 style='color:white;font-size:1.8rem;margin:0;font-weight:700'>📊 Customer Churn Dashboard</h1>
  <p style='color:#8b9cc8;margin:0.3rem 0 0;font-size:0.9rem'>
    Analyse prédictive de l'attrition clients — Telco Dataset
  </p>
</div>""", unsafe_allow_html=True)

k1, k2, k3, k4, k5 = st.columns(5)
with k1:
    st.markdown(f"""<div class="kpi-card"><div class="kpi-label">Total Clients</div>
        <div class="kpi-value">{len(filtered):,}</div>
        <div class="kpi-delta-pos">↑ Dataset filtré</div></div>""", unsafe_allow_html=True)
with k2:
    cc = "kpi-delta-neg" if churn_rate > 0.2 else "kpi-delta-pos"
    st.markdown(f"""<div class="kpi-card"><div class="kpi-label">Taux de Churn</div>
        <div class="kpi-value" style="color:#f87171">{churn_rate:.1%}</div>
        <div class="{cc}">{'⚠ Critique' if churn_rate > 0.25 else '✓ Acceptable'}</div></div>""",
        unsafe_allow_html=True)
with k3:
    st.markdown(f"""<div class="kpi-card"><div class="kpi-label">Clients Perdus</div>
        <div class="kpi-value" style="color:#f87171">{len(churn_df):,}</div>
        <div class="kpi-delta-neg">↓ {len(churn_df)/len(filtered)*100:.1f}% du total</div></div>""",
        unsafe_allow_html=True)
with k4:
    st.markdown(f"""<div class="kpi-card"><div class="kpi-label">Revenu Mensuel</div>
        <div class="kpi-value">${monthly_rev:,.0f}</div>
        <div class="kpi-delta-pos">$ actif/mois</div></div>""", unsafe_allow_html=True)
with k5:
    st.markdown(f"""<div class="kpi-card"><div class="kpi-label">Revenu Perdu/Mois</div>
        <div class="kpi-value" style="color:#f87171">${lost_rev:,.0f}</div>
        <div class="kpi-delta-neg">↑ dû au churn</div></div>""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# TABS
# ─────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs([
    "📈 Vue Générale", "🔍 Analyse Segments", "🤖 Modèle ML", "🎯 Prédiction Client"
])

# ── TAB 1 ────────────────────────────────────
with tab1:
    c1, c2 = st.columns([1, 2])
    with c1:
        fig = go.Figure(go.Pie(
            values=[len(churn_df), len(stay_df)], labels=['Churned', 'Actif'],
            hole=0.65, marker=dict(colors=[COLORS['churn'], COLORS['stay']]),
            textinfo='percent', textfont=dict(size=13, color='white')))
        fig.add_annotation(text=f"<b>{churn_rate:.0%}</b><br><span style='font-size:10px'>CHURN</span>",
            x=0.5, y=0.5, showarrow=False, font=dict(size=20, color=COLORS['churn']))
        chart_layout(fig, "Répartition Churn", 320); st.plotly_chart(fig, use_container_width=True)
    with c2:
        ct = filtered.groupby(['Contract', 'Churn']).size().reset_index(name='count')
        fig = px.bar(ct, x='Contract', y='count', color='Churn', barmode='group',
            color_discrete_map={'Yes': COLORS['churn'], 'No': COLORS['stay']},
            labels={'count': 'Clients', 'Contract': 'Type de Contrat'})
        chart_layout(fig, "Churn par Type de Contrat", 320)
        fig.update_traces(marker_line_width=0, opacity=0.9); st.plotly_chart(fig, use_container_width=True)

    c3, c4 = st.columns(2)
    with c3:
        fig = go.Figure()
        fig.add_trace(go.Histogram(x=churn_df['tenure'], name='Churned',
            marker_color=COLORS['churn'], opacity=0.7, nbinsx=30))
        fig.add_trace(go.Histogram(x=stay_df['tenure'], name='Actif',
            marker_color=COLORS['stay'], opacity=0.7, nbinsx=30))
        fig.update_layout(barmode='overlay')
        chart_layout(fig, "Distribution Ancienneté (mois)", 320); st.plotly_chart(fig, use_container_width=True)
    with c4:
        fig = go.Figure()
        fig.add_trace(go.Box(y=churn_df['MonthlyCharges'], name='Churned',
            marker_color=COLORS['churn'], fillcolor='rgba(248,113,113,0.2)', line_color=COLORS['churn']))
        fig.add_trace(go.Box(y=stay_df['MonthlyCharges'], name='Actif',
            marker_color=COLORS['stay'], fillcolor='rgba(74,222,128,0.2)', line_color=COLORS['stay']))
        chart_layout(fig, "Charges Mensuelles ($)", 320); st.plotly_chart(fig, use_container_width=True)

# ── TAB 2 ────────────────────────────────────
with tab2:
    c1, c2 = st.columns(2)
    with c1:
        seg = filtered.groupby('InternetService')['Churn_binary'].agg(['mean','count']).reset_index()
        seg.columns = ['InternetService','churn_rate','count']
        fig = px.bar(seg, x='InternetService', y='churn_rate', color='churn_rate',
            color_continuous_scale=['#4ade80','#f59e0b','#f87171'],
            text=seg['churn_rate'].apply(lambda x: f'{x:.1%}'),
            labels={'churn_rate':'Taux de Churn','InternetService':'Service Internet'})
        fig.update_traces(textposition='outside', marker_line_width=0)
        fig.update_coloraxes(showscale=False)
        chart_layout(fig, "Taux de Churn par Service Internet", 320); st.plotly_chart(fig, use_container_width=True)
    with c2:
        seg = filtered.groupby('PaymentMethod')['Churn_binary'].mean().reset_index()
        seg.columns = ['PaymentMethod','churn_rate']
        seg['PaymentMethod'] = seg['PaymentMethod'].str.replace(' (automatic)', '\n(auto)', regex=False)
        fig = px.bar(seg.sort_values('churn_rate'), x='churn_rate', y='PaymentMethod',
            orientation='h', color='churn_rate',
            color_continuous_scale=['#4ade80','#f59e0b','#f87171'],
            text=seg.sort_values('churn_rate')['churn_rate'].apply(lambda x: f'{x:.1%}'),
            labels={'churn_rate':'Taux de Churn','PaymentMethod':''})
        fig.update_traces(textposition='outside', marker_line_width=0)
        fig.update_coloraxes(showscale=False)
        chart_layout(fig, "Taux de Churn par Mode de Paiement", 320); st.plotly_chart(fig, use_container_width=True)

    c3, c4 = st.columns(2)
    with c3:
        sample = filtered.sample(min(2000, len(filtered)), random_state=42)
        fig = px.scatter(sample, x='tenure', y='MonthlyCharges', color='Churn',
            color_discrete_map={'Yes': COLORS['churn'], 'No': COLORS['stay']}, opacity=0.5,
            labels={'tenure':'Ancienneté (mois)','MonthlyCharges':'Charges/Mois ($)'},
            hover_data=['Contract','InternetService'])
        chart_layout(fig, "Ancienneté vs Charges Mensuelles", 340); st.plotly_chart(fig, use_container_width=True)
    with c4:
        pivot = filtered.groupby(['SeniorCitizen_label','gender'])['Churn_binary'].mean().unstack()
        fig = go.Figure(go.Heatmap(
            z=pivot.values, x=pivot.columns.tolist(), y=pivot.index.tolist(),
            colorscale=[[0,'#4ade80'],[0.5,'#f59e0b'],[1,'#f87171']],
            text=[[f'{v:.1%}' for v in row] for row in pivot.values],
            texttemplate="%{text}", textfont=dict(size=16, color='white'),
            showscale=True, zmin=0, zmax=0.5))
        chart_layout(fig, "Taux de Churn : Senior × Genre", 320); st.plotly_chart(fig, use_container_width=True)

# ── TAB 3 ────────────────────────────────────
with tab3:
    tn, fp, fn, tp = cm.ravel()
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0

    m1, m2, m3 = st.columns(3)
    with m1:
        st.markdown(f"""<div class="kpi-card" style="margin-top:0.5rem">
            <div class="kpi-label">Accuracy</div>
            <div class="kpi-value" style="color:#6366f1">{accuracy:.1%}</div>
            <div class="kpi-delta-pos">✓ {model_name}</div></div>""", unsafe_allow_html=True)
    with m2:
        st.markdown(f"""<div class="kpi-card" style="margin-top:0.5rem">
            <div class="kpi-label">AUC-ROC</div>
            <div class="kpi-value" style="color:#06b6d4">{auc:.3f}</div>
            <div class="kpi-delta-pos">{'✓ Excellent' if auc > 0.85 else '✓ Good'}</div></div>""",
            unsafe_allow_html=True)
    with m3:
        st.markdown(f"""<div class="kpi-card" style="margin-top:0.5rem">
            <div class="kpi-label">Recall (Churn)</div>
            <div class="kpi-value" style="color:#f59e0b">{recall:.1%}</div>
            <div class="kpi-delta-pos">Precision: {precision:.1%}</div></div>""", unsafe_allow_html=True)

    ml1, ml2 = st.columns(2)
    with ml1:
        if len(importances) > 0:
            fig = go.Figure(go.Bar(
                x=importances.values, y=importances.index, orientation='h',
                marker=dict(color=importances.values,
                    colorscale=[[0,'#2d3250'],[0.5,'#6366f1'],[1,'#06b6d4']], line_width=0)))
            chart_layout(fig, "Feature Importance", 400); st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Feature importances not available for this model bundle.")
    with ml2:
        fig = go.Figure(go.Heatmap(
            z=cm, x=['Predicted: Active', 'Predicted: Churn'],
            y=['Actual: Active', 'Actual: Churn'],
            colorscale=[[0,'#1e2130'],[1,'#6366f1']],
            text=cm, texttemplate='<b>%{text}</b>',
            textfont=dict(size=20, color='white'), showscale=False))
        for xi, yi, lbl in [(0,0,'TN'),(1,0,'FP'),(0,1,'FN'),(1,1,'TP')]:
            fig.add_annotation(x=xi, y=yi, text=lbl, showarrow=False,
                font=dict(size=11, color='#8b9cc8'), yshift=-20)
        chart_layout(fig, "Confusion Matrix", 400); st.plotly_chart(fig, use_container_width=True)

# ── TAB 4 ────────────────────────────────────
with tab4:
    st.markdown('<div class="section-title">🎯 Predict Churn Risk for a Customer</div>',
                unsafe_allow_html=True)
    p1, p2, p3 = st.columns(3)

    with p1:
        gender     = st.selectbox("Gender", ["Male", "Female"])
        senior     = st.selectbox("Senior Citizen", ["No", "Yes"])
        partner    = st.selectbox("Partner", ["Yes", "No"])
        dependents = st.selectbox("Dependents", ["Yes", "No"])
        tenure_val = st.slider("Tenure (months)", 0, 72, 12)
    with p2:
        phone      = st.selectbox("Phone Service", ["Yes", "No"])
        multilines = st.selectbox("Multiple Lines", ["Yes", "No", "No phone service"])
        internet   = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
        contract   = st.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])
        paperless  = st.selectbox("Paperless Billing", ["Yes", "No"])
    with p3:
        payment = st.selectbox("Payment Method", [
            "Electronic check", "Mailed check",
            "Bank transfer (automatic)", "Credit card (automatic)"])
        monthly_val = st.slider("Monthly Charges ($)", 18, 120, 65)
        total_val   = monthly_val * tenure_val
        st.markdown(f"**Estimated Total Charges:** ${total_val:,.0f}")

    if st.button("🔮 Predict Churn Risk", use_container_width=True):
        input_dict = {
            'SeniorCitizen': 1 if senior == "Yes" else 0,
            'tenure': tenure_val, 'MonthlyCharges': monthly_val,
            'TotalCharges': float(total_val),
            'gender': gender, 'Partner': partner, 'Dependents': dependents,
            'PhoneService': phone, 'MultipleLines': multilines,
            'InternetService': internet, 'Contract': contract,
            'PaperlessBilling': paperless, 'PaymentMethod': payment,
        }
        proba = predict_single(bundle, input_dict)

        r1, r2 = st.columns([1, 2])
        with r1:
            if proba > threshold:
                st.markdown(f"""<div class="pred-high">
                    <div class="pred-text">⚠️ CHURN RISK</div>
                    <div class="pred-sub">Probability: {proba:.1%}</div></div>""",
                    unsafe_allow_html=True)
            else:
                st.markdown(f"""<div class="pred-low">
                    <div class="pred-text">✅ ACTIVE CLIENT</div>
                    <div class="pred-sub">Retention probability: {1-proba:.1%}</div></div>""",
                    unsafe_allow_html=True)
        with r2:
            gauge_color = COLORS['churn'] if proba > threshold else COLORS['stay']
            fig = go.Figure(go.Indicator(
                mode="gauge+number+delta", value=proba * 100,
                domain={'x': [0, 1], 'y': [0, 1]},
                title={'text': "Churn Risk Score (%)", 'font': {'size': 14, 'color': COLORS['text']}},
                delta={'reference': 26.5, 'increasing': {'color': COLORS['churn']},
                       'decreasing': {'color': COLORS['stay']}},
                gauge={
                    'axis': {'range': [0, 100], 'tickcolor': COLORS['text'],
                             'tickfont': {'color': COLORS['text']}},
                    'bar': {'color': gauge_color}, 'bgcolor': COLORS['bg'],
                    'steps': [
                        {'range': [0, 30],   'color': 'rgba(74,222,128,0.15)'},
                        {'range': [30, 60],  'color': 'rgba(245,158,11,0.15)'},
                        {'range': [60, 100], 'color': 'rgba(248,113,113,0.15)'},
                    ],
                    'threshold': {'line': {'color': 'white', 'width': 2},
                                  'thickness': 0.75, 'value': threshold * 100}
                },
                number={'suffix': '%', 'font': {'color': gauge_color, 'size': 36}}
            ))
            fig.update_layout(paper_bgcolor='rgba(0,0,0,0)',
                font=dict(color=COLORS['text']), height=250,
                margin=dict(l=20, r=20, t=30, b=10))
            st.plotly_chart(fig, use_container_width=True)

        st.markdown('<div class="section-title">💡 Retention Recommendations</div>',
                    unsafe_allow_html=True)
        recs = []
        if contract == "Month-to-month":
            recs.append("📋 Offer a 1- or 2-year commitment plan with a discount")
        if internet == "Fiber optic" and proba > 0.4:
            recs.append("📡 Check Fiber network satisfaction — quality issues drive churn")
        if payment == "Electronic check":
            recs.append("💳 Encourage automatic payment — linked to higher retention")
        if tenure_val < 12:
            recs.append("🎁 Enroll in new-customer loyalty program (0–12 months)")
        if monthly_val > 80:
            recs.append("💰 Offer a tailored plan or bundle to reduce monthly cost")
        if multilines == "No" and phone == "Yes":
            recs.append("📱 Upsell Multiple Lines — add-ons correlate with lower churn")
        if not recs:
            recs.append("✅ Low-risk customer — maintain current service quality")
        for r in recs:
            st.markdown(f"- {r}")

# ─────────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────────
st.markdown("<br>", unsafe_allow_html=True)
st.markdown(f"""
<div style='text-align:center;color:#4a5568;font-size:0.75rem;
            padding:1rem 0;border-top:1px solid #2d3250'>
    Churn Analytics Dashboard &nbsp;•&nbsp; IBM Telco Dataset &nbsp;•&nbsp;
    {model_name} &nbsp;|&nbsp; Threshold: {threshold:.2f} &nbsp;|&nbsp; AUC-ROC: {auc:.3f}
</div>
""", unsafe_allow_html=True)
