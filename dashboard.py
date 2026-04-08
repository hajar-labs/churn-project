import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
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

    /* KPI Cards */
    .kpi-card {
        background: linear-gradient(135deg, #1e2130 0%, #252a3d 100%);
        border: 1px solid #2d3250;
        border-radius: 16px;
        padding: 1.4rem 1.6rem;
        text-align: center;
        box-shadow: 0 4px 20px rgba(0,0,0,0.3);
        transition: transform 0.2s;
    }
    .kpi-card:hover { transform: translateY(-3px); }
    .kpi-label { color: #8b9cc8; font-size: 0.75rem; font-weight: 600; letter-spacing: 0.1em; text-transform: uppercase; margin-bottom: 0.5rem; }
    .kpi-value { color: #ffffff; font-size: 2rem; font-weight: 700; line-height: 1; }
    .kpi-delta-pos { color: #4ade80; font-size: 0.8rem; margin-top: 0.3rem; }
    .kpi-delta-neg { color: #f87171; font-size: 0.8rem; margin-top: 0.3rem; }

    /* Section headers */
    .section-title {
        color: #c7d2fe;
        font-size: 1.1rem;
        font-weight: 600;
        margin: 1.5rem 0 0.8rem 0;
        padding-left: 0.5rem;
        border-left: 3px solid #6366f1;
    }

    /* Sidebar */
    [data-testid="stSidebar"] {
        background: #13151f;
        border-right: 1px solid #2d3250;
    }
    [data-testid="stSidebar"] .css-1d391kg { padding: 1rem; }

    /* Plotly chart background */
    .js-plotly-plot { border-radius: 12px; }

    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] { background: #1e2130; border-radius: 10px; padding: 4px; }
    .stTabs [data-baseweb="tab"] { background: transparent; color: #8b9cc8; border-radius: 8px; }
    .stTabs [aria-selected="true"] { background: #6366f1 !important; color: white !important; }

    /* Prediction card */
    .pred-high { background: linear-gradient(135deg, #7f1d1d, #991b1b); border: 1px solid #ef4444; border-radius: 12px; padding: 1rem; }
    .pred-low { background: linear-gradient(135deg, #14532d, #166534); border: 1px solid #22c55e; border-radius: 12px; padding: 1rem; }
    .pred-text { color: white; font-size: 1.3rem; font-weight: 700; text-align: center; }
    .pred-sub { color: rgba(255,255,255,0.7); font-size: 0.85rem; text-align: center; margin-top: 0.3rem; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# DATA LOADING & PREPROCESSING
# ─────────────────────────────────────────────
@st.cache_data
def load_data():
    """Load Telco churn dataset (from GitHub or generate synthetic)"""
    try:
        url = "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv"
        df = pd.read_csv(url)
    except:
        # Fallback: synthetic dataset matching Telco schema
        np.random.seed(42)
        n = 7043
        genders = np.random.choice(['Male', 'Female'], n)
        senior = np.random.choice([0, 1], n, p=[0.84, 0.16])
        partner = np.random.choice(['Yes', 'No'], n)
        dependents = np.random.choice(['Yes', 'No'], n, p=[0.3, 0.7])
        tenure = np.random.randint(0, 73, n)
        phone = np.random.choice(['Yes', 'No'], n, p=[0.9, 0.1])
        multilines = np.where(phone == 'No', 'No phone service', np.random.choice(['Yes', 'No'], n))
        internet = np.random.choice(['DSL', 'Fiber optic', 'No'], n, p=[0.34, 0.44, 0.22])
        contract = np.random.choice(['Month-to-month', 'One year', 'Two year'], n, p=[0.55, 0.21, 0.24])
        paperless = np.random.choice(['Yes', 'No'], n, p=[0.59, 0.41])
        payment = np.random.choice(['Electronic check', 'Mailed check', 'Bank transfer (automatic)', 'Credit card (automatic)'], n)
        monthly = np.round(np.random.uniform(18, 118, n), 2)
        total = np.round(monthly * tenure + np.random.normal(0, 50, n), 2)
        total = np.clip(total, 0, None)
        # Churn logic (higher for fiber + month-to-month + short tenure)
        churn_prob = 0.1 + 0.3*(internet=='Fiber optic') + 0.25*(contract=='Month-to-month') - 0.003*tenure
        churn_prob = np.clip(churn_prob, 0.02, 0.95)
        churn = np.where(np.random.random(n) < churn_prob, 'Yes', 'No')
        df = pd.DataFrame({
            'customerID': [f'ID-{i:04d}' for i in range(n)],
            'gender': genders, 'SeniorCitizen': senior,
            'Partner': partner, 'Dependents': dependents, 'tenure': tenure,
            'PhoneService': phone, 'MultipleLines': multilines,
            'InternetService': internet, 'Contract': contract,
            'PaperlessBilling': paperless, 'PaymentMethod': payment,
            'MonthlyCharges': monthly, 'TotalCharges': total.astype(str), 'Churn': churn
        })
    # Clean
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce').fillna(0)
    df['Churn_binary'] = (df['Churn'] == 'Yes').astype(int)
    df['SeniorCitizen_label'] = df['SeniorCitizen'].map({0: 'No', 1: 'Yes'})
    return df

@st.cache_resource
def train_model(df):
    features = ['SeniorCitizen', 'tenure', 'MonthlyCharges', 'TotalCharges',
                'gender', 'Partner', 'Dependents', 'PhoneService',
                'InternetService', 'Contract', 'PaperlessBilling', 'PaymentMethod']
    X = df[features].copy()
    y = df['Churn_binary']
    # Encode categoricals
    le_dict = {}
    for col in X.select_dtypes(include='object').columns:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col].astype(str))
        le_dict[col] = le
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestClassifier(n_estimators=200, max_depth=8, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])
    cm = confusion_matrix(y_test, y_pred)
    importances = pd.Series(model.feature_importances_, index=features).sort_values(ascending=True)
    return model, le_dict, accuracy, auc, cm, importances, features

# ─────────────────────────────────────────────
# PLOTLY THEME
# ─────────────────────────────────────────────
COLORS = {
    'churn': '#f87171',
    'stay': '#4ade80',
    'primary': '#6366f1',
    'secondary': '#8b5cf6',
    'accent': '#06b6d4',
    'bg': '#1e2130',
    'grid': '#2d3250',
    'text': '#c7d2fe',
}
PALETTE = ['#6366f1', '#8b5cf6', '#06b6d4', '#4ade80', '#f59e0b', '#f87171']

def chart_layout(fig, title="", height=380):
    fig.update_layout(
        title=dict(text=title, font=dict(size=14, color=COLORS['text']), x=0.02),
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color=COLORS['text'], family='Inter'),
        height=height,
        margin=dict(l=10, r=10, t=40, b=10),
        legend=dict(bgcolor='rgba(0,0,0,0)', font=dict(size=11)),
        xaxis=dict(gridcolor=COLORS['grid'], linecolor=COLORS['grid'], tickfont=dict(size=11)),
        yaxis=dict(gridcolor=COLORS['grid'], linecolor=COLORS['grid'], tickfont=dict(size=11)),
    )
    return fig

# ─────────────────────────────────────────────
# LOAD DATA
# ─────────────────────────────────────────────
df = load_data()
model, le_dict, accuracy, auc, cm, importances, features = train_model(df)

# ─────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🔮 Churn Analytics")
    st.markdown("<hr style='border-color:#2d3250;margin:0.5rem 0 1rem'>", unsafe_allow_html=True)

    st.markdown("### 🎛️ Filtres")
    selected_contract = st.multiselect("Type de contrat",
        df['Contract'].unique().tolist(),
        default=df['Contract'].unique().tolist())

    selected_internet = st.multiselect("Service Internet",
        df['InternetService'].unique().tolist(),
        default=df['InternetService'].unique().tolist())

    tenure_range = st.slider("Ancienneté (mois)", 0, 72, (0, 72))

    monthly_range = st.slider("Charges mensuelles ($)", 
        int(df['MonthlyCharges'].min()), int(df['MonthlyCharges'].max()),
        (int(df['MonthlyCharges'].min()), int(df['MonthlyCharges'].max())))

    st.markdown("<hr style='border-color:#2d3250;margin:1rem 0'>", unsafe_allow_html=True)
    st.markdown(f"**Modèle ML:** Random Forest")
    st.markdown(f"**Accuracy:** `{accuracy:.1%}`")
    st.markdown(f"**AUC-ROC:** `{auc:.3f}`")
    st.markdown(f"**Dataset:** `{len(df):,}` clients")

# ─────────────────────────────────────────────
# FILTER DATA
# ─────────────────────────────────────────────
filtered = df[
    (df['Contract'].isin(selected_contract)) &
    (df['InternetService'].isin(selected_internet)) &
    (df['tenure'].between(*tenure_range)) &
    (df['MonthlyCharges'].between(*monthly_range))
]

churn_df = filtered[filtered['Churn'] == 'Yes']
stay_df = filtered[filtered['Churn'] == 'No']
churn_rate = filtered['Churn_binary'].mean()
total_rev = filtered['TotalCharges'].sum()
monthly_rev = filtered['MonthlyCharges'].sum()
lost_rev = churn_df['MonthlyCharges'].sum()

# ─────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────
st.markdown("""
<div style='padding:1rem 0 0.5rem'>
    <h1 style='color:white;font-size:1.8rem;margin:0;font-weight:700'>
        📊 Customer Churn Dashboard
    </h1>
    <p style='color:#8b9cc8;margin:0.3rem 0 0;font-size:0.9rem'>
        Analyse prédictive de l'attrition clients — Telco Dataset
    </p>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# KPI CARDS
# ─────────────────────────────────────────────
k1, k2, k3, k4, k5 = st.columns(5)

with k1:
    st.markdown(f"""<div class="kpi-card">
        <div class="kpi-label">Total Clients</div>
        <div class="kpi-value">{len(filtered):,}</div>
        <div class="kpi-delta-pos">↑ Dataset filtré</div>
    </div>""", unsafe_allow_html=True)

with k2:
    color = "kpi-delta-neg" if churn_rate > 0.2 else "kpi-delta-pos"
    st.markdown(f"""<div class="kpi-card">
        <div class="kpi-label">Taux de Churn</div>
        <div class="kpi-value" style="color:#f87171">{churn_rate:.1%}</div>
        <div class="{color}">{'⚠ Critique' if churn_rate > 0.25 else '✓ Acceptable'}</div>
    </div>""", unsafe_allow_html=True)

with k3:
    st.markdown(f"""<div class="kpi-card">
        <div class="kpi-label">Clients Perdus</div>
        <div class="kpi-value" style="color:#f87171">{len(churn_df):,}</div>
        <div class="kpi-delta-neg">↓ {len(churn_df)/len(filtered)*100:.1f}% du total</div>
    </div>""", unsafe_allow_html=True)

with k4:
    st.markdown(f"""<div class="kpi-card">
        <div class="kpi-label">Revenu Mensuel</div>
        <div class="kpi-value">${monthly_rev:,.0f}</div>
        <div class="kpi-delta-pos">$ actif/mois</div>
    </div>""", unsafe_allow_html=True)

with k5:
    st.markdown(f"""<div class="kpi-card">
        <div class="kpi-label">Revenu Perdu/Mois</div>
        <div class="kpi-value" style="color:#f87171">${lost_rev:,.0f}</div>
        <div class="kpi-delta-neg">↑ dû au churn</div>
    </div>""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# TABS
# ─────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs(["📈 Vue Générale", "🔍 Analyse Segments", "🤖 Modèle ML", "🎯 Prédiction Client"])

# ══════════════════════════════════════════════
# TAB 1 — VUE GÉNÉRALE
# ══════════════════════════════════════════════
with tab1:
    row1_l, row1_r = st.columns([1, 2])

    with row1_l:
        # Donut churn
        fig_donut = go.Figure(go.Pie(
            values=[len(churn_df), len(stay_df)],
            labels=['Churned', 'Actif'],
            hole=0.65,
            marker=dict(colors=[COLORS['churn'], COLORS['stay']]),
            textinfo='percent',
            textfont=dict(size=13, color='white'),
        ))
        fig_donut.add_annotation(
            text=f"<b>{churn_rate:.0%}</b><br><span style='font-size:10px'>CHURN</span>",
            x=0.5, y=0.5, showarrow=False,
            font=dict(size=20, color=COLORS['churn'])
        )
        chart_layout(fig_donut, "Répartition Churn", 320)
        st.plotly_chart(fig_donut, use_container_width=True)

    with row1_r:
        # Churn by contract type — grouped bar
        ct = filtered.groupby(['Contract', 'Churn']).size().reset_index(name='count')
        fig_bar = px.bar(ct, x='Contract', y='count', color='Churn',
            barmode='group',
            color_discrete_map={'Yes': COLORS['churn'], 'No': COLORS['stay']},
            labels={'count': 'Clients', 'Contract': 'Type de Contrat'})
        chart_layout(fig_bar, "Churn par Type de Contrat", 320)
        fig_bar.update_traces(marker_line_width=0, opacity=0.9)
        st.plotly_chart(fig_bar, use_container_width=True)

    row2_l, row2_r = st.columns(2)

    with row2_l:
        # Tenure distribution
        fig_hist = go.Figure()
        fig_hist.add_trace(go.Histogram(
            x=churn_df['tenure'], name='Churned',
            marker_color=COLORS['churn'], opacity=0.7, nbinsx=30))
        fig_hist.add_trace(go.Histogram(
            x=stay_df['tenure'], name='Actif',
            marker_color=COLORS['stay'], opacity=0.7, nbinsx=30))
        fig_hist.update_layout(barmode='overlay')
        chart_layout(fig_hist, "Distribution Ancienneté (mois)", 320)
        st.plotly_chart(fig_hist, use_container_width=True)

    with row2_r:
        # Monthly charges boxplot
        fig_box = go.Figure()
        fig_box.add_trace(go.Box(
            y=churn_df['MonthlyCharges'], name='Churned',
            marker_color=COLORS['churn'], fillcolor='rgba(248,113,113,0.2)',
            line_color=COLORS['churn']))
        fig_box.add_trace(go.Box(
            y=stay_df['MonthlyCharges'], name='Actif',
            marker_color=COLORS['stay'], fillcolor='rgba(74,222,128,0.2)',
            line_color=COLORS['stay']))
        chart_layout(fig_box, "Charges Mensuelles ($)", 320)
        st.plotly_chart(fig_box, use_container_width=True)

# ══════════════════════════════════════════════
# TAB 2 — ANALYSE SEGMENTS
# ══════════════════════════════════════════════
with tab2:
    r1, r2 = st.columns(2)

    with r1:
        # Churn rate by internet service
        seg_internet = filtered.groupby('InternetService')['Churn_binary'].agg(['mean','count']).reset_index()
        seg_internet.columns = ['InternetService', 'churn_rate', 'count']
        fig_int = px.bar(seg_internet, x='InternetService', y='churn_rate',
            color='churn_rate',
            color_continuous_scale=['#4ade80', '#f59e0b', '#f87171'],
            text=seg_internet['churn_rate'].apply(lambda x: f'{x:.1%}'),
            labels={'churn_rate': 'Taux de Churn', 'InternetService': 'Service Internet'})
        fig_int.update_traces(textposition='outside', marker_line_width=0)
        fig_int.update_coloraxes(showscale=False)
        chart_layout(fig_int, "Taux de Churn par Service Internet", 320)
        st.plotly_chart(fig_int, use_container_width=True)

    with r2:
        # Churn by payment method
        seg_pay = filtered.groupby('PaymentMethod')['Churn_binary'].mean().reset_index()
        seg_pay.columns = ['PaymentMethod', 'churn_rate']
        seg_pay['PaymentMethod'] = seg_pay['PaymentMethod'].str.replace(' (automatic)', '\n(auto)', regex=False)
        fig_pay = px.bar(seg_pay.sort_values('churn_rate'), x='churn_rate', y='PaymentMethod',
            orientation='h',
            color='churn_rate',
            color_continuous_scale=['#4ade80', '#f59e0b', '#f87171'],
            text=seg_pay.sort_values('churn_rate')['churn_rate'].apply(lambda x: f'{x:.1%}'),
            labels={'churn_rate': 'Taux de Churn', 'PaymentMethod': ''})
        fig_pay.update_traces(textposition='outside', marker_line_width=0)
        fig_pay.update_coloraxes(showscale=False)
        chart_layout(fig_pay, "Taux de Churn par Mode de Paiement", 320)
        st.plotly_chart(fig_pay, use_container_width=True)

    r3, r4 = st.columns(2)

    with r3:
        # Scatter: tenure vs monthly charges
        sample = filtered.sample(min(2000, len(filtered)), random_state=42)
        fig_sc = px.scatter(sample, x='tenure', y='MonthlyCharges',
            color='Churn',
            color_discrete_map={'Yes': COLORS['churn'], 'No': COLORS['stay']},
            opacity=0.5, size_max=5,
            labels={'tenure': 'Ancienneté (mois)', 'MonthlyCharges': 'Charges/Mois ($)'},
            hover_data=['Contract', 'InternetService'])
        chart_layout(fig_sc, "Ancienneté vs Charges Mensuelles", 340)
        st.plotly_chart(fig_sc, use_container_width=True)

    with r4:
        # Churn rate by senior + gender heatmap
        pivot = filtered.groupby(['SeniorCitizen_label', 'gender'])['Churn_binary'].mean().unstack()
        fig_heat = go.Figure(go.Heatmap(
            z=pivot.values,
            x=pivot.columns.tolist(),
            y=pivot.index.tolist(),
            colorscale=[[0,'#4ade80'],[0.5,'#f59e0b'],[1,'#f87171']],
            text=[[f'{v:.1%}' for v in row] for row in pivot.values],
            texttemplate="%{text}",
            textfont=dict(size=16, color='white'),
            showscale=True,
            zmin=0, zmax=0.5
        ))
        chart_layout(fig_heat, "Taux de Churn : Senior × Genre", 320)
        st.plotly_chart(fig_heat, use_container_width=True)

# ══════════════════════════════════════════════
# TAB 3 — MODÈLE ML
# ══════════════════════════════════════════════
with tab3:
    m1, m2, m3 = st.columns(3)

    with m1:
        st.markdown(f"""<div class="kpi-card" style="margin-top:0.5rem">
            <div class="kpi-label">Accuracy</div>
            <div class="kpi-value" style="color:#6366f1">{accuracy:.1%}</div>
            <div class="kpi-delta-pos">✓ Random Forest</div>
        </div>""", unsafe_allow_html=True)

    with m2:
        st.markdown(f"""<div class="kpi-card" style="margin-top:0.5rem">
            <div class="kpi-label">AUC-ROC</div>
            <div class="kpi-value" style="color:#06b6d4">{auc:.3f}</div>
            <div class="kpi-delta-pos">{'✓ Excellent' if auc > 0.85 else '✓ Bon'}</div>
        </div>""", unsafe_allow_html=True)

    with m3:
        tn, fp, fn, tp = cm.ravel()
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        st.markdown(f"""<div class="kpi-card" style="margin-top:0.5rem">
            <div class="kpi-label">Recall (Churn)</div>
            <div class="kpi-value" style="color:#f59e0b">{recall:.1%}</div>
            <div class="kpi-delta-pos">Précision: {precision:.1%}</div>
        </div>""", unsafe_allow_html=True)

    ml1, ml2 = st.columns(2)

    with ml1:
        # Feature importance
        fig_imp = go.Figure(go.Bar(
            x=importances.values,
            y=importances.index,
            orientation='h',
            marker=dict(
                color=importances.values,
                colorscale=[[0, '#2d3250'], [0.5, '#6366f1'], [1, '#06b6d4']],
                line_width=0
            )
        ))
        chart_layout(fig_imp, "Importance des Variables", 400)
        st.plotly_chart(fig_imp, use_container_width=True)

    with ml2:
        # Confusion matrix
        fig_cm = go.Figure(go.Heatmap(
            z=cm,
            x=['Prédit: Actif', 'Prédit: Churn'],
            y=['Réel: Actif', 'Réel: Churn'],
            colorscale=[[0, '#1e2130'], [1, '#6366f1']],
            text=cm,
            texttemplate='<b>%{text}</b>',
            textfont=dict(size=20, color='white'),
            showscale=False
        ))
        fig_cm.add_annotation(x=0, y=0, text="TN", showarrow=False, font=dict(size=11, color='#8b9cc8'), yshift=-20)
        fig_cm.add_annotation(x=1, y=0, text="FP", showarrow=False, font=dict(size=11, color='#8b9cc8'), yshift=-20)
        fig_cm.add_annotation(x=0, y=1, text="FN", showarrow=False, font=dict(size=11, color='#8b9cc8'), yshift=-20)
        fig_cm.add_annotation(x=1, y=1, text="TP", showarrow=False, font=dict(size=11, color='#8b9cc8'), yshift=-20)
        chart_layout(fig_cm, "Matrice de Confusion", 400)
        st.plotly_chart(fig_cm, use_container_width=True)

# ══════════════════════════════════════════════
# TAB 4 — PRÉDICTION CLIENT
# ══════════════════════════════════════════════
with tab4:
    st.markdown('<div class="section-title">🎯 Prédire le risque de churn d\'un client</div>', unsafe_allow_html=True)

    p1, p2, p3 = st.columns(3)

    with p1:
        gender = st.selectbox("Genre", ["Male", "Female"])
        senior = st.selectbox("Senior Citizen", ["Non (0)", "Oui (1)"])
        partner = st.selectbox("Partenaire", ["Yes", "No"])
        dependents = st.selectbox("Personnes à charge", ["Yes", "No"])
        tenure = st.slider("Ancienneté (mois)", 0, 72, 12)

    with p2:
        phone = st.selectbox("Service Téléphonique", ["Yes", "No"])
        internet = st.selectbox("Service Internet", ["DSL", "Fiber optic", "No"])
        contract = st.selectbox("Type de Contrat", ["Month-to-month", "One year", "Two year"])
        paperless = st.selectbox("Facturation Dématérialisée", ["Yes", "No"])

    with p3:
        payment = st.selectbox("Mode de Paiement", [
            "Electronic check", "Mailed check",
            "Bank transfer (automatic)", "Credit card (automatic)"])
        monthly = st.slider("Charges Mensuelles ($)", 18, 120, 65)
        total = monthly * tenure
        st.markdown(f"**Charges Totales estimées:** ${total:,.0f}")

    if st.button("🔮 Prédire le risque de Churn", use_container_width=True):
        senior_val = 1 if "Oui" in senior else 0
        input_data = {
            'SeniorCitizen': senior_val, 'tenure': tenure,
            'MonthlyCharges': monthly, 'TotalCharges': float(total),
            'gender': gender, 'Partner': partner, 'Dependents': dependents,
            'PhoneService': phone, 'InternetService': internet,
            'Contract': contract, 'PaperlessBilling': paperless,
            'PaymentMethod': payment
        }
        input_df = pd.DataFrame([input_data])
        for col in input_df.select_dtypes(include='object').columns:
            if col in le_dict:
                try:
                    input_df[col] = le_dict[col].transform(input_df[col].astype(str))
                except:
                    input_df[col] = 0

        proba = model.predict_proba(input_df[features])[0][1]
        prediction = "CHURN" if proba > 0.5 else "ACTIF"

        res1, res2 = st.columns([1, 2])
        with res1:
            if proba > 0.5:
                st.markdown(f"""<div class="pred-high">
                    <div class="pred-text">⚠️ Risque de CHURN</div>
                    <div class="pred-sub">Probabilité: {proba:.1%}</div>
                </div>""", unsafe_allow_html=True)
            else:
                st.markdown(f"""<div class="pred-low">
                    <div class="pred-text">✅ Client ACTIF</div>
                    <div class="pred-sub">Probabilité de rester: {1-proba:.1%}</div>
                </div>""", unsafe_allow_html=True)

        with res2:
            # Gauge chart
            fig_gauge = go.Figure(go.Indicator(
                mode="gauge+number+delta",
                value=proba * 100,
                domain={'x': [0, 1], 'y': [0, 1]},
                title={'text': "Score de Risque Churn (%)", 'font': {'size': 14, 'color': COLORS['text']}},
                delta={'reference': 26.5, 'increasing': {'color': COLORS['churn']}, 'decreasing': {'color': COLORS['stay']}},
                gauge={
                    'axis': {'range': [0, 100], 'tickcolor': COLORS['text'], 'tickfont': {'color': COLORS['text']}},
                    'bar': {'color': COLORS['churn'] if proba > 0.5 else COLORS['stay']},
                    'bgcolor': COLORS['bg'],
                    'steps': [
                        {'range': [0, 30], 'color': 'rgba(74,222,128,0.15)'},
                        {'range': [30, 60], 'color': 'rgba(245,158,11,0.15)'},
                        {'range': [60, 100], 'color': 'rgba(248,113,113,0.15)'}
                    ],
                    'threshold': {'line': {'color': 'white', 'width': 2}, 'thickness': 0.75, 'value': 50}
                },
                number={'suffix': '%', 'font': {'color': COLORS['churn'] if proba > 0.5 else COLORS['stay'], 'size': 36}}
            ))
            fig_gauge.update_layout(
                paper_bgcolor='rgba(0,0,0,0)',
                font=dict(color=COLORS['text']),
                height=250, margin=dict(l=20, r=20, t=30, b=10)
            )
            st.plotly_chart(fig_gauge, use_container_width=True)

        # Recommendations
        st.markdown('<div class="section-title">💡 Recommandations de Rétention</div>', unsafe_allow_html=True)
        recs = []
        if contract == "Month-to-month": recs.append("📋 Proposer une offre d'engagement 1 ou 2 ans avec remise")
        if internet == "Fiber optic" and proba > 0.4: recs.append("📡 Vérifier la satisfaction qualité réseau Fiber")
        if payment == "Electronic check": recs.append("💳 Encourager le prélèvement automatique (fidélité +)")
        if tenure < 12: recs.append("🎁 Offrir un programme de fidélité nouveau client (0-12 mois)")
        if monthly > 80: recs.append("💰 Proposer un plan tarifaire adapté / bundle économique")
        if not recs: recs.append("✅ Client peu risqué — maintenir la qualité de service")
        for r in recs:
            st.markdown(f"- {r}")

# ─────────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────────
st.markdown("<br>", unsafe_allow_html=True)
st.markdown("""
<div style='text-align:center;color:#4a5568;font-size:0.75rem;padding:1rem 0;border-top:1px solid #2d3250'>
    Churn Analytics Dashboard • Dataset Telco IBM • Random Forest Classifier
</div>
""", unsafe_allow_html=True)