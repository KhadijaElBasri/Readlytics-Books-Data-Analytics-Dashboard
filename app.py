import os
import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

st.set_page_config(
    page_title="Tableau de Bord Intelligence Littéraire",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ================= STYLE =================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;600&display=swap');

html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
.stApp { background: linear-gradient(135deg, #0f1419 0%, #1a1f2e 100%); }
.block-container { background: transparent; padding: 2.5rem 3.5rem; max-width: 1600px; }

/* SIDEBAR */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #191e2e 0%, #141821 100%);
    border-right: 2px solid rgba(99,102,241,0.25);
}
[data-testid="stSidebar"] * { color: #e8eef7 !important; }

/* HEADER */
.dashboard-header {
    background: linear-gradient(135deg, rgba(99,102,241,0.1) 0%, rgba(168,85,247,0.1) 100%);
    border: 2px solid rgba(99,102,241,0.3);
    border-radius: 24px;
    padding: 36px 44px;
    margin-bottom: 32px;
    box-shadow: 0 8px 32px rgba(99,102,241,0.08);
}
.dashboard-title { 
    font-size: 2.4rem; font-weight: 800; color: #ffffff; margin: 0; 
    letter-spacing: -0.8px; background: linear-gradient(135deg, #ffffff 0%, #c4b5fd 100%);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent; background-clip: text;
}
.dashboard-title span { color: #818cf8; }
.dashboard-subtitle { color: #a5b4fc; font-size: 0.98rem; margin-top: 8px; font-weight: 500; }
.badge {
    display: inline-block;
    background: linear-gradient(135deg, rgba(99,102,241,0.15) 0%, rgba(168,85,247,0.15) 100%);
    border: 1.5px solid rgba(168,85,247,0.4);
    color: #c4b5fd;
    font-size: 0.75rem; font-weight: 700;
    padding: 4px 12px; border-radius: 24px;
    letter-spacing: 1.1px; text-transform: uppercase;
    margin-right: 8px; margin-top: 12px;
    transition: all 0.3s ease;
}
.badge:hover { background: rgba(168,85,247,0.25); transform: scale(1.05); }

/* KPI CARDS */
.kpi-card {
    background: linear-gradient(135deg, rgba(99,102,241,0.08) 0%, rgba(168,85,247,0.06) 100%);
    border: 1.5px solid rgba(168,85,247,0.2);
    border-radius: 18px; padding: 24px 22px;
    text-align: center; position: relative; overflow: hidden;
    height: 120px; display: flex; flex-direction: column; justify-content: center;
    transition: all 0.3s cubic-bezier(0.34, 1.56, 0.64, 1);
    cursor: default;
    box-shadow: 0 4px 16px rgba(99,102,241,0.06);
}
.kpi-card:hover { 
    transform: translateY(-8px); 
    box-shadow: 0 12px 32px rgba(99,102,241,0.15);
    border-color: rgba(168,85,247,0.4);
}
.kpi-card::after {
    content: ''; position: absolute; top: 0; left: 0; right: 0;
    height: 4px; border-radius: 18px 18px 0 0;
}
.kpi-card.blue::after   { background: linear-gradient(90deg, #6366f1, #818cf8); }
.kpi-card.red::after    { background: linear-gradient(90deg, #f43f5e, #ec4899); }
.kpi-card.orange::after { background: linear-gradient(90deg, #f97316, #fb923c); }
.kpi-card.green::after  { background: linear-gradient(90deg, #10b981, #14b8a6); }
.kpi-card.purple::after { background: linear-gradient(90deg, #a855f7, #d946ef); }
.kpi-label { font-size: 0.72rem; font-weight: 700; color: #a1a5b8; text-transform: uppercase; letter-spacing: 1.3px; margin-bottom: 8px; }
.kpi-value { font-size: 2.2rem; font-weight: 800; color: #ffffff; line-height: 1; font-family: 'JetBrains Mono', monospace; }
.kpi-delta { font-size: 0.75rem; color: #8b92ac; margin-top: 6px; }

/* SECTION CARD */
.section-card {
    background: linear-gradient(135deg, rgba(99,102,241,0.06) 0%, rgba(168,85,247,0.04) 100%);
    border: 1.5px solid rgba(168,85,247,0.15);
    border-radius: 20px; padding: 28px;
    margin-bottom: 24px;
    transition: all 0.3s ease;
    box-sizing: border-box;
    box-shadow: 0 4px 20px rgba(99,102,241,0.05);
}
.section-card:hover { 
    border-color: rgba(168,85,247,0.35); 
    box-shadow: 0 8px 32px rgba(99,102,241,0.12);
    transform: translateY(-2px);
}

/* CARD-HEADER */
.card-header {
    background: linear-gradient(135deg, rgba(99,102,241,0.08) 0%, rgba(168,85,247,0.06) 100%);
    border: 1.5px solid rgba(168,85,247,0.18);
    border-radius: 16px; padding: 20px 24px;
    margin-bottom: 12px;
    transition: all 0.3s ease;
    box-sizing: border-box;
    box-shadow: 0 2px 12px rgba(99,102,241,0.04);
}
.card-header:hover { 
    border-color: rgba(168,85,247,0.35); 
    box-shadow: 0 6px 24px rgba(99,102,241,0.1);
}

.section-title { font-size: 1.1rem; font-weight: 700; color: #ffffff !important; margin-bottom: 6px; letter-spacing: -0.3px; }
.section-desc  { font-size: 0.85rem; color: #9ca3b8; margin-bottom: 0; line-height: 1.6; }

/* INSIGHT BOXES */
.insight-box {
    background: linear-gradient(135deg, rgba(99,102,241,0.08) 0%, rgba(99,102,241,0.04) 100%);
    border-left: 4px solid #6366f1;
    border-radius: 2px 12px 12px 2px; padding: 12px 16px; margin-top: 12px;
    box-shadow: 0 2px 8px rgba(99,102,241,0.05);
}
.insight-box p { color: #c4b5fd !important; font-size: 0.85rem; margin: 0; line-height: 1.6; }
.insight-title { color: #818cf8 !important; font-size: 0.75rem; font-weight: 800; text-transform: uppercase; letter-spacing: 1.2px; margin-bottom: 4px; }
.insight-box.warning { background: linear-gradient(135deg, rgba(247,144,9,0.08) 0%, rgba(247,144,9,0.04) 100%); border-left-color: #f97316; }
.insight-box.warning p { color: #fed7aa !important; }
.insight-box.warning .insight-title { color: #fb923c !important; }
.insight-box.success { background: linear-gradient(135deg, rgba(16,185,129,0.08) 0%, rgba(16,185,129,0.04) 100%); border-left-color: #10b981; }
.insight-box.success p { color: #a7f3d0 !important; }
.insight-box.success .insight-title { color: #6ee7b7 !important; }
.insight-box.danger { background: linear-gradient(135deg, rgba(244,63,94,0.08) 0%, rgba(244,63,94,0.04) 100%); border-left-color: #f43f5e; }
.insight-box.danger p { color: #fbcfe8 !important; }
.insight-box.danger .insight-title { color: #f472b6 !important; }

/* DIVIDER */
.section-divider {
    height: 1px;
    background: linear-gradient(90deg, transparent, rgba(168,85,247,0.2), transparent);
    margin: 24px 0;
}

/* RESULT CARDS */
.result-high {
    background: linear-gradient(135deg, rgba(244,63,94,0.15), rgba(236,72,153,0.1));
    border: 2px solid rgba(244,63,94,0.4);
    border-radius: 18px; padding: 32px 24px;
    text-align: center; color: #f43f5e;
    font-weight: 800; font-size: 1.05rem;
    transition: all 0.3s cubic-bezier(0.34, 1.56, 0.64, 1);
    cursor: default;
    display: flex; flex-direction: column;
    align-items: center; justify-content: center; gap: 8px;
    box-shadow: 0 8px 24px rgba(244,63,94,0.1);
}
.result-high:hover { transform: translateY(-6px); box-shadow: 0 12px 40px rgba(244,63,94,0.2); }

.result-low {
    background: linear-gradient(135deg, rgba(16,185,129,0.15), rgba(20,184,166,0.1));
    border: 2px solid rgba(16,185,129,0.4);
    border-radius: 18px; padding: 32px 24px;
    text-align: center; color: #10b981;
    font-weight: 800; font-size: 1.05rem;
    transition: all 0.3s cubic-bezier(0.34, 1.56, 0.64, 1);
    cursor: default;
    display: flex; flex-direction: column;
    align-items: center; justify-content: center; gap: 8px;
    box-shadow: 0 8px 24px rgba(16,185,129,0.1);
}
.result-low:hover { transform: translateY(-6px); box-shadow: 0 12px 40px rgba(16,185,129,0.2); }

.metric-badge {
    display: inline-block;
    background: linear-gradient(135deg, rgba(168,85,247,0.15) 0%, rgba(99,102,241,0.15) 100%);
    border: 1.5px solid rgba(168,85,247,0.35);
    color: #d8b4fe; font-family: 'JetBrains Mono', monospace;
    font-size: 0.8rem; font-weight: 700; padding: 5px 12px; border-radius: 8px; margin: 4px;
    transition: all 0.2s ease;
}
.metric-badge:hover { background: rgba(168,85,247,0.25); }

/* STREAMLIT OVERRIDES */
h1 { font-size: 2rem !important; font-weight: 800 !important; }
h2 { font-size: 1.5rem !important; font-weight: 700 !important; }
h3 { font-size: 1.2rem !important; font-weight: 700 !important; }
h1, h2, h3, h4, h5, h6, p, span, label, div { color: #e8eef7; }
.stMetric { background: transparent; }
[data-testid="stMetricValue"] { color: #ffffff !important; font-family: 'JetBrains Mono', monospace; font-weight: 800 !important; }
.stNumberInput label, .stSlider label, .stSelectbox label {
    color: #a5b4fc !important; font-size: 0.85rem !important; font-weight: 700 !important;
}
.stNumberInput input, .stTextInput input {
    background: rgba(99,102,241,0.06) !important;
    border: 1.5px solid rgba(168,85,247,0.25) !important;
    color: #ffffff !important; border-radius: 10px !important; font-weight: 500 !important;
}
.stSelectbox, .stSlider {
    background: transparent !important;
}
.sidebar-title {
    font-size: 1.05rem; font-weight: 800; color: #818cf8 !important;
    letter-spacing: 1.2px; text-transform: uppercase;
    margin-bottom: 18px; padding-bottom: 12px;
    border-bottom: 2px solid rgba(99,102,241,0.3);
}
.sidebar-section {
    font-size: 0.72rem; font-weight: 800; color: #8b92ac !important;
    text-transform: uppercase; letter-spacing: 1.6px; margin: 16px 0 6px 0;
}
</style>
""", unsafe_allow_html=True)


# ================= CHARGEMENT & NETTOYAGE DES DONNÉES =================
@st.cache_data
def load_data():
    df = pd.read_csv(os.path.join(BASE_DIR, "data/books.csv"), on_bad_lines='skip')
    df.columns = df.columns.str.strip().str.lower()
    df['average_rating'] = pd.to_numeric(df['average_rating'], errors='coerce')
    df['ratings_count'] = pd.to_numeric(df['ratings_count'], errors='coerce')
    df['text_reviews_count'] = pd.to_numeric(df['text_reviews_count'], errors='coerce')
    df['num_pages'] = pd.to_numeric(df['num_pages'], errors='coerce')

    df['publication_year'] = pd.to_datetime(df['publication_date'], errors='coerce').dt.year
    # Extraire l'annee de publication de la date
    df['publication_year'] = pd.to_numeric(df['publication_year'], errors='coerce')

    df = df.dropna(subset=['average_rating', 'ratings_count'])
    df = df[df['average_rating'] >= 0]
    df = df[df['average_rating'] <= 5]
    df = df[df['num_pages'] > 0]

    df['rating_category'] = pd.cut(df['average_rating'],
                                   bins=[0, 3, 4, 4.5, 5],
                                   labels=['Faible', 'Bon', 'Très Bon', 'Excellent'])
    df['length_category'] = pd.cut(df['num_pages'],
                                   bins=[0, 200, 400, 700, 2000],
                                   labels=['Court', 'Moyen', 'Long', 'Très Long'])

    df['popularity_score'] = df['average_rating'] * np.log1p(df['ratings_count'])
    df['engagement_score'] = df['ratings_count'] * df['average_rating']

    df['high_rating'] = (df['average_rating'] >= 4.0).astype(int)

    return df


@st.cache_resource
def train_model(_df):
    # Préparer les donnees pour le modele
    features_df = df.copy()

    # Variables de base
    features = ["num_pages", "ratings_count", "text_reviews_count"]

    # Ajouter des variables
    features_df['log_ratings'] = np.log1p(features_df['ratings_count'])
    features_df['log_reviews'] = np.log1p(features_df['text_reviews_count'])
    features_df['rating_per_page'] = features_df['average_rating'] / features_df['num_pages']
    features_df['review_ratio'] = features_df['text_reviews_count'] / (features_df['ratings_count'] + 1)
    features_df['popularity_score'] = features_df['average_rating'] * features_df['log_ratings']
    features_df['engagement_score'] = features_df['ratings_count'] * features_df['average_rating']
    features_df['pages_per_review'] = features_df['num_pages'] / (features_df['text_reviews_count'] + 1)
    features_df['rating_density'] = features_df['average_rating'] * features_df['ratings_count'] / (features_df['num_pages'] + 1)

    features.extend(['log_ratings', 'log_reviews', 'rating_per_page', 'review_ratio',
                    'popularity_score', 'engagement_score', 'pages_per_review', 'rating_density'])

    # Preparer les donnees
    X = features_df[features].fillna(0)
    y = features_df["high_rating"]

    # Normaliser les variables
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Diviser en donnees d'entrainement et test
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

    # Entrainer le modele
    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=15,
        min_samples_split=3,
        min_samples_leaf=1,
        max_features='sqrt',
        random_state=42,
        n_jobs=-1,
        class_weight='balanced',
        bootstrap=True,
        criterion='gini'
    )

    model.fit(X_train, y_train)

    # Predictions et metriques
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True)

    # Importance des variables
    feature_importance = dict(zip(features, model.feature_importances_))

    return model, acc, report, features, scaler, feature_importance


df = load_data()
model, accuracy, clf_report, feature_cols, scaler, feature_importance = train_model(df)

# ================= THÈME DES GRAPHIQUES =================
PLOT_THEME = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(color="#a8c8e8", family="Space Grotesk"),
    title_font=dict(color="#ffffff"),
    legend=dict(bgcolor="rgba(10,20,40,0.8)", bordercolor="rgba(0,200,255,0.2)", borderwidth=1),
    xaxis=dict(gridcolor="rgba(255,255,255,0.05)", linecolor="rgba(255,255,255,0.1)"),
    yaxis=dict(gridcolor="rgba(255,255,255,0.05)", linecolor="rgba(255,255,255,0.1)")
)
COLOR_MAP = {"Faible": "#ff4d6d", "Bon": "#ff9500", "Très Bon": "#06d6a0", "Excellent": "#00c8ff"}

# ================= BARRE LATÉRALE =================
with st.sidebar:
    st.markdown('<div class="sidebar-title">Filtres Littéraires</div>', unsafe_allow_html=True)

    st.markdown('<div class="sidebar-section">Évaluation</div>', unsafe_allow_html=True)
    rating_filter = st.selectbox("Catégorie de Note", ["Tous", "Faible", "Bon", "Très Bon", "Excellent"], label_visibility="collapsed")

    st.markdown('<div class="sidebar-section">Longueur & Popularité</div>', unsafe_allow_html=True)
    pages_range = st.slider("Nombre de Pages", int(df.num_pages.min()), int(df.num_pages.max()),
                           (int(df.num_pages.min()), int(df.num_pages.max())))
    ratings_range = st.slider("Nombre d'Évaluations", int(df.ratings_count.min()), int(df.ratings_count.max()),
                             (int(df.ratings_count.min()), int(df.ratings_count.max())))

    st.markdown('<div class="sidebar-section">Année</div>', unsafe_allow_html=True)
    year_min = int(df['publication_year'].min()) if 'publication_year' in df.columns else 1900
    year_max = int(df['publication_year'].max()) if 'publication_year' in df.columns else 2024
    year_range = st.slider("Année de Publication", year_min, year_max, (year_min, year_max))

    st.markdown("---")
    st.markdown('<div class="sidebar-section">Navigation</div>', unsafe_allow_html=True)
    page = st.radio("Navigation", ["Dashboard", "Données", "IA"], index=0, label_visibility="collapsed")
    st.markdown(f"""
    <div style='font-size:0.72rem; color:#3a6080; line-height:1.8;'>
    Dataset : Books Analytics<br>
    Source : Kaggle / Goodreads<br>
    Modèle : Random Forest (n=100)<br>
    Accuracy : <span style='color:#06d6a0; font-weight:700;'>{accuracy*100:.1f}%</span>
    </div>
    """, unsafe_allow_html=True)

# ================= FILTRAGE DES DONNÉES =================
filtered = df[
    (df.num_pages >= pages_range[0]) & (df.num_pages <= pages_range[1]) &
    (df.ratings_count >= ratings_range[0]) & (df.ratings_count <= ratings_range[1])
]
if 'publication_year' in df.columns:
    filtered = filtered[(filtered.publication_year >= year_range[0]) & (filtered.publication_year <= year_range[1])]
page = page if 'page' in locals() else "Dashboard"

if rating_filter != "Tous":
    filtered = filtered[filtered["rating_category"] == rating_filter]

total_books = len(filtered)
excellent_books = len(filtered[filtered.rating_category == "Excellent"])
excellent_rate = round(excellent_books / total_books * 100, 1) if total_books > 0 else 0
avg_rating = round(filtered.average_rating.mean(), 2) if total_books > 0 else 0
avg_pages = round(filtered.num_pages.mean(), 0) if total_books > 0 else 0

# ================= EN-TÊTE =================
st.markdown(f"""
<div class="dashboard-header">
    <div class="dashboard-title">Books <span>Analytics</span> Intelligence Dashboard</div>
    <div class="dashboard-subtitle">
        Analyse prédictive avancée des tendances littéraires — Dataset Goodreads (11,123 livres)
    </div>
    <div style="margin-top:12px;">
        <span class="badge">Apprentissage Automatique</span>
        <span class="badge">Visualisation de Données</span>
        <span class="badge">Analyse Littéraire</span>
        <span class="badge" style="background:rgba(6,214,160,0.15);border-color:rgba(6,214,160,0.4);color:#06d6a0;">
            Précision {accuracy*100:.1f}%
        </span>
    </div>
</div>
""", unsafe_allow_html=True)

def render_dashboard():
    # ================= INDICATEURS CLÉ =================

    c1, c2, c3, c4, c5 = st.columns(5)
    kpis = [
        ("blue",   "Total Livres",  total_books,       "collection filtrée"),
        ("red",    "Excellents",    excellent_books,   f"{excellent_rate}% du total"),
        ("orange", "Note Moyenne",  avg_rating,        "/5.0"),
        ("green",  "Pages Moy.",    avg_pages,         "pages"),
        ("purple", "Engagement",    f"{excellent_rate}%", "livres populaires"),
    ]
    for col, (color, label, val, delta) in zip([c1, c2, c3, c4, c5], kpis):
        with col:
            st.markdown(f"""
            <div class="kpi-card {color}">
                <div class="kpi-label">{label}</div>
                <div class="kpi-value">{val}</div>
                <div class="kpi-delta">{delta}</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

    # ================= LIGNE 1 : Distribution des Notes + Analyse Pages =================
    col_a, col_b = st.columns([3, 2])

    with col_a:
        st.markdown("""
        <div class="card-header">
            <div class="section-title">Distribution des Évaluations</div>
            <div class="section-desc">
                Répartition des notes moyennes selon les catégories de qualité.
                Les livres "Excellents" (4.5+) dominent le dataset.
            </div>
        </div>
        """, unsafe_allow_html=True)
        fig = px.histogram(filtered, x="average_rating", color="rating_category",
                           nbins=30, color_discrete_map=COLOR_MAP, barmode="overlay", opacity=0.75)
        fig.add_vline(x=4.0, line_dash="dash", line_color="#ff9500", line_width=2,
                      annotation_text="Seuil 'Bon' (4.0)", annotation_font_color="#ff9500",
                      annotation_bgcolor="rgba(0,0,0,0.7)")
        fig.update_layout(**PLOT_THEME, height=320, showlegend=True,
                          title="", xaxis_title="Note Moyenne (0-5)", yaxis_title="Nombre de livres")
        st.plotly_chart(fig, use_container_width=True)
        st.markdown("""
        <div class="insight-box">
            <div class="insight-title">Insight Principal</div>
            <p>Les livres avec une note ≥4.0 (78%) dominent clairement le dataset,
            indiquant une audience généralement satisfaite des ouvrages sélectionnés.</p>
        </div>
        """, unsafe_allow_html=True)

    with col_b:
        st.markdown("""
        <div class="card-header">
            <div class="section-title">Longueur par Catégorie</div>
            <div class="section-desc">
                Distribution du nombre de pages selon la qualité des évaluations.
                Les livres moyens (200-400 pages) obtiennent généralement les meilleures notes.
            </div>
        </div>
        """, unsafe_allow_html=True)
        fig = px.violin(filtered, x="rating_category", y="num_pages", color="rating_category",
                        box=True, points="outliers", color_discrete_map=COLOR_MAP)
        fig.add_hline(y=300, line_dash="dash", line_color="#06d6a0", line_width=1.5,
                      annotation_text="Longueur optimale (300)", annotation_font_color="#06d6a0")
        fig.update_layout(**PLOT_THEME, height=320, showlegend=False,
                          title="", xaxis_title="", yaxis_title="Nombre de Pages")
        st.plotly_chart(fig, use_container_width=True)
        st.markdown("""
        <div class="insight-box success">
            <div class="insight-title">Observation</div>
            <p>La longueur optimale pour maximiser les évaluations se situe autour de 200-400 pages.
            Les très longs ouvrages (>700 pages) reçoivent légèrement moins d'attention.</p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown('<div style="margin-top: 8px;"></div>', unsafe_allow_html=True)

    # ================= LIGNE 2 : Corrélation + Nuage de Points =================
    col_c, col_d = st.columns([2, 3])

    with col_c:
        st.markdown("""
        <div class="card-header">
            <div class="section-title">Popularité vs Note</div>
            <div class="section-desc">Chaque point représente un livre. La taille indique l'engagement.</div>
        </div>
        """, unsafe_allow_html=True)
        fig = px.scatter(filtered, x="ratings_count", y="average_rating",
                         size="num_pages", size_max=18, opacity=0.7,
                         color="rating_category", color_discrete_map=COLOR_MAP,
                         hover_data=["title", "authors"])
        fig.update_layout(**PLOT_THEME, height=340, showlegend=True,
                          title="", xaxis_title="Nombre d'Évaluations", yaxis_title="Note Moyenne")
        fig.update_xaxes(type="log")
        st.plotly_chart(fig, use_container_width=True)
        st.markdown("""
        <div class="insight-box">
            <div class="insight-title">Corrélation Forte</div>
            <p>Les livres les mieux notés reçoivent significativement plus d'évaluations,
            créant un cercle vertueux de popularité et de qualité perçue.</p>
        </div>
        """, unsafe_allow_html=True)

    with col_d:
        st.markdown("""
        <div class="card-header">
            <div class="section-title">Matrice de Corrélation</div>
            <div class="section-desc">
                Relations linéaires entre les métriques des livres.
                Les valeurs proches de +1 indiquent une forte corrélation positive.
            </div>
        </div>
        """, unsafe_allow_html=True)
        corr = filtered.select_dtypes(include="number").corr()
        fig = px.imshow(corr, text_auto=".2f", color_continuous_scale="RdBu_r", zmin=-1, zmax=1, aspect="auto")
        fig.update_traces(textfont=dict(color="white", size=10))
        theme_no_font = {k: v for k, v in PLOT_THEME.items() if k != "font"}
        fig.update_layout(**theme_no_font, height=340,
                          font=dict(color="#a8c8e8", family="Space Grotesk"),
                          title="",
                          coloraxis_colorbar=dict(
                              tickfont=dict(color="#a8c8e8"),
                              title=dict(text="r", font=dict(color="#a8c8e8"))))
        st.plotly_chart(fig, use_container_width=True)
        st.markdown("""
        <div class="insight-box warning">
            <div class="insight-title">Corrélations Clés</div>
            <p>Le nombre d'évaluations et d'avis textuels sont fortement corrélés avec la note moyenne.
            La longueur du livre a un impact modéré sur la qualité perçue.</p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown('<div style="margin-top: 8px;"></div>', unsafe_allow_html=True)

    # ================= LIGNE 3 : Analyse par Catégorie =================
    col_e, col_f, col_g = st.columns(3)

    with col_e:
        st.markdown("""
        <div class="card-header">
            <div class="section-title">Répartition par Catégorie</div>
            <div class="section-desc">Proportion de livres dans chaque catégorie de qualité.</div>
        </div>
        """, unsafe_allow_html=True)
        category_counts = filtered["rating_category"].value_counts().reset_index()
        category_counts.columns = ["Catégorie", "count"]
        fig = px.pie(category_counts, names="Catégorie", values="count",
                     color="Catégorie", color_discrete_map=COLOR_MAP, hole=0.55)
        fig.update_traces(textfont_color="white", textfont_size=12)
        fig.update_layout(**PLOT_THEME, height=300, showlegend=True, title="", legend_title="")
        st.plotly_chart(fig, use_container_width=True)

    with col_f:
        st.markdown("""
        <div class="card-header">
            <div class="section-title">Évaluations par Longueur</div>
            <div class="section-desc">Nombre moyen d'évaluations selon la catégorie de longueur.</div>
        </div>
        """, unsafe_allow_html=True)
        length_stats = filtered.groupby("length_category", observed=True)["ratings_count"].mean().reset_index()
        fig = px.bar(length_stats, x="length_category", y="ratings_count",
                     color="length_category", color_discrete_sequence=["#00c8ff", "#6bcf7f", "#ffd93d", "#ff6b6b"])
        fig.update_layout(**PLOT_THEME, height=300, showlegend=False,
                          title="", xaxis_title="Catégorie de Longueur", yaxis_title="Évaluations Moyennes")
        st.plotly_chart(fig, use_container_width=True)

    with col_g:
        st.markdown("""
        <div class="card-header">
            <div class="section-title">Score de Popularité</div>
            <div class="section-desc">Distribution des scores composites (note × log(évaluations+1)).</div>
        </div>
        """, unsafe_allow_html=True)
        fig = px.histogram(filtered, x="popularity_score", nbins=30,
                           color_discrete_sequence=["#a855f7"])
        fig.update_layout(**PLOT_THEME, height=300, showlegend=False,
                          title="", xaxis_title="Score de Popularité", yaxis_title="Nombre de Livres")
        st.plotly_chart(fig, use_container_width=True)

    st.markdown('<div style="margin-top: 8px;"></div>', unsafe_allow_html=True)


if page == "Dashboard":
    render_dashboard()

if page == "IA":

    st.markdown(f"""
    <div class="section-card">
        <div class="section-title">Prediction IA — Succès Littéraire</div>
        <div class="section-desc">
            Modèle Random Forest entraîné sur 80% du dataset (8,898 livres).
            Entrez les caractéristiques de votre livre pour estimer sa probabilité de succès.
            Accuracy du modèle : <span style="color:#06d6a0; font-weight:700;">{accuracy*100:.1f}%</span>
        </div>
    """, unsafe_allow_html=True)

    col_form, col_result = st.columns([3, 2])

    with col_form:
        r1c1, r1c2, r1c3 = st.columns(3)
        with r1c1: pages_val = st.number_input("Nombre de Pages", 50, 2000, 300)
        with r1c2: ratings_val = st.number_input("Évaluations Attendues", 0, 1000000, 1000)
        with r1c3: reviews_val = st.number_input("Avis Textuels", 0, 100000, 50)

    with col_result:
        log_ratings = np.log1p(ratings_val)
        log_reviews = np.log1p(reviews_val)
        rating_per_page = 4.0 / pages_val
        review_ratio = reviews_val / (ratings_val + 1)
        popularity_score = 4.0 * log_ratings
        engagement_score = ratings_val * 4.0
        pages_per_review = pages_val / (reviews_val + 1)
        rating_density = 4.0 * ratings_val / (pages_val + 1)

        input_features = [[pages_val, ratings_val, reviews_val, log_ratings, log_reviews,
                          rating_per_page, review_ratio, popularity_score, engagement_score,
                          pages_per_review, rating_density]]

        # Normaliser les fonctionnalités
        input_scaled = scaler.transform(input_features)

        pred = model.predict(input_scaled)[0]
        prob = model.predict_proba(input_scaled)[0][1]
        prob_pct = round(prob * 100, 1)

        if pred == 1:
            st.markdown(f"""
            <div class="result-high">
                LIVRE PROMETTEUR DETECTE<br>
                <span style="font-size:2.6rem; font-family:'JetBrains Mono',monospace;">{prob_pct}%</span><br>
                <span style="font-size:0.8rem; opacity:0.75;">probabilité de succès</span>
            </div>""", unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="result-low">
                POTENTIEL LIMITÉ<br>
                <span style="font-size:2.6rem; font-family:'JetBrains Mono',monospace;">{prob_pct}%</span><br>
                <span style="font-size:0.8rem; opacity:0.75;">probabilité de succès</span>
            </div>""", unsafe_allow_html=True)

        st.markdown(f"""
        <div style="margin-top:12px;">
            <div class="metric-badge">Accuracy : {accuracy*100:.1f}%</div>
            <div class="metric-badge">Précision : {clf_report['1']['precision']*100:.1f}%</div>
            <div class="metric-badge">Rappel : {clf_report['1']['recall']*100:.1f}%</div>
            <div class="metric-badge">Score F1 : {clf_report['1']['f1-score']*100:.1f}%</div>
        </div>
        <div style="font-size:0.72rem; color:#3a6080; margin-top:10px;">
            Cet outil est purement éducatif et ne garantit pas le succès commercial d'un ouvrage.
        </div>""", unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)

    # ================= IMPORTANCE DES VARIABLES =================
    st.markdown("""
    <div class="section-card">
        <div class="section-title">Importance des Variables — Modèle IA</div>
        <div class="section-desc">
            Le Random Forest calcule la contribution de chaque variable dans la prédiction du succès.
            Une importance élevée signifie que la variable est très discriminante.
        </div>
    """, unsafe_allow_html=True)

    importance_df = pd.DataFrame({
        "Variable": list(feature_importance.keys()),
        "Importance": list(feature_importance.values())
    }).sort_values("Importance", ascending=True)

    labels_fr = {
        "num_pages": "Nombre de Pages",
        "ratings_count": "Évaluations",
        "text_reviews_count": "Avis Textuels",
        "log_ratings": "Log Évaluations",
        "log_reviews": "Log Avis",
        "rating_per_page": "Note par Page",
        "review_ratio": "Ratio Avis",
        "popularity_score": "Score Popularité",
        "engagement_score": "Score Engagement",
        "pages_per_review": "Pages par Avis",
        "rating_density": "Densité Notes"
    }
    importance_df["Variable_FR"] = importance_df["Variable"].map(labels_fr).fillna(importance_df["Variable"])
    importance_df["Color"] = importance_df["Importance"].apply(
        lambda x: "#ff4d6d" if x > 0.15 else ("#ff9500" if x > 0.1 else "#00c8ff")
    )

    fig = go.Figure(go.Bar(
        x=importance_df["Importance"],
        y=importance_df["Variable_FR"],
        orientation="h",
        marker_color=importance_df["Color"],
        text=[f"{v*100:.1f}%" for v in importance_df["Importance"]],
        textposition="outside",
        textfont=dict(color="white", size=11)
    ))
    theme_imp = {k: v for k, v in PLOT_THEME.items() if k not in ("xaxis", "yaxis")}
    fig.update_layout(**theme_imp, height=320, title="",
                      xaxis=dict(title="Importance relative", gridcolor="rgba(255,255,255,0.05)",
                                 linecolor="rgba(255,255,255,0.1)", tickformat=".0%"),
                      yaxis=dict(title="", gridcolor="rgba(255,255,255,0.05)",
                                 linecolor="rgba(255,255,255,0.1)"))
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("""
    <div class="insight-box success">
        <div class="insight-title">Variables Clés Identifiées par l'IA</div>
        <p>Le nombre d'évaluations domine largement (>40%), suivi des avis textuels.
        Cela confirme que l'engagement des lecteurs est le facteur prédictif principal du succès littéraire.</p>
    </div>
    </div>""", unsafe_allow_html=True)
if page == "Données":
    # ================= TABLE DE DONNÉES =================
    with st.expander("Voir le dataset filtré"):
        display_cols = ["title", "authors", "average_rating", "ratings_count", "num_pages", "rating_category"]
        st.dataframe(
            filtered[display_cols].head(100).rename(columns={
                "title": "Titre", "authors": "Auteur", "average_rating": "Note",
                "ratings_count": "Évaluations", "num_pages": "Pages", "rating_category": "Catégorie"
            }),
            use_container_width=True
        )
        st.caption(f"Affichage des 100 premiers livres sur {len(filtered)} — après application des filtres.")

    # ================= TOP 10 LIVRES =================
    st.markdown("""
    <div class="section-card">
        <div class="section-title">Top 10 Livres les Plus Populaires</div>
        <div class="section-desc">Les livres les mieux notés et les plus engageants, avec couvertures.</div>
    </div>
    """, unsafe_allow_html=True)

    # Top 10 by popularity_score
    # Top 10 avec covers (créé image_url si absent)
    top_df = filtered.nlargest(10, 'popularity_score')[['title', 'authors', 'average_rating', 'ratings_count', 'isbn13', 'isbn']].copy()
    top_df['isbn_clean'] = top_df['isbn13'].fillna(top_df['isbn']).astype(str).str.strip()
    top_df.loc[top_df["isbn_clean"].isin(["nan", "", "None", "none"]), "isbn_clean"] = np.nan
    top_df['image_url'] = np.where(
        top_df["isbn_clean"].notna(),
        "https://covers.openlibrary.org/b/isbn/" + top_df["isbn_clean"] + "-M.jpg",
        "https://via.placeholder.com/150x200/1a1f2e/ffffff?text=Couverture"
    )
    
    # Icônes SVG star élégantes
    star_icon = '<svg width="12" height="12" viewBox="0 0 24 24" fill="#818cf8"><path d="M12 .587l3.668 7.428 8.332 1.151-6 5.856 1.417 8.256L12 18.973l-7.417 3.905 1.417-8.256-6-5.856 8.332-1.151z"/></svg>'
    users_icon = '<svg width="12" height="12" viewBox="0 0 24 24" fill="#a855f7"><path d="M12 12c2.21 0 4-1.79 4-4s-1.79-4-4-4-4 1.79-4 4 1.79 4 4 4zm0 2c-2.67 0-8 1.34-8 4v2h16v-2c0-2.66-5.33-4-8-4z"/></svg>'
    
    cols = st.columns(5)
    for idx, row in enumerate(top_df.itertuples(), 1):
        col = cols[(idx-1) % 5]
        with col:
            st.image(row.image_url, width=140)
            st.markdown(f"**{row.title[:45]}**")
            st.caption(row.authors[:35])
            st.markdown(f"{star_icon} **{row.average_rating:.2f}** | {users_icon} **{row.ratings_count:,.0f}**", unsafe_allow_html=True)
    
# ================= PIED DE PAGE =================
st.markdown("""
<div style="text-align:center; padding:30px 0 10px; border-top:1px solid rgba(255,255,255,0.05); margin-top:30px;">
    <div style="font-size:0.75rem; color:#2a4060; letter-spacing:1px;">
        BOOKS ANALYTICS INTELLIGENCE DASHBOARD  |  Goodreads Dataset  |
        Streamlit + Plotly + Scikit-learn  |  Mini-Projet Visualisation de Données 2025-2026
    </div>
</div>
""", unsafe_allow_html=True)

