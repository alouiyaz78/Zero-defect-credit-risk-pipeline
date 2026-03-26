import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from pathlib import Path

# ==========================================
# 1. CONFIGURATION ET CHARGEMENT
# ==========================================
st.set_page_config(page_title="Simulateur de Risque Crédit", page_icon="🏦", layout="wide")

MONTANT_PRET_MOYEN = 10000
MARGE_BENEFICIAIRE = 0.10

@st.cache_resource
def load_model():
    return joblib.load(Path("models/xgboost_baseline.pkl"))

@st.cache_data
def load_data():
    return pd.read_csv(Path("data/processed/test_sample_dashboard.csv"))

def build_age_group(days_birth_series: pd.Series) -> pd.Series:
    age_years = (-days_birth_series / 365.25).round(0)
    bins = [0, 30, 40, 50, 60, 120]
    labels = ['<=30', '31-40', '41-50', '51-60', '60+']
    return pd.cut(age_years, bins=bins, labels=labels, include_lowest=True)

def fairness_by_group(df_features: pd.DataFrame, y_true: pd.Series, y_pred: np.ndarray, group_col: str) -> pd.DataFrame:
    tmp = df_features.copy()
    tmp['y_true'] = np.asarray(y_true)
    tmp['y_pred'] = np.asarray(y_pred)
    tmp = tmp[tmp[group_col].notna()].copy()

    rows = []
    global_acceptance = (tmp['y_pred'] == 0).mean() if len(tmp) else np.nan

    for group_value, grp in tmp.groupby(group_col):
        tn, fp, fn, tp = confusion_matrix(grp['y_true'], grp['y_pred'], labels=[0, 1]).ravel()
        acceptance_rate = (grp['y_pred'] == 0).mean()
        refusal_rate = (grp['y_pred'] == 1).mean()
        default_rate = grp['y_true'].mean()
        fpr = fp / (fp + tn) if (fp + tn) else np.nan
        fnr = fn / (fn + tp) if (fn + tp) else np.nan

        rows.append({
            'groupe': str(group_value),
            'effectif': len(grp),
            'taux_defaut_reel': default_rate,
            'taux_acceptation': acceptance_rate,
            'taux_refus': refusal_rate,
            'false_positive_rate': fpr,
            'false_negative_rate': fnr,
            'ecart_acceptation_vs_global': acceptance_rate - global_acceptance,
        })

    return pd.DataFrame(rows).sort_values('effectif', ascending=False)

# Chargement initial
model = load_model()
df = load_data()

X = df.drop(columns=['TARGET'])
y_true = df['TARGET']
y_probs = model.predict_proba(X)[:, 1]

# ==========================================
# 2. INTERFACE UTILISATEUR (SYCHRONISATION SLIDER/INPUT)
# ==========================================
st.sidebar.title("🏦 Réglages MLOps")
st.sidebar.markdown("Ajustez le niveau de sévérité de l'algorithme.")

if 'seuil' not in st.session_state:
    st.session_state.seuil = 0.44

# Fonctions de synchronisation pour éviter les lags
def sync_input():
    st.session_state.seuil = st.session_state.champ_saisie

def sync_slider():
    st.session_state.seuil = st.session_state.curseur

st.sidebar.number_input(
    "Saisie précise du seuil",
    min_value=0.01, max_value=0.99, value=st.session_state.seuil,
    step=0.01, key="champ_saisie", on_change=sync_input
)

st.sidebar.slider(
    "Ajustement rapide",
    min_value=0.01, max_value=0.99, value=st.session_state.seuil,
    step=0.01, key="curseur", on_change=sync_slider
)

threshold = st.session_state.seuil
y_pred_custom = (y_probs >= threshold).astype(int)

# Préparation équité
fairness_df = X.copy()
if 'DAYS_BIRTH' in fairness_df.columns:
    fairness_df['AGE_GROUP'] = build_age_group(fairness_df['DAYS_BIRTH'])

available_groups = [
    col for col in ['CODE_GENDER', 'AGE_GROUP', 'NAME_EDUCATION_TYPE', 'NAME_FAMILY_STATUS']
    if col in fairness_df.columns and fairness_df[col].nunique(dropna=True) > 1
]

# ==========================================
# 3. CRÉATION DES ONGLETS
# ==========================================
st.title("Tableau de Bord : Octroi de Crédit")

tab_macro, tab_equite, tab_micro = st.tabs([
    "📊 Vue Macro (Finance)", "⚖️ Vue Équité", "🔍 Vue Micro (Explicabilité)"
])

# ONGLET 1 : MACRO
with tab_macro:
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred_custom, labels=[0, 1]).ravel()
    resultat_net = (tn * MONTANT_PRET_MOYEN * MARGE_BENEFICIAIRE) - (fn * MONTANT_PRET_MOYEN)

    if resultat_net > 0:
        st.success(f"### 📈 RÉSULTAT NET DE L'AGENCE : + {resultat_net:,.0f} €")
    else:
        st.error(f"### 📉 RÉSULTAT NET DE L'AGENCE : {resultat_net:,.0f} €")

    st.markdown("---")
    c1, c2, c3 = st.columns(3)
    c1.error(f"🚨 Perte\n**-{fn*MONTANT_PRET_MOYEN:,.0f}€**")
    c2.info(f"💶 CA\n**+{tn*MONTANT_PRET_MOYEN*MARGE_BENEFICIAIRE:,.0f}€**")
    c3.warning(f"⚠️ Manque à gagner\n**-{fp*MONTANT_PRET_MOYEN*MARGE_BENEFICIAIRE:,.0f}€**")

    st.markdown("---")
    cg, cm = st.columns(2)
    with cg:
        st.subheader("⚖️ Volumes")
        df_vol = pd.DataFrame({"Statut": ["Accordés", "Refusés"], "Volume": [tn+fn, tp+fp]}).set_index("Statut")
        st.bar_chart(df_vol)
    with cm:
        st.subheader("📊 Matrice de Confusion")
        fig, ax = plt.subplots(figsize=(5, 3))
        sns.heatmap(confusion_matrix(y_true, y_pred_custom), annot=True, fmt='d', cmap='Blues', cbar=False, 
                    xticklabels=['Accordé', 'Refusé'], yticklabels=['Solvable', 'Défaut'], ax=ax)
        st.pyplot(fig)

# ONGLET 2 : ÉQUITÉ
with tab_equite:
    if not available_groups:
        st.info("Aucun groupe trouvé.")
    else:
        selected_group = st.selectbox("Groupe à analyser", available_groups)
        report = fairness_by_group(fairness_df, y_true, y_pred_custom, selected_group)
        
        # Correction 2026 : width='stretch' au lieu de use_container_width
        display_report = report.copy()
        for col in report.columns:
            if 'taux' in col or 'rate' in col or 'ecart' in col:
                display_report[col] = (display_report[col] * 100).round(2)
        
        st.dataframe(display_report, width="stretch", hide_index=True)
        st.bar_chart(report.set_index('groupe')[['taux_acceptation', 'false_positive_rate']])

# ONGLET 3 : MICRO (SHAP)
with tab_micro:
    st.subheader("🔍 Explication par dossier")
    
    # Filtrage optimisé pour la fluidité
    filtre = st.radio("Dossiers :", ["🔴 Refusés", "🟢 Accordés", "⚪ Tous"], horizontal=True)
    if "Refusés" in filtre: ids = np.where(y_pred_custom == 1)[0]
    elif "Accordés" in filtre: ids = np.where(y_pred_custom == 0)[0]
    else: ids = np.arange(len(y_pred_custom))

    client_id = st.selectbox(f"Dossier ({len(ids)} filtrés) :", ids, format_func=lambda x: f"Client N° {x}")

    if st.button("📊 Générer l'analyse SHAP", type="primary"):
        with st.spinner('Analyse SHAP en cours...'):
            import shap
            # Extraction des étapes du pipeline
            preprocessor = model.named_steps['preprocessor']
            classifier = model.named_steps['classifier']
            
            client_data = X.iloc[[client_id]]
            client_trans = preprocessor.transform(client_data)
            
            # Reconstruction des noms de colonnes
            ohe = preprocessor.named_transformers_['cat'].named_steps['onehot']
            features = X.select_dtypes(exclude=['object']).columns.tolist() + list(ohe.get_feature_names_out())

            explainer = shap.TreeExplainer(classifier)
            shap_vals = explainer.shap_values(client_trans)

            fig_shap, ax = plt.subplots(figsize=(8, 4))
            exp = shap.Explanation(values=shap_vals[0], base_values=explainer.expected_value, 
                                   data=np.round(client_trans[0], 2), feature_names=features)
            shap.waterfall_plot(exp, max_display=10, show=False)
            st.pyplot(fig_shap)
            st.info(f"Score de risque : **{y_probs[client_id]*100:.1f}%**")