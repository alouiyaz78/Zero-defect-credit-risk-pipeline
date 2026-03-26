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

model = load_model()
df = load_data()

X = df.drop(columns=['TARGET'])
y_true = df['TARGET']
y_probs = model.predict_proba(X)[:, 1]

# ==========================================
# 2. INTERFACE UTILISATEUR (BARRE LATÉRALE GLOBALE)
# ==========================================
st.sidebar.title("🏦 Réglages MLOps")
st.sidebar.markdown("Ajustez le niveau de sévérité de l'algorithme.")

if 'seuil' not in st.session_state:
    st.session_state.seuil = 0.44  # On met ton excellent seuil par défaut !

def update_slider():
    st.session_state.seuil = st.session_state.champ_saisie

def update_champ():
    st.session_state.seuil = st.session_state.curseur

st.sidebar.number_input("Saisie précise du seuil", min_value=0.01, max_value=0.99, 
                        value=st.session_state.seuil, step=0.01, 
                        key="champ_saisie", on_change=update_slider)

st.sidebar.slider("Ajustement rapide", min_value=0.01, max_value=0.99, 
                  value=st.session_state.seuil, step=0.01, 
                  key="curseur", on_change=update_champ)

threshold = st.session_state.seuil
y_pred_custom = (y_probs >= threshold).astype(int)

# ==========================================
# 3. CRÉATION DES ONGLETS PRINCIPAUX
# ==========================================
st.title("Tableau de Bord : Octroi de Crédit")

# Création des deux onglets
tab_macro, tab_micro = st.tabs(["📊 Vue Macro (Simulateur Financier)", "🔍 Vue Micro (Explicabilité Client)"])

# ==========================================
# ONGLET 1 : LA VUE AGENCE (FINANCES ET VOLUMES)
# ==========================================
with tab_macro:
    # Calculs financiers
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred_custom, labels=[0, 1]).ravel()
    gains_interets_tn = tn * (MONTANT_PRET_MOYEN * MARGE_BENEFICIAIRE) 
    perte_totale_fn = fn * MONTANT_PRET_MOYEN  
    manque_a_gagner_fp = fp * (MONTANT_PRET_MOYEN * MARGE_BENEFICIAIRE) 
    resultat_net = gains_interets_tn - perte_totale_fn

    # Bilan
    if resultat_net > 0:
        st.success(f"### 📈 RÉSULTAT NET DE L'AGENCE : + {resultat_net:,.0f} €")
    else:
        st.error(f"### 📉 RÉSULTAT NET DE L'AGENCE : {resultat_net:,.0f} €")

    st.markdown("---")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.error(f"🚨 Capital Perdu\n\n**-{perte_totale_fn:,.0f} €**\n\n*{fn} faux négatifs*")
    with col2:
        st.info(f"💶 Chiffre d'Affaires\n\n**+{gains_interets_tn:,.0f} €**\n\n*{tn} vrais négatifs*")
    with col3:
        st.warning(f"⚠️ Manque à gagner\n\n**-{manque_a_gagner_fp:,.0f} €**\n\n*{fp} faux positifs*")

    st.markdown("---")
    col_graphe, col_matrice = st.columns(2)
    with col_graphe:
        st.subheader("⚖️ Volumes de Décision")
        df_volumes = pd.DataFrame({
            "Statut": ["Accordés", "Refusés"],
            "Volume": [tn + fn, tp + fp]
        }).set_index("Statut")
        st.bar_chart(df_volumes, color=["#1f77b4"])
        st.caption(f"**Taux d'acceptation :** {((tn + fn) / len(y_true) * 100):.1f} %")

    with col_matrice:
        st.subheader("📊 Matrice de Confusion")
        fig, ax = plt.subplots(figsize=(5, 3))
        cm_custom = confusion_matrix(y_true, y_pred_custom)
        sns.heatmap(cm_custom, annot=True, fmt='d', cmap='Blues', cbar=False,
                    xticklabels=['Accordé', 'Refusé'], yticklabels=['Solvable', 'Défaut'])
        st.pyplot(fig)

# ==========================================
# ONGLET 2 : LA VUE CLIENT (SHAP)
# ==========================================
with tab_micro:
    st.subheader("Moteur d'explicabilité des décisions algorithmiques")
    st.markdown("Sélectionnez un dossier spécifique pour justifier le choix de l'IA.")

    col_filtre, col_client = st.columns(2)
    with col_filtre:
        filtre_statut = st.radio(
            "Filtrer les dossiers :", 
            ["🔴 Seulement les Refusés", "🟢 Seulement les Accordés", "⚪ Tous"]
        )

    if "Refusés" in filtre_statut:
        dossiers_disponibles = np.where(y_pred_custom == 1)[0]
    elif "Accordés" in filtre_statut:
        dossiers_disponibles = np.where(y_pred_custom == 0)[0]
    else:
        dossiers_disponibles = np.arange(len(y_pred_custom))

    with col_client:
        client_choisi = st.selectbox(
            f"Dossiers ({len(dossiers_disponibles)} filtrés) :", 
            dossiers_disponibles,
            format_func=lambda x: f"Dossier N° {x}" 
        )

    if st.button("📊 Générer le rapport SHAP pour ce dossier", type="primary"):
        with st.spinner('Analyse des variables en cours...'):
            import shap
            
            client_data = X.iloc[[client_choisi]]
            preprocessor = model.named_steps['preprocessor']
            xgb_classifier = model.named_steps['classifier']
            
            client_transformed = preprocessor.transform(client_data)
            
            cat_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()
            num_cols = X.select_dtypes(exclude=['object', 'category']).columns.tolist()
            ohe_features = preprocessor.named_transformers_['cat'].named_steps['onehot'].get_feature_names_out(cat_cols)
            all_features = num_cols + list(ohe_features)
            
            explainer = shap.TreeExplainer(xgb_classifier)
            shap_values = explainer.shap_values(client_transformed)
            
            fig_shap, ax_shap = plt.subplots(figsize=(8, 5))
            shap_val_client = shap.Explanation(
                values=shap_values[0],
                base_values=explainer.expected_value,
                data=np.round(client_transformed[0], 2),
                feature_names=all_features
            )
            
            shap.waterfall_plot(shap_val_client, max_display=10, show=False)
            plt.tight_layout()
            st.pyplot(fig_shap)
            
            proba_defaut = y_probs[client_choisi] * 100
            st.warning(f"💡 Risque de défaut de paiement estimé à **{proba_defaut:.1f}%** (Le seuil actuel de refus est fixé à {threshold*100:.0f}%).")