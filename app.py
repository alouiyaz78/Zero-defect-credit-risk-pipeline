import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

# ==========================================
# 1. Configuration de la page
# ==========================================
st.set_page_config(page_title="Dashboard Risque de Crédit", page_icon="🏦", layout="wide")
st.title("🏦 Dashboard d'Octroi de Crédit (Scoring)")
st.markdown("""
Cette application permet de simuler l'impact financier du seuil de décision du modèle d'Intelligence Artificielle.
* **Faux Négatif** (Prêt accordé à un profil à risque) : Coût de **10**
* **Faux Positif** (Prêt refusé à un bon client) : Coût de **1**
""")

# ==========================================
# 2. Chargement des données et du modèle (Mis en cache pour la vitesse)
# ==========================================
@st.cache_resource
def load_model_and_data():
    # Adapter les chemins selon ton arborescence
    model = joblib.load("models/xgboost_baseline.pkl")
    data = pd.read_csv("data/processed/test_sample_dashboard.csv")
    
    # Séparation X et y
    X = data.drop(columns=['TARGET'])
    y = data['TARGET']
    
    # Calculer les probabilités une seule fois
    # predict_proba renvoie [Probabilité_0, Probabilité_1]
    y_prob = model.predict_proba(X)[:, 1] 
    
    return y, y_prob

try:
    y_true, y_prob = load_model_and_data()
except Exception as e:
    st.error(f"Erreur lors du chargement des fichiers : {e}. Vérifiez que vous avez bien exécuté la sauvegarde dans le notebook.")
    st.stop()

# ==========================================
# 3. Barre latérale (Sidebar) - Contrôle du Seuil
# ==========================================
st.sidebar.header("⚙️ Paramètres du Modèle")
st.sidebar.write("Ajustez le seuil de tolérance au risque de la banque :")

# Le fameux slider !
threshold = st.sidebar.slider(
    "Seuil de Probabilité (Threshold)", 
    min_value=0.01, 
    max_value=0.99, 
    value=0.50, 
    step=0.01,
    help="Si la probabilité de défaut dépasse ce seuil, le crédit est refusé."
)

# ==========================================
# 4. Calculs basés sur le seuil dynamique
# ==========================================
# Si la proba > seuil, alors on prédit 1 (Défaut), sinon 0 (Remboursé)
y_pred_dynamic = (y_prob >= threshold).astype(int)

# Calcul de la matrice de confusion
tn, fp, fn, tp = confusion_matrix(y_true, y_pred_dynamic, labels=[0, 1]).ravel()

# Calcul des coûts
cout_fn = fn * 10
cout_fp = fp * 1
cout_total = cout_fn + cout_fp

# ==========================================
# 5. Affichage des KPIs (Indicateurs clés)
# ==========================================
st.markdown("### 📊 Indicateurs de Performance Financière")

col1, col2, col3, col4 = st.columns(4)
col1.metric("Coût Total Simulé", f"{cout_total:,.0f} pts", delta=f"{cout_fp} (FP) + {cout_fn} (FN)", delta_color="inverse")
col2.metric("Vrais Défauts Détectés (TP)", f"{tp}", help="Bons blocages")
col3.metric("Faux Négatifs (FN)", f"{fn}", help="Erreurs critiques : Prêts accordés à tort", delta="- Coût fort", delta_color="inverse")
col4.metric("Faux Positifs (FP)", f"{fp}", help="Erreurs commerciales : Prêts refusés à tort")

# ==========================================
# 6. Visualisation de la Matrice de Confusion
# ==========================================
st.markdown("### 📉 Matrice de Confusion Interactive")

fig, ax = plt.subplots(figsize=(6, 4))
cm_matrix = np.array([[tn, fp], [fn, tp]])

sns.heatmap(cm_matrix, annot=True, fmt='d', cmap='Blues', cbar=False,
            xticklabels=['Prédit Solvable (0)', 'Prédit Défaut (1)'], 
            yticklabels=['Vrai Solvable (0)', 'Vrai Défaut (1)'],
            linewidths=1, linecolor='black', ax=ax)

ax.set_ylabel('Réalité Terrain', fontweight='bold')
ax.set_xlabel('Prédiction du Modèle', fontweight='bold')
ax.set_title(f'Matrice avec Seuil = {threshold:.2f}', pad=15)

# Centrer le graphique
col_g1, col_g2, col_g3 = st.columns([1, 2, 1])
with col_g2:
    st.pyplot(fig)