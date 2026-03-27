import streamlit as st
import streamlit.components.v1 as components
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
st.set_page_config(
    page_title="Simulateur de Risque Crédit",
    page_icon="🏦",
    layout="wide"
)

MONTANT_PRET_MOYEN = 10000
MARGE_BENEFICIAIRE = 0.10

# Hypothèses métier FIXES du projet
COUT_FAUX_NEGATIF = 10
COUT_FAUX_POSITIF = 1


@st.cache_resource
def load_model():
    return joblib.load(Path("models/xgboost_baseline.pkl"))


@st.cache_data
def load_data():
    return pd.read_csv(Path("data/processed/test_sample_dashboard.csv"))


def build_age_group(days_birth_series: pd.Series) -> pd.Series:
    age_years = (-days_birth_series / 365.25).round(0)
    bins = [0, 30, 40, 50, 60, 120]
    labels = ["<=30", "31-40", "41-50", "51-60", "60+"]
    return pd.cut(age_years, bins=bins, labels=labels, include_lowest=True)


def fairness_by_group(
    df_features: pd.DataFrame,
    y_true: pd.Series,
    y_pred: np.ndarray,
    group_col: str
) -> pd.DataFrame:
    tmp = df_features.copy()
    tmp["y_true"] = np.asarray(y_true)
    tmp["y_pred"] = np.asarray(y_pred)
    tmp = tmp[tmp[group_col].notna()].copy()

    rows = []
    global_acceptance = (tmp["y_pred"] == 0).mean() if len(tmp) else np.nan

    for group_value, grp in tmp.groupby(group_col):
        tn, fp, fn, tp = confusion_matrix(
            grp["y_true"], grp["y_pred"], labels=[0, 1]
        ).ravel()

        acceptance_rate = (grp["y_pred"] == 0).mean()
        refusal_rate = (grp["y_pred"] == 1).mean()
        default_rate = grp["y_true"].mean()
        fpr = fp / (fp + tn) if (fp + tn) else np.nan
        fnr = fn / (fn + tp) if (fn + tp) else np.nan

        rows.append({
            "groupe": str(group_value),
            "effectif": len(grp),
            "taux_defaut_reel": default_rate,
            "taux_acceptation": acceptance_rate,
            "taux_refus": refusal_rate,
            "false_positive_rate": fpr,
            "false_negative_rate": fnr,
            "ecart_acceptation_vs_global": acceptance_rate - global_acceptance,
        })

    return pd.DataFrame(rows).sort_values("effectif", ascending=False)


def business_cost(
    y_true: pd.Series,
    y_pred: np.ndarray,
    cost_fn: int = COUT_FAUX_NEGATIF,
    cost_fp: int = COUT_FAUX_POSITIF
) -> int:
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return int(fn * cost_fn + fp * cost_fp)


def business_cost_normalized(
    y_true: pd.Series,
    y_pred: np.ndarray,
    cost_fn: int = COUT_FAUX_NEGATIF,
    cost_fp: int = COUT_FAUX_POSITIF
) -> float:
    return business_cost(y_true, y_pred, cost_fn, cost_fp) / len(y_true)


@st.cache_data
def find_best_threshold(
    y_true: pd.Series,
    y_probs: np.ndarray,
    cost_fn: int = COUT_FAUX_NEGATIF,
    cost_fp: int = COUT_FAUX_POSITIF
):
    thresholds = np.linspace(0.10, 0.90, 81)

    best_threshold = 0.50
    best_cost = float("inf")
    rows = []

    for t in thresholds:
        y_pred = (y_probs >= t).astype(int)
        cost = business_cost(y_true, y_pred, cost_fn, cost_fp)

        tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

        rows.append({
            "threshold": round(float(t), 2),
            "cost": cost,
            "tn": int(tn),
            "fp": int(fp),
            "fn": int(fn),
            "tp": int(tp),
        })

        if cost < best_cost:
            best_cost = cost
            best_threshold = t

    curve_df = pd.DataFrame(rows)
    return round(float(best_threshold), 2), int(best_cost), curve_df


# Chargement initial
model = load_model()
df = load_data()

X = df.drop(columns=["TARGET"])
y_true = df["TARGET"]
y_probs = model.predict_proba(X)[:, 1]

# Seuil recommandé calculé à partir des coûts métier FIXES du projet
best_threshold, best_cost, threshold_curve = find_best_threshold(y_true, y_probs)

# ==========================================
# 2. SIDEBAR / NAVIGATION ET RÉGLAGES
# ==========================================
st.sidebar.title("🏦 Navigation")
page = st.sidebar.radio(
    "Aller à",
    [
        "📊 Vue Macro (Finance)",
        "⚖️ Vue Équité",
        "🔍 Vue Micro (Explicabilité)",
        "📉 Drift Monitoring",
    ]
)

st.sidebar.markdown("---")
st.sidebar.title("⚙️ Réglages du projet")

# État initial du seuil de probabilité
if "seuil" not in st.session_state:
    st.session_state.seuil = 0.44

if "champ_saisie" not in st.session_state:
    st.session_state.champ_saisie = 0.44

if "curseur" not in st.session_state:
    st.session_state.curseur = 0.44


def apply_manual_threshold():
    value = float(st.session_state.champ_saisie)
    st.session_state.seuil = value
    st.session_state.curseur = value


def apply_slider_threshold():
    value = float(st.session_state.curseur)
    st.session_state.seuil = value
    st.session_state.champ_saisie = value


def apply_best_threshold():
    value = float(best_threshold)
    st.session_state.seuil = value
    st.session_state.champ_saisie = value
    st.session_state.curseur = value


# Bloc 1 : seuil de probabilité
st.sidebar.markdown("## 🎯 Seuil de probabilité")
st.sidebar.caption(
    "Le modèle produit une probabilité de défaut. "
    "Le seuil ci-dessous détermine à partir de quelle probabilité un dossier est classé comme risqué."
)

st.sidebar.info(
    "Exemple : si la probabilité de défaut d'un client est 0.60 et que le seuil vaut 0.50, "
    "le dossier sera classé en défaut / refusé."
)

st.sidebar.button(
    "🎯 Appliquer le seuil recommandé",
    on_click=apply_best_threshold
)

st.sidebar.metric("Seuil de probabilité recommandé", f"{best_threshold:.2f}")

st.sidebar.number_input(
    "Modifier manuellement le seuil de probabilité",
    min_value=0.01,
    max_value=0.99,
    step=0.01,
    key="champ_saisie",
    on_change=apply_manual_threshold
)

st.sidebar.slider(
    "Ajustement rapide du seuil de probabilité",
    min_value=0.01,
    max_value=0.99,
    step=0.01,
    key="curseur",
    on_change=apply_slider_threshold
)

threshold = float(st.session_state.seuil)
y_pred_custom = (y_probs >= threshold).astype(int)

# Bloc 2 : hypothèses métier fixes
current_cost = business_cost(y_true, y_pred_custom)
current_cost_norm = business_cost_normalized(y_true, y_pred_custom)

st.sidebar.markdown("---")
st.sidebar.markdown("## 💼 Hypothèses métier du projet")
st.sidebar.caption(
    "Ces hypothèses sont FIXES dans ce dashboard et proviennent du notebook final du projet. "
    "Elles servent à calculer le seuil recommandé."
)

st.sidebar.write(f"**Coût d'un faux négatif (FN)** : {COUT_FAUX_NEGATIF}")
st.sidebar.write(f"**Coût d'un faux positif (FP)** : {COUT_FAUX_POSITIF}")

st.sidebar.metric("Seuil de probabilité actuel", f"{threshold:.2f}")
st.sidebar.metric("Coût métier actuel", f"{current_cost:,}")
st.sidebar.metric("Coût métier minimal estimé", f"{best_cost:,}")
st.sidebar.metric("Coût moyen / dossier", f"{current_cost_norm:.4f}")

if abs(threshold - best_threshold) < 1e-9:
    st.sidebar.success("Le seuil de probabilité actuel est déjà le seuil recommandé.")
else:
    st.sidebar.info(
        f"Écart entre seuil actuel et seuil recommandé : {threshold - best_threshold:+.2f}"
    )

# Préparation équité
fairness_df = X.copy()
if "DAYS_BIRTH" in fairness_df.columns:
    fairness_df["AGE_GROUP"] = build_age_group(fairness_df["DAYS_BIRTH"])

available_groups = [
    col for col in [
        "CODE_GENDER",
        "AGE_GROUP",
        "NAME_EDUCATION_TYPE",
        "NAME_FAMILY_STATUS"
    ]
    if col in fairness_df.columns and fairness_df[col].nunique(dropna=True) > 1
]

# ==========================================
# 3. TITRE PRINCIPAL
# ==========================================
st.title("Tableau de Bord : Octroi de Crédit")

# ==========================================
# 4. PAGE MACRO
# ==========================================
if page == "📊 Vue Macro (Finance)":
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred_custom, labels=[0, 1]).ravel()

    resultat_net = (
        (tn * MONTANT_PRET_MOYEN * MARGE_BENEFICIAIRE)
        - (fn * MONTANT_PRET_MOYEN)
    )

    if resultat_net > 0:
        st.success(f"### 📈 RÉSULTAT NET DE L'AGENCE : + {resultat_net:,.0f} €")
    else:
        st.error(f"### 📉 RÉSULTAT NET DE L'AGENCE : {resultat_net:,.0f} €")

    st.markdown("---")

    c1, c2, c3 = st.columns(3)
    c1.error(f"🚨 Perte\n**-{fn * MONTANT_PRET_MOYEN:,.0f} €**")
    c2.info(f"💶 CA\n**+{tn * MONTANT_PRET_MOYEN * MARGE_BENEFICIAIRE:,.0f} €**")
    c3.warning(f"⚠️ Manque à gagner\n**-{fp * MONTANT_PRET_MOYEN * MARGE_BENEFICIAIRE:,.0f} €**")

    st.markdown("---")
    st.subheader("🎯 Lecture du seuil de probabilité")

    k1, k2, k3 = st.columns(3)
    k1.metric("Seuil utilisé", f"{threshold:.2f}")
    k2.metric("Seuil recommandé", f"{best_threshold:.2f}")
    k3.metric("Coût métier actuel", f"{current_cost:,}")

    st.caption(
        "La courbe ci-dessous montre comment le coût métier évolue selon le seuil de probabilité choisi. "
        "Le seuil recommandé est celui qui minimise ce coût métier, selon les hypothèses fixes du projet."
    )

    st.line_chart(threshold_curve.set_index("threshold")["cost"])

    st.markdown("---")

    cg, cm = st.columns(2)

    with cg:
        st.subheader("⚖️ Volumes")
        df_vol = pd.DataFrame({
            "Statut": ["Accordés", "Refusés"],
            "Volume": [tn + fn, tp + fp]
        }).set_index("Statut")
        st.bar_chart(df_vol)

    with cm:
        st.subheader("📊 Matrice de confusion")
        fig, ax = plt.subplots(figsize=(5, 3))
        sns.heatmap(
            confusion_matrix(y_true, y_pred_custom),
            annot=True,
            fmt="d",
            cmap="Blues",
            cbar=False,
            xticklabels=["Accordé", "Refusé"],
            yticklabels=["Solvable", "Défaut"],
            ax=ax
        )
        ax.set_xlabel("Prédiction")
        ax.set_ylabel("Réalité")
        st.pyplot(fig)

# ==========================================
# 5. PAGE ÉQUITÉ
# ==========================================
elif page == "⚖️ Vue Équité":
    st.subheader("⚖️ Analyse d'équité")

    if not available_groups:
        st.info("Aucun groupe exploitable trouvé dans les données.")
    else:
        selected_group = st.selectbox("Groupe à analyser", available_groups)
        report = fairness_by_group(fairness_df, y_true, y_pred_custom, selected_group)

        display_report = report.copy()
        for col in display_report.columns:
            if "taux" in col or "rate" in col or "ecart" in col:
                display_report[col] = (display_report[col] * 100).round(2)

        st.dataframe(display_report, width="stretch", hide_index=True)
        st.bar_chart(
            report.set_index("groupe")[["taux_acceptation", "false_positive_rate"]]
        )

# ==========================================
# 6. PAGE MICRO / SHAP
# ==========================================
elif page == "🔍 Vue Micro (Explicabilité)":
    st.subheader("🔍 Explication par dossier")

    filtre = st.radio(
        "Dossiers :",
        ["🔴 Refusés", "🟢 Accordés", "⚪ Tous"],
        horizontal=True
    )

    if "Refusés" in filtre:
        ids = np.where(y_pred_custom == 1)[0]
    elif "Accordés" in filtre:
        ids = np.where(y_pred_custom == 0)[0]
    else:
        ids = np.arange(len(y_pred_custom))

    client_id = st.selectbox(
        f"Dossier ({len(ids)} filtrés) :",
        ids,
        format_func=lambda x: f"Client N° {x}"
    )

    if st.button("📊 Générer l'analyse SHAP", type="primary"):
        with st.spinner("Analyse SHAP en cours..."):
            import shap

            preprocessor = model.named_steps["preprocessor"]
            classifier = model.named_steps["classifier"]

            client_data = X.iloc[[client_id]]
            client_trans = preprocessor.transform(client_data)

            numeric_features = X.select_dtypes(exclude=["object"]).columns.tolist()
            ohe = preprocessor.named_transformers_["cat"].named_steps["onehot"]
            cat_features = list(ohe.get_feature_names_out())
            features = numeric_features + cat_features

            if hasattr(client_trans, "toarray"):
                client_trans_dense = client_trans.toarray()
            else:
                client_trans_dense = np.asarray(client_trans)

            explainer = shap.TreeExplainer(classifier)
            shap_vals = explainer.shap_values(client_trans_dense)

            fig_shap, ax = plt.subplots(figsize=(8, 4))
            exp = shap.Explanation(
                values=shap_vals[0],
                base_values=explainer.expected_value,
                data=np.round(client_trans_dense[0], 2),
                feature_names=features
            )
            shap.waterfall_plot(exp, max_display=10, show=False)
            st.pyplot(fig_shap)

            st.info(f"Score de risque : **{y_probs[client_id] * 100:.1f}%**")

# ==========================================
# 7. PAGE DRIFT
# ==========================================
elif page == "📉 Drift Monitoring":
    st.subheader("📉 Data Drift Monitoring")
    st.info(
        "Ce rapport Evidently compare les distributions entre les données "
        "de référence et les données courantes afin de détecter un drift."
    )

    try:
        drift_path = Path("drift/data_drift_report.html")

        if drift_path.exists():
            with open(drift_path, "r", encoding="utf-8") as f:
                html_content = f.read()

            components.html(html_content, height=900, scrolling=True)
        else:
            st.warning(
                "⚠️ Rapport de drift introuvable. Lancez le pipeline pour le générer."
            )

    except Exception as e:
        st.error(f"Erreur lors du chargement du rapport de drift : {e}")