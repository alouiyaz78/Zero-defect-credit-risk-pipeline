# 💳 Zero-Defect Credit Risk Pipeline

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge.svg)](https://zero-defect-credit-risk-pipeline-fxqrngtu5bmqpctnghwnqn.streamlit.app/)

![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)
![MLOps Status](https://img.shields.io/badge/MLOps-Production--Ready-brightgreen.svg)
![License](https://img.shields.io/badge/License-MIT-yellow.svg)

## 🌟 Aperçu
![Dashboard Screenshot](screenshots/dashboard.png)

Ce projet implémente un pipeline **MLOps de bout en bout** pour le scoring de crédit. L'approche **"Zéro Défaut"** ne se contente pas de prédire : elle garantit la fiabilité du système à chaque étape, de l'ingestion des données à la surveillance en production.

👉 **[Accéder au Dashboard Live](https://zero-defect-credit-risk-pipeline-fxqrngtu5bmqpctnghwnqn.streamlit.app/)**

---

## 🛡️ Les 4 Piliers "Zéro Défaut"

| Pilier | Technologie | Impact Métier |
| :--- | :--- | :--- |
| **Validation Qualité** | `Pandera` & `YAML` | Garantit l'intégrité des données avant le scoring (Data QC). |
| **Optimisation Coût** | `XGBoost` (Seuil 0.44) | Minimise les pertes financières réelles (Pénalité FN=10, FP=1). |
| **Interprétabilité** | `SHAP` | Explique chaque décision de crédit pour la conformité bancaire. |
| **Éthique & Drift** | `Fairlearn` & `Evidently` | Surveille les biais (âge/genre) et l'obsolescence du modèle. |

---

## 🏗️ Structure du Projet

* **`app.py`** : Dashboard Streamlit interactif (Simulateur, Business Metrics et Fairness).
* **`PIPELINE_MASTER.ipynb`** : Orchestration centrale (Qualité ➡️ Engineering ➡️ Modèle ➡️ Drift).
* **`configs/`** : Contrats de données (`YAML`) et schémas de validation (`Pandera`).
* **`models/`** : Modèle final exporté (`xgboost_baseline.pkl`).
* **`data/processed/`** : Données optimisées et échantillonnées pour la performance web.
* **`Dockerfile`** : Configuration pour le déploiement conteneurisé.

---

## 🛠️ Stack Technique
* **Data Science :** Pandas, NumPy, Scikit-learn, XGBoost.
* **Interprétabilité :** SHAP (Shapley Additive Explanations).
* **Monitoring :** Evidently AI (Data Drift), Fairlearn (Fairness Metrics).
* **Déploiement & DevOps :** Docker, Streamlit Cloud, Poetry (gestion des dépendances).

---

## 🚀 Installation et Utilisation

### 1. Cloner le projet
```bash
git clone [https://github.com/alouiyaz78/Zero-defect-credit-risk-pipeline.git](https://github.com/alouiyaz78/Zero-defect-credit-risk-pipeline.git)
cd Zero-defect-credit-risk-pipeline

2. Lancement avec Docker (Recommandé)
Docker garantit que l'application fonctionnera sur n'importe quelle machine sans installation manuelle de Python.

Bash
# Construire l'image
docker build -t credit-scoring-app .

# Lancer le conteneur
docker run -p 8501:8501 credit-scoring-app
L'application sera accessible sur : http://localhost:8501

3. Installation manuelle (Poetry)
Bash
# Installer les dépendances
poetry install

# Lancer l'application Streamlit
poetry run streamlit run app.py

👥 L'Équipe
Projet réalisé avec rigueur par :

Yazid ALOUI

Malik

Radouane
