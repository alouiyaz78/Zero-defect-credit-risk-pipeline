# Zero-Defect Credit Risk Pipeline

## Description
Ce projet implémente un pipeline MLOps complet pour le scoring de crédit. L'approche **"Zéro Défaut"** garantit la fiabilité du système grâce à une validation stricte de la qualité des données (Data QC), une optimisation basée sur les coûts financiers réels et une surveillance continue de l'équité (Fairness) et de la dérive (Drift).

## Objectifs du Projet
* **Validation Qualité :** Détection automatique des anomalies via Pandera et fichiers de configuration YAML.
* **Optimisation Mémoire :** Gestion "Memory Safe" pour le traitement de datasets volumineux sans crash RAM.
* **Performance Métier :** Optimisation du seuil de décision (0.44) pour minimiser la perte financière (Pénalité FN=10, FP=1).
* **Éthique & Monitoring :** Analyse de l'équité par tranches d'âge et détection du Data Drift avec Evidently AI.

## Structure du Projet
* **`PIPELINE_MASTER.ipynb`** : Notebook central orchestrant tout le flux (Qualité -> Engineering -> Modèle -> Drift).
* **`app_fairness.py`** : Dashboard Streamlit interactif incluant le simulateur de risque et l'analyse d'équité.
* **`application_quality.yaml`** : Configuration technique des tests de qualité (Contrat de données).
* **`pandera_schemas.py`** : Scripts de validation automatisée des schémas.
* **`models/`** : Répertoire du modèle final exporté (`xgboost_baseline.pkl`).
* **`notebooks/archives/`** : Historique complet des étapes de recherche et développement.

## Stack Technique
* **Langage :** Python 3.11
* **Data & Validation :** Pandas, NumPy, Pandera
* **Machine Learning :** Scikit-learn, XGBoost, SHAP
* **MLOps :** Evidently AI, Fairlearn
* **Interface & Déploiement :** Streamlit, Poetry, Docker

## Installation et Utilisation

### 1. Installation des dépendances
Assurez-vous d'avoir Poetry installé, puis lancez la commande suivante à la racine :
```bash
poetry install
2. Lancement du Dashboard Streamlit
Pour accéder à l'interface interactive (Simulateur, Business Metrics et Fairness) :

Bash
poetry run streamlit run app_fairness.py
3. Exécution du Pipeline Master
Le notebook principal peut être ouvert pour reproduire l'intégralité du pipeline de données :

Bash
poetry run jupyter notebook PIPELINE_MASTER.ipynb
Déploiement via Docker

1. Construction de l'image Docker
Bash
docker build -t credit-scoring-app .
2. Lancement du conteneur
Bash
docker run -p 8501:8501 credit-scoring-app
L'application sera ensuite accessible sur votre navigateur à l'adresse : http://localhost:8501

Équipe

Yazid 

Malik

Radouane