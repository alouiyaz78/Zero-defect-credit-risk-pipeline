# 1. Utilisation d'une image Python légère
FROM python:3.11-slim

# 2. Définition du répertoire de travail dans le conteneur
WORKDIR /app

# 3. Installation des dépendances système nécessaires (XGBoost, SHAP, etc.)
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/apt-get/lists/*

# 4. Installation de Poetry
RUN pip install --no-cache-dir poetry

# 5. Copie uniquement des fichiers de dépendances pour optimiser le cache Docker
COPY pyproject.toml poetry.lock ./

# 6. Installation des bibliothèques sans créer de venv (plus simple en Docker)
RUN poetry config virtualenvs.create false \
    && poetry install --no-interaction --no-ansi --no-root

# 7. Copie de l'intégralité du projet (incluant configs/, data/, models/)
COPY . .

# 8. Exposition du port utilisé par Streamlit
EXPOSE 8501

# 9. Commande de lancement de l'application
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]