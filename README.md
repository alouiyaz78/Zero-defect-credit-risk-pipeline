# Zero-Defect Credit Risk Pipeline

This project aims to build a robust credit risk pipeline that integrates data quality validation at every stage of the machine learning lifecycle.

## Objective

The goal is to design a system capable of detecting issues across:

* Data quality (missing values, outliers, schema validation)
* Feature engineering and data leakage
* Model performance and robustness
* Fairness across different groups
* Data drift after deployment

## Dataset

The project is based on the Home Credit dataset, including:

* application_train.csv
* application_test.csv
* bureau.csv
* previous_application.csv
* installments_payments.csv
* credit_card_balance.csv

## Tech Stack

* Python
* Pandas, NumPy
* Scikit-learn
* Pandera
* Fairlearn
* Matplotlib / Seaborn
* Poetry

## Project Structure

```
data/
notebooks/
src/
tests/
configs/
reports/
```

## Setup

```bash
poetry install
poetry run jupyter notebook
```

## Status

Project initialization completed.
Currently working on Data Quality validation.

## Team

* Yazid Aloui
* Malik
* Radouane
