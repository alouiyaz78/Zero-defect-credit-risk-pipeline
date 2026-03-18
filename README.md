
# Zero-Defect Credit Risk Pipeline

This project focuses on building a credit risk pipeline that does more than train a predictive model. The main objective is to design a workflow that checks data quality, validates transformations, evaluates model reliability, and detects distribution changes over time.

The project is based on the Home Credit dataset and is organized around the idea that a machine learning system should be monitored and challenged at every stage of the data lifecycle, not only at prediction time.

## Project goal

The goal is to develop a robust credit scoring pipeline capable of identifying weaknesses before, during, and after modeling. The project includes:

- data quality profiling
- schema validation
- outlier detection
- feature engineering checks
- leakage prevention
- model validation with appropriate business-oriented metrics
- fairness analysis
- drift detection and monitoring

Rather than focusing only on model accuracy, this project emphasizes trust, reproducibility, and quality control.

## Dataset

The work uses the Home Credit dataset, which contains a main application table and several related tables describing historical credit behavior, previous applications, installments, and account balances.

Main files used in the project include:

- `application_train.csv`
- `application_test.csv`
- `bureau.csv`
- `bureau_balance.csv`
- `previous_application.csv`
- `installments_payments.csv`
- `credit_card_balance.csv`
- `POS_CASH_balance.csv`
- `HomeCredit_columns_description.csv`

The first stage of the project starts with the main application dataset before progressively integrating additional relational tables for feature enrichment.

## Project structure

```text
zero-defect-credit-risk-pipeline/
├── data/
│   ├── raw/
│   ├── processed/
│   └── drift/
├── notebooks/
├── src/
├── tests/
├── configs/
├── reports/
├── pyproject.toml
├── poetry.lock
└── README.md
