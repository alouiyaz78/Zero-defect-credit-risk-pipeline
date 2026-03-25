import pandas as pd
from pandera import Check, Column, DataFrameSchema


# =========================================================
# application_train
# =========================================================
application_train_schema = DataFrameSchema(
    {
        "SK_ID_CURR": Column(int, nullable=False, unique=True),
        "TARGET": Column(int, checks=Check.isin([0, 1]), nullable=False),
        "AMT_INCOME_TOTAL": Column(float, checks=Check.greater_than_or_equal_to(0), nullable=False),
        "DAYS_BIRTH": Column(int, checks=Check.less_than(0), nullable=False),
    },
    strict=False,
    coerce=True,
)

def validate_application_train(df: pd.DataFrame) -> pd.DataFrame:
    return application_train_schema.validate(df, lazy=True)


# =========================================================
# bureau
# =========================================================
bureau_schema = DataFrameSchema(
    {
        "SK_ID_CURR": Column(int, nullable=False),
        "SK_ID_BUREAU": Column(int, nullable=False, unique=True),
        "AMT_CREDIT_SUM": Column(float, checks=Check.greater_than_or_equal_to(0), nullable=True),
    },
    strict=False,
    coerce=True,
)

def validate_bureau(df: pd.DataFrame) -> pd.DataFrame:
    return bureau_schema.validate(df, lazy=True)


# =========================================================
# previous_application
# =========================================================
previous_application_schema = DataFrameSchema(
    {
        "SK_ID_PREV": Column(int, nullable=False, unique=True),
        "SK_ID_CURR": Column(int, nullable=False),
        "AMT_APPLICATION": Column(float, checks=Check.greater_than_or_equal_to(0), nullable=True),
        "AMT_CREDIT": Column(float, checks=Check.greater_than_or_equal_to(0), nullable=True),
    },
    strict=False,
    coerce=True,
)

def validate_previous_application(df: pd.DataFrame) -> pd.DataFrame:
    return previous_application_schema.validate(df, lazy=True)


# =========================================================
# installments_payments
# =========================================================
installments_payments_schema = DataFrameSchema(
    {
        "SK_ID_PREV": Column(int, nullable=False),
        "SK_ID_CURR": Column(int, nullable=False),
        "AMT_INSTALMENT": Column(float, checks=Check.greater_than_or_equal_to(0), nullable=True),
        "AMT_PAYMENT": Column(float, checks=Check.greater_than_or_equal_to(0), nullable=True),
    },
    strict=False,
    coerce=True,
)

def validate_installments_payments(df: pd.DataFrame) -> pd.DataFrame:
    return installments_payments_schema.validate(df, lazy=True)


# =========================================================
# credit_card_balance
# =========================================================
credit_card_balance_schema = DataFrameSchema(
    {
        "SK_ID_PREV": Column(int, nullable=False),
        "SK_ID_CURR": Column(int, nullable=False),
        "AMT_BALANCE": Column(float, nullable=True),
        "AMT_CREDIT_LIMIT_ACTUAL": Column(float, checks=Check.greater_than_or_equal_to(0), nullable=True),
    },
    strict=False,
    coerce=True,
)

def validate_credit_card_balance(df: pd.DataFrame) -> pd.DataFrame:
    return credit_card_balance_schema.validate(df, lazy=True)


# =========================================================
# POS_CASH_balance
# =========================================================
pos_cash_balance_schema = DataFrameSchema(
    {
        "SK_ID_PREV": Column(int, nullable=False),
        "SK_ID_CURR": Column(int, nullable=False),
        "CNT_INSTALMENT": Column(float, checks=Check.greater_than_or_equal_to(0), nullable=True),
        "CNT_INSTALMENT_FUTURE": Column(float, checks=Check.greater_than_or_equal_to(0), nullable=True),
    },
    strict=False,
    coerce=True,
)

def validate_pos_cash_balance(df: pd.DataFrame) -> pd.DataFrame:
    return pos_cash_balance_schema.validate(df, lazy=True)


# =========================================================
# bureau_balance
# =========================================================
bureau_balance_schema = DataFrameSchema(
    {
        "SK_ID_BUREAU": Column(int, nullable=False),
        "MONTHS_BALANCE": Column(int, nullable=False),
        "STATUS": Column(str, nullable=True),
    },
    strict=False,
    coerce=True,
)

def validate_bureau_balance(df: pd.DataFrame) -> pd.DataFrame:
    return bureau_balance_schema.validate(df, lazy=True)