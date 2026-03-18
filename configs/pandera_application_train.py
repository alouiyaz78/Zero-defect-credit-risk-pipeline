import pandas as pd
import pandera as pa
from pandera import Check, Column, DataFrameSchema


application_train_schema = DataFrameSchema(
    {
        "SK_ID_CURR": Column(
            int,
            nullable=False,
            unique=True,
            required=True,
        ),
        "TARGET": Column(
            int,
            checks=Check.isin([0, 1]),
            nullable=False,
            required=True,
        ),
        "CNT_CHILDREN": Column(
            float,
            checks=[
                Check.greater_than_or_equal_to(0),
                Check.less_than_or_equal_to(20),
            ],
            nullable=True,
            required=True,
        ),
        "AMT_INCOME_TOTAL": Column(
            float,
            checks=[
                Check.greater_than_or_equal_to(0),
                Check.less_than_or_equal_to(1e8),
            ],
            nullable=False,
            required=True,
        ),
        "AMT_CREDIT": Column(
            float,
            checks=[
                Check.greater_than_or_equal_to(0),
                Check.less_than_or_equal_to(1e8),
            ],
            nullable=False,
            required=True,
        ),
        "AMT_ANNUITY": Column(
            float,
            checks=[
                Check.greater_than_or_equal_to(0),
                Check.less_than_or_equal_to(1e7),
            ],
            nullable=True,
            required=True,
        ),
        "AMT_GOODS_PRICE": Column(
            float,
            checks=[
                Check.greater_than_or_equal_to(0),
                Check.less_than_or_equal_to(1e8),
            ],
            nullable=True,
            required=True,
        ),
        "DAYS_BIRTH": Column(
            int,
            checks=Check.less_than(0),
            nullable=False,
            required=True,
        ),
        "DAYS_EMPLOYED": Column(
            int,
            checks=Check(lambda s: ((s <= 0) | (s == 365243)).all()),
            nullable=False,
            required=True,
        ),
        "CODE_GENDER": Column(
            str,
            checks=Check.isin(["M", "F", "XNA"]),
            nullable=False,
            required=True,
        ),
        "FLAG_OWN_CAR": Column(
            str,
            checks=Check.isin(["Y", "N"]),
            nullable=False,
            required=True,
        ),
        "FLAG_OWN_REALTY": Column(
            str,
            checks=Check.isin(["Y", "N"]),
            nullable=False,
            required=True,
        ),
        "NAME_CONTRACT_TYPE": Column(
            str,
            checks=Check.isin(["Cash loans", "Revolving loans"]),
            nullable=False,
            required=True,
        ),
    },
    strict=False,
    coerce=True,
)


def add_derived_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ajoute des colonnes dérivées utiles à la validation métier.
    """
    df = df.copy()
    df["AGE_YEARS"] = -df["DAYS_BIRTH"] / 365.25
    return df


def validate_application_train(df: pd.DataFrame) -> pd.DataFrame:
    """
    Valide le dataset application_train avec Pandera
    puis applique des contrôles métier complémentaires.
    """
    validated_df = application_train_schema.validate(df, lazy=True)
    validated_df = add_derived_columns(validated_df)

    if (validated_df["AGE_YEARS"] < 18).any():
        raise ValueError("Des clients avec un âge inférieur à 18 ans ont été détectés.")

    if (validated_df["AGE_YEARS"] > 100).any():
        raise ValueError("Des clients avec un âge supérieur à 100 ans ont été détectés.")

    annuity_not_null = validated_df["AMT_ANNUITY"].notna()
    invalid_annuity_count = (
        validated_df.loc[annuity_not_null, "AMT_ANNUITY"]
        > validated_df.loc[annuity_not_null, "AMT_CREDIT"]
    ).sum()

    if invalid_annuity_count > 0:
        raise ValueError(
            "Certaines annuités sont supérieures au montant du crédit. "
            "Cette situation peut signaler une incohérence dans les données."
        )

    return validated_df