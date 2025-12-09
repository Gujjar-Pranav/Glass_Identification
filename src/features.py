import pandas as pd


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add engineered features for the Glass dataset.
    Only creates a feature if the required base columns exist.
    """
    df = df.copy()

    # 1. Ratios between key oxides
    if {"Na", "Ca"}.issubset(df.columns):
        df["Na_Ca_ratio"] = df["Na"] / (df["Ca"] + 1e-6)

    if {"Mg", "Ca"}.issubset(df.columns):
        df["Mg_Ca_ratio"] = df["Mg"] / (df["Ca"] + 1e-6)

    # 2. Interaction between Al and Si
    if {"Al", "Si"}.issubset(df.columns):
        df["Al_Si_interaction"] = df["Al"] * df["Si"]

    # 3. Non-linear transformation of RI + binning
    if "RI" in df.columns:
        df["RI_squared"] = df["RI"] ** 2

        if "RI_bin" not in df.columns:
            df["RI_bin"] = pd.qcut(
                df["RI"],
                q=4,
                labels=False,
                duplicates="drop"
            )
            df = pd.get_dummies(df, columns=["RI_bin"], drop_first=True)

    return df
