import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE


def load_data(path: str = "data/glass.xlsx") -> pd.DataFrame:
    """Load the raw glass dataset."""
    return pd.read_excel(path, sheet_name="glass")


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Handle zeros, remove duplicates, and drop constant columns."""
    df = df.copy()

    # Handle zeros in numeric columns
    if "Type" in df.columns:
        num_cols = df.columns.drop("Type")
    else:
        num_cols = df.columns

    df[num_cols] = df[num_cols].replace(0, df[num_cols].median())

    # Drop duplicates
    df = df.drop_duplicates()

    # Drop constant columns
    constant_cols = [c for c in df.columns if df[c].nunique() == 1]
    if constant_cols:
        df = df.drop(columns=constant_cols)

    return df


def winsorize_outliers(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clip outliers using IQR rule for all numeric columns,
    excluding 'Type' and 'Ba' if present.
    """
    df = df.copy()

    exclude_cols = {"Type", "Ba"}
    num_cols = [c for c in df.columns if c not in exclude_cols]

    if not num_cols:
        return df

    Q1 = df[num_cols].quantile(0.25)
    Q3 = df[num_cols].quantile(0.75)
    IQR = Q3 - Q1

    for col in num_cols:
        lower = Q1[col] - 1.5 * IQR[col]
        upper = Q3[col] + 1.5 * IQR[col]
        df[col] = df[col].clip(lower=lower, upper=upper)

    return df


def full_preprocess(df_raw: pd.DataFrame) -> pd.DataFrame:
    """
    Apply the full preprocessing pipeline used in training:
    - clean_data
    - winsorize_outliers
    - add_features
    """
    from src.features import add_features  # local import to avoid circular

    df = clean_data(df_raw)
    df = winsorize_outliers(df)
    df = add_features(df)
    return df


def split_scale_smote(X, y, test_size=0.2, random_state=42):
    """
    Split into train/test, scale numeric features, then apply SMOTE on train.
    Returns:
    - X_train_bal (scaled & balanced)
    - X_test_scaled (scaled)
    - y_train_bal
    - y_test
    - scaler
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    sm = SMOTE(random_state=random_state)
    X_train_bal, y_train_bal = sm.fit_resample(X_train_scaled, y_train)

    return X_train_bal, X_test_scaled, y_train_bal, y_test, scaler
