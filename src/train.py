import json
import joblib
from pathlib import Path

from sklearn.ensemble import (
    RandomForestClassifier,
    BaggingClassifier,
    AdaBoostClassifier,
    GradientBoostingClassifier,
    StackingClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from src.data_prep import load_data, full_preprocess, split_scale_smote
from src.features import add_features  # noqa: F401  (used indirectly)

MODELS_DIR = Path("models")
MODELS_DIR.mkdir(exist_ok=True)


def train_all_models(data_path: str = "data/glass.xlsx"):
    # 1. Data
    df = load_data(data_path)
    df = full_preprocess(df)

    X = df.drop("Type", axis=1)
    y = df["Type"]

    X_train_bal, X_test_scaled, y_train_bal, y_test, scaler = split_scale_smote(X, y)

    # 2. Models
    rf_baseline = RandomForestClassifier(random_state=42)

    rf_tuned = RandomForestClassifier(
        n_estimators=500,
        max_depth=20,
        max_features="sqrt",
        min_samples_split=2,
        min_samples_leaf=1,
        class_weight="balanced",
        bootstrap=False,
        random_state=42,
    )

    bagging = BaggingClassifier(
        estimator=DecisionTreeClassifier(),
        n_estimators=200,
        random_state=42,
    )

    ada = AdaBoostClassifier(
        n_estimators=300,
        learning_rate=0.5,
        random_state=42,
    )

    gb = GradientBoostingClassifier(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=3,
        random_state=42,
    )

    estimators = [
        ("rf", rf_tuned),
        ("gb", gb),
        ("ada", ada),
    ]

    stacking = StackingClassifier(
        estimators=estimators,
        final_estimator=LogisticRegression(max_iter=1000),
        n_jobs=-1,
    )

    models = {
        "rf_baseline": rf_baseline,
        "rf_tuned": rf_tuned,
        "bagging": bagging,
        "ada": ada,
        "gb": gb,
        "stacking": stacking,
    }

    metrics = {}

    # 3. Train & evaluate
    for name, model in models.items():
        model.fit(X_train_bal, y_train_bal)
        y_pred = model.predict(X_test_scaled)

        acc = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred, output_dict=True)
        cm = confusion_matrix(y_test, y_pred).tolist()

        metrics[name] = {
            "accuracy": acc,
            "classification_report": report,
            "confusion_matrix": cm,
        }

        print(f"\n=== {name.upper()} ===")
        print("Accuracy:", acc)

    # 4. Save best model + artifacts (stacking as final model)
    joblib.dump(stacking, MODELS_DIR / "stacking_model.joblib")
    joblib.dump(scaler, MODELS_DIR / "scaler.joblib")
    joblib.dump(X.columns.tolist(), MODELS_DIR / "feature_columns.joblib")

    with open(MODELS_DIR / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    print("\n✅ Saved best model and metrics to 'models/' directory.")


if __name__ == "__main__":
    train_all_models()
