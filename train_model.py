import json
import joblib
import pandas as pd
from pathlib import Path

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeRegressor


DATA_PATH = Path("coal_quality_data.csv")
MODEL_PATH = Path("coal_quality_model.pkl")
METRICS_PATH = Path("model_metrics.json")


FEATURE_COLUMNS = [
    "mine_location",
    "moisture",
    "ash",
    "volatile_matter",
    "fixed_carbon",
    "sulphur",
    "temperature",
    "depth",
]

TARGET_COLUMN = "gcv"


def load_data() -> pd.DataFrame:
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"{DATA_PATH} not found. Keep coal_quality_data.csv in the same folder as train_model.py"
        )

    df = pd.read_csv(DATA_PATH)

    required_columns = FEATURE_COLUMNS + [TARGET_COLUMN]
    missing = [col for col in required_columns if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in dataset: {missing}")

    df = df.dropna(subset=required_columns).copy()
    return df


def build_preprocessor() -> ColumnTransformer:
    numeric_features = [
        "moisture",
        "ash",
        "volatile_matter",
        "fixed_carbon",
        "sulphur",
        "temperature",
        "depth",
    ]
    categorical_features = ["mine_location"]

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_features),
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features),
        ]
    )
    return preprocessor


def evaluate_model(name, model, X_train, X_test, y_train, y_test):
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    rmse = mean_squared_error(y_test, predictions) ** 0.5

    return {
        "model_name": name,
        "mae": round(mean_absolute_error(y_test, predictions), 3),
        "rmse": round(rmse, 3),
        "r2_score": round(r2_score(y_test, predictions), 4),
        "trained_model": model,
    }


def main():
    df = load_data()

    X = df[FEATURE_COLUMNS]
    y = df[TARGET_COLUMN]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    preprocessor = build_preprocessor()

    candidate_models = [
        (
            "Linear Regression",
            Pipeline(
                steps=[
                    ("preprocessor", preprocessor),
                    ("model", LinearRegression()),
                ]
            ),
        ),
        (
            "Decision Tree",
            Pipeline(
                steps=[
                    ("preprocessor", preprocessor),
                    ("model", DecisionTreeRegressor(random_state=42, max_depth=8)),
                ]
            ),
        ),
        (
            "Random Forest",
            Pipeline(
                steps=[
                    ("preprocessor", preprocessor),
                    (
                        "model",
                        RandomForestRegressor(
                            n_estimators=250,
                            random_state=42,
                            max_depth=None,
                            min_samples_leaf=2,
                            n_jobs=-1,
                        ),
                    ),
                ]
            ),
        ),
    ]

    results = []
    for name, model in candidate_models:
        result = evaluate_model(name, model, X_train, X_test, y_train, y_test)
        results.append(result)
        print(
            f"{name}: MAE={result['mae']}, RMSE={result['rmse']}, R2={result['r2_score']}"
        )

    # For this hackathon app, we deliberately save Random Forest as the final model.
    # Reason: it performs strongly on tabular data and provides feature importance,
    # which makes the dashboard more explainable for judges.
    final_result = next(item for item in results if item["model_name"] == "Random Forest")
    final_model = final_result["trained_model"]

    joblib.dump(final_model, MODEL_PATH)

    metrics_to_save = {
        "final_model": final_result["model_name"],
        "mae": final_result["mae"],
        "rmse": final_result["rmse"],
        "r2_score": final_result["r2_score"],
        "features": FEATURE_COLUMNS,
        "target": TARGET_COLUMN,
        "rows_used": len(df),
    }

    with open(METRICS_PATH, "w") as f:
        json.dump(metrics_to_save, f, indent=4)

    print("\nFinal Random Forest model saved successfully.")
    print("Note: Linear Regression may score extremely high because this synthetic dataset appears formula-based.")
    print("Random Forest is saved because it gives strong accuracy and useful feature importance for the dashboard.")
    print(f"Model file: {MODEL_PATH}")
    print(f"Metrics file: {METRICS_PATH}")


if __name__ == "__main__":
    main()
