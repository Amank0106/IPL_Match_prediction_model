"""
train_model.py
================
Trains the IPL match-winner prediction pipeline and saves it to disk.

This reproduces the modeling logic from the original notebook.ipynb as
faithfully as possible:

    - Same features   : season, city, team1, team2, toss_winner, toss_decision
    - Same target      : winner
    - Same preprocessing: OneHotEncoder(handle_unknown="ignore") on the
      categorical columns, season passed through untouched.
    - Same algorithm   : RandomForestClassifier, selected in the notebook
      after comparing it against LogisticRegression, KNeighborsClassifier,
      and DecisionTreeClassifier.
      
Run with:
    python train_model.py
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

DATA_PATH = Path(__file__).parent / "data.csv"
MODEL_PATH = Path(__file__).parent / "pipeline.pkl"

CATEGORICAL_FEATURES = ["team1", "team2", "toss_winner", "toss_decision", "city"]
NUMERIC_FEATURES = ["season"]
FEATURE_ORDER = ["season", "city", "team1", "team2", "toss_winner", "toss_decision"]
RANDOM_STATE = 42


def load_and_prepare_data(path: Path = DATA_PATH) -> tuple[pd.DataFrame, pd.Series]:
   
    df = pd.read_csv(path)

    data = df.drop(
        columns=[
            "id",
            "player_of_match",
            "umpire1",
            "umpire2",
            "umpire3",
            "date",
            "result",
            "dl_applied",
            "win_by_runs",
            "win_by_wickets",
        ]
    )
    data = data.dropna(subset=["city", "winner"])

    y = data["winner"]
    x = data.drop(columns=["winner", "venue"])
    return x, y


def build_pipeline() -> Pipeline:
    """Build the exact preprocessing + model pipeline used in the notebook."""
    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES),
            ("num", "passthrough", NUMERIC_FEATURES),
        ]
    )
    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", RandomForestClassifier(random_state=RANDOM_STATE)),
        ]
    )
    return pipeline


def train() -> dict:
    x, y = load_and_prepare_data()

    x_train, x_test, y_train, y_test = train_test_split(
        x, y, random_state=RANDOM_STATE, test_size=0.2
    )

    pipeline = build_pipeline()
    pipeline.fit(x_train, y_train)

    predictions = pipeline.predict(x_test)
    accuracy = accuracy_score(y_test, predictions)

    bundle = {
        "pipeline": pipeline,
        "accuracy": accuracy,
        "model_name": "Random Forest Classifier",
        "feature_order": FEATURE_ORDER,
        "categorical_features": CATEGORICAL_FEATURES,
        "numeric_features": NUMERIC_FEATURES,
        "classes": sorted(pipeline.classes_.tolist()),
        "n_matches_used": int(len(x)),
        "n_train": int(len(x_train)),
        "n_test": int(len(x_test)),
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "random_state": RANDOM_STATE,
    }

    joblib.dump(bundle, MODEL_PATH)

    print(f"Trained on {bundle['n_train']} matches, tested on {bundle['n_test']}.")
    print(f"Test accuracy: {accuracy:.4f} ({accuracy * 100:.2f}%)")
    print(f"Classes learned ({len(bundle['classes'])}): {bundle['classes']}")
    print(f"Saved pipeline bundle -> {MODEL_PATH}")

    return bundle


if __name__ == "__main__":
    train()
