"""Train the Fraudlens XGBoost fraud-risk classifier."""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass

import joblib
import pandas as pd
from sklearn.metrics import average_precision_score, precision_recall_fscore_support, roc_auc_score
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

from fraudlens import config
from fraudlens.ml.features import FEATURE_COLUMNS, build_training_features, select_feature_matrix


@dataclass
class TrainingMetrics:
    n_train: int
    n_test: int
    fraud_rate: float
    roc_auc: float
    average_precision: float
    precision_at_0_5: float
    recall_at_0_5: float
    f1_at_0_5: float


def load_transactions() -> pd.DataFrame:
    return pd.read_csv(config.TRANSACTIONS_CSV)


def train_model(transactions: pd.DataFrame) -> tuple[XGBClassifier, TrainingMetrics]:
    """Engineer features, fit an XGBoost classifier, and evaluate on a held-out split."""
    features_df = build_training_features(transactions)
    X = select_feature_matrix(features_df)
    y = features_df["is_fraud"].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=config.RANDOM_SEED, stratify=y
    )

    pos = int(y_train.sum())
    neg = int(len(y_train) - pos)
    scale_pos_weight = (neg / pos) if pos > 0 else 1.0

    model = XGBClassifier(
        n_estimators=250,
        max_depth=4,
        learning_rate=0.08,
        subsample=0.9,
        colsample_bytree=0.9,
        scale_pos_weight=scale_pos_weight,
        eval_metric="logloss",
        random_state=config.RANDOM_SEED,
    )
    model.fit(X_train, y_train)

    proba = model.predict_proba(X_test)[:, 1]
    preds = (proba >= 0.5).astype(int)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_test, preds, average="binary", zero_division=0
    )

    metrics = TrainingMetrics(
        n_train=len(X_train),
        n_test=len(X_test),
        fraud_rate=float(y.mean()),
        roc_auc=float(roc_auc_score(y_test, proba)),
        average_precision=float(average_precision_score(y_test, proba)),
        precision_at_0_5=float(precision),
        recall_at_0_5=float(recall),
        f1_at_0_5=float(f1),
    )
    return model, metrics


def save_artifacts(model: XGBClassifier, metrics: TrainingMetrics) -> None:
    config.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, config.MODEL_PATH)
    config.METRICS_PATH.write_text(json.dumps(asdict(metrics), indent=2))
    config.FEATURE_SPEC_PATH.write_text(json.dumps({"feature_columns": FEATURE_COLUMNS}, indent=2))
