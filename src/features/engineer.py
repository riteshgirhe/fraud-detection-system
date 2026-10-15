from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


ALLOWED_TYPES = ["CASH_IN", "CASH_OUT", "DEBIT", "PAYMENT", "TRANSFER"]


def safe_divide(numerator: pd.Series, denominator: pd.Series, eps: float = 1e-8) -> pd.Series:
    denom = denominator.replace(0, np.nan)
    return numerator.div(denom.fillna(eps))


def engineer_features(df: pd.DataFrame, keep_target: bool = True) -> pd.DataFrame:
    """Create model-ready features for PaySim transactions.

    The same logic is designed to be used both during training and at inference time,
    so no target-derived or post-hoc features are introduced.
    """
    if df.empty:
        return df.copy()

    frame = df.copy()
    required_columns = {"step", "type", "amount", "oldbalanceOrg", "newbalanceOrig", "oldbalanceDest", "newbalanceDest"}
    missing = sorted(required_columns - set(frame.columns))
    if missing:
        raise ValueError(f"Missing required columns for feature engineering: {missing}")

    if "nameDest" not in frame:
        frame["nameDest"] = ""
    frame["type"] = frame["type"].astype(str).str.upper().str.strip()
    frame["hour_of_day"] = frame["step"] % 24
    frame["day"] = frame["step"] // 24

    frame["errorBalanceOrig"] = frame["newbalanceOrig"] + frame["amount"] - frame["oldbalanceOrg"]
    frame["errorBalanceDest"] = frame["oldbalanceDest"] + frame["amount"] - frame["newbalanceDest"]
    frame["orig_balance_zero_after"] = (frame["newbalanceOrig"] == 0).astype(int)
    frame["dest_balance_zero_before"] = (frame["oldbalanceDest"] == 0).astype(int)
    frame["amount_to_balance_ratio"] = safe_divide(frame["amount"], frame["oldbalanceOrg"] + frame["oldbalanceDest"])
    frame["log_amount"] = np.log1p(frame["amount"].clip(lower=0))

    frame["is_merchant_dest"] = frame["nameDest"].fillna("").astype(str).str.startswith("M").astype(int)

    type_frame = pd.get_dummies(
        frame["type"].astype(pd.CategoricalDtype(categories=ALLOWED_TYPES, ordered=False)),
        prefix="type",
        dtype=np.int8,
    )
    for col in [c for c in [f"type_{t}" for t in ALLOWED_TYPES] if c not in type_frame.columns]:
        type_frame[col] = 0
    type_frame = type_frame[[f"type_{t}" for t in ALLOWED_TYPES]]
    frame = pd.concat([frame, type_frame], axis=1)

    id_columns = ["nameOrig", "nameDest"]
    drop_columns = [col for col in ["type", "nameOrig", "nameDest", "isFlaggedFraud"] if col in frame.columns]
    frame = frame.drop(columns=drop_columns, errors="ignore")

    if not keep_target and "isFraud" in frame.columns:
        frame = frame.drop(columns=["isFraud"], errors="ignore")
    elif keep_target and "isFraud" in frame.columns:
        frame["isFraud"] = frame["isFraud"].astype(int)

    feature_columns = [
        "step",
        "amount",
        "oldbalanceOrg",
        "newbalanceOrig",
        "oldbalanceDest",
        "newbalanceDest",
        "hour_of_day",
        "day",
        "errorBalanceOrig",
        "errorBalanceDest",
        "orig_balance_zero_after",
        "dest_balance_zero_before",
        "amount_to_balance_ratio",
        "log_amount",
        "is_merchant_dest",
        *[f"type_{t}" for t in ALLOWED_TYPES],
    ]
    missing = [col for col in feature_columns if col not in frame.columns]
    if missing:
        raise ValueError(f"Feature engineering failed to add columns: {missing}")

    return frame[feature_columns + (["isFraud"] if "isFraud" in frame.columns else [])].copy()


def get_feature_columns() -> list[str]:
    return [
        "step",
        "amount",
        "oldbalanceOrg",
        "newbalanceOrig",
        "oldbalanceDest",
        "newbalanceDest",
        "hour_of_day",
        "day",
        "errorBalanceOrig",
        "errorBalanceDest",
        "orig_balance_zero_after",
        "dest_balance_zero_before",
        "amount_to_balance_ratio",
        "log_amount",
        "is_merchant_dest",
        *[f"type_{t}" for t in ALLOWED_TYPES],
    ]


class FeatureEngineerTransformer(BaseEstimator, TransformerMixin):
    """Scikit-learn transformer applying the shared transaction feature logic."""

    def __init__(self, include_merchant_feature: bool = True):
        self.include_merchant_feature = include_merchant_feature

    def fit(self, X: pd.DataFrame, y=None):
        self.feature_names_in_ = np.asarray(X.columns, dtype=object)
        self.n_features_in_ = len(self.feature_names_in_)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if not isinstance(X, pd.DataFrame):
            raise TypeError("Feature engineering expects a pandas DataFrame.")
        features = engineer_features(X, keep_target=False)
        if not self.include_merchant_feature:
            features = features.drop(columns=["is_merchant_dest"])
        return features

    def get_feature_names_out(self, input_features=None) -> np.ndarray:
        columns = get_feature_columns()
        if not self.include_merchant_feature:
            columns.remove("is_merchant_dest")
        return np.asarray(columns, dtype=object)
