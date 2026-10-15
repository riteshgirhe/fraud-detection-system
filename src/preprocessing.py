from __future__ import annotations

from pathlib import Path

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.config import RANDOM_STATE


TARGET = 'isFraud'
NUMERIC_FEATURES = [
    'step', 'amount', 'oldbalanceOrg', 'newbalanceOrig', 'oldbalanceDest', 'newbalanceDest',
    'hour_of_day', 'day', 'errorBalanceOrig', 'errorBalanceDest', 'orig_balance_zero_after',
    'dest_balance_zero_before', 'amount_to_balance_ratio', 'log_amount', 'is_merchant_dest'
]
CATEGORICAL_FEATURES = [f'type_{kind}' for kind in ['CASH_IN', 'CASH_OUT', 'DEBIT', 'PAYMENT', 'TRANSFER']]


def build_preprocessor() -> ColumnTransformer:
    numeric_transformer = Pipeline([('scaler', StandardScaler())])
    categorical_transformer = Pipeline([('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))])
    return ColumnTransformer(
        transformers=[
            ('numeric', numeric_transformer, NUMERIC_FEATURES),
            ('categorical', categorical_transformer, CATEGORICAL_FEATURES),
        ],
        remainder='drop',
    )


def fit_preprocessor(train_df: pd.DataFrame, output_path: str | Path = 'models/preprocessor.joblib') -> Pipeline:
    X = train_df.drop(columns=[TARGET], errors='ignore')
    preprocessor = build_preprocessor()
    preprocessor.fit(X)
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(preprocessor, output)
    return preprocessor


def load_preprocessor(path: str | Path = 'models/preprocessor.joblib'):
    return joblib.load(path)
