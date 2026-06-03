"""Model scoring: turn mention<->candidate pairs into match probabilities.

`make_scorer` builds a callable `candidates_df -> match_probability Series`. The real
XGBoost model and featurizer are loaded lazily by default; both are injectable so the
wiring tests without ML deps.
"""
from __future__ import annotations

import os
import pickle
from typing import Callable, Optional

import pandas as pd

_MODEL_PATH = os.path.join(os.path.dirname(__file__), "models", "best_model_xgboost.pkl")
_model = None


def load_model():
    """Lazily load and cache the pickled XGBoost model."""
    global _model
    if _model is None:
        with open(_MODEL_PATH, "rb") as f:
            _model = pickle.load(f)
    return _model


def make_scorer(model=None, featurize_fn: Optional[Callable] = None) -> Callable:
    """Build a scorer. With no args it uses the real model + featurizer (loaded lazily)."""

    def scorer(candidates: pd.DataFrame) -> pd.Series:
        m = model if model is not None else load_model()
        ff = featurize_fn
        if ff is None:
            from resolve.features import featurize as ff  # lazy: pulls in heavy deps
        feats = ff(candidates)
        cols = [c for c in feats.columns if c in m.feature_names_in_]
        X = pd.DataFrame(feats[cols].values, columns=cols, index=feats.index)
        proba = m.predict_proba(X)[:, 1]
        return pd.Series(proba, index=candidates.index)

    return scorer


def xgboost_scorer(candidates: pd.DataFrame) -> pd.Series:
    """Default scorer used by PostMatcher (real model + featurizer)."""
    return make_scorer()(candidates)
