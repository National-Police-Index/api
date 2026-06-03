"""Scoring wiring test with a fake model + fake featurizer (no ML deps).

A separate integration test (-m integration) exercises the real XGBoost model.
"""
import numpy as np
import pandas as pd

from resolve.scoring import make_scorer


class FakeModel:
    feature_names_in_ = np.array(["f1", "f2"])

    def predict_proba(self, X):
        # positive-class prob = mean of the two features (deterministic, in [0,1])
        p1 = X.mean(axis=1).to_numpy()
        return np.column_stack([1 - p1, p1])


def fake_featurize(candidates):
    df = candidates.copy()
    df["f1"] = [0.9, 0.1][: len(df)]
    df["f2"] = [0.9, 0.1][: len(df)]
    df["extra_col_ignored"] = 1.0
    return df


def test_scorer_returns_positive_class_probabilities_aligned_to_index():
    candidates = pd.DataFrame(
        [{"post_person_nbr": "P1"}, {"post_person_nbr": "P2"}],
        index=[5, 9],  # non-default index to verify alignment
    )
    scorer = make_scorer(model=FakeModel(), featurize_fn=fake_featurize)
    probs = scorer(candidates)
    assert list(probs.index) == [5, 9]
    assert probs.iloc[0] == 0.9  # P1 high
    assert probs.iloc[1] == 0.1  # P2 low


def test_scorer_uses_only_model_feature_columns():
    candidates = pd.DataFrame([{"post_person_nbr": "P1"}])
    scorer = make_scorer(model=FakeModel(), featurize_fn=fake_featurize)
    # extra_col_ignored must not break prediction
    probs = scorer(candidates)
    assert len(probs) == 1
