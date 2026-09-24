"""Conservative baseline model and candidate labeling."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from mlchallenge.contracts import truth_mapping
from mlchallenge.features import FEATURE_COLUMNS


def label_candidate_pairs(
    feature_frame: pd.DataFrame,
    ground_truth: pd.DataFrame,
) -> np.ndarray:
    truth = truth_mapping(ground_truth)
    labels = [
        int(str(row.candidate_entity_id) in truth[str(row.source1_entity_id)])
        for row in feature_frame.itertuples(index=False)
    ]
    return np.asarray(labels, dtype=np.int8)


@dataclass
class BaselineMatcher:
    """Scaled logistic regression used to establish an interpretable baseline."""

    regularization_c: float = 1.0
    max_iter: int = 2_000

    def __post_init__(self) -> None:
        self.pipeline = Pipeline(
            [
                ("scale", StandardScaler()),
                (
                    "classifier",
                    LogisticRegression(
                        C=self.regularization_c,
                        class_weight="balanced",
                        max_iter=self.max_iter,
                        random_state=0,
                    ),
                ),
            ]
        )

    def fit(self, feature_frame: pd.DataFrame, labels: np.ndarray) -> BaselineMatcher:
        unique = np.unique(labels)
        if not np.array_equal(unique, np.array([0, 1], dtype=unique.dtype)):
            raise ValueError(f"training candidates must contain both classes; observed {unique}")
        self.pipeline.fit(feature_frame.loc[:, FEATURE_COLUMNS], labels)
        return self

    def predict_scores(self, feature_frame: pd.DataFrame) -> pd.DataFrame:
        probabilities = self.pipeline.predict_proba(feature_frame.loc[:, FEATURE_COLUMNS])[:, 1]
        return pd.DataFrame(
            {
                "source1_entity_id": feature_frame["source1_entity_id"].astype(str),
                "candidate_entity_id": feature_frame["candidate_entity_id"].astype(str),
                "score": probabilities,
            }
        )
