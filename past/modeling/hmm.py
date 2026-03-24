from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Dict

import numpy as np  # type: ignore
import pandas as pd  # type: ignore

from hmmlearn.hmm import GaussianHMM  # type: ignore


@dataclass
class HMMConfig:
    n_components: int
    covariance_type: str = "diag"
    n_iter: int = 200
    tol: float = 1e-3
    random_state: int = 42


class HMMModel:
    def __init__(self, config: HMMConfig) -> None:
        self.config = config
        self.model = GaussianHMM(
            n_components=config.n_components,
            covariance_type=config.covariance_type,
            n_iter=config.n_iter,
            tol=config.tol,
            random_state=config.random_state,
            verbose=False,
        )

    def fit(self, X: np.ndarray, lengths: Optional[List[int]] = None) -> None:
        """
        Fit the HMM on concatenated sequences.

        Parameters
        ----------
        X : np.ndarray
            Shape (n_samples, n_features). Concatenated windows
            from many matches.
        lengths : list[int], optional
            Length of each match sequence.
        """
        self.model.fit(X, lengths)

    def predict_states(self, X: np.ndarray, lengths: Optional[List[int]] = None) -> np.ndarray:
        """
        Decode the most likely hidden state sequence (Viterbi).
        """
        return self.model.predict(X, lengths)

    def predict_state_proba(self, X: np.ndarray, lengths: Optional[List[int]] = None) -> np.ndarray:
        """
        Posterior probability of each state at each timestep.
        """
        return self.model.predict_proba(X, lengths)

    def score(self, X: np.ndarray, lengths: Optional[List[int]] = None) -> float:
        """
        Log-likelihood of the data under the model.
        Useful for model selection (AIC/BIC).
        """
        return self.model.score(X, lengths)

    @property
    def transition_matrix(self) -> np.ndarray:
        return self.model.transmat_

    @property
    def start_probabilities(self) -> np.ndarray:
        return self.model.startprob_

    @property
    def state_means(self) -> np.ndarray:
        return self.model.means_

    @property
    def state_covariances(self) -> np.ndarray:
        return self.model.covars_