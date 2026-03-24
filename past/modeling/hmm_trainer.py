"""
HMM Trainer

Trains Gaussian Hidden Markov Model on behavioral features.

Learns state transitions and emission distributions for tilt detection.
"""

import numpy as np
import pandas as pd
import pickle
from pathlib import Path
from typing import Optional, Tuple, List

from .hmm import HMMModel, HMMConfig


class HMMTrainer:
    """
    Trains and persists Hidden Markov Models for tilt detection.
    
    Learns state dynamics from feature sequences and provides
    model serialization.
    """
    
    def __init__(self, n_states: int = 3, covariance_type: str = "diag",
                 n_iter: int = 200, random_state: int = 42):
        """
        Initialize HMM trainer.
        
        Args:
            n_states (int): Number of behavioral states (typically 3)
            covariance_type (str): 'diag', 'tied', or 'spherical'
            n_iter (int): Max EM iterations for training
            random_state (int): Random seed for reproducibility
        """
        self.config = HMMConfig(
            n_components=n_states,
            covariance_type=covariance_type,
            n_iter=n_iter,
            random_state=random_state
        )
        self.model = HMMModel(self.config)
        self.is_fitted = False
        self.feature_names = None
    
    def train(self, X: pd.DataFrame, lengths: Optional[List[int]] = None) -> None:
        """
        Train HMM on feature sequences.
        
        Args:
            X (pd.DataFrame): Training features [n_total_frames, n_features]
            lengths (list[int]): Sequence lengths if concatenated
        """
        X_values = X.values if isinstance(X, pd.DataFrame) else X
        self.feature_names = X.columns.tolist() if isinstance(X, pd.DataFrame) else None
        
        self.model.fit(X_values, lengths)
        self.is_fitted = True
    
    def predict_states(self, X: pd.DataFrame) -> np.ndarray:
        """
        Predict hidden states for data.
        
        Args:
            X (pd.DataFrame): Features [n_frames, n_features]
            
        Returns:
            np.ndarray: Predicted state IDs [n_frames]
        """
        if not self.is_fitted:
            raise ValueError("Model not trained. Call train() first.")
        
        X_values = X.values if isinstance(X, pd.DataFrame) else X
        return self.model.predict_states(X_values)
    
    def predict_log_proba(self, X: pd.DataFrame) -> np.ndarray:
        """
        Get log probability of observations under model.
        
        Args:
            X (pd.DataFrame): Features
            
        Returns:
            np.ndarray: Log probability per frame
        """
        if not self.is_fitted:
            raise ValueError("Model not trained. Call train() first.")
        
        X_values = X.values if isinstance(X, pd.DataFrame) else X
        return self.model.score(X_values)
    
    def save_model(self, filepath: str) -> None:
        """
        Save trained model to disk.
        
        Args:
            filepath (str): Path to save model
        """
        if not self.is_fitted:
            raise ValueError("Model not trained. Call train() first.")
        
        model_data = {
            'config': self.config,
            'model': self.model,
            'feature_names': self.feature_names
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
    
    @classmethod
    def load_model(cls, filepath: str) -> 'HMMTrainer':
        """
        Load trained model from disk.
        
        Args:
            filepath (str): Path to saved model
            
        Returns:
            HMMTrainer: Loaded trainer instance
        """
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        
        trainer = cls(
            n_states=model_data['config'].n_components,
            covariance_type=model_data['config'].covariance_type,
            n_iter=model_data['config'].n_iter,
            random_state=model_data['config'].random_state
        )
        trainer.model = model_data['model']
        trainer.is_fitted = True
        trainer.feature_names = model_data['feature_names']
        
        return trainer
    
    @property
    def transition_matrix(self) -> np.ndarray:
        return self.model.transition_matrix
    
    @property
    def state_means(self) -> np.ndarray:
        return self.model.state_means
