"""
Tilt Detector

Performs inference on trained HMM to identify tilt episodes.

Detects behavioral regime changes and onset of tilted states.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional

from .hmm import HMMModel


class TiltDetector:
    """
    Detects tilt episodes using trained HMM.
    
    Identifies sustained transitions to high-risk states and
    computes tilt onset/offset times.
    """
    
    def __init__(self, hmm_model: HMMModel, risky_states: List[int],
                 min_sustained_frames: int = 10):
        """
        Initialize tilt detector.
        
        Args:
            hmm_model: Fitted HMMModel instance
            risky_states (list): State IDs indicating tilt/risk
            min_sustained_frames (int): Min frames for tilt onset
        """
        self.model = hmm_model
        self.risky_states = set(risky_states)
        self.min_frames = min_sustained_frames
    
    def predict_states(self, X: pd.DataFrame) -> np.ndarray:
        """
        Predict hidden states using Viterbi decoding.
        
        Args:
            X (pd.DataFrame): Feature matrix
            
        Returns:
            np.ndarray: Most likely state sequence
        """
        X_values = X.values if isinstance(X, pd.DataFrame) else X
        return self.model.predict_states(X_values)
    
    def compute_state_log_proba(self, X: pd.DataFrame) -> np.ndarray:
        """
        Compute log probability of each state at each timestep.
        
        Args:
            X (pd.DataFrame): Feature matrix
            
        Returns:
            np.ndarray: State log probabilities [n_frames, n_states]
        """
        X_values = X.values if isinstance(X, pd.DataFrame) else X
        return self.model.predict_state_proba(X_values)
    
    def detect_tilt_episodes(self, states: np.ndarray) -> List[Tuple[int, int]]:
        """
        Identify tilt episodes from state sequence.
        
        Args:
            states (np.ndarray): Predicted state sequence
            
        Returns:
            list[tuple]: List of (start_frame, end_frame) tilt episodes
        """
        episodes = []
        in_tilt = False
        start_frame = 0
        
        for i, state in enumerate(states):
            if state in self.risky_states:
                if not in_tilt:
                    in_tilt = True
                    start_frame = i
            else:
                if in_tilt:
                    # Check if sustained long enough
                    duration = i - start_frame
                    if duration >= self.min_frames:
                        episodes.append((start_frame, i - 1))
                    in_tilt = False
        
        # Handle case where tilt continues to end
        if in_tilt:
            duration = len(states) - start_frame
            if duration >= self.min_frames:
                episodes.append((start_frame, len(states) - 1))
        
        return episodes
    
    def detect_tilt_onset(self, X: pd.DataFrame) -> Tuple[np.ndarray, List[Tuple[int, int]]]:
        """
        Full tilt detection pipeline.
        
        Args:
            X (pd.DataFrame): Feature matrix
            
        Returns:
            tuple: (predicted_states, tilt_episodes)
        """
        states = self.predict_states(X)
        episodes = self.detect_tilt_episodes(states)
        return states, episodes
    
    def compute_tilt_probability(self, X: pd.DataFrame) -> np.ndarray:
        """
        Compute probability of being in tilt state at each timestep.
        
        Args:
            X (pd.DataFrame): Feature matrix
            
        Returns:
            np.ndarray: Tilt probability per frame [n_frames]
        """
        state_proba = self.compute_state_log_proba(X)
        # Convert log probabilities to probabilities
        state_proba = np.exp(state_proba)
        # Sum probabilities of risky states
        tilt_proba = np.sum(state_proba[:, list(self.risky_states)], axis=1)
        return tilt_proba