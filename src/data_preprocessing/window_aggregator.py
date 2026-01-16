import json
import logging
import os
import time
import requests # type: ignore
import pandas as pd # type: ignore 

from typing import Dict, Iterable, List, Optional

class WindowAggregator:
    def __init__(self, events_df: pd.DataFrame, window_size: int = 600000):
        self.events_df = events_df
        self.window_size = window_size  # in milliseconds
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)

    def aggregate_events(self) -> pd.DataFrame:
        if self.events_df.empty:
            self.logger.warning("The events DataFrame is empty. Returning an empty DataFrame.")
            return pd.DataFrame()

        min_timestamp = self.events_df['timestamp'].min()
        max_timestamp = self.events_df['timestamp'].max()
        bins = range(min_timestamp, max_timestamp + self.window_size, self.window_size)
        
        self.events_df['time_window'] = pd.cut(self.events_df['timestamp'], bins=bins, right=False)

        aggregated_data = self.events_df.groupby(['time_window', 'event_type']).size().unstack(fill_value=0).reset_index()
        
        return aggregated_data
    
    def save_aggregated_data(self, output_path: str) -> None:
        aggregated_df = self.aggregate_events()
        aggregated_df.to_csv(output_path, index=False)
        self.logger.info(f"Aggregated data saved to {output_path}")