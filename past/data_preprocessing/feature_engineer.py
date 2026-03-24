from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np  # type: ignore
import pandas as pd  # type: ignore


class TimelineParser:
    def __init__(self, timelines: List[Dict]):
        self.timelines = timelines
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)

    def parse_timelines(self) -> pd.DataFrame:
        all_events = []
        for timeline in self.timelines:
            match_id = timeline.get("metadata", {}).get("matchId")
            for frame in timeline.get("info", {}).get("frames", []):
                timestamp = frame.get("timestamp", 0)
                for event in frame.get("events", []):
                    event_data = {
                        "match_id": match_id,
                        "timestamp": timestamp,
                        "event_type": event.get("type"),
                        "participant_id": event.get("participantId"),
                        "position": event.get("position"),
                        "killer_id": event.get("killerId"),
                        "victim_id": event.get("victimId"),
                        "assisting_participant_ids": event.get("assistingParticipantIds"),
                        "item_id": event.get("itemId"),
                        "skill_slot": event.get("skillSlot"),
                        "level_up_type": event.get("levelUpType"),
                        "ward_type": event.get("wardType"),
                    }
                    all_events.append(event_data)
        return pd.DataFrame(all_events)

    def parse_frames(self) -> pd.DataFrame:
        """
        Parse per-participant frame stats for windowed delta features.
        These stats enable relative and volatility features without using outcome targets.
        """
        all_frames = []
        for timeline in self.timelines:
            match_id = timeline.get("metadata", {}).get("matchId")
            for frame in timeline.get("info", {}).get("frames", []):
                timestamp = frame.get("timestamp", 0)
                participant_frames = frame.get("participantFrames", {})
                for participant_id, pdata in participant_frames.items():
                    damage_stats = pdata.get("damageStats", {}) or {}
                    position = pdata.get("position", {}) or {}
                    all_frames.append(
                        {
                            "match_id": match_id,
                            "timestamp": timestamp,
                            "participant_id": int(participant_id),
                            "total_gold": pdata.get("totalGold"),
                            "current_gold": pdata.get("currentGold"),
                            "level": pdata.get("level"),
                            "xp": pdata.get("xp"),
                            "minions_killed": pdata.get("minionsKilled"),
                            "jungle_minions_killed": pdata.get("jungleMinionsKilled"),
                            "damage_done_to_champions": damage_stats.get("totalDamageDoneToChampions"),
                            "damage_taken": damage_stats.get("totalDamageTaken"),
                            "position_x": position.get("x"),
                            "position_y": position.get("y"),
                        }
                    )
        return pd.DataFrame(all_frames)


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

        min_timestamp = self.events_df["timestamp"].min()
        max_timestamp = self.events_df["timestamp"].max()
        bins = range(min_timestamp, max_timestamp + self.window_size, self.window_size)

        self.events_df["time_window"] = pd.cut(self.events_df["timestamp"], bins=bins, right=False)

        aggregated_data = (
            self.events_df.groupby(["time_window", "event_type"]).size().unstack(fill_value=0).reset_index()
        )

        return aggregated_data

    def save_aggregated_data(self, output_path: str) -> None:
        aggregated_df = self.aggregate_events()
        aggregated_df.to_csv(output_path, index=False)
        self.logger.info("Aggregated data saved to %s", output_path)


@dataclass
class FeatureConfig:
    """
    Configuration for windowed feature generation.
    """
    window_sizes_ms: List[int]
    river_sum_threshold: int = 15000
    deep_enemy_sum_threshold: int = 17000
    teammate_radius: int = 2000
    rolling_windows: int = 3
    objective_after_death_ms: int = 30000


class FeatureEngineer:
    """
    Build HMM emission sequences from timeline events and frames.
    This module converts raw Riot timeline data into windowed, behavioral
    observation vectors suitable for unsupervised sequence models (e.g., HMMs).
    Features emphasize relative change, volatility, coordination, and risk,
    not match outcome or absolute skill.
    """

    def __init__(self, config: Optional[FeatureConfig] = None) -> None:
        self.config = config or FeatureConfig(window_sizes_ms=[10000, 30000, 60000])

    def build_emission_matrix(
        self,
        events_df: pd.DataFrame,
        frames_df: pd.DataFrame,
        match_id: str,
        participant_to_team: Optional[Dict[int, int]] = None,
        lane_opponents: Optional[Dict[int, int]] = None,
        participant_baselines: Optional[Dict[int, Dict[str, float]]] = None,
    ) -> pd.DataFrame:
        """
        Convert raw timeline events/frames into HMM-ready emission observations.
        Each row corresponds to a single participant in a single fixed time window.
        Downstream modeling code should group rows by (match_id, participant_id)
        and treat the ordered windows as an observation sequence.
        """
        if events_df is None:
            events_df = pd.DataFrame()
        if frames_df is None:
            frames_df = pd.DataFrame()

        participant_baselines = participant_baselines or {}

        feature_frames = []
        for window_size_ms in self.config.window_sizes_ms:
            window_features = self._build_features_for_window(
                events_df=events_df,
                frames_df=frames_df,
                window_size_ms=window_size_ms,
                match_id=match_id,
                participant_to_team=participant_to_team or {},
                lane_opponents=lane_opponents or {},
                participant_baselines=participant_baselines,
            )
            feature_frames.append(window_features)

        if not feature_frames:
            return pd.DataFrame()

        return pd.concat(feature_frames, ignore_index=True)

    def _build_features_for_window(
        self,
        events_df: pd.DataFrame,
        frames_df: pd.DataFrame,
        window_size_ms: int,
        match_id: str,
        participant_to_team: Dict[int, int],
        lane_opponents: Dict[int, int],
        participant_baselines: Dict[int, Dict[str, float]],
    ) -> pd.DataFrame:
        window_events = self._add_window_index(events_df, window_size_ms)
        window_frames = self._add_window_index(frames_df, window_size_ms)

        participant_ids = self._collect_participants(window_events, window_frames)
        if not participant_ids:
            return pd.DataFrame()

        base_index = self._build_base_index(
            participant_ids=participant_ids,
            window_starts=self._collect_window_starts(window_events, window_frames),
        )

        event_features = self._event_features(window_events, participant_to_team)
        frame_features = self._frame_features(window_frames)

        features = base_index.merge(event_features, on=["participant_id", "window_start_ms"], how="left")
        features = features.merge(frame_features, on=["participant_id", "window_start_ms"], how="left")

        count_columns = [
            "kills",
            "deaths",
            "assists",
            "solo_kills",
            "solo_deaths",
            "grouped_deaths",
            "repeated_deaths_same_opponent",
            "wards_placed",
            "wards_killed",
            "items_purchased",
            "team_kills",
        ]
        for col in count_columns:
            if col not in features:
                features[col] = 0
        features[count_columns] = features[count_columns].fillna(0)

        features["window_size_ms"] = window_size_ms

        features = self._relative_features(features, lane_opponents)
        features = self._interaction_history_features(features, window_size_ms)
        features = self._synergy_features(features, participant_to_team, participant_baselines, window_size_ms)
        features = self._risk_features(window_events, features, participant_to_team)
        features = self._directional_features(features)
        features = self._position_cluster_features(window_frames, features, participant_to_team)
        features = self._objective_after_death_features(window_events, features, participant_to_team)

        features["match_id"] = match_id

        numeric_cols = [col for col in features.columns if col not in {"match_id"}]
        features[numeric_cols] = features[numeric_cols].apply(pd.to_numeric, errors="ignore")
        # NOTE: Returned features are emissions only.
        # No labels, targets, or outcome variables should be added here.
        return features

    @staticmethod
    def _add_window_index(df: pd.DataFrame, window_size_ms: int) -> pd.DataFrame:
        if df.empty:
            return df.copy()
        df = df.copy()
        df["window_start_ms"] = (df["timestamp"] // window_size_ms) * window_size_ms
        return df

    @staticmethod
    def _collect_participants(events_df: pd.DataFrame, frames_df: pd.DataFrame) -> List[int]:
        participant_ids = set()
        if not events_df.empty and "participant_id" in events_df:
            participant_ids.update(events_df["participant_id"].dropna().astype(int).tolist())
        if not frames_df.empty and "participant_id" in frames_df:
            participant_ids.update(frames_df["participant_id"].dropna().astype(int).tolist())
        return sorted(participant_ids)

    @staticmethod
    def _collect_window_starts(events_df: pd.DataFrame, frames_df: pd.DataFrame) -> List[int]:
        starts = set()
        if not events_df.empty and "window_start_ms" in events_df:
            starts.update(events_df["window_start_ms"].dropna().astype(int).tolist())
        if not frames_df.empty and "window_start_ms" in frames_df:
            starts.update(frames_df["window_start_ms"].dropna().astype(int).tolist())
        return sorted(starts)

    @staticmethod
    def _build_base_index(participant_ids: List[int], window_starts: List[int]) -> pd.DataFrame:
        index = pd.MultiIndex.from_product(
            [participant_ids, window_starts], names=["participant_id", "window_start_ms"]
        )
        return index.to_frame(index=False)

    def _event_features(self, events_df: pd.DataFrame, participant_to_team: Dict[int, int]) -> pd.DataFrame:
        """
        Event-based interaction signals: kills, deaths, assists, and repeated opponent deaths.
        These capture short-horizon relational pressure and coordination breakdowns.
        """
        if events_df.empty:
            return pd.DataFrame(columns=["participant_id", "window_start_ms"])

        events_df = events_df.copy()
        events_df["assisting_participant_ids"] = events_df["assisting_participant_ids"].apply(
            lambda ids: ids if isinstance(ids, list) else []
        )

        kill_events = events_df[events_df["event_type"] == "CHAMPION_KILL"].copy()
        kills = (
            kill_events.groupby(["window_start_ms", "killer_id"])
            .size()
            .rename("kills")
            .reset_index()
            .rename(columns={"killer_id": "participant_id"})
        )
        deaths = (
            kill_events.groupby(["window_start_ms", "victim_id"])
            .size()
            .rename("deaths")
            .reset_index()
            .rename(columns={"victim_id": "participant_id"})
        )

        assists_rows = []
        for _, row in kill_events.iterrows():
            for assister_id in row["assisting_participant_ids"]:
                assists_rows.append(
                    {
                        "participant_id": assister_id,
                        "window_start_ms": row["window_start_ms"],
                        "assists": 1,
                    }
                )
        assists = (
            pd.DataFrame(assists_rows)
            .groupby(["participant_id", "window_start_ms"])
            .sum()
            .reset_index()
            if assists_rows
            else pd.DataFrame(columns=["participant_id", "window_start_ms", "assists"])
        )

        solo_kills = kill_events[kill_events["assisting_participant_ids"].apply(len) == 0]
        solo_kills = (
            solo_kills.groupby(["window_start_ms", "killer_id"])
            .size()
            .rename("solo_kills")
            .reset_index()
            .rename(columns={"killer_id": "participant_id"})
        )

        solo_deaths = kill_events[kill_events["assisting_participant_ids"].apply(len) == 0]
        solo_deaths = (
            solo_deaths.groupby(["window_start_ms", "victim_id"])
            .size()
            .rename("solo_deaths")
            .reset_index()
            .rename(columns={"victim_id": "participant_id"})
        )
        grouped_deaths = kill_events[kill_events["assisting_participant_ids"].apply(len) > 0]
        grouped_deaths = (
            grouped_deaths.groupby(["window_start_ms", "victim_id"])
            .size()
            .rename("grouped_deaths")
            .reset_index()
            .rename(columns={"victim_id": "participant_id"})
        )

        repeated_deaths = []
        for victim_id, victim_events in kill_events.sort_values("timestamp").groupby("victim_id"):
            prev_killer = None
            for _, event in victim_events.iterrows():
                killer_id = event.get("killer_id")
                if killer_id is not None and killer_id == prev_killer:
                    repeated_deaths.append(
                        {
                            "participant_id": victim_id,
                            "window_start_ms": event["window_start_ms"],
                            "repeated_deaths_same_opponent": 1,
                        }
                    )
                prev_killer = killer_id
        repeated_deaths = (
            pd.DataFrame(repeated_deaths)
            .groupby(["participant_id", "window_start_ms"])
            .sum()
            .reset_index()
            if repeated_deaths
            else pd.DataFrame(columns=["participant_id", "window_start_ms", "repeated_deaths_same_opponent"])
        )

        ward_placements = self._count_events(events_df, "WARD_PLACED", "participant_id", "wards_placed")
        ward_kills = self._count_events(events_df, "WARD_KILL", "participant_id", "wards_killed")
        item_purchases = self._count_events(events_df, "ITEM_PURCHASED", "participant_id", "items_purchased")

        event_features = (
            kills.merge(deaths, on=["participant_id", "window_start_ms"], how="outer")
            .merge(assists, on=["participant_id", "window_start_ms"], how="outer")
            .merge(solo_kills, on=["participant_id", "window_start_ms"], how="outer")
            .merge(solo_deaths, on=["participant_id", "window_start_ms"], how="outer")
            .merge(grouped_deaths, on=["participant_id", "window_start_ms"], how="outer")
            .merge(repeated_deaths, on=["participant_id", "window_start_ms"], how="outer")
            .merge(ward_placements, on=["participant_id", "window_start_ms"], how="outer")
            .merge(ward_kills, on=["participant_id", "window_start_ms"], how="outer")
            .merge(item_purchases, on=["participant_id", "window_start_ms"], how="outer")
        )
        event_features = event_features.fillna(0)

        event_features["team_id"] = event_features["participant_id"].map(participant_to_team)
        team_kills = (
            kills.assign(team_id=lambda df: df["participant_id"].map(participant_to_team))
            .groupby(["window_start_ms", "team_id"])["kills"]
            .sum()
            .rename("team_kills")
            .reset_index()
        )
        event_features = event_features.merge(team_kills, on=["window_start_ms", "team_id"], how="left")
        event_features["team_kills"] = event_features["team_kills"].fillna(0)
        return event_features.drop(columns=["team_id"])

    @staticmethod
    def _count_events(
        events_df: pd.DataFrame, event_type: str, participant_col: str, output_name: str
    ) -> pd.DataFrame:
        subset = events_df[events_df["event_type"] == event_type]
        if subset.empty:
            return pd.DataFrame(columns=["participant_id", "window_start_ms", output_name])
        return (
            subset.groupby(["window_start_ms", participant_col])
            .size()
            .rename(output_name)
            .reset_index()
            .rename(columns={participant_col: "participant_id"})
        )

    def _frame_features(self, frames_df: pd.DataFrame) -> pd.DataFrame:
        # All frame-based features are computed as within-window deltas only.
        # This prevents monotonic time leakage and keeps observations local.
        """
        Frame delta features for relative strength and volatility without outcome leakage.
        """
        if frames_df.empty:
            return pd.DataFrame(columns=["participant_id", "window_start_ms"])

        frames_df = frames_df.copy()
        frames_df = frames_df.sort_values(["participant_id", "timestamp"])

        def window_delta(group: pd.DataFrame, col: str) -> pd.Series:
            if col not in group or group[col].isna().all():
                return pd.Series({"delta_" + col: np.nan})
            return pd.Series({"delta_" + col: group[col].iloc[-1] - group[col].iloc[0]})

        delta_frames = []
        for (participant_id, window_start_ms), group in frames_df.groupby(
            ["participant_id", "window_start_ms"]
        ):
            row = {
                "participant_id": participant_id,
                "window_start_ms": window_start_ms,
            }
            row.update(window_delta(group, "total_gold"))
            row.update(window_delta(group, "minions_killed"))
            row.update(window_delta(group, "jungle_minions_killed"))
            row.update(window_delta(group, "xp"))
            row.update(window_delta(group, "damage_done_to_champions"))
            row.update(window_delta(group, "damage_taken"))
            delta_frames.append(row)

        delta_df = pd.DataFrame(delta_frames)
        if delta_df.empty:
            return pd.DataFrame(columns=["participant_id", "window_start_ms"])

        delta_df["damage_trade_efficiency"] = delta_df["delta_damage_done_to_champions"] / (
            delta_df["delta_damage_taken"].replace(0, np.nan)
        )
        return delta_df

    def _relative_features(self, features: pd.DataFrame, lane_opponents: Dict[int, int]) -> pd.DataFrame:
        """
        Opponent-relative deltas emphasize relational superiority over absolute strength.
        """
        if not lane_opponents:
            return features

        opponent_lookup = pd.Series(lane_opponents, name="opponent_id")
        features = features.copy()
        features["opponent_id"] = features["participant_id"].map(opponent_lookup)

        cols = ["delta_total_gold", "delta_minions_killed", "damage_trade_efficiency"]
        for col in cols:
            if col not in features:
                features[col] = np.nan

        opponent_cols = features[["participant_id", "window_start_ms"] + cols].rename(
            columns={c: f"{c}_opp" for c in cols}
        )
        opponent_cols = opponent_cols.rename(columns={"participant_id": "opponent_id"})
        features = features.merge(opponent_cols, on=["opponent_id", "window_start_ms"], how="left")

        features["gold_delta_vs_opponent"] = features["delta_total_gold"] - features["delta_total_gold_opp"]
        features["cs_delta_vs_opponent"] = features["delta_minions_killed"] - features["delta_minions_killed_opp"]
        features["damage_efficiency_delta_vs_opponent"] = (
            features["damage_trade_efficiency"] - features["damage_trade_efficiency_opp"]
        )
        return features.drop(columns=["opponent_id"])

    def _interaction_history_features(self, features: pd.DataFrame, window_size_ms: int) -> pd.DataFrame:
        # Rolling aggregates below intentionally compress recent history.
        # They act as short-horizon memory proxies so that the hidden state
        # can summarize longer-term context while preserving Markov structure.
        """
        Rolling summaries compress recent behavior into HMM-friendly emissions.
        """
        features = features.sort_values(["participant_id", "window_start_ms"]).copy()
        window_minutes = max(window_size_ms / 60000.0, 1e-6)

        features["actions_per_min"] = (
            features[["kills", "deaths", "assists", "wards_placed", "wards_killed", "items_purchased"]]
            .fillna(0)
            .sum(axis=1)
            / window_minutes
        )
        features["rolling_action_var"] = (
            features.groupby("participant_id")["actions_per_min"]
            .rolling(self.config.rolling_windows, min_periods=1)
            .var()
            .reset_index(level=0, drop=True)
        )

        for col in ["delta_total_gold", "delta_minions_killed"]:
            if col not in features:
                features[col] = np.nan
            features[f"rolling_{col}_volatility"] = (
                features.groupby("participant_id")[col]
                .rolling(self.config.rolling_windows, min_periods=1)
                .std()
                .reset_index(level=0, drop=True)
            )

        features["aggression_index"] = (
            features[["kills", "assists"]].fillna(0).sum(axis=1)
            + features.get("delta_damage_done_to_champions", pd.Series(0, index=features.index)).fillna(0)
            / 1000.0
        )
        features["aggression_change"] = (
            features.groupby("participant_id")["aggression_index"].diff().fillna(0)
        )

        features["death_in_prev_window"] = (
            features.groupby("participant_id")["deaths"].shift(1).fillna(0) > 0
        )
        features["aggression_change_after_death"] = np.where(
            features["death_in_prev_window"], features["aggression_change"], 0
        )
        last_death_ts = (
            features.groupby("participant_id")["window_start_ms"]
            .apply(lambda s: s.where(features.loc[s.index, "deaths"] > 0).ffill())
            .reset_index(level=0, drop=True)
        )
        time_since_death_min = (features["window_start_ms"] - last_death_ts).fillna(0) / 60000.0
        features["death_rate_after_last_death"] = features["deaths"].fillna(0) / time_since_death_min.replace(
            0, np.nan
        )
        return features

    def _synergy_features(
        self,
        features: pd.DataFrame,
        participant_to_team: Dict[int, int],
        participant_baselines: Dict[int, Dict[str, float]],
        window_size_ms: int,
    ) -> pd.DataFrame:
        """
        Coordination features track kill participation and assist deviations from baseline.
        """
        features = features.copy()
        window_minutes = max(window_size_ms / 60000.0, 1e-6)

        features["kill_participation"] = (
            (features["kills"].fillna(0) + features["assists"].fillna(0))
            / features["team_kills"].replace(0, np.nan)
        )
        kp_baseline = features["participant_id"].map(
            lambda pid: participant_baselines.get(pid, {}).get("kill_participation")
        )
        kp_baseline = kp_baseline.fillna(
            features.groupby("participant_id")["kill_participation"].transform("mean")
        )
        features["kill_participation_dev"] = features["kill_participation"] - kp_baseline

        features["solo_death_fraction"] = features["solo_deaths"] / features["deaths"].replace(0, np.nan)
        features["grouped_death_fraction"] = features["grouped_deaths"] / features["deaths"].replace(0, np.nan)

        assist_rate = features["assists"].fillna(0) / window_minutes
        assist_baseline = features["participant_id"].map(
            lambda pid: participant_baselines.get(pid, {}).get("assist_rate")
        )
        assist_baseline = assist_baseline.fillna(assist_rate.groupby(features["participant_id"]).transform("mean"))
        features["assist_rate_change"] = assist_rate - assist_baseline
        return features

    def _risk_features(
        self,
        events_df: pd.DataFrame,
        features: pd.DataFrame,
        participant_to_team: Dict[int, int],
    ) -> pd.DataFrame:
        """
        Overextension indicators from positional deaths and enemy-side kills.
        """
        if events_df.empty:
            features["deaths_past_river"] = 0
            features["deaths_deep_enemy"] = 0
            features["enemy_side_kills"] = 0
            features["enemy_side_kills_per_min"] = 0
            return features

        deaths = events_df[events_df["event_type"] == "CHAMPION_KILL"].copy()
        deaths = deaths.rename(columns={"victim_id": "participant_id"})
        deaths["team_id"] = deaths["participant_id"].map(participant_to_team)

        deaths["x"] = deaths["position"].apply(lambda p: p.get("x") if isinstance(p, dict) else np.nan)
        deaths["y"] = deaths["position"].apply(lambda p: p.get("y") if isinstance(p, dict) else np.nan)
        deaths["sum_xy"] = deaths[["x", "y"]].sum(axis=1)

        def past_river(row: pd.Series) -> bool:
            if pd.isna(row["sum_xy"]) or pd.isna(row["team_id"]):
                return False
            if row["team_id"] == 100:
                return row["sum_xy"] > self.config.river_sum_threshold
            return row["sum_xy"] < self.config.river_sum_threshold

        def deep_enemy(row: pd.Series) -> bool:
            if pd.isna(row["sum_xy"]) or pd.isna(row["team_id"]):
                return False
            if row["team_id"] == 100:
                return row["sum_xy"] > self.config.deep_enemy_sum_threshold
            return row["sum_xy"] < (2 * self.config.river_sum_threshold - self.config.deep_enemy_sum_threshold)

        deaths["past_river"] = deaths.apply(past_river, axis=1)
        deaths["deep_enemy"] = deaths.apply(deep_enemy, axis=1)

        deaths_past_river = (
            deaths[deaths["past_river"]]
            .groupby(["participant_id", "window_start_ms"])
            .size()
            .rename("deaths_past_river")
            .reset_index()
        )
        deaths_deep_enemy = (
            deaths[deaths["deep_enemy"]]
            .groupby(["participant_id", "window_start_ms"])
            .size()
            .rename("deaths_deep_enemy")
            .reset_index()
        )

        kills = events_df[events_df["event_type"] == "CHAMPION_KILL"].copy()
        kills = kills.rename(columns={"killer_id": "participant_id"})
        kills["team_id"] = kills["participant_id"].map(participant_to_team)
        kills["x"] = kills["position"].apply(lambda p: p.get("x") if isinstance(p, dict) else np.nan)
        kills["y"] = kills["position"].apply(lambda p: p.get("y") if isinstance(p, dict) else np.nan)
        kills["sum_xy"] = kills[["x", "y"]].sum(axis=1)
        kills["enemy_side"] = kills.apply(past_river, axis=1)
        enemy_side_kills = (
            kills[kills["enemy_side"]]
            .groupby(["participant_id", "window_start_ms"])
            .size()
            .rename("enemy_side_kills")
            .reset_index()
        )

        features = features.merge(deaths_past_river, on=["participant_id", "window_start_ms"], how="left")
        features = features.merge(deaths_deep_enemy, on=["participant_id", "window_start_ms"], how="left")
        features = features.merge(enemy_side_kills, on=["participant_id", "window_start_ms"], how="left")
        features[["deaths_past_river", "deaths_deep_enemy", "enemy_side_kills"]] = features[
            ["deaths_past_river", "deaths_deep_enemy", "enemy_side_kills"]
        ].fillna(0)
        window_minutes = max(features["window_size_ms"].iloc[0] / 60000.0, 1e-6)
        features["enemy_side_kills_per_min"] = features["enemy_side_kills"] / window_minutes
        return features

    def _directional_features(self, features: pd.DataFrame) -> pd.DataFrame:
        """
        Directional context features encode pressure orientation without implying success.
        """
        features = features.copy()
        if "gold_delta_vs_opponent" in features:
            features["relative_pressure_sign"] = np.sign(
                features["gold_delta_vs_opponent"].fillna(0)
            )
        else:
            features["relative_pressure_sign"] = 0.0
        return features

    def _position_cluster_features(
        self,
        frames_df: pd.DataFrame,
        features: pd.DataFrame,
        participant_to_team: Dict[int, int],
    ) -> pd.DataFrame:
        """
        Team proximity signals reflect grouping or isolation over time.
        """
        if frames_df.empty or "position_x" not in frames_df:
            features["teammates_nearby"] = np.nan
            features["distance_to_team_centroid"] = np.nan
            features["solo_engage_no_teammates"] = np.nan
            return features

        latest_positions = (
            frames_df.sort_values(["participant_id", "timestamp"])
            .groupby(["participant_id", "window_start_ms"])
            .tail(1)
        )
        latest_positions["team_id"] = latest_positions["participant_id"].map(participant_to_team)
        latest_positions = latest_positions.dropna(subset=["position_x", "position_y", "team_id"])

        centroid = (
            latest_positions.groupby(["window_start_ms", "team_id"])[["position_x", "position_y"]]
            .mean()
            .rename(columns={"position_x": "centroid_x", "position_y": "centroid_y"})
            .reset_index()
        )
        positions = latest_positions.merge(centroid, on=["window_start_ms", "team_id"], how="left")
        positions["distance_to_team_centroid"] = np.sqrt(
            (positions["position_x"] - positions["centroid_x"]) ** 2
            + (positions["position_y"] - positions["centroid_y"]) ** 2
        )

        teammate_counts = []
        for (window_start_ms, team_id), group in positions.groupby(["window_start_ms", "team_id"]):
            coords = group[["position_x", "position_y"]].to_numpy()
            pids = group["participant_id"].to_numpy()
            for idx, pid in enumerate(pids):
                distances = np.sqrt(((coords - coords[idx]) ** 2).sum(axis=1))
                nearby = int((distances <= self.config.teammate_radius).sum()) - 1
                teammate_counts.append(
                    {
                        "participant_id": pid,
                        "window_start_ms": window_start_ms,
                        "teammates_nearby": nearby,
                    }
                )
        teammate_counts = pd.DataFrame(teammate_counts)

        features = features.merge(
            positions[["participant_id", "window_start_ms", "distance_to_team_centroid"]],
            on=["participant_id", "window_start_ms"],
            how="left",
        )
        features = features.merge(teammate_counts, on=["participant_id", "window_start_ms"], how="left")
        features["solo_engage_no_teammates"] = np.where(
            (features["solo_kills"].fillna(0) > 0) & (features["teammates_nearby"].fillna(0) <= 0),
            features["solo_kills"].fillna(0),
            0,
        )
        return features

    def _objective_after_death_features(
        self,
        events_df: pd.DataFrame,
        features: pd.DataFrame,
        participant_to_team: Dict[int, int],
    ) -> pd.DataFrame:
        """
        Objective contests after deaths capture risky re-engagement patterns.
        """
        if events_df.empty:
            features["team_objective_after_death"] = 0
            return features

        deaths = events_df[events_df["event_type"] == "CHAMPION_KILL"].copy()
        deaths = deaths.rename(columns={"victim_id": "participant_id"})
        deaths["team_id"] = deaths["participant_id"].map(participant_to_team)

        objectives = events_df[
            events_df["event_type"].isin(["ELITE_MONSTER_KILL", "BUILDING_KILL", "TOWER_KILL"])
        ].copy()
        objectives["team_id"] = objectives["killer_id"].map(participant_to_team)

        results = []
        for _, death in deaths.iterrows():
            window_start_ms = death["window_start_ms"]
            end_ts = death["timestamp"] + self.config.objective_after_death_ms
            team_id = death["team_id"]
            if pd.isna(team_id):
                continue
            window_objectives = objectives[
                (objectives["timestamp"] >= death["timestamp"])
                & (objectives["timestamp"] <= end_ts)
                & (objectives["team_id"] == team_id)
            ]
            if not window_objectives.empty:
                results.append(
                    {
                        "participant_id": death["participant_id"],
                        "window_start_ms": window_start_ms,
                        "team_objective_after_death": 1,
                    }
                )
        results_df = (
            pd.DataFrame(results)
            .groupby(["participant_id", "window_start_ms"])
            .sum()
            .reset_index()
            if results
            else pd.DataFrame(columns=["participant_id", "window_start_ms", "team_objective_after_death"])
        )
        features = features.merge(results_df, on=["participant_id", "window_start_ms"], how="left")
        features["team_objective_after_death"] = features["team_objective_after_death"].fillna(0)
        return features
