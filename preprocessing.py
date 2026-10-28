"""Feature scaling and session utilities for VibeFlow."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.data_loader import AUDIO_FEATURES


@dataclass(frozen=True)
class FeatureMatrix:
    """Numeric track representation.

    scaled:       0-1 (catalog min-max), for charts and the vibe profile. Constant columns -> 0.5.
    standardized: z-scores, for similarity so every feature has comparable influence. Constant columns -> 0.
    """
    track_ids: list[str]
    features: list[str]
    scaled: np.ndarray
    standardized: np.ndarray
    index: dict[str, int]
    notes: list[str]


def build_feature_matrix(tracks: pd.DataFrame) -> FeatureMatrix:
    """Impute missing values (column median) and build scaled/standardized matrices."""
    notes: list[str] = []
    features: list[str] = []
    for feat in AUDIO_FEATURES:
        if feat not in tracks.columns:
            continue
        if tracks[feat].notna().sum() == 0:
            notes.append(f"Feature '{feat}' has no valid values and was excluded.")
            continue
        features.append(feat)
    if not features:
        raise ValueError("No usable audio features are available.")

    values = tracks[features].astype(float).copy()
    for feat in features:
        n_missing = int(values[feat].isna().sum())
        if n_missing:
            values[feat] = values[feat].fillna(values[feat].median())
            notes.append(f"Imputed {n_missing} missing '{feat}' value(s) with the column median.")

    arr = values.to_numpy()
    lo, hi = arr.min(axis=0), arr.max(axis=0)
    span = hi - lo
    mean, std = arr.mean(axis=0), arr.std(axis=0)
    for feat, s_span in zip(features, span):
        if s_span == 0:
            notes.append(f"Feature '{feat}' is constant across the catalog and carries no information.")

    scaled = np.where(span > 0, (arr - lo) / np.where(span > 0, span, 1.0), 0.5)
    standardized = np.where(std > 0, (arr - mean) / np.where(std > 0, std, 1.0), 0.0)

    track_ids = tracks["track_id"].astype(str).tolist()
    return FeatureMatrix(track_ids=track_ids, features=features, scaled=scaled,
                         standardized=standardized,
                         index={t: i for i, t in enumerate(track_ids)}, notes=notes)


def session_order(sessions: pd.DataFrame) -> list[str]:
    """Session IDs in chronological order (by first timestamp; falls back to ID order)."""
    firsts = sessions.groupby("session_id")["timestamp"].min().reset_index()
    firsts = firsts.sort_values(["timestamp", "session_id"], na_position="last")
    return firsts["session_id"].tolist()


def build_session_sequences(sessions: pd.DataFrame) -> dict[str, list[str]]:
    """Map session_id -> ordered list of track_ids, with sessions in chronological order."""
    seqs = {sid: g.sort_values("position")["track_id"].tolist()
            for sid, g in sessions.groupby("session_id")}
    return {sid: seqs[sid] for sid in session_order(sessions)}