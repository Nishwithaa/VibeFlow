"""Safe CSV loading and validation for VibeFlow.

Functions return (cleaned DataFrame, DataQualityReport). Problems are reported,
never silently hidden. Only unusable files raise DataValidationError.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TRACKS_PATH = PROJECT_ROOT / "data" / "demo_tracks.csv"
DEFAULT_SESSIONS_PATH = PROJECT_ROOT / "data" / "demo_sessions.csv"

REQUIRED_TRACK_COLUMNS = ["track_id", "track_name", "artist"]
REQUIRED_SESSION_COLUMNS = ["session_id", "user_id", "track_id", "position"]

AUDIO_FEATURES = ["valence", "energy", "danceability", "tempo",
                  "acousticness", "instrumentalness", "speechiness", "loudness"]

# Plausibility ranges. Values outside are treated as invalid (set to missing and reported).
VALID_RANGES = {
    "valence": (0.0, 1.0), "energy": (0.0, 1.0), "danceability": (0.0, 1.0),
    "acousticness": (0.0, 1.0), "instrumentalness": (0.0, 1.0), "speechiness": (0.0, 1.0),
    "tempo": (30.0, 250.0), "loudness": (-60.0, 5.0),
}
POPULARITY_RANGE = (0.0, 100.0)
MIN_AUDIO_FEATURES = 2


class DataValidationError(Exception):
    """Raised when a data file is missing or cannot be used at all."""


@dataclass
class DataQualityReport:
    """Collects human-readable data quality messages."""
    issues: list[str] = field(default_factory=list)

    def add(self, message: str) -> None:
        self.issues.append(message)

    def extend(self, other: "DataQualityReport") -> None:
        self.issues.extend(other.issues)

    @property
    def has_issues(self) -> bool:
        return bool(self.issues)


def _read_csv(path: str | Path, required: list[str], id_columns: list[str]) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        raise DataValidationError(
            f"File not found: {path}. If this is the demo data, run: python scripts/generate_demo_data.py")
    try:
        df = pd.read_csv(path, dtype={c: str for c in id_columns})
    except (pd.errors.EmptyDataError, pd.errors.ParserError) as exc:
        raise DataValidationError(f"Could not parse {path.name}: {exc}") from exc
    df.columns = [str(c).strip() for c in df.columns]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise DataValidationError(f"{path.name} is missing required column(s): {', '.join(missing)}")
    if df.empty:
        raise DataValidationError(f"{path.name} contains no rows.")
    return df


def _clean_text(series: pd.Series) -> pd.Series:
    """Strip whitespace; blank strings become missing."""
    s = series.astype("string").str.strip()
    return s.mask(s == "", pd.NA)


def load_tracks(path: str | Path = DEFAULT_TRACKS_PATH) -> tuple[pd.DataFrame, DataQualityReport]:
    """Load and validate the track catalog."""
    report = DataQualityReport()
    df = _read_csv(path, REQUIRED_TRACK_COLUMNS, ["track_id"])

    df["track_id"] = _clean_text(df["track_id"])
    bad_id = df["track_id"].isna()
    if bad_id.any():
        report.add(f"Dropped {int(bad_id.sum())} track row(s) with a missing track_id.")
        df = df[~bad_id].copy()

    duplicated = df["track_id"].duplicated(keep="first")
    if duplicated.any():
        report.add(f"Dropped {int(duplicated.sum())} duplicate track_id row(s) (kept the first occurrence).")
        df = df[~duplicated].copy()
    df["track_id"] = df["track_id"].astype(str)

    names = _clean_text(df["track_name"])
    if names.isna().any():
        report.add(f"{int(names.isna().sum())} track(s) had no name; their track_id is used as the name.")
    df["track_name"] = names.fillna(df["track_id"]).astype(str)

    artists = _clean_text(df["artist"])
    if artists.isna().any():
        report.add(f"{int(artists.isna().sum())} track(s) had no artist; labelled 'Unknown artist'.")
    df["artist"] = artists.fillna("Unknown artist").astype(str)

    present = [f for f in AUDIO_FEATURES if f in df.columns]
    absent = [f for f in AUDIO_FEATURES if f not in df.columns]
    if len(present) < MIN_AUDIO_FEATURES:
        raise DataValidationError(
            f"Need at least {MIN_AUDIO_FEATURES} audio feature columns; found: {present or 'none'}.")
    if absent:
        report.add(f"Audio feature column(s) not present and will not be used: {', '.join(absent)}.")

    for feat in present:
        raw_missing = int(df[feat].isna().sum())
        numeric = pd.to_numeric(df[feat], errors="coerce")
        not_numeric = int(numeric.isna().sum()) - raw_missing
        if not_numeric > 0:
            report.add(f"'{feat}': {not_numeric} non-numeric value(s) were set to missing.")
        lo, hi = VALID_RANGES[feat]
        out_of_range = (numeric < lo) | (numeric > hi)
        if out_of_range.any():
            report.add(f"'{feat}': {int(out_of_range.sum())} value(s) outside [{lo}, {hi}] were set to missing.")
        df[feat] = numeric.mask(out_of_range).astype(float)
        remaining = int(df[feat].isna().sum())
        if remaining:
            report.add(f"'{feat}': {remaining} missing value(s) (will be median-imputed for modelling).")

    if "popularity" in df.columns:
        pop = pd.to_numeric(df["popularity"], errors="coerce")
        lo, hi = POPULARITY_RANGE
        bad_pop = (pop < lo) | (pop > hi)
        if bad_pop.any():
            report.add(f"'popularity': {int(bad_pop.sum())} value(s) outside [{lo}, {hi}] were set to missing.")
        df["popularity"] = pop.mask(bad_pop).astype(float)
    else:
        report.add("No 'popularity' column; the popularity baseline will fall back to catalog order.")

    if "genre" in df.columns:
        df["genre"] = _clean_text(df["genre"]).fillna("unknown").astype(str)
    else:
        df["genre"] = "unknown"
    if "source_type" in df.columns:
        df["source_type"] = _clean_text(df["source_type"]).fillna("unknown").astype(str)
    else:
        df["source_type"] = "unknown"

    return df.reset_index(drop=True), report


def load_sessions(path: str | Path = DEFAULT_SESSIONS_PATH,
                  valid_track_ids: set[str] | None = None) -> tuple[pd.DataFrame, DataQualityReport]:
    """Load and validate ordered listening sessions.

    Rows referencing tracks not in `valid_track_ids` are dropped (and reported).
    Positions are renumbered 1..n within each session after cleaning.
    """
    report = DataQualityReport()
    df = _read_csv(path, REQUIRED_SESSION_COLUMNS, ["session_id", "user_id", "track_id"])

    for col in ["session_id", "user_id", "track_id"]:
        df[col] = _clean_text(df[col])
    missing_key = df["session_id"].isna() | df["track_id"].isna()
    if missing_key.any():
        report.add(f"Dropped {int(missing_key.sum())} session row(s) with a missing session_id or track_id.")
        df = df[~missing_key].copy()
    if df["user_id"].isna().any():
        report.add(f"{int(df['user_id'].isna().sum())} row(s) had no user_id; labelled 'unknown'.")
    df["user_id"] = df["user_id"].fillna("unknown").astype(str)
    df["session_id"] = df["session_id"].astype(str)
    df["track_id"] = df["track_id"].astype(str)

    pos = pd.to_numeric(df["position"], errors="coerce")
    invalid_pos = pos.isna() | (pos < 1) | (pos != pos.round())
    if invalid_pos.any():
        report.add(f"Dropped {int(invalid_pos.sum())} row(s) with an invalid position (must be an integer >= 1).")
    df = df[~invalid_pos].copy()
    df["position"] = pos[~invalid_pos].astype(int)

    if valid_track_ids is not None:
        unknown = ~df["track_id"].isin(valid_track_ids)
        if unknown.any():
            report.add(f"Dropped {int(unknown.sum())} session row(s) referencing tracks not in the catalog.")
            df = df[~unknown].copy()

    dup_pos = df.duplicated(subset=["session_id", "position"], keep="first")
    if dup_pos.any():
        report.add(f"Dropped {int(dup_pos.sum())} row(s) with a repeated (session_id, position) pair.")
        df = df[~dup_pos].copy()

    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce", format="ISO8601")
        n_bad_ts = int(df["timestamp"].isna().sum())
        if n_bad_ts:
            report.add(f"{n_bad_ts} row(s) have a missing/unparseable timestamp.")
    else:
        df["timestamp"] = pd.NaT
        report.add("No 'timestamp' column; sessions will be ordered by session_id instead.")

    if "source_type" in df.columns:
        df["source_type"] = _clean_text(df["source_type"]).fillna("unknown").astype(str)
    else:
        df["source_type"] = "unknown"

    df = df.sort_values(["session_id", "position"]).reset_index(drop=True)
    renumbered = df.groupby("session_id").cumcount() + 1
    if (renumbered != df["position"]).any():
        report.add("Positions were renumbered 1..n within sessions after cleaning (order preserved).")
        df["position"] = renumbered

    sizes = df.groupby("session_id").size()
    short = int((sizes < 2).sum())
    if short:
        report.add(f"{short} session(s) have fewer than 2 tracks and cannot provide transitions.")
    if df.empty:
        raise DataValidationError("No valid session rows remain after cleaning.")

    return df, report