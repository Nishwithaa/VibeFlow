"""Recommendation engine for VibeFlow.

Components
----------
* Popularity baseline    - uses the 'popularity' column; falls back to catalog order if absent.
* Content-based          - cosine similarity between a recency-weighted "session vibe"
                           vector and every candidate track (z-scored audio features).
* Sequential transitions - skip-gram style transition counts learned ONLY from the
                           sessions passed to fit(), with additive smoothing.
* Hybrid                 - transparent weighted sum of min-max normalized components.
                           Unavailable components are dropped and weights renormalized.
* Explanations           - built from the numbers actually computed for each track.

Leakage note: the model learns nothing from session data except via fit().
For evaluation, call fit() with training sessions only.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

from src.preprocessing import FeatureMatrix, build_feature_matrix, build_session_sequences

# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #
METHODS = ("hybrid", "content", "sequence", "popularity")
METHOD_LABELS = {
    "hybrid": "Hybrid (content + sequence + popularity)",
    "content": "Content-based (audio similarity)",
    "sequence": "Sequential (observed transitions)",
    "popularity": "Popularity baseline",
}

DEFAULT_WEIGHTS = {"content": 0.50, "sequence": 0.35, "popularity": 0.15}
RECENCY_DECAY = 0.75        # each step back in the session is worth 75% of the next one
SEQ_WINDOW = 3              # a track "follows" the previous 3 tracks in training sessions
LAG_DECAY = 0.5             # a track 2 steps later counts half as much as the next track
SMOOTHING_ALPHA = 0.1       # additive smoothing for sparse transition counts
TIE_BREAK_EPS = 1e-3        # content similarity breaks ties for the pure 'sequence' method
MAX_K = 50
MAX_DENSE_TRACKS = 20_000   # transition matrix is dense: n x n floats
REFERENCE_SAMPLE_SIZE = 500
REFERENCE_SEED = 42
LOW_THRESHOLD, HIGH_THRESHOLD = 0.35, 0.65   # on the 0-1 catalog-relative scale
MAX_DESCRIPTION_PHRASES = 4

FEATURE_PHRASES = {
    "valence": ("darker / lower valence", "brighter / higher valence"),
    "energy": ("low energy", "high energy"),
    "danceability": ("less danceable", "highly danceable"),
    "tempo": ("slower tempo", "faster tempo"),
    "acousticness": ("more electronic or produced", "acoustic-leaning"),
    "instrumentalness": ("vocal-led", "more instrumental"),
    "speechiness": ("little spoken word", "speech-heavy"),
    "loudness": ("quieter", "louder"),
}


# --------------------------------------------------------------------------- #
# Result containers
# --------------------------------------------------------------------------- #
@dataclass
class VibeProfile:
    """Computed audio-feature summary of a listening sequence (not a verdict on emotion)."""
    track_ids: list[str]
    track_names: list[str]
    features: list[str]
    profile: dict[str, float]            # recency-weighted mean, 0-1 catalog-relative scale
    trend: dict[str, float] | None       # later half minus earlier half (scaled units)
    contrast_ratio: float | None         # mean consecutive distance / mean random-pair distance
    contrast_label: str
    description: str
    recency_weights: list[float]
    vector_scaled: np.ndarray = field(repr=False)
    vector_standardized: np.ndarray = field(repr=False)


@dataclass
class Recommendation:
    rank: int
    track_id: str
    track_name: str
    artist: str
    genre: str
    popularity: float | None
    score: float                         # 0-1, relative to the candidates in this request
    components: dict[str, float]         # normalized component scores (0-1)
    tag: str                             # "Familiar pick", "Exploratory pick" or ""
    explanation: str


@dataclass
class RecommendationResult:
    recommendations: list[Recommendation]
    method_requested: str
    method_used: str
    weights_used: dict[str, float] | None
    notes: list[str]
    vibe: VibeProfile | None
    ignored_track_ids: list[str]
    n_candidates: int

    @property
    def is_empty(self) -> bool:
        return len(self.recommendations) == 0

    def to_dataframe(self) -> pd.DataFrame:
        base_cols = ["rank", "track_id", "track_name", "artist", "genre", "popularity", "score"]
        if not self.recommendations:
            return pd.DataFrame(columns=base_cols + ["tag", "explanation"])
        rows = []
        for r in self.recommendations:
            row: dict[str, Any] = {
                "rank": r.rank, "track_id": r.track_id, "track_name": r.track_name,
                "artist": r.artist, "genre": r.genre, "popularity": r.popularity,
                "score": round(r.score, 4),
            }
            for name, value in r.components.items():
                row[f"{name}_score"] = round(value, 4)
            row["tag"] = r.tag
            row["explanation"] = r.explanation
            rows.append(row)
        return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# Small helpers
# --------------------------------------------------------------------------- #
def recency_weights(n: int, decay: float = RECENCY_DECAY) -> np.ndarray:
    """Weights that sum to 1, giving the most recent item the largest weight."""
    if n <= 0:
        return np.array([], dtype=float)
    w = decay ** np.arange(n - 1, -1, -1, dtype=float)
    return w / w.sum()


def _minmax(values: np.ndarray) -> np.ndarray:
    """Scale to 0-1. A constant array carries no ranking information and becomes all zeros."""
    values = np.asarray(values, dtype=float)
    if values.size == 0:
        return values
    lo, hi = float(values.min()), float(values.max())
    if hi - lo <= 1e-12:
        return np.zeros_like(values)
    return (values - lo) / (hi - lo)


def validate_weights(weights: Mapping[str, float] | None) -> tuple[dict[str, float], list[str]]:
    """Return usable hybrid weights plus notes. Missing keys count as 0. Falls back to defaults."""
    if weights is None:
        return dict(DEFAULT_WEIGHTS), []
    notes: list[str] = []
    try:
        cleaned = {key: float(weights.get(key, 0.0)) for key in DEFAULT_WEIGHTS}
        unknown = [str(k) for k in weights if k not in DEFAULT_WEIGHTS]
    except (AttributeError, TypeError, ValueError):
        return dict(DEFAULT_WEIGHTS), ["Weights could not be read; default weights were used."]
    if unknown:
        notes.append(f"Ignored unknown weight name(s): {', '.join(unknown)}.")
    if any((not np.isfinite(v)) or v < 0 for v in cleaned.values()) or sum(cleaned.values()) <= 0:
        notes.append("Weights must be non-negative and not all zero; default weights were used.")
        return dict(DEFAULT_WEIGHTS), notes
    return cleaned, notes


# --------------------------------------------------------------------------- #
# Recommender
# --------------------------------------------------------------------------- #
class VibeFlowRecommender:
    """Content + sequence + popularity recommender over a track catalog."""

    def __init__(self, tracks: pd.DataFrame, feature_matrix: FeatureMatrix | None = None) -> None:
        if tracks is None or len(tracks) == 0:
            raise ValueError("The track catalog is empty; cannot build a recommender.")
        missing = [c for c in ("track_id", "track_name", "artist") if c not in tracks.columns]
        if missing:
            raise ValueError(f"Track catalog is missing required column(s): {', '.join(missing)}")
        tracks = tracks.reset_index(drop=True)
        if tracks["track_id"].duplicated().any():
            raise ValueError("Track catalog contains duplicate track_id values.")
        if len(tracks) > MAX_DENSE_TRACKS:
            raise ValueError(
                f"Catalog has {len(tracks)} tracks; this MVP's dense transition matrix supports "
                f"at most {MAX_DENSE_TRACKS}.")

        fm = feature_matrix if feature_matrix is not None else build_feature_matrix(tracks)
        if fm.track_ids != tracks["track_id"].astype(str).tolist():
            raise ValueError("feature_matrix rows do not match the track catalog order.")

        self.tracks = tracks
        self.fm = fm
        self.n = len(tracks)
        self.names = tracks["track_name"].astype(str).tolist()
        self.artists = tracks["artist"].astype(str).tolist()
        self.genres = (tracks["genre"].astype(str).tolist() if "genre" in tracks.columns
                       else ["unknown"] * self.n)

        if "popularity" in tracks.columns:
            pop = pd.to_numeric(tracks["popularity"], errors="coerce").to_numpy(dtype=float)
        else:
            pop = np.full(self.n, np.nan)
        self._pop_raw = pop
        self.popularity_available = bool(np.isfinite(pop).any())
        if self.popularity_available:
            self._pop_filled = np.where(np.isfinite(pop), pop, np.nanmedian(pop))
        else:
            self._pop_filled = np.zeros(self.n)

        self.is_fitted = False
        self.n_training_sessions = 0
        self._weighted: np.ndarray | None = None   # lag-weighted follow counts
        self._raw_counts: np.ndarray | None = None  # unweighted follow counts (for explanations)
        self._ref_distance = self._reference_distance()

    # ------------------------------------------------------------------ fit
    def fit(self, train_sequences: Mapping[str, Sequence[str]] | Iterable[Sequence[str]] | None) -> "VibeFlowRecommender":
        """Learn transition counts from ordered training sessions (and nothing else)."""
        if train_sequences is None:
            sequences: list[Sequence[str]] = []
        elif hasattr(train_sequences, "values"):
            sequences = list(train_sequences.values())  # type: ignore[union-attr]
        else:
            try:
                sequences = list(train_sequences)
            except TypeError as exc:
                raise TypeError("train_sequences must be a dict of sequences or an iterable of sequences.") from exc

        weighted = np.zeros((self.n, self.n), dtype=float)
        raw = np.zeros((self.n, self.n), dtype=float)
        used_sessions = 0
        for seq in sequences:
            idx = [self.fm.index[str(t)] for t in seq if str(t) in self.fm.index]
            if len(idx) < 2:
                continue
            used_sessions += 1
            for pos in range(1, len(idx)):
                target = idx[pos]
                for lag in range(1, SEQ_WINDOW + 1):
                    src_pos = pos - lag
                    if src_pos < 0:
                        break
                    src = idx[src_pos]
                    if src == target:
                        continue
                    weighted[src, target] += LAG_DECAY ** (lag - 1)
                    raw[src, target] += 1.0

        self._weighted, self._raw_counts = weighted, raw
        self.n_training_sessions = used_sessions
        self.is_fitted = used_sessions > 0 and raw.sum() > 0
        return self

    @property
    def sequence_available(self) -> bool:
        return bool(self.is_fitted)

    def top_transitions(self, n: int = 10) -> pd.DataFrame:
        """Most frequent observed (track -> later track) pairs in the training sessions."""
        cols = ["from_track", "to_track", "count"]
        if not self.is_fitted or self._raw_counts is None:
            return pd.DataFrame(columns=cols)
        flat = self._raw_counts.ravel()
        order = np.argsort(-flat, kind="stable")[: max(int(n), 1)]
        rows = []
        for pos in order:
            if flat[pos] <= 0:
                break
            src, dst = divmod(int(pos), self.n)
            rows.append({"from_track": self.names[src], "to_track": self.names[dst], "count": int(flat[pos])})
        return pd.DataFrame(rows, columns=cols)

    # ------------------------------------------------------------- the vibe
    def estimate_vibe(self, context_track_ids: Iterable[str] | None) -> VibeProfile | None:
        """Public helper: vibe of a listening sequence, or None if no valid tracks."""
        valid, _ = self._resolve_context(context_track_ids, [])
        return self._build_vibe(valid) if valid else None

    def _reference_distance(self) -> float:
        """Mean distance between random catalog pairs (z-scored space); a yardstick for contrast."""
        x = self.fm.standardized
        if len(x) < 2:
            return 0.0
        if len(x) > REFERENCE_SAMPLE_SIZE:
            rng = np.random.default_rng(REFERENCE_SEED)
            x = x[rng.choice(len(x), REFERENCE_SAMPLE_SIZE, replace=False)]
        dist = np.sqrt(((x[:, None, :] - x[None, :, :]) ** 2).sum(axis=-1))
        return float(dist[np.triu_indices(len(x), k=1)].mean())

    def _build_vibe(self, valid_ids: list[str]) -> VibeProfile:
        idx = [self.fm.index[t] for t in valid_ids]
        w = recency_weights(len(idx))
        scaled_rows = self.fm.scaled[idx]
        std_rows = self.fm.standardized[idx]
        vec_scaled = w @ scaled_rows
        vec_std = w @ std_rows
        features = list(self.fm.features)
        profile = {f: float(v) for f, v in zip(features, vec_scaled)}

        trend: dict[str, float] | None = None
        contrast_ratio: float | None = None
        if len(idx) >= 2:
            half = len(idx) // 2
            delta = scaled_rows[len(idx) - half:].mean(axis=0) - scaled_rows[:half].mean(axis=0)
            trend = {f: float(v) for f, v in zip(features, delta)}
            consecutive = float(np.linalg.norm(np.diff(std_rows, axis=0), axis=1).mean())
            if self._ref_distance > 0:
                contrast_ratio = consecutive / self._ref_distance
        if contrast_ratio is None:
            label = "Single track: no transitions to compare yet."
        elif contrast_ratio < 0.8:
            label = "Smooth: consecutive tracks are more alike than random catalog pairs."
        elif contrast_ratio <= 1.2:
            label = "Mixed: consecutive tracks differ about as much as random catalog pairs."
        else:
            label = "Contrasting: consecutive tracks differ more than random catalog pairs."

        return VibeProfile(
            track_ids=list(valid_ids), track_names=[self.names[i] for i in idx], features=features,
            profile=profile, trend=trend, contrast_ratio=contrast_ratio, contrast_label=label,
            description=self._describe(profile), recency_weights=[float(x) for x in w],
            vector_scaled=vec_scaled, vector_standardized=vec_std)

    @staticmethod
    def _describe(profile: dict[str, float]) -> str:
        phrases: list[tuple[float, str]] = []
        for feat, value in profile.items():
            if feat not in FEATURE_PHRASES:
                continue
            low, high = FEATURE_PHRASES[feat]
            if value >= HIGH_THRESHOLD:
                phrases.append((value - 0.5, high))
            elif value <= LOW_THRESHOLD:
                phrases.append((0.5 - value, low))
        phrases.sort(key=lambda p: -p[0])
        chosen = [text for _, text in phrases[:MAX_DESCRIPTION_PHRASES]]
        body = ", ".join(chosen) if chosen else "mid-range on most features"
        return (f"Relative to this catalog: {body}. "
                "This is a summary of audio features, not a measurement of emotion.")

    # ------------------------------------------------------------ recommend
    def recommend(self, context_track_ids: Iterable[str] | None = None, k: int = 10,
                  method: str = "hybrid", weights: Mapping[str, float] | None = None,
                  diversity: float = 0.0, exclude_track_ids: Iterable[str] | None = None,
                  explain: bool = True) -> RecommendationResult:
        """Rank next-track candidates for a listening sequence (oldest -> newest).

        Tracks already in the sequence (and any in exclude_track_ids) are never returned.
        Problems produce notes and fallbacks rather than exceptions.
        """
        notes: list[str] = []
        k = self._sanitize_k(k, notes)
        requested = method
        if method not in METHODS:
            notes.append(f"Unknown method '{method}'; using 'hybrid'.")
            method = "hybrid"
        diversity = self._sanitize_diversity(diversity, notes)
        weights_valid, weight_notes = validate_weights(weights)
        if method == "hybrid":
            notes.extend(weight_notes)

        context, ignored = self._resolve_context(context_track_ids, notes)
        vibe = self._build_vibe(context) if context else None
        ctx_idx = [self.fm.index[t] for t in context]

        method_used = method
        if vibe is None and method != "popularity":
            notes.append("No valid tracks in the listening sequence, so there is no session to personalize "
                         "from. Showing the popularity baseline instead.")
            method_used = "popularity"
        if len(context) == 1 and method_used in ("sequence", "hybrid"):
            notes.append("Only one track is in the session: the vibe equals that track and only one "
                         "transition source is available.")

        # Candidate generation: everything except tracks already played / explicitly excluded.
        mask = np.ones(self.n, dtype=bool)
        mask[ctx_idx] = False
        if exclude_track_ids is not None:
            try:
                for t in exclude_track_ids:
                    if str(t) in self.fm.index:
                        mask[self.fm.index[str(t)]] = False
            except TypeError:
                notes.append("exclude_track_ids could not be read and was ignored.")
        cand = np.flatnonzero(mask)
        if cand.size == 0:
            notes.append("No candidate tracks remain after excluding tracks already in the session.")
            return RecommendationResult([], requested, method_used, None, notes, vibe, ignored, 0)
        if cand.size < k:
            notes.append(f"Only {cand.size} candidate track(s) are available; fewer than {k} are returned.")

        # ---- Component scores (raw) ----
        content_raw = None
        if vibe is not None:
            content_raw = cosine_similarity(vibe.vector_standardized.reshape(1, -1),
                                            self.fm.standardized[cand]).ravel()
        seq = self._sequence_scores(ctx_idx, cand) if vibe is not None else None

        norm: dict[str, np.ndarray] = {}
        if content_raw is not None:
            norm["content"] = _minmax(content_raw)
        if seq is not None:
            norm["sequence"] = _minmax(seq[0])
        if self.popularity_available:
            norm["popularity"] = _minmax(self._pop_filled[cand])

        # ---- Final score by method ----
        weights_used: dict[str, float] | None = None
        if method_used == "sequence" and seq is None:
            self._note_sequence_unavailable(notes, "Falling back to content-based ranking.")
            method_used = "content"

        if method_used == "popularity":
            if self.popularity_available:
                score = norm["popularity"]
            else:
                notes.append("No popularity data is available; the baseline follows catalog order.")
                score = 1.0 - cand / max(self.n - 1, 1)
        elif method_used == "content":
            score = norm["content"]
        elif method_used == "sequence":
            score = _minmax(norm["sequence"] + TIE_BREAK_EPS * norm["content"])
            if int((seq[2].sum(axis=0) > 0).sum()) < k:  # type: ignore[index]
                notes.append("Fewer candidates than requested have an observed transition; the remaining "
                             "slots are ordered by audio similarity.")
        else:  # hybrid
            available = list(norm.keys())
            for comp, w in weights_valid.items():
                if w > 0 and comp not in available:
                    if comp == "sequence":
                        self._note_sequence_unavailable(
                            notes, "Its weight was redistributed to the remaining components.")
                    elif comp == "popularity":
                        notes.append("No popularity data is available; its weight was redistributed.")
            positive = {c: weights_valid[c] for c in available if weights_valid[c] > 0}
            if not positive:
                notes.append("All requested weights were for unavailable components; default weights "
                             "were used for the available ones.")
                positive = {c: DEFAULT_WEIGHTS[c] for c in available}
            total = sum(positive.values())
            weights_used = {c: w / total for c, w in positive.items()}
            score = sum(weights_used[c] * norm[c] for c in weights_used)

        # ---- Selection (optionally diversity-aware) ----
        if diversity > 0 and cand.size > 1:
            chosen = self._mmr_select(cand, score, k, diversity)
        else:
            chosen = [int(p) for p in np.argsort(-score, kind="stable")[:k]]

        # ---- Familiar / exploratory tag ----
        familiar_cut = float(np.quantile(content_raw, 2 / 3)) if content_raw is not None else None
        if familiar_cut is not None:
            notes.append("'Familiar' means a pick's audio similarity is in the top third of all candidates; "
                         "'Exploratory' means it is lower.")

        recs: list[Recommendation] = []
        for rank, pos in enumerate(chosen, start=1):
            j = int(cand[pos])
            tag = ""
            if content_raw is not None and familiar_cut is not None:
                tag = "Familiar pick" if content_raw[pos] >= familiar_cut else "Exploratory pick"
            pop_val = float(self._pop_raw[j]) if np.isfinite(self._pop_raw[j]) else None
            recs.append(Recommendation(
                rank=rank, track_id=self.fm.track_ids[j], track_name=self.names[j],
                artist=self.artists[j], genre=self.genres[j], popularity=pop_val,
                score=float(score[pos]),
                components={c: float(arr[pos]) for c, arr in norm.items()},
                tag=tag,
                explanation=(self._explain(method_used, pos, j, vibe, content_raw, seq, norm,
                                           weights_used, tag) if explain else "")))

        return RecommendationResult(recs, requested, method_used, weights_used, notes, vibe,
                                    ignored, int(cand.size))

    # ------------------------------------------------------------- internals
    def _resolve_context(self, ids: Iterable[str] | None, notes: list[str]) -> tuple[list[str], list[str]]:
        if ids is None:
            return [], []
        if isinstance(ids, str):
            ids = [ids]
        try:
            raw = [str(t).strip() for t in ids]
        except TypeError:
            notes.append("The listening sequence could not be read and was treated as empty.")
            return [], []
        valid = [t for t in raw if t in self.fm.index]
        ignored = [t for t in raw if t not in self.fm.index]
        if ignored:
            shown = ", ".join(ignored[:5]) + (" ..." if len(ignored) > 5 else "")
            notes.append(f"Ignored {len(ignored)} unknown track ID(s): {shown}.")
        return valid, ignored

    @staticmethod
    def _sanitize_k(k: Any, notes: list[str]) -> int:
        try:
            k = int(k)
        except (TypeError, ValueError):
            notes.append("k was not a number; using 10.")
            return 10
        if k < 1:
            notes.append("k must be at least 1; using 1.")
            return 1
        if k > MAX_K:
            notes.append(f"k is capped at {MAX_K}.")
            return MAX_K
        return k

    @staticmethod
    def _sanitize_diversity(value: Any, notes: list[str]) -> float:
        try:
            value = float(value)
        except (TypeError, ValueError):
            notes.append("diversity was not a number; using 0.")
            return 0.0
        if not np.isfinite(value):
            return 0.0
        return float(min(max(value, 0.0), 1.0))

    def _note_sequence_unavailable(self, notes: list[str], consequence: str) -> None:
        if not self.is_fitted:
            notes.append("No training sessions were loaded, so the sequence component is unavailable. "
                         + consequence)
        else:
            notes.append("None of the recent tracks has an observed transition in the training sessions, "
                         "so the sequence component is unavailable. " + consequence)

    def _sequence_scores(self, ctx_idx: list[int], cand: np.ndarray):
        """Return (score, contribution matrix, raw count matrix, used context indices) or None."""
        if not self.is_fitted or self._weighted is None or self._raw_counts is None or not ctx_idx:
            return None
        used = ctx_idx[-SEQ_WINDOW:]
        w = recency_weights(len(used))
        rows = self._weighted[used]
        prob = (rows + SMOOTHING_ALPHA) / (rows.sum(axis=1, keepdims=True) + SMOOTHING_ALPHA * self.n)
        contrib = w[:, None] * prob[:, cand]
        counts = self._raw_counts[np.ix_(used, cand)]
        if counts.sum() <= 0:
            return None
        return contrib.sum(axis=0), contrib, counts, used

    def _mmr_select(self, cand: np.ndarray, score: np.ndarray, k: int, diversity: float) -> list[int]:
        """Greedy maximal-marginal-relevance selection; returns positions into `cand`."""
        pool_size = min(cand.size, max(k * 5, 50))
        pool = np.argsort(-score, kind="stable")[:pool_size]
        sim = (cosine_similarity(self.fm.standardized[cand[pool]]) + 1.0) / 2.0
        selected: list[int] = []
        remaining = list(range(len(pool)))
        while remaining and len(selected) < k:
            if not selected:
                best = remaining[0]
            else:
                redundancy = sim[np.ix_(remaining, selected)].max(axis=1)
                mmr = (1.0 - diversity) * score[pool[remaining]] - diversity * redundancy
                best = remaining[int(np.argmax(mmr))]
            selected.append(best)
            remaining.remove(best)
        return [int(pool[i]) for i in selected]

    # ---------------------------------------------------------- explanations
    def _closest_features(self, vibe: VibeProfile, j: int, n: int = 2) -> str:
        diffs = np.abs(vibe.vector_scaled - self.fm.scaled[j])
        order = np.argsort(diffs, kind="stable")[:n]
        parts = [f"{self.fm.features[i]} ({self.fm.scaled[j][i]:.2f} vs session {vibe.vector_scaled[i]:.2f})"
                 for i in order]
        return " and ".join(parts)

    def _sequence_sentence(self, seq, pos: int) -> str:
        _, contrib, counts, used = seq
        col_counts = counts[:, pos]
        if col_counts.max() <= 0:
            return "No transition from your recent tracks to this one was observed in the training sessions."
        masked = np.where(col_counts > 0, contrib[:, pos], -np.inf)
        t = int(np.argmax(masked))
        text = (f"In the training sessions, '{self.names[used[t]]}' was followed by this track "
                f"{int(col_counts[t])} time(s) within the next {SEQ_WINDOW} plays")
        others = int((col_counts > 0).sum()) - 1
        if others > 0:
            text += f" (and {others} other recent track(s) also led to it)"
        return text + "."

    def _pop_text(self, j: int) -> str:
        if np.isfinite(self._pop_raw[j]):
            return f"{self._pop_raw[j]:.0f}/100"
        return f"missing (catalog median {self._pop_filled[j]:.0f}/100 used)"

    def _explain(self, method_used: str, pos: int, j: int, vibe: VibeProfile | None,
                 content_raw, seq, norm: dict[str, np.ndarray],
                 weights_used: dict[str, float] | None, tag: str) -> str:
        parts: list[str] = []
        if method_used == "popularity":
            if self.popularity_available:
                return f"Ranked by catalog popularity ({self._pop_text(j)}); this ignores your session."
            return "No popularity data is available, so this follows catalog order; it ignores your session."

        if tag:
            parts.append(f"{tag}.")

        if method_used in ("content", "hybrid") and vibe is not None and content_raw is not None:
            parts.append(f"Audio-profile similarity {content_raw[pos]:.2f} (cosine on standardized features); "
                         f"closest on {self._closest_features(vibe, j)}.")

        if method_used == "sequence" and seq is not None:
            sentence = self._sequence_sentence(seq, pos)
            if sentence.startswith("No transition") and content_raw is not None:
                sentence += f" Ordered here by audio similarity ({content_raw[pos]:.2f})."
            parts.append(sentence)
        elif method_used == "hybrid":
            if seq is not None:
                parts.append(self._sequence_sentence(seq, pos))
            if weights_used and "popularity" in weights_used:
                parts.append(f"Popularity: {self._pop_text(j)}.")
            if weights_used:
                breakdown = " + ".join(f"{weights_used[c] * norm[c][pos]:.2f} {c}" for c in weights_used)
                parts.append(f"Score = {breakdown}.")
        return " ".join(parts)


# --------------------------------------------------------------------------- #
# Convenience builder
# --------------------------------------------------------------------------- #
def build_recommender(tracks: pd.DataFrame, sessions: pd.DataFrame | None = None,
                      train_session_ids: Iterable[str] | None = None) -> VibeFlowRecommender:
    """Create a recommender and (optionally) fit it on cleaned sessions from load_sessions().

    Pass `train_session_ids` to fit on a subset (e.g. the training split) and avoid leakage.
    """
    recommender = VibeFlowRecommender(tracks)
    if sessions is not None and len(sessions) > 0:
        sequences = build_session_sequences(sessions)
        if train_session_ids is not None:
            keep = set(train_session_ids)
            sequences = {sid: seq for sid, seq in sequences.items() if sid in keep}
        recommender.fit(sequences)
    return recommender