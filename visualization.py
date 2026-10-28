"""Plotly charts for VibeFlow (dark theme, violet / blue / magenta accents).

Every function returns a plotly Figure and never raises on empty or missing data;
it returns a labelled placeholder figure instead, so the app can always render it.
All charts are drawn from computed data passed in. Nothing here is hardcoded.
"""
from __future__ import annotations

import math
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from src.data_loader import AUDIO_FEATURES
from src.preprocessing import FeatureMatrix
from src.recommender import SEQ_WINDOW, RecommendationResult, VibeProfile

# --------------------------------------------------------------------------- #
# Theme
# --------------------------------------------------------------------------- #
VIOLET = "#8B5CF6"
BLUE = "#3B82F6"
MAGENTA = "#D946EF"
BG = "#0E0B1A"
SURFACE = "#171229"
TEXT = "#E5E7EB"
MUTED = "#9CA3AF"
GRID = "rgba(255,255,255,0.08)"
FONT_FAMILY = "Inter, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif"

PALETTE = [VIOLET, BLUE, MAGENTA, "#22D3EE", "#A78BFA", "#60A5FA",
           "#F472B6", "#818CF8", "#C084FC", "#38BDF8", "#E879F9", "#6366F1"]
COMPONENT_COLORS = {"content": VIOLET, "sequence": BLUE, "popularity": MAGENTA}

DEFAULT_RANDOM_PAIRS = 3000
RANDOM_SEED = 42


def _label(feature: str) -> str:
    return feature.replace("_", " ").capitalize()


def _short(text: str, n: int = 26) -> str:
    text = str(text)
    return text if len(text) <= n else text[: n - 1] + "…"


def apply_theme(fig: go.Figure, title: str | None = None, height: int | None = None) -> go.Figure:
    """Apply the shared dark styling to a figure."""
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family=FONT_FAMILY, color=TEXT, size=13),
        margin=dict(l=40, r=20, t=60 if title else 30, b=40),
        legend=dict(bgcolor="rgba(0,0,0,0)"),
        hoverlabel=dict(bgcolor=SURFACE, font_color=TEXT),
    )
    if title:
        fig.update_layout(title=dict(text=title, x=0.01, font=dict(size=16)))
    if height:
        fig.update_layout(height=height)
    fig.update_xaxes(gridcolor=GRID, zerolinecolor=GRID)
    fig.update_yaxes(gridcolor=GRID, zerolinecolor=GRID)
    return fig


def empty_figure(message: str, height: int = 300) -> go.Figure:
    """Placeholder figure with a centered message (used for empty / error states)."""
    fig = go.Figure()
    fig.add_annotation(text=message, x=0.5, y=0.5, xref="paper", yref="paper",
                       showarrow=False, font=dict(size=14, color=MUTED))
    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False)
    return apply_theme(fig, height=height)


def _usable_features(tracks: pd.DataFrame | None, features: Iterable[str] | None = None) -> list[str]:
    if tracks is None or len(tracks) == 0:
        return []
    wanted = list(features) if features is not None else AUDIO_FEATURES
    return [f for f in wanted if f in tracks.columns and tracks[f].notna().any()]


# --------------------------------------------------------------------------- #
# Current Vibe charts
# --------------------------------------------------------------------------- #
def vibe_radar(vibe: VibeProfile | None, height: int = 380) -> go.Figure:
    """Radar chart of the recency-weighted session profile (0-1, relative to this catalog)."""
    if vibe is None or not vibe.features:
        return empty_figure("Add tracks to see the session's audio profile.", height)
    labels = [_label(f) for f in vibe.features]
    values = [float(vibe.profile[f]) for f in vibe.features]
    fig = go.Figure(go.Scatterpolar(
        r=values + values[:1], theta=labels + labels[:1], fill="toself",
        line=dict(color=MAGENTA, width=2), fillcolor="rgba(139,92,246,0.35)",
        hovertemplate="%{theta}: %{r:.2f}<extra></extra>"))
    fig.update_layout(
        showlegend=False,
        polar=dict(bgcolor="rgba(0,0,0,0)",
                   radialaxis=dict(range=[0, 1], gridcolor=GRID, linecolor=GRID,
                                   tickfont=dict(size=10, color=MUTED)),
                   angularaxis=dict(gridcolor=GRID, linecolor=GRID)))
    return apply_theme(fig, "Session audio profile (0–1, relative to catalog)", height)


def vibe_profile_bars(vibe: VibeProfile | None, height: int = 380) -> go.Figure:
    """Horizontal bars of the same profile; easier to read exact values than the radar."""
    if vibe is None or not vibe.features:
        return empty_figure("Add tracks to see the session's audio profile.", height)
    labels = [_label(f) for f in vibe.features]
    values = [float(vibe.profile[f]) for f in vibe.features]
    fig = go.Figure(go.Bar(
        x=values, y=labels, orientation="h", marker_color=VIOLET,
        text=[f"{v:.2f}" for v in values], textposition="outside", cliponaxis=False,
        hovertemplate="%{y}: %{x:.2f}<extra></extra>"))
    fig.update_xaxes(range=[0, 1.1], title="Catalog-relative value (0–1)")
    fig.update_yaxes(autorange="reversed")
    return apply_theme(fig, "Feature values for this session", height)


def vibe_trend_chart(vibe: VibeProfile | None, height: int = 340) -> go.Figure:
    """Later half of the session minus earlier half, per feature (scaled units)."""
    if vibe is None or vibe.trend is None:
        return empty_figure("Add at least 2 tracks to see how the session changes.", height)
    feats = list(vibe.trend.keys())
    deltas = [float(vibe.trend[f]) for f in feats]
    fig = go.Figure(go.Bar(
        x=[_label(f) for f in feats], y=deltas,
        marker_color=[MAGENTA if d >= 0 else BLUE for d in deltas],
        hovertemplate="%{x}: %{y:+.2f}<extra></extra>"))
    bound = max(0.1, max(abs(d) for d in deltas) * 1.15)
    fig.update_yaxes(range=[-bound, bound], title="Later half − earlier half")
    return apply_theme(fig, "How the session shifts (pink = rising, blue = falling)", height)


def sequence_feature_chart(vibe: VibeProfile | None, fm: FeatureMatrix | None,
                           height: int = 380) -> go.Figure:
    """Feature values (0-1) for each track in listening order. Click legend entries to toggle."""
    if vibe is None or fm is None or not vibe.track_ids:
        return empty_figure("Add tracks to see features across the sequence.", height)
    idx = [fm.index[t] for t in vibe.track_ids if t in fm.index]
    if not idx:
        return empty_figure("None of the selected tracks have features.", height)
    x_labels = [f"{i + 1}. {_short(vibe.track_names[i], 18)}" for i in range(len(idx))]
    shown_by_default = {"valence", "energy", "danceability"}
    fig = go.Figure()
    for col, feat in enumerate(fm.features):
        fig.add_trace(go.Scatter(
            x=x_labels, y=fm.scaled[idx, col], mode="lines+markers", name=_label(feat),
            line=dict(color=PALETTE[col % len(PALETTE)], width=2),
            visible=True if feat in shown_by_default else "legendonly",
            hovertemplate=f"{_label(feat)}: " + "%{y:.2f}<extra></extra>"))
    fig.update_yaxes(range=[0, 1.05], title="Catalog-relative value (0–1)")
    return apply_theme(fig, "Features across the listening sequence", height)


# --------------------------------------------------------------------------- #
# Recommendation charts
# --------------------------------------------------------------------------- #
def recommendation_score_chart(result: RecommendationResult | None, height: int | None = None) -> go.Figure:
    """Bar chart of ranking scores. For hybrid results, bars are stacked by weighted component."""
    recs = result.recommendations if result is not None else []
    if not recs:
        return empty_figure("No recommendations to display.")
    labels = [f"{r.rank}. {_short(r.track_name)}" for r in recs]
    hover = [f"{r.track_name} — {r.artist}" for r in recs]
    fig = go.Figure()
    if result is not None and result.weights_used:
        for comp, weight in result.weights_used.items():
            values = [weight * r.components.get(comp, 0.0) for r in recs]
            fig.add_bar(y=labels, x=values, orientation="h", name=comp.capitalize(),
                        marker_color=COMPONENT_COLORS.get(comp, BLUE), customdata=hover,
                        hovertemplate="%{customdata}<br>" + comp + " contribution: %{x:.3f}<extra></extra>")
        fig.update_layout(barmode="stack")
        xtitle = "Weighted contribution to hybrid score"
    else:
        fig.add_bar(y=labels, x=[r.score for r in recs], orientation="h", marker_color=VIOLET,
                    customdata=hover, name="Score",
                    hovertemplate="%{customdata}<br>score: %{x:.3f}<extra></extra>")
        xtitle = "Score (0–1, relative to the candidates)"
    fig.update_yaxes(autorange="reversed")
    fig.update_xaxes(title=xtitle)
    return apply_theme(fig, "How each recommendation was scored",
                       height or max(320, 34 * len(recs) + 120))


def recommendation_map(tracks: pd.DataFrame | None, result: RecommendationResult | None,
                       vibe: VibeProfile | None, x_feature: str = "valence",
                       y_feature: str = "energy", height: int = 460) -> go.Figure:
    """Catalog scatter with the session's tracks (numbered, violet) and recommendations (pink)."""
    if tracks is None or len(tracks) == 0:
        return empty_figure("No catalog loaded.", height)
    if x_feature not in tracks.columns or y_feature not in tracks.columns:
        return empty_figure(f"Features '{x_feature}' / '{y_feature}' are not in the catalog.", height)

    catalog = tracks.dropna(subset=[x_feature, y_feature])
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=catalog[x_feature], y=catalog[y_feature], mode="markers", name="Catalog",
        marker=dict(size=7, color="rgba(148,163,184,0.35)"),
        hovertext=[f"{n} — {a}" for n, a in zip(catalog["track_name"], catalog["artist"])],
        hovertemplate="%{hovertext}<br>%{x:.2f}, %{y:.2f}<extra></extra>"))

    lookup = tracks.set_index("track_id")
    if vibe is not None and vibe.track_ids:
        positions: dict[str, list[int]] = {}
        for p, tid in enumerate(vibe.track_ids, start=1):
            positions.setdefault(tid, []).append(p)
        ids = [t for t in positions if t in lookup.index
               and pd.notna(lookup.loc[t, x_feature]) and pd.notna(lookup.loc[t, y_feature])]
        if ids:
            fig.add_trace(go.Scatter(
                x=[lookup.loc[t, x_feature] for t in ids], y=[lookup.loc[t, y_feature] for t in ids],
                mode="markers+text", name="Your session",
                text=["/".join(map(str, positions[t])) for t in ids], textposition="top center",
                marker=dict(size=13, color=VIOLET, line=dict(width=1, color=TEXT)),
                hovertext=[f"{lookup.loc[t, 'track_name']} — {lookup.loc[t, 'artist']}" for t in ids],
                hovertemplate="%{hovertext}<br>%{x:.2f}, %{y:.2f}<extra></extra>"))

    if result is not None and result.recommendations:
        recs = [r for r in result.recommendations if r.track_id in lookup.index
                and pd.notna(lookup.loc[r.track_id, x_feature]) and pd.notna(lookup.loc[r.track_id, y_feature])]
        if recs:
            fig.add_trace(go.Scatter(
                x=[lookup.loc[r.track_id, x_feature] for r in recs],
                y=[lookup.loc[r.track_id, y_feature] for r in recs],
                mode="markers+text", name="Recommended", text=[str(r.rank) for r in recs],
                textposition="top center",
                marker=dict(size=13, symbol="diamond", color=MAGENTA, line=dict(width=1, color=TEXT)),
                hovertext=[f"#{r.rank} {r.track_name} — {r.artist}" for r in recs],
                hovertemplate="%{hovertext}<br>%{x:.2f}, %{y:.2f}<extra></extra>"))

    fig.update_xaxes(title=_label(x_feature))
    fig.update_yaxes(title=_label(y_feature))
    return apply_theme(fig, f"Where the session and recommendations sit: {_label(x_feature)} vs {_label(y_feature)}",
                       height)


# --------------------------------------------------------------------------- #
# Analytics charts
# --------------------------------------------------------------------------- #
def feature_distribution_chart(tracks: pd.DataFrame | None, features: Iterable[str] | None = None) -> go.Figure:
    """Grid of histograms, one per audio feature."""
    feats = _usable_features(tracks, features)
    if not feats:
        return empty_figure("No audio features available.")
    cols = min(4, len(feats))
    rows = math.ceil(len(feats) / cols)
    fig = make_subplots(rows=rows, cols=cols, subplot_titles=[_label(f) for f in feats],
                        vertical_spacing=0.16 if rows > 1 else 0.1)
    for i, feat in enumerate(feats):
        fig.add_trace(go.Histogram(x=tracks[feat].dropna(), nbinsx=20, marker_color=VIOLET,
                                   showlegend=False,
                                   hovertemplate=f"{_label(feat)}: " + "%{x}<br>tracks: %{y}<extra></extra>"),
                      row=i // cols + 1, col=i % cols + 1)
    fig.update_annotations(font=dict(size=12, color=MUTED))
    return apply_theme(fig, "Distribution of audio features", 240 * rows + 70)


def correlation_heatmap(tracks: pd.DataFrame | None, features: Iterable[str] | None = None) -> go.Figure:
    """Pearson correlations between audio features (constant columns are skipped)."""
    feats = [f for f in _usable_features(tracks, features) if tracks[f].nunique(dropna=True) > 1]
    if len(feats) < 2:
        return empty_figure("Need at least two varying audio features for a correlation chart.")
    corr = tracks[feats].corr()
    labels = [_label(f) for f in feats]
    fig = go.Figure(go.Heatmap(
        z=corr.values, x=labels, y=labels, zmin=-1, zmax=1,
        colorscale=[[0.0, BLUE], [0.5, "#1A1530"], [1.0, MAGENTA]],
        text=np.round(corr.values, 2), texttemplate="%{text}",
        hovertemplate="%{y} vs %{x}: %{z:.2f}<extra></extra>",
        colorbar=dict(title="r")))
    fig.update_yaxes(autorange="reversed")
    return apply_theme(fig, "Correlation between audio features", 480)


def feature_scatter(tracks: pd.DataFrame | None, x_feature: str = "valence", y_feature: str = "energy",
                    color_by: str | None = "genre", height: int = 480) -> go.Figure:
    """Scatter of two audio features across the catalog, optionally coloured by a column."""
    if tracks is None or len(tracks) == 0:
        return empty_figure("No catalog loaded.", height)
    if x_feature not in tracks.columns or y_feature not in tracks.columns:
        return empty_figure(f"Features '{x_feature}' / '{y_feature}' are not in the catalog.", height)
    data = tracks.dropna(subset=[x_feature, y_feature])
    if data.empty:
        return empty_figure("No tracks have values for both selected features.", height)
    color = color_by if color_by and color_by in data.columns else None
    fig = px.scatter(data, x=x_feature, y=y_feature, color=color, color_discrete_sequence=PALETTE,
                     hover_name="track_name" if "track_name" in data.columns else None,
                     hover_data=["artist"] if "artist" in data.columns else None)
    fig.update_traces(marker=dict(size=8, opacity=0.85, line=dict(width=0)))
    fig.update_xaxes(title=_label(x_feature))
    fig.update_yaxes(title=_label(y_feature))
    if color:
        fig.update_layout(legend_title_text=_label(color))
    return apply_theme(fig, f"{_label(x_feature)} vs {_label(y_feature)} across the catalog", height)


def popularity_chart(tracks: pd.DataFrame | None, height: int = 320) -> go.Figure:
    """Histogram of the popularity column (if present)."""
    if tracks is None or "popularity" not in tracks.columns or not tracks["popularity"].notna().any():
        return empty_figure("No popularity data available.", height)
    fig = go.Figure(go.Histogram(x=tracks["popularity"].dropna(), nbinsx=20, marker_color=BLUE,
                                 hovertemplate="popularity: %{x}<br>tracks: %{y}<extra></extra>"))
    fig.update_xaxes(title="Popularity (0–100)")
    fig.update_yaxes(title="Tracks")
    return apply_theme(fig, "Popularity distribution", height)


def session_length_chart(sessions: pd.DataFrame | None, height: int = 320) -> go.Figure:
    """Number of sessions at each session length."""
    if sessions is None or len(sessions) == 0:
        return empty_figure("No session data available.", height)
    counts = sessions.groupby("session_id").size().value_counts().sort_index()
    fig = go.Figure(go.Bar(x=counts.index.astype(int), y=counts.values, marker_color=VIOLET,
                           hovertemplate="%{x} tracks: %{y} sessions<extra></extra>"))
    fig.update_xaxes(title="Tracks per session", dtick=1)
    fig.update_yaxes(title="Sessions")
    return apply_theme(fig, "Session lengths", height)


def top_transitions_chart(transitions: pd.DataFrame | None, height: int | None = None) -> go.Figure:
    """Bar chart of the most frequent 'followed within a few plays' pairs from training sessions.

    Expects the DataFrame returned by VibeFlowRecommender.top_transitions().
    """
    needed = {"from_track", "to_track", "count"}
    if transitions is None or len(transitions) == 0 or not needed.issubset(transitions.columns):
        return empty_figure("No transitions learned. Load session data to see them.")
    labels = [f"{_short(a, 22)} → {_short(b, 22)}"
              for a, b in zip(transitions["from_track"], transitions["to_track"])]
    fig = go.Figure(go.Bar(
        y=labels, x=transitions["count"], orientation="h", marker_color=BLUE,
        customdata=np.column_stack([transitions["from_track"], transitions["to_track"]]),
        hovertemplate="%{customdata[0]} → %{customdata[1]}<br>count: %{x}<extra></extra>"))
    fig.update_yaxes(autorange="reversed")
    fig.update_xaxes(title="Times observed in training sessions")
    return apply_theme(fig, f"Most frequent pairs (second track within {SEQ_WINDOW} plays of the first)",
                       height or max(320, 30 * len(transitions) + 120))


def transition_distance_chart(sequences: Mapping[str, Sequence[str]] | Iterable[Sequence[str]] | None,
                              fm: FeatureMatrix | None, n_random_pairs: int = DEFAULT_RANDOM_PAIRS,
                              seed: int = RANDOM_SEED, height: int = 380) -> go.Figure:
    """Compare audio distance between consecutive tracks with distance between random pairs.

    Distances are Euclidean in the standardized feature space. Consecutive distances smaller
    than random ones mean sessions tend to move between similar tracks. On the synthetic demo
    data this reflects structure planted by the generator, not real listener behaviour.
    """
    if sequences is None or fm is None or len(fm.track_ids) < 2:
        return empty_figure("Need sessions and at least two tracks.", height)
    seq_list = list(sequences.values()) if hasattr(sequences, "values") else list(sequences)
    consecutive: list[float] = []
    for seq in seq_list:
        idx = [fm.index[str(t)] for t in seq if str(t) in fm.index]
        if len(idx) >= 2:
            consecutive.extend(np.linalg.norm(np.diff(fm.standardized[idx], axis=0), axis=1).tolist())
    if not consecutive:
        return empty_figure("No consecutive track pairs found in the sessions.", height)

    rng = np.random.default_rng(seed)
    a = rng.integers(0, len(fm.track_ids), size=n_random_pairs)
    b = rng.integers(0, len(fm.track_ids), size=n_random_pairs)
    keep = a != b
    random_d = np.linalg.norm(fm.standardized[a[keep]] - fm.standardized[b[keep]], axis=1)

    fig = go.Figure()
    fig.add_trace(go.Histogram(x=consecutive, histnorm="probability density", opacity=0.65,
                               marker_color=MAGENTA, nbinsx=30,
                               name=f"Consecutive tracks (mean {np.mean(consecutive):.2f})"))
    fig.add_trace(go.Histogram(x=random_d, histnorm="probability density", opacity=0.55,
                               marker_color=BLUE, nbinsx=30,
                               name=f"Random pairs (mean {random_d.mean():.2f})"))
    fig.update_layout(barmode="overlay")
    fig.update_xaxes(title="Audio distance (standardized features)")
    fig.update_yaxes(title="Density")
    return apply_theme(fig, "Do sessions move between similar tracks?", height)