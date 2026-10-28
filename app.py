"""VibeFlow - Streamlit app.

Run from the project root:  streamlit run app.py

Screens: Discover, Current Vibe, Recommendations, Analytics, Data & Methodology.
All numbers and charts come from the loaded CSVs and the implemented models.
"""
from __future__ import annotations

import html
import inspect
import re
from typing import Any

import pandas as pd
import streamlit as st

from src import visualization as viz
from src.data_loader import DEFAULT_SESSIONS_PATH, DEFAULT_TRACKS_PATH, DataValidationError, load_sessions, load_tracks
from src.preprocessing import build_feature_matrix, build_session_sequences
from src.recommender import (DEFAULT_WEIGHTS, LAG_DECAY, METHOD_LABELS, METHODS, RECENCY_DECAY,
                             SEQ_WINDOW, SMOOTHING_ALPHA, VibeFlowRecommender)

st.set_page_config(page_title="VibeFlow", page_icon="🎧", layout="wide")

PAGES = ["Discover", "Current Vibe", "Recommendations", "Analytics", "Data & Methodology"]
MAX_SESSION_TRACKS = 30
MAX_PICKER_OPTIONS = 300
DEFAULT_K = 10
MAX_UI_K = 20
EXAMPLE_SESSION_LENGTH = 4
TOP_TRANSITIONS_DEFAULT = 15

CUSTOM_CSS = f"""
<style>
.vf-hero-title {{
    font-size: 3.2rem; font-weight: 800; line-height: 1.1; margin: 0 0 .3rem 0;
    background: linear-gradient(90deg, {viz.VIOLET}, {viz.BLUE} 50%, {viz.MAGENTA});
    -webkit-background-clip: text; background-clip: text; color: transparent;
}}
.vf-hero-sub {{ color: {viz.MUTED}; font-size: 1.1rem; max-width: 48rem; margin-bottom: 1.2rem; }}
.vf-side-title {{
    font-size: 1.6rem; font-weight: 800; margin: 0 0 .5rem 0;
    background: linear-gradient(90deg, {viz.VIOLET}, {viz.MAGENTA});
    -webkit-background-clip: text; background-clip: text; color: transparent;
}}
.vf-rank {{ font-size: 2.1rem; font-weight: 800; color: {viz.MAGENTA}; line-height: 1.1; }}
.vf-track {{ font-size: 1.15rem; font-weight: 700; color: {viz.TEXT}; }}
.vf-meta {{ color: {viz.MUTED}; font-size: .9rem; margin-top: 2px; }}
.vf-pill {{
    display: inline-block; padding: 1px 10px; border-radius: 999px; font-size: .72rem;
    font-weight: 600; margin-left: 10px; border: 1px solid; vertical-align: middle;
}}
.vf-pill-familiar {{ color: {viz.VIOLET}; border-color: {viz.VIOLET}; background: rgba(139,92,246,.15); }}
.vf-pill-explore {{ color: {viz.MAGENTA}; border-color: {viz.MAGENTA}; background: rgba(217,70,239,.15); }}
.vf-step-num {{ color: {viz.MAGENTA}; font-weight: 800; font-size: 1.4rem; }}
</style>
"""

_MD_SPECIAL = re.compile(r"([\\`*_\[\]<>$#~|])")
_PLOTLY_HAS_WIDTH = "width" in inspect.signature(st.plotly_chart).parameters


# --------------------------------------------------------------------------- #
# Small helpers
# --------------------------------------------------------------------------- #
def md(text: Any) -> str:
    """Escape characters that Streamlit's markdown would interpret (data-derived text only)."""
    return _MD_SPECIAL.sub(r"\\\1", str(text))


def label(feature: str) -> str:
    return feature.replace("_", " ").capitalize()


def show_chart(fig, key: str) -> None:
    """Render a Plotly figure full-width, working across Streamlit versions."""
    if _PLOTLY_HAS_WIDTH:
        st.plotly_chart(fig, key=key, theme=None, width="stretch")
    else:
        st.plotly_chart(fig, key=key, theme=None, use_container_width=True)


# --------------------------------------------------------------------------- #
# Data loading (cached once per server process)
# --------------------------------------------------------------------------- #
@st.cache_resource(show_spinner="Loading catalog and sessions...")
def load_app_data() -> dict[str, Any]:
    """Load CSVs, build features, and fit the recommender. Returns {'error': msg} on fatal problems."""
    try:
        tracks, track_report = load_tracks()
    except DataValidationError as exc:
        return {"error": str(exc)}
    except Exception as exc:  # noqa: BLE001 - show any unexpected failure in the UI
        return {"error": f"Unexpected problem while loading the track catalog: {exc}"}

    try:
        fm = build_feature_matrix(tracks)
        recommender = VibeFlowRecommender(tracks, fm)
    except ValueError as exc:
        return {"error": f"Could not build the recommender: {exc}"}
    except Exception as exc:  # noqa: BLE001
        return {"error": f"Unexpected problem while preparing features: {exc}"}

    sessions = pd.DataFrame(columns=["session_id", "user_id", "track_id", "position", "timestamp", "source_type"])
    sequences: dict[str, list[str]] = {}
    session_issues: list[str] = []
    session_error: str | None = None
    try:
        sessions, session_report = load_sessions(valid_track_ids=set(tracks["track_id"]))
        session_issues = session_report.issues
        sequences = build_session_sequences(sessions)
        recommender.fit(sequences)
    except DataValidationError as exc:
        session_error = str(exc)
    except Exception as exc:  # noqa: BLE001
        session_error = f"Unexpected problem while loading sessions: {exc}"

    labels = {tid: f"{name} — {artist} ({genre})" for tid, name, artist, genre
              in zip(tracks["track_id"], tracks["track_name"], tracks["artist"], tracks["genre"])}
    synthetic = bool(tracks["source_type"].str.lower().eq("synthetic").any())
    if "source_type" in sessions.columns and len(sessions):
        synthetic = synthetic or bool(sessions["source_type"].str.lower().eq("synthetic").any())

    example = next((tuple(seq[:EXAMPLE_SESSION_LENGTH]) for seq in sequences.values()
                    if len(seq) >= EXAMPLE_SESSION_LENGTH), None)

    return {
        "tracks": tracks, "sessions": sessions, "fm": fm, "recommender": recommender,
        "sequences": sequences, "labels": labels, "is_synthetic": synthetic,
        "quality_notes": track_report.issues + session_issues + fm.notes,
        "session_error": session_error, "example_session": example,
    }


# --------------------------------------------------------------------------- #
# Session state and callbacks
# --------------------------------------------------------------------------- #
def init_state(ctx: dict[str, Any]) -> None:
    st.session_state.setdefault("page", PAGES[0])
    st.session_state.setdefault("session_tracks", [])
    valid = set(ctx["fm"].index)
    st.session_state["session_tracks"] = [t for t in st.session_state["session_tracks"] if t in valid]


def go_to(page: str) -> None:
    st.session_state["page"] = page


def add_track(track_id: str) -> None:
    session = st.session_state["session_tracks"]
    if track_id not in session and len(session) < MAX_SESSION_TRACKS:
        session.append(track_id)


def remove_track(index: int) -> None:
    session = st.session_state["session_tracks"]
    if 0 <= index < len(session):
        session.pop(index)


def move_track(index: int, step: int) -> None:
    session = st.session_state["session_tracks"]
    target = index + step
    if 0 <= index < len(session) and 0 <= target < len(session):
        session[index], session[target] = session[target], session[index]


def reset_session() -> None:
    st.session_state["session_tracks"] = []


def load_example(track_ids: tuple[str, ...]) -> None:
    st.session_state["session_tracks"] = list(track_ids)
    st.session_state["page"] = "Current Vibe"


# --------------------------------------------------------------------------- #
# Shared UI pieces
# --------------------------------------------------------------------------- #
def synthetic_banner(ctx: dict[str, Any]) -> None:
    if ctx["is_synthetic"]:
        st.info("**Synthetic demonstration data.** The tracks, artists and listening sessions here are artificial, "
                "generated by `scripts/generate_demo_data.py`. Patterns in them show how the software behaves, "
                "not how real listeners behave.")
    if ctx["session_error"]:
        st.warning(f"Session data could not be loaded, so sequence-based recommendations are unavailable. "
                   f"Details: {ctx['session_error']}")


def render_error(message: str) -> None:
    st.title("VibeFlow")
    st.error(message)
    st.markdown("**Things to try** (PowerShell, from the project root with the virtual environment active):")
    st.code("python scripts/generate_demo_data.py\npython scripts/check_data.py", language="powershell")
    st.caption(f"Expected files: {DEFAULT_TRACKS_PATH.name} and {DEFAULT_SESSIONS_PATH.name} in the data folder.")


# --------------------------------------------------------------------------- #
# Screen A: Discover
# --------------------------------------------------------------------------- #
def render_discover(ctx: dict[str, Any]) -> None:
    st.markdown('<p class="vf-hero-title">VibeFlow</p>', unsafe_allow_html=True)
    st.markdown('<p class="vf-hero-sub">Pick a few tracks in the order you would listen to them. VibeFlow '
                'summarizes that session\'s audio profile, checks which tracks tended to follow which in '
                'listening sessions, and suggests what to play next, with a reason for every suggestion.</p>',
                unsafe_allow_html=True)
    st.caption("Research question: can the sequence of songs someone plays reveal their current vibe and help "
               "predict what they want next? This is a prototype for exploring that question, not a claim of "
               "a novel method.")

    c1, c2, c3 = st.columns(3)
    steps = [
        ("1", "Build a session", "Choose tracks and arrange them in listening order."),
        ("2", "Read the vibe", "See a recency-weighted audio profile of the session and how it shifts."),
        ("3", "Get next tracks", "Compare audio similarity, observed transitions and popularity, with explanations."),
    ]
    for col, (num, title, body) in zip((c1, c2, c3), steps):
        with col.container(border=True):
            st.markdown(f'<span class="vf-step-num">{num}</span>', unsafe_allow_html=True)
            st.markdown(f"**{title}**")
            st.caption(body)

    st.write("")
    n_sessions = ctx["sessions"]["session_id"].nunique() if len(ctx["sessions"]) else 0
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Tracks in catalog", len(ctx["tracks"]))
    m2.metric("Listening sessions", n_sessions)
    m3.metric("Sessions used to learn transitions", ctx["recommender"].n_training_sessions)
    m4.metric("Audio features used", len(ctx["fm"].features))

    st.write("")
    session = st.session_state["session_tracks"]
    b1, b2, _ = st.columns([2, 3, 4])
    b1.button("Start a session →", type="primary", on_click=go_to, args=("Current Vibe",), key="disc_start")
    if ctx["example_session"]:
        b2.button("Load an example session from the data", on_click=load_example,
                  args=(ctx["example_session"],), key="disc_example",
                  help=f"Uses the first {EXAMPLE_SESSION_LENGTH} tracks of the earliest session in the dataset.")
    if session:
        st.success(f"You already have {len(session)} track(s) in your session.")
        st.button("Continue to recommendations →", on_click=go_to, args=("Recommendations",), key="disc_continue")


# --------------------------------------------------------------------------- #
# Screen B: Current Vibe
# --------------------------------------------------------------------------- #
def render_current_vibe(ctx: dict[str, Any]) -> None:
    st.header("Current Vibe")
    tracks, labels, fm, recommender = ctx["tracks"], ctx["labels"], ctx["fm"], ctx["recommender"]
    session = st.session_state["session_tracks"]
    vibe = recommender.estimate_vibe(session) if session else None
    weights = vibe.recency_weights if vibe is not None and len(vibe.recency_weights) == len(session) else []

    left, right = st.columns([5, 6], gap="large")
    with left:
        st.subheader("Build your session")
        search = st.text_input("Search by track or artist", key="track_search", placeholder="Type to filter")
        genres = ["All genres"] + sorted(tracks["genre"].unique())
        genre = st.selectbox("Genre", genres, key="genre_filter")

        pool = tracks
        if genre != "All genres":
            pool = pool[pool["genre"] == genre]
        needle = search.strip().lower()
        if needle:
            pool = pool[pool["track_name"].str.lower().str.contains(needle, regex=False)
                        | pool["artist"].str.lower().str.contains(needle, regex=False)]
        pool = pool[~pool["track_id"].isin(session)]
        options = pool["track_id"].tolist()[:MAX_PICKER_OPTIONS]

        if len(session) >= MAX_SESSION_TRACKS:
            st.warning(f"Sessions are limited to {MAX_SESSION_TRACKS} tracks.")
        elif not options:
            st.info("No tracks match these filters (tracks already in your session are hidden).")
        else:
            if len(pool) > MAX_PICKER_OPTIONS:
                st.caption(f"Showing the first {MAX_PICKER_OPTIONS} of {len(pool)} matches; refine the search.")
            choice = st.selectbox("Next track in your session", options, format_func=lambda t: labels[t])
            st.button("Add to session", type="primary", on_click=add_track, args=(choice,), key="cv_add")

        st.subheader("Listening order")
        if not session:
            st.info("Your session is empty. Add tracks above, or load an example from the Discover screen.")
        else:
            for i, tid in enumerate(session):
                c0, c1, c2, c3, c4 = st.columns([0.6, 5, 0.9, 0.9, 0.9])
                c0.markdown(f"**{i + 1}**")
                row = tracks.loc[tracks["track_id"] == tid].iloc[0]
                weight_text = f" · weight {weights[i]:.2f}" if weights else ""
                c1.markdown(f"**{md(row['track_name'])}**  \n{md(row['artist'])} · {md(row['genre'])}{weight_text}")
                c2.button("↑", key=f"up_{i}", on_click=move_track, args=(i, -1), disabled=(i == 0),
                          help="Play earlier")
                c3.button("↓", key=f"down_{i}", on_click=move_track, args=(i, 1),
                          disabled=(i == len(session) - 1), help="Play later")
                c4.button("✕", key=f"rm_{i}", on_click=remove_track, args=(i,), help="Remove from session")
            if weights:
                st.caption("'Weight' is how much each track counts toward the vibe; recent tracks count more.")
            a1, a2 = st.columns(2)
            a1.button("Reset session", on_click=reset_session, key="cv_reset")
            a2.button("Get recommendations →", type="primary", on_click=go_to, args=("Recommendations",),
                      key="cv_recs")

    with right:
        st.subheader("Estimated vibe")
        if vibe is None:
            st.info("Add at least one track to see the session's audio profile.")
        else:
            st.markdown(md(vibe.description))
            m1, m2 = st.columns(2)
            m1.metric("Tracks in session", len(session))
            m2.metric("Consecutive-track contrast",
                      f"{vibe.contrast_ratio:.2f}×" if vibe.contrast_ratio is not None else "n/a",
                      help="Average audio distance between consecutive tracks divided by the average distance "
                           "between random catalog pairs. Below 1 = smoother than random.")
            st.caption(vibe.contrast_label)
        show_chart(viz.vibe_radar(vibe), "cv_radar")

    if vibe is not None:
        tab1, tab2, tab3 = st.tabs(["Feature values", "How the session shifts", "Across the sequence"])
        with tab1:
            show_chart(viz.vibe_profile_bars(vibe), "cv_bars")
        with tab2:
            show_chart(viz.vibe_trend_chart(vibe), "cv_trend")
            st.caption("Compares the later half of the session with the earlier half. Needs at least 2 tracks.")
        with tab3:
            show_chart(viz.sequence_feature_chart(vibe, fm), "cv_seq")
            st.caption("Click legend entries to show or hide features.")


# --------------------------------------------------------------------------- #
# Screen C: Recommendations
# --------------------------------------------------------------------------- #
def render_score_details(ctx: dict[str, Any], rec, result) -> None:
    fm = ctx["fm"]
    if rec.components:
        rows = []
        for name, value in rec.components.items():
            weight = (result.weights_used or {}).get(name)
            rows.append({
                "component": name,
                "normalized score (0-1)": round(value, 3),
                "weight": round(weight, 3) if weight is not None else None,
                "contribution": round(weight * value, 3) if weight is not None else None,
            })
        st.dataframe(pd.DataFrame(rows), hide_index=True)
    if result.vibe is not None and rec.track_id in fm.index:
        idx = fm.index[rec.track_id]
        table = pd.DataFrame({
            "feature": [label(f) for f in fm.features],
            "this track (0-1)": [round(float(fm.scaled[idx, c]), 2) for c in range(len(fm.features))],
            "session (0-1)": [round(float(result.vibe.profile[f]), 2) for f in fm.features],
        })
        st.dataframe(table, hide_index=True)
        st.caption("Values are relative to this catalog (0 = lowest track, 1 = highest).")


def render_recommendations(ctx: dict[str, Any]) -> None:
    st.header("Recommendations")
    recommender, fm, tracks = ctx["recommender"], ctx["fm"], ctx["tracks"]
    session = st.session_state["session_tracks"]
    if not session:
        st.info("Build a listening session first, then come back for next-track suggestions.")
        st.button("Build a session →", type="primary", on_click=go_to, args=("Current Vibe",), key="rec_build")
        return

    c1, c2, c3 = st.columns([3, 2, 2])
    method = c1.selectbox("Method", METHODS, format_func=lambda m: METHOD_LABELS[m], key="rec_method")
    k = c2.slider("Number of recommendations", 1, MAX_UI_K, DEFAULT_K, key="rec_k")
    diversity = c3.slider("Diversity", 0.0, 1.0, 0.0, 0.05, key="rec_div",
                          help="0 = rank purely by score. Higher values avoid near-duplicate picks (MMR re-ranking).")
    weights = None
    if method == "hybrid":
        with st.expander("Hybrid weights (normalized automatically)"):
            w1, w2, w3 = st.columns(3)
            weights = {
                "content": w1.slider("Content", 0.0, 1.0, float(DEFAULT_WEIGHTS["content"]), 0.05, key="w_content"),
                "sequence": w2.slider("Sequence", 0.0, 1.0, float(DEFAULT_WEIGHTS["sequence"]), 0.05, key="w_seq"),
                "popularity": w3.slider("Popularity", 0.0, 1.0, float(DEFAULT_WEIGHTS["popularity"]), 0.05,
                                        key="w_pop"),
            }
            st.caption("These defaults are a starting point, not tuned values. Tuning belongs on a validation split.")

    try:
        with st.spinner("Ranking tracks..."):
            result = recommender.recommend(session, k=k, method=method, weights=weights, diversity=diversity)
    except Exception as exc:  # noqa: BLE001
        st.error(f"Something went wrong while generating recommendations: {exc}")
        return

    last_name = tracks.loc[tracks["track_id"] == session[-1], "track_name"].iloc[0]
    st.subheader(f"Next up after “{last_name}”")
    if result.vibe is not None:
        st.caption(result.vibe.description)

    if result.method_used != result.method_requested:
        st.warning(f"Showing **{METHOD_LABELS[result.method_used]}** instead of "
                   f"**{METHOD_LABELS.get(result.method_requested, result.method_requested)}**; see notes below.")
    legend_notes = [n for n in result.notes if n.startswith("'Familiar'")]
    for note in result.notes:
        if note not in legend_notes:
            st.info(note)

    if result.is_empty:
        st.warning("No recommendations could be generated for this session.")
        return

    for rec in result.recommendations:
        with st.container(border=True):
            c_rank, c_main, c_score, c_btn = st.columns([0.8, 5, 2.2, 1.8])
            c_rank.markdown(f'<div class="vf-rank">{rec.rank}</div>', unsafe_allow_html=True)
            pill = ""
            if rec.tag:
                css = "vf-pill-familiar" if rec.tag.startswith("Familiar") else "vf-pill-explore"
                pill = f'<span class="vf-pill {css}">{html.escape(rec.tag)}</span>'
            pop_text = f"Popularity {rec.popularity:.0f}/100" if rec.popularity is not None else "Popularity n/a"
            c_main.markdown(
                f'<div class="vf-track">{html.escape(rec.track_name)}{pill}</div>'
                f'<div class="vf-meta">{html.escape(rec.artist)} · {html.escape(rec.genre)} · {pop_text}</div>',
                unsafe_allow_html=True)
            c_score.progress(min(max(float(rec.score), 0.0), 1.0))
            c_score.caption(f"Relative score {rec.score:.2f}")
            c_btn.button("＋ Add to session", key=f"add_{rec.track_id}", on_click=add_track, args=(rec.track_id,),
                         disabled=len(session) >= MAX_SESSION_TRACKS)
            if rec.explanation:
                st.markdown(f"**Why:** {md(rec.explanation)}")
            with st.expander("Score details"):
                render_score_details(ctx, rec, result)

    st.caption(f"{result.n_candidates} candidate tracks were ranked (tracks already in your session are excluded). "
               "Scores are relative to these candidates, not probabilities.")
    for note in legend_notes:
        st.caption(note)

    tab_scores, tab_map = st.tabs(["Score breakdown", "Feature map"])
    with tab_scores:
        show_chart(viz.recommendation_score_chart(result), "rec_scores")
    with tab_map:
        feats = fm.features
        mx, my = st.columns(2)
        x_feat = mx.selectbox("X axis", feats, index=feats.index("valence") if "valence" in feats else 0,
                              format_func=label, key="map_x")
        y_feat = my.selectbox("Y axis", feats,
                              index=feats.index("energy") if "energy" in feats else min(1, len(feats) - 1),
                              format_func=label, key="map_y")
        show_chart(viz.recommendation_map(tracks, result, result.vibe, x_feat, y_feat), "rec_map")
        st.caption("Violet dots are your session (numbered by play order); pink diamonds are recommendations.")

    st.download_button("Download these recommendations (CSV)", result.to_dataframe().to_csv(index=False),
                       file_name="vibeflow_recommendations.csv", mime="text/csv", key="rec_download")


# --------------------------------------------------------------------------- #
# Screen D: Analytics
# --------------------------------------------------------------------------- #
def render_analytics(ctx: dict[str, Any]) -> None:
    st.header("Analytics")
    tracks, sessions, fm, recommender = ctx["tracks"], ctx["sessions"], ctx["fm"], ctx["recommender"]
    has_sessions = len(sessions) > 0
    if ctx["is_synthetic"]:
        st.caption("All charts below describe synthetic data. Structure you see (genre clusters, smooth "
                   "transitions) was planted by the data generator.")

    sizes = sessions.groupby("session_id").size() if has_sessions else pd.Series(dtype=int)
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Tracks", len(tracks))
    m2.metric("Sessions", int(sizes.size))
    m3.metric("Users", int(sessions["user_id"].nunique()) if has_sessions else 0)
    m4.metric("Average session length", f"{sizes.mean():.1f}" if has_sessions else "n/a")

    tab_cat, tab_sess, tab_eval = st.tabs(["Catalog", "Sessions & transitions", "Evaluation"])

    with tab_cat:
        show_chart(viz.feature_distribution_chart(tracks), "an_dist")
        c1, c2 = st.columns(2)
        with c1:
            show_chart(viz.correlation_heatmap(tracks), "an_corr")
        with c2:
            show_chart(viz.popularity_chart(tracks), "an_pop")
        st.subheader("Explore two features")
        feats = fm.features
        s1, s2, s3 = st.columns(3)
        x_feat = s1.selectbox("X axis", feats, index=feats.index("valence") if "valence" in feats else 0,
                              format_func=label, key="an_x")
        y_feat = s2.selectbox("Y axis", feats,
                              index=feats.index("energy") if "energy" in feats else min(1, len(feats) - 1),
                              format_func=label, key="an_y")
        color_choice = s3.selectbox("Colour by", ["genre", "source_type", "None"], key="an_color")
        show_chart(viz.feature_scatter(tracks, x_feat, y_feat, None if color_choice == "None" else color_choice),
                   "an_scatter")

    with tab_sess:
        if not has_sessions:
            st.info("No session data is loaded, so session and transition charts are unavailable.")
        else:
            c1, c2 = st.columns(2)
            with c1:
                show_chart(viz.session_length_chart(sessions), "an_len")
            with c2:
                show_chart(viz.transition_distance_chart(ctx["sequences"], fm), "an_dist_trans")
            st.caption("If consecutive tracks are closer in audio space than random pairs, sessions tend to move "
                       "between similar tracks. Here that mostly reflects how the demo data was generated.")
            n_pairs = st.slider("Number of transition pairs to show", 5, 30, TOP_TRANSITIONS_DEFAULT, key="an_top_n")
            show_chart(viz.top_transitions_chart(recommender.top_transitions(n_pairs)), "an_top")
            st.caption(f"Pairs are counted when the second track appears within {SEQ_WINDOW} plays after the "
                       "first, across all loaded sessions.")

    with tab_eval:
        st.info("**Offline evaluation is not implemented yet** (planned for Stage 3). No performance numbers are "
                "shown anywhere in this app because none have been computed. When added, results will use a "
                "chronological split, training-only transition statistics, and honest baselines.")


# --------------------------------------------------------------------------- #
# Screen E: Data & Methodology
# --------------------------------------------------------------------------- #
def render_methodology(ctx: dict[str, Any]) -> None:
    st.header("Data & Methodology")
    tracks, sessions, fm = ctx["tracks"], ctx["sessions"], ctx["fm"]
    source_labels = ", ".join(sorted(tracks["source_type"].unique()))
    w = DEFAULT_WEIGHTS

    st.subheader("Dataset")
    st.markdown(
        f"- **Files:** `data/{DEFAULT_TRACKS_PATH.name}` ({len(tracks)} tracks) and "
        f"`data/{DEFAULT_SESSIONS_PATH.name}` ({len(sessions)} session rows).\n"
        f"- **Source labels found in the tracks file:** {md(source_labels)}.\n"
        "- **Synthetic vs real:** the demo files are produced by `scripts/generate_demo_data.py` with a fixed "
        "random seed, so they are identical on every run. Track names, artists, audio features, popularity values "
        "and listening sessions are all artificial and contain no third-party data.\n"
        "- **Planted structure:** the generator creates six audio 'vibe' groups, preferred transitions between "
        "groups, and favoured successor tracks. Models can therefore find these patterns, but this proves the "
        "software works, not that it helps real listeners.\n"
        "- **Using real data later:** replace the two CSVs with properly licensed files that have the same column "
        "names. Real data has not been verified or used in this project yet.")

    st.subheader("Feature representation")
    st.markdown(
        f"- **Features used ({len(fm.features)}):** {', '.join(label(f) for f in fm.features)}.\n"
        "- **Missing values** are filled with the column median; out-of-range values are treated as missing and "
        "reported rather than hidden.\n"
        "- **Two scalings:** min-max (0 to 1, for charts and profiles) and z-scores (for similarity), so features "
        "with large numeric ranges, like tempo and loudness, do not dominate.")

    st.subheader("Vibe estimation")
    st.markdown(
        f"The session's vibe is a **recency-weighted average** of its tracks' features (each step back in time is "
        f"worth {RECENCY_DECAY:.2f}× the next). The app also reports how the later half of the session differs from "
        "the earlier half, and how much consecutive tracks contrast compared with random pairs. This summarizes "
        "audio features; it is not a measurement of emotion.")

    st.subheader("Recommendation methods")
    st.markdown(
        "- **Popularity baseline:** ranks by the catalog's `popularity` column (catalog order if it is missing).\n"
        "- **Content-based:** cosine similarity between the session vibe vector and each candidate track.\n"
        f"- **Sequential:** counts how often track B appears within the next {SEQ_WINDOW} plays after track A in "
        f"the loaded sessions (a play {SEQ_WINDOW - 1} steps later counts {LAG_DECAY ** (SEQ_WINDOW - 1):.2f}× the "
        f"next one), converts counts to smoothed probabilities (additive smoothing {SMOOTHING_ALPHA}), and "
        "combines the session's most recent tracks with recency weights.\n"
        "- **Hybrid:** `hybrid_score = w_content * content + w_sequence * sequence + w_popularity * popularity` "
        f"after min-max normalizing each component. Default weights: content {w['content']:.2f}, "
        f"sequence {w['sequence']:.2f}, popularity {w['popularity']:.2f} (untuned). Unavailable components are "
        "dropped and the remaining weights renormalized.\n"
        "- **Diversity slider:** greedy re-ranking that trades score against similarity to already chosen picks.\n"
        "- **Explanations** quote the numbers actually computed for each track. Tracks already in the session are "
        "never recommended.")

    st.subheader("Evaluation design")
    st.markdown(
        "Not implemented yet. Planned: chronological session split, training-only transition statistics, "
        "next-track prediction from preceding context, Hit Rate@K, Precision@K, Recall@K, NDCG@K, coverage and "
        "diversity, with weights tuned on a validation split only.")

    st.subheader("Known limitations")
    st.markdown(
        "- The demo catalog is small and synthetic, so conclusions about real listeners cannot be drawn.\n"
        f"- Personalization comes from the current session only; `user_id` is loaded but not used by the models "
        "yet.\n"
        "- Eight audio features cannot capture lyrics, culture, artist identity or context.\n"
        "- Transition counts are sparse; unseen transitions fall back to content similarity.\n"
        "- Transition statistics are fitted on every loaded session, so the app is a demo, not an evaluation. "
        "Leakage-safe fitting arrives with the evaluation stage.")

    with st.expander("Data quality notes from loading"):
        if ctx["quality_notes"]:
            for note in ctx["quality_notes"]:
                st.write(f"- {note}")
        else:
            st.write("No data quality issues were reported.")
    with st.expander("Preview: tracks"):
        st.dataframe(tracks.head(25), hide_index=True)
    with st.expander("Preview: sessions"):
        st.dataframe(sessions.head(25), hide_index=True)


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main() -> None:
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)
    ctx = load_app_data()
    if "error" in ctx:
        render_error(ctx["error"])
        st.stop()
    init_state(ctx)

    with st.sidebar:
        st.markdown('<p class="vf-side-title">VibeFlow</p>', unsafe_allow_html=True)
        st.radio("Go to", PAGES, key="page", label_visibility="collapsed")
        st.divider()
        n_session = len(st.session_state["session_tracks"])
        st.caption(f"Current session: {n_session} track(s)")
        if n_session:
            st.button("Reset session", on_click=reset_session, key="side_reset")
        st.divider()
        n_sessions = ctx["sessions"]["session_id"].nunique() if len(ctx["sessions"]) else 0
        st.caption(f"{len(ctx['tracks'])} tracks · {n_sessions} sessions"
                   + (" · synthetic demo data" if ctx["is_synthetic"] else ""))

    synthetic_banner(ctx)
    page = st.session_state["page"]
    if page == "Discover":
        render_discover(ctx)
    elif page == "Current Vibe":
        render_current_vibe(ctx)
    elif page == "Recommendations":
        render_recommendations(ctx)
    elif page == "Analytics":
        render_analytics(ctx)
    else:
        render_methodology(ctx)


main()