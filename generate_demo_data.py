"""Generate the deterministic SYNTHETIC demonstration dataset for VibeFlow.

Run from the project root:  python scripts/generate_demo_data.py

All tracks, artists and listening sessions are artificial. Structure is planted
on purpose (vibe archetypes, archetype transition preferences, favoured
successor tracks), so results on this data demonstrate that the software works,
NOT how well it would work for real listeners.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

SEED = 42
TRACKS_PER_ARCHETYPE = 20
ARTISTS_PER_ARCHETYPE = 5
N_USERS = 40
N_SESSIONS = 400
MIN_SESSION_LEN, MAX_SESSION_LEN = 5, 10
SUCCESSOR_PROB = 0.25      # chance the next track is one of the current track's favoured successors
USER_PREF_PROB = 0.85      # chance a session starts in one of the user's two preferred archetypes

DATA_DIR = Path(__file__).resolve().parents[1] / "data"

# Mean audio features per archetype (synthetic "vibes").
ARCHETYPES = {
    "dreamy": {
        "genres": ["dream pop", "indie electronic", "shoegaze"],
        "mean": dict(valence=0.45, energy=0.40, danceability=0.50, tempo=105,
                     acousticness=0.35, instrumentalness=0.25, speechiness=0.04, loudness=-10.0),
    },
    "dark_bass": {
        "genres": ["dark electronic", "bass music", "industrial"],
        "mean": dict(valence=0.20, energy=0.85, danceability=0.65, tempo=140,
                     acousticness=0.05, instrumentalness=0.30, speechiness=0.08, loudness=-5.0),
    },
    "party_pop": {
        "genres": ["dance pop", "synth pop", "electropop"],
        "mean": dict(valence=0.80, energy=0.80, danceability=0.80, tempo=122,
                     acousticness=0.10, instrumentalness=0.01, speechiness=0.06, loudness=-5.5),
    },
    "chill_acoustic": {
        "genres": ["acoustic", "folk", "singer-songwriter"],
        "mean": dict(valence=0.55, energy=0.25, danceability=0.50, tempo=95,
                     acousticness=0.80, instrumentalness=0.10, speechiness=0.04, loudness=-14.0),
    },
    "hype_rap": {
        "genres": ["hip hop", "trap", "rap"],
        "mean": dict(valence=0.50, energy=0.80, danceability=0.75, tempo=135,
                     acousticness=0.10, instrumentalness=0.01, speechiness=0.30, loudness=-6.0),
    },
    "melancholic": {
        "genres": ["sad pop", "ambient", "piano ballad"],
        "mean": dict(valence=0.20, energy=0.30, danceability=0.40, tempo=90,
                     acousticness=0.60, instrumentalness=0.15, speechiness=0.04, loudness=-12.0),
    },
}
ARCH_NAMES = list(ARCHETYPES)

# Row = current archetype, column = next archetype (same order as ARCH_NAMES). Rows sum to 1.
TRANSITIONS = np.array([
    [0.45, 0.20, 0.10, 0.10, 0.00, 0.15],  # dreamy
    [0.10, 0.40, 0.10, 0.00, 0.30, 0.10],  # dark_bass
    [0.10, 0.10, 0.40, 0.05, 0.30, 0.05],  # party_pop
    [0.25, 0.00, 0.05, 0.40, 0.00, 0.30],  # chill_acoustic
    [0.00, 0.30, 0.25, 0.00, 0.40, 0.05],  # hype_rap
    [0.25, 0.10, 0.00, 0.25, 0.05, 0.35],  # melancholic
])

FEATURE_SD = dict(valence=0.10, energy=0.10, danceability=0.08, tempo=8.0,
                  acousticness=0.12, instrumentalness=0.08, speechiness=0.03, loudness=1.5)
FEATURE_CLIP = dict(valence=(0, 1), energy=(0, 1), danceability=(0, 1), tempo=(60, 200),
                    acousticness=(0, 1), instrumentalness=(0, 1), speechiness=(0, 1), loudness=(-30, 0))
DECIMALS = dict(tempo=1, loudness=2)  # everything else: 3

ADJECTIVES = ["Neon", "Velvet", "Midnight", "Paper", "Glass", "Silver", "Hollow", "Golden",
              "Quiet", "Electric", "Faded", "Violet", "Distant", "Burning", "Soft"]
NOUNS = ["Harbor", "Echo", "Garden", "Signal", "Skyline", "Memory", "Static", "Orbit",
         "Weather", "Mirror", "Satellite", "Tide", "Window", "Pulse", "Horizon"]
FIRST_NAMES = ["Ari", "Mika", "Juno", "Rio", "Sana", "Teo", "Lena", "Kai", "Nova", "Iris", "Odin", "Zara"]
LAST_NAMES = ["Vale", "Okoro", "Lund", "Reyes", "Marsh", "Kapoor", "Frost", "Ibarra", "Sato", "Quill", "Hale", "Dorn"]


def make_artists(rng: np.random.Generator) -> list[list[str]]:
    """Return one list of synthetic artist names per archetype."""
    combos = [f"{f} {l}" for f in FIRST_NAMES for l in LAST_NAMES]
    picks = rng.choice(len(combos), size=len(ARCH_NAMES) * ARTISTS_PER_ARCHETYPE, replace=False)
    names = [combos[i] for i in picks]
    return [names[i * ARTISTS_PER_ARCHETYPE:(i + 1) * ARTISTS_PER_ARCHETYPE] for i in range(len(ARCH_NAMES))]


def make_tracks(rng: np.random.Generator) -> pd.DataFrame:
    """Create the synthetic catalog. Track IDs are assigned after shuffling so ID order leaks nothing."""
    artists = make_artists(rng)
    rows = []
    for a_idx, name in enumerate(ARCH_NAMES):
        spec = ARCHETYPES[name]
        for _ in range(TRACKS_PER_ARCHETYPE):
            row = {"archetype": a_idx}  # internal only; dropped before saving
            for feat, mean in spec["mean"].items():
                lo, hi = FEATURE_CLIP[feat]
                value = float(np.clip(rng.normal(mean, FEATURE_SD[feat]), lo, hi))
                row[feat] = round(value, DECIMALS.get(feat, 3))
            row["genre"] = spec["genres"][int(rng.integers(len(spec["genres"])))]
            row["artist"] = artists[a_idx][int(rng.integers(ARTISTS_PER_ARCHETYPE))]
            row["popularity"] = int(round(float(rng.beta(2, 3)) * 100))
            rows.append(row)

    order = rng.permutation(len(rows))
    rows = [rows[i] for i in order]
    name_combos = [f"{a} {n}" for a in ADJECTIVES for n in NOUNS]
    name_idx = rng.choice(len(name_combos), size=len(rows), replace=False)
    for i, row in enumerate(rows):
        row["track_id"] = f"T{i + 1:03d}"
        row["track_name"] = name_combos[name_idx[i]]
        row["source_type"] = "synthetic"
    return pd.DataFrame(rows)


def make_sessions(tracks: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    """Create ordered synthetic listening sessions from the archetype/successor structure."""
    n = len(tracks)
    arch = tracks["archetype"].to_numpy()
    ids = tracks["track_id"].tolist()
    popularity = tracks["popularity"].to_numpy(dtype=float)
    by_arch = {a: np.where(arch == a)[0] for a in range(len(ARCH_NAMES))}

    # Two favoured successor tracks per track (planted track-level sequential signal).
    successors: dict[int, list[int]] = {}
    for i in range(n):
        picks: list[int] = []
        while len(picks) < 2:
            a = int(rng.choice(len(ARCH_NAMES), p=TRANSITIONS[arch[i]]))
            j = int(rng.choice(by_arch[a]))
            if j != i and j not in picks:
                picks.append(j)
        successors[i] = picks

    def pick_from_archetype(a: int, used: set[int]) -> int:
        pool = [int(i) for i in by_arch[a] if int(i) not in used]
        if not pool:
            pool = [i for i in range(n) if i not in used]
        weights = popularity[pool] + 10.0  # more popular tracks are somewhat more likely
        return int(rng.choice(pool, p=weights / weights.sum()))

    user_prefs = [rng.choice(len(ARCH_NAMES), size=2, replace=False) for _ in range(N_USERS)]
    start = pd.Timestamp("2025-01-01 08:00:00")
    rows = []
    for s in range(N_SESSIONS):
        user = int(rng.integers(N_USERS))
        if rng.random() < USER_PREF_PROB:
            a = int(rng.choice(user_prefs[user]))
        else:
            a = int(rng.integers(len(ARCH_NAMES)))
        length = int(rng.integers(MIN_SESSION_LEN, MAX_SESSION_LEN + 1))

        current = pick_from_archetype(a, set())
        seq = [current]
        for _ in range(length - 1):
            nxt = None
            if rng.random() < SUCCESSOR_PROB:
                candidates = [j for j in successors[current] if j not in seq]
                if candidates:
                    nxt = int(rng.choice(candidates))
            if nxt is None:
                a_next = int(rng.choice(len(ARCH_NAMES), p=TRANSITIONS[arch[current]]))
                nxt = pick_from_archetype(a_next, set(seq))
            seq.append(nxt)
            current = nxt

        session_start = start + pd.Timedelta(hours=6 * s)
        for pos, idx in enumerate(seq, start=1):
            rows.append({
                "session_id": f"S{s + 1:04d}",
                "user_id": f"U{user + 1:03d}",
                "track_id": ids[idx],
                "position": pos,
                "timestamp": (session_start + pd.Timedelta(minutes=3 * (pos - 1))).isoformat(),
                "source_type": "synthetic",
            })
    return pd.DataFrame(rows)


def main() -> None:
    rng = np.random.default_rng(SEED)
    tracks = make_tracks(rng)
    sessions = make_sessions(tracks, rng)

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    columns = ["track_id", "track_name", "artist", "genre", "valence", "energy", "danceability",
               "tempo", "acousticness", "instrumentalness", "speechiness", "loudness",
               "popularity", "source_type"]
    tracks[columns].to_csv(DATA_DIR / "demo_tracks.csv", index=False)
    sessions.to_csv(DATA_DIR / "demo_sessions.csv", index=False)
    print(f"Wrote {len(tracks)} synthetic tracks and {len(sessions)} synthetic session rows "
          f"({sessions['session_id'].nunique()} sessions) to {DATA_DIR}")


if __name__ == "__main__":
    main()
    