"""Quick check that data loading and preprocessing work. Run: python scripts/check_data.py"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data_loader import DataValidationError, load_sessions, load_tracks
from src.preprocessing import build_feature_matrix, build_session_sequences


def main() -> int:
    try:
        tracks, track_report = load_tracks()
        sessions, session_report = load_sessions(valid_track_ids=set(tracks["track_id"]))
    except DataValidationError as exc:
        print(f"DATA ERROR: {exc}")
        return 1

    fm = build_feature_matrix(tracks)
    seqs = build_session_sequences(sessions)
    lengths = [len(s) for s in seqs.values()]

    print(f"Tracks: {len(tracks)} | Sessions: {len(seqs)} | Session rows: {len(sessions)}")
    print(f"Session length min/max: {min(lengths)}/{max(lengths)}")
    print(f"Features used: {fm.features}")
    print(f"Scaled matrix shape: {fm.scaled.shape} | min={fm.scaled.min():.2f} max={fm.scaled.max():.2f}")
    print(f"Data source labels: {sorted(tracks['source_type'].unique())}")
    issues = track_report.issues + session_report.issues + fm.notes
    print("Data quality notes:" if issues else "Data quality notes: none")
    for note in issues:
        print(f"  - {note}")
    first_id, first_seq = next(iter(seqs.items()))
    print(f"First chronological session {first_id}: {first_seq}")
    return 0


if __name__ == "__main__":
    sys.exit(main())