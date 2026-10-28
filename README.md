# VibeFlow 
### A Personalized Sequential Music Recommendation System

VibeFlow is a music recommendation project designed to explore how a listener's recent sequence of songs can help represent their current musical vibe and guide what they might want to hear next.

Unlike recommendations based only on a single track, VibeFlow explores a session-based approach that considers audio characteristics and listening sequence patterns to recommend tracks that fit the listener's evolving vibe.

> Project status: Working MVP using synthetic demonstration data. Recommendations and evaluation still need to be validated using real public music data.

 Features

- Session Building: Create a listening session by selecting genres and tracks.
- Current Vibe Analysis: Explore audio characteristics such as valence, energy, danceability, tempo, acousticness, instrumentalness, speechiness, and loudness.
- Next-Track Recommendations: Generate a list of ten recommended tracks based on the current session.
- Recommendation Explanations: Inspect the reasons provided for individual recommendations.
- Analytics Dashboard: Explore visual summaries of tracks, listening sessions, and recommendation-related information.
- Interactive Interface: Navigate the project through a Streamlit dashboard.

How It Works

1. Build the Session: Select tracks to form a listening sequence.
2. Read the Vibe: Represent the session using audio features.
3. Get Next Tracks: Generate and inspect recommended tracks and their explanations.

The project is designed to explore the relationship between audio-feature similarity and sequential listening patterns.

Technology Stack

- Language: Python
- Interface: Streamlit
- Data Processing: Pandas, NumPy
- Machine Learning Utilities: Scikit-learn
- Visualisation: Plotly
- Testing: Pytest

Project Structure

```text
VibeFlow/
├── app.py
├── requirements.txt
├── data/
│   ├── demo_tracks.csv
│   └── demo_sessions.csv
├── scripts/
│   ├── generate_demo_data.py
│   └── check_data.py
└── src/
    ├── __init__.py
    ├── data_loader.py
    ├── preprocessing.py
    ├── recommender.py
    └── visualization.py
```

Getting Started

Prerequisites

- Python 3.11 or later recommended
- pip
- Git (optional, if cloning the repository from the command line)

1. Clone the repository

```bash
git clone https://github.com/Nishwithaa/VibeFlow.git
cd VibeFlow
```

Alternatively, download the repository as a ZIP file from GitHub and extract it.

2. Create and activate a virtual environment

Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

If PowerShell blocks activation, you can use a process-scoped execution policy for the current terminal:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
```

3. Install dependencies

```bash
python -m pip install -r requirements.txt
```

4. Generate the demonstration data

```bash
python scripts/generate_demo_data.py
```

This generates the synthetic track and listening-session CSV files in the `data/` directory.

5. Launch the application

```bash
streamlit run app.py
```

Open the local URL printed by Streamlit, usually `http://localhost:8501`.

Data and Limitations

The current demonstration dataset is **synthetic**, created to develop and test the application's workflow. Its tracks, listening sessions, and audio-feature values should not be interpreted as verified real-world music data.

The current working MVP demonstrates the interface and recommendation workflow. Recommendation quality, generalisation to real listening behaviour, and performance against baseline methods have not yet been established through a formal evaluation.

Planned Improvements

- Identify and integrate a suitable public music dataset with appropriate usage permissions.
- Validate audio features and recommendation behaviour on real data.
- Compare recommendations against simple baseline methods.
- Evaluate recommendation quality using suitable offline metrics.
- Improve sequential and hybrid recommendation strategies.
- Add project screenshots and document experimental results.

Author

P. U. Nishwitha

Developed as a personal project exploring music recommendation, data processing, and interactive analytics.
