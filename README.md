# Music Comparison App

Compare a student's recording against a reference recording, measure differences in
**pitch**, **dynamics**, and **tone**, see where they diverge, and get actionable feedback.

See [PLAN.md](PLAN.md) for the full design and roadmap.

## Setup (Windows)

Prerequisites: [uv](https://docs.astral.sh/uv/), [FFmpeg](https://ffmpeg.org/) (`winget install Gyan.FFmpeg`), Git.

This repo lives in a OneDrive folder, so the virtual environment is kept **outside** it.
Do this once (it persists for your Windows user):

```powershell
[Environment]::SetEnvironmentVariable("UV_PROJECT_ENVIRONMENT", "C:\dev\venvs\music-comparison-app", "User")
```

Then open a **new** terminal in the project folder:

```powershell
uv sync                     # create the venv and install everything from uv.lock
uv run pre-commit install   # run lint checks automatically before every commit
```

## Everyday commands

| Command | What it does |
|---|---|
| `uv run pytest` | Run the tests |
| `uv run ruff check .` | Lint (find problems) |
| `uv run ruff format .` | Auto-format code |
| `uv run mypy` | Type-check `src/` and `app/` |
| `uv run streamlit run app/streamlit_app.py` | Start the app |
| `uv run pytest --durations=10` | Run the tests and list the slowest ones |

The full test suite takes a few minutes: most of it is real pitch tracking on
synthetic audio, plus headless runs of the Streamlit pages.

## Using the app

1. **Compare** page: upload a reference recording and your own take of the same
   passage (WAV, MP3, FLAC, OGG, or M4A; 1 s to 5 min). Pick your instrument in
   the sidebar so the tips fit it.
2. You get scores for pitch, dynamics, and tone, a ranked list of what to work on,
   charts of where you differ, and a player for each problem spot.
3. Rate tips with 👍/👎. Ratings are saved and will train a future model.
4. **History** page: your saved sessions and how your scores changed over time.

Works best when:
- both recordings are of **one instrument playing one line** (a solo reference
  is better than one with piano, orchestra, or lots of reverb; the app warns
  you when the reference sounds accompanied);
- both are trimmed to **the same passage** (count-ins and extra bars are
  detected and reported, but aren't scored).

## Your data and privacy

Everything stays **on this computer**, in the `data/` folder of the project
(which is in `.gitignore`, so it never reaches GitHub). Set the
`MUSICCOMPARE_DATA_DIR` environment variable to keep it somewhere else.

- Saved by default: scores, measurements (pitch/loudness/tone curves), tips,
  and your ratings. Switch off "Save to my practice history" in the sidebar to
  save nothing.
- Your **recordings** are saved only if you tick "Also keep copies of the
  recordings". Unticking it deletes the copies for that take.
- Comparisons of two different pieces are never saved.
- The History page has a "Delete history" button that removes everything.
