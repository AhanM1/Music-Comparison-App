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
| `uv run mypy` | Type-check `src/` |
| `uv run streamlit run app/streamlit_app.py` | Start the app (from Phase 1) |
