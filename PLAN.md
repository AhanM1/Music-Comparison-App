# Music Comparison App — Project Plan

> Compare a student's recording against a reference recording, measure differences in
> **pitch**, **dynamics**, and **tone**, visualize where they diverge, and give
> actionable feedback. Built with Streamlit + librosa + numpy/scipy + matplotlib,
> and designed so a machine-learning model can later replace or augment the feedback logic.

This document is both a build plan and a learning guide. Each phase lists the
concepts you'll pick up, so you can learn the "why" while we build the "what".

---

## 1. How the app works (the big picture)

```
 ┌───────────┐   ┌───────────┐   ┌──────────┐   ┌──────────┐   ┌──────────┐   ┌──────────┐
 │  Upload   │──▶│ Preprocess│──▶│  Align   │──▶│ Extract  │──▶│ Compare  │──▶│ Feedback │
 │ ref + user│   │ mono, SR, │   │  (DTW)   │   │ features │   │ & score  │   │  engine  │
 └───────────┘   │ trim, norm│   └──────────┘   │pitch/dyn/│   └──────────┘   └────┬─────┘
                 └───────────┘                  │  tone    │                       │
                                                └──────────┘                       ▼
                                                                         ┌──────────────────┐
                                                                         │ Visualize + show │
                                                                         │  in Streamlit    │
                                                                         └──────────────────┘
```

The key insight: **the two recordings won't line up in time.** The student may play
slower, start late, or rush a passage. Before we can compare note 12 to note 12, we
have to *align* the recordings. That step (Dynamic Time Warping) is the heart of the app.

---

## 2. Architecture

### 2.1 Guiding principle: keep the "brain" separate from the "face"

All audio analysis lives in a plain Python package (`musiccompare/`) that knows
**nothing** about Streamlit. The Streamlit app is a thin layer that calls into it.

Why this matters:
- **Testable** — you can unit-test pitch detection without launching a web page.
- **Swappable UI** — if you later want a mobile app or REST API (FastAPI), the core is reused unchanged.
- **ML-ready** — the ML model plugs into the same interfaces the rule-based code uses.

This is called a **layered architecture** (or "ports and adapters" in its stricter form).

### 2.2 Layers

| Layer | Package | Responsibility | Depends on |
|---|---|---|---|
| **UI** | `app/` | Streamlit pages, widgets, session state | `musiccompare` |
| **Pipeline** | `musiccompare/pipeline.py` | Orchestrates the steps end-to-end | all core modules |
| **Feedback** | `musiccompare/feedback/` | Turns differences into advice (rules now, ML later) | `models` |
| **Analysis** | `musiccompare/analysis/` | Alignment, comparison, scoring | `features`, `models` |
| **Features** | `musiccompare/features/` | Pitch, dynamics, tone extraction | `audio`, `models` |
| **Audio I/O** | `musiccompare/audio/` | Load, resample, trim, normalize | librosa, soundfile |
| **Visualization** | `musiccompare/viz/` | matplotlib figures (return `Figure`, never `plt.show()`) | `models` |
| **Models** | `musiccompare/models.py` | Typed dataclasses shared by everyone | nothing |
| **Data** | `musiccompare/data/` | Save sessions/features for future ML training | `models` |

Arrows only point **downward**. The UI can import analysis; analysis never imports UI.

### 2.3 Folder structure

```
MusicComparisonApp/
├── app/
│   ├── streamlit_app.py          # entry point: `streamlit run app/streamlit_app.py`
│   ├── pages/                    # extra Streamlit pages (history, settings)
│   └── components.py             # reusable UI widgets
├── src/
│   └── musiccompare/
│       ├── __init__.py
│       ├── config.py             # sample rate, hop length, thresholds (one place!)
│       ├── models.py             # dataclasses: AudioClip, FeatureSet, Comparison, Feedback
│       ├── pipeline.py           # compare(ref_path, user_path) -> ComparisonResult
│       ├── audio/
│       │   ├── loader.py         # load, to mono, resample
│       │   └── preprocess.py     # trim silence, loudness normalize
│       ├── features/
│       │   ├── pitch.py          # pYIN f0 tracking, cents deviation
│       │   ├── dynamics.py       # RMS energy -> dB, envelope
│       │   └── tone.py           # spectral centroid, MFCCs, flatness, rolloff
│       ├── analysis/
│       │   ├── alignment.py      # chroma + DTW warping path
│       │   ├── compare.py        # per-frame diffs on the aligned path
│       │   ├── segments.py       # group bad frames into "problem regions"
│       │   └── scoring.py        # 0–100 scores per dimension
│       ├── feedback/
│       │   ├── base.py           # FeedbackProvider protocol (the ML "socket")
│       │   ├── rules.py          # RuleBasedFeedback (v1)
│       │   └── ml.py             # MLFeedback (stub for later)
│       ├── viz/
│       │   ├── pitch_plot.py
│       │   ├── dynamics_plot.py
│       │   ├── tone_plot.py
│       │   └── overview.py       # waveform with highlighted problem regions
│       └── data/
│           └── store.py          # persist features + feedback + user ratings
├── tests/
│   ├── conftest.py               # fixtures: synthetic sine tones
│   ├── test_pitch.py
│   ├── test_dynamics.py
│   ├── test_alignment.py
│   └── test_pipeline.py
├── notebooks/                    # Jupyter experiments (learning playground)
├── sample_audio/                 # small reference/user pairs for demos
├── .github/workflows/ci.yml      # run lint + tests on every push
├── .streamlit/config.toml        # theme, upload size limit
├── pyproject.toml                # dependencies + tool config
├── .pre-commit-config.yaml
├── .gitignore
├── README.md
└── PLAN.md                       # this file
```

> **Why `src/` layout?** It forces you to install your package (`pip install -e .`)
> rather than relying on "whatever folder I happen to be in". That catches import bugs
> early and is the modern Python packaging standard.

### 2.4 The data models (the contract between layers)

```python
@dataclass(frozen=True)
class AudioClip:
    y: np.ndarray          # samples, mono, float32
    sr: int                # sample rate
    name: str

@dataclass(frozen=True)
class FeatureSet:
    times: np.ndarray      # frame timestamps (s)
    f0_hz: np.ndarray      # pitch per frame (NaN = unvoiced)
    voiced_prob: np.ndarray
    rms_db: np.ndarray     # loudness per frame
    centroid_hz: np.ndarray
    mfcc: np.ndarray       # (n_mfcc, n_frames)
    extractor_version: str # bump when feature code changes -> keeps ML data consistent

@dataclass(frozen=True)
class ProblemRegion:
    start_s: float
    end_s: float
    dimension: Literal["pitch", "dynamics", "tone"]
    severity: float        # 0..1
    detail: dict           # e.g. {"mean_cents": +32}

@dataclass(frozen=True)
class ComparisonResult:
    ref: FeatureSet
    user: FeatureSet
    warp_path: np.ndarray
    scores: dict[str, float]
    regions: list[ProblemRegion]

@dataclass(frozen=True)
class FeedbackItem:
    region: ProblemRegion | None
    message: str           # "You're ~30 cents sharp in bars 4–5..."
    tip: str               # "Relax embouchure / check finger placement..."
    confidence: float
    source: str            # "rules-v1" or "ml-v1"
```

### 2.5 The ML "socket": `FeedbackProvider`

```python
class FeedbackProvider(Protocol):
    def generate(self, result: ComparisonResult, context: PlayerContext) -> list[FeedbackItem]: ...
```

- **v1:** `RuleBasedFeedback` — hand-written thresholds ("if mean cents deviation > 20 → you're sharp").
- **Later:** `MLFeedback` — a trained model that takes the same `ComparisonResult` and returns the same `FeedbackItem`s.
- The UI chooses a provider from config. **No other code changes.** This is the
  *Strategy pattern*, and it's what makes the app "ML-ready" today.

`PlayerContext` (instrument, skill level, piece) is captured from day one even though
rules barely use it — ML will need it, and you can't retroactively collect it.

---

## 3. The audio science (what each feature means)

| Dimension | What the musician hears | What we compute | librosa function |
|---|---|---|---|
| **Pitch** | In tune / sharp / flat | Fundamental frequency f0 per frame → difference in **cents** (100 cents = 1 semitone) | `librosa.pyin` |
| **Dynamics** | Loud/soft, crescendos, accents | RMS energy → decibels; compare the *shape* of the loudness curve | `librosa.feature.rms`, `librosa.amplitude_to_db` |
| **Tone** | Bright/dark, airy/focused, thin/full | Spectral centroid (brightness), MFCCs (timbre "fingerprint"), spectral flatness (noisiness/breathiness) | `librosa.feature.spectral_centroid`, `.mfcc`, `.spectral_flatness` |
| **Timing** (bonus) | Rushing / dragging | Slope of the DTW warp path; onset comparison | `librosa.sequence.dtw`, `librosa.onset.onset_detect` |

Important subtleties we'll handle:
- **Cents, not Hz.** 10 Hz off at A2 (110 Hz) is terrible; 10 Hz off at A6 (1760 Hz) is barely noticeable. Cents = `1200 * log2(f_user / f_ref)` is perceptually fair.
- **Relative dynamics.** Different mics/rooms give different absolute levels. We normalize, then compare the *contour* (did you crescendo where the reference did?), not raw dB.
- **Align on chroma, compare on everything.** Chroma (pitch-class energy) is robust to tone/loudness differences, so it's the best signal for DTW alignment.
- **Octave errors.** Pitch trackers sometimes jump an octave; we'll detect and fold those.

---

## 4. Dev tools

| Tool | Purpose | Why this one |
|---|---|---|
| **Python 3.12** | Language | Already installed; supported by librosa |
| **uv** | Virtual envs + dependency management | Very fast, one tool replaces pip/venv/pip-tools, produces a lockfile (`uv.lock`) for reproducible installs |
| **pyproject.toml** | Single config file for deps and tools | Modern standard (replaces setup.py/requirements.txt) |
| **Git + GitHub** | Version control, backup, portfolio | Industry standard |
| **Ruff** | Linter + formatter | Replaces flake8/black/isort; extremely fast |
| **mypy** | Static type checking | Catches shape/type bugs between layers early |
| **pytest** | Testing | Simple, powerful fixtures |
| **pre-commit** | Runs ruff/mypy automatically before each commit | Keeps code clean without remembering to |
| **GitHub Actions** | CI: lint + test on every push | Free for public repos; catches "works on my machine" |
| **Jupyter** (dev only) | Explore audio interactively | Best way to *see* what features look like before coding them |
| **FFmpeg** | Decode MP3/M4A uploads | `soundfile` handles WAV/FLAC/OGG; FFmpeg covers the rest |
| **VS Code** + Python, Ruff extensions | Editor | Good debugger, notebook support |

### Runtime dependencies
```
streamlit, librosa, numpy, scipy, matplotlib, soundfile
```
### Dev dependencies
```
pytest, pytest-cov, ruff, mypy, pre-commit, jupyter
```
### Future ML dependencies (Phase 7+)
```
scikit-learn, pandas, pyarrow, joblib   (later maybe: torch, a pretrained audio model)
```

### ⚠️ Environment note: OneDrive
This folder is inside **OneDrive**. OneDrive tries to sync the thousands of files in a
`.venv/` folder, which is slow and can corrupt environments. Options:
1. **Recommended:** move the project to e.g. `C:\dev\MusicComparisonApp` (GitHub becomes your backup).
2. Or keep it here and tell uv to put the venv elsewhere (`UV_PROJECT_ENVIRONMENT`).

Also note the folder name has a typo (`Comparision`) — moving is a good chance to fix it.

---

## 5. Build phases & learning goals

Each phase ends with something that **runs** and a commit to GitHub.

### Phase 0 — Project setup
- Install uv and FFmpeg; create GitHub repo; `git init`; `pyproject.toml`; folder skeleton; ruff/pytest/pre-commit; CI workflow.
- ✅ Done when: `uv run pytest` passes (with a trivial test) and CI goes green on GitHub.
- 🎓 **Learn:** virtual environments, why lockfiles exist, git basics (commit, branch, push), what CI is.

### Phase 1 — Audio I/O and a hello-world UI
- `audio/loader.py`: load any file → mono float32 at a fixed sample rate (22,050 Hz).
- `audio/preprocess.py`: trim leading/trailing silence, peak/loudness normalize.
- Streamlit page: two file uploaders, audio players, waveform plots.
- ✅ Done when: you upload two files and see both waveforms.
- 🎓 **Learn:** sample rate & Nyquist, what a waveform array actually is, mono vs stereo, Streamlit's rerun model and `st.session_state`.

### Phase 2 — Feature extraction (in a notebook first)
- Explore pitch/RMS/centroid in `notebooks/01_features.ipynb` on real recordings.
- Move the code into `features/pitch.py`, `dynamics.py`, `tone.py`.
- Tests using **synthetic audio**: a generated 440 Hz sine must detect as A4; a 450 Hz sine must read ≈ +39 cents.
- ✅ Done when: tests pass and the app shows the pitch curve and loudness curve of each recording.
- 🎓 **Learn:** the STFT and spectrograms, frames & hop length, pYIN pitch tracking, decibels, MFCCs, testing with known ground truth.

### Phase 3 — Alignment (the hard, fun part)
- `analysis/alignment.py`: chroma features → `librosa.sequence.dtw` → warping path.
- Visualize the DTW cost matrix and path (great for intuition).
- Tests: a reference and a time-stretched copy should align along a predictable line.
- ✅ Done when: the app shows both pitch curves overlaid **in sync**, even if you played slower.
- 🎓 **Learn:** dynamic programming, DTW, why chroma is tempo/timbre robust, `librosa.effects.time_stretch`.

### Phase 4 — Comparison, scoring, problem regions
- `analysis/compare.py`: per-frame cents error, dB contour error, centroid/MFCC distance along the warp path.
- `analysis/segments.py`: smooth errors (median filter via scipy), threshold, merge nearby frames into `ProblemRegion`s.
- `analysis/scoring.py`: map errors to 0–100 per dimension.
- ✅ Done when: the app lists "Pitch: 82/100, Dynamics: 67/100, Tone: 90/100" plus problem time ranges.
- 🎓 **Learn:** signal smoothing, robust statistics (median vs mean), designing metrics that match human perception.

### Phase 5 — Visualizations
- Pitch overlay with shaded sharp/flat regions; loudness contour overlay; spectral centroid comparison; side-by-side spectrograms; an overview timeline with color-coded problem regions.
- Click/select a region → play just that snippet of each recording.
- ✅ Done when: a musician can *see* where they went wrong without reading numbers.
- 🎓 **Learn:** matplotlib's object-oriented API (`fig, ax`), designing readable charts, caching in Streamlit (`@st.cache_data`).

### Phase 6 — Rule-based feedback
- `feedback/base.py` (Protocol) + `feedback/rules.py`.
- Rules per dimension and per instrument, e.g.:
  - consistently sharp across the piece → "check tuning before you start"
  - sharp only on loud notes → "you may be over-blowing / pressing too hard on crescendos"
  - flat dynamic range vs reference → "exaggerate your dynamics; your ff and pp are only 6 dB apart vs 20 dB in the reference"
  - higher spectral flatness → "tone is breathier/noisier than reference; focus airstream / bow contact"
- ✅ Done when: the app shows a ranked list of specific, actionable tips tied to time ranges.
- 🎓 **Learn:** the Strategy pattern, Python Protocols, turning data into human-friendly language.

### Phase 7 — Data collection (ML groundwork)
- `data/store.py`: on each analysis, save features + comparison + feedback + `PlayerContext` (SQLite for metadata, Parquet/NPZ for arrays).
- Add a 👍/👎 "Was this tip helpful?" button per feedback item → **labels for training**.
- History page: track scores over time.
- ✅ Done when: every session is saved and you can see your progress chart.
- 🎓 **Learn:** data schemas, why labeled data is the bottleneck in ML, privacy (ask consent before storing recordings).

### Phase 8 — Machine learning (future)
Progressive options, simplest first:
1. **Feedback ranking:** train a classifier (scikit-learn) on 👍/👎 data to predict which tips a user finds helpful → reorder tips.
2. **Error-type classification:** from region features, predict cause ("intonation drift", "unstable vibrato", "weak attack") instead of hand-written rules.
3. **Pretrained audio embeddings** (e.g. a model like CREPE for pitch or a general audio encoder for timbre) for richer tone comparison.
4. **Personalization:** use a user's history to tailor advice to their recurring weaknesses.
5. **LLM-written feedback:** pass the structured `ComparisonResult` to an LLM to phrase advice conversationally.

Because of Phase 6's `FeedbackProvider` interface and Phase 7's data, each of these is a drop-in.
- 🎓 **Learn:** train/test splits, overfitting, evaluation metrics, model versioning.

### Phase 9 — Polish & deploy
- Error handling (silent files, very different pieces, huge uploads), loading spinners, README with screenshots.
- Deploy free on **Streamlit Community Cloud** (connects straight to your GitHub repo).

---

## 6. Testing strategy

- **Synthetic fixtures** (in `conftest.py`): generate sine tones, chirps, and amplitude ramps with numpy where the "right answer" is known exactly. This is how you test DSP code with confidence.
- **Unit tests** per feature module; **integration test** for `pipeline.compare()` on a short real clip pair.
- **No tests for Streamlit layout** — the UI is thin on purpose.
- Target: core package ≥ 80% coverage (`pytest --cov`).

## 7. Config & tunables (all in `config.py`)

```python
SAMPLE_RATE = 22_050
HOP_LENGTH = 512          # ~23 ms per frame
FMIN, FMAX = "C2", "C7"   # pitch search range (instrument-specific later)
PITCH_TOLERANCE_CENTS = 15
DYNAMICS_TOLERANCE_DB = 4
MIN_REGION_SECONDS = 0.25
```
One place to tune means no "magic numbers" scattered through the code.

## 8. Risks & open questions

| Risk | Mitigation |
|---|---|
| Polyphonic audio (piano chords) breaks pYIN | v1 targets **monophonic** instruments/voice; document it; consider chroma-based comparison for polyphonic later |
| Background noise / bad mics | Silence trimming, voiced-probability threshold, warn on low SNR |
| Recordings of different pieces | Check DTW alignment cost; warn "these don't look like the same piece" |
| Streamlit reruns make analysis slow | `@st.cache_data` keyed on file hash |
| Large uploads | Limit in `.streamlit/config.toml`; cap duration (e.g. 5 min) |

**Questions for you:**
1. Which instrument(s) do you play / want to target first? (Affects pitch range and feedback wording.)
2. Will the reference usually be a professional recording, a teacher, or a previous take of yours?
3. Public GitHub repo (portfolio + free CI) or private?

---

## 9. How we'll work together (learning mode)

- I'll build in **small steps, one phase at a time**, and explain each file before/after writing it.
- At the end of each phase I'll suggest a **small exercise** for you to do yourself (e.g. "add spectral rolloff to `tone.py` and write its test").
- Notebooks are your playground — break things there, not in `src/`.
- Commit after every working step with a clear message so you can see the project's history grow.

## 10. Immediate next steps

1. Decide on the OneDrive question (move folder or not).
2. Create an **empty** GitHub repo (no README/license, so it doesn't conflict) named `music-comparison-app`, and send me the URL.
3. Install uv: `powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"`
4. Install FFmpeg: `winget install Gyan.FFmpeg`
5. Then we start Phase 0 together.
