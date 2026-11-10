"""Central place for every tunable number in the app.

Keeping these here (instead of scattering "magic numbers" through the code) means
you can tune behaviour in one spot, and tests can refer to the same values.
"""

from typing import Final

# --- Audio ---------------------------------------------------------------
# Every clip is resampled to this rate (samples per second) after loading.
# 22,050 Hz captures frequencies up to ~11 kHz (the Nyquist limit = SR / 2),
# which covers the fundamentals and main overtones of most instruments.
SAMPLE_RATE: Final = 22_050

# Number of samples between successive analysis frames.
# 512 / 22,050 Hz ≈ 23 ms per frame.
HOP_LENGTH: Final = 512

# Window length for each analysis frame (samples). 2048 / 22,050 Hz ≈ 93 ms:
# long enough to contain several cycles of the lowest note we track (C2 ≈ 65 Hz).
FRAME_LENGTH: Final = 2048

# Usable recording length, after leading/trailing silence is trimmed.
# Shorter than ~1 s there isn't enough audio to measure habits, align, or
# judge tone (and the FFT windows are longer than the clip itself). Longer
# than 5 minutes, analysis takes over half a minute per clip.
MIN_DURATION_S: Final = 1.0
MAX_DURATION_S: Final = 300.0

# --- Dynamics ------------------------------------------------------------
# Quietest loudness we report. Anything below is treated as silence.
SILENCE_DB: Final = -80.0

# --- Tone ----------------------------------------------------------------
N_MFCC: Final = 13

# Background-noise removal for brightness (see features/tone.py). At each
# frequency, this percentile of its level over time estimates the noise, and
# anything under NOISE_FLOOR_MARGIN times that is ignored. For white noise the
# 25th percentile is ~0.76x its typical level, so a margin of 4 removes
# ~99% of noise-only values while notes, far louder, are kept.
NOISE_FLOOR_PERCENTILE: Final = 25.0
NOISE_FLOOR_MARGIN: Final = 4.0

# --- Alignment -----------------------------------------------------------
# Frames quieter than this get a neutral chroma vector, so background noise in
# rests isn't mistaken for notes.
CHROMA_SILENCE_DB: Final = -50.0

# DTW memory budget: max cells in its cost grid (ref frames x user frames).
# 16M cells ≈ 128 MB per float64 matrix; clips up to ~90 s align at full detail.
MAX_DTW_CELLS: Final = 16_000_000

# Extra cost for a DTW step where only one recording moves forward. Without it,
# DTW has no reason to prefer the diagonal inside a held note (every frame of
# the note looks the same), so even identical recordings wander off it.
DTW_STEP_PENALTY: Final = 0.01

# Width (s) of the moving average applied to the mapping. Inside a held note
# DTW only knows where the note starts and ends; averaging spreads the timing
# evenly across the note. ~0.35 s keeps note middles within ~3 frames and note
# starts within ~2 frames on synthetic tests (legato and detached).
ALIGNMENT_SMOOTHING_S: Final = 0.35

# Mean chroma distance along the path above which the recordings probably
# aren't the same piece (0 = identical, 1 = unrelated).
ALIGNMENT_COST_WARNING: Final = 0.35

# Unmatched material (a count-in, a repeat, a skipped passage) makes the DTW
# path run straight along one recording while the other stands still. Runs at
# least this long, whose notes differ this much (mean chroma distance), are
# reported. On synthetic tests, lingering on a held note costs ~0 and truly
# unmatched notes 0.48-0.92, so 0.3 separates them with margin.
UNMATCHED_MIN_S: Final = 0.5
UNMATCHED_MIN_COST: Final = 0.3

# --- Pitch ---------------------------------------------------------------
# Search range for the pitch tracker, as note names (librosa understands these).
FMIN: Final = "C2"
FMAX: Final = "C7"

# --- Comparison ----------------------------------------------------------
# Frames quieter than this count as silence when comparing loudness and tone.
SOUNDING_DB: Final = -50.0

# Width (s) of the median filter applied to every difference curve before
# scoring. Longer than one vibrato cycle (~0.2 s at 5 Hz), so two players whose
# vibratos wobble out of step aren't marked out of tune, and long enough to
# ignore the 1-2 frame glitches alignment leaves where notes change.
COMPARISON_SMOOTHING_S: Final = 0.25

# Differences within these tolerances count as perfect...
PITCH_TOLERANCE_CENTS: Final = 15.0  # 100 cents = 1 semitone
DYNAMICS_TOLERANCE_DB: Final = 4.0
TONE_TOLERANCE_OCTAVES: Final = 0.25  # brightness: 0.25 octave ≈ 19% brighter/darker

# ...and differences this large (or larger) score 0, with a straight line between.
PITCH_ZERO_SCORE_CENTS: Final = 50.0  # a quarter-tone: clearly out of tune
DYNAMICS_ZERO_SCORE_DB: Final = 12.0  # roughly two dynamic levels (e.g. mf vs pp)
TONE_ZERO_SCORE_OCTAVES: Final = 1.0  # spectral centroid doubled or halved

# A reference note with no clear note from you counts as "missing" only when
# that lasts this long: pitch trackers drop out for a moment at note changes.
MISSING_MIN_S: Final = 0.5

MIN_REGION_SECONDS: Final = 0.25  # ignore problem regions shorter than this
REGION_MERGE_GAP_S: Final = 0.2  # join same-direction problems closer than this

# --- Feedback --------------------------------------------------------------
# A whole-take habit needs at least this many frames (~1 s) of evidence.
MIN_HABIT_FRAMES: Final = 40

# "Out of tune throughout" needs this share of the pitched frames off in the
# same direction (by more than half the tolerance), so one bad passage
# doesn't get reported as a tuning problem.
TUNING_HABIT_FRACTION: Final = 0.75

# "Pitch changes with loudness" compares your loudest third of notes with
# your softest third, but only when those are at least this many dB apart.
LOUD_SOFT_MIN_SPREAD_DB: Final = 6.0

# Dynamic range = loudness percentile 90 minus percentile 10 (the extremes,
# ignoring the odd spike), over the frames where you're playing.
DYNAMIC_RANGE_PERCENTILES: Final = (10.0, 90.0)

# Spectral flatness spans orders of magnitude (each 10 dB more noise is ~10x
# flatter), so it's compared in dB. Clean synthetic audio reaches -100 dB,
# which no real recording does, so values are floored here: below it, the
# noise is far too quiet to hear.
FLATNESS_FLOOR_DB: Final = -40.0
NOISE_TOLERANCE_DB: Final = 10.0  # this much noisier than the reference is flagged

# How much each dimension counts when ranking advice: wrong notes stand out
# most, and the tone measurement is the least reliable of the three.
DIMENSION_WEIGHTS: Final = {"pitch": 1.0, "dynamics": 0.8, "tone": 0.6}
# How sure we are a measurement is real (1 = certain).
DIMENSION_CONFIDENCE: Final = {"pitch": 0.9, "dynamics": 0.8, "tone": 0.6}
# Pickups, tone knobs, and amps change an electric guitar's brightness far more
# than playing does, so tone advice for it is much less certain.
ELECTRIC_TONE_CONFIDENCE: Final = 0.3
# Commercial recordings are usually compressed (their dynamics squeezed), so
# "your dynamics are wider than the reference" is often the mastering, not you.
DYNAMICS_WIDER_CONFIDENCE: Final = 0.4
# A noisy room or microphone looks exactly like a noisy tone.
NOISE_CONFIDENCE: Final = 0.5

# --- Playback --------------------------------------------------------------
# Extra audio (s) played before and after a problem spot, so your ear hears
# it in context instead of starting mid-note.
SNIPPET_PAD_S: Final = 0.5

# --- Storage ---------------------------------------------------------------
# Where practice sessions are saved: a SQLite database plus one NPZ file of
# measurements per session (and, only if you opt in, the recordings).
# Set the environment variable to store them somewhere else (tests use this
# to write into a throwaway folder). The default is relative to the folder you
# start the app from; /data/ is in .gitignore so it never reaches GitHub.
DATA_DIR_ENV: Final = "MUSICCOMPARE_DATA_DIR"
DEFAULT_DATA_DIR: Final = "data"
