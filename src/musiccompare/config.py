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

# --- Dynamics ------------------------------------------------------------
# Quietest loudness we report. Anything below is treated as silence.
SILENCE_DB: Final = -80.0

# --- Tone ----------------------------------------------------------------
N_MFCC: Final = 13

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

MIN_REGION_SECONDS: Final = 0.25  # ignore problem regions shorter than this
REGION_MERGE_GAP_S: Final = 0.2  # join same-direction problems closer than this
