"""Turn a clip back into a sound file, e.g. for an in-browser audio player."""

import io

import soundfile as sf

from musiccompare.models import AudioClip


def to_wav_bytes(clip: AudioClip) -> bytes:
    """The clip as a 16-bit WAV file, in memory."""
    buffer = io.BytesIO()
    sf.write(buffer, clip.y, clip.sr, format="WAV", subtype="PCM_16")
    return buffer.getvalue()
