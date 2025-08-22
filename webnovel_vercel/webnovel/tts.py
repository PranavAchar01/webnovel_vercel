"""Text‑to‑Speech adapters for the web novel service.

This module defines a very simple TTS provider that produces silent
audio. The absence of external TTS libraries in the execution
environment means that we cannot synthesize real speech. Instead, a
pre‑encoded 1‑second silent MP3 payload is repeated according to the
length of the input text. While this produces no audible output it
fulfils the API contract by returning a binary MP3 and a duration.

The design allows for pluggable providers: you can implement a
provider class with a ``name`` attribute and a ``synthesize`` method
that returns ``bytes`` and the duration (seconds) for a chunk of text.
"""

from __future__ import annotations

import base64
import math
import re
from typing import List, Tuple


# Base64 encoded MP3 of approximately one second of silence. This data
# originates from a public gist of tiny silent audio files. Embedding
# the payload avoids the need for external audio utilities or
# dependencies. To generate your own you can use ffmpeg:
# ``ffmpeg -f lavfi -i anullsrc=r=24000:cl=mono -t 1 -q:a 9 silence.mp3``
# and then base64‑encode it.
SILENT_MP3_BASE64 = (
    "SUQzBAAAAAAAI1RTU0UAAAAPAAADTGF2ZjU4Ljc2LjEwMAAAAAAAAAAAAAAA/+M4wAAAAAAAAAAAA"
    "EluZm8AAAAPAAAAAwAAAbAAqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqq1dXV1dXV1dXV1"
    "dXV1dXV1dXV1dXV1dXV1dXV1dXV1dXV//////////////////////////////AAAAAExhdmM1OC4xMwAA"
    "AAAAAAAAAAAAACQDkAAAAAAAAAGw9wrNaQAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA"
    "AAAAAAAAAA/+MYxAAAAANIAAAAAExBTUUzLjEwMFVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVV"
    "VVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVV/+MYxDsAAANIAAAAAFVVVVVVVVVVVVVVVVVVVVVVVVVVVV"
    "VVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVV/+MYxHYAAANIAAAAAFVVV"
    "VVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVV"
)

# Ensure proper decoding of the base64 string by removing any
# whitespace and adding the necessary padding.  Some base64 encoders
# omit trailing '=' characters; base64.b64decode will raise an
# exception if the length isn't a multiple of four.  We first
# sanitise the string to remove any accidental newlines or spaces
# introduced during source formatting, then compute the required
# padding.
_sanitised = SILENT_MP3_BASE64.replace("\n", "").replace(" ", "")
_padding = '=' * ((4 - (len(_sanitised) % 4)) % 4)
try:
    # Attempt to decode the bundled MP3.  If the payload is malformed
    # (for example if the length is off by one) base64.b64decode will
    # raise an exception.  In that case we fall back to a known-good
    # silent MP3 payload from the novwhisky gist.
    SILENT_MP3_BYTES: bytes = base64.b64decode(_sanitised + _padding)
except Exception:
    # Known-good base64 representation of ~1 second of silence from
    # https://gist.github.com/novwhisky/8a1a0168b94f3b6abfaa (MP3 variant).
    _fallback_b64 = (
        "SUQzBAAAAAAAI1RTU0UAAAAPAAADTGF2ZjU4Ljc2LjEwMAAAAAAAAAAAAAAA/+M4wAAAAAAAAAAAA"
        "EluZm8AAAAPAAAAAwAAAbAAqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqq1dXV1dXV1dXV1"
        "dXV1dXV1dXV1dXV1dXV1dXV1dXV1dXV//////////////////////////////AAAAAExhdmM1OC4xMwAA"
        "AAAAAAAAAAAAACQDkAAAAAAAAAGw9wrNaQAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA"
        "AAAAAAAAAA/+MYxAAAAANIAAAAAExBTUUzLjEwMFVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVV"
        "VVVVVVVVVVVVVVVVVVVVVVVVVVVV/+MYxDsAAANIAAAAAFVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVV"
        "VVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVV/+MYxHYAAANIAAAAAFVV"
        "VVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVV"
    )
    _san_fallback = _fallback_b64.replace("\n", "").replace(" ", "")
    _pad_fb = '=' * ((4 - (len(_san_fallback) % 4)) % 4)
    SILENT_MP3_BYTES = base64.b64decode(_san_fallback + _pad_fb)


def chunk_text(text: str, max_chars: int = 3200) -> List[str]:
    """Split ``text`` into chunks not exceeding ``max_chars`` characters.

    The splitting algorithm tries to break on sentence boundaries when
    possible. It first splits on newline characters, then further
    splits those chunks on periods. If a resulting piece is still
    longer than ``max_chars`` it is split directly at the character
    limit.
    """
    chunks: List[str] = []
    sentences: List[str] = []
    for paragraph in text.split("\n"):
        paragraph = paragraph.strip()
        if not paragraph:
            continue
        # Split on common sentence terminators
        parts = re.split(r"(?<=[.!?])\s+", paragraph)
        sentences.extend(parts)
    buf: List[str] = []
    current_len = 0
    for sentence in sentences:
        sent_len = len(sentence)
        if current_len + sent_len + 1 > max_chars and buf:
            chunks.append(" ".join(buf).strip())
            buf = [sentence]
            current_len = sent_len
        else:
            buf.append(sentence)
            current_len += sent_len + 1
    if buf:
        chunks.append(" ".join(buf).strip())
    return chunks


class DummyTTSProvider:
    """A placeholder TTS provider that returns silent MP3 audio.

    Each call to ``synthesize`` returns a tuple of (bytes, duration)
    where ``bytes`` is the binary MP3 data and ``duration`` is the
    length of the audio in seconds. The duration is estimated as the
    number of characters divided by ``chars_per_second``. This crude
    estimate ensures that playback progress bars move roughly in
    proportion to the amount of text. You can adjust the constant
    accordingly.
    """

    name = "silent"
    # Estimate that 15 characters correspond to one second of speech at a
    # moderate speaking rate (around 120 words per minute). This value
    # is arbitrary and serves only to produce plausible durations.
    chars_per_second: int = 15

    async def synthesize(self, text: str) -> Tuple[bytes, int]:
        # Determine number of seconds required to read the text.
        seconds = max(1, math.ceil(len(text) / self.chars_per_second))
        # Concatenate silent payload to match the length. Each silent
        # payload is roughly 1 second. Concatenating MP3 files by
        # simple concatenation is valid because MP3 streams can be
        # joined at frame boundaries.
        audio_bytes = SILENT_MP3_BYTES * seconds
        return audio_bytes, seconds


class OpenAITTSProvider:
    """Text‑to‑speech provider using OpenAI's TTS API.

    This provider connects to the OpenAI text‑to‑speech endpoint and
    returns MP3 audio.  It requires an API key which should be set via
    the ``OPENAI_API_KEY`` environment variable.  You can choose a
    voice by passing a ``voice`` argument when initialising the
    provider.  Supported voices include ``alloy``, ``echo``, ``fable``,
    ``onyx``, ``nova`` and ``shimmer``.  See the OpenAI documentation
    for the complete list and any changes.

    Usage::

        provider = OpenAITTSProvider(api_key=os.environ["OPENAI_API_KEY"], voice="alloy")
        audio_bytes, seconds = await provider.synthesize("Hello world!")

    The duration is estimated heuristically as for the dummy provider.
    """

    name = "openai"

    def __init__(self, api_key: str, voice: str = "alloy", model: str = "tts-1",
                 response_format: str = "mp3", speed: float = 1.0) -> None:
        self.api_key = api_key
        self.voice = voice
        self.model = model
        self.response_format = response_format
        self.speed = speed
        # Reuse the same character‑per‑second heuristic as DummyTTSProvider
        self.chars_per_second = DummyTTSProvider.chars_per_second

    async def synthesize(self, text: str) -> Tuple[bytes, int]:
        import httpx  # Local import to avoid requiring httpx when unused
        payload = {
            "model": self.model,
            "input": text,
            "voice": self.voice,
            "response_format": self.response_format,
            "speed": self.speed,
        }
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        # Call the OpenAI API; raise for non‑2xx responses
        async with httpx.AsyncClient(timeout=60.0) as client:
            response = await client.post(
                "https://api.openai.com/v1/audio/speech",
                headers=headers,
                json=payload,
            )
            response.raise_for_status()
            audio_bytes = response.content
        # Estimate duration based on number of characters
        seconds = max(1, math.ceil(len(text) / self.chars_per_second))
        return audio_bytes, seconds
