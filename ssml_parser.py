"""SSML parser for TTS prosody control."""

import re
import math
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field


@dataclass
class Prosody:
    """Prosody parameters for TTS."""
    rate: float = 1.0
    pitch_st: float = 0.0
    volume: float = 1.0
    is_break: bool = False
    break_ms: int = 0


@dataclass
class VoiceSpec:
    """Voice specification from SSML."""
    name: str = "default"
    emotion: str = "neutral"


@dataclass
class Segment:
    """A parsed SSML segment."""
    text: str
    voice: VoiceSpec = field(default_factory=VoiceSpec)
    prosody: Prosody = field(default_factory=Prosody)


# ── Parsing Helpers ───────────────────────────────────────────
_RATE_KEYWORDS = {"x-slow": 0.6, "slow": 0.8, "medium": 1.0, "fast": 1.25, "x-fast": 1.5}
_PITCH_KEYWORDS = {"x-low": -4, "low": -2, "medium": 0, "high": 2, "x-high": 4}
_VOLUME_KEYWORDS = {"silent": 0, "x-soft": 0.25, "soft": 0.5, "medium": 1.0, "loud": 1.5, "x-loud": 2.0}
_BREAK_STRENGTHS = {"none": 0, "x-weak": 50, "weak": 100, "medium": 200, "strong": 400, "x-strong": 750}


def _parse_rate(value):
    if not value:
        return 1.0
    s = value.strip().lower()
    if s in _RATE_KEYWORDS:
        return _RATE_KEYWORDS[s]
    m = re.match(r"([\d.]+)%$", s)
    return float(m.group(1)) / 100 if m else 1.0


def _parse_pitch(value):
    if not value:
        return 0.0
    s = value.strip().lower()
    if s in _PITCH_KEYWORDS:
        return float(_PITCH_KEYWORDS[s])
    m = re.match(r"([+-]?[\d.]+)st$", s)
    if m:
        return float(m.group(1))
    m = re.match(r"([\d.]+)%$", s)
    if m:
        p = float(m.group(1)) / 100
        return 12 * math.log2(p) if p > 0 else 0
    return 0.0


def _parse_volume(value):
    if not value:
        return 1.0
    s = value.strip().lower()
    if s in _VOLUME_KEYWORDS:
        return _VOLUME_KEYWORDS[s]
    m = re.match(r"([+-]?[\d.]+)db$", s)
    if m:
        return 10 ** (float(m.group(1)) / 20)
    m = re.match(r"([\d.]+)%$", s)
    return float(m.group(1)) / 100 if m else 1.0


def _parse_time(value):
    if not value:
        return 0
    m = re.match(r"([\d.]+)ms$", value.strip().lower())
    if m:
        return int(float(m.group(1)))
    m = re.match(r"([\d.]+)s$", value.strip().lower())
    return int(float(m.group(1)) * 1000) if m else 0


# ── Main Parser ───────────────────────────────────────────────
def parse_ssml(text):
    """Parse SSML text into segments.
    
    Returns:
        List of (text, VoiceSpec, Prosody) tuples
    """
    s = text.strip()
    if not s:
        return []
    
    try:
        if not s.startswith("<"):
            s = f"<speak>{s}</speak>"
        root = ET.fromstring(s)
    except ET.ParseError:
        return [(text, VoiceSpec(), Prosody())]

    segments = []

    def _recurse(node, voice, pros):
        tag = node.tag.lower()
        
        # Handle break elements
        if "break" in tag:
            ms = _parse_time(node.attrib.get("time", "0ms"))
            if not ms:
                strength = node.attrib.get("strength", "").lower()
                ms = _BREAK_STRENGTHS.get(strength, 200)
            segments.append(("", voice, Prosody(is_break=True, break_ms=ms)))
            return

        # Handle voice elements
        new_voice = voice
        if "voice" in tag:
            new_voice = VoiceSpec(
                name=node.attrib.get("name", "default"),
                emotion=node.attrib.get("emotion", "neutral"),
            )

        # Handle prosody elements
        new_pros = pros
        if "prosody" in tag:
            new_pros = Prosody(
                rate=pros.rate * _parse_rate(node.attrib.get("rate", "")),
                pitch_st=pros.pitch_st + _parse_pitch(node.attrib.get("pitch", "")),
                volume=pros.volume * _parse_volume(node.attrib.get("volume", "")),
            )

        # Add text content
        if node.text and node.text.strip():
            segments.append((node.text.strip(), new_voice, new_pros))

        # Recurse into children
        for child in node:
            _recurse(child, new_voice, new_pros)
            if child.tail and child.tail.strip():
                segments.append((child.tail.strip(), new_voice, new_pros))

    _recurse(root, VoiceSpec(), Prosody())
    return [(t, v, p) for t, v, p in segments if t or p.is_break]


def wrap_with_prosody(text, pitch_pct, speed_pct, volume_pct):
    """Wrap text with prosody SSML tags based on slider values."""
    st = 12 * math.log2(pitch_pct) if pitch_pct > 0 else 0
    return (
        f'<prosody pitch="{st:+.1f}st" '
        f'rate="{speed_pct * 100:.0f}%" volume="{volume_pct * 100:.0f}%">'
        f'{text}</prosody>'
    )