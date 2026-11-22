"""Voice and emotion profile definitions."""

# ═══════════════════════════════════════════════════════════════
# VOICE PROFILES
# ═══════════════════════════════════════════════════════════════
VOICES = {
    "default": {"f0": 125, "fscale": 1.0, "bright": 1.0, "breath": 0.05},
    "male":    {"f0": 110, "fscale": 0.92, "bright": 0.85, "breath": 0.06},
    "female":  {"f0": 220, "fscale": 1.08, "bright": 1.2, "breath": 0.12},
    "robot":   {"f0": 140, "fscale": 1.0, "bright": 1.6, "breath": 0.0},
    "child":   {"f0": 300, "fscale": 1.15, "bright": 1.4, "breath": 0.15},
    "deep":    {"f0": 85, "fscale": 0.85, "bright": 0.8, "breath": 0.03},
    "bright":  {"f0": 160, "fscale": 1.1, "bright": 1.3, "breath": 0.08},
}

# ═══════════════════════════════════════════════════════════════
# EMOTION PROFILES
# ═══════════════════════════════════════════════════════════════
EMOTIONS = {
    "neutral": {"ps": 1.0, "vol": 1.0, "spd": 1.0, "bright": 1.0,
                "pause": 1.0, "stress": 1.0, "breath": 1.0},
    "happy":   {"ps": 1.2, "vol": 1.15, "spd": 1.1, "bright": 1.2,
                "pause": 0.8, "stress": 1.25, "breath": 1.1},
    "sad":     {"ps": 0.85, "vol": 0.85, "spd": 0.85, "bright": 0.85,
                "pause": 1.4, "stress": 0.8, "breath": 1.3},
    "angry":   {"ps": 1.1, "vol": 1.3, "spd": 1.2, "bright": 1.4,
                "pause": 0.6, "stress": 1.5, "breath": 0.8},
    "calm":    {"ps": 0.95, "vol": 0.9, "spd": 0.9, "bright": 0.95,
                "pause": 1.2, "stress": 0.9, "breath": 1.1},
}


def get_voice(name):
    """Get voice profile by name."""
    return VOICES.get(name, VOICES["default"])


def get_emotion(name):
    """Get emotion profile by name."""
    return EMOTIONS.get(name, EMOTIONS["neutral"])


def get_voice_info(name):
    """Get formatted voice info string."""
    v = get_voice(name)
    return f"F0: {v['f0']}Hz | Scale: {v['fscale']:.2f} | Bright: {v['bright']:.2f}"


def get_emotion_info(name):
    """Get formatted emotion info string."""
    e = get_emotion(name)
    return f"Pitch: {e['ps']:.2f}x | Vol: {e['vol']:.2f}x | Speed: {e['spd']:.2f}x"