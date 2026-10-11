"""Configuration constants for the FFT TTS Engine."""

import numpy as np
from scipy.signal import lfilter, butter, resample

try:
    import sounddevice as sd
    HAS_SD = True
except ImportError:
    HAS_SD = False

try:
    from g2p_en import G2p
    HAS_G2P = True
except ImportError:
    HAS_G2P = False

# ── Audio Settings ────────────────────────────────────────────
SR = 22050
EPS = 1e-9
BLOCKSIZE = 1024
WINDOW_DUR = 0.5
WINDOW_SIZE = int(SR * WINDOW_DUR)
SPEC_FFT = 512
SPEC_COLS = int(WINDOW_DUR * SR / 128)

# ── Default Settings ──────────────────────────────────────────
DEFAULTS = {
    'sample_rate': 22050,
    'blocksize': 1024,
    'spec_fft': 512,
    'wave_dur': 0.5,
    'default_f0': 125,
    'breathiness': 0.05,
}

# ── SSML Demo ─────────────────────────────────────────────────
DEMO_SSML = """<voice name="female" emotion="happy">
  <prosody pitch="+2st" rate="110%" volume="115%">
    Hello there! I am so glad you are listening.
  </prosody>
</voice>

<voice name="male" emotion="sad">
  <prosody pitch="-3st" rate="85%" volume="85%">
    I feel a little low today... but it will be okay.
  </prosody>
</voice>

<voice name="robot" emotion="angry">
  <prosody rate="105%" volume="130%">
    Warning! System overload detected.
  </prosody>
</voice>

<voice name="child" emotion="happy">
  <prosody pitch="+4st" rate="115%">
    Yay! This is so much fun!
  </prosody>
</voice>

<voice name="deep" emotion="calm">
  <prosody pitch="-2st" rate="90%">
    Take a deep breath. Everything is fine.
  </prosody>"""

# ── Presets ───────────────────────────────────────────────────
PRESETS = {
    "Welcome": '<voice name="female"><prosody pitch="+2st">Welcome to the FFT Text-to-Speech Engine!</prosody></voice>',
    "Story": '<voice name="deep" emotion="calm"><prosody rate="90%">Once upon a time, in a land far away, there lived a brave young hero.</prosody></voice>',
    "Robot": '<voice name="robot"><prosody rate="105%" volume="130%">Systems online. All diagnostics complete.</prosody></voice>',
    "News": '<voice name="male" emotion="neutral"><prosody rate="110%">Breaking news: Scientists have discovered a new form of renewable energy.</prosody></voice>',
    "Poem": '<voice name="female" emotion="calm"><prosody rate="85%" pitch="+1st">The wind whispers softly through the trees, carrying secrets on the gentle breeze.</prosody></voice>',
}