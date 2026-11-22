#!/usr/bin/env python3
"""
tts_fft_ssml_singlefile.py
Single-file, rule-based FFT TTS with SSML, voices, emotions — no G2P (maps characters to phoneme-like synthesis).

Dependencies:
  - numpy
  - (optional) sounddevice for playback: pip install sounddevice

Usage:
  - Run as script to synth demo SSML, writes demo_out.wav, optionally plays.
  - Import synthesize_ssml(ssml_string) to get numpy audio array for programmatic use.

Note: This is a demonstration/prototype. Tweak FORMANTS/PROFILES/MAPPINGS for different results.
"""

import math
import re
import wave
from dataclasses import dataclass
from typing import List, Tuple, Dict
import numpy as np

# ------------- Global audio parameters -------------
SR = 16000
EPS = 1e-9

# ------------- Formants (vowel-like targets) -------------
# Basic formant triplets (F1,F2,F3) and approximate bandwidths (Hz)
FORMANTS = {
    "A": ((730,80),(1090,90),(2440,120)),
    "E": ((530,70),(1840,90),(2480,120)),
    "I": ((270,60),(2290,100),(3010,150)),
    "O": ((570,80),( 840,80),(2410,120)),
    "U": ((300,60),( 870,80),(2240,120)),
    "AH":((600,80),(1040,90),(2250,130)),
    "ER":((500,80),(1500,100),(2500,140)),
    # approximant-ish and nasals
    "L": ((350,70),(1700,120),(2600,150)),
    "R": ((300,70),(1200,120),(2600,150)),
    "M": ((250,60),(1000,120),(2200,150)),
    "N": ((300,60),(1200,120),(2400,150)),
}

# ------------- Fricative bands (Hz) -------------
FRIC_BANDS = {
    "s": (4000, 8000),
    "sh": (2000, 5000),
    "f": (4000, 6500),
    "h": (1000, 4000),
    "th": (3000, 5500),
    "z": (4000, 8000),
}

# ------------- Voice timbre profiles -------------
VOICE_PROFILES = {
    "default": {"f0": 125.0, "formant_scale": 1.0, "brightness": 1.0, "breathiness": 0.05},
    "male":    {"f0": 110.0, "formant_scale": 0.92, "brightness": 0.85, "breathiness": 0.06},
    "female":  {"f0": 220.0, "formant_scale": 1.08, "brightness": 1.2,  "breathiness": 0.12},
    "robot":   {"f0": 140.0, "formant_scale": 1.0,  "brightness": 1.6,  "breathiness": 0.0},
    "child":   {"f0": 300.0, "formant_scale": 1.15, "brightness": 1.4,  "breathiness": 0.15},
}

# ------------- Emotion profiles (affect pitch, volume, rate, brightness, pauses, stress) -------------
EMOTION_PROFILES = {
    "neutral": {"pitch_scale":1.0, "volume":1.0, "speed":1.0, "brightness":1.0, "pause_factor":1.0, "stress":1.0},
    "happy":   {"pitch_scale":1.2, "volume":1.15,"speed":1.15,"brightness":1.25,"pause_factor":0.8,"stress":1.25},
    "sad":     {"pitch_scale":0.85,"volume":0.85,"speed":0.8, "brightness":0.85,"pause_factor":1.4,"stress":0.8},
    "angry":   {"pitch_scale":1.05,"volume":1.3, "speed":1.25,"brightness":1.5,"pause_factor":0.6,"stress":1.6},
    "calm":    {"pitch_scale":0.95,"volume":0.9, "speed":0.9, "brightness":0.95,"pause_factor":1.2,"stress":0.9},
}

# ------------- Utility: formant envelope & bandpass via FFT -------------
def gaussian_formant_envelope(freqs: np.ndarray, formants: Tuple[Tuple[float,float], ...], strength=1.0) -> np.ndarray:
    env = np.full_like(freqs, 1e-6, dtype=np.float64)
    for (fc, bw) in formants:
        sigma = bw / (2*np.sqrt(2*np.log(2)) + EPS)
        env += strength * np.exp(-0.5 * ((freqs - fc) / (sigma + EPS))**2)
    return env

def bandpass_mask(freqs: np.ndarray, f_lo: float, f_hi: float, edge=300.0) -> np.ndarray:
    # Soft raised-cosine edges
    env = np.zeros_like(freqs)
    lo1 = max(0.0, f_lo - edge)
    lo2 = f_lo + edge
    hi1 = max(0.0, f_hi - edge)
    hi2 = f_hi + edge
    # rise
    rise = np.clip((freqs - lo1) / (lo2 - lo1 + EPS), 0.0, 1.0)
    fall = np.clip((hi2 - freqs) / (hi2 - hi1 + EPS), 0.0, 1.0)
    env = rise * fall
    return env

# ------------- Excitations -------------
def impulse_train(duration: float, f0: float) -> np.ndarray:
    n = max(1, int(SR * duration))
    t = np.arange(n) / SR
    # band-limited saw-ish glottal excitation with tiny jitter
    phase = 2.0 * np.pi * (f0 * t + 0.001 * np.random.randn(n))
    saw = (np.pi - (phase % (2*np.pi)))
    exc = np.tanh(0.6 * saw)
    # gentle lowpass to reduce very high harmonics
    fft = np.fft.rfft(exc)
    freqs = np.fft.rfftfreq(len(exc), 1.0/SR)
    lp = 1.0 / (1.0 + (freqs / 5000.0)**4)
    exc = np.fft.irfft(fft * lp)
    return exc

def white_noise(duration: float, amp=0.3) -> np.ndarray:
    n = max(1, int(SR * duration))
    return np.random.randn(n) * amp

# ------------- Fade helper & overlap append -------------
def apply_fades(x: np.ndarray, fade_ms: int = 8) -> np.ndarray:
    if x.size == 0: return x
    n = len(x)
    m = max(1, int(SR * fade_ms / 1000.0))
    w = np.ones(n, dtype=np.float64)
    ramp = np.linspace(0.0, 1.0, m)
    w[:m] = ramp
    w[-m:] = ramp[::-1]
    y = x * w
    peak = np.max(np.abs(y)) + EPS
    return 0.95 * y / peak

def overlap_append(a: np.ndarray, b: np.ndarray, xf_ms=8) -> np.ndarray:
    if a.size == 0:
        return b.copy()
    m = max(1, int(SR * xf_ms / 1000.0))
    out = np.zeros(len(a) + len(b) - m, dtype=np.float64)
    out[:len(a)] = a
    fade = np.linspace(0.0, 1.0, m)
    out[len(a)-m:len(a)] = (1.0 - fade) * out[len(a)-m:len(a)] + fade * b[:m]
    out[len(a):] = b[m:]
    return out

# ------------- Intonation: emotion pitch curves -------------
def emotion_pitch_curve(emotion: str, t: np.ndarray, base_f0: float) -> np.ndarray:
    if t.size == 0:
        return np.array([], dtype=float)
    if emotion == "happy":
        # rising + small vibrato-like wiggle
        return base_f0 * (1.05 + 0.12 * (t / t.max()) + 0.03 * np.sin(2*np.pi*3*t))
    if emotion == "sad":
        # falling, flat
        return base_f0 * (0.98 - 0.18 * (t / t.max()))
    if emotion == "angry":
        # jittery, up/down spikes
        return base_f0 * (1.0 + 0.18 * np.sign(np.sin(2*np.pi*6*t)))
    if emotion == "calm":
        # gentle slow undulation
        return base_f0 * (1.0 + 0.03 * np.sin(2*np.pi*1.5*t))
    # neutral
    return base_f0 * np.ones_like(t)

# ----------------- Helper: match array lengths -----------------
def match_length(a: np.ndarray, b: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Trims the longer array to the shorter array length to avoid broadcasting errors.
    Returns two arrays of equal length.
    """
    n = min(len(a), len(b))
    return a[:n], b[:n]

# ------------- Synthesis primitives (voiced vowel-like, fricative, stop, nasal) -------------
def synthesize_voiced_formant(formant_key: str, duration: float, base_f0: float, voice_profile: dict, emotion_profile: dict) -> np.ndarray:
    # duration scaled by emotion speed
    duration = max(0.01, duration / max(1e-3, emotion_profile.get("speed", 1.0)))
    n = max(1, int(SR * duration))
    t = np.linspace(0.0, duration, n, False)
    
    # dynamic F0 curve
    base_f0_scaled = base_f0 * emotion_profile.get("pitch_scale", 1.0)
    f0_curve = emotion_pitch_curve(
        emotion_profile.get("name","neutral") if isinstance(emotion_profile, dict) and "name" in emotion_profile else "neutral", 
        t, 
        base_f0_scaled
    )
    if np.all(f0_curve == 0) or len(f0_curve) != len(t):
        f0_curve = base_f0_scaled * np.ones_like(t)
    
    # integrate phase
    phase = 2.0 * np.pi * np.cumsum(f0_curve) / SR
    exc = np.sin(phase)
    
    # shape spectrum by formants
    formants = FORMANTS.get(formant_key, FORMANTS["AH"])
    fft_exc = np.fft.rfft(exc)
    freqs = np.fft.rfftfreq(len(exc), 1.0 / SR)
    fs = [(fc * voice_profile.get("formant_scale", 1.0), bw) for (fc, bw) in formants]
    env = gaussian_formant_envelope(freqs, tuple(fs), strength=1.0 * voice_profile.get("brightness", 1.0) * emotion_profile.get("brightness", 1.0))
    shaped = np.fft.irfft(fft_exc * env)
    
    # breathiness mix (length-safe)
    breath = voice_profile.get("breathiness", 0.0) * emotion_profile.get("breathiness", 0.0) if "breathiness" in emotion_profile else voice_profile.get("breathiness", 0.0)
    if breath > 0:
        wn = white_noise(duration, amp=0.07)
        shaped, wn = match_length(shaped, wn)
        shaped += breath * wn
    
    shaped = apply_fades(shaped, fade_ms=8)
    
    # emotion volume
    shaped *= emotion_profile.get("volume", 1.0)
    
    return shaped

def synth_fricative(symbol: str, duration: float, voice_profile: dict, emotion_profile: dict) -> np.ndarray:
    duration = max(0.01, duration / max(1e-3, emotion_profile.get("speed", 1.0)))
    n = max(1, int(SR * duration))
    noise = white_noise(duration, amp=0.5)
    fft_noise = np.fft.rfft(noise)
    freqs = np.fft.rfftfreq(len(noise), 1.0/SR)
    if symbol in FRIC_BANDS:
        f_lo, f_hi = FRIC_BANDS[symbol]
    else:
        f_lo, f_hi = 2000, 8000
    env = bandpass_mask(freqs, f_lo*voice_profile.get("formant_scale",1.0), f_hi*voice_profile.get("formant_scale",1.0))
    shaped = np.fft.irfft(fft_noise * env)
    # slight voicing for z-like fricatives based on emotion
    if symbol in ("z",):
        voiced = synthesize_voiced_formant("AH", duration, voice_profile["f0"], voice_profile, emotion_profile) * 0.15
        shaped = (1.0 - 0.15) * shaped + 0.15 * voiced
    shaped = apply_fades(shaped, fade_ms=6)
    shaped *= emotion_profile.get("volume", 1.0)
    return shaped

def synthesize_stop(symbol: str, duration: float, voice_profile: dict, emotion_profile: dict) -> np.ndarray:
    duration = max(0.03, duration / max(1e-3, emotion_profile.get("speed", 1.0)))
    total_n = int(SR * duration)
    # closure and burst proportions
    closure_n = int(total_n * 0.6)
    burst_n = min(total_n - closure_n, int(SR * 0.06))
    release_n = total_n - closure_n - burst_n
    part_closure = np.zeros(closure_n)
    burst = white_noise(burst_n, amp=0.8)
    # emphasize mid freq for burst
    fft_b = np.fft.rfft(burst)
    freqs = np.fft.rfftfreq(len(burst), 1.0/SR)
    env = bandpass_mask(freqs, 800, 4500)
    burst = np.fft.irfft(fft_b * (0.5 + env))
    burst = apply_fades(burst, fade_ms=3)
    release = np.zeros(release_n)
    if release_n > 0:
        # short schwa-like release
        release = synthesize_voiced_formant("AH", release_n / SR, voice_profile["f0"], voice_profile, emotion_profile) * 0.4
        release = apply_fades(release, fade_ms=5)
    out = np.concatenate([part_closure, burst, release])
    out *= emotion_profile.get("volume", 1.0)
    return out

def synthesize_nasal(symbol: str, duration: float, voice_profile: dict, emotion_profile: dict) -> np.ndarray:
    duration = max(0.04, duration / max(1e-3, emotion_profile.get("speed", 1.0)))
    if symbol == "m":
        form = ((250,80),(1000,120),(2200,150))
    elif symbol == "n":
        form = ((300,80),(1200,120),(2400,150))
    else:
        form = FORMANTS.get("N")
    n = max(1, int(SR * duration))
    t = np.linspace(0.0, duration, n, False)
    # low-energy periodic plus formant shaping
    base_f0 = voice_profile.get("f0", 120.0) * emotion_profile.get("pitch_scale",1.0)
    phase = 2.0 * np.pi * base_f0 * t
    exc = 0.6 * np.sin(phase)
    fft_exc = np.fft.rfft(exc)
    freqs = np.fft.rfftfreq(len(exc), 1.0/SR)
    env = gaussian_formant_envelope(freqs, form, strength=0.8)
    shaped = np.fft.irfft(fft_exc * env)
    shaped = apply_fades(shaped, fade_ms=6)
    shaped *= emotion_profile.get("volume", 1.0)
    return shaped

# ------------- Character-to-synthesis mapping (NO G2P) -------------
# This maps characters directly to approximate behaviors.
VOWEL_LETTERS = set("aeiouyAEIOUY")
CONSONANT_TO_TYPE = {
    # fricatives
    's': ('fric','s'), 'S':('fric','s'),
    'z': ('fric','z'), 'Z':('fric','z'),
    'f': ('fric','f'), 'F':('fric','f'),
    'h': ('fric','h'), 'H':('fric','h'),
    # affricate-ish
    'c': ('fric','sh'), 'C':('fric','sh'),
    # stops (plosives)
    'p': ('stop','p'), 'P':('stop','p'),
    't': ('stop','t'), 'T':('stop','t'),
    'k': ('stop','k'), 'K':('stop','k'),
    'b': ('stop','b'), 'B':('stop','b'),
    'd': ('stop','d'), 'D':('stop','d'),
    'g': ('stop','g'), 'G':('stop','g'),
    # nasals
    'm': ('nasal','m'), 'M':('nasal','m'),
    'n': ('nasal','n'), 'N':('nasal','n'),
    # approximants (treat as voiced short vowels)
    'l': ('approx','L'), 'L':('approx','L'),
    'r': ('approx','R'), 'R':('approx','R'),
    'w': ('approx','W'), 'W':('approx','W'),
    'y': ('approx','Y'), 'Y':('approx','Y'),
}

# ------------- SSML parsing (subset) -------------
@dataclass
class Prosody:
    rate: float = 1.0
    pitch_semitones: float = 0.0
    volume_mul: float = 1.0
    is_break: bool = False
    break_ms: int = 0

@dataclass
class VoiceSpec:
    name: str = "default"
    emotion: str = "neutral"

def parse_rate(v: str) -> float:
    if not v: return 1.0
    s = v.strip().lower()
    keywords = {"x-slow":0.6,"slow":0.8,"medium":1.0,"fast":1.25,"x-fast":1.5}
    if s in keywords: return keywords[s]
    m = re.match(r"([0-9]+)%$", s)
    if m: return float(m.group(1))/100.0
    try: return float(s)
    except: return 1.0

def parse_pitch_to_st(v: str) -> float:
    if not v: return 0.0
    s = v.strip().lower()
    keywords = {"x-low":-4.0,"low":-2.0,"medium":0.0,"high":2.0,"x-high":4.0}
    if s in keywords: return keywords[s]
    m = re.match(r"([+-]?[0-9]+)st$", s)
    if m: return float(m.group(1))
    m2 = re.match(r"([0-9]+)%$", s)
    if m2:
        pct = float(m2.group(1))/100.0
        return 12.0 * math.log2(pct) if pct>0 else 0.0
    try: return float(s)
    except: return 0.0

def parse_volume(v: str) -> float:
    if not v: return 1.0
    s = v.strip().lower()
    keywords = {"silent":0.0,"x-soft":0.25,"soft":0.5,"medium":1.0,"loud":1.5,"x-loud":2.0}
    if s in keywords: return keywords[s]
    m = re.match(r"([+-]?[0-9]+)db$", s)
    if m:
        db = float(m.group(1))
        return 10**(db/20.0)
    m2 = re.match(r"([0-9]+)%$", s)
    if m2: return float(m2.group(1))/100.0
    try: return float(s)
    except: return 1.0

def parse_time_ms(v: str) -> int:
    if not v: return 0
    s = v.strip().lower()
    m = re.match(r"([0-9]+)ms$", s)
    if m: return int(m.group(1))
    m2 = re.match(r"([0-9]+(?:\.[0-9]+)?)s$", s)
    if m2: return int(float(m2.group(1)) * 1000)
    try: return int(float(s))
    except: return 0

def combine_prosody(parent: Prosody, attrs: Dict[str,str]) -> Prosody:
    p = Prosody(rate=parent.rate, pitch_semitones=parent.pitch_semitones, volume_mul=parent.volume_mul)
    if 'rate' in attrs: p.rate *= parse_rate(attrs['rate'])
    if 'pitch' in attrs: p.pitch_semitones += parse_pitch_to_st(attrs['pitch'])
    if 'volume' in attrs: p.volume_mul *= parse_volume(attrs['volume'])
    return p

def combine_voice(parent: VoiceSpec, attrs: Dict[str,str]) -> VoiceSpec:
    v = VoiceSpec(name=parent.name, emotion=parent.emotion)
    if 'name' in attrs: v.name = attrs['name']
    if 'emotion' in attrs: v.emotion = attrs['emotion']
    return v

import xml.etree.ElementTree as ET
def extract_segments_from_ssml(ssml_text: str) -> List[Tuple[str, VoiceSpec, Prosody]]:
    s = ssml_text.strip()
    if not s:
        return []
    try:
        if not s.startswith("<"):
            s = f"<speak>{s}</speak>"
        root = ET.fromstring(s if s.strip().startswith("<speak") else f"<speak>{s}</speak>")
    except Exception:
        # fallback plain text
        return [(ssml_text, VoiceSpec(), Prosody())]
    segments = []
    def recurse(node, current_voice: VoiceSpec, current_prosody: Prosody):
        tag = node.tag.lower()
        if tag.endswith("break"):
            br_ms = parse_time_ms(node.attrib.get("time") or node.attrib.get("strength") or "0ms")
            segments.append(("", current_voice, Prosody(rate=current_prosody.rate, pitch_semitones=current_prosody.pitch_semitones, volume_mul=current_prosody.volume_mul, is_break=True, break_ms=br_ms)))
            return
        new_voice = combine_voice(current_voice, node.attrib) if tag.endswith("voice") else current_voice
        new_prosody = combine_prosody(current_prosody, node.attrib) if tag.endswith("prosody") else current_prosody
        if node.text and node.text.strip():
            segments.append((node.text, new_voice, new_prosody))
        for c in node:
            recurse(c, new_voice, new_prosody)
            if c.tail and c.tail.strip():
                segments.append((c.tail, new_voice, new_prosody))
    recurse(root, VoiceSpec(), Prosody())
    # clean up empty text segments
    out = []
    for text, v, p in segments:
        t = text.replace("\n", " ").strip()
        if t == "" and not p.is_break:
            continue
        out.append((t, v, p))
    return out

# ------------- High-level segment synthesizer (text chunk -> audio) -------------
def synth_segment_text(text: str, prosody: Prosody, voice_name: str, emotion_name: str) -> np.ndarray:
    # voice/emotion profiles
    voice_profile = VOICE_PROFILES.get(voice_name, VOICE_PROFILES["default"])
    emotion_profile = EMOTION_PROFILES.get(emotion_name, EMOTION_PROFILES["neutral"]).copy()
    # include emotion name for pitch curve decisions
    emotion_profile["name"] = emotion_name
    # set volume multiplicative with prosody volume
    emotion_profile["volume"] = emotion_profile.get("volume",1.0) * prosody.volume_mul
    # combine prosody pitch semitone offset top of emotion pitch scale
    pitch_semitone_offset = prosody.pitch_semitones
    # effective base f0
    base_f0 = voice_profile.get("f0", 125.0) * emotion_profile.get("pitch_scale",1.0) * (2.0 ** (pitch_semitone_offset / 12.0))
    # split text into characters tokens, synth each char
    audio = np.zeros(0, dtype=np.float64)
    # simple word-level pause length
    base_pause = 0.06
    # identify letters and punctuation
    chars = list(text)
    # simple stress heuristic: stress every 2nd vowel (per continuous text); influenced by emotion stress factor
    vowel_count = 0
    for i, ch in enumerate(chars):
        if ch.isspace():
            pause_dur = base_pause * emotion_profile.get("pause_factor",1.0) / prosody.rate
            audio = np.concatenate([audio, np.zeros(int(SR * pause_dur))])
            continue
        # punctuation influences pause and intonation (short mapping)
        if ch in ",;:":
            audio = np.concatenate([audio, np.zeros(int(SR * 0.12 * emotion_profile.get("pause_factor",1.0)) )])
            continue
        if ch in ".!?":
            audio = np.concatenate([audio, np.zeros(int(SR * 0.22 * emotion_profile.get("pause_factor",1.0)) )])
            continue
        # map to synthetic behavior
        if ch in VOWEL_LETTERS:
            # choose a formant target by vowel letter
            target = "AH"
            cl = ch.lower()
            if cl == 'a': target = "A"
            elif cl == 'e': target = "E"
            elif cl == 'i': target = "I"
            elif cl == 'o': target = "O"
            elif cl == 'u': target = "U"
            elif cl == 'y': target = "I"
            # duration baseline
            dur = 0.14 if target != "I" else 0.12
            # stress heuristic
            stressed = (vowel_count % 2 == 0)
            vowel_count += 1
            if stressed:
                dur *= 1.0 * emotion_profile.get("stress", 1.0) * (1.15 if prosody.rate <= 1.0 else 1.05)
            # scale by prosody rate (rate >1 -> shorter durations)
            dur = dur / max(1e-3, prosody.rate)
            # dynamic pitch handled inside synthesize_voiced_formant via emotion_profile (and base_f0)
            seg = synthesize_voiced_formant(target, dur, base_f0, voice_profile, emotion_profile)
            audio = overlap_append(audio, seg, xf_ms=10)
            continue
        # consonant mapping table
        mapping = CONSONANT_TO_TYPE.get(ch, None)
        if mapping:
            typ, sym = mapping
            if typ == 'fric':
                dur = 0.09 / max(1e-3, prosody.rate)
                seg = synth_fricative(sym, dur, voice_profile, emotion_profile)
                audio = overlap_append(audio, seg, xf_ms=6)
                continue
            if typ == 'stop':
                dur = 0.06 / max(1e-3, prosody.rate)
                seg = synthesize_stop(sym, dur, voice_profile, emotion_profile)
                audio = overlap_append(audio, seg, xf_ms=6)
                continue
            if typ == 'nasal':
                dur = 0.11 / max(1e-3, prosody.rate)
                seg = synthesize_nasal(sym, dur, voice_profile, emotion_profile)
                audio = overlap_append(audio, seg, xf_ms=8)
                continue
            if typ == 'approx':
                # short voiced approximant -> treat as short voiced formant
                dur = 0.08 / max(1e-3, prosody.rate)
                seg = synthesize_voiced_formant(sym if sym in FORMANTS else "AH", dur, base_f0, voice_profile, emotion_profile)
                audio = overlap_append(audio, seg, xf_ms=8)
                continue
        # fallback: produce a soft schwa-like brief sound
        seg = synthesize_voiced_formant("AH", 0.06 / max(1e-3, prosody.rate), base_f0, voice_profile, emotion_profile) * 0.6
        audio = overlap_append(audio, seg, xf_ms=6)
    # apply overall prosody volume multiply
    audio *= prosody.volume_mul
    # final normalization safe-scaling
    if audio.size > 0:
        peak = np.max(np.abs(audio)) + EPS
        audio = audio * (0.95 / peak)
    return audio

# ------------- Top-level SSML -> waveform -------------
def synthesize_ssml(ssml_text: str, out_wav: str = None, play: bool = False) -> np.ndarray:
    segments = extract_segments_from_ssml(ssml_text)
    out = np.zeros(0, dtype=np.float64)
    for text, voice_spec, prosody in segments:
        if prosody.is_break:
            n = int(SR * (prosody.break_ms / 1000.0))
            out = np.concatenate([out, np.zeros(n)])
            continue
        seg_audio = synth_segment_text(text, prosody, voice_spec.name, voice_spec.emotion)
        out = np.concatenate([out, seg_audio])
    # final normalization and optional write
    if out.size > 0:
        peak = np.max(np.abs(out)) + EPS
        out = out * (0.95 / peak)
    if out_wav:
        write_wav(out_wav, out, SR)
    if play:
        try:
            import sounddevice as sd
            sd.play(out, SR); sd.wait()
        except Exception as e:
            print("Playback failed (sounddevice may be missing):", e)
    return out

# ------------- WAV write helper -------------
def write_wav(path: str, x: np.ndarray, sr: int = SR):
    x16 = np.clip(x, -1.0, 1.0)
    x16 = (x16 * 32767.0).astype(np.int16)
    with wave.open(path, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(x16.tobytes())

# ------------- Demo SSML & main -------------
if __name__ == "__main__":
    demo = r"""
<speak>
  <voice name="female" emotion="happy">
    <prosody rate="110%" pitch="+3st" volume="110%">
      Hello there! <break time="180ms"/> I'm so glad you're listening.
    </prosody>
  </voice>

  <voice name="male" emotion="sad">
    <prosody rate="85%" pitch="-3st" volume="85%">
      I feel a little low today... <break time="250ms"/> but it's okay.
    </prosody>
  </voice>

  <voice name="robot" emotion="angry">
    <prosody rate="105%" pitch="+0st" volume="120%">
      Warning! System overload detected.
    </prosody>
  </voice>

  Plain text fallback: this is still synthesized (no SSML).
</speak>
"""
    print("Synthesizing demo SSML (this may take a few seconds)...")
    y = synthesize_ssml(demo, out_wav="demo_out.wav", play=False)
    print(f"Wrote demo_out.wav ({len(y)/SR:.2f}s).")
    # Try playback if sounddevice available
    try:
        import sounddevice as sd
        print("Playing demo...")
        sd.play(y, SR); sd.wait()
    except Exception:
        print("Install sounddevice to enable playback: pip install sounddevice")

