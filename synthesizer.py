"""FFT formant synthesis, Kelly-Lochbaum synthesis, and high-level orchestrator."""

import re
import numpy as np
from config import SR, EPS
from phonemes import (FORMANTS, NASAL_FMT, APPROX_FMT, FRIC_BANDS,
                      PH_TYPE, PH_DUR, KL_AREAS, get_phoneme_formants)
from profiles import VOICES, EMOTIONS, get_voice, get_emotion
from text_processing import text_to_phonemes
from ssml_parser import parse_ssml
from audio_utils import (wnoise, apply_fades, gaussian_envelope, bandpass_mask,
                         f0_curve, apply_warmth, match_length, normalize_audio)


# ═══════════════════════════════════════════════════════════════
# FFT FORMANT SYNTHESIS
# ═══════════════════════════════════════════════════════════════
def _get_fmts(key):
    """Get formant data for phoneme."""
    return get_phoneme_formants(key)


def fft_voiced(key, dur, f0, vp, ep, emo="neutral"):
    """Synthesize voiced phoneme using FFT formant method."""
    dur = max(0.01, dur / max(EPS, ep["spd"]))
    n = max(1, int(SR * dur))
    t = np.linspace(0, dur, n, False)
    f0s = f0 * ep["ps"] * (2 ** (ep["pst"] / 12))
    fc = f0_curve(emo, t, f0s)
    if len(fc) != n:
        fc = f0s * np.ones(n)

    exc = np.sin(2 * np.pi * np.cumsum(fc) / SR)
    fmts = _get_fmts(key)
    sc = [(fc_ * vp["fscale"], bw) for fc_, bw in fmts]

    fe = np.fft.rfft(exc)
    fr = np.fft.rfftfreq(len(exc), 1.0 / SR)
    br = vp["bright"] * ep["bright"]
    env = gaussian_envelope(fr, sc, br)
    shaped = np.fft.irfft(fe * env)

    breath = vp["breath"] * ep["breath"]
    if breath > 0:
        wn = wnoise(dur, 0.07)
        shaped, wn = match_length(shaped, wn)
        shaped += breath * wn

    return apply_fades(shaped) * ep["vol"]


def fft_fric(sym, dur, vp, ep):
    """Synthesize fricative using FFT filtering."""
    dur = max(0.01, dur / max(EPS, ep["spd"]))
    ns = wnoise(dur, 0.5)
    fn = np.fft.rfft(ns)
    fr = np.fft.rfftfreq(len(ns), 1.0 / SR)
    flo, fhi = FRIC_BANDS.get(sym, (2000, 8000))
    fs = vp["fscale"]
    env = bandpass_mask(fr, flo * fs, fhi * fs)
    return apply_fades(np.fft.irfft(fn * env), 6) * ep["vol"]


def fft_stop(sym, dur, vp, ep, emo="neutral"):
    """Synthesize stop consonant."""
    dur = max(0.03, dur / max(EPS, ep["spd"]))
    tot = int(SR * dur)
    cn = int(tot * 0.6)
    bn = min(tot - cn, int(SR * 0.06))
    rn = tot - cn - bn

    burst = wnoise(bn / SR, 0.8)
    fb = np.fft.rfft(burst)
    fr = np.fft.rfftfreq(len(burst), 1.0 / SR)
    burst = np.fft.irfft(fb * (0.5 + bandpass_mask(fr, 800, 4500)))
    burst = apply_fades(burst, 3)

    rel = np.zeros(rn)
    if rn > 0:
        rel = fft_voiced("AH", rn / SR, vp["f0"], vp, ep, emo) * 0.4
        rel = apply_fades(rel, 5)

    return np.concatenate([np.zeros(cn), burst, rel]) * ep["vol"]


# ═══════════════════════════════════════════════════════════════
# KELLY-LOCHBAUM SYNTHESIS
# ═══════════════════════════════════════════════════════════════
def _kl_filter(exc, areas):
    """Apply Kelly-Lochbaum vocal tract filter."""
    nt = len(areas)
    N = len(exc)
    out = np.zeros(N)
    fw = np.zeros(nt)
    bw = np.zeros(nt)

    for n in range(N):
        r = np.zeros(nt - 1)
        for k in range(nt - 1):
            r[k] = (areas[k + 1] - areas[k]) / (areas[k + 1] + areas[k] + EPS)

        fw[0] = exc[n] + r[0] * bw[0]
        for k in range(1, nt - 1):
            fw[k] = (1 + r[k - 1]) * fw[k - 1] - r[k - 1] * bw[k]
            bw[k] = (1 - r[k]) * bw[k + 1] + r[k] * fw[k]

        fw[nt - 1] = (1 + r[nt - 2]) * fw[nt - 2]

        if n > 1:
            out[n] = fw[nt - 1] - 0.97 * out[n - 1] + 0.3 * out[n - 2]
        elif n > 0:
            out[n] = fw[nt - 1] - 0.97 * out[n - 1]
        else:
            out[n] = fw[nt - 1]

        for k in range(nt - 2, 0, -1):
            bw[k] = (1 - r[k]) * bw[k + 1] + r[k] * fw[k]
        bw[0] = (1 - r[0]) * bw[1] + r[0] * fw[0]

    return out


def kl_phoneme(ph, dur, f0, vp, ep, emo="neutral"):
    """Synthesize phoneme using Kelly-Lochbaum model."""
    dur = max(0.01, dur / max(EPS, ep["spd"]))
    n = max(1, int(SR * dur))
    t = np.linspace(0, dur, n, False)
    f0s = f0 * ep["ps"]
    fc = f0_curve(emo, t, f0s)
    if len(fc) != n:
        fc = f0s * np.ones(n)

    pt = PH_TYPE.get(ph, "silence")
    if pt == "fricative":
        exc = wnoise(dur, 0.5)
    elif pt == "stop":
        cn = int(n * 0.6)
        be = wnoise((n - cn) / SR, 0.8)
        exc = np.concatenate([np.zeros(cn), be])
        exc = exc[:n] if len(exc) >= n else np.pad(exc, (0, n - len(exc)))
    else:
        exc = np.sin(2 * np.pi * np.cumsum(fc) / SR)

    areas = [a * vp["fscale"] for a in KL_AREAS.get(ph, [2.0] * 8)]
    out = _kl_filter(exc, areas)

    breath = vp["breath"] * ep["breath"]
    if breath > 0 and pt in ("vowel", "nasal", "approximant"):
        wn = wnoise(dur, 0.07)
        out, wn = match_length(out, wn)
        out += breath * wn

    return apply_fades(out) * ep["vol"]


# ═══════════════════════════════════════════════════════════════
# HIGH-LEVEL SYNTHESIZER
# ═══════════════════════════════════════════════════════════════
def _synth_phones(phones, pros, vname, ename, engine):
    """Synthesize a sequence of phonemes."""
    vp = get_voice(vname)
    ep = get_emotion(ename).copy()
    ep["vol"] *= pros.volume
    ep["spd"] *= pros.rate
    ep["pst"] = ep.get("pst", 0) + pros.pitch_st

    base_f0 = vp["f0"] * ep["ps"] * (2 ** (ep["pst"] / 12))
    blocks = []
    ph_rng = []
    ci = 0

    # Split phones into words
    words_ph = []
    cur = []
    for p in phones:
        if p == " ":
            if cur:
                words_ph.append(cur)
            cur = []
        else:
            cur.append(p)
    if cur:
        words_ph.append(cur)

    w_rng = []
    for wp in words_ph:
        ws = ci
        for pi, ph in enumerate(wp):
            pt = PH_TYPE.get(ph, "silence")
            dur = PH_DUR.get(pt, 0.08)
            if pi == 0 and pt == "vowel":
                dur *= ep["stress"]

            if engine == "kl":
                sig = kl_phoneme(ph, dur, base_f0, vp, ep, ename)
            elif pt == "fricative":
                sig = fft_fric(ph, dur, vp, ep)
            elif pt == "stop":
                sig = fft_stop(ph, dur, vp, ep, ename)
            else:
                sig = fft_voiced(ph, dur, base_f0, vp, ep, ename)

            blocks.append(sig)
            ph_rng.append((ci, ci + len(sig), ph))
            ci += len(sig)

        w_rng.append((ws, ci, ""))
        pause = np.zeros(int(SR * 0.06 * ep["pause"]))
        blocks.append(pause)
        ci += len(pause)

    if not blocks:
        return np.zeros(0), [], []
    return np.concatenate(blocks), ph_rng, w_rng


def synthesize_ssml(ssml_text, engine="fft", warmth=0, prog_cb=None):
    """Main synthesis function - converts SSML to audio.
    
    Args:
        ssml_text: SSML input text
        engine: "fft" or "kl" synthesis engine
        warmth: Warmth effect level (0-100)
        prog_cb: Optional progress callback function
        
    Returns:
        Tuple of (audio, phoneme_ranges, word_ranges)
    """
    segs = parse_ssml(ssml_text)
    all_a = []
    all_ph = []
    all_w = []
    off = 0
    ts = len(segs)

    for si, (txt, vs, pr) in enumerate(segs):
        if prog_cb:
            prog_cb(int(si / ts * 100))
        if pr.is_break:
            p = np.zeros(int(SR * pr.break_ms / 1000))
            all_a.append(p)
            off += len(p)
            continue

        phones = text_to_phonemes(txt)
        if not phones:
            continue

        a, ph, w = _synth_phones(phones, pr, vs.name, vs.emotion, engine)
        all_a.append(a)
        all_ph.extend([(s + off, e + off, p) for s, e, p in ph])
        all_w.extend([(s + off, e + off, ww) for s, e, ww in w])
        off += len(a)

    if not all_a:
        return np.zeros(0), [], []

    audio = np.concatenate(all_a)

    # Fill in word text
    words = []
    for txt, vs, pr in segs:
        if not pr.is_break and txt:
            words.extend(re.findall(r'\S+', txt))
    for i, (s, e, _) in enumerate(all_w):
        if i < len(words):
            all_w[i] = (s, e, words[i])

    if warmth > 0:
        audio = apply_warmth(audio, warmth)
    
    audio = normalize_audio(audio, 0.95)
    
    if prog_cb:
        prog_cb(100)

    return audio, all_ph, all_w