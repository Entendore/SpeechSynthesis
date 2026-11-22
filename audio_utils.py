"""Low-level audio utility functions."""

import numpy as np
from scipy.signal import lfilter, butter
from config import SR, EPS


def wnoise(dur, amp=0.3):
    """Generate white noise."""
    return np.random.randn(max(1, int(SR * dur))) * amp


def apply_fades(x, ms=8):
    """Apply fade in/out to audio."""
    if x.size == 0:
        return x
    n = len(x)
    m = max(1, int(SR * ms / 1000))
    w = np.ones(n)
    r = np.linspace(0, 1, m)
    w[:m] = r
    w[-m:] = r[::-1]
    pk = np.max(np.abs(x * w)) + EPS
    return 0.95 * x * w / pk


def overlap_add(a, b, ms=8):
    """Overlap-add two audio segments."""
    if a.size == 0:
        return b.copy()
    m = max(1, int(SR * ms / 1000))
    out = np.zeros(len(a) + len(b) - m)
    out[:len(a)] = a
    f = np.linspace(0, 1, m)
    out[len(a) - m:len(a)] = (1 - f) * out[len(a) - m:len(a)] + f * b[:m]
    out[len(a):] = b[m:]
    return out


def gaussian_envelope(freqs, fmts, strength=1.0):
    """Generate Gaussian formant envelope."""
    env = np.full_like(freqs, 1e-6)
    for fc, bw in fmts:
        sigma = bw / (2 * np.sqrt(2 * np.log(2)) + EPS)
        env += strength * np.exp(-0.5 * ((freqs - fc) / (sigma + EPS)) ** 2)
    return env


def bandpass_mask(freqs, flo, fhi, edge=300):
    """Generate bandpass filter mask."""
    return (np.clip((freqs - max(0, flo - edge)) / (2 * edge + EPS), 0, 1)
            * np.clip((fhi + edge - freqs) / (2 * edge + EPS), 0, 1))


def f0_curve(emo, t, f0):
    """Generate F0 contour based on emotion."""
    if t.size == 0:
        return np.array([])
    mx = t.max() + EPS
    
    curves = {
        "happy": lambda: f0 * (1.05 + 0.12 * (t / mx) + 0.03 * np.sin(2 * np.pi * 3 * t)),
        "sad": lambda: f0 * (0.98 - 0.18 * (t / mx)),
        "angry": lambda: f0 * (1.0 + 0.18 * np.sign(np.sin(2 * np.pi * 6 * t))),
        "calm": lambda: f0 * (1.0 + 0.03 * np.sin(2 * np.pi * 1.5 * t)),
    }
    
    return curves.get(emo, lambda: f0 * np.ones_like(t))()


def apply_warmth(x, w):
    """Apply warmth effect (chorus + lowpass)."""
    if w == 0 or x.size == 0:
        return x
    d = w / 100 * 0.7
    sh = int(0.025 * SR)
    x = x + d * np.roll(x, sh)
    b, a = butter(2, 400 / (SR / 2), btype='low')
    x = lfilter(b, a, x)
    pk = np.max(np.abs(x)) + EPS
    return x / pk


def match_length(a, b):
    """Match lengths of two arrays."""
    n = min(len(a), len(b))
    return a[:n], b[:n]


def normalize_audio(x, target=0.95):
    """Normalize audio to target level."""
    if x.size == 0:
        return x
    pk = np.max(np.abs(x)) + EPS
    return x * (target / pk)


def write_wav(path, x, sr=SR):
    """Write audio to WAV file."""
    import wave
    x16 = np.clip(x, -1, 1)
    x16 = (x16 * 32767).astype(np.int16)
    with wave.open(path, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(x16.tobytes())


def read_wav(path):
    """Read audio from WAV file."""
    import wave
    with wave.open(path, "rb") as wf:
        sr = wf.getframerate()
        n_frames = wf.getnframes()
        data = wf.readframes(n_frames)
        x = np.frombuffer(data, dtype=np.int16).astype(np.float32) / 32767.0
        return x, sr


def apply_reverb(audio, decay=0.5, mix=0.3):
    """Apply simple reverb effect."""
    result = audio.copy()
    delay_samples = int(0.03 * SR)
    reverb = np.zeros_like(result)
    
    for i in range(6):
        d = int(delay_samples * (i + 1) ** 1.3)
        g = decay ** (i + 1)
        if d < len(result):
            padded = np.zeros(len(result) + d)
            padded[:len(result)] = result
            reverb += g * padded[d:d + len(result)]
    
    reverb = normalize_audio(reverb)
    return (1 - mix) * result + mix * reverb


def apply_echo(audio, delay=0.25, feedback=0.4):
    """Apply echo effect."""
    delay_samples = int(delay * SR)
    echo = np.zeros_like(audio)
    
    for i in range(8):
        d = delay_samples * (i + 1)
        g = feedback ** i
        if d < len(audio):
            padded = np.zeros(len(audio) + d)
            padded[:len(audio)] = audio
            echo += g * padded[d:d + len(audio)]
    
    echo = normalize_audio(echo)
    return 0.7 * audio + 0.3 * echo


def apply_eq(audio, low_db=0, mid_db=0, high_db=0):
    """Apply 3-band EQ."""
    freq = np.fft.rfftfreq(len(audio), 1.0 / SR)
    eq_curve = np.ones_like(freq)
    
    low_gain = 10 ** (low_db / 20)
    mid_gain = 10 ** (mid_db / 20)
    high_gain = 10 ** (high_db / 20)
    
    eq_curve *= np.clip((300 - freq) / 300, 0, 1) * (low_gain - 1) + 1
    mid_mask = np.clip((freq - 300) / 200, 0, 1) * np.clip((3000 - freq) / 500, 0, 1)
    eq_curve += mid_mask * (mid_gain - 1)
    eq_curve *= np.clip((freq - 3000) / 500, 0, 1) * (high_gain - 1) + 1
    
    spectrum = np.fft.rfft(audio)
    spectrum *= eq_curve
    return np.fft.irfft(spectrum, len(audio))


def estimate_pitch(audio, frame_size=1024):
    """Simple pitch estimation using autocorrelation."""
    pitches = []
    for i in range(0, len(audio) - frame_size, frame_size // 2):
        frame = audio[i:i + frame_size]
        frame = frame * np.hanning(len(frame))
        
        # Autocorrelation
        corr = np.correlate(frame, frame, mode='full')
        corr = corr[len(corr) // 2:]
        
        # Find first peak after initial decline
        min_lag = int(SR / 500)  # Max 500 Hz
        max_lag = int(SR / 50)   # Min 50 Hz
        
        if len(corr) > max_lag:
            search = corr[min_lag:max_lag]
            if len(search) > 0:
                peak_idx = np.argmax(search) + min_lag
                pitch = SR / peak_idx
                if 50 < pitch < 500:
                    pitches.append(pitch)
                else:
                    pitches.append(0)
            else:
                pitches.append(0)
        else:
            pitches.append(0)
    
    return pitches