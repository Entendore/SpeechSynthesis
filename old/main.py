#!/usr/bin/env python3
import sys, re, wave
import numpy as np
import sounddevice as sd
from scipy.signal import resample
from PyQt6.QtWidgets import QApplication, QWidget, QVBoxLayout, QLabel, QSlider, QPushButton, QTextEdit
from PyQt6.QtCore import Qt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
import matplotlib.animation as animation

# ---------------- Config ----------------
SR = 22050
WINDOW_DURATION = 0.5
WINDOW_SIZE = int(SR*WINDOW_DURATION)
SPEC_FFT_SIZE = 512
SPEC_HOP_SIZE = 128
SPEC_N_COLS = int(WINDOW_DURATION*SR/SPEC_HOP_SIZE)

# ---------------- Phonemes ----------------
PHONEMES = {
    "AH":[(700,80),(1220,90),(2600,120)],
    "IY":[(300,60),(2200,90),(3000,120)],
    "EY":[(500,70),(2000,80),(2600,100)],
    "OW":[(400,60),(750,70),(2400,100)],
    "OY":[(400,70),(1200,80),(2400,100)],
    "AA":[(700,70),(1200,80),(2600,100)],
    "SH":[(0,2500)], "F":[(0,2000)], "S":[(0,3000)],
    "P":[(0,1)], "T":[(0,1)], "K":[(0,1)],
    "B":[(0,1)], "D":[(0,1)], "G":[(0,1)]
}
PHONEME_TYPES = {
    "AH":"vowel","IY":"vowel","EY":"vowel","OW":"vowel","OY":"vowel","AA":"vowel",
    "S":"fricative","SH":"fricative","F":"fricative",
    "P":"stop","T":"stop","K":"stop","B":"stop","D":"stop","G":"stop"
}
PHONEME_DURATIONS = {
    "vowel":0.12,"approximant":0.10,"fricative":0.08,"stop":0.06,"silence":0.05
}
PHONEME_COLORS = {
    "vowel": "cyan",
    "stop": "magenta",
    "fricative": "yellow",
    "silence": "gray",
    "approximant": "green"
}

# ---------------- Voices & Emotions ----------------
VOICES = {
    "default":{"f0":120,"formant_scale":1.0,"breathiness":0.05,"brightness":1.0},
    "deep":{"f0":90,"formant_scale":0.85,"breathiness":0.03,"brightness":0.9},
    "bright":{"f0":150,"formant_scale":1.1,"breathiness":0.08,"brightness":1.2},
    "robot":{"f0":130,"formant_scale":1.0,"breathiness":0.0,"brightness":1.0}
}
EMOTIONS = {
    "neutral":{"pitch_scale":1.0,"volume":1.0,"breathiness":1.0,"speed":1.0},
    "happy":{"pitch_scale":1.2,"volume":1.2,"breathiness":1.1,"speed":1.1},
    "angry":{"pitch_scale":1.1,"volume":1.3,"breathiness":0.9,"speed":1.2}
}

# ---------------- Utilities ----------------
def white_noise(duration, amp=0.05):
    n = int(SR*duration)
    return np.random.randn(n).astype(np.float32)*amp

def apply_fades(sig, fade_ms=5):
    n_fade = int(SR*fade_ms/1000)
    fade_in = np.linspace(0,1,n_fade)
    fade_out = np.linspace(1,0,n_fade)
    sig[:n_fade]*=fade_in
    sig[-n_fade:]*=fade_out
    return sig

def phrase_pitch_curve(duration, f0_base=1.0, phrase_type="statement"):
    n=int(SR*duration)
    t=np.linspace(0,duration,n)
    if phrase_type=="statement":
        return f0_base*(1-0.05*np.cos(np.pi*t/duration))
    elif phrase_type=="question":
        return f0_base*(1+0.15*t/duration)
    else:
        return f0_base*np.ones_like(t)

def apply_word_stress(n, stress_level=0.1):
    return 1.0 + stress_level*np.sin(np.linspace(0,np.pi,n))

def get_phoneme_duration(phoneme, rate=1.0):
    ptype = PHONEME_TYPES.get(phoneme,"silence")
    return PHONEME_DURATIONS.get(ptype,0.08)/rate

def interpolate_formants_smooth(start_formants, end_formants, n):
    interpolated=[]
    n_formants=len(start_formants)
    for i in range(n_formants):
        f_start,bw_start=start_formants[i]
        f_end,bw_end=end_formants[i]
        t = np.linspace(0,1,n)
        alpha = 0.5 - 0.5*np.cos(np.pi*t)
        freqs = f_start + (f_end-f_start)*alpha
        bws = bw_start + (bw_end-bw_start)*alpha
        interpolated.append((freqs,bws))
    return interpolated

def apply_emotional_dynamics(sig, emotion='neutral'):
    if emotion=='happy':
        sig *= 1.1
    elif emotion=='angry':
        sig *= 1.2
    return sig

def pitch_shift_block(block, pitch_factor):
    n = len(block)
    new_n = max(1, int(n / pitch_factor))
    return resample(block, new_n).astype(np.float32)

# ---------------- Phoneme Synthesis ----------------
def synthesize_voiced(phoneme, duration, voice_profile, emotion_profile, next_phoneme=None):
    duration = max(0.01, duration/emotion_profile.get("speed",1.0))
    n = int(SR*duration)
    t = np.linspace(0,duration,n,False)
    base_f0 = voice_profile.get("f0",120) * emotion_profile.get("pitch_scale",1.0)
    phase = 2*np.pi*np.cumsum(phrase_pitch_curve(duration, base_f0))/SR
    exc = np.sin(phase)
    start_formants = PHONEMES.get(phoneme, PHONEMES["AH"])
    end_formants = PHONEMES.get(next_phoneme, start_formants)
    interpolated_formants = interpolate_formants_smooth(start_formants, end_formants, n)
    fft_exc=np.fft.rfft(exc)
    freqs_fft=np.fft.rfftfreq(len(exc),1.0/SR)
    env=np.zeros_like(freqs_fft)
    for freqs,bws in interpolated_formants:
        for f_curve,bw_curve in zip(freqs,bws):
            env += np.exp(-0.5*((freqs_fft-f_curve)/bw_curve)**2)
    env = np.clip(env,0,10)
    shaped = np.fft.irfft(fft_exc*env).astype(np.float32)
    breath = voice_profile.get("breathiness",0.05)*emotion_profile.get("breathiness",1.0)
    wn = white_noise(duration,amp=0.07)
    min_len = min(len(shaped), len(wn))
    shaped[:min_len] += wn[:min_len]*breath
    stress_curve = apply_word_stress(n, stress_level=0.1)
    intonation_curve = phrase_pitch_curve(duration,1.0,'statement')
    shaped *= stress_curve * intonation_curve
    shaped = apply_fades(shaped, fade_ms=8)
    shaped *= emotion_profile.get("volume",1.0)
    return shaped

def synthesize_fricative(p,dur,voice_profile,emotion_profile):
    return white_noise(dur, amp=0.1*emotion_profile.get("volume",1.0))

def synthesize_stop(p,dur,voice_profile,emotion_profile):
    n=int(SR*dur)
    sig=np.zeros(n,dtype=np.float32)
    burst=int(0.02*SR)
    sig[-burst:] = np.random.randn(burst).astype(np.float32)*0.3*emotion_profile.get("volume",1.0)
    return sig

# ---------------- Text to Phonemes ----------------
def text_to_phonemes(text):
    digraphs = {"sh":"SH","ch":"CH","th":"TH","ng":"NG","oo":"OW","ee":"IY","ay":"EY","ow":"OW"}
    single_letters = {"a":"AH","i":"IY","e":"EH","o":"OW","u":"UH","y":"OY"}
    phonemes=[]
    i=0
    text=text.lower()
    while i<len(text):
        found=False
        for dg,ph in digraphs.items():
            if text.startswith(dg,i):
                phonemes.append(ph)
                i+=len(dg)
                found=True
                break
        if found: continue
        c=text[i]
        if c in single_letters:
            phonemes.append(single_letters[c])
        elif c.isalpha():
            phonemes.append("AH")
        i+=1
    return phonemes

# ---------------- SSML Parsing ----------------
SSML_TAG_RE = re.compile(r"<(voice|prosody)(.*?)>(.*?)</\1>", re.DOTALL)
def parse_ssml(text):
    segments=[]
    pos=0
    for m in SSML_TAG_RE.finditer(text):
        if m.start()>pos:
            segments.append({"text":text[pos:m.start()],"voice":"default","emotion":"neutral","rate":1.0,"volume":1.0})
        tag, attrs, inner = m.groups()
        voice="default"; emotion="neutral"; rate=1.0; volume=1.0
        if tag=="voice":
            m_name = re.search(r'name="(.*?)"',attrs)
            if m_name: voice=m_name.group(1)
            m_em = re.search(r'emotion="(.*?)"',attrs)
            if m_em: emotion=m_em.group(1)
        elif tag=="prosody":
            m_rate = re.search(r'rate="([\d\.]+)"',attrs)
            m_vol = re.search(r'volume="([\d\.]+)"',attrs)
            if m_rate: rate=float(m_rate.group(1))
            if m_vol: volume=float(m_vol.group(1))
        segments.append({"text":inner,"voice":voice,"emotion":emotion,"rate":rate,"volume":volume})
        pos=m.end()
    if pos<len(text):
        segments.append({"text":text[pos:],"voice":"default","emotion":"neutral","rate":1.0,"volume":1.0})
    return segments

# ---------------- Audio Generation with Word Tracking ----------------
def synthesize_ssml_audio_with_words(ssml_segments):
    audio_blocks=[]
    phoneme_ranges=[]
    word_ranges=[]
    curr_idx=0
    for seg in ssml_segments:
        text=seg["text"]
        voice_prof=VOICES.get(seg["voice"], VOICES["default"])
        emotion_prof=EMOTIONS.get(seg["emotion"],EMOTIONS["neutral"])
        rate = seg["rate"]
        words = re.findall(r'\S+|\s+', text)
        for word in words:
            phonemes = text_to_phonemes(word)
            word_start = curr_idx
            for i,p in enumerate(phonemes):
                next_p=phonemes[i+1] if i+1<len(phonemes) else None
                ptype = PHONEME_TYPES.get(p,"silence")
                dur = get_phoneme_duration(p, rate)
                if ptype in ["vowel","approximant"]:
                    s=synthesize_voiced(p,dur,voice_prof,emotion_prof,next_phoneme=next_p)
                elif ptype=="fricative":
                    s=synthesize_fricative(p,dur,voice_prof,emotion_prof)
                elif ptype=="stop":
                    s=synthesize_stop(p,dur,voice_prof,emotion_prof)
                else:
                    s=np.zeros(int(SR*dur),dtype=np.float32)
                s=apply_emotional_dynamics(s, seg["emotion"])
                audio_blocks.append(s)
                start=curr_idx
                end=curr_idx+len(s)
                phoneme_ranges.append((start,end,p))
                curr_idx=end
            word_end = curr_idx
            word_ranges.append((word_start, word_end, word.strip()))
    audio=np.concatenate(audio_blocks)
    return audio, phoneme_ranges, word_ranges


# ---------------- GUI ----------------
class TTSControlPanel(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("FFT TTS Controls")
        layout=QVBoxLayout()
        self.text_input=QTextEdit()
        self.text_input.setPlaceholderText("Enter SSML text here...")
        layout.addWidget(self.text_input)

        self.pitch_slider=QSlider(Qt.Orientation.Horizontal)
        self.pitch_slider.setRange(50,200)
        self.pitch_slider.setValue(100)
        layout.addWidget(QLabel("Pitch (%)"))
        layout.addWidget(self.pitch_slider)

        self.emotion_slider=QSlider(Qt.Orientation.Horizontal)
        self.emotion_slider.setRange(50,150)
        self.emotion_slider.setValue(100)
        layout.addWidget(QLabel("Emotion Factor (%)"))
        layout.addWidget(self.emotion_slider)

        self.play_button=QPushButton("Play SSML")
        layout.addWidget(self.play_button)
        self.setLayout(layout)

    def get_values(self):
        return self.pitch_slider.value()/100.0, self.emotion_slider.value()/100.0


def play_demo(panel):
    ssml_text = panel.text_input.toPlainText()
    audio, phoneme_ranges, word_ranges = synthesize_ssml_audio_with_words(parse_ssml(ssml_text))
    n = len(audio)
    recorded_audio = np.zeros_like(audio)
    t = np.arange(n)/SR
    blocksize = 1024
    idx = [0]

    fig, (ax_wave, ax_spec) = plt.subplots(2,1,figsize=(10,6))

    # Waveform + phoneme highlight
    line_wave, = ax_wave.plot([], [])
    highlight_wave = Polygon([[0,0],[0,0],[0,0],[0,0]], closed=True, color='cyan', alpha=0.3)
    ax_wave.add_patch(highlight_wave)
    ax_wave.set_ylim(-1,1)
    ax_wave.set_xlim(0, WINDOW_DURATION)
    ax_wave.set_title("Scrolling Waveform with Phoneme Highlight")
    phoneme_text = ax_wave.text(0.5, 1.05, "", transform=ax_wave.transAxes,
                                ha='center', va='bottom', fontsize=10, color='white')
    word_text = ax_wave.text(0.5, -0.15, "", transform=ax_wave.transAxes,
                             ha='center', va='top', fontsize=10, color='lime')

    # Spectrogram
    spec_data = np.zeros((SPEC_FFT_SIZE//2, SPEC_N_COLS))
    im_spec = ax_spec.imshow(20*np.log10(spec_data+1e-6), aspect='auto', origin='lower', cmap='magma',
                             extent=[0, WINDOW_DURATION, 0, SR/2])
    ax_spec.set_ylim(0, 5000)
    ax_spec.set_xlim(0, WINDOW_DURATION)
    ax_spec.set_title("Scrolling Spectrogram")

    spec_highlight = ax_spec.add_patch(Polygon([[0,0],[0,0],[0,SR/2],[0,SR/2]], closed=True, color='cyan', alpha=0.2))

    stream = sd.OutputStream(channels=1, samplerate=SR, blocksize=blocksize, dtype='float32')
    stream.start()

    def update(frame):
        pitch_scale, emotion_factor = panel.get_values()
        end_idx = min(idx[0]+blocksize, n)
        raw_block = audio[idx[0]:end_idx]*emotion_factor
        block = pitch_shift_block(raw_block, pitch_scale)
        if len(block)<blocksize:
            block = np.pad(block,(0,blocksize-len(block)),'constant')
        else:
            block = block[:blocksize]
        stream.write(block)
        recorded_audio[idx[0]:idx[0]+len(block)] = block[:min(len(block), n-idx[0])]

        # Waveform scrolling
        win_start = max(0, idx[0]-WINDOW_SIZE//2)
        win_end = win_start + WINDOW_SIZE
        t_win = t[win_start:win_end]
        y_win = audio[win_start:win_end]
        line_wave.set_data(t_win-t_win[0], y_win)

        # Phoneme highlight
        current_phoneme=""
        for start,end,p in phoneme_ranges:
            if start <= idx[0] < end:
                current_phoneme=p
                ptype = PHONEME_TYPES.get(p,"silence")
                highlight_wave.set_xy(np.array([
                    [t[start]-t_win[0], -1],[t[end]-t_win[0], -1],
                    [t[end]-t_win[0], 1],[t[start]-t_win[0], 1]
                ]))
                highlight_wave.set_color(PHONEME_COLORS.get(ptype,"cyan"))
                break
        phoneme_text.set_text(current_phoneme)
        phoneme_text.set_color(PHONEME_COLORS.get(ptype,"white"))

        # Word highlight
        current_word=""
        for start,end,w in word_ranges:
            if start <= idx[0] < end:
                current_word = w
                break
        word_text.set_text(current_word)

        # Spectrogram update
        fft_block = np.fft.fft(block, SPEC_FFT_SIZE)
        spec_col = np.abs(fft_block[:SPEC_FFT_SIZE//2])
        spec_data[:, :-1] = spec_data[:, 1:]
        spec_data[:, -1] = spec_col
        im_spec.set_data(20*np.log10(spec_data+1e-6))
        im_spec.set_extent([0, WINDOW_DURATION, 0, SR/2])
        spec_highlight.set_color(PHONEME_COLORS.get(ptype,"cyan"))

        idx[0] += blocksize
        return line_wave, highlight_wave, im_spec, phoneme_text, word_text

    ani = animation.FuncAnimation(fig, update, interval=30, blit=False)
    plt.show()
    stream.stop()

    # Save WAV
    wav_file = wave.open("recorded_demo.wav",'wb')
    wav_file.setnchannels(1)
    wav_file.setsampwidth(2)
    wav_file.setframerate(SR)
    wav_file.writeframes((recorded_audio*32767).astype(np.int16).tobytes())
    wav_file.close()
    print("Recording saved to recorded_demo.wav")


if __name__=="__main__":
    app = QApplication(sys.argv)
    panel = TTSControlPanel()
    panel.show()

    demo_ssml="""
<voice name="bright" emotion="happy">Hello there!</voice>
<prosody rate="1.1" volume="1.0">This is a real-time interactive FFT TTS demo.</prosody>
<voice name="deep" emotion="angry">Watch out!</voice>
<voice name="robot" emotion="neutral">Beep boop.</voice>
"""
    panel.text_input.setPlainText(demo_ssml)

    panel.play_button.clicked.connect(lambda: play_demo(panel))
    sys.exit(app.exec())
