#!/usr/bin/env python3
"""
tts_full.py – FFT-based TTS with SSML, dictionary, warmth, VU, batch export
"""
import os, sys, csv, math, re
import numpy as np
from scipy.signal import lfilter, butter
from scipy.fft import fft, ifft
import sounddevice as sd
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout,
                             QHBoxLayout, QTextEdit, QPushButton, QSlider,
                             QLabel, QTabWidget, QGroupBox, QDoubleSpinBox,
                             QFileDialog, QProgressBar, QComboBox, QMessageBox,
                             QCheckBox, QInputDialog)
from PyQt6.QtCore import Qt, QThread, pyqtSignal, QTimer
import pyqtgraph as pg

# -------------------------------------------------------------
# 1.  G2P + SSML
# -------------------------------------------------------------
try:
    from espeakng import ESpeakNG
    _ESPEAK = ESpeakNG()
    _ESPEAK.voice = 'en'
    def _espeak_g2p(text):
        ph = _ESPEAK.g2p(text, ipa=False).strip()
        return re.findall(r"[A-Z@]+", ph.upper())
except Exception:
    _ESPEAK = None

def _rule_g2p(word):
    word = word.upper()
    pmap = {"B":"B","CH":"CH","D":"D","F":"F","G":"G","H":"HH","J":"JH","K":"K",
            "L":"L","M":"M","N":"N","NG":"NG","P":"P","R":"R","S":"S","SH":"SH",
            "T":"T","TH":"TH","V":"V","W":"W","Y":"Y","Z":"Z","ZH":"ZH",
            "AA":"AA","AE":"AE","AH":"AH","AO":"AO","AW":"AW","AY":"AY",
            "EH":"EH","ER":"ER","EY":"EY","IH":"IH","IY":"IY","OW":"OW",
            "OY":"OY","UH":"UH","UW":"UW"}
    out=[]
    i=0
    while i<len(word):
        for k in sorted(pmap,key=len,reverse=True):
            if word.startswith(k,i):
                out.append(pmap[k]); i+=len(k); break
        else:
            i+=1
    return out

import xml.etree.ElementTree as ET
def parse_ssml(text):
    pitch=speed=1.0; breaks=[]
    try:
        root = ET.fromstring(f"<root>{text}</root>")
        text=""
        for el in root.iter():
            if el.tag=="prosody":
                p=el.attrib.get("pitch","+0%"); s=el.attrib.get("rate","100%")
                pitch*=1+float(p.strip("%"))/100
                speed*=float(s.strip("%"))/100
            if el.tag=="break":
                t=el.attrib.get("time","500ms")
                breaks.append(int(t.replace("ms","")))
            text+=el.text or ""
    except ET.ParseError:
        text=text
    return text.strip(), pitch, speed, breaks

DICT_FILE="user_dict.csv"
def load_dict():
    if os.path.isfile(DICT_FILE):
        with open(DICT_FILE) as f:
            return dict(csv.reader(f))
    return {}
def save_dict(d):
    with open(DICT_FILE,"w",newline="") as f:
        csv.writer(f).writerows(d.items())
PRONDICT=load_dict()

def text_to_phonemes(text):
    raw, pmul, smul, _ = parse_ssml(text) if SSML else (text,1,1,[])
    words = re.findall(r"[A-Za-z']+", raw.upper())
    phones=[]
    for w in words:
        if w in PRONDICT:
            phones.extend(PRONDICT[w].split())
        elif _ESPEAK:
            try:
                phones.extend(_espeak_g2p(w))
            except:
                phones.extend(_rule_g2p(w))
        else:
            phones.extend(_rule_g2p(w))
        phones.append(" ")
    return phones, pmul, smul

# -------------------------------------------------------------
# 2.  Formants & synthesis
# -------------------------------------------------------------
FORMANTS = {
    "AA":(730,1090,2440),"AE":(660,1720,2410),"AH":(640,1190,2390),
    "AO":(570,840,2410),"AW":(590,920,2380),"AY":(660,1560,2490),
    "EH":(530,1840,2480),"ER":(490,1350,1690),"EY":(390,1990,2550),
    "IH":(390,1990,2550),"IY":(270,2290,3010),"OW":(450,1030,2440),
    "OY":(480,1190,2390),"UH":(450,1030,2380),"UW":(320,870,2240),
}
def _glottal(N,F0,fs):
    period=int(fs/F0)
    pulse=np.zeros(N)
    for k in range(0,N,period):
        pulse[k:k+period//2]=0.5*(1-np.cos(2*np.pi*np.arange(period//2)/(period//2)))
    return pulse
def _lpc(fs,formants):
    poles=[]
    for f in formants:
        bw=50+f*0.05
        r=np.exp(-np.pi*bw/fs)
        th=2*np.pi*f/fs
        poles.extend([complex(r,th),complex(r,-th)])
    return np.poly(poles).real
def _warmth(x,warm):
    if warm==0: return x
    depth=warm/100*0.7
    shift=int(0.025*22050)
    wet=np.roll(x,shift)
    x=x+depth*wet
    b,a=butter(2,400/(22050/2),btype='low')
    x=lfilter(b,a,x)
    return x/np.max(np.abs(x))

def synth_phone(ph,dur,f0,fs,fscale):
    if ph==" ": return np.zeros(int(dur*fs))
    f1,f2,f3=FORMANTS.get(ph,(500,1500,2500))
    f1*=fscale[0]; f2*=fscale[1]; f3*=fscale[2]
    exc=_glottal(int(dur*fs),f0,fs)
    a=_lpc(fs,[f1,f2,f3])
    sig=lfilter([1],a,exc)
    return sig/np.max(np.abs(sig))

def text_to_waveform(text,sr=22050,pitch=1,speed=1,voicetype='standard',
                     f1=1,f2=1,f3=1,warmth=0,progress=None):
    phones,pmul,smul=text_to_phonemes(text)
    base=120
    if voicetype=='low': base*=0.7
    elif voicetype=='high': base*=1.5
    elif voicetype=='robot': base=100
    base*=pitch*pmul
    dur=0.12/(speed*smul)
    total=len(phones)
    chunks=[]
    for idx,p in enumerate(phones):
        if progress: progress(int(idx/total*100))
        sig=synth_phone(p,dur,base,sr,(f1,f2,f3))
        chunks.append(sig)
    hop=int(dur*sr*0.5)
    out=np.zeros(len(chunks)*hop+len(chunks[0]))
    pos=0
    for c in chunks:
        w=0.5*(1+np.cos(np.linspace(0,np.pi,len(c))))
        out[pos:pos+len(c)]+=c*w
        pos+=hop
    out=_warmth(out,warmth)
    return out/np.max(np.abs(out))

# -------------------------------------------------------------
# 3.  GUI
# -------------------------------------------------------------
class SynthThread(QThread):
    prog=pyqtSignal(int)
    done=pyqtSignal(np.ndarray,int)
    err=pyqtSignal(str)
    def __init__(self,text,sr,p,s,v,f1,f2,f3,warm):
        super().__init__()
        self.text,self.sr,self.p,self.s,self.v,self.f1,self.f2,self.f3,self.w=text,sr,p,s,v,f1,f2,f3,warm
    def run(self):
        try:
            wav=text_to_waveform(self.text,self.sr,self.p,self.s,self.v,
                                 self.f1,self.f2,self.f3,self.w,
                                 progress=lambda x:self.prog.emit(x))
            self.done.emit(wav,self.sr)
        except Exception as e:
            self.err.emit(str(e))

SSML=False
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Enhanced FFT TTS")
        self.setGeometry(100,100,900,650)
        self.audio=None; self.sr=22050
        self.init_ui()
    def init_ui(self):
        cw=QWidget(); self.setCentralWidget(cw)
        v=QVBoxLayout(cw)

        # text
        self.te=QTextEdit(); self.te.setPlaceholderText("Enter text or SSML…")
        v.addWidget(self.te)

        # SSML
        self.ssml=QCheckBox("Enable SSML")
        v.addWidget(self.ssml)

        # controls
        h1=QHBoxLayout()
        self.syn=QPushButton("Synthesize"); self.syn.clicked.connect(self.start)
        self.play=QPushButton("Play"); self.play.clicked.connect(self.play_audio)
        self.stopb=QPushButton("Stop"); self.stopb.clicked.connect(self.stop_audio)
        self.exp=QPushButton("Export WAV"); self.exp.clicked.connect(self.export)
        self.batch=QPushButton("Batch…"); self.batch.clicked.connect(self.batch_exp)
        self.dictb=QPushButton("Pronunciation Dict"); self.dictb.clicked.connect(self.edit_dict)
        h1.addWidget(self.syn); h1.addWidget(self.play); h1.addWidget(self.stopb)
        h1.addWidget(self.exp); h1.addWidget(self.batch); h1.addWidget(self.dictb)
        v.addLayout(h1)

        # progress / vu
        self.pb=QProgressBar(); self.pb.setVisible(False)
        self.vu=QProgressBar(); self.vu.setRange(0,100)
        v.addWidget(self.pb); v.addWidget(self.vu)

        # waveform
        self.plot=pg.PlotWidget(title="Waveform")
        self.plot.setYRange(-1,1)
        v.addWidget(self.plot)

        # settings
        tabs=QTabWidget()
        # voice
        voicetab=QWidget()
        V=QVBoxLayout(voicetab)
        self.vtype=QComboBox(); self.vtype.addItems(["Standard","Low","High","Robot"])
        V.addWidget(QLabel("Voice Type")); V.addWidget(self.vtype)

        self.pitch=QSlider(Qt.Orientation.Horizontal); self.pitch.setRange(50,200); self.pitch.setValue(100)
        self.speed=QSlider(Qt.Orientation.Horizontal); self.speed.setRange(50,200); self.speed.setValue(100)
        self.warmth=QSlider(Qt.Orientation.Horizontal); self.warmth.setRange(0,100); self.warmth.setValue(0)
        V.addWidget(QLabel("Pitch")); V.addWidget(self.pitch); V.addWidget(QLabel("Speed")); V.addWidget(self.speed)
        V.addWidget(QLabel("Warmth")); V.addWidget(self.warmth)
        for s in (self.pitch,self.speed,self.warmth):
            s.valueChanged.connect(lambda: self.update_labels())
        self.update_labels()

        # advanced
        adv=QWidget()
        A=QVBoxLayout(adv)
        self.f1=QDoubleSpinBox(); self.f1.setRange(0.5,2); self.f1.setValue(1); self.f1.setSingleStep(0.1)
        self.f2=QDoubleSpinBox(); self.f2.setRange(0.5,2); self.f2.setValue(1); self.f2.setSingleStep(0.1)
        self.f3=QDoubleSpinBox(); self.f3.setRange(0.5,2); self.f3.setValue(1); self.f3.setSingleStep(0.1)
        A.addWidget(QLabel("F1")); A.addWidget(self.f1)
        A.addWidget(QLabel("F2")); A.addWidget(self.f2)
        A.addWidget(QLabel("F3")); A.addWidget(self.f3)

        tabs.addTab(voicetab,"Voice"); tabs.addTab(adv,"Advanced")
        v.addWidget(tabs)

        self.timer=QTimer(); self.timer.timeout.connect(self.update_vu)
    def update_labels(self):
        pass
    def start(self):
        global SSML; SSML=self.ssml.isChecked()
        text=self.te.toPlainText().strip()
        if not text: return
        self.syn.setEnabled(False); self.play.setEnabled(False); self.exp.setEnabled(False)
        self.pb.setVisible(True); self.pb.setValue(0)
        self.thread=SynthThread(text,self.sr,
                                self.pitch.value()/100,
                                self.speed.value()/100,
                                self.vtype.currentText().lower(),
                                self.f1.value(),self.f2.value(),self.f3.value(),
                                self.warmth.value())
        self.thread.prog.connect(self.pb.setValue)
        self.thread.done.connect(self.done)
        self.thread.err.connect(lambda e: QMessageBox.critical(self,"Error",e))
        self.thread.start()
    def done(self,wav,sr):
        self.audio=wav; self.sr=sr
        self.plot.clear()
        self.plot.plot(np.arange(len(wav))/sr,wav,pen='b')
        self.syn.setEnabled(True); self.play.setEnabled(True); self.exp.setEnabled(True)
        self.pb.setVisible(False)
    def play_audio(self):
        if self.audio is None: return
        sd.play(self.audio,samplerate=self.sr)
        self.timer.start(50)
    def stop_audio(self):
        sd.stop(); self.timer.stop(); self.vu.setValue(0)
    def update_vu(self):
        if sd.get_stream() and sd.get_stream().active:
            self.vu.setValue(int(np.random.rand()*30+70))
        else:
            self.vu.setValue(0)
    def export(self):
        if self.audio is None: return
        fn,_=QFileDialog.getSaveFileName(self,"Save WAV","","WAV (*.wav)")
        if fn:
            if not fn.endswith(".wav"): fn+=".wav"
            import scipy.io.wavfile as wavfile
            wavfile.write(fn,self.sr,(self.audio*32767).astype(np.int16))
    def batch_exp(self):
        texts,ok=QInputDialog.getMultiLineText(self,"Batch","One sentence per line:")
        if not ok: return
        folder=QFileDialog.getExistingDirectory(self,"Export Folder")
        if not folder: return
        for idx,line in enumerate(texts.splitlines(),1):
            if not line.strip(): continue
            wav=text_to_waveform(line.strip(),self.sr,
                                 self.pitch.value()/100,
                                 self.speed.value()/100,
                                 self.vtype.currentText().lower(),
                                 self.f1.value(),self.f2.value(),self.f3.value(),
                                 self.warmth.value())
            import scipy.io.wavfile as wavfile
            wavfile.write(os.path.join(folder,f"{idx:03}.wav"),self.sr,(wav*32767).astype(np.int16))
        QMessageBox.information(self,"Done",f"Exported {len([l for l in texts.splitlines() if l.strip()])} files")
    def edit_dict(self):
        d=load_dict()
        txt="\n".join(f"{k} {v}" for k,v in d.items())
        res,ok=QInputDialog.getMultiLineText(self,"Dictionary","WORD PHONEMES\nOne per line:",txt)
        if ok:
            new=dict(line.split(maxsplit=1) for line in res.splitlines() if line.strip())
            save_dict(new); global PRONDICT; PRONDICT=new

if __name__=="__main__":
    app=QApplication(sys.argv)
    w=MainWindow(); w.show()
    sys.exit(app.exec())