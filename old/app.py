#!/usr/bin/env python3
"""
Unified FFT TTS Engine (PySide6)
=================================
Merges rule-based FFT formant synthesis, Kelly-Lochbaum vocal tract modeling,
comprehensive SSML parsing, multiple voice/emotion profiles, real-time scrolling
visualization with phoneme/word highlighting, pronunciation dictionary, batch
export, warmth effect, and a dark-themed PySide6 GUI.

Dependencies:
    pip install numpy scipy sounddevice PySide6 matplotlib g2p_en  (g2p_en optional)
"""

import os, sys, csv, math, re, wave
from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional
import numpy as np
from scipy.signal import lfilter, butter, resample

import matplotlib
matplotlib.use("QtAgg")
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QTextEdit, QPushButton, QSlider, QLabel, QDoubleSpinBox,
    QFileDialog, QProgressBar, QComboBox, QMessageBox, QInputDialog,
)
from PySide6.QtCore import Qt, QThread, Signal, QTimer
from PySide6.QtGui import QPalette, QColor

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

# ═══════════════════════════════════════════════════════════════
# 1. CONFIGURATION
# ═══════════════════════════════════════════════════════════════
SR = 22050
EPS = 1e-9
BLOCKSIZE = 1024
WINDOW_DUR = 0.5
WINDOW_SIZE = int(SR * WINDOW_DUR)
SPEC_FFT = 512
SPEC_COLS = int(WINDOW_DUR * SR / 128)

# ═══════════════════════════════════════════════════════════════
# 2. PHONEME DATABASE
# ═══════════════════════════════════════════════════════════════
FORMANTS = {
    "AA":((730,80),(1090,90),(2440,120)),"AE":((660,80),(1720,90),(2410,120)),
    "AH":((640,80),(1190,90),(2390,120)),"AO":((570,80),(840,80),(2410,120)),
    "AW":((590,80),(920,80),(2380,120)),  "AY":((660,80),(1560,90),(2490,120)),
    "EH":((530,70),(1840,90),(2480,120)), "ER":((490,80),(1350,100),(1690,140)),
    "EY":((390,70),(1990,90),(2550,120)), "IH":((390,70),(1990,90),(2550,120)),
    "IY":((270,60),(2290,100),(3010,150)),"OW":((450,80),(1030,80),(2440,120)),
    "OY":((480,80),(1190,90),(2390,120)), "UH":((450,80),(1030,80),(2380,120)),
    "UW":((320,60),(870,80),(2240,120)),
}
NASAL_FMT = {
    "M":((250,60),(1000,120),(2200,150)),
    "N":((300,60),(1200,120),(2400,150)),"NG":((300,60),(1200,120),(2400,150)),
}
APPROX_FMT = {
    "L":((350,70),(1700,120),(2600,150)),"R":((300,70),(1200,120),(2600,150)),
    "W":((350,70),(700,100),(2200,140)), "Y":((300,60),(2100,100),(2800,140)),
}
FRIC_BANDS = {
    "S":(4000,8000),"SH":(2000,5000),"F":(4000,6500),"TH":(3000,5500),
    "HH":(1000,4000),"Z":(4000,8000),"V":(4000,6500),"DH":(3000,5500),"ZH":(2000,5000),
}

PH_TYPE = {}
for _p in FORMANTS:    PH_TYPE[_p]="vowel"
for _p in NASAL_FMT:   PH_TYPE[_p]="nasal"
for _p in APPROX_FMT:  PH_TYPE[_p]="approximant"
for _p in FRIC_BANDS:  PH_TYPE[_p]="fricative"
for _p in list("PTKBGD"):  PH_TYPE[_p]="stop"

PH_COLOR = {"vowel":"#00e5ff","nasal":"#76ff03","approximant":"#69f0ae",
            "fricative":"#ffea00","stop":"#ff4081","silence":"#9e9e9e"}
PH_DUR = {"vowel":0.13,"nasal":0.11,"approximant":0.09,
          "fricative":0.09,"stop":0.06,"silence":0.05}

KL_AREAS = {
    "AA":[3.5,4.5,6.0,8.0,9.5,8.0,5.5,4.0],"AE":[2.5,3.0,4.0,5.5,8.0,8.5,6.0,4.5],
    "AH":[2.0,2.5,3.8,5.8,7.8,8.0,6.5,5.2],"AO":[2.8,3.8,5.5,7.5,8.5,7.0,5.0,3.8],
    "AW":[2.5,3.5,5.0,7.0,8.5,8.0,6.0,4.5],"AY":[2.0,3.0,5.0,7.0,8.5,8.0,6.0,4.5],
    "EH":[2.0,2.5,3.5,5.0,7.5,8.0,6.5,5.0],"ER":[1.8,2.2,3.5,5.5,7.5,8.2,7.2,6.0],
    "EY":[1.8,2.3,3.5,5.0,7.0,8.5,7.2,5.8],"IH":[1.2,1.6,2.5,4.0,6.5,8.5,7.8,6.5],
    "IY":[1.0,1.3,2.0,3.5,6.0,8.0,7.5,6.0],"OW":[1.5,2.0,3.2,5.0,7.0,8.5,7.5,6.2],
    "OY":[1.5,2.0,3.2,5.0,7.0,8.5,7.5,6.2],"UH":[1.8,2.2,3.5,5.5,7.5,8.0,7.0,6.0],
    "UW":[1.2,1.5,2.5,4.0,6.0,8.0,7.8,6.5],
}
for _c in list("PTKBGD"): KL_AREAS[_c]=[0.1]*8
for _c in list("SFZV"):    KL_AREAS[_c]=[0.2]*8
KL_AREAS["SH"]=KL_AREAS["ZH"]=[0.3]*8
KL_AREAS["HH"]=[1.0]*8
for _c in list("MN"):      KL_AREAS[_c]=[0.1]*8
KL_AREAS["NG"]=[0.1]*8
KL_AREAS["L"]=[0.4]*8; KL_AREAS["R"]=[0.5]*8
KL_AREAS["W"]=[0.7]*8; KL_AREAS["Y"]=[0.6]*8
KL_AREAS["TH"]=KL_AREAS["DH"]=[0.2]*8
KL_AREAS["CH"]=[0.2]*8; KL_AREAS["JH"]=[0.2]*8
for _p in FORMANTS:
    KL_AREAS.setdefault(_p, [2.0]*8)

# ═══════════════════════════════════════════════════════════════
# 3. VOICE & EMOTION PROFILES
# ═══════════════════════════════════════════════════════════════
VOICES = {
    "default":{"f0":125,"fscale":1.0,"bright":1.0,"breath":0.05},
    "male":   {"f0":110,"fscale":0.92,"bright":0.85,"breath":0.06},
    "female": {"f0":220,"fscale":1.08,"bright":1.2,"breath":0.12},
    "robot":  {"f0":140,"fscale":1.0,"bright":1.6,"breath":0.0},
    "child":  {"f0":300,"fscale":1.15,"bright":1.4,"breath":0.15},
    "deep":   {"f0":85,"fscale":0.85,"bright":0.8,"breath":0.03},
    "bright": {"f0":160,"fscale":1.1,"bright":1.3,"breath":0.08},
}
EMOTIONS = {
    "neutral":{"ps":1.0,"vol":1.0,"spd":1.0,"bright":1.0,"pause":1.0,"stress":1.0,"breath":1.0},
    "happy":  {"ps":1.2,"vol":1.15,"spd":1.1,"bright":1.2,"pause":0.8,"stress":1.25,"breath":1.1},
    "sad":    {"ps":0.85,"vol":0.85,"spd":0.85,"bright":0.85,"pause":1.4,"stress":0.8,"breath":1.3},
    "angry":  {"ps":1.1,"vol":1.3,"spd":1.2,"bright":1.4,"pause":0.6,"stress":1.5,"breath":0.8},
    "calm":   {"ps":0.95,"vol":0.9,"spd":0.9,"bright":0.95,"pause":1.2,"stress":0.9,"breath":1.1},
}

# ═══════════════════════════════════════════════════════════════
# 4. TEXT NORMALIZATION
# ═══════════════════════════════════════════════════════════════
_ABB = {r'\bdr\.':'doctor',r'\bmr\.':'mister',r'\bmrs\.':'missus',
        r'\bms\.':'miss',r'\bprof\.':'professor',r'\betc\.':'et cetera',
        r'\bvs\.':'versus',r'\bst\.':'saint',r'\binc\.':'incorporated',
        r'\bltd\.':'limited'}
_SYM = {r'\$(\d+)':r'\1 dollars',r'€(\d+)':r'\1 euros',r'£(\d+)':r'\1 pounds',
        r'%':'percent',r'&':'and'}
_UN = ['','one','two','three','four','five','six','seven','eight','nine']
_TE = ['ten','eleven','twelve','thirteen','fourteen','fifteen',
       'sixteen','seventeen','eighteen','nineteen']
_TN = ['','ten','twenty','thirty','forty','fifty','sixty','seventy','eighty','ninety']

def _n2w(n):
    if n==0: return 'zero'
    if n<10: return _UN[n]
    if n<20: return _TE[n-10]
    if n<100: return _TN[n//10]+(' '+_UN[n%10] if n%10 else '')
    return _UN[n//100]+' hundred'+(' '+_n2w(n%100) if n%100 else '')

def normalize_text(t):
    t=t.lower()
    for p,r in _ABB.items(): t=re.sub(p,r,t)
    for p,r in _SYM.items(): t=re.sub(p,r,t)
    t=re.sub(r'\b\d+\b',lambda m:_n2w(int(m.group())) if int(m.group())<1000 else m.group(),t)
    return t

# ═══════════════════════════════════════════════════════════════
# 5. GRAPHENE-TO-PHONEME
# ═══════════════════════════════════════════════════════════════
DICT_FILE="user_dict.csv"
def load_dict():
    if os.path.isfile(DICT_FILE):
        with open(DICT_FILE) as f: return dict(csv.reader(f))
    return {}
def save_dict(d):
    with open(DICT_FILE,"w",newline="") as f: csv.writer(f).writerows(d.items())
PRON_DICT=load_dict()

_IPA2ARPA={'i':'IY','ɪ':'IH','e':'EY','ɛ':'EH','æ':'AE','ɑ':'AA','ɔ':'AO',
           'ʊ':'UH','u':'UW','ʌ':'AH','ə':'AH','ɝ':'ER','aɪ':'AY','eɪ':'EY',
           'ɔɪ':'OY','aʊ':'AW','oʊ':'OW','p':'P','b':'B','t':'T','d':'D',
           'k':'K','g':'G','f':'F','v':'V','θ':'TH','ð':'DH','s':'S','z':'Z',
           'ʃ':'SH','ʒ':'ZH','h':'HH','m':'M','n':'N','ŋ':'NG','l':'L','r':'R',
           'w':'W','j':'Y'}

_DIG = [("NG","NG"),("SH","SH"),("CH","CH"),("TH","TH"),("WH","HH"),("PH","F"),
        ("CK","K"),("KN","N"),("WR","R"),("GN","N"),("TION","SH N"),("SION","ZH N"),
        ("EE","IY"),("EA","IY"),("OO","UW"),("OU","AW"),("OI","OY"),
        ("AI","EY"),("AY","EY"),("OW","OW"),("AW","AW"),("EW","UW"),
        ("IGH","IY"),("EIGH","EY"),("AR","AA R"),("ER","ER"),("IR","ER"),
        ("OR","AO R"),("UR","ER"),("AL","AO L"),("LE","AH L")]

def _rule_g2p(word):
    w=word.upper(); out=[]; i=0
    while i<len(w):
        hit=False
        for dg,ph in _DIG:
            if w.startswith(dg,i):
                if ph: out.extend(ph.split())
                i+=len(dg); hit=True; break
        if hit: continue
        c=w[i]
        if c in "AEIOU":
            if c=='E' and i==len(w)-1 and out: i+=1; continue
            vm={'A':'AE','E':'EH','I':'IH','O':'AA','U':'AH'}
            if i+2<len(w) and w[i+1] not in "AEIOU" and w[i+2]=='E':
                lm={'A':'EY','E':'IY','I':'IY','O':'OW','U':'UW'}
                out.append(lm.get(c,vm.get(c,'AH'))); i+=1
            else:
                out.append(vm.get(c,'AH'))
        elif c=='Y': out.append("IY" if i>0 else "AY")
        else:
            cm={'B':'B','C':'K','D':'D','F':'F','G':'G','H':'HH','J':'JH',
                'K':'K','L':'L','M':'M','N':'N','P':'P','Q':'K','R':'R',
                'S':'S','T':'T','V':'V','W':'W','X':'K S','Z':'Z'}
            if c in cm: out.extend(cm[c].split())
        i+=1
    return out or ['AH']

_g2p_cache = {}
def text_to_phonemes(text):
    text=normalize_text(text)
    words=re.findall(r"[A-Za-z']+",text.upper())
    phones=[]
    for w in words:
        if w in PRON_DICT:
            phones.extend(PRON_DICT[w].split())
        elif HAS_G2P:
            if w not in _g2p_cache:
                try:
                    ipa=G2p()(w)
                    mapped=[_IPA2ARPA.get(p,'AH') for p in ipa if p in _IPA2ARPA]
                    _g2p_cache[w]=mapped or _rule_g2p(w)
                except Exception:
                    _g2p_cache[w]=_rule_g2p(w)
            phones.extend(_g2p_cache[w])
        else:
            phones.extend(_rule_g2p(w))
        phones.append(" ")
    return phones

# ═══════════════════════════════════════════════════════════════
# 6. SSML PARSER
# ═══════════════════════════════════════════════════════════════
@dataclass
class Prosody:
    rate:float=1.0; pitch_st:float=0.0; volume:float=1.0
    is_break:bool=False; break_ms:int=0

@dataclass
class VoiceSpec:
    name:str="default"; emotion:str="neutral"

def _prate(v):
    if not v: return 1.0
    s=v.strip().lower()
    kw={"x-slow":0.6,"slow":0.8,"medium":1.0,"fast":1.25,"x-fast":1.5}
    if s in kw: return kw[s]
    m=re.match(r"([\d.]+)%$",s)
    return float(m.group(1))/100 if m else 1.0

def _ppitch(v):
    if not v: return 0.0
    s=v.strip().lower()
    kw={"x-low":-4,"low":-2,"medium":0,"high":2,"x-high":4}
    if s in kw: return float(kw[s])
    m=re.match(r"([+-]?[\d.]+)st$",s)
    if m: return float(m.group(1))
    m=re.match(r"([\d.]+)%$",s)
    if m:
        p=float(m.group(1))/100
        return 12*math.log2(p) if p>0 else 0
    return 0.0

def _pvol(v):
    if not v: return 1.0
    s=v.strip().lower()
    kw={"silent":0,"x-soft":0.25,"soft":0.5,"medium":1.0,"loud":1.5,"x-loud":2.0}
    if s in kw: return kw[s]
    m=re.match(r"([+-]?[\d.]+)db$",s)
    if m: return 10**(float(m.group(1))/20)
    m=re.match(r"([\d.]+)%$",s)
    return float(m.group(1))/100 if m else 1.0

def _ptime(v):
    if not v: return 0
    m=re.match(r"([\d.]+)ms$",v.strip().lower())
    if m: return int(float(m.group(1)))
    m=re.match(r"([\d.]+)s$",v.strip().lower())
    return int(float(m.group(1))*1000) if m else 0

import xml.etree.ElementTree as ET

def parse_ssml(text):
    s=text.strip()
    if not s: return []
    try:
        if not s.startswith("<"): s=f"<speak>{s}</speak>"
        root=ET.fromstring(s)
    except ET.ParseError:
        return [(text,VoiceSpec(),Prosody())]
    segs=[]
    def _rec(node,voice,pros):
        tag=node.tag.lower()
        if "break" in tag:
            ms=_ptime(node.attrib.get("time","0ms"))
            if not ms:
                st=node.attrib.get("strength","").lower()
                ms={"none":0,"x-weak":50,"weak":100,"medium":200,"strong":400,"x-strong":750}.get(st,200)
            segs.append(("",voice,Prosody(is_break=True,break_ms=ms)))
            return
        nv=voice
        if "voice" in tag:
            nv=VoiceSpec(name=node.attrib.get("name","default"),
                         emotion=node.attrib.get("emotion","neutral"))
        np_=pros
        if "prosody" in tag:
            np_=Prosody(rate=pros.rate*_prate(node.attrib.get("rate","")),
                        pitch_st=pros.pitch_st+_ppitch(node.attrib.get("pitch","")),
                        volume=pros.volume*_pvol(node.attrib.get("volume","")))
        if node.text and node.text.strip():
            segs.append((node.text.strip(),nv,np_))
        for c in node:
            _rec(c,nv,np_)
            if c.tail and c.tail.strip():
                segs.append((c.tail.strip(),nv,np_))
    _rec(root,VoiceSpec(),Prosody())
    return [(t,v,p) for t,v,p in segs if t or p.is_break]

# ═══════════════════════════════════════════════════════════════
# 7. AUDIO UTILITIES
# ═══════════════════════════════════════════════════════════════
def wnoise(dur,amp=0.3):
    return np.random.randn(max(1,int(SR*dur)))*amp

def _fades(x,ms=8):
    if x.size==0: return x
    n=len(x); m=max(1,int(SR*ms/1000))
    w=np.ones(n); r=np.linspace(0,1,m)
    w[:m]=r; w[-m:]=r[::-1]
    pk=np.max(np.abs(x*w))+EPS
    return 0.95*x*w/pk

def _ovlap(a,b,ms=8):
    if a.size==0: return b.copy()
    m=max(1,int(SR*ms/1000))
    out=np.zeros(len(a)+len(b)-m)
    out[:len(a)]=a
    f=np.linspace(0,1,m)
    out[len(a)-m:len(a)]=(1-f)*out[len(a)-m:len(a)]+f*b[:m]
    out[len(a):]=b[m:]
    return out

def _gauss_env(freqs,fmts,strength=1.0):
    env=np.full_like(freqs,1e-6)
    for fc,bw in fmts:
        sig=bw/(2*np.sqrt(2*np.log(2))+EPS)
        env+=strength*np.exp(-0.5*((freqs-fc)/(sig+EPS))**2)
    return env

def _bp_mask(freqs,flo,fhi,edge=300):
    return (np.clip((freqs-max(0,flo-edge))/(2*edge+EPS),0,1)
            *np.clip((fhi+edge-freqs)/(2*edge+EPS),0,1))

def _f0_curve(emo,t,f0):
    if t.size==0: return np.array([])
    mx=t.max()+EPS
    if emo=="happy":   return f0*(1.05+0.12*(t/mx)+0.03*np.sin(2*np.pi*3*t))
    if emo=="sad":     return f0*(0.98-0.18*(t/mx))
    if emo=="angry":   return f0*(1.0+0.18*np.sign(np.sin(2*np.pi*6*t)))
    if emo=="calm":    return f0*(1.0+0.03*np.sin(2*np.pi*1.5*t))
    return f0*np.ones_like(t)

def _warmth(x,w):
    if w==0 or x.size==0: return x
    d=w/100*0.7; sh=int(0.025*SR)
    x=x+d*np.roll(x,sh)
    b,a=butter(2,400/(SR/2),btype='low')
    x=lfilter(b,a,x)
    pk=np.max(np.abs(x))+EPS
    return x/pk

def _mlen(a,b):
    n=min(len(a),len(b)); return a[:n],b[:n]

# ═══════════════════════════════════════════════════════════════
# 8. FFT FORMANT SYNTHESIS
# ═══════════════════════════════════════════════════════════════
def _get_fmts(key):
    if key in FORMANTS: return FORMANTS[key]
    if key in NASAL_FMT: return NASAL_FMT[key]
    if key in APPROX_FMT: return APPROX_FMT[key]
    return FORMANTS["AH"]

def fft_voiced(key,dur,f0,vp,ep,emo="neutral"):
    dur=max(0.01,dur/max(EPS,ep["spd"]))
    n=max(1,int(SR*dur)); t=np.linspace(0,dur,n,False)
    f0s=f0*ep["ps"]*(2**(ep["pst"]/12))
    fc=_f0_curve(emo,t,f0s)
    if len(fc)!=n: fc=f0s*np.ones(n)
    exc=np.sin(2*np.pi*np.cumsum(fc)/SR)
    fmts=_get_fmts(key)
    sc=[(fc_*vp["fscale"],bw) for fc_,bw in fmts]
    fe=np.fft.rfft(exc); fr=np.fft.rfftfreq(len(exc),1.0/SR)
    br=vp["bright"]*ep["bright"]
    env=_gauss_env(fr,sc,br)
    shaped=np.fft.irfft(fe*env)
    breath=vp["breath"]*ep["breath"]
    if breath>0:
        wn=wnoise(dur,0.07); shaped,wn=_mlen(shaped,wn); shaped+=breath*wn
    return _fades(shaped)*ep["vol"]

def fft_fric(sym,dur,vp,ep):
    dur=max(0.01,dur/max(EPS,ep["spd"]))
    ns=wnoise(dur,0.5); fn=np.fft.rfft(ns); fr=np.fft.rfftfreq(len(ns),1.0/SR)
    flo,fhi=FRIC_BANDS.get(sym,(2000,8000))
    fs=vp["fscale"]
    env=_bp_mask(fr,flo*fs,fhi*fs)
    return _fades(np.fft.irfft(fn*env),6)*ep["vol"]

def fft_stop(sym,dur,vp,ep,emo="neutral"):
    dur=max(0.03,dur/max(EPS,ep["spd"])); tot=int(SR*dur)
    cn=int(tot*0.6); bn=min(tot-cn,int(SR*0.06)); rn=tot-cn-bn
    burst=wnoise(bn/SR,0.8)
    fb=np.fft.rfft(burst); fr=np.fft.rfftfreq(len(burst),1.0/SR)
    burst=np.fft.irfft(fb*(0.5+_bp_mask(fr,800,4500)))
    burst=_fades(burst,3)
    rel=np.zeros(rn)
    if rn>0:
        rel=fft_voiced("AH",rn/SR,vp["f0"],vp,ep,emo)*0.4
        rel=_fades(rel,5)
    return np.concatenate([np.zeros(cn),burst,rel])*ep["vol"]

# ═══════════════════════════════════════════════════════════════
# 9. KELLY-LOCHBAUM SYNTHESIS
# ═══════════════════════════════════════════════════════════════
def _kl_filter(exc,areas):
    nt=len(areas); N=len(exc); out=np.zeros(N)
    fw=np.zeros(nt); bw=np.zeros(nt)
    for n in range(N):
        r=np.zeros(nt-1)
        for k in range(nt-1):
            r[k]=(areas[k+1]-areas[k])/(areas[k+1]+areas[k]+EPS)
        fw[0]=exc[n]+r[0]*bw[0]
        for k in range(1,nt-1):
            fw[k]=(1+r[k-1])*fw[k-1]-r[k-1]*bw[k]
            bw[k]=(1-r[k])*bw[k+1]+r[k]*fw[k]
        fw[nt-1]=(1+r[nt-2])*fw[nt-2]
        if n>1:   out[n]=fw[nt-1]-0.97*out[n-1]+0.3*out[n-2]
        elif n>0: out[n]=fw[nt-1]-0.97*out[n-1]
        else:     out[n]=fw[nt-1]
        for k in range(nt-2,0,-1):
            bw[k]=(1-r[k])*bw[k+1]+r[k]*fw[k]
        bw[0]=(1-r[0])*bw[1]+r[0]*fw[0]
    return out

def kl_phoneme(ph,dur,f0,vp,ep,emo="neutral"):
    dur=max(0.01,dur/max(EPS,ep["spd"]))
    n=max(1,int(SR*dur)); t=np.linspace(0,dur,n,False)
    f0s=f0*ep["ps"]; fc=_f0_curve(emo,t,f0s)
    if len(fc)!=n: fc=f0s*np.ones(n)
    pt=PH_TYPE.get(ph,"silence")
    if pt=="fricative":
        exc=wnoise(dur,0.5)
    elif pt=="stop":
        cn=int(n*0.6); be=wnoise((n-cn)/SR,0.8)
        exc=np.concatenate([np.zeros(cn),be])
        exc=exc[:n] if len(exc)>=n else np.pad(exc,(0,n-len(exc)))
    else:
        exc=np.sin(2*np.pi*np.cumsum(fc)/SR)
    areas=[a*vp["fscale"] for a in KL_AREAS.get(ph,[2.0]*8)]
    out=_kl_filter(exc,areas)
    breath=vp["breath"]*ep["breath"]
    if breath>0 and pt in ("vowel","nasal","approximant"):
        wn=wnoise(dur,0.07); out,wn=_mlen(out,wn); out+=breath*wn
    return _fades(out)*ep["vol"]

# ═══════════════════════════════════════════════════════════════
# 10. HIGH-LEVEL SYNTHESIZER
# ═══════════════════════════════════════════════════════════════
def _synth_phones(phones,pros,vname,ename,engine):
    vp=VOICES.get(vname,VOICES["default"])
    ep=EMOTIONS.get(ename,EMOTIONS["neutral"]).copy()
    ep["vol"]*=pros.volume; ep["spd"]*=pros.rate
    ep["pst"]=ep.get("pst",0)+pros.pitch_st
    base_f0=vp["f0"]*ep["ps"]*(2**(ep["pst"]/12))
    blocks=[]; ph_rng=[]; ci=0
    words_ph=[]; cur=[]
    for p in phones:
        if p==" ":
            if cur: words_ph.append(cur); cur=[]
        else: cur.append(p)
    if cur: words_ph.append(cur)
    w_rng=[]
    for wp in words_ph:
        ws=ci
        for pi,ph in enumerate(wp):
            pt=PH_TYPE.get(ph,"silence")
            dur=PH_DUR.get(pt,0.08)
            if pi==0 and pt=="vowel": dur*=ep["stress"]
            if engine=="kl":
                sig=kl_phoneme(ph,dur,base_f0,vp,ep,ename)
            elif pt=="fricative":
                sig=fft_fric(ph,dur,vp,ep)
            elif pt=="stop":
                sig=fft_stop(ph,dur,vp,ep,ename)
            else:
                sig=fft_voiced(ph,dur,base_f0,vp,ep,ename)
            blocks.append(sig); ph_rng.append((ci,ci+len(sig),ph)); ci+=len(sig)
        w_rng.append((ws,ci,""))
        pause=np.zeros(int(SR*0.06*ep["pause"]))
        blocks.append(pause); ci+=len(pause)
    if not blocks: return np.zeros(0),[],[]
    return np.concatenate(blocks),ph_rng,w_rng

def synthesize_ssml(ssml_text,engine="fft",warmth=0,prog_cb=None):
    segs=parse_ssml(ssml_text)
    all_a=[]; all_ph=[]; all_w=[]; off=0; ts=len(segs)
    for si,(txt,vs,pr) in enumerate(segs):
        if prog_cb: prog_cb(int(si/ts*100))
        if pr.is_break:
            p=np.zeros(int(SR*pr.break_ms/1000))
            all_a.append(p); off+=len(p); continue
        phones=text_to_phonemes(txt)
        if not phones: continue
        a,ph,w=_synth_phones(phones,pr,vs.name,vs.emotion,engine)
        all_a.append(a)
        all_ph.extend([(s+off,e+off,p) for s,e,p in ph])
        all_w.extend([(s+off,e+off,ww) for s,e,ww in w])
        off+=len(a)
    if not all_a: return np.zeros(0),[],[]
    audio=np.concatenate(all_a)
    words=[]
    for txt,vs,pr in segs:
        if not pr.is_break and txt: words.extend(re.findall(r'\S+',txt))
    for i,(s,e,_) in enumerate(all_w):
        if i<len(words): all_w[i]=(s,e,words[i])
    if warmth>0: audio=_warmth(audio,warmth)
    pk=np.max(np.abs(audio))+EPS
    audio=audio*(0.95/pk)
    if prog_cb: prog_cb(100)
    return audio,all_ph,all_w

def write_wav(path,x,sr=SR):
    x16=np.clip(x,-1,1); x16=(x16*32767).astype(np.int16)
    with wave.open(path,"wb") as wf:
        wf.setnchannels(1); wf.setsampwidth(2); wf.setframerate(sr)
        wf.writeframes(x16.tobytes())

# ═══════════════════════════════════════════════════════════════
# 11. GUI  (PySide6)
# ═══════════════════════════════════════════════════════════════
class SynthThread(QThread):
    progress = Signal(int)
    done     = Signal(object, object, object)
    error    = Signal(str)

    def __init__(self,text,engine,warmth):
        super().__init__()
        self.text=text; self.engine=engine; self.warmth=warmth

    def run(self):
        try:
            a,ph,w=synthesize_ssml(self.text,self.engine,self.warmth,
                                   prog_cb=lambda x:self.progress.emit(x))
            self.done.emit(a,ph,w)
        except Exception as e:
            self.error.emit(str(e))


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Unified FFT TTS Engine")
        self.setGeometry(60,60,1080,820)
        self.audio=None; self.ph_ranges=[]; self.w_ranges=[]
        self.playing=False; self.play_idx=0; self.stream=None
        self._build()

    # ── build UI ──────────────────────────────────────────────
    def _build(self):
        cw=QWidget(); self.setCentralWidget(cw); mv=QVBoxLayout(cw)

        self.te=QTextEdit()
        self.te.setPlaceholderText("Enter text or SSML…")
        self.te.setMaximumHeight(110)
        mv.addWidget(self.te)

        r1=QHBoxLayout()
        r1.addWidget(QLabel("Voice:"))
        self.vc=QComboBox(); self.vc.addItems(list(VOICES.keys()))
        r1.addWidget(self.vc)
        r1.addWidget(QLabel("Emotion:"))
        self.ec=QComboBox(); self.ec.addItems(list(EMOTIONS.keys()))
        r1.addWidget(self.ec)
        r1.addWidget(QLabel("Engine:"))
        self.enc=QComboBox()
        self.enc.addItems(["FFT Formant","Kelly-Lochbaum"])
        r1.addWidget(self.enc)
        mv.addLayout(r1)

        r2=QHBoxLayout()
        for lbl,lo,hi,val,attr in [
            ("Pitch",50,200,100,"pitch_sl"),("Speed",50,200,100,"speed_sl"),
            ("Volume",0,200,100,"vol_sl"),("Warmth",0,100,0,"warm_sl")]:
            v=QVBoxLayout(); v.addWidget(QLabel(lbl))
            s=QSlider(Qt.Orientation.Horizontal)
            s.setRange(lo,hi); s.setValue(val)
            lb=QLabel(f"{val}%")
            s.valueChanged.connect(lambda x,l=lb:l.setText(f"{x}%"))
            v.addWidget(s); v.addWidget(lb); r2.addLayout(v)
            setattr(self,attr,s)
        mv.addLayout(r2)

        r3=QHBoxLayout()
        for i,(lbl,df) in enumerate([("F1",1.0),("F2",1.0),("F3",1.0)]):
            r3.addWidget(QLabel(f"{lbl}:"))
            sp=QDoubleSpinBox(); sp.setRange(0.5,2.0)
            sp.setValue(df); sp.setSingleStep(0.05)
            r3.addWidget(sp); setattr(self,f"f{i+1}",sp)
        mv.addLayout(r3)

        r4=QHBoxLayout()
        self.syn_btn=QPushButton("🔊 Synthesize")
        self.syn_btn.clicked.connect(self._synth)
        self.play_btn=QPushButton("▶ Play")
        self.play_btn.clicked.connect(self._play)
        self.play_btn.setEnabled(False)
        self.stop_btn=QPushButton("⏹ Stop")
        self.stop_btn.clicked.connect(self._stop)
        self.exp_btn=QPushButton("💾 Export")
        self.exp_btn.clicked.connect(self._export)
        self.exp_btn.setEnabled(False)
        self.bat_btn=QPushButton("📦 Batch")
        self.bat_btn.clicked.connect(self._batch)
        self.dic_btn=QPushButton("📖 Dict")
        self.dic_btn.clicked.connect(self._dict)
        for b in (self.syn_btn,self.play_btn,self.stop_btn,
                  self.exp_btn,self.bat_btn,self.dic_btn):
            r4.addWidget(b)
        mv.addLayout(r4)

        r5=QHBoxLayout()
        self.pb=QProgressBar(); self.pb.setVisible(False)
        r5.addWidget(self.pb)
        self.vu=QProgressBar(); self.vu.setRange(0,100)
        self.vu.setMaximumWidth(160); self.vu.setFormat("VU %v%")
        r5.addWidget(self.vu)
        mv.addLayout(r5)

        self.stat=QLabel("Ready")
        self.stat.setStyleSheet(
            "padding:4px;background:#1a1a2e;color:#e0e0e0;border-radius:4px;")
        mv.addWidget(self.stat)

        self.fig=Figure(figsize=(10,4),facecolor='#1a1a2e')
        self.canvas=FigureCanvas(self.fig)
        self.canvas.setMinimumHeight(260)
        self.ax_w=self.fig.add_subplot(211)
        self.ax_s=self.fig.add_subplot(212)
        self.fig.subplots_adjust(hspace=0.45,left=0.06,right=0.98,top=0.94,bottom=0.08)
        for ax in (self.ax_w,self.ax_s):
            ax.set_facecolor('#0d1117')
            ax.tick_params(colors='#888')
            for sp in ax.spines.values(): sp.set_color('#333')
        self.ax_w.set_title("Waveform",color='#ccc',fontsize=10)
        self.ax_w.set_ylim(-1,1)
        self.ax_s.set_title("Spectrogram",color='#ccc',fontsize=10)
        self.ax_s.set_ylim(0,5000)
        mv.addWidget(self.canvas)

        self.ptimer=QTimer(); self.ptimer.timeout.connect(self._tick)
        self.te.setPlainText(DEMO_SSML)

    # ── helpers ───────────────────────────────────────────────
    def _eng(self):
        return "kl" if "kelly" in self.enc.currentText().lower() else "fft"

    def _wrap_prosody(self,text):
        ps=self.pitch_sl.value()/100
        sp=self.speed_sl.value()/100
        vo=self.vol_sl.value()/100
        st=12*math.log2(ps) if ps>0 else 0
        return (f'<prosody pitch="{st:+.1f}st" '
                f'rate="{sp*100:.0f}%" volume="{vo*100:.0f}%">'
                f'{text}</prosody>')

    # ── synthesis ─────────────────────────────────────────────
    def _synth(self):
        text=self.te.toPlainText().strip()
        if not text:
            QMessageBox.warning(self,"Warning","Enter text first.")
            return
        if not text.startswith("<"):
            text=self._wrap_prosody(text)
        else:
            text=self._wrap_prosody(text)
        self.syn_btn.setEnabled(False); self.play_btn.setEnabled(False)
        self.exp_btn.setEnabled(False); self.pb.setVisible(True)
        self.pb.setValue(0); self.stat.setText("Synthesizing…")
        self.th=SynthThread(text,self._eng(),self.warm_sl.value())
        self.th.progress.connect(self.pb.setValue)
        self.th.done.connect(self._on_done)
        self.th.error.connect(self._on_err)
        self.th.start()

    def _on_done(self,a,ph,w):
        self.audio=a; self.ph_ranges=ph; self.w_ranges=w
        self._draw_static()
        self.syn_btn.setEnabled(True); self.play_btn.setEnabled(True)
        self.exp_btn.setEnabled(True); self.pb.setVisible(False)
        d=len(a)/SR
        self.stat.setText(
            f"Done: {d:.2f}s | {len(ph)} phonemes | {len(w)} words")

    def _on_err(self,msg):
        QMessageBox.critical(self,"Error",msg)
        self.syn_btn.setEnabled(True); self.pb.setVisible(False)
        self.stat.setText("Error")

    # ── static waveform drawing ───────────────────────────────
    def _draw_static(self):
        a=self.audio; ax=self.ax_w
        ax.clear(); ax.set_facecolor('#0d1117')
        ax.set_title("Waveform",color='#ccc',fontsize=10); ax.set_ylim(-1,1)
        t=np.arange(len(a))/SR
        for s,e,p in self.ph_ranges:
            pt=PH_TYPE.get(p,"silence"); c=PH_COLOR.get(pt,"#888")
            se=min(e,len(a)-1)
            ax.axvspan(t[s],t[se],color=c,alpha=0.2)
        ax.plot(t,a,color='#4fc3f7',linewidth=0.5)
        self.ph_txt=ax.text(0.5,1.08,"",transform=ax.transAxes,ha='center',
                            va='bottom',fontsize=11,color='white',fontweight='bold')
        self.wd_txt=ax.text(0.5,-0.18,"",transform=ax.transAxes,ha='center',
                            va='top',fontsize=10,color='#76ff03')
        self.canvas.draw()

    # ── live playback ─────────────────────────────────────────
    def _play(self):
        if self.audio is None: return
        if not HAS_SD:
            QMessageBox.warning(self,"Warning","pip install sounddevice"); return
        self.playing=True; self.play_idx=0
        self.stream=sd.OutputStream(
            channels=1,samplerate=SR,blocksize=BLOCKSIZE,dtype='float32')
        self.stream.start(); self.ptimer.start(30)
        self.play_btn.setEnabled(False); self.syn_btn.setEnabled(False)
        self.stat.setText("Playing…")

        ax=self.ax_w; ax.clear(); ax.set_facecolor('#0d1117')
        ax.set_title("Waveform ▶",color='#ccc',fontsize=10)
        ax.set_ylim(-1,1); ax.set_xlim(0,WINDOW_DUR)
        self.wline,=ax.plot([],[],color='#4fc3f7',linewidth=0.8)
        self.hl_patch=None
        self.ph_txt=ax.text(0.5,1.08,"",transform=ax.transAxes,ha='center',
                            va='bottom',fontsize=11,color='white',fontweight='bold')
        self.wd_txt=ax.text(0.5,-0.18,"",transform=ax.transAxes,ha='center',
                            va='top',fontsize=10,color='#76ff03')

        axs=self.ax_s; axs.clear(); axs.set_facecolor('#0d1117')
        axs.set_title("Spectrogram ▶",color='#ccc',fontsize=10)
        axs.set_ylim(0,5000); axs.set_xlim(0,WINDOW_DUR)
        self.live_spec=np.zeros((SPEC_FFT//2,SPEC_COLS))
        self.simg=axs.imshow(20*np.log10(self.live_spec+1e-6),
                             aspect='auto',origin='lower',cmap='magma',
                             extent=[0,WINDOW_DUR,0,SR/2],
                             interpolation='bilinear')
        self.canvas.draw()

    def _tick(self):
        if not self.playing or self.audio is None:
            self._stop(); return
        n=len(self.audio)
        ei=min(self.play_idx+BLOCKSIZE,n)
        blk=self.audio[self.play_idx:ei]
        vol=self.vol_sl.value()/100
        blk=(blk*vol).astype(np.float32)
        if len(blk)<BLOCKSIZE:
            blk=np.pad(blk,(0,BLOCKSIZE-len(blk)))
        else:
            blk=blk[:BLOCKSIZE]
        self.stream.write(blk)

        ws=max(0,self.play_idx-WINDOW_SIZE//2); we=ws+WINDOW_SIZE
        tw=np.arange(ws,min(we,n))/SR; yw=self.audio[ws:min(we,n)]
        self.wline.set_data(tw-tw[0] if len(tw) else [],yw)
        self.ax_w.set_xlim(0,WINDOW_DUR)

        cph=""; cpt="silence"
        for s,e,p in self.ph_ranges:
            if s<=self.play_idx<e:
                cph=p; cpt=PH_TYPE.get(p,"silence")
                c=PH_COLOR.get(cpt,"#888")
                ts_=s/SR-ws/SR; te_=e/SR-ws/SR
                if self.hl_patch:
                    self.hl_patch.remove()
                self.hl_patch=self.ax_w.axvspan(ts_,te_,color=c,alpha=0.3)
                break
        self.ph_txt.set_text(f"[{cph}]" if cph else "")
        self.ph_txt.set_color(PH_COLOR.get(cpt,"white"))

        cw=""
        for s,e,w in self.w_ranges:
            if s<=self.play_idx<e: cw=w; break
        self.wd_txt.set_text(f'"{cw}"' if cw else "")

        fb=np.fft.fft(blk,SPEC_FFT)
        sc=np.abs(fb[:SPEC_FFT//2])
        self.live_spec[:,:-1]=self.live_spec[:,1:]
        self.live_spec[:,-1]=sc
        self.simg.set_data(20*np.log10(self.live_spec+1e-6))

        rms=np.sqrt(np.mean(blk**2))
        self.vu.setValue(int(min(100,rms*500)))
        self.canvas.draw_idle()

        self.play_idx+=BLOCKSIZE
        if self.play_idx>=n: self._stop()

    def _stop(self):
        self.playing=False; self.ptimer.stop()
        if self.stream:
            try: self.stream.stop(); self.stream.close()
            except Exception: pass
            self.stream=None
        self.vu.setValue(0)
        self.play_btn.setEnabled(self.audio is not None)
        self.syn_btn.setEnabled(True)
        if self.audio is not None:
            self.stat.setText("Stopped"); self._draw_static()

    # ── export / batch / dict ─────────────────────────────────
    def _export(self):
        if self.audio is None: return
        fn,_=QFileDialog.getSaveFileName(self,"Save WAV","","WAV (*.wav)")
        if fn:
            if not fn.endswith(".wav"): fn+=".wav"
            write_wav(fn,self.audio)
            self.stat.setText(f"Exported: {fn}")

    def _batch(self):
        txt,ok=QInputDialog.getMultiLineText(
            self,"Batch","One sentence per line:")
        if not ok: return
        fld=QFileDialog.getExistingDirectory(self,"Folder")
        if not fld: return
        cnt=0
        for i,line in enumerate(txt.splitlines(),1):
            if not line.strip(): continue
            try:
                wrapped=self._wrap_prosody(line.strip())
                a,_,_=synthesize_ssml(wrapped,self._eng(),self.warm_sl.value())
                write_wav(os.path.join(fld,f"{i:03}.wav"),a)
                cnt+=1
            except Exception as e:
                self.stat.setText(f"Error line {i}: {e}"); return
        self.stat.setText(f"Batch: {cnt} files → {fld}")
        QMessageBox.information(self,"Done",f"Exported {cnt} files.")

    def _dict(self):
        global PRON_DICT
        d=load_dict()
        txt="\n".join(f"{k} {v}" for k,v in d.items())
        res,ok=QInputDialog.getMultiLineText(
            self,"Pronunciation Dictionary",
            "WORD PHONEMES (one per line)\ne.g. hello HH AH L OW",txt)
        if ok:
            nd={}
            for ln in res.splitlines():
                pts=ln.strip().split(maxsplit=1)
                if len(pts)==2: nd[pts[0]]=pts[1]
            save_dict(nd); PRON_DICT=nd
            self.stat.setText(f"Dict: {len(nd)} entries")

    def closeEvent(self,event):
        self._stop(); event.accept()

# ═══════════════════════════════════════════════════════════════
# 12. DEMO SSML & MAIN
# ═══════════════════════════════════════════════════════════════
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
  </prosody>
</voice>"""


def main():
    app=QApplication(sys.argv)
    app.setStyle("Fusion")
    pal=QPalette()
    pal.setColor(QPalette.ColorRole.Window,        QColor(26,26,46))
    pal.setColor(QPalette.ColorRole.WindowText,    QColor(224,224,224))
    pal.setColor(QPalette.ColorRole.Base,          QColor(13,17,23))
    pal.setColor(QPalette.ColorRole.AlternateBase, QColor(30,30,50))
    pal.setColor(QPalette.ColorRole.Text,          QColor(224,224,224))
    pal.setColor(QPalette.ColorRole.Button,        QColor(40,40,65))
    pal.setColor(QPalette.ColorRole.ButtonText,    QColor(224,224,224))
    pal.setColor(QPalette.ColorRole.Highlight,     QColor(79,195,247))
    pal.setColor(QPalette.ColorRole.HighlightedText, QColor(0,0,0))
    app.setPalette(pal)
    w=MainWindow(); w.show()
    sys.exit(app.exec())


if __name__=="__main__":
    main()