"""Text normalization and grapheme-to-phoneme conversion."""

import os
import csv
import re
from config import HAS_G2P, EPS

# ═══════════════════════════════════════════════════════════════
# TEXT NORMALIZATION
# ═══════════════════════════════════════════════════════════════
_ABB = {
    r'\bdr\.': 'doctor', r'\bmr\.': 'mister', r'\bmrs\.': 'missus',
    r'\bms\.': 'miss', r'\bprof\.': 'professor', r'\betc\.': 'et cetera',
    r'\bvs\.': 'versus', r'\bst\.': 'saint', r'\binc\.': 'incorporated',
    r'\bltd\.': 'limited',
}

_SYM = {
    r'\$(\d+)': r'\1 dollars', r'€(\d+)': r'\1 euros',
    r'£(\d+)': r'\1 pounds', r'%': 'percent', r'&': 'and',
}

_UN = ['', 'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine']
_TE = ['ten', 'eleven', 'twelve', 'thirteen', 'fourteen', 'fifteen',
       'sixteen', 'seventeen', 'eighteen', 'nineteen']
_TN = ['', 'ten', 'twenty', 'thirty', 'forty', 'fifty',
       'sixty', 'seventy', 'eighty', 'ninety']


def _n2w(n):
    if n == 0: return 'zero'
    if n < 10: return _UN[n]
    if n < 20: return _TE[n - 10]
    if n < 100: return _TN[n // 10] + (' ' + _UN[n % 10] if n % 10 else '')
    return _UN[n // 100] + ' hundred' + (' ' + _n2w(n % 100) if n % 100 else '')


def normalize_text(t):
    """Normalize text for TTS processing."""
    t = t.lower()
    for p, r in _ABB.items():
        t = re.sub(p, r, t)
    for p, r in _SYM.items():
        t = re.sub(p, r, t)
    t = re.sub(r'\b\d+\b', 
               lambda m: _n2w(int(m.group())) if int(m.group()) < 1000 else m.group(), 
               t)
    return t


# ═══════════════════════════════════════════════════════════════
# PRONUNCIATION DICTIONARY
# ═══════════════════════════════════════════════════════════════
DICT_FILE = "user_dict.csv"


def load_dict():
    """Load user pronunciation dictionary."""
    if os.path.isfile(DICT_FILE):
        with open(DICT_FILE) as f:
            return dict(csv.reader(f))
    return {}


def save_dict(d):
    """Save user pronunciation dictionary."""
    with open(DICT_FILE, "w", newline="") as f:
        csv.writer(f).writerows(d.items())


PRON_DICT = load_dict()


def update_pron_dict(new_dict):
    """Update global pronunciation dictionary."""
    global PRON_DICT
    save_dict(new_dict)
    PRON_DICT = new_dict


# ═══════════════════════════════════════════════════════════════
# GRAPHENE-TO-PHONEME
# ═══════════════════════════════════════════════════════════════
_IPA2ARPA = {
    'i': 'IY', 'ɪ': 'IH', 'e': 'EY', 'ɛ': 'EH', 'æ': 'AE', 'ɑ': 'AA',
    'ɔ': 'AO', 'ʊ': 'UH', 'u': 'UW', 'ʌ': 'AH', 'ə': 'AH', 'ɝ': 'ER',
    'aɪ': 'AY', 'eɪ': 'EY', 'ɔɪ': 'OY', 'aʊ': 'AW', 'oʊ': 'OW',
    'p': 'P', 'b': 'B', 't': 'T', 'd': 'D', 'k': 'K', 'g': 'G',
    'f': 'F', 'v': 'V', 'θ': 'TH', 'ð': 'DH', 's': 'S', 'z': 'Z',
    'ʃ': 'SH', 'ʒ': 'ZH', 'h': 'HH', 'm': 'M', 'n': 'N', 'ŋ': 'NG',
    'l': 'L', 'r': 'R', 'w': 'W', 'j': 'Y',
}

_DIGRAPH_RULES = [
    ("NG", "NG"), ("SH", "SH"), ("CH", "CH"), ("TH", "TH"), ("WH", "HH"),
    ("PH", "F"), ("CK", "K"), ("KN", "N"), ("WR", "R"), ("GN", "N"),
    ("TION", "SH N"), ("SION", "ZH N"), ("EE", "IY"), ("EA", "IY"),
    ("OO", "UW"), ("OU", "AW"), ("OI", "OY"), ("AI", "EY"), ("AY", "EY"),
    ("OW", "OW"), ("AW", "AW"), ("EW", "UW"), ("IGH", "IY"), ("EIGH", "EY"),
    ("AR", "AA R"), ("ER", "ER"), ("IR", "ER"), ("OR", "AO R"), ("UR", "ER"),
    ("AL", "AO L"), ("LE", "AH L"),
]


def _rule_g2p(word):
    """Simple rule-based grapheme-to-phoneme conversion."""
    w = word.upper()
    out = []
    i = 0
    while i < len(w):
        hit = False
        for dg, ph in _DIGRAPH_RULES:
            if w.startswith(dg, i):
                if ph:
                    out.extend(ph.split())
                i += len(dg)
                hit = True
                break
        if hit:
            continue
        c = w[i]
        if c in "AEIOU":
            if c == 'E' and i == len(w) - 1 and out:
                i += 1
                continue
            vm = {'A': 'AE', 'E': 'EH', 'I': 'IH', 'O': 'AA', 'U': 'AH'}
            if i + 2 < len(w) and w[i + 1] not in "AEIOU" and w[i + 2] == 'E':
                lm = {'A': 'EY', 'E': 'IY', 'I': 'IY', 'O': 'OW', 'U': 'UW'}
                out.append(lm.get(c, vm.get(c, 'AH')))
                i += 1
            else:
                out.append(vm.get(c, 'AH'))
        elif c == 'Y':
            out.append("IY" if i > 0 else "AY")
        else:
            cm = {
                'B': 'B', 'C': 'K', 'D': 'D', 'F': 'F', 'G': 'G', 'H': 'HH',
                'J': 'JH', 'K': 'K', 'L': 'L', 'M': 'M', 'N': 'N', 'P': 'P',
                'Q': 'K', 'R': 'R', 'S': 'S', 'T': 'T', 'V': 'V', 'W': 'W',
                'X': 'K S', 'Z': 'Z',
            }
            if c in cm:
                out.extend(cm[c].split())
        i += 1
    return out or ['AH']


_g2p_cache = {}


def text_to_phonemes(text):
    """Convert text to phoneme sequence."""
    text = normalize_text(text)
    words = re.findall(r"[A-Za-z']+", text.upper())
    phones = []
    for w in words:
        if w in PRON_DICT:
            phones.extend(PRON_DICT[w].split())
        elif HAS_G2P:
            if w not in _g2p_cache:
                try:
                    from g2p_en import G2p
                    ipa = G2p()(w)
                    mapped = [_IPA2ARPA.get(p, 'AH') for p in ipa if p in _IPA2ARPA]
                    _g2p_cache[w] = mapped or _rule_g2p(w)
                except Exception:
                    _g2p_cache[w] = _rule_g2p(w)
            phones.extend(_g2p_cache[w])
        else:
            phones.extend(_rule_g2p(w))
        phones.append(" ")
    return phones


def clear_g2p_cache():
    """Clear the G2P cache."""
    global _g2p_cache
    _g2p_cache = {}