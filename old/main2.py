import numpy as np
import scipy.io.wavfile as wav
from g2p_en import G2p
import re
from typing import List, Dict, Tuple, Optional
import matplotlib.pyplot as plt
from dataclasses import dataclass

@dataclass
class PhonemeParams:
    areas: List[float]
    duration: float
    f0_contour: Optional[List[float]] = None
    amplitude: float = 1.0

class EnhancedIPASynthesizer:
    def __init__(self, sample_rate: int = 22050):
        self.sample_rate = sample_rate
        self.g2p = G2p()
        
        # Enhanced vocal tract tube models with more precise area functions
        self.phoneme_params = self._create_enhanced_phoneme_params()
        
        # Improved duration model
        self.phoneme_durations = {
            'short_vowel': 0.15, 'long_vowel': 0.25, 'diphthong': 0.3,
            'stop': 0.08, 'fricative': 0.12, 'affricate': 0.18,
            'nasal': 0.18, 'liquid': 0.16, 'glide': 0.14,
            'silence': 0.1
        }
        
        # Phoneme classification
        self.phoneme_categories = {
            'short_vowel': ['ɪ', 'ɛ', 'æ', 'ʊ', 'ʌ', 'ə'],
            'long_vowel': ['i', 'e', 'ɑ', 'ɔ', 'u', 'ɝ'],
            'diphthong': ['aɪ', 'eɪ', 'ɔɪ', 'aʊ', 'oʊ'],
            'stop': ['p', 'b', 't', 'd', 'k', 'g'],
            'fricative': ['f', 'v', 'θ', 'ð', 's', 'z', 'ʃ', 'ʒ', 'h'],
            'nasal': ['m', 'n', 'ŋ'],
            'liquid': ['l', 'r'],
            'glide': ['j', 'w']
        }

    def _create_enhanced_phoneme_params(self) -> Dict[str, PhonemeParams]:
        """Create enhanced vocal tract parameters for each phoneme."""
        params = {}
        
        # Vowels with more accurate area functions based on research data
        params['i'] = PhonemeParams([1.0, 1.3, 2.0, 3.5, 6.0, 8.0, 7.5, 6.0], 0.25)  # beet
        params['ɪ'] = PhonemeParams([1.2, 1.6, 2.5, 4.0, 6.5, 8.5, 7.8, 6.5], 0.15)  # bit
        params['e'] = PhonemeParams([1.5, 2.0, 3.0, 4.5, 7.0, 8.5, 7.0, 5.5], 0.25)  # bait
        params['ɛ'] = PhonemeParams([2.0, 2.5, 3.5, 5.0, 7.5, 8.0, 6.5, 5.0], 0.15)  # bet
        params['æ'] = PhonemeParams([2.5, 3.0, 4.0, 5.5, 8.0, 8.5, 6.0, 4.5], 0.15)  # bat
        params['ɑ'] = PhonemeParams([3.5, 4.5, 6.0, 8.0, 9.5, 8.0, 5.5, 4.0], 0.25)  # father
        params['ɔ'] = PhonemeParams([2.8, 3.8, 5.5, 7.5, 8.5, 7.0, 5.0, 3.8], 0.25)  # caught
        params['ʊ'] = PhonemeParams([1.8, 2.2, 3.5, 5.5, 7.5, 8.0, 7.0, 6.0], 0.15)  # book
        params['u'] = PhonemeParams([1.2, 1.5, 2.5, 4.0, 6.0, 8.0, 7.8, 6.5], 0.25)  # boot
        params['ʌ'] = PhonemeParams([2.2, 2.8, 4.0, 6.0, 8.0, 8.2, 6.8, 5.5], 0.15)  # but
        params['ə'] = PhonemeParams([2.0, 2.5, 3.8, 5.8, 7.8, 8.0, 6.5, 5.2], 0.15)  # about
        params['ɝ'] = PhonemeParams([1.8, 2.2, 3.5, 5.5, 7.5, 8.2, 7.2, 6.0], 0.25)  # bird
        
        # Diphthongs with dynamic area transitions
        params['aɪ'] = PhonemeParams([3.0, 4.0, 5.5, 7.0, 8.5, 8.0, 6.0, 4.5], 0.3)
        params['eɪ'] = PhonemeParams([1.8, 2.3, 3.5, 5.0, 7.0, 8.5, 7.2, 5.8], 0.3)
        params['oʊ'] = PhonemeParams([1.5, 2.0, 3.2, 5.0, 7.0, 8.5, 7.5, 6.2], 0.3)
        
        # Consonants
        params['p'] = PhonemeParams([0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1], 0.08)
        params['b'] = PhonemeParams([0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1], 0.08)
        params['t'] = PhonemeParams([0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1], 0.08)
        params['d'] = PhonemeParams([0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1], 0.08)
        params['k'] = PhonemeParams([0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1], 0.08)
        params['g'] = PhonemeParams([0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1], 0.08)
        
        params['s'] = PhonemeParams([0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2], 0.12)
        params['z'] = PhonemeParams([0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2], 0.12)
        params['ʃ'] = PhonemeParams([0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3], 0.12)
        params['ʒ'] = PhonemeParams([0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3], 0.12)
        params['h'] = PhonemeParams([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0], 0.1)
        
        params['m'] = PhonemeParams([0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1], 0.18)
        params['n'] = PhonemeParams([0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1], 0.18)
        params['ŋ'] = PhonemeParams([0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1], 0.18)
        
        params['l'] = PhonemeParams([0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4], 0.16)
        params['r'] = PhonemeParams([0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5], 0.16)
        params['w'] = PhonemeParams([0.7, 0.7, 0.7, 0.7, 0.7, 0.7, 0.7, 0.7], 0.14)
        params['j'] = PhonemeParams([0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6], 0.14)
        
        return params

    def _enhanced_normalize_text(self, text: str) -> str:
        """Enhanced text normalization with more comprehensive rules."""
        text = text.lower()
        
        # Expand abbreviations
        abbreviation_map = {
            r'\bdr\.': 'doctor',
            r'\bmr\.': 'mister',
            r'\bmrs\.': 'missus',
            r'\bms\.': 'miss',
            r'\bprof\.': 'professor',
            r'\betc\.': 'et cetera',
            r'\bvs\.': 'versus',
            r'\bst\.': 'saint',
            r'\bave\.': 'avenue',
            r'\bblvd\.': 'boulevard',
            r'\binc\.': 'incorporated',
            r'\bltd\.': 'limited',
        }
        
        for pattern, replacement in abbreviation_map.items():
            text = re.sub(pattern, replacement, text)
        
        # Number expansion
        text = self._expand_numbers(text)
        
        # Currency and symbols
        symbol_map = {
            r'\$(\d+)': r'\1 dollars',
            r'€(\d+)': r'\1 euros',
            r'£(\d+)': r'\1 pounds',
            r'%': 'percent',
            r'&': 'and',
        }
        
        for pattern, replacement in symbol_map.items():
            text = re.sub(pattern, replacement, text)
        
        return text

    def _expand_numbers(self, text: str) -> str:
        """Expand numbers to words with improved handling."""
        def number_to_words(num):
            """Convert number to words (0-999)."""
            units = ['', 'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine']
            teens = ['ten', 'eleven', 'twelve', 'thirteen', 'fourteen', 'fifteen', 
                    'sixteen', 'seventeen', 'eighteen', 'nineteen']
            tens = ['', 'ten', 'twenty', 'thirty', 'forty', 'fifty', 
                   'sixty', 'seventy', 'eighty', 'ninety']
            
            if num == 0:
                return 'zero'
            elif num < 10:
                return units[num]
            elif num < 20:
                return teens[num - 10]
            elif num < 100:
                return tens[num // 10] + (' ' + units[num % 10] if num % 10 != 0 else '')
            else:
                return units[num // 100] + ' hundred' + (' ' + number_to_words(num % 100) if num % 100 != 0 else '')
        
        def replace_number(match):
            num = int(match.group(0))
            if num < 1000:
                return number_to_words(num)
            else:
                return match.group(0)  # Return as digits for large numbers
        
        return re.sub(r'\b\d+\b', replace_number, text)

    def text_to_ipa_with_prosody(self, text: str) -> List[str]:
        """Convert text to IPA with enhanced prosody markers."""
        text = self._enhanced_normalize_text(text)
        phonemes = self.g2p(text)
        
        processed_phonemes = []
        word_count = 0
        sentence_end = False
        
        for i, phoneme in enumerate(phonemes):
            if phoneme == ' ':
                continue
            
            # Handle punctuation and prosody
            if phoneme in [',', ';']:
                processed_phonemes.append('<PAUSE:0.15>')
            elif phoneme in ['.', '!', '?']:
                processed_phonemes.append('<PAUSE:0.4>')
                sentence_end = True
            elif phoneme in ['ˈ', 'ˌ']:
                # Stress markers - we'll handle these during synthesis
                continue
            else:
                # Add word boundary markers
                if (i > 0 and phonemes[i-1] == ' ') or word_count == 0:
                    processed_phonemes.append('<WORD_START>')
                    word_count += 1
                
                processed_phonemes.append(phoneme)
                
                # Add sentence boundary marker
                if sentence_end:
                    processed_phonemes.append('<SENTENCE_END>')
                    sentence_end = False
        
        return processed_phonemes

    def _generate_enhanced_glottal_pulse(self, duration: float, F0: float, 
                                        jitter: float = 0.01, shimmer: float = 0.05) -> np.ndarray:
        """Generate glottal pulse with natural variations."""
        t = np.arange(0, duration, 1/self.sample_rate)
        pulse_train = np.zeros_like(t)
        
        # Add natural variations
        jitter_values = np.random.normal(0, jitter, len(t))
        shimmer_values = np.random.normal(1, shimmer, len(t))
        
        current_time = 0
        sample_index = 0
        
        while sample_index < len(t):
            # Add jitter to period
            current_period = (1.0 / F0) * (1 + jitter_values[sample_index])
            T1 = 0.4 * current_period
            T2 = 0.16 * current_period
            
            # Generate one pulse
            while sample_index < len(t) and t[sample_index] < current_time + current_period:
                time_in_period = t[sample_index] - current_time
                
                if time_in_period < T1:
                    pulse_val = 0.5 * (1 - np.cos(np.pi * time_in_period / T1))
                elif time_in_period < T1 + T2:
                    pulse_val = np.cos(np.pi * (time_in_period - T1) / (2 * T2))
                else:
                    pulse_val = 0.0
                
                # Add shimmer (amplitude variation)
                pulse_train[sample_index] = pulse_val * shimmer_values[sample_index]
                sample_index += 1
            
            current_time += current_period
        
        return pulse_train

    def _advanced_kelly_lochbaum(self, excitation: np.ndarray, areas: List[float], 
                                formant_transition: Optional[List[float]] = None) -> np.ndarray:
        """Enhanced Kelly-Lochbaum filter with formant transitions."""
        num_tubes = len(areas)
        N = len(excitation)
        output = np.zeros(N)
        
        # Dynamic area adjustment for formant transitions
        if formant_transition:
            area_factors = np.linspace(1.0, formant_transition[0], N)
            dynamic_areas = [areas.copy() for _ in range(N)]
            for i in range(N):
                for j in range(num_tubes):
                    dynamic_areas[i][j] = areas[j] * area_factors[i]
        else:
            dynamic_areas = [areas] * N
        
        # Initialize waves
        f = np.zeros(num_tubes)
        b = np.zeros(num_tubes)
        
        for n in range(N):
            current_areas = dynamic_areas[n]
            
            # Calculate reflection coefficients for current areas
            r = np.zeros(num_tubes - 1)
            for k in range(num_tubes - 1):
                r[k] = (current_areas[k + 1] - current_areas[k]) / (current_areas[k + 1] + current_areas[k])
            
            # Input excitation
            f[0] = excitation[n] + r[0] * b[0]
            
            # Propagate through tubes
            for k in range(1, num_tubes - 1):
                f[k] = (1 + r[k-1]) * f[k-1] - r[k-1] * b[k]
                b[k] = (1 - r[k]) * b[k+1] + r[k] * f[k]
            
            # Output
            f[num_tubes-1] = (1 + r[num_tubes-2]) * f[num_tubes-2]
            b[num_tubes-1] = 0
            
            # Radiation effect with improved filter
            if n > 1:
                output[n] = f[num_tubes-1] - 0.97 * output[n-1] + 0.3 * output[n-2]
            elif n > 0:
                output[n] = f[num_tubes-1] - 0.97 * output[n-1]
            else:
                output[n] = f[num_tubes-1]
            
            # Update backward waves
            for k in range(num_tubes - 2, 0, -1):
                b[k] = (1 - r[k]) * b[k+1] + r[k] * f[k]
            b[0] = (1 - r[0]) * b[1] + r[0] * f[0]
        
        return output

    def synthesize_with_prosody(self, phoneme: str, duration: float = 0.2, 
                               base_F0: float = 120, position: int = 0, 
                               total_phonemes: int = 1) -> np.ndarray:
        """Synthesize phoneme with prosodic variations."""
        if phoneme not in self.phoneme_params:
            phoneme = 'ə'  # Default to schwa
        
        params = self.phoneme_params[phoneme]
        
        # Dynamic F0 contour based on position
        f0_contour = self._generate_f0_contour(duration, base_F0, position, total_phonemes)
        
        # Generate excitation
        if self._is_voiced(phoneme):
            excitation = self._generate_enhanced_glottal_pulse(duration, base_F0)
        else:
            num_samples = int(duration * self.sample_rate)
            excitation = np.random.randn(num_samples) * 0.3
        
        # Apply vocal tract filter
        waveform = self._advanced_kelly_lochbaum(excitation, params.areas)
        
        # Apply dynamic amplitude envelope
        waveform = self._apply_dynamic_envelope(waveform, phoneme)
        
        return waveform

    def _generate_f0_contour(self, duration: float, base_F0: float, 
                            position: int, total_phonemes: int) -> np.ndarray:
        """Generate F0 contour with declination and emphasis."""
        num_samples = int(duration * self.sample_rate)
        t = np.linspace(0, duration, num_samples)
        
        # Base declination
        declination = 1.0 - 0.2 * (position / total_phonemes)
        
        # Add slight rising-falling pattern for vowels
        contour = base_F0 * declination * (1 + 0.1 * np.sin(2 * np.pi * t / duration))
        
        return contour

    def _is_voiced(self, phoneme: str) -> bool:
        """Check if phoneme is voiced."""
        voiceless = ['p', 't', 'k', 'f', 'θ', 's', 'ʃ', 'h']
        return phoneme not in voiceless

    def _apply_dynamic_envelope(self, waveform: np.ndarray, phoneme: str) -> np.ndarray:
        """Apply dynamic amplitude envelope based on phoneme type."""
        envelope = np.ones_like(waveform)
        num_samples = len(waveform)
        
        # Different envelope shapes for different phoneme types
        if self._is_vowel(phoneme):
            # Smooth bell-shaped envelope for vowels
            attack = int(0.1 * num_samples)
            release = int(0.2 * num_samples)
            sustain = num_samples - attack - release
            
            if attack > 0:
                envelope[:attack] = np.linspace(0, 1, attack)
            if sustain > 0:
                envelope[attack:attack+sustain] = 1.0
            if release > 0:
                envelope[attack+sustain:] = np.linspace(1, 0, release)
                
        elif self._is_fricative(phoneme):
            # Quick attack, sustained, quick release for fricatives
            attack = int(0.05 * num_samples)
            release = int(0.1 * num_samples)
            sustain = num_samples - attack - release
            
            if attack > 0:
                envelope[:attack] = np.linspace(0, 1, attack)
            if sustain > 0:
                envelope[attack:attack+sustain] = 1.0
            if release > 0:
                envelope[attack+sustain:] = np.linspace(1, 0, release)
        
        return waveform * envelope

    def _is_vowel(self, phoneme: str) -> bool:
        """Check if phoneme is a vowel."""
        vowels = ['i', 'ɪ', 'e', 'ɛ', 'æ', 'ɑ', 'ɔ', 'ʊ', 'u', 'ʌ', 'ə', 'ɝ']
        return phoneme in vowels

    def _is_fricative(self, phoneme: str) -> bool:
        """Check if phoneme is a fricative."""
        fricatives = ['f', 'v', 'θ', 'ð', 's', 'z', 'ʃ', 'ʒ', 'h']
        return phoneme in fricatives

# Example usage with enhanced features
if __name__ == "__main__":
    synthesizer = EnhancedIPASynthesizer()
    
    # Test enhanced text normalization
    test_texts = [
        "Hello world!",
        "Dr. Smith lives at 123 Main St.",
        "It costs $50, which is about 25% less.",
        "The meeting is at 3:30 PM, etc."
    ]
    
    for text in test_texts:
        print(f"Original: {text}")
        normalized = synthesizer._enhanced_normalize_text(text)
        print(f"Normalized: {normalized}")
        phonemes = synthesizer.text_to_ipa_with_prosody(text)
        print(f"Phonemes: {phonemes}")
        print()
    
    # Synthesize sample phrases
    sample_phrases = [
        "hello world",
        "test message",
        "articulatory synthesis"
    ]import numpy as np
import scipy.io.wavfile as wav
from g2p_en import G2p
import re
from typing import List, Dict, Tuple, Optional
import matplotlib.pyplot as plt
from dataclasses import dataclass

@dataclass
class PhonemeParams:
    areas: List[float]
    duration: float
    f0_contour: Optional[List[float]] = None
    amplitude: float = 1.0

class EnhancedIPASynthesizer:
    def __init__(self, sample_rate: int = 22050):
        self.sample_rate = sample_rate
        self.g2p = G2p()
        
        # Enhanced vocal tract tube models with more precise area functions
        self.phoneme_params = self._create_enhanced_phoneme_params()
        
        # Improved duration model
        self.phoneme_durations = {
            'short_vowel': 0.15, 'long_vowel': 0.25, 'diphthong': 0.3,
            'stop': 0.08, 'fricative': 0.12, 'affricate': 0.18,
            'nasal': 0.18, 'liquid': 0.16, 'glide': 0.14,
            'silence': 0.1
        }
        
        # Phoneme classification
        self.phoneme_categories = {
            'short_vowel': ['ɪ', 'ɛ', 'æ', 'ʊ', 'ʌ', 'ə'],
            'long_vowel': ['i', 'e', 'ɑ', 'ɔ', 'u', 'ɝ'],
            'diphthong': ['aɪ', 'eɪ', 'ɔɪ', 'aʊ', 'oʊ'],
            'stop': ['p', 'b', 't', 'd', 'k', 'g'],
            'fricative': ['f', 'v', 'θ', 'ð', 's', 'z', 'ʃ', 'ʒ', 'h'],
            'nasal': ['m', 'n', 'ŋ'],
            'liquid': ['l', 'r'],
            'glide': ['j', 'w']
        }

    def _create_enhanced_phoneme_params(self) -> Dict[str, PhonemeParams]:
        """Create enhanced vocal tract parameters for each phoneme."""
        params = {}
        
        # Vowels with more accurate area functions based on research data
        params['i'] = PhonemeParams([1.0, 1.3, 2.0, 3.5, 6.0, 8.0, 7.5, 6.0], 0.25)  # beet
        params['ɪ'] = PhonemeParams([1.2, 1.6, 2.5, 4.0, 6.5, 8.5, 7.8, 6.5], 0.15)  # bit
        params['e'] = PhonemeParams([1.5, 2.0, 3.0, 4.5, 7.0, 8.5, 7.0, 5.5], 0.25)  # bait
        params['ɛ'] = PhonemeParams([2.0, 2.5, 3.5, 5.0, 7.5, 8.0, 6.5, 5.0], 0.15)  # bet
        params['æ'] = PhonemeParams([2.5, 3.0, 4.0, 5.5, 8.0, 8.5, 6.0, 4.5], 0.15)  # bat
        params['ɑ'] = PhonemeParams([3.5, 4.5, 6.0, 8.0, 9.5, 8.0, 5.5, 4.0], 0.25)  # father
        params['ɔ'] = PhonemeParams([2.8, 3.8, 5.5, 7.5, 8.5, 7.0, 5.0, 3.8], 0.25)  # caught
        params['ʊ'] = PhonemeParams([1.8, 2.2, 3.5, 5.5, 7.5, 8.0, 7.0, 6.0], 0.15)  # book
        params['u'] = PhonemeParams([1.2, 1.5, 2.5, 4.0, 6.0, 8.0, 7.8, 6.5], 0.25)  # boot
        params['ʌ'] = PhonemeParams([2.2, 2.8, 4.0, 6.0, 8.0, 8.2, 6.8, 5.5], 0.15)  # but
        params['ə'] = PhonemeParams([2.0, 2.5, 3.8, 5.8, 7.8, 8.0, 6.5, 5.2], 0.15)  # about
        params['ɝ'] = PhonemeParams([1.8, 2.2, 3.5, 5.5, 7.5, 8.2, 7.2, 6.0], 0.25)  # bird
        
        # Diphthongs with dynamic area transitions
        params['aɪ'] = PhonemeParams([3.0, 4.0, 5.5, 7.0, 8.5, 8.0, 6.0, 4.5], 0.3)
        params['eɪ'] = PhonemeParams([1.8, 2.3, 3.5, 5.0, 7.0, 8.5, 7.2, 5.8], 0.3)
        params['oʊ'] = PhonemeParams([1.5, 2.0, 3.2, 5.0, 7.0, 8.5, 7.5, 6.2], 0.3)
        
        # Consonants
        params['p'] = PhonemeParams([0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1], 0.08)
        params['b'] = PhonemeParams([0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1], 0.08)
        params['t'] = PhonemeParams([0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1], 0.08)
        params['d'] = PhonemeParams([0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1], 0.08)
        params['k'] = PhonemeParams([0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1], 0.08)
        params['g'] = PhonemeParams([0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1], 0.08)
        
        params['s'] = PhonemeParams([0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2], 0.12)
        params['z'] = PhonemeParams([0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2], 0.12)
        params['ʃ'] = PhonemeParams([0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3], 0.12)
        params['ʒ'] = PhonemeParams([0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3], 0.12)
        params['h'] = PhonemeParams([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0], 0.1)
        
        params['m'] = PhonemeParams([0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1], 0.18)
        params['n'] = PhonemeParams([0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1], 0.18)
        params['ŋ'] = PhonemeParams([0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1], 0.18)
        
        params['l'] = PhonemeParams([0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4], 0.16)
        params['r'] = PhonemeParams([0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5], 0.16)
        params['w'] = PhonemeParams([0.7, 0.7, 0.7, 0.7, 0.7, 0.7, 0.7, 0.7], 0.14)
        params['j'] = PhonemeParams([0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6], 0.14)
        
        return params

    def _enhanced_normalize_text(self, text: str) -> str:
        """Enhanced text normalization with more comprehensive rules."""
        text = text.lower()
        
        # Expand abbreviations
        abbreviation_map = {
            r'\bdr\.': 'doctor',
            r'\bmr\.': 'mister',
            r'\bmrs\.': 'missus',
            r'\bms\.': 'miss',
            r'\bprof\.': 'professor',
            r'\betc\.': 'et cetera',
            r'\bvs\.': 'versus',
            r'\bst\.': 'saint',
            r'\bave\.': 'avenue',
            r'\bblvd\.': 'boulevard',
            r'\binc\.': 'incorporated',
            r'\bltd\.': 'limited',
        }
        
        for pattern, replacement in abbreviation_map.items():
            text = re.sub(pattern, replacement, text)
        
        # Number expansion
        text = self._expand_numbers(text)
        
        # Currency and symbols
        symbol_map = {
            r'\$(\d+)': r'\1 dollars',
            r'€(\d+)': r'\1 euros',
            r'£(\d+)': r'\1 pounds',
            r'%': 'percent',
            r'&': 'and',
        }
        
        for pattern, replacement in symbol_map.items():
            text = re.sub(pattern, replacement, text)
        
        return text

    def _expand_numbers(self, text: str) -> str:
        """Expand numbers to words with improved handling."""
        def number_to_words(num):
            """Convert number to words (0-999)."""
            units = ['', 'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine']
            teens = ['ten', 'eleven', 'twelve', 'thirteen', 'fourteen', 'fifteen', 
                    'sixteen', 'seventeen', 'eighteen', 'nineteen']
            tens = ['', 'ten', 'twenty', 'thirty', 'forty', 'fifty', 
                   'sixty', 'seventy', 'eighty', 'ninety']
            
            if num == 0:
                return 'zero'
            elif num < 10:
                return units[num]
            elif num < 20:
                return teens[num - 10]
            elif num < 100:
                return tens[num // 10] + (' ' + units[num % 10] if num % 10 != 0 else '')
            else:
                return units[num // 100] + ' hundred' + (' ' + number_to_words(num % 100) if num % 100 != 0 else '')
        
        def replace_number(match):
            num = int(match.group(0))
            if num < 1000:
                return number_to_words(num)
            else:
                return match.group(0)  # Return as digits for large numbers
        
        return re.sub(r'\b\d+\b', replace_number, text)

    def text_to_ipa_with_prosody(self, text: str) -> List[str]:
        """Convert text to IPA with enhanced prosody markers."""
        text = self._enhanced_normalize_text(text)
        phonemes = self.g2p(text)
        
        processed_phonemes = []
        word_count = 0
        sentence_end = False
        
        for i, phoneme in enumerate(phonemes):
            if phoneme == ' ':
                continue
            
            # Handle punctuation and prosody
            if phoneme in [',', ';']:
                processed_phonemes.append('<PAUSE:0.15>')
            elif phoneme in ['.', '!', '?']:
                processed_phonemes.append('<PAUSE:0.4>')
                sentence_end = True
            elif phoneme in ['ˈ', 'ˌ']:
                # Stress markers - we'll handle these during synthesis
                continue
            else:
                # Add word boundary markers
                if (i > 0 and phonemes[i-1] == ' ') or word_count == 0:
                    processed_phonemes.append('<WORD_START>')
                    word_count += 1
                
                processed_phonemes.append(phoneme)
                
                # Add sentence boundary marker
                if sentence_end:
                    processed_phonemes.append('<SENTENCE_END>')
                    sentence_end = False
        
        return processed_phonemes

    def _generate_enhanced_glottal_pulse(self, duration: float, F0: float, 
                                        jitter: float = 0.01, shimmer: float = 0.05) -> np.ndarray:
        """Generate glottal pulse with natural variations."""
        t = np.arange(0, duration, 1/self.sample_rate)
        pulse_train = np.zeros_like(t)
        
        # Add natural variations
        jitter_values = np.random.normal(0, jitter, len(t))
        shimmer_values = np.random.normal(1, shimmer, len(t))
        
        current_time = 0
        sample_index = 0
        
        while sample_index < len(t):
            # Add jitter to period
            current_period = (1.0 / F0) * (1 + jitter_values[sample_index])
            T1 = 0.4 * current_period
            T2 = 0.16 * current_period
            
            # Generate one pulse
            while sample_index < len(t) and t[sample_index] < current_time + current_period:
                time_in_period = t[sample_index] - current_time
                
                if time_in_period < T1:
                    pulse_val = 0.5 * (1 - np.cos(np.pi * time_in_period / T1))
                elif time_in_period < T1 + T2:
                    pulse_val = np.cos(np.pi * (time_in_period - T1) / (2 * T2))
                else:
                    pulse_val = 0.0
                
                # Add shimmer (amplitude variation)
                pulse_train[sample_index] = pulse_val * shimmer_values[sample_index]
                sample_index += 1
            
            current_time += current_period
        
        return pulse_train

    def _advanced_kelly_lochbaum(self, excitation: np.ndarray, areas: List[float], 
                                formant_transition: Optional[List[float]] = None) -> np.ndarray:
        """Enhanced Kelly-Lochbaum filter with formant transitions."""
        num_tubes = len(areas)
        N = len(excitation)
        output = np.zeros(N)
        
        # Dynamic area adjustment for formant transitions
        if formant_transition:
            area_factors = np.linspace(1.0, formant_transition[0], N)
            dynamic_areas = [areas.copy() for _ in range(N)]
            for i in range(N):
                for j in range(num_tubes):
                    dynamic_areas[i][j] = areas[j] * area_factors[i]
        else:
            dynamic_areas = [areas] * N
        
        # Initialize waves
        f = np.zeros(num_tubes)
        b = np.zeros(num_tubes)
        
        for n in range(N):
            current_areas = dynamic_areas[n]
            
            # Calculate reflection coefficients for current areas
            r = np.zeros(num_tubes - 1)
            for k in range(num_tubes - 1):
                r[k] = (current_areas[k + 1] - current_areas[k]) / (current_areas[k + 1] + current_areas[k])
            
            # Input excitation
            f[0] = excitation[n] + r[0] * b[0]
            
            # Propagate through tubes
            for k in range(1, num_tubes - 1):
                f[k] = (1 + r[k-1]) * f[k-1] - r[k-1] * b[k]
                b[k] = (1 - r[k]) * b[k+1] + r[k] * f[k]
            
            # Output
            f[num_tubes-1] = (1 + r[num_tubes-2]) * f[num_tubes-2]
            b[num_tubes-1] = 0
            
            # Radiation effect with improved filter
            if n > 1:
                output[n] = f[num_tubes-1] - 0.97 * output[n-1] + 0.3 * output[n-2]
            elif n > 0:
                output[n] = f[num_tubes-1] - 0.97 * output[n-1]
            else:
                output[n] = f[num_tubes-1]
            
            # Update backward waves
            for k in range(num_tubes - 2, 0, -1):
                b[k] = (1 - r[k]) * b[k+1] + r[k] * f[k]
            b[0] = (1 - r[0]) * b[1] + r[0] * f[0]
        
        return output

    def synthesize_with_prosody(self, phoneme: str, duration: float = 0.2, 
                               base_F0: float = 120, position: int = 0, 
                               total_phonemes: int = 1) -> np.ndarray:
        """Synthesize phoneme with prosodic variations."""
        if phoneme not in self.phoneme_params:
            phoneme = 'ə'  # Default to schwa
        
        params = self.phoneme_params[phoneme]
        
        # Dynamic F0 contour based on position
        f0_contour = self._generate_f0_contour(duration, base_F0, position, total_phonemes)
        
        # Generate excitation
        if self._is_voiced(phoneme):
            excitation = self._generate_enhanced_glottal_pulse(duration, base_F0)
        else:
            num_samples = int(duration * self.sample_rate)
            excitation = np.random.randn(num_samples) * 0.3
        
        # Apply vocal tract filter
        waveform = self._advanced_kelly_lochbaum(excitation, params.areas)
        
        # Apply dynamic amplitude envelope
        waveform = self._apply_dynamic_envelope(waveform, phoneme)
        
        return waveform

    def _generate_f0_contour(self, duration: float, base_F0: float, 
                            position: int, total_phonemes: int) -> np.ndarray:
        """Generate F0 contour with declination and emphasis."""
        num_samples = int(duration * self.sample_rate)
        t = np.linspace(0, duration, num_samples)
        
        # Base declination
        declination = 1.0 - 0.2 * (position / total_phonemes)
        
        # Add slight rising-falling pattern for vowels
        contour = base_F0 * declination * (1 + 0.1 * np.sin(2 * np.pi * t / duration))
        
        return contour

    def _is_voiced(self, phoneme: str) -> bool:
        """Check if phoneme is voiced."""
        voiceless = ['p', 't', 'k', 'f', 'θ', 's', 'ʃ', 'h']
        return phoneme not in voiceless

    def _apply_dynamic_envelope(self, waveform: np.ndarray, phoneme: str) -> np.ndarray:
        """Apply dynamic amplitude envelope based on phoneme type."""
        envelope = np.ones_like(waveform)
        num_samples = len(waveform)
        
        # Different envelope shapes for different phoneme types
        if self._is_vowel(phoneme):
            # Smooth bell-shaped envelope for vowels
            attack = int(0.1 * num_samples)
            release = int(0.2 * num_samples)
            sustain = num_samples - attack - release
            
            if attack > 0:
                envelope[:attack] = np.linspace(0, 1, attack)
            if sustain > 0:
                envelope[attack:attack+sustain] = 1.0
            if release > 0:
                envelope[attack+sustain:] = np.linspace(1, 0, release)
                
        elif self._is_fricative(phoneme):
            # Quick attack, sustained, quick release for fricatives
            attack = int(0.05 * num_samples)
            release = int(0.1 * num_samples)
            sustain = num_samples - attack - release
            
            if attack > 0:
                envelope[:attack] = np.linspace(0, 1, attack)
            if sustain > 0:
                envelope[attack:attack+sustain] = 1.0
            if release > 0:
                envelope[attack+sustain:] = np.linspace(1, 0, release)
        
        return waveform * envelope

    def _is_vowel(self, phoneme: str) -> bool:
        """Check if phoneme is a vowel."""
        vowels = ['i', 'ɪ', 'e', 'ɛ', 'æ', 'ɑ', 'ɔ', 'ʊ', 'u', 'ʌ', 'ə', 'ɝ']
        return phoneme in vowels

    def _is_fricative(self, phoneme: str) -> bool:
        """Check if phoneme is a fricative."""
        fricatives = ['f', 'v', 'θ', 'ð', 's', 'z', 'ʃ', 'ʒ', 'h']
        return phoneme in fricatives

# Example usage with enhanced features
if __name__ == "__main__":
    synthesizer = EnhancedIPASynthesizer()
    
    # Test enhanced text normalization
    test_texts = [
        "Hello world!",
        "Dr. Smith lives at 123 Main St.",
        "It costs $50, which is about 25% less.",
        "The meeting is at 3:30 PM, etc."
    ]
    
    for text in test_texts:
        print(f"Original: {text}")
        normalized = synthesizer._enhanced_normalize_text(text)
        print(f"Normalized: {normalized}")
        phonemes = synthesizer.text_to_ipa_with_prosody(text)
        print(f"Phonemes: {phonemes}")
        print()
    
    # Synthesize sample phrases
    sample_phrases = [
        "hello world",
        "test message",
        "articulatory synthesis"
    ]
    
    for i, phrase in enumerate(sample_phrases):
        output_file = f"enhanced_{i}.wav"
        print(f"Synthesizing: {phrase}")
        
        # Get phonemes with prosody
        phoneme_list = synthesizer.text_to_ipa_with_prosody(phrase)
        print(f"Phoneme sequence: {phoneme_list}")
        
        # Synthesize and save
        waveform = synthesizer.concatenate_phonemes(phoneme_list)
        waveform_int16 = np.int16(waveform * 32767)
        wav.write(output_file, synthesizer.sample_rate, waveform_int16)
        print(f"Saved to {output_file}\n")
    
    for i, phrase in enumerate(sample_phrases):
        output_file = f"enhanced_{i}.wav"
        print(f"Synthesizing: {phrase}")
        
        # Get phonemes with prosody
        phoneme_list = synthesizer.text_to_ipa_with_prosody(phrase)
        print(f"Phoneme sequence: {phoneme_list}")
        
        # Synthesize and save
        waveform = synthesizer.concatenate_phonemes(phoneme_list)
        waveform_int16 = np.int16(waveform * 32767)
        wav.write(output_file, synthesizer.sample_rate, waveform_int16)
        print(f"Saved to {output_file}\n")