"""Dialog windows for the TTS Engine."""

from PySide6.QtWidgets import (
    QDialog, QDialogButtonBox, QFormLayout, QSpinBox, QDoubleSpinBox,
    QCheckBox, QVBoxLayout, QLabel, QMessageBox, QTextBrowser, QTabWidget,
    QWidget
)
from PySide6.QtCore import Qt

from phonemes import PH_TYPE, PH_COLOR, FORMANTS, NASAL_FMT, APPROX_FMT, FRIC_BANDS, PH_DUR


# ═══════════════════════════════════════════════════════════════
# SETTINGS DIALOG
# ═══════════════════════════════════════════════════════════════
class SettingsDialog(QDialog):
    """Application settings dialog."""
    
    def __init__(self, current_settings, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Settings")
        self.setMinimumWidth(400)
        self.settings = current_settings.copy()
        self._setup_ui()
    
    def _setup_ui(self):
        layout = QFormLayout(self)
        
        # Audio settings
        self.sr_spin = QSpinBox()
        self.sr_spin.setRange(8000, 48000)
        self.sr_spin.setSingleStep(1000)
        self.sr_spin.setValue(self.settings.get('sample_rate', 22050))
        layout.addRow("Sample Rate:", self.sr_spin)
        
        self.block_spin = QSpinBox()
        self.block_spin.setRange(256, 4096)
        self.block_spin.setSingleStep(256)
        self.block_spin.setValue(self.settings.get('blocksize', 1024))
        layout.addRow("Block Size:", self.block_spin)
        
        # Display settings
        self.spec_fft_spin = QSpinBox()
        self.spec_fft_spin.setRange(256, 2048)
        self.spec_fft_spin.setSingleStep(128)
        self.spec_fft_spin.setValue(self.settings.get('spec_fft', 512))
        layout.addRow("Spectrogram FFT:", self.spec_fft_spin)
        
        self.wave_dur_spin = QDoubleSpinBox()
        self.wave_dur_spin.setRange(0.1, 2.0)
        self.wave_dur_spin.setSingleStep(0.1)
        self.wave_dur_spin.setValue(self.settings.get('wave_dur', 0.5))
        layout.addRow("Waveform Window (s):", self.wave_dur_spin)
        
        # Synthesis settings
        self.default_f0_spin = QSpinBox()
        self.default_f0_spin.setRange(50, 400)
        self.default_f0_spin.setValue(self.settings.get('default_f0', 125))
        layout.addRow("Default F0 (Hz):", self.default_f0_spin)
        
        self.breath_spin = QDoubleSpinBox()
        self.breath_spin.setRange(0.0, 0.5)
        self.breath_spin.setSingleStep(0.01)
        self.breath_spin.setValue(self.settings.get('breathiness', 0.05))
        layout.addRow("Breathiness:", self.breath_spin)
        
        # Buttons
        btns = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        btns.accepted.connect(self.accept)
        btns.rejected.connect(self.reject)
        layout.addRow(btns)

    def get_settings(self):
        return {
            'sample_rate': self.sr_spin.value(),
            'blocksize': self.block_spin.value(),
            'spec_fft': self.spec_fft_spin.value(),
            'wave_dur': self.wave_dur_spin.value(),
            'default_f0': self.default_f0_spin.value(),
            'breathiness': self.breath_spin.value(),
        }


# ═══════════════════════════════════════════════════════════════
# AUDIO EFFECTS DIALOG
# ═══════════════════════════════════════════════════════════════
class EffectsDialog(QDialog):
    """Audio effects configuration dialog."""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Audio Effects")
        self.setMinimumWidth(350)
        self._setup_ui()
    
    def _setup_ui(self):
        layout = QFormLayout(self)
        
        # Reverb
        self.reverb_check = QCheckBox("Enable Reverb")
        self.reverb_decay = QDoubleSpinBox()
        self.reverb_decay.setRange(0.1, 0.99)
        self.reverb_decay.setSingleStep(0.05)
        self.reverb_decay.setValue(0.5)
        self.reverb_mix = QDoubleSpinBox()
        self.reverb_mix.setRange(0.0, 1.0)
        self.reverb_mix.setSingleStep(0.05)
        self.reverb_mix.setValue(0.3)
        
        layout.addRow(self.reverb_check)
        layout.addRow("  Decay:", self.reverb_decay)
        layout.addRow("  Mix:", self.reverb_mix)
        
        # Echo
        self.echo_check = QCheckBox("Enable Echo")
        self.echo_delay = QDoubleSpinBox()
        self.echo_delay.setRange(0.05, 1.0)
        self.echo_delay.setSingleStep(0.05)
        self.echo_delay.setValue(0.25)
        self.echo_feedback = QDoubleSpinBox()
        self.echo_feedback.setRange(0.0, 0.9)
        self.echo_feedback.setSingleStep(0.05)
        self.echo_feedback.setValue(0.4)
        
        layout.addRow(self.echo_check)
        layout.addRow("  Delay (s):", self.echo_delay)
        layout.addRow("  Feedback:", self.echo_feedback)
        
        # EQ
        self.eq_check = QCheckBox("Enable EQ")
        self.eq_low = QDoubleSpinBox()
        self.eq_low.setRange(-12, 12)
        self.eq_low.setValue(0)
        self.eq_mid = QDoubleSpinBox()
        self.eq_mid.setRange(-12, 12)
        self.eq_mid.setValue(0)
        self.eq_high = QDoubleSpinBox()
        self.eq_high.setRange(-12, 12)
        self.eq_high.setValue(0)
        
        layout.addRow(self.eq_check)
        layout.addRow("  Low (dB):", self.eq_low)
        layout.addRow("  Mid (dB):", self.eq_mid)
        layout.addRow("  High (dB):", self.eq_high)
        
        # Buttons
        btns = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        btns.accepted.connect(self.accept)
        btns.rejected.connect(self.reject)
        layout.addRow(btns)
    
    def apply_effects(self, audio):
        """Apply configured effects to audio."""
        from audio_utils import apply_reverb, apply_echo, apply_eq, normalize_audio
        import numpy as np
        
        result = audio.copy()
        
        if self.reverb_check.isChecked():
            result = apply_reverb(result, self.reverb_decay.value(), self.reverb_mix.value())
        
        if self.echo_check.isChecked():
            result = apply_echo(result, self.echo_delay.value(), self.echo_feedback.value())
        
        if self.eq_check.isChecked():
            result = apply_eq(result, self.eq_low.value(), self.eq_mid.value(), self.eq_high.value())
        
        return normalize_audio(result)


# ═══════════════════════════════════════════════════════════════
# HELP DIALOGS
# ═══════════════════════════════════════════════════════════════
class SSMLHelpDialog(QDialog):
    """SSML reference dialog."""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("SSML Reference")
        self.setMinimumSize(600, 500)
        self._setup_ui()
    
    def _setup_ui(self):
        layout = QVBoxLayout(self)
        
        browser = QTextBrowser()
        browser.setOpenExternalLinks(False)
        
        html = """
        <h2>SSML Reference</h2>
        
        <h3>&lt;voice&gt;</h3>
        <p>Switch voice profile</p>
        <pre>&lt;voice name="female" emotion="happy"&gt;...&lt;/voice&gt;</pre>
        <p><b>Attributes:</b></p>
        <ul>
            <li><code>name</code>: default, male, female, robot, child, deep, bright</li>
            <li><code>emotion</code>: neutral, happy, sad, angry, calm</li>
        </ul>
        
        <h3>&lt;prosody&gt;</h3>
        <p>Adjust speech characteristics</p>
        <pre>&lt;prosody pitch="+2st" rate="110%" volume="115%"&gt;...&lt;/prosody&gt;</pre>
        <p><b>Attributes:</b></p>
        <ul>
            <li><code>pitch</code>: semitones (+2st, -3st) or percentage (120%)</li>
            <li><code>rate</code>: percentage (80%, 120%) or keyword (slow, fast)</li>
            <li><code>volume</code>: percentage (50%, 150%) or keyword (soft, loud)</li>
        </ul>
        
        <h3>&lt;break&gt;</h3>
        <p>Insert pause</p>
        <pre>&lt;break time="500ms" /&gt;
&lt;break strength="strong" /&gt;</pre>
        <p><b>Strength values:</b> none, x-weak, weak, medium, strong, x-strong</p>
        
        <h3>Example</h3>
        <pre>&lt;voice name="female" emotion="happy"&gt;
  &lt;prosody pitch="+2st" rate="110%"&gt;
    Hello! This is a test.
  &lt;/prosody&gt;
  &lt;break time="200ms" /&gt;
  And here is more text.
&lt;/voice&gt;</pre>
        """
        browser.setHtml(html)
        layout.addWidget(browser)


class PhonemeHelpDialog(QDialog):
    """Phoneme reference dialog."""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Phoneme Reference")
        self.setMinimumSize(500, 500)
        self._setup_ui()
    
    def _setup_ui(self):
        layout = QVBoxLayout(self)
        
        tabs = QTabWidget()
        
        # Vowels
        vowels_html = self._build_phoneme_table(FORMANTS, "vowel")
        tab1 = QTextBrowser()
        tab1.setHtml(vowels_html)
        tabs.addTab(tab1, "Vowels")
        
        # Nasals
        nasals_html = self._build_phoneme_table(NASAL_FMT, "nasal")
        tab2 = QTextBrowser()
        tab2.setHtml(nasals_html)
        tabs.addTab(tab2, "Nasals")
        
        # Approximants
        approx_html = self._build_phoneme_table(APPROX_FMT, "approximant")
        tab3 = QTextBrowser()
        tab3.setHtml(approx_html)
        tabs.addTab(tab3, "Approximants")
        
        # Fricatives
        fric_html = "<h2>Fricatives</h2><table>"
        fric_html += "<tr><th>Phoneme</th><th>Band (Hz)</th><th>Example</th></tr>"
        examples = {"S": "sun", "SH": "ship", "F": "fish", "TH": "think",
                    "HH": "hat", "Z": "zoo", "V": "van", "DH": "the", "ZH": "measure"}
        for ph, (lo, hi) in FRIC_BANDS.items():
            fric_html += f"<tr><td style='color:{PH_COLOR['fricative']}'>{ph}</td>"
            fric_html += f"<td>{lo}-{hi}</td>"
            fric_html += f"<td>{examples.get(ph, '-')}</td></tr>"
        fric_html += "</table>"
        tab4 = QTextBrowser()
        tab4.setHtml(fric_html)
        tabs.addTab(tab4, "Fricatives")
        
        # Stops
        stops_html = "<h2>Stops</h2><p>"
        stops_html += "P, T, K (voiceless) | B, D, G (voiced)</p>"
        tab5 = QTextBrowser()
        tab5.setHtml(stops_html)
        tabs.addTab(tab5, "Stops")
        
        layout.addWidget(tabs)
    
    def _build_phoneme_table(self, data, ph_type):
        color = PH_COLOR.get(ph_type, "#888")
        html = f"<h2>{ph_type.capitalize()}s</h2><table>"
        html += "<tr><th>Phoneme</th><th>F1 (Hz)</th><th>F2 (Hz)</th><th>F3 (Hz)</th><th>Duration</th></tr>"
        for ph, (f1, f2, f3) in data.items():
            dur = PH_DUR.get(ph_type, 0.1) * 1000
            html += f"<tr><td style='color:{color}; font-weight:bold'>{ph}</td>"
            html += f"<td>{f1[0]}±{f1[1]}</td>"
            html += f"<td>{f2[0]}±{f2[1]}</td>"
            html += f"<td>{f3[0]}±{f3[1]}</td>"
            html += f"<td>~{dur:.0f}ms</td></tr>"
        html += "</table>"
        return html


class AboutDialog(QDialog):
    """About dialog."""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("About")
        self.setFixedSize(400, 250)
        self._setup_ui()
    
    def _setup_ui(self):
        layout = QVBoxLayout(self)
        
        layout.addWidget(QLabel("🔊"))
        
        title = QLabel("<h2>Unified FFT TTS Engine</h2>")
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(title)
        
        version = QLabel("Version 1.0.0")
        version.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(version)
        
        desc = QLabel(
            "A formant-based text-to-speech engine\n"
            "using FFT synthesis and Kelly-Lochbaum modeling.\n\n"
            "Built with PySide6, NumPy, and SciPy."
        )
        desc.setAlignment(Qt.AlignmentFlag.AlignCenter)
        desc.setStyleSheet("color: #888;")
        layout.addWidget(desc)
        
        layout.addStretch()
        
        btn = QPushButton("OK")
        btn.clicked.connect(self.accept)
        layout.addWidget(btn)