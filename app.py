#!/usr/bin/env python3
"""
Unified FFT TTS Engine (PySide6) - Main Entry Point
====================================================
Run this file to launch the GUI application.

Dependencies:
    pip install numpy scipy sounddevice PySide6 matplotlib g2p_en
"""

import sys
import math
import os
import numpy as np

import matplotlib
matplotlib.use("QtAgg")
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QTextEdit, QPushButton, QSlider, QLabel, QDoubleSpinBox,
    QFileDialog, QProgressBar, QComboBox, QMessageBox, QInputDialog,
    QMenuBar, QMenu, QToolBar, QGroupBox, QStatusBar, QTabWidget
)
from PySide6.QtCore import Qt, QThread, Signal, QTimer, QSettings
from PySide6.QtGui import (
    QAction, QKeySequence, QFont, QPalette, QColor, QShortcut
)

# Local imports
from config import (SR, EPS, BLOCKSIZE, WINDOW_SIZE, WINDOW_DUR, 
                    SPEC_FFT, SPEC_COLS, HAS_SD, DEFAULTS, DEMO_SSML, PRESETS)
from phonemes import PH_TYPE, PH_COLOR
from profiles import VOICES, EMOTIONS, get_voice_info, get_emotion_info
from text_processing import load_dict, save_dict, update_pron_dict
from synthesizer import synthesize_ssml
from audio_utils import write_wav, read_wav, estimate_pitch
from widgets import SSMLHighlighter, PhonemeTimeline, PhonemeInfoDock, PhonemeBreakdownWidget
from dialogs import SettingsDialog, EffectsDialog, SSMLHelpDialog, PhonemeHelpDialog, AboutDialog


# ═══════════════════════════════════════════════════════════════
# SYNTHESIS THREAD
# ═══════════════════════════════════════════════════════════════
class SynthThread(QThread):
    progress = Signal(int)
    done = Signal(object, object, object)
    error = Signal(str)

    def __init__(self, text, engine, warmth, effects=None):
        super().__init__()
        self.text = text
        self.engine = engine
        self.warmth = warmth
        self.effects = effects

    def run(self):
        try:
            a, ph, w = synthesize_ssml(
                self.text, self.engine, self.warmth,
                prog_cb=lambda x: self.progress.emit(x)
            )
            if self.effects and a is not None and len(a) > 0:
                a = self.effects.apply_effects(a)
            self.done.emit(a, ph, w)
        except Exception as e:
            self.error.emit(str(e))


# ═══════════════════════════════════════════════════════════════
# MAIN WINDOW
# ═══════════════════════════════════════════════════════════════
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Unified FFT TTS Engine")
        self.setGeometry(60, 60, 1200, 900)
        
        # State
        self.audio = None
        self.ph_ranges = []
        self.w_ranges = []
        self.playing = False
        self.paused = False
        self.play_idx = 0
        self.looping = False
        self.stream = None
        self.effects_settings = None
        self.recent_files = []
        self.settings = QSettings("FFT_TTS", "Engine")
        
        # Build UI
        self._build_menu()
        self._build_toolbar()
        self._build_ui()
        self._build_docks()
        self._build_statusbar()
        self._load_settings()
        self._setup_shortcuts()

    def _build_menu(self):
        menubar = self.menuBar()
        
        # File menu
        file_menu = menubar.addMenu("&File")
        self._add_action(file_menu, "New", "Ctrl+N", self._new_file)
        self._add_action(file_menu, "Open...", "Ctrl+O", self._open_file)
        self._add_action(file_menu, "Save Text...", "Ctrl+S", self._save_text)
        file_menu.addSeparator()
        self.recent_menu = file_menu.addMenu("Recent Files")
        file_menu.addSeparator()
        self._add_action(file_menu, "Export WAV...", "Ctrl+E", self._export)
        self._add_action(file_menu, "Batch Export...", None, self._batch)
        file_menu.addSeparator()
        self._add_action(file_menu, "Exit", "Ctrl+Q", self.close)
        
        # Edit menu
        edit_menu = menubar.addMenu("&Edit")
        self._add_action(edit_menu, "Undo", "Ctrl+Z", lambda: self.te.undo())
        self._add_action(edit_menu, "Redo", "Ctrl+Y", lambda: self.te.redo())
        edit_menu.addSeparator()
        self._add_action(edit_menu, "Pronunciation Dictionary...", None, self._dict)
        
        # View menu
        view_menu = menubar.addMenu("&View")
        self.phoneme_dock_action = QAction("Phoneme Info", self, checkable=True, checked=True)
        self.phoneme_dock_action.triggered.connect(
            lambda: self.phoneme_dock.setVisible(self.phoneme_dock_action.isChecked()))
        view_menu.addAction(self.phoneme_dock_action)
        view_menu.addSeparator()
        self._add_action(view_menu, "Zoom In", "Ctrl+=", lambda: self._zoom_waveform(0.8))
        self._add_action(view_menu, "Zoom Out", "Ctrl+-", lambda: self._zoom_waveform(1.25))
        self._add_action(view_menu, "Fit Waveform", "Ctrl+0", lambda: self._zoom_waveform(None))
        
        # Tools menu
        tools_menu = menubar.addMenu("&Tools")
        self._add_action(tools_menu, "Audio Effects...", None, self._open_effects)
        self._add_action(tools_menu, "Reset Effects", None, self._reset_effects)
        tools_menu.addSeparator()
        self._add_action(tools_menu, "Settings...", None, self._open_settings)
        
        # Presets menu
        presets_menu = menubar.addMenu("&Presets")
        for name, text in PRESETS.items():
            act = QAction(name, self)
            act.triggered.connect(lambda checked, t=text: self.te.setPlainText(t))
            presets_menu.addAction(act)
        
        # Help menu
        help_menu = menubar.addMenu("&Help")
        self._add_action(help_menu, "SSML Reference...", None, lambda: SSMLHelpDialog(self).exec())
        self._add_action(help_menu, "Phoneme Reference...", None, lambda: PhonemeHelpDialog(self).exec())
        help_menu.addSeparator()
        self._add_action(help_menu, "About...", None, lambda: AboutDialog(self).exec())

    def _add_action(self, menu, text, shortcut, callback):
        act = QAction(text, self)
        if shortcut:
            act.setShortcut(shortcut)
        act.triggered.connect(callback)
        menu.addAction(act)
        return act

    def _build_toolbar(self):
        toolbar = QToolBar("Main Toolbar")
        toolbar.setMovable(False)
        self.addToolBar(toolbar)
        
        self.syn_btn = QPushButton("🔊 Synthesize")
        self.syn_btn.setToolTip("Synthesize (Ctrl+Enter)")
        self.syn_btn.clicked.connect(self._synth)
        toolbar.addWidget(self.syn_btn)
        
        toolbar.addSeparator()
        
        self.play_btn = QPushButton("▶ Play")
        self.play_btn.setToolTip("Play (Space)")
        self.play_btn.clicked.connect(self._play)
        self.play_btn.setEnabled(False)
        toolbar.addWidget(self.play_btn)
        
        self.pause_btn = QPushButton("⏸ Pause")
        self.pause_btn.setToolTip("Pause (P)")
        self.pause_btn.clicked.connect(self._toggle_pause)
        self.pause_btn.setEnabled(False)
        toolbar.addWidget(self.pause_btn)
        
        self.stop_btn = QPushButton("⏹ Stop")
        self.stop_btn.setToolTip("Stop (Esc)")
        self.stop_btn.clicked.connect(self._stop)
        toolbar.addWidget(self.stop_btn)
        
        self.loop_btn = QPushButton("🔁 Loop")
        self.loop_btn.setToolTip("Loop (L)")
        self.loop_btn.setCheckable(True)
        self.loop_btn.clicked.connect(lambda: setattr(self, 'looping', self.loop_btn.isChecked()))
        toolbar.addWidget(self.loop_btn)
        
        toolbar.addSeparator()
        
        self.exp_btn = QPushButton("💾 Export")
        self.exp_btn.clicked.connect(self._export)
        self.exp_btn.setEnabled(False)
        toolbar.addWidget(self.exp_btn)
        
        self.effects_btn = QPushButton("🎛 Effects")
        self.effects_btn.clicked.connect(self._open_effects)
        toolbar.addWidget(self.effects_btn)
        
        toolbar.addSeparator()
        
        toolbar.addWidget(QLabel(" VU: "))
        self.vu = QProgressBar()
        self.vu.setRange(0, 100)
        self.vu.setMaximumWidth(120)
        self.vu.setMaximumHeight(20)
        toolbar.addWidget(self.vu)
        
        toolbar.addWidget(QLabel(" Time: "))
        self.time_label = QLabel("0:00.0 / 0:00.0")
        self.time_label.setMinimumWidth(100)
        toolbar.addWidget(self.time_label)

    def _build_ui(self):
        cw = QWidget()
        self.setCentralWidget(cw)
        main_layout = QVBoxLayout(cw)
        main_layout.setContentsMargins(5, 5, 5, 5)
        
        # Text input
        text_group = QGroupBox("Input Text / SSML")
        text_layout = QVBoxLayout(text_group)
        self.te = QTextEdit()
        self.te.setPlaceholderText("Enter text or SSML here...")
        self.te.setMaximumHeight(120)
        self.te.setFont(QFont("Consolas", 10))
        self.ssml_highlighter = SSMLHighlighter(self.te.document())
        text_layout.addWidget(self.te)
        main_layout.addWidget(text_group)
        
        # Controls row 1
        controls1 = QHBoxLayout()
        
        voice_group = QGroupBox("Voice")
        voice_layout = QVBoxLayout(voice_group)
        self.vc = QComboBox()
        self.vc.addItems(list(VOICES.keys()))
        self.vc.currentTextChanged.connect(self._update_voice_info)
        voice_layout.addWidget(self.vc)
        self.voice_info_label = QLabel("")
        self.voice_info_label.setStyleSheet("color: #888; font-size: 10px;")
        voice_layout.addWidget(self.voice_info_label)
        controls1.addWidget(voice_group)
        
        emotion_group = QGroupBox("Emotion")
        emotion_layout = QVBoxLayout(emotion_group)
        self.ec = QComboBox()
        self.ec.addItems(list(EMOTIONS.keys()))
        self.ec.currentTextChanged.connect(self._update_emotion_info)
        emotion_layout.addWidget(self.ec)
        self.emotion_info_label = QLabel("")
        self.emotion_info_label.setStyleSheet("color: #888; font-size: 10px;")
        emotion_layout.addWidget(self.emotion_info_label)
        controls1.addWidget(emotion_group)
        
        engine_group = QGroupBox("Engine")
        engine_layout = QVBoxLayout(engine_group)
        self.enc = QComboBox()
        self.enc.addItems(["FFT Formant", "Kelly-Lochbaum"])
        engine_layout.addWidget(self.enc)
        controls1.addWidget(engine_group)
        main_layout.addLayout(controls1)
        
        # Controls row 2
        sliders_group = QGroupBox("Prosody Adjustments")
        sliders_layout = QHBoxLayout(sliders_group)
        
        for lbl, lo, hi, val, attr in [
            ("Pitch", 50, 200, 100, "pitch_sl"),
            ("Speed", 50, 200, 100, "speed_sl"),
            ("Volume", 0, 200, 100, "vol_sl"),
            ("Warmth", 0, 100, 0, "warm_sl"),
        ]:
            v = QVBoxLayout()
            v.addWidget(QLabel(lbl))
            s = QSlider(Qt.Orientation.Horizontal)
            s.setRange(lo, hi)
            s.setValue(val)
            lb = QLabel(f"{val}%")
            lb.setMinimumWidth(35)
            s.valueChanged.connect(lambda x, l=lb: l.setText(f"{x}%"))
            v.addWidget(s)
            v.addWidget(lb)
            sliders_layout.addLayout(v)
            setattr(self, attr, s)
        main_layout.addWidget(sliders_group)
        
        # Controls row 3
        formant_group = QGroupBox("Formant Scaling")
        formant_layout = QHBoxLayout(formant_group)
        for i, (lbl, df) in enumerate([("F1", 1.0), ("F2", 1.0), ("F3", 1.0)]):
            formant_layout.addWidget(QLabel(f"{lbl}:"))
            sp = QDoubleSpinBox()
            sp.setRange(0.5, 2.0)
            sp.setValue(df)
            sp.setSingleStep(0.05)
            formant_layout.addWidget(sp)
            setattr(self, f"f{i + 1}", sp)
        main_layout.addWidget(formant_group)
        
        # Buttons
        buttons_layout = QHBoxLayout()
        for text, callback in [
            ("📦 Batch", self._batch), ("📖 Dict", self._dict),
            ("📋 Demo", lambda: self.te.setPlainText(DEMO_SSML)),
            ("🗑 Clear", self.te.clear),
        ]:
            btn = QPushButton(text)
            btn.clicked.connect(callback)
            buttons_layout.addWidget(btn)
        buttons_layout.addStretch()
        self.pb = QProgressBar()
        self.pb.setVisible(False)
        self.pb.setMaximumWidth(200)
        buttons_layout.addWidget(self.pb)
        main_layout.addLayout(buttons_layout)
        
        # Visualization
        viz_group = QGroupBox("Visualization")
        viz_layout = QVBoxLayout(viz_group)
        self.viz_tabs = QTabWidget()
        
        # Main viz tab
        viz_widget = QWidget()
        viz_inner = QVBoxLayout(viz_widget)
        self.fig = Figure(figsize=(10, 4), facecolor='#1a1a2e')
        self.canvas = FigureCanvas(self.fig)
        self.canvas.setMinimumHeight(200)
        self.ax_w = self.fig.add_subplot(211)
        self.ax_s = self.fig.add_subplot(212)
        self._setup_axes()
        viz_inner.addWidget(self.canvas)
        self.timeline = PhonemeTimeline()
        self.timeline.phoneme_clicked.connect(self._seek_to_position)
        viz_inner.addWidget(self.timeline)
        self.viz_tabs.addTab(viz_widget, "Waveform/Spectrogram")
        
        # Phoneme breakdown tab
        self.phoneme_list = PhonemeBreakdownWidget()
        self.viz_tabs.addTab(self.phoneme_list, "Phoneme Breakdown")
        
        viz_layout.addWidget(self.viz_tabs)
        main_layout.addWidget(viz_group)
        
        self.te.setPlainText(DEMO_SSML)
        self._update_voice_info()
        self._update_emotion_info()

    def _build_docks(self):
        self.phoneme_dock = PhonemeInfoDock(self)
        self.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, self.phoneme_dock)

    def _build_statusbar(self):
        self.statusbar = QStatusBar()
        self.setStatusBar(self.statusbar)
        self.stat = QLabel("Ready")
        self.stat.setStyleSheet("padding: 2px 8px; background: #1a1a2e; color: #e0e0e0; border-radius: 3px;")
        self.statusbar.addPermanentWidget(self.stat)

    def _setup_axes(self):
        self.fig.clear()
        self.ax_w = self.fig.add_subplot(211)
        self.ax_s = self.fig.add_subplot(212)
        for ax in (self.ax_w, self.ax_s):
            ax.set_facecolor('#0d1117')
            ax.tick_params(colors='#888')
            for sp in ax.spines.values():
                sp.set_color('#333')
        self.ax_w.set_title("Waveform", color='#ccc', fontsize=10)
        self.ax_w.set_ylim(-1, 1)
        self.ax_s.set_title("Spectrogram", color='#ccc', fontsize=10)
        self.ax_s.set_ylim(0, 5000)
        self.fig.subplots_adjust(hspace=0.45, left=0.06, right=0.98, top=0.94, bottom=0.08)

    def _setup_shortcuts(self):
        QShortcut(QKeySequence("Ctrl+Enter"), self, self._synth)
        QShortcut(QKeySequence("Space"), self, self._play)
        QShortcut(QKeySequence("P"), self, self._toggle_pause)
        QShortcut(QKeySequence("Escape"), self, self._stop)
        QShortcut(QKeySequence("Left"), self, lambda: self._seek_relative(-0.5))
        QShortcut(QKeySequence("Right"), self, lambda: self._seek_relative(0.5))

    # ── Settings Management ───────────────────────────────────
    def _load_settings(self):
        self.recent_files = self.settings.value("recent_files", [], list)
        self._update_recent_menu()
        if self.settings.value("default_voice", "default", str) in VOICES:
            self.vc.setCurrentText(self.settings.value("default_voice", "default", str))

    def _save_settings(self):
        self.settings.setValue("default_voice", self.vc.currentText())
        self.settings.setValue("recent_files", self.recent_files[:10])

    def _update_recent_menu(self):
        self.recent_menu.clear()
        if not self.recent_files:
            act = QAction("(None)", self)
            act.setEnabled(False)
            self.recent_menu.addAction(act)
        else:
            for path in self.recent_files[:10]:
                act = QAction(os.path.basename(path), self)
                act.triggered.connect(lambda checked, p=path: self._load_file(p))
                self.recent_menu.addAction(act)

    # ── Info Updates ──────────────────────────────────────────
    def _update_voice_info(self):
        self.voice_info_label.setText(get_voice_info(self.vc.currentText()))

    def _update_emotion_info(self):
        self.emotion_info_label.setText(get_emotion_info(self.ec.currentText()))

    def _format_time(self, seconds):
        return f"{int(seconds)//60}:{seconds%60:05.2f}"

    # ── Helpers ───────────────────────────────────────────────
    def _eng(self):
        return "kl" if "kelly" in self.enc.currentText().lower() else "fft"

    def _wrap_prosody(self, text):
        ps, sp, vo = self.pitch_sl.value()/100, self.speed_sl.value()/100, self.vol_sl.value()/100
        st = 12 * math.log2(ps) if ps > 0 else 0
        return f'<prosody pitch="{st:+.1f}st" rate="{sp*100:.0f}%" volume="{vo*100:.0f}%">{text}</prosody>'

    # ── File Operations ───────────────────────────────────────
    def _new_file(self):
        self.te.clear()
        self.audio = None
        self.ph_ranges = []
        self.w_ranges = []
        self.play_btn.setEnabled(False)
        self.exp_btn.setEnabled(False)
        self.timeline.set_data([], [], 0)
        self.stat.setText("New")

    def _open_file(self):
        fn, _ = QFileDialog.getOpenFileName(self, "Open", "", "Text/SSML (*.txt *.ssml);;All (*)")
        if fn:
            self._load_file(fn)

    def _load_file(self, fn):
        with open(fn, 'r', encoding='utf-8') as f:
            self.te.setPlainText(f.read())
        if fn in self.recent_files:
            self.recent_files.remove(fn)
        self.recent_files.insert(0, fn)
        self._update_recent_menu()
        self.stat.setText(f"Loaded: {os.path.basename(fn)}")

    def _save_text(self):
        fn, _ = QFileDialog.getSaveFileName(self, "Save", "", "Text (*.txt);;SSML (*.ssml)")
        if fn:
            with open(fn, 'w', encoding='utf-8') as f:
                f.write(self.te.toPlainText())
            self.stat.setText(f"Saved: {fn}")

    # ── Synthesis ─────────────────────────────────────────────
    def _synth(self):
        text = self.te.toPlainText().strip()
        if not text:
            QMessageBox.warning(self, "Warning", "Enter text first.")
            return
        text = self._wrap_prosody(text)
        
        self.syn_btn.setEnabled(False)
        self.play_btn.setEnabled(False)
        self.exp_btn.setEnabled(False)
        self.pb.setVisible(True)
        self.pb.setValue(0)
        self.stat.setText("Synthesizing…")
        
        self.th = SynthThread(text, self._eng(), self.warm_sl.value(), self.effects_settings)
        self.th.progress.connect(self.pb.setValue)
        self.th.done.connect(self._on_done)
        self.th.error.connect(self._on_err)
        self.th.start()

    def _on_done(self, a, ph, w):
        self.audio = a
        self.ph_ranges = ph
        self.w_ranges = w
        self._draw_static()
        self.timeline.set_data(ph, w, len(a) if a is not None else 0)
        self.phoneme_list.set_phoneme_data(ph, w)
        self.syn_btn.setEnabled(True)
        self.play_btn.setEnabled(True)
        self.exp_btn.setEnabled(True)
        self.pb.setVisible(False)
        d = len(a) / SR if a is not None else 0
        self.stat.setText(f"Done: {d:.2f}s | {len(ph)} phonemes")
        self._update_time_display(0, len(a) if a is not None else 0)

    def _on_err(self, msg):
        QMessageBox.critical(self, "Error", msg)
        self.syn_btn.setEnabled(True)
        self.pb.setVisible(False)
        self.stat.setText("Error")

    # ── Visualization ─────────────────────────────────────────
    def _draw_static(self):
        if self.audio is None:
            return
        a = self.audio
        ax = self.ax_w
        ax.clear()
        ax.set_facecolor('#0d1117')
        ax.set_title("Waveform", color='#ccc', fontsize=10)
        ax.set_ylim(-1, 1)
        
        t = np.arange(len(a)) / SR
        for s, e, p in self.ph_ranges:
            pt = PH_TYPE.get(p, "silence")
            c = PH_COLOR.get(pt, "#888")
            ax.axvspan(t[s], t[min(e, len(a)-1)], color=c, alpha=0.2)
        ax.plot(t, a, color='#4fc3f7', linewidth=0.5)
        
        self.ph_txt = ax.text(0.5, 1.08, "", transform=ax.transAxes, ha='center', fontsize=11, color='white', fontweight='bold')
        self.wd_txt = ax.text(0.5, -0.18, "", transform=ax.transAxes, ha='center', fontsize=10, color='#76ff03')
        self.canvas.draw()

    def _zoom_waveform(self, factor):
        if self.audio is None:
            return
        if factor is None:
            self.ax_w.set_xlim(0, len(self.audio) / SR)
        else:
            xlim = self.ax_w.get_xlim()
            center = (xlim[0] + xlim[1]) / 2
            width = (xlim[1] - xlim[0]) * factor
            self.ax_w.set_xlim(center - width/2, center + width/2)
        self.canvas.draw_idle()

    # ── Playback ──────────────────────────────────────────────
    def _play(self):
        if self.audio is None or not HAS_SD:
            if not HAS_SD:
                QMessageBox.warning(self, "Warning", "pip install sounddevice")
            return
        
        if self.paused:
            self.paused = False
            self.ptimer.start(30)
            self.stat.setText("Playing…")
            return
        
        self.playing = True
        self.play_idx = 0
        
        import sounddevice as sd
        self.stream = sd.OutputStream(channels=1, samplerate=SR, blocksize=BLOCKSIZE, dtype='float32')
        self.stream.start()
        self.ptimer = QTimer()
        self.ptimer.timeout.connect(self._tick)
        self.ptimer.start(30)
        self.play_btn.setEnabled(False)
        self.pause_btn.setEnabled(True)
        self.syn_btn.setEnabled(False)
        self.stat.setText("Playing…")
        self._setup_playback_axes()

    def _setup_playback_axes(self):
        self.ax_w.clear()
        self.ax_w.set_facecolor('#0d1117')
        self.ax_w.set_title("Waveform ▶", color='#ccc', fontsize=10)
        self.ax_w.set_ylim(-1, 1)
        self.ax_w.set_xlim(0, WINDOW_DUR)
        self.wline, = self.ax_w.plot([], [], color='#4fc3f7', linewidth=0.8)
        self.hl_patch = None
        self.ph_txt = self.ax_w.text(0.5, 1.08, "", transform=self.ax_w.transAxes, ha='center', fontsize=11, color='white', fontweight='bold')
        self.wd_txt = self.ax_w.text(0.5, -0.18, "", transform=self.ax_w.transAxes, ha='center', fontsize=10, color='#76ff03')
        
        self.ax_s.clear()
        self.ax_s.set_facecolor('#0d1117')
        self.ax_s.set_title("Spectrogram ▶", color='#ccc', fontsize=10)
        self.ax_s.set_ylim(0, 5000)
        self.ax_s.set_xlim(0, WINDOW_DUR)
        self.live_spec = np.zeros((SPEC_FFT // 2, SPEC_COLS))
        self.simg = self.ax_s.imshow(20 * np.log10(self.live_spec + 1e-6), aspect='auto', origin='lower', cmap='magma', extent=[0, WINDOW_DUR, 0, SR/2], interpolation='bilinear')
        self.canvas.draw()

    def _tick(self):
        if not hasattr(self, 'live_spec') or not self.playing or self.audio is None:
            self._stop()
            return
        
        n = len(self.audio)
        ei = min(self.play_idx + BLOCKSIZE, n)
        blk = self.audio[self.play_idx:ei]
        vol = self.vol_sl.value() / 100
        blk = (blk * vol).astype(np.float32)
        if len(blk) < BLOCKSIZE:
            blk = np.pad(blk, (0, BLOCKSIZE - len(blk)))
        self.stream.write(blk)
        
        # Update waveform
        ws = max(0, self.play_idx - WINDOW_SIZE // 2)
        we = ws + WINDOW_SIZE
        tw = np.arange(ws, min(we, n)) / SR
        self.wline.set_data(tw - tw[0] if len(tw) else [], self.audio[ws:min(we, n)])
        self.ax_w.set_xlim(0, WINDOW_DUR)
        
        # Highlight phoneme
        cph, cpt = "", "silence"
        for s, e, p in self.ph_ranges:
            if s <= self.play_idx < e:
                cph, cpt = p, PH_TYPE.get(p, "silence")
                c = PH_COLOR.get(cpt, "#888")
                if self.hl_patch:
                    self.hl_patch.remove()
                self.hl_patch = self.ax_w.axvspan(s/SR - ws/SR, e/SR - ws/SR, color=c, alpha=0.3)
                break
        self.ph_txt.set_text(f"[{cph}]" if cph else "")
        self.ph_txt.set_color(PH_COLOR.get(cpt, "white"))
        self.phoneme_dock.update_phoneme(cph, *[x for x in [s, e] if cph][:2] if cph else [None, None])
        
        cw = next((w for s, e, w in self.w_ranges if s <= self.play_idx < e), "")
        self.wd_txt.set_text(f'"{cw}"' if cw else "")
        
        # Update spectrogram
        fb = np.fft.fft(blk, SPEC_FFT)
        self.live_spec[:, :-1] = self.live_spec[:, 1:]
        self.live_spec[:, -1] = np.abs(fb[:SPEC_FFT // 2])
        self.simg.set_data(20 * np.log10(self.live_spec + 1e-6))
        
        # VU meter
        self.vu.setValue(int(min(100, np.sqrt(np.mean(blk ** 2)) * 500)))
        self._update_time_display(self.play_idx, n)
        self.timeline.set_position(self.play_idx)
        self.canvas.draw_idle()
        
        self.play_idx += BLOCKSIZE
        if self.play_idx >= n:
            if self.looping:
                self.play_idx = 0
            else:
                self._stop()

    def _toggle_pause(self):
        if self.playing:
            self.paused = True
            self.ptimer.stop()
            self.stat.setText("Paused")

    def _stop(self):
        self.playing = False
        self.paused = False
        self.ptimer.stop()
        if self.stream:
            try:
                self.stream.stop()
                self.stream.close()
            except Exception:
                pass
            self.stream = None
        self.vu.setValue(0)
        self.play_btn.setEnabled(self.audio is not None)
        self.pause_btn.setEnabled(False)
        self.syn_btn.setEnabled(True)
        self.phoneme_dock.update_phoneme(None)
        if self.audio is not None:
            self._draw_static()
            self._update_time_display(0, len(self.audio))
            self.timeline.set_position(-1)

    def _seek_relative(self, seconds):
        if self.audio is None:
            return
        new_pos = self.play_idx + int(seconds * SR)
        self.play_idx = max(0, min(new_pos, len(self.audio)))

    def _seek_to_position(self, pos):
        if self.audio is not None and self.playing:
            self.play_idx = max(0, min(pos, len(self.audio)))

    def _update_time_display(self, current, total):
        ct = current / SR if SR > 0 else 0
        tt = total / SR if SR > 0 else 0
        self.time_label.setText(f"{self._format_time(ct)} / {self._format_time(tt)}")

    # ── Export / Batch / Dict ─────────────────────────────────
    def _export(self):
        if self.audio is None:
            return
        fn, _ = QFileDialog.getSaveFileName(self, "Save WAV", "", "WAV (*.wav)")
        if fn:
            if not fn.endswith(".wav"):
                fn += ".wav"
            write_wav(fn, self.audio)
            self.stat.setText(f"Exported: {fn}")

    def _batch(self):
        txt, ok = QInputDialog.getMultiLineText(self, "Batch", "One sentence per line:")
        if not ok:
            return
        fld = QFileDialog.getExistingDirectory(self, "Folder")
        if not fld:
            return
        
        cnt = 0
        for i, line in enumerate(txt.splitlines(), 1):
            if not line.strip():
                continue
            try:
                wrapped = self._wrap_prosody(line.strip())
                a, _, _ = synthesize_ssml(wrapped, self._eng(), self.warm_sl.value())
                write_wav(os.path.join(fld, f"{i:03}.wav"), a)
                cnt += 1
            except Exception as e:
                self.stat.setText(f"Error line {i}: {e}")
                return
        self.stat.setText(f"Batch: {cnt} files → {fld}")
        QMessageBox.information(self, "Done", f"Exported {cnt} files.")

    def _dict(self):
        d = load_dict()
        txt = "\n".join(f"{k} {v}" for k, v in d.items())
        res, ok = QInputDialog.getMultiLineText(self, "Pronunciation Dictionary", "WORD PHONEMES\n(hello HH AH L OW)", txt)
        if ok:
            nd = {}
            for ln in res.splitlines():
                pts = ln.strip().split(maxsplit=1)
                if len(pts) == 2:
                    nd[pts[0]] = pts[1]
            update_pron_dict(nd)
            self.stat.setText(f"Dict: {len(nd)} entries")

    # ── Effects & Settings ────────────────────────────────────
    def _open_effects(self):
        dlg = EffectsDialog(self)
        if dlg.exec():
            self.effects_settings = dlg

    def _reset_effects(self):
        self.effects_settings = None
        self.stat.setText("Effects reset")

    def _open_settings(self):
        dlg = SettingsDialog(DEFAULTS, self)
        if dlg.exec():
            new_settings = dlg.get_settings()
            self.settings.setValue("settings", new_settings)
            self.stat.setText("Settings saved")

    # ── Cleanup ───────────────────────────────────────────────
    def closeEvent(self, event):
        self._stop()
        self._save_settings()
        event.accept()


# ═══════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════
def main():
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    
    pal = QPalette()
    pal.setColor(QPalette.ColorRole.Window, QColor(26, 26, 46))
    pal.setColor(QPalette.ColorRole.WindowText, QColor(224, 224, 224))
    pal.setColor(QPalette.ColorRole.Base, QColor(13, 17, 23))
    pal.setColor(QPalette.ColorRole.AlternateBase, QColor(30, 30, 50))
    pal.setColor(QPalette.ColorRole.Text, QColor(224, 224, 224))
    pal.setColor(QPalette.ColorRole.Button, QColor(40, 40, 65))
    pal.setColor(QPalette.ColorRole.ButtonText, QColor(224, 224, 224))
    pal.setColor(QPalette.ColorRole.Highlight, QColor(79, 195, 247))
    pal.setColor(QPalette.ColorRole.HighlightedText, QColor(0, 0, 0))
    app.setPalette(pal)
    
    w = MainWindow()
    w.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()