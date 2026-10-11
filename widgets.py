"""Reusable UI widgets for the TTS Engine."""

import re
import numpy as np
from PySide6.QtWidgets import (
    QWidget, QLabel, QDockWidget, QGroupBox, QVBoxLayout, QFormLayout,
    QTextBrowser
)
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QPainter, QPen, QBrush, QColor, QFont, QTextCharFormat, QSyntaxHighlighter

from config import SR, EPS
from phonemes import PH_TYPE, PH_COLOR, FRIC_BANDS, get_phoneme_formants

import matplotlib
matplotlib.use("QtAgg")
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure


# ═══════════════════════════════════════════════════════════════
# SSML SYNTAX HIGHLIGHTER
# ═══════════════════════════════════════════════════════════════
class SSMLHighlighter(QSyntaxHighlighter):
    """Syntax highlighter for SSML text."""
    
    def __init__(self, doc):
        super().__init__(doc)
        self._formats = {
            'tag': self._make_format("#ff7043", bold=True),
            'attr': self._make_format("#ffca28"),
            'val': self._make_format("#69f0ae"),
            'comment': self._make_format("#666", italic=True),
        }
        self._patterns = [
            (r'<!--.*?-->', 'comment'),
            (r'<[/]?[\w]+', 'tag'),
            (r'[\w-]+=', 'attr'),
            (r'"[^"]*"', 'val'),
        ]

    @staticmethod
    def _make_format(color, bold=False, italic=False):
        fmt = QTextCharFormat()
        fmt.setForeground(QColor(color))
        if bold:
            fmt.setFontWeight(QFont.Bold)
        if italic:
            fmt.setFontItalic(True)
        return fmt

    def highlightBlock(self, text):
        for pattern, fmt_name in self._patterns:
            for match in re.finditer(pattern, text):
                self.setFormat(match.start(), match.end() - match.start(), self._formats[fmt_name])


# ═══════════════════════════════════════════════════════════════
# PHONEME TIMELINE WIDGET
# ═══════════════════════════════════════════════════════════════
class PhonemeTimeline(QWidget):
    """Visual timeline showing phoneme sequence."""
    
    phoneme_clicked = Signal(int)
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumHeight(50)
        self.setMaximumHeight(80)
        self.ph_ranges = []
        self.w_ranges = []
        self.current_pos = -1
        self.total_samples = 0
        self.setStyleSheet("background: #0d1117; border-radius: 4px;")
    
    def set_data(self, ph_ranges, w_ranges, total_samples):
        """Update timeline with phoneme data."""
        self.ph_ranges = ph_ranges
        self.w_ranges = w_ranges
        self.total_samples = total_samples
        self.update()
    
    def set_position(self, pos):
        """Update current playback position."""
        self.current_pos = pos
        self.update()
    
    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        
        w = self.width()
        h = self.height()
        
        painter.fillRect(0, 0, w, h, QColor("#0d1117"))
        
        if not self.ph_ranges or self.total_samples == 0:
            painter.setPen(QColor("#666"))
            painter.drawText(event.rect(), Qt.AlignmentFlag.AlignCenter, "No phoneme data")
            return
        
        # Draw word boundaries
        painter.setPen(QPen(QColor("#444"), 1, Qt.PenStyle.DashLine))
        for s, e, word in self.w_ranges:
            x1 = int(s / self.total_samples * w)
            painter.drawLine(x1, 0, x1, h)
        
        # Draw phonemes
        margin = 5
        ph_h = h - 2 * margin
        
        for s, e, ph in self.ph_ranges:
            x1 = int(s / self.total_samples * w)
            x2 = int(e / self.total_samples * w)
            ph_type = PH_TYPE.get(ph, "silence")
            color = QColor(PH_COLOR.get(ph_type, "#888"))
            
            painter.setBrush(QBrush(color))
            painter.setPen(Qt.PenStyle.NoPen)
            painter.drawRect(x1, margin, max(1, x2 - x1), ph_h)
            
            if x2 - x1 > 15:
                painter.setPen(QColor("black"))
                painter.setFont(QFont("Monospace", 7))
                painter.drawText(x1 + 2, margin + ph_h // 2 + 4, ph)
        
        # Draw playback position
        if self.current_pos >= 0:
            x = int(self.current_pos / self.total_samples * w)
            painter.setPen(QPen(QColor("white"), 2))
            painter.drawLine(x, 0, x, h)
    
    def mousePressEvent(self, event):
        if self.total_samples > 0:
            x = event.position().x()
            sample_pos = int(x / self.width() * self.total_samples)
            self.phoneme_clicked.emit(sample_pos)


# ═══════════════════════════════════════════════════════════════
# PHONEME INFO DOCK
# ═══════════════════════════════════════════════════════════════
class PhonemeInfoDock(QDockWidget):
    """Dock widget showing detailed phoneme information."""
    
    def __init__(self, parent=None):
        super().__init__("Phoneme Info", parent)
        self.setAllowedAreas(
            Qt.DockWidgetArea.LeftDockWidgetArea | 
            Qt.DockWidgetArea.RightDockWidgetArea
        )
        self._setup_ui()
    
    def _setup_ui(self):
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        self.current_phoneme = QLabel("—")
        self.current_phoneme.setFont(QFont("Monospace", 18, QFont.Bold))
        self.current_phoneme.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.current_phoneme)
        
        self.ph_type_label = QLabel("Type: -")
        layout.addWidget(self.ph_type_label)
        
        self.duration_label = QLabel("Duration: -")
        layout.addWidget(self.duration_label)
        
        # Formant display
        formant_group = QGroupBox("Formants")
        formant_layout = QFormLayout(formant_group)
        self.f1_label = QLabel("-")
        self.f2_label = QLabel("-")
        self.f3_label = QLabel("-")
        formant_layout.addRow("F1:", self.f1_label)
        formant_layout.addRow("F2:", self.f2_label)
        formant_layout.addRow("F3:", self.f3_label)
        layout.addWidget(formant_group)
        
        # Mini formant chart
        self.mini_fig = Figure(figsize=(3, 2), facecolor='#1a1a2e')
        self.mini_canvas = FigureCanvas(self.mini_fig)
        self.mini_ax = self.mini_fig.add_subplot(111)
        self.mini_ax.set_facecolor('#0d1117')
        self.mini_fig.subplots_adjust(left=0.15, right=0.95, top=0.9, bottom=0.15)
        layout.addWidget(self.mini_canvas)
        
        # Legend
        legend_group = QGroupBox("Phoneme Legend")
        legend_layout = QVBoxLayout(legend_group)
        for ph_type, color in PH_COLOR.items():
            lbl = QLabel(f"  ■ {ph_type.capitalize()}")
            lbl.setStyleSheet(f"color: {color}; font-weight: bold;")
            legend_layout.addWidget(lbl)
        layout.addWidget(legend_group)
        
        layout.addStretch()
        self.setWidget(widget)
    
    def update_phoneme(self, phoneme, start_idx=None, end_idx=None):
        """Update displayed phoneme information."""
        if not phoneme:
            self.current_phoneme.setText("—")
            self.ph_type_label.setText("Type: -")
            self.duration_label.setText("Duration: -")
            self.f1_label.setText("-")
            self.f2_label.setText("-")
            self.f3_label.setText("-")
            self._clear_mini_chart()
            return
        
        ph_type = PH_TYPE.get(phoneme, "silence")
        color = PH_COLOR.get(ph_type, "#888")
        
        self.current_phoneme.setText(f"[{phoneme}]")
        self.current_phoneme.setStyleSheet(f"color: {color};")
        self.ph_type_label.setText(f"Type: {ph_type}")
        
        if start_idx is not None and end_idx is not None:
            dur = (end_idx - start_idx) / SR * 1000
            self.duration_label.setText(f"Duration: {dur:.1f} ms")
        
        fmts = get_phoneme_formants(phoneme) if phoneme in PH_TYPE else None
        self._draw_mini_chart(phoneme, fmts, color)
        
        if fmts:
            self.f1_label.setText(f"{fmts[0][0]} Hz")
            self.f2_label.setText(f"{fmts[1][0]} Hz")
            self.f3_label.setText(f"{fmts[2][0]} Hz")
        elif phoneme in FRIC_BANDS:
            self.f1_label.setText("-")
            self.f2_label.setText("-")
            self.f3_label.setText("-")
        else:
            self.f1_label.setText("-")
            self.f2_label.setText("-")
            self.f3_label.setText("-")
    
    def _clear_mini_chart(self):
        self.mini_ax.clear()
        self.mini_ax.set_facecolor('#0d1117')
        self.mini_canvas.draw_idle()
    
    def _draw_mini_chart(self, phoneme, fmts, color):
        self.mini_ax.clear()
        self.mini_ax.set_facecolor('#0d1117')
        
        if fmts:
            freqs = np.linspace(0, 5000, 1000)
            envelope = np.zeros_like(freqs)
            for fc, bw in fmts:
                sigma = bw / (2 * np.sqrt(2 * np.log(2)) + EPS)
                envelope += np.exp(-0.5 * ((freqs - fc) / (sigma + EPS)) ** 2)
            
            self.mini_ax.fill_between(freqs, envelope, alpha=0.3, color=color)
            self.mini_ax.plot(freqs, envelope, color=color, linewidth=1.5)
            for i, (fc, bw) in enumerate(fmts):
                self.mini_ax.axvline(fc, color='white', linestyle='--', alpha=0.5, linewidth=0.5)
                self.mini_ax.text(fc, 1.1, f"F{i+1}", color='white', fontsize=7, ha='center')
        elif phoneme in FRIC_BANDS:
            flo, fhi = FRIC_BANDS[phoneme]
            self.mini_ax.axvspan(flo, fhi, alpha=0.3, color=color)
            self.mini_ax.text((flo + fhi) / 2, 0.5, f"{flo}-{fhi}Hz", 
                            color='white', fontsize=8, ha='center')
        
        self.mini_ax.set_xlim(0, 5000)
        self.mini_ax.set_ylim(0, 1.3)
        self.mini_ax.tick_params(colors='#666', labelsize=7)
        for sp in self.mini_ax.spines.values():
            sp.set_color('#333')
        self.mini_canvas.draw_idle()


# ═══════════════════════════════════════════════════════════════
# PHONEME BREAKDOWN WIDGET
# ═══════════════════════════════════════════════════════════════
class PhonemeBreakdownWidget(QTextBrowser):
    """Widget displaying phoneme breakdown with colors."""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFont(QFont("Monospace", 10))
        self.setStyleSheet("background: #0d1117; color: #e0e0e0;")
    
    def set_phoneme_data(self, ph_ranges, w_ranges):
        """Display phoneme breakdown."""
        html = "<table style='width:100%; border-collapse: collapse;'>"
        html += "<tr style='border-bottom: 1px solid #333;'>"
        html += "<th>#</th><th>Phoneme</th><th>Type</th><th>Start</th><th>End</th><th>Duration</th></tr>"
        
        for i, (s, e, ph) in enumerate(ph_ranges):
            pt = PH_TYPE.get(ph, "silence")
            color = PH_COLOR.get(pt, "#888")
            dur = (e - s) / SR * 1000
            
            html += f"<tr>"
            html += f"<td>{i+1}</td>"
            html += f"<td style='color:{color}; font-weight:bold;'>{ph}</td>"
            html += f"<td style='color:{color};'>{pt}</td>"
            html += f"<td>{s/SR:.3f}s</td>"
            html += f"<td>{e/SR:.3f}s</td>"
            html += f"<td>{dur:.1f}ms</td>"
            html += f"</tr>"
        
        html += "</table>"
        self.setHtml(html)