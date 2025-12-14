from __future__ import annotations

import time
from PyQt6.QtCore import Qt, pyqtSlot, QThread, QTimer, QCoreApplication
from PyQt6.QtGui import QFont
from PyQt6.QtWidgets import (
    QLabel,
    QMainWindow,
    QVBoxLayout,
    QWidget,
    QPushButton,
    QHBoxLayout,
    QTextEdit,
    QCheckBox,
    QComboBox,
    QSpinBox,
    QGroupBox,
    QFormLayout,
)

from translator.core.state import state
from translator.ui.region_selector import RegionSelector
from translator.ui.overlay import Overlay
from translator.services.monitor_service import MonitorService
from translator.services.translate_service import TranslateService
from translator.input.hotkey import GlobalHotkey
from translator.input.text_replace import ClipboardController


class MainWindow(QMainWindow):
    """Control panel for the translator application."""

    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Translator Control Panel")
        self.resize(450, 500)
        
        self._region_selector = RegionSelector()
        self._region_selector.region_selected.connect(self._on_region_selected)
        
        self._overlay = Overlay()

        central = QWidget()
        layout = QVBoxLayout()
        
        # Status Section
        self._status_label = QLabel("Status: Idle")
        self._status_label.setFont(QFont("Segoe UI", 10, QFont.Weight.Bold))
        layout.addWidget(self._status_label)

        # Controls
        controls_layout = QHBoxLayout()
        
        self._btn_select_region = QPushButton("Select Region")
        self._btn_select_region.clicked.connect(self._start_region_selection)
        controls_layout.addWidget(self._btn_select_region)
        
        self._btn_toggle_monitor = QPushButton("Start Monitoring")
        self._btn_toggle_monitor.setCheckable(True)
        self._btn_toggle_monitor.clicked.connect(self._toggle_monitoring)
        controls_layout.addWidget(self._btn_toggle_monitor)
        
        layout.addLayout(controls_layout)
        
        # Settings Group
        settings_group = QGroupBox("Settings")
        settings_form = QFormLayout()
        
        # Languages
        self._combo_source_lang = QComboBox()
        self._combo_source_lang.addItems(["Auto", "en", "zh", "ja", "ko", "fr", "de", "es"])
        settings_form.addRow("Source Language:", self._combo_source_lang)
        
        self._combo_target_lang = QComboBox()
        self._combo_target_lang.addItems(["zh", "en", "ja", "ko", "fr", "de", "es"])
        self._combo_target_lang.setCurrentText("zh")
        self._combo_target_lang.currentTextChanged.connect(self._on_target_lang_changed)
        settings_form.addRow("Target Language:", self._combo_target_lang)
        
        # Interval
        self._spin_interval = QSpinBox()
        self._spin_interval.setRange(100, 5000)
        self._spin_interval.setSingleStep(100)
        self._spin_interval.setValue(200)
        self._spin_interval.setSuffix(" ms")
        self._spin_interval.valueChanged.connect(self._on_interval_changed)
        settings_form.addRow("OCR Interval:", self._spin_interval)
        
        # Toggles
        self._chk_always_on_top = QCheckBox("Always on Top")
        self._chk_always_on_top.toggled.connect(self._toggle_always_on_top)
        settings_form.addRow("", self._chk_always_on_top)
        
        self._chk_overlay_on_text = QCheckBox("Overlay on Text")
        self._chk_overlay_on_text.setChecked(True)
        settings_form.addRow("", self._chk_overlay_on_text)
        
        settings_group.setLayout(settings_form)
        layout.addWidget(settings_group)

        # Info Display
        layout.addWidget(QLabel("Last OCR Text:"))
        self._text_display = QTextEdit()
        self._text_display.setReadOnly(True)
        layout.addWidget(self._text_display)

        central.setLayout(layout)
        self.setCentralWidget(central)
        
        # Service & Thread
        self._monitor_thread = QThread()
        self._monitor_service = MonitorService()
        self._monitor_service.moveToThread(self._monitor_thread)
        
        # Connect signals
        self._monitor_thread.started.connect(self._monitor_service.start_monitoring)
        self._monitor_service.text_detected.connect(self._on_text_detected)
        self._monitor_service.translation_finished.connect(self._on_translation_finished)
        self._monitor_service.error_occurred.connect(self._on_error)
        
        self._monitor_thread.start()
        
        # Hotkey & Input
        self._hotkey = GlobalHotkey(on_double_space=self._on_double_space_triggered)
        self._hotkey.start()
        self._clipboard = ClipboardController()
        self._translate_service = TranslateService()

    def closeEvent(self, event) -> None:
        """Handle application exit."""
        self._monitor_service.stop_monitoring()
        self._monitor_thread.quit()
        self._monitor_thread.wait(1000)
        self._hotkey.stop()
        self._overlay.close()
        event.accept()
        QCoreApplication.quit()

    @pyqtSlot(str)
    def _on_target_lang_changed(self, lang: str) -> None:
        self._monitor_service.set_target_language(lang)

    @pyqtSlot(int)
    def _on_interval_changed(self, ms: int) -> None:
        self._monitor_service.set_interval(ms / 1000.0)

    @pyqtSlot(bool)
    def _toggle_always_on_top(self, checked: bool) -> None:
        flags = self.windowFlags()
        if checked:
            flags |= Qt.WindowType.WindowStaysOnTopHint
        else:
            flags &= ~Qt.WindowType.WindowStaysOnTopHint
        self.setWindowFlags(flags)
        self.show()

    def _on_double_space_triggered(self) -> None:
        """Handle double space hotkey."""
        # This runs in hotkey thread, need to be careful with UI updates
        # But clipboard operations are fine.
        # We should probably emit a signal or use QTimer.singleShot to run in main thread
        QTimer.singleShot(0, self._process_selection_translation)

    def _process_selection_translation(self) -> None:
        # Try to undo the spaces first (Ctrl+Z x 2)
        # This is risky but necessary if spaces replaced selection
        self._clipboard.undo_spaces()
        
        text = self._clipboard.get_selected_text()
        if not text:
            self._status_label.setText("Status: No text selected")
            return
            
        self._status_label.setText("Status: Translating selection...")
        try:
            target_lang = self._combo_target_lang.currentText()
            translated = self._translate_service.translate(text, target_lang)
            self._clipboard.paste_text(translated)
            self._status_label.setText("Status: Translation pasted")
        except Exception as e:
            self._status_label.setText(f"Error: {e}")

    @pyqtSlot()
    def _start_region_selection(self) -> None:
        self.hide()  # Hide main window to see screen
        self._region_selector.start_selection()

    @pyqtSlot(tuple)
    def _on_region_selected(self, region: tuple[int, int, int, int]) -> None:
        self.show()
        state.monitoring_region = region
        self._status_label.setText(f"Region Selected: {region}")
        
    @pyqtSlot(bool)
    def _toggle_monitoring(self, checked: bool) -> None:
        state.is_monitoring = checked
        if checked:
            self._btn_toggle_monitor.setText("Stop Monitoring")
            self._status_label.setText("Status: Monitoring...")
            if not state.monitoring_region:
                self._status_label.setText("Warning: No region selected!")
        else:
            self._btn_toggle_monitor.setText("Start Monitoring")
            self._status_label.setText("Status: Idle")
            self._overlay.hide()

    @pyqtSlot(str)
    def _on_text_detected(self, text: str) -> None:
        self._text_display.setText(text)

    @pyqtSlot(str)
    def _on_translation_finished(self, text: str) -> None:
        self._overlay.update_text(text)
        
        if self._chk_overlay_on_text.isChecked() and state.monitoring_region:
            x, y, w, h = state.monitoring_region
            self._overlay.move(x, y)
            
        if not self._overlay.isVisible():
            self._overlay.show()
        self._text_display.setText(text)

    @pyqtSlot(str)
    def _on_error(self, error: str) -> None:
        self._status_label.setText(f"Error: {error}")
