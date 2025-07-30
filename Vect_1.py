import sys
import pyqtgraph as pg
import os
import numpy as np
import pygame
import time
import tempfile
import subprocess
from PyQt6.QtWidgets import (
    QApplication, QWidget, QLabel, QVBoxLayout, QHBoxLayout,
    QSlider, QComboBox, QFileDialog, QPushButton, QDoubleSpinBox, QCheckBox
)
from PyQt6.QtCore import Qt, QTimer, QPointF
from PyQt6.QtGui import QPixmap, QPainter, QColor, QPen, QImage

pg.setConfigOptions(useOpenGL=True, antialias=True)

# === AUDIO ENGINE ===
class AudioMuteMixin:
    def toggle_mute(self):
        self.is_muted = not getattr(self, 'is_muted', False)
        if hasattr(self.audio, 'channel') and self.audio.channel:
            if self.is_muted:
                self.audio.channel.set_volume(0, 0)
            else:
                self.audio.channel.set_volume(self.audio.left_amp, self.audio.right_amp)

def generate_wave(wave_type, freq, amp, samplerate=44100, duration=1.0):
    t = np.linspace(0, duration, int(samplerate * duration), endpoint=False)
    if wave_type == 'sine':
        return amp * np.sin(2 * np.pi * freq * t)
    elif wave_type == 'square':
        return amp * np.sign(np.sin(2 * np.pi * freq * t))
    elif wave_type == 'triangle':
        return amp * 2 * np.arcsin(np.sin(2 * np.pi * freq * t)) / np.pi
    elif wave_type == 'sawtooth':
        return amp * 2 * (t * freq - np.floor(0.5 + t * freq))
    else:
        return np.zeros_like(t)

class AudioOutput:
    def __init__(self):
        pygame.mixer.init(frequency=44100, size=-16, channels=2)
        self.left_freq = 440
        self.right_freq = 440
        self.left_wave = 'sine'
        self.right_wave = 'sine'
        self.left_amp = 0.2
        self.right_amp = 0.2
        self.buffer_time = 1.0
        self.last_params = None
        self.channel = None
        self.latest_stereo = None
        self.is_file_mode = False
        self.generate_continuous_buffer()

        self.timer = QTimer()
        self.timer.timeout.connect(self.check_update_needed)
        self.timer.start(100)

    def check_update_needed(self):
        if self.is_file_mode:
            return
        current_params = (self.left_wave, self.left_freq, self.left_amp, self.right_wave, self.right_freq, self.right_amp)
        if current_params != self.last_params:
            self.generate_continuous_buffer()

    def generate_continuous_buffer(self):
        if self.is_file_mode:
            return
        self.last_params = (self.left_wave, self.left_freq, self.left_amp, self.right_wave, self.right_freq, self.right_amp)
        duration = 10.0
        l = generate_wave(self.left_wave, self.left_freq, self.left_amp, duration=duration)
        r = generate_wave(self.right_wave, self.right_freq, self.right_amp, duration=duration)
        stereo = np.vstack((l, r)).T
        self.latest_stereo = stereo.copy()
        stereo_int = (stereo * 32767).astype(np.int16)
        new_sound = pygame.sndarray.make_sound(stereo_int.copy())
        if self.channel:
            self.channel.stop()
        self.channel = new_sound.play(loops=-1)

    def play_audio_file(self, file_path):
        self.is_file_mode = True
        if self.channel:
            self.channel.stop()

        with tempfile.NamedTemporaryFile(delete=False, suffix=".raw") as tmp:
            raw_path = tmp.name

        cmd = [
            "ffmpeg", "-y", "-i", file_path,
            "-f", "s16le", "-acodec", "pcm_s16le", "-ac", "2", "-ar", "44100", raw_path
        ]
        subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

        raw_audio = np.fromfile(raw_path, dtype=np.int16)
        os.remove(raw_path)

        stereo = raw_audio.reshape(-1, 2) / 32768.0
        self.latest_stereo = stereo.copy()

        temp_wav = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
        wav_path = temp_wav.name
        temp_wav.close()

        cmd_wav = ["ffmpeg", "-y", "-i", file_path, wav_path]
        subprocess.run(cmd_wav, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

        sound = pygame.mixer.Sound(wav_path)
        self.channel = sound.play()
        self.file_start_time = time.time()
        self.file_duration = len(self.latest_stereo) / 44100

class MainWindow(QWidget, AudioMuteMixin):
    def __init__(self):
        super().__init__()
        self.audio = AudioOutput()
        self.trail_alpha = 180
        self.glow_intensity = 100
        self.scale_factor = 0.45
        self.x_scale_factor = 0.45
        self.y_scale_factor = 0.45
        self.hue = 120

        layout = QVBoxLayout()

        mode_layout = QHBoxLayout()
        self.mode_selector = QComboBox()
        self.mode_selector.addItems(["Tone Generator", "Audio File"])
        self.mode_selector.currentTextChanged.connect(self.switch_audio_mode)
        self.load_button = QPushButton("Load File")
        self.load_button.clicked.connect(self.load_audio_file)
        self.load_button.setEnabled(False)
        mode_layout.addWidget(QLabel("Mode:"))
        mode_layout.addWidget(self.mode_selector)
        mode_layout.addWidget(self.load_button)
        layout.addLayout(mode_layout)

        control_panel = QHBoxLayout()
        self.left_waveform = QComboBox()
        self.left_waveform.addItems(["sine", "square", "triangle", "sawtooth"])
        self.left_waveform.currentTextChanged.connect(lambda text: setattr(self.audio, 'left_wave', text))
        control_panel.addWidget(QLabel("Left Wave"))
        control_panel.addWidget(self.left_waveform)

        self.left_freq_slider = QSlider(Qt.Orientation.Horizontal)
        self.left_freq_slider.setRange(20, 2000)
        self.left_freq_slider.setValue(self.audio.left_freq)
        self.left_freq_slider.valueChanged.connect(lambda val: setattr(self.audio, 'left_freq', val))
        self.left_freq_spin = QDoubleSpinBox()
        self.left_freq_spin.setRange(20, 2000)
        self.left_freq_spin.setDecimals(1)
        self.left_freq_spin.setSingleStep(0.1)
        self.left_freq_spin.setValue(self.audio.left_freq)
        self.left_freq_spin.valueChanged.connect(lambda val: setattr(self.audio, 'left_freq', val))
        self.left_freq_slider.valueChanged.connect(lambda val: self.left_freq_spin.setValue(float(val)))
        self.left_freq_spin.valueChanged.connect(lambda val: self.left_freq_slider.setValue(int(val)))
        control_panel.addWidget(QLabel("Left Freq"))
        control_panel.addWidget(self.left_freq_slider)
        control_panel.addWidget(self.left_freq_spin)

        self.right_waveform = QComboBox()
        self.right_waveform.addItems(["sine", "square", "triangle", "sawtooth"])
        self.right_waveform.currentTextChanged.connect(lambda text: setattr(self.audio, 'right_wave', text))
        control_panel.addWidget(QLabel("Right Wave"))
        control_panel.addWidget(self.right_waveform)

        self.right_freq_slider = QSlider(Qt.Orientation.Horizontal)
        self.right_freq_slider.setRange(20, 2000)
        self.right_freq_slider.setValue(self.audio.right_freq)
        self.right_freq_slider.valueChanged.connect(lambda val: setattr(self.audio, 'right_freq', val))
        self.right_freq_spin = QDoubleSpinBox()
        self.right_freq_spin.setRange(20, 2000)
        self.right_freq_spin.setDecimals(1)
        self.right_freq_spin.setSingleStep(0.1)
        self.right_freq_spin.setValue(self.audio.right_freq)
        self.right_freq_spin.valueChanged.connect(lambda val: setattr(self.audio, 'right_freq', val))
        self.right_freq_slider.valueChanged.connect(lambda val: self.right_freq_spin.setValue(float(val)))
        self.right_freq_spin.valueChanged.connect(lambda val: self.right_freq_slider.setValue(int(val)))
        control_panel.addWidget(QLabel("Right Freq"))
        control_panel.addWidget(self.right_freq_slider)
        control_panel.addWidget(self.right_freq_spin)

        self.left_vol_slider = QSlider(Qt.Orientation.Horizontal)
        self.left_vol_slider.setRange(0, 100)
        self.left_vol_slider.setValue(int(self.audio.left_amp * 100))
        self.left_vol_slider.valueChanged.connect(lambda val: setattr(self.audio, 'left_amp', val / 100))
        control_panel.addWidget(QLabel("Left Vol"))
        control_panel.addWidget(self.left_vol_slider)

        self.right_vol_slider = QSlider(Qt.Orientation.Horizontal)
        self.right_vol_slider.setRange(0, 100)
        self.right_vol_slider.setValue(int(self.audio.right_amp * 100))
        self.right_vol_slider.valueChanged.connect(lambda val: setattr(self.audio, 'right_amp', val / 100))
        control_panel.addWidget(QLabel("Right Vol"))
        control_panel.addWidget(self.right_vol_slider)

        self.mute_button = QPushButton("Mute")
        self.mute_button.clicked.connect(self.toggle_mute)
        control_panel.addWidget(self.mute_button)
        layout.addLayout(control_panel)

        scope_section = QHBoxLayout()
        self.scope = QLabel()
        self.scope.setFixedSize(512, 512)
        scope_section.addWidget(self.scope)

        scope_controls = QVBoxLayout()
        self.invert_y_checkbox = QCheckBox("Invert Y")
        self.invert_y_checkbox.setChecked(False)
        scope_controls.addWidget(self.invert_y_checkbox)

        self.x_scale_slider = QSlider(Qt.Orientation.Horizontal)
        self.x_scale_slider.setRange(10, 200)
        self.x_scale_slider.setValue(int(self.x_scale_factor * 100))
        self.x_scale_slider.valueChanged.connect(lambda val: setattr(self, 'x_scale_factor', val / 100.0))
        scope_controls.addWidget(QLabel("X Scale"))
        scope_controls.addWidget(self.x_scale_slider)

        self.y_scale_slider = QSlider(Qt.Orientation.Horizontal)
        self.y_scale_slider.setRange(10, 200)
        self.y_scale_slider.setValue(int(self.y_scale_factor * 100))
        self.y_scale_slider.valueChanged.connect(lambda val: setattr(self, 'y_scale_factor', val / 100.0))
        scope_controls.addWidget(QLabel("Y Scale"))
        scope_controls.addWidget(self.y_scale_slider)

        self.trail_slider = QSlider(Qt.Orientation.Horizontal)
        self.trail_slider.setRange(0, 255)
        self.trail_slider.setValue(self.trail_alpha)
        self.trail_slider.valueChanged.connect(lambda val: setattr(self, 'trail_alpha', val))
        scope_controls.addWidget(QLabel("Trail"))
        scope_controls.addWidget(self.trail_slider)

        self.hue_slider = QSlider(Qt.Orientation.Horizontal)
        self.hue_slider.setRange(0, 360)
        self.hue_slider.setValue(self.hue)
        self.hue_slider.valueChanged.connect(lambda val: setattr(self, 'hue', val))
        scope_controls.addWidget(QLabel("Hue"))
        scope_controls.addWidget(self.hue_slider)

        scope_section.addLayout(scope_controls)
        layout.addLayout(scope_section)

        self.setLayout(layout)
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_scope)
        self.timer.start(33)
        self.keyPressEvent = self.handle_key

    def switch_audio_mode(self, mode):
        if mode == "Audio File":
            self.load_button.setEnabled(True)
            self.audio.is_file_mode = True
        else:
            self.load_button.setEnabled(False)
            self.audio.is_file_mode = False
            self.audio.generate_continuous_buffer()

    def load_audio_file(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Open Audio File", "", "Audio Files (*.mp3 *.wav *.flac)")
        if file_path:
            self.audio.play_audio_file(file_path)

    def update_scope(self):
        if self.audio.latest_stereo is None:
            return
        buffer = self.audio.latest_stereo
        window_size = 2048
        w, h = 512, 512
        img = QImage(w, h, QImage.Format.Format_ARGB32)
        img.fill(QColor(0, 0, 0, 255))
        painter = QPainter(img)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        center_x, center_y = w // 2, h // 2
        x_scale = min(w, h) * self.x_scale_factor
        y_scale = min(w, h) * self.y_scale_factor
        invert_y = self.invert_y_checkbox.isChecked()

        if self.audio.is_file_mode:
            elapsed = time.time() - getattr(self.audio, 'file_start_time', 0)
            offset = int(elapsed * 44100)
            if offset + window_size > len(buffer):
                offset = max(0, len(buffer) - window_size)
        else:
            t = time.time()
            offset = int((t * 44100) % (len(buffer) - window_size))

        data = buffer[offset:offset + window_size]
        path = []
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        for point in data:
            x = float(center_x + point[0] * x_scale)
            y = float(center_y + point[1] * y_scale if invert_y else center_y - point[1] * y_scale)
            path.append(QPointF(x, y))

        trail_len = len(path)
        for i in range(1, trail_len):
            age = i / trail_len
            fade_alpha = int((1.0 - age) * self.trail_alpha)
            glow_width = max(1, int(1 + (1 - age) * 2))
            color = QColor.fromHsv(self.hue, 255, 255, fade_alpha)
            pen = QPen(color)
            pen.setWidth(glow_width)
            painter.setPen(pen)
            painter.drawLine(path[i - 1], path[i])

        if path:
            last_point = path[-1]
            glow_pen = QPen(QColor.fromHsv(self.hue, 255, 255, self.glow_intensity))
            glow_pen.setWidth(6)
            painter.setPen(glow_pen)
            painter.drawPoint(last_point)
            core_pen = QPen(QColor(255, 255, 255))
            core_pen.setWidth(1)
            painter.setPen(core_pen)
            painter.drawPoint(last_point)

        painter.end()
        self.scope.setPixmap(QPixmap.fromImage(img))

    def handle_key(self, event):
        if event.key() == Qt.Key.Key_Escape:
            if hasattr(self.audio, 'channel') and self.audio.channel:
                self.audio.channel.stop()
            self.close()

if __name__ == '__main__':
    app = QApplication(sys.argv)
    win = MainWindow()
    win.show()
    sys.exit(app.exec())
    


