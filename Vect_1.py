import sys
import pyqtgraph as pg
import os
import numpy as np
import pygame
import time
import tempfile
import subprocess
try:
    from pydub import AudioSegment
    PYDUB_AVAILABLE = True
except ImportError:
    PYDUB_AVAILABLE = False

try:
    from scipy.io import wavfile
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
from PyQt6.QtWidgets import (
    QApplication, QWidget, QLabel, QSizePolicy, QVBoxLayout, QHBoxLayout,
    QSlider, QComboBox, QFileDialog, QPushButton, QDoubleSpinBox, QCheckBox, QStackedWidget, QFrame, QMainWindow
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
        pygame.mixer.init(frequency=44100, size=-16, channels=2, buffer=1024)
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
        
        # File playback variables - much simpler approach
        self.file_audio_data = None
        self.file_start_time = None
        self.file_sample_rate = 44100
        self.file_playing = False
        
        self.generate_continuous_buffer()

        self.timer = QTimer()
        self.timer.timeout.connect(self.check_update_needed)
        self.timer.start(33)  # Back to 30fps

    def check_update_needed(self):
        if self.is_file_mode:
            self.update_file_playback()
        else:
            current_params = (self.left_wave, self.left_freq, self.left_amp, self.right_wave, self.right_freq, self.right_amp)
            if current_params != self.last_params:
                self.generate_continuous_buffer()
    
    def update_file_playback(self):
        """Simple file playback - just calculate position based on time"""
        if not self.file_playing or self.file_audio_data is None:
            return
        
        # Initialize start time if not set
        if self.file_start_time is None:
            self.file_start_time = time.time()
            
        # Calculate where we should be in the file
        elapsed = time.time() - self.file_start_time
        total_duration = len(self.file_audio_data) / self.file_sample_rate
        
        # Loop the file
        playback_position = elapsed % total_duration
        sample_position = int(playback_position * self.file_sample_rate)
        
        # Extract a window for display - much smaller window
        window_size = 1024
        start_pos = sample_position
        end_pos = start_pos + window_size
        
        # Handle wraparound
        if end_pos > len(self.file_audio_data):
            # Split across the boundary
            part1 = self.file_audio_data[start_pos:]
            needed = window_size - len(part1)
            part2 = self.file_audio_data[:needed] if needed > 0 else np.empty((0, 2))
            self.latest_stereo = np.vstack([part1, part2]) if len(part2) > 0 else part1
        else:
            self.latest_stereo = self.file_audio_data[start_pos:end_pos]

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

        try:
            print(f"Loading audio file: {file_path}")
            
            if PYDUB_AVAILABLE:
                # Use pydub for audio conversion
                audio = AudioSegment.from_file(file_path)
                
                # Convert to stereo if mono
                if audio.channels == 1:
                    audio = audio.set_channels(2)
                
                # Convert to 44.1kHz, 16-bit
                audio = audio.set_frame_rate(44100)
                audio = audio.set_sample_width(2)
                
                # Get raw audio data and store it
                raw_data = audio.raw_data
                audio_array = np.frombuffer(raw_data, dtype=np.int16)
                self.file_audio_data = audio_array.reshape(-1, 2) / 32768.0
                self.file_sample_rate = 44100
                
                print(f"Audio loaded: {len(self.file_audio_data)} samples, {len(self.file_audio_data)/44100:.2f} seconds")
                
                # Create temporary WAV for pygame playback
                temp_wav = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
                wav_path = temp_wav.name
                temp_wav.close()
                
                # Export as WAV for pygame
                audio.export(wav_path, format="wav")
                
            else:
                # Fallback for WAV files
                if not file_path.lower().endswith(('.wav', '.ogg')):
                    raise Exception("Without pydub, only WAV and OGG files are supported")
                
                import wave
                with wave.open(file_path, 'rb') as wav_file:
                    frames = wav_file.readframes(-1)
                    sample_rate = wav_file.getframerate()
                    channels = wav_file.getnchannels()
                    
                    audio_array = np.frombuffer(frames, dtype=np.int16)
                    
                    if channels == 1:
                        audio_array = np.repeat(audio_array, 2)
                    
                    if sample_rate != 44100:
                        # Simple resampling
                        resample_factor = 44100 / sample_rate
                        new_length = int(len(audio_array) * resample_factor)
                        indices = np.linspace(0, len(audio_array) - 1, new_length)
                        audio_array = np.interp(indices, np.arange(len(audio_array)), audio_array).astype(np.int16)
                    
                    self.file_audio_data = audio_array.reshape(-1, 2) / 32768.0
                    self.file_sample_rate = 44100
                    wav_path = file_path

            # Initialize playback timing
            self.playback_start_time = time.time()
            self.file_position = 0
            
            # Start pygame playback
            sound = pygame.mixer.Sound(wav_path)
            self.channel = sound.play(loops=-1)  # Loop the audio
            
            # Initialize the display buffer
            self.latest_stereo = self.file_audio_data[:2048].copy()
            
            # Clean up temporary file
            if PYDUB_AVAILABLE and wav_path != file_path:
                def cleanup_later():
                    time.sleep(2)
                    try:
                        os.remove(wav_path)
                    except:
                        pass
                import threading
                threading.Thread(target=cleanup_later, daemon=True).start()
                    
        except Exception as e:
            print(f"Error loading audio file: {e}")
            print("Falling back to tone generator mode")
            self.is_file_mode = False
            self.generate_continuous_buffer()

class FullscreenScopeWindow(QWidget):
    def __init__(self, main_window):
        super().__init__()
        self.main_window = main_window
        self.setWindowTitle("Vectorscope - Fullscreen")
        self.setWindowFlags(Qt.WindowType.Window)
        
        layout = QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        
        # Create a new scope label for fullscreen
        self.scope = QLabel()
        self.scope.setMinimumSize(800, 600)
        self.scope.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        layout.addWidget(self.scope)
        
        # Add a button to exit fullscreen
        exit_btn = QPushButton("Exit Fullscreen (ESC)")
        exit_btn.clicked.connect(self.close)
        layout.addWidget(exit_btn)
        
        self.setLayout(layout)
    
    def keyPressEvent(self, event):
        if event.key() == Qt.Key.Key_Escape:
            self.close()
        super().keyPressEvent(event)
    
    def closeEvent(self, event):
        self.main_window.exit_fullscreen()
        event.accept()

class MainWindow(QMainWindow, AudioMuteMixin):
    def __init__(self):
        super().__init__()
        self.audio = AudioOutput()
        self.trail_alpha = 180
        self.glow_intensity = 100
        self.scale_factor = 0.45
        self.x_scale_factor = 0.45
        self.y_scale_factor = 0.45
        self.hue = 120
        self.fullscreen_window = None

        # Create central widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        layout = QVBoxLayout(central_widget)

        # Mode selection
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

        # Control panel
        control_panel = QHBoxLayout()
        
        # Left channel controls
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

        # Right channel controls
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

        # Volume controls
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

        # Mute and fullscreen buttons
        self.mute_button = QPushButton("Mute")
        self.mute_button.clicked.connect(self.toggle_mute)
        control_panel.addWidget(self.mute_button)
        
        self.toggle_scope_btn = QPushButton("Fullscreen Scope")
        self.toggle_scope_btn.clicked.connect(self.toggle_scope_fullscreen)
        control_panel.addWidget(self.toggle_scope_btn)
        
        layout.addLayout(control_panel)

        # Scope section
        scope_section = QHBoxLayout()
        
        # Main scope display
        self.scope = QLabel()
        self.scope.setMinimumSize(512, 512)
        self.scope.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        scope_section.addWidget(self.scope)

        # Scope controls
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

        # Set up timer for scope updates
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_scope)
        self.timer.start(33)  # ~30 FPS

    def toggle_scope_fullscreen(self):
        if self.fullscreen_window is None:
            self.fullscreen_window = FullscreenScopeWindow(self)
            self.fullscreen_window.showMaximized()
            self.toggle_scope_btn.setText("Exit Fullscreen Scope")
        else:
            self.fullscreen_window.close()

    def exit_fullscreen(self):
        if self.fullscreen_window:
            self.fullscreen_window = None
            self.toggle_scope_btn.setText("Fullscreen Scope")

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

    def update_scope_display(self, scope_widget):
        if self.audio.latest_stereo is None:
            return
            
        buffer = self.audio.latest_stereo
        w = scope_widget.width()
        h = scope_widget.height()
        
        if w <= 0 or h <= 0:
            return
            
        img = QImage(w, h, QImage.Format.Format_ARGB32)
        img.fill(QColor(0, 0, 0, 255))
        painter = QPainter(img)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        center_x, center_y = w // 2, h // 2
        x_scale = min(w, h) * self.x_scale_factor
        y_scale = min(w, h) * self.y_scale_factor
        invert_y = self.invert_y_checkbox.isChecked()

        if self.audio.is_file_mode:
            # Better timing calculation for file playback
            if hasattr(self.audio, 'channel') and self.audio.channel:
                # Check if sound is still playing
                if not self.audio.channel.get_busy():
                    # Sound finished, loop back to beginning
                    self.audio.file_start_time = time.time()
                    if hasattr(self.audio, 'file_duration'):
                        # Restart playback
                        try:
                            temp_wav = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
                            wav_path = temp_wav.name
                            temp_wav.close()
                            
                            # Re-export from our stored audio data
                            stereo_int = (self.audio.latest_stereo * 32767).astype(np.int16)
                            from scipy.io import wavfile
                            wavfile.write(wav_path, 44100, stereo_int)
                            
                            sound = pygame.mixer.Sound(wav_path)
                            self.audio.channel = sound.play()
                        except:
                            pass
            
            elapsed = time.time() - getattr(self.audio, 'file_start_time', 0)
            sample_offset = int(elapsed * 44100)
            
            # Use a smaller window for smoother display
            window_size = 1024  # Reduced from 2048
            
            # Ensure we don't go past the end
            if sample_offset + window_size > len(buffer):
                sample_offset = max(0, len(buffer) - window_size)
                
            # If we're at the very end, wrap around
            if sample_offset >= len(buffer) - window_size:
                sample_offset = 0
                
        else:
            # Tone generator mode
            window_size = 1024
            t = time.time()
            sample_offset = int((t * 44100) % (len(buffer) - window_size))

        # Extract the current window of audio data
        data = buffer[sample_offset:sample_offset + window_size]
        
        # Skip some samples for performance (display every nth sample)
        display_step = max(1, len(data) // 512)  # Show max 512 points
        data = data[::display_step]
        
        path = []
        
        for point in data:
            x = float(center_x + point[0] * x_scale)
            y = float(center_y + point[1] * y_scale if invert_y else center_y - point[1] * y_scale)
            path.append(QPointF(x, y))

        # Draw the trail
        trail_len = len(path)
        if trail_len > 1:
            for i in range(1, trail_len):
                age = i / trail_len
                fade_alpha = int((1.0 - age) * self.trail_alpha)
                if fade_alpha > 5:  # Only draw if visible enough
                    glow_width = max(1, int(1 + (1 - age) * 2))
                    color = QColor.fromHsv(self.hue, 255, 255, fade_alpha)
                    pen = QPen(color)
                    pen.setWidth(glow_width)
                    painter.setPen(pen)
                    painter.drawLine(path[i - 1], path[i])

        # Draw the current point with glow
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
        scope_widget.setPixmap(QPixmap.fromImage(img))

    def update_scope(self):
        # Update main scope
        self.update_scope_display(self.scope)
        
        # Update fullscreen scope if it exists
        if self.fullscreen_window and self.fullscreen_window.isVisible():
            self.update_scope_display(self.fullscreen_window.scope)

    def keyPressEvent(self, event):
        if event.key() == Qt.Key.Key_Escape:
            if self.fullscreen_window:
                self.fullscreen_window.close()
            else:
                if hasattr(self.audio, 'channel') and self.audio.channel:
                    self.audio.channel.stop()
                self.close()
        super().keyPressEvent(event)

    def closeEvent(self, event):
        if self.fullscreen_window:
            self.fullscreen_window.close()
        if hasattr(self.audio, 'channel') and self.audio.channel:
            self.audio.channel.stop()
        event.accept()

if __name__ == '__main__':
    app = QApplication(sys.argv)
    
    window = MainWindow()
    window.setWindowTitle("Vectorscope App")
    window.show()

    sys.exit(app.exec())