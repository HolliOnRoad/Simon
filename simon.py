import sys
import json
import threading
import subprocess
import tempfile
import shutil
import zipfile
from pathlib import Path
import os
from dataclasses import dataclass
from typing import List, Optional

import numpy as np
import sounddevice as sd
import requests

from PySide6 import QtCore, QtGui, QtWidgets

try:
    from faster_whisper import WhisperModel
except Exception:
    WhisperModel = None


@dataclass
class ChatMessage:
    role: str
    content: str


APP_VERSION = "1.0.0"
DEFAULT_UPDATE_URL = "https://raw.githubusercontent.com/HolliOnRoad/Simon/main/updates/simon.json"

STT_PRESETS = [
    ("fast", "Fast", {"model": "tiny", "device": "auto", "compute": "int8"}),
    ("balanced", "Balanced", {"model": "base", "device": "auto", "compute": "int8"}),
    ("accurate", "Accurate", {"model": "small", "device": "auto", "compute": "int8_float16"}),
    ("best", "Best", {"model": "large-v3", "device": "auto", "compute": "int8_float16"}),
]


def compare_versions(a: str, b: str) -> int:
    def parse(v: str):
        return [int(x) for x in v.strip().split(".") if x.isdigit()]
    pa = parse(a or "0")
    pb = parse(b or "0")
    max_len = max(len(pa), len(pb))
    pa += [0] * (max_len - len(pa))
    pb += [0] * (max_len - len(pb))
    if pa > pb:
        return 1
    if pa < pb:
        return -1
    return 0


def get_app_bundle_path() -> Optional[Path]:
    path = Path(sys.argv[0]).resolve()
    for parent in path.parents:
        if parent.suffix == ".app":
            return parent
    return None


def get_update_target_dir(app_path: Optional[Path]) -> Path:
    if app_path and os.access(app_path.parent, os.W_OK):
        return app_path.parent
    home_apps = Path.home() / "Applications"
    home_apps.mkdir(parents=True, exist_ok=True)
    return home_apps


def stage_update_from_zip(zip_path: Path, target_dir: Path, app_name: str) -> Path:
    extract_dir = Path(tempfile.mkdtemp(prefix="simon_update_"))
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(extract_dir)
    candidates = list(extract_dir.rglob("*.app"))
    if not candidates:
        raise RuntimeError("No .app found in update zip.")
    new_app = candidates[0]
    staged = target_dir / f"{app_name}.app.new"
    if staged.exists():
        shutil.rmtree(staged)
    shutil.copytree(new_app, staged, symlinks=True)
    return staged


def launch_update_helper(old_app: Path, new_app: Path, pid: int):
    script = f"""#!/bin/zsh
OLD=\"{old_app}\"
NEW=\"{new_app}\"
PID={pid}
while kill -0 $PID 2>/dev/null; do sleep 0.5; done
rm -rf \"$OLD\"
mv \"$NEW\" \"$OLD\"
open \"$OLD\"
"""
    helper = Path(tempfile.mkdtemp(prefix="simon_update_helper_")) / "update.sh"
    helper.write_text(script)
    helper.chmod(0o755)
    subprocess.Popen(["/bin/zsh", str(helper)], close_fds=True)


class AudioRecorder(QtCore.QObject):
    levelChanged = QtCore.Signal(float)

    def __init__(self, samplerate: int = 16000):
        super().__init__()
        self.samplerate = samplerate
        self.actual_samplerate = samplerate
        self.channels = 1
        self._frames: List[np.ndarray] = []
        self._stream: Optional[sd.InputStream] = None
        self.is_running = False
        self.device: Optional[int] = None

    def start(self, device: Optional[int] = None, samplerate: Optional[int] = None) -> None:
        if self.is_running:
            return
        self._frames = []
        self.device = device
        self.actual_samplerate = samplerate or self.samplerate
        self._stream = sd.InputStream(
            samplerate=self.actual_samplerate,
            channels=self.channels,
            device=self.device,
            dtype="float32",
            callback=self._callback,
        )
        self._stream.start()
        self.is_running = True

    def stop(self) -> tuple[np.ndarray, int]:
        if not self.is_running:
            return np.array([], dtype=np.float32), int(self.actual_samplerate)
        if self._stream is not None:
            self._stream.stop()
            self._stream.close()
            self._stream = None
        self.is_running = False
        if not self._frames:
            return np.array([], dtype=np.float32), int(self.actual_samplerate)
        audio = np.concatenate(self._frames, axis=0).squeeze()
        return audio.astype(np.float32), int(self.actual_samplerate)

    def _callback(self, indata, frames, time_info, status):
        if status:
            pass
        self._frames.append(indata.copy())
        rms = float(np.sqrt(np.mean(indata ** 2)))
        level = max(0.0, min(rms * 20.0, 1.0))
        self.levelChanged.emit(level)


class LevelMonitor(QtCore.QObject):
    levelChanged = QtCore.Signal(float)
    error = QtCore.Signal(str)

    def __init__(self, samplerate: int = 16000):
        super().__init__()
        self.samplerate = samplerate
        self._stream: Optional[sd.InputStream] = None
        self.is_running = False
        self.device: Optional[int] = None

    def start(self, device: Optional[int] = None, samplerate: Optional[int] = None) -> None:
        if self.is_running:
            return
        self.device = device
        actual_rate = samplerate or self.samplerate
        try:
            self._stream = sd.InputStream(
                samplerate=actual_rate,
                channels=1,
                device=self.device,
                dtype="float32",
                callback=self._callback,
            )
            self._stream.start()
            self.is_running = True
        except Exception as exc:
            self.error.emit(str(exc))

    def stop(self) -> None:
        if not self.is_running:
            return
        if self._stream is not None:
            self._stream.stop()
            self._stream.close()
            self._stream = None
        self.is_running = False

    def _callback(self, indata, frames, time_info, status):
        if status:
            pass
        rms = float(np.sqrt(np.mean(indata ** 2)))
        level = max(0.0, min(rms * 20.0, 1.0))
        self.levelChanged.emit(level)


class WhisperManager:
    def __init__(self):
        self._model = None
        self._config = None

    def transcribe(
        self,
        audio: np.ndarray,
        language: str,
        model_name: str,
        device: str,
        compute_type: str,
    ) -> tuple[str, Optional[str]]:
        if WhisperModel is None:
            raise RuntimeError("faster-whisper is not installed.")
        if audio.size == 0:
            return "", None

        warning = None
        config = (model_name, device, compute_type)
        if self._model is None or self._config != config:
            try:
                self._model = WhisperModel(model_name, device=device, compute_type=compute_type)
                self._config = config
            except Exception as exc:
                warning = f"STT fallback to CPU/int8 (reason: {exc})"
                try:
                    self._model = WhisperModel(model_name, device="cpu", compute_type="int8")
                    self._config = (model_name, "cpu", "int8")
                except Exception as exc2:
                    raise RuntimeError(f"Failed to init Whisper model: {exc2}") from exc2
        segments, _ = self._model.transcribe(
            audio,
            language=language,
            beam_size=5,
            vad_filter=True,
            vad_parameters={"min_silence_duration_ms": 500},
        )
        text = "".join(segment.text for segment in segments).strip()
        return text, warning


def resample_audio(audio: np.ndarray, src_rate: int, dst_rate: int) -> np.ndarray:
    if audio.size == 0 or src_rate == dst_rate:
        return audio
    duration = audio.shape[0] / float(src_rate)
    target_len = int(duration * dst_rate)
    if target_len <= 1:
        return audio
    x_old = np.linspace(0.0, duration, num=audio.shape[0], endpoint=False)
    x_new = np.linspace(0.0, duration, num=target_len, endpoint=False)
    return np.interp(x_new, x_old, audio).astype(np.float32)


class WorkerSignals(QtCore.QObject):
    finished = QtCore.Signal(object)
    error = QtCore.Signal(str)


class TranscriptionWorker(QtCore.QRunnable):
    def __init__(
        self,
        whisper: WhisperManager,
        audio: np.ndarray,
        language: str,
        model_name: str,
        device: str,
        compute_type: str,
    ):
        super().__init__()
        self.whisper = whisper
        self.audio = audio
        self.language = language
        self.model_name = model_name
        self.device = device
        self.compute_type = compute_type
        self.signals = WorkerSignals()

    def run(self):
        try:
            text, warning = self.whisper.transcribe(
                self.audio, self.language, self.model_name, self.device, self.compute_type
            )
            self.signals.finished.emit({"text": text, "warning": warning})
        except Exception as exc:
            self.signals.error.emit(str(exc))


class MicTestWorker(QtCore.QRunnable):
    def __init__(self, device: Optional[int], samplerate: int, duration: float = 1.5):
        super().__init__()
        self.device = device
        self.samplerate = samplerate
        self.duration = duration
        self.signals = WorkerSignals()

    def run(self):
        try:
            frames = int(self.duration * self.samplerate)
            recording = sd.rec(
                frames,
                samplerate=self.samplerate,
                channels=1,
                dtype="float32",
                device=self.device,
            )
            sd.wait()
            rms = float(np.sqrt(np.mean(recording ** 2))) if recording.size else 0.0
            level = max(0.0, min(rms * 20.0, 1.0))
            self.signals.finished.emit(level)
        except Exception as exc:
            self.signals.error.emit(str(exc))


class UpdateCheckWorker(QtCore.QRunnable):
    def __init__(self, update_url: str, current_version: str):
        super().__init__()
        self.update_url = update_url
        self.current_version = current_version
        self.signals = WorkerSignals()

    def run(self):
        try:
            resp = requests.get(self.update_url, timeout=20)
            if resp.status_code != 200:
                raise RuntimeError(f"Update check HTTP {resp.status_code}")
            data = resp.json()
            latest = str(data.get("version", "")).strip()
            url = str(data.get("url", "")).strip()
            if not latest or not url:
                raise RuntimeError("Update JSON missing version or url")
            if compare_versions(latest, self.current_version) > 0:
                self.signals.finished.emit({"status": "update", "version": latest, "url": url})
            else:
                self.signals.finished.emit({"status": "up_to_date", "version": latest})
        except Exception as exc:
            self.signals.error.emit(str(exc))


class UpdateDownloadWorker(QtCore.QRunnable):
    def __init__(self, url: str, app_name: str):
        super().__init__()
        self.url = url
        self.app_name = app_name
        self.signals = WorkerSignals()

    def run(self):
        try:
            resp = requests.get(self.url, stream=True, timeout=60)
            if resp.status_code != 200:
                raise RuntimeError(f"Download HTTP {resp.status_code}")
            tmp_fd, tmp_path = tempfile.mkstemp(suffix=".zip")
            with os.fdopen(tmp_fd, "wb") as f:
                for chunk in resp.iter_content(chunk_size=1024 * 1024):
                    if chunk:
                        f.write(chunk)
            self.signals.finished.emit({"zip": tmp_path})
        except Exception as exc:
            self.signals.error.emit(str(exc))


class LLMWorker(QtCore.QRunnable):
    def __init__(self, provider: str, model: str, api_base: str, api_key: str, messages: List[ChatMessage]):
        super().__init__()
        self.provider = provider
        self.model = model
        self.api_base = api_base
        self.api_key = api_key
        self.messages = messages
        self.signals = WorkerSignals()

    def run(self):
        try:
            reply = self._send()
            self.signals.finished.emit(reply)
        except Exception as exc:
            self.signals.error.emit(str(exc))

    def _send(self) -> str:
        payload_messages = [{"role": m.role, "content": m.content} for m in self.messages]

        if self.provider == "ollama":
            base = self.api_base or "http://localhost:11434"
            url = base.rstrip("/") + "/api/chat"
            payload = {
                "model": self.model or "llama3.1:8b",
                "messages": payload_messages,
                "stream": False,
            }
            resp = requests.post(url, json=payload, timeout=120)
            if resp.status_code != 200:
                raise RuntimeError(f"Ollama HTTP {resp.status_code}")
            data = resp.json()
            if "message" in data and "content" in data["message"]:
                return data["message"]["content"]
            if "response" in data:
                return data["response"]
            raise RuntimeError("Invalid Ollama response")

        if self.provider == "openai":
            if not self.api_key:
                raise RuntimeError("Missing API key")
            if not self.model:
                raise RuntimeError("Missing model name")
            base = self.api_base or "https://api.openai.com/v1"
            url = base.rstrip("/") + "/chat/completions"
            payload = {
                "model": self.model,
                "messages": payload_messages,
                "temperature": 0.7,
            }
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            }
            resp = requests.post(url, headers=headers, json=payload, timeout=120)
            if resp.status_code != 200:
                raise RuntimeError(f"API HTTP {resp.status_code}")
            data = resp.json()
            choices = data.get("choices") or []
            if choices:
                message = choices[0].get("message") or {}
                content = message.get("content")
                if content:
                    return content
            raise RuntimeError("Invalid API response")

        raise RuntimeError("Unknown provider")


def escape_html(text: str) -> str:
    return (
        text.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace("\n", "<br>")
    )


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Simon")
        self.resize(980, 720)

        self.settings = QtCore.QSettings("Simon", "Simon")
        self.threadpool = QtCore.QThreadPool.globalInstance()
        self.recorder = AudioRecorder()
        self.recorder.levelChanged.connect(self.on_level)
        self.monitor = LevelMonitor()
        self.monitor.levelChanged.connect(self.on_monitor_level)
        self.monitor.error.connect(self.on_monitor_error)
        self.whisper = WhisperManager()
        self.messages: List[ChatMessage] = []
        self._monitor_paused = False

        self._build_ui()
        self._load_settings()
        if self.auto_update_check.isChecked() and self.update_url_edit.text().strip():
            QtCore.QTimer.singleShot(1200, self.on_check_updates_clicked)

    def _build_ui(self):
        central = QtWidgets.QWidget()
        self.setCentralWidget(central)

        main_layout = QtWidgets.QVBoxLayout(central)
        main_layout.setSpacing(8)

        settings_box = QtWidgets.QGroupBox("Settings")
        settings_layout = QtWidgets.QGridLayout(settings_box)

        self.language_combo = QtWidgets.QComboBox()
        self.language_combo.addItem("Deutsch (DE)", "de")
        self.language_combo.addItem("English (US)", "en")

        self.provider_combo = QtWidgets.QComboBox()
        self.provider_combo.addItem("Local (Ollama)", "ollama")
        self.provider_combo.addItem("API (OpenAI-compatible)", "openai")
        self.provider_combo.currentIndexChanged.connect(self.on_provider_changed)

        self.model_edit = QtWidgets.QLineEdit()
        self.base_url_edit = QtWidgets.QLineEdit()
        self.api_key_edit = QtWidgets.QLineEdit()
        self.api_key_edit.setEchoMode(QtWidgets.QLineEdit.Password)

        self.stt_model_combo = QtWidgets.QComboBox()
        self.stt_model_combo.addItem("tiny", "tiny")
        self.stt_model_combo.addItem("base", "base")
        self.stt_model_combo.addItem("small", "small")
        self.stt_model_combo.addItem("medium", "medium")
        self.stt_model_combo.addItem("large-v3", "large-v3")

        self.stt_device_combo = QtWidgets.QComboBox()
        self.stt_device_combo.addItem("Auto (GPU if available)", "auto")
        self.stt_device_combo.addItem("CPU", "cpu")

        self.stt_compute_combo = QtWidgets.QComboBox()
        self.stt_compute_combo.addItem("int8 (fast)", "int8")
        self.stt_compute_combo.addItem("int8_float16 (balanced)", "int8_float16")
        self.stt_compute_combo.addItem("float16 (accurate)", "float16")

        self.stt_preset_combo = QtWidgets.QComboBox()
        self.stt_preset_combo.addItem("Custom", "custom")
        for key, label, _ in STT_PRESETS:
            self.stt_preset_combo.addItem(label, key)

        self.input_device_combo = QtWidgets.QComboBox()
        self.refresh_devices_button = QtWidgets.QPushButton("Refresh")
        self.refresh_devices_button.clicked.connect(self.populate_input_devices)
        self.monitor_check = QtWidgets.QCheckBox("Monitor")
        self.monitor_bar = QtWidgets.QProgressBar()
        self.monitor_bar.setRange(0, 100)
        self.monitor_bar.setValue(0)
        self.monitor_bar.setTextVisible(False)
        self.test_mic_button = QtWidgets.QPushButton("Auto-Test")
        self.test_mic_button.clicked.connect(self.on_test_mic_clicked)

        self.update_url_edit = QtWidgets.QLineEdit()
        self.update_url_edit.setPlaceholderText(DEFAULT_UPDATE_URL)
        self.auto_update_check = QtWidgets.QCheckBox("Auto-Check Updates")
        self.check_update_button = QtWidgets.QPushButton("Check Now")
        self.check_update_button.clicked.connect(self.on_check_updates_clicked)
        self.version_label = QtWidgets.QLabel(f"Version {APP_VERSION}")
        self.version_label.setAlignment(QtCore.Qt.AlignRight | QtCore.Qt.AlignVCenter)

        self.tts_combo = QtWidgets.QComboBox()
        self.tts_combo.addItem("System Voice", "system")
        self.tts_combo.addItem("Piper (HTTP)", "piper")
        self.tts_voice_edit = QtWidgets.QLineEdit()
        self.piper_endpoint_edit = QtWidgets.QLineEdit()

        self.system_prompt_edit = QtWidgets.QLineEdit()

        self.auto_speak_check = QtWidgets.QCheckBox("Auto-Speak")
        self.auto_send_check = QtWidgets.QCheckBox("Auto-Send on Stop")

        self._preset_lock = False
        self.stt_preset_combo.currentIndexChanged.connect(self.on_stt_preset_changed)
        self.stt_model_combo.currentIndexChanged.connect(self.on_stt_settings_changed)
        self.stt_device_combo.currentIndexChanged.connect(self.on_stt_settings_changed)
        self.stt_compute_combo.currentIndexChanged.connect(self.on_stt_settings_changed)
        self.monitor_check.toggled.connect(self.on_monitor_toggled)
        self.input_device_combo.currentIndexChanged.connect(self.on_input_device_changed)

        row = 0
        settings_layout.addWidget(QtWidgets.QLabel("Language"), row, 0)
        settings_layout.addWidget(self.language_combo, row, 1)
        settings_layout.addWidget(QtWidgets.QLabel("LLM"), row, 2)
        settings_layout.addWidget(self.provider_combo, row, 3)
        settings_layout.addWidget(QtWidgets.QLabel("Model"), row, 4)
        settings_layout.addWidget(self.model_edit, row, 5)
        settings_layout.addWidget(self.auto_speak_check, row, 6)

        row += 1
        settings_layout.addWidget(QtWidgets.QLabel("STT Model"), row, 0)
        settings_layout.addWidget(self.stt_model_combo, row, 1)
        settings_layout.addWidget(QtWidgets.QLabel("Device"), row, 2)
        settings_layout.addWidget(self.stt_device_combo, row, 3)
        settings_layout.addWidget(QtWidgets.QLabel("Compute"), row, 4)
        settings_layout.addWidget(self.stt_compute_combo, row, 5)
        settings_layout.addWidget(QtWidgets.QLabel("Preset"), row, 6)
        settings_layout.addWidget(self.stt_preset_combo, row, 7)

        row += 1
        settings_layout.addWidget(QtWidgets.QLabel("Input Device"), row, 0)
        settings_layout.addWidget(self.input_device_combo, row, 1, 1, 5)
        settings_layout.addWidget(self.refresh_devices_button, row, 6, 1, 2)

        row += 1
        settings_layout.addWidget(QtWidgets.QLabel("Mic Monitor"), row, 0)
        settings_layout.addWidget(self.monitor_check, row, 1)
        settings_layout.addWidget(self.monitor_bar, row, 2, 1, 4)
        settings_layout.addWidget(self.test_mic_button, row, 6, 1, 2)

        row += 1
        settings_layout.addWidget(QtWidgets.QLabel("API Base URL"), row, 0)
        settings_layout.addWidget(self.base_url_edit, row, 1, 1, 4)
        settings_layout.addWidget(QtWidgets.QLabel("API Key"), row, 5)
        settings_layout.addWidget(self.api_key_edit, row, 6, 1, 2)

        row += 1
        settings_layout.addWidget(QtWidgets.QLabel("Updates"), row, 0)
        settings_layout.addWidget(self.update_url_edit, row, 1, 1, 4)
        settings_layout.addWidget(self.auto_update_check, row, 5)
        settings_layout.addWidget(self.check_update_button, row, 6, 1, 2)

        row += 1
        settings_layout.addWidget(QtWidgets.QLabel("TTS"), row, 0)
        settings_layout.addWidget(self.tts_combo, row, 1)
        settings_layout.addWidget(QtWidgets.QLabel("TTS Voice"), row, 2)
        settings_layout.addWidget(self.tts_voice_edit, row, 3)
        settings_layout.addWidget(QtWidgets.QLabel("Piper Endpoint"), row, 4)
        settings_layout.addWidget(self.piper_endpoint_edit, row, 5, 1, 3)

        row += 1
        settings_layout.addWidget(QtWidgets.QLabel("System Prompt"), row, 0)
        settings_layout.addWidget(self.system_prompt_edit, row, 1, 1, 7)
        row += 1
        settings_layout.addWidget(self.version_label, row, 6, 1, 2)

        main_layout.addWidget(settings_box)

        self.chat_view = QtWidgets.QTextEdit()
        self.chat_view.setReadOnly(True)
        self.chat_view.setAcceptRichText(True)
        self.chat_view.setPlaceholderText("Conversation will appear here...")
        main_layout.addWidget(self.chat_view, 1)

        level_layout = QtWidgets.QHBoxLayout()
        self.listen_status = QtWidgets.QLabel("Not Listening")
        self.level_bar = QtWidgets.QProgressBar()
        self.level_bar.setRange(0, 100)
        self.level_bar.setValue(0)
        self.level_bar.setTextVisible(False)
        level_layout.addWidget(self.listen_status)
        level_layout.addWidget(self.level_bar, 1)
        main_layout.addLayout(level_layout)

        input_layout = QtWidgets.QHBoxLayout()
        self.input_edit = QtWidgets.QPlainTextEdit()
        self.input_edit.setPlaceholderText("Type a message...")
        self.input_edit.setFixedHeight(70)
        input_layout.addWidget(self.input_edit, 1)

        button_layout = QtWidgets.QVBoxLayout()
        self.listen_button = QtWidgets.QPushButton("Listen")
        self.listen_button.clicked.connect(self.on_listen_clicked)
        self.send_button = QtWidgets.QPushButton("Send")
        self.send_button.clicked.connect(self.on_send_clicked)
        button_layout.addWidget(self.listen_button)
        button_layout.addWidget(self.send_button)
        button_layout.addStretch(1)

        input_layout.addLayout(button_layout)
        main_layout.addLayout(input_layout)

        status_layout = QtWidgets.QHBoxLayout()
        self.status_label = QtWidgets.QLabel("Idle")
        status_layout.addWidget(self.status_label)
        status_layout.addStretch(1)
        status_layout.addWidget(self.auto_send_check)
        main_layout.addLayout(status_layout)

        self.populate_input_devices()

    def _load_settings(self):
        self.language_combo.setCurrentIndex(
            self.language_combo.findData(self.settings.value("language", "en"))
        )
        self.provider_combo.setCurrentIndex(
            self.provider_combo.findData(self.settings.value("provider", "ollama"))
        )
        self.model_edit.setText(self.settings.value("model", "llama3.1:8b"))
        self.base_url_edit.setText(self.settings.value("api_base", "http://localhost:11434"))
        self.api_key_edit.setText(self.settings.value("api_key", ""))
        self.stt_model_combo.setCurrentIndex(
            self.stt_model_combo.findData(self.settings.value("stt_model", "small"))
        )
        self.stt_device_combo.setCurrentIndex(
            self.stt_device_combo.findData(self.settings.value("stt_device", "auto"))
        )
        self.stt_compute_combo.setCurrentIndex(
            self.stt_compute_combo.findData(self.settings.value("stt_compute", "int8"))
        )
        self.sync_preset_from_stt()
        saved_device = self.settings.value("stt_input_device", -1)
        try:
            saved_device = int(saved_device)
        except Exception:
            saved_device = -1
        index = self.input_device_combo.findData(saved_device)
        if index >= 0:
            self.input_device_combo.setCurrentIndex(index)
        self.monitor_check.setChecked(self.settings.value("mic_monitor", False, bool))
        self.update_url_edit.setText(self.settings.value("update_url", DEFAULT_UPDATE_URL))
        self.auto_update_check.setChecked(self.settings.value("auto_update", False, bool))
        self.tts_combo.setCurrentIndex(
            self.tts_combo.findData(self.settings.value("tts_engine", "system"))
        )
        self.tts_voice_edit.setText(self.settings.value("tts_voice", ""))
        self.piper_endpoint_edit.setText(self.settings.value("piper_endpoint", "http://localhost:5002/tts"))
        self.system_prompt_edit.setText(
            self.settings.value("system_prompt", "Du bist ein hilfreicher KI-Agent. Antworte kurz und klar.")
        )
        self.auto_speak_check.setChecked(self.settings.value("auto_speak", True, bool))
        self.auto_send_check.setChecked(self.settings.value("auto_send", True, bool))

    def closeEvent(self, event):
        self._save_settings()
        super().closeEvent(event)

    def _save_settings(self):
        self.settings.setValue("language", self.language_combo.currentData())
        self.settings.setValue("provider", self.provider_combo.currentData())
        self.settings.setValue("model", self.model_edit.text().strip())
        self.settings.setValue("api_base", self.base_url_edit.text().strip())
        self.settings.setValue("api_key", self.api_key_edit.text().strip())
        self.settings.setValue("stt_model", self.stt_model_combo.currentData())
        self.settings.setValue("stt_device", self.stt_device_combo.currentData())
        self.settings.setValue("stt_compute", self.stt_compute_combo.currentData())
        self.settings.setValue("stt_preset", self.stt_preset_combo.currentData())
        self.settings.setValue("stt_input_device", self.input_device_combo.currentData())
        self.settings.setValue("mic_monitor", self.monitor_check.isChecked())
        self.settings.setValue("update_url", self.update_url_edit.text().strip())
        self.settings.setValue("auto_update", self.auto_update_check.isChecked())
        self.settings.setValue("tts_engine", self.tts_combo.currentData())
        self.settings.setValue("tts_voice", self.tts_voice_edit.text().strip())
        self.settings.setValue("piper_endpoint", self.piper_endpoint_edit.text().strip())
        self.settings.setValue("system_prompt", self.system_prompt_edit.text().strip())
        self.settings.setValue("auto_speak", self.auto_speak_check.isChecked())
        self.settings.setValue("auto_send", self.auto_send_check.isChecked())

    def on_provider_changed(self):
        provider = self.provider_combo.currentData()
        if provider == "ollama":
            if not self.base_url_edit.text().strip():
                self.base_url_edit.setText("http://localhost:11434")
            if not self.model_edit.text().strip():
                self.model_edit.setText("llama3.1:8b")
        else:
            if self.base_url_edit.text().strip() == "http://localhost:11434":
                self.base_url_edit.setText("")

    def populate_input_devices(self):
        current = self.input_device_combo.currentData()
        self.input_device_combo.clear()
        self.input_device_combo.addItem("System Default", -1)
        try:
            devices = sd.query_devices()
        except Exception as exc:
            self.status_label.setText(f"Audio devices error: {exc}")
            return

        for idx, dev in enumerate(devices):
            try:
                if dev.get("max_input_channels", 0) <= 0:
                    continue
                name = dev.get("name", f"Device {idx}")
                label = f"{idx}: {name}"
                self.input_device_combo.addItem(label, idx)
            except Exception:
                continue

        if current is not None:
            index = self.input_device_combo.findData(current)
            if index >= 0:
                self.input_device_combo.setCurrentIndex(index)
        if self.monitor_check.isChecked():
            self.restart_monitor()

    def on_stt_preset_changed(self):
        if self._preset_lock:
            return
        preset_key = self.stt_preset_combo.currentData()
        if preset_key == "custom":
            return
        preset = next((cfg for key, _, cfg in STT_PRESETS if key == preset_key), None)
        if not preset:
            return
        self._preset_lock = True
        self.stt_model_combo.setCurrentIndex(self.stt_model_combo.findData(preset["model"]))
        self.stt_device_combo.setCurrentIndex(self.stt_device_combo.findData(preset["device"]))
        self.stt_compute_combo.setCurrentIndex(self.stt_compute_combo.findData(preset["compute"]))
        self._preset_lock = False

    def on_stt_settings_changed(self):
        if self._preset_lock:
            return
        self.sync_preset_from_stt()

    def sync_preset_from_stt(self):
        current = {
            "model": self.stt_model_combo.currentData(),
            "device": self.stt_device_combo.currentData(),
            "compute": self.stt_compute_combo.currentData(),
        }
        preset_key = "custom"
        for key, _, cfg in STT_PRESETS:
            if cfg == current:
                preset_key = key
                break
        self._preset_lock = True
        index = self.stt_preset_combo.findData(preset_key)
        if index < 0:
            index = 0
        self.stt_preset_combo.setCurrentIndex(index)
        self._preset_lock = False

    def on_input_device_changed(self):
        if self.monitor_check.isChecked():
            self.restart_monitor()

    def on_monitor_toggled(self, checked: bool):
        if checked:
            self.start_monitor()
        else:
            self.stop_monitor()

    def start_monitor(self):
        device_index = self.input_device_combo.currentData()
        samplerate = 16000
        if device_index is not None and device_index != -1:
            try:
                dev_info = sd.query_devices(int(device_index))
                samplerate = int(dev_info.get("default_samplerate", 16000))
            except Exception:
                samplerate = 16000
        self.monitor.start(device=None if device_index == -1 else device_index, samplerate=samplerate)

    def stop_monitor(self):
        self.monitor.stop()
        self.monitor_bar.setValue(0)

    def restart_monitor(self):
        self.stop_monitor()
        self.start_monitor()

    def on_monitor_level(self, level: float):
        self.monitor_bar.setValue(int(level * 100))

    def on_monitor_error(self, message: str):
        self.status_label.setText(f"Monitor error: {message}")

    def on_test_mic_clicked(self):
        device_index = self.input_device_combo.currentData()
        samplerate = 16000
        if device_index is not None and device_index != -1:
            try:
                dev_info = sd.query_devices(int(device_index))
                samplerate = int(dev_info.get("default_samplerate", 16000))
            except Exception:
                samplerate = 16000

        if self.monitor_check.isChecked():
            self.stop_monitor()
            self._monitor_paused = True
        else:
            self._monitor_paused = False

        self.status_label.setText("Testing mic...")
        self.test_mic_button.setEnabled(False)

        worker = MicTestWorker(None if device_index == -1 else device_index, samplerate)
        worker.signals.finished.connect(self.on_test_mic_done)
        worker.signals.error.connect(self.on_test_mic_error)
        self.threadpool.start(worker)

    def on_test_mic_done(self, level: object):
        try:
            level_val = float(level)
        except Exception:
            level_val = 0.0
        self.monitor_bar.setValue(int(level_val * 100))
        self.status_label.setText(f"Mic test ok (level {level_val:.2f})")
        self.test_mic_button.setEnabled(True)
        if self.monitor_check.isChecked() and self._monitor_paused:
            self.start_monitor()
        self._monitor_paused = False

    def on_test_mic_error(self, message: str):
        self.status_label.setText(f"Mic test error: {message}")
        self.test_mic_button.setEnabled(True)
        if self.monitor_check.isChecked() and self._monitor_paused:
            self.start_monitor()
        self._monitor_paused = False

    def on_check_updates_clicked(self):
        url = self.update_url_edit.text().strip()
        if not url:
            self.status_label.setText("Update URL not set")
            return
        self.status_label.setText("Checking for updates...")
        self.check_update_button.setEnabled(False)
        worker = UpdateCheckWorker(url, APP_VERSION)
        worker.signals.finished.connect(self.on_update_check_done)
        worker.signals.error.connect(self.on_update_check_error)
        self.threadpool.start(worker)

    def on_update_check_done(self, result: object):
        self.check_update_button.setEnabled(True)
        if not isinstance(result, dict):
            self.status_label.setText("Update check failed")
            return
        status = result.get("status")
        if status == "up_to_date":
            self.status_label.setText("Up to date")
            return
        if status == "update":
            latest = result.get("version", "?")
            url = result.get("url", "")
            msg = QtWidgets.QMessageBox(self)
            msg.setWindowTitle("Update available")
            msg.setText(f"Version {latest} is available. Download and install now?")
            msg.setStandardButtons(QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No)
            if msg.exec() == QtWidgets.QMessageBox.Yes:
                self.status_label.setText("Downloading update...")
                self.check_update_button.setEnabled(False)
                worker = UpdateDownloadWorker(url, "Simon")
                worker.signals.finished.connect(self.on_update_download_done)
                worker.signals.error.connect(self.on_update_download_error)
                self.threadpool.start(worker)
            else:
                self.status_label.setText("Update postponed")
            return
        self.status_label.setText("Update check unknown status")

    def on_update_check_error(self, message: str):
        self.check_update_button.setEnabled(True)
        self.status_label.setText(f"Update check error: {message}")

    def on_update_download_done(self, result: object):
        self.check_update_button.setEnabled(True)
        if not isinstance(result, dict):
            self.status_label.setText("Update download failed")
            return
        zip_path = result.get("zip")
        if not zip_path:
            self.status_label.setText("Update download failed")
            return
        try:
            app_path = get_app_bundle_path()
            target_dir = get_update_target_dir(app_path)
            staged = stage_update_from_zip(Path(zip_path), target_dir, "Simon")
            old_app = app_path or (target_dir / "Simon.app")
            self.status_label.setText("Installing update...")
            launch_update_helper(old_app, staged, os.getpid())
            QtWidgets.QMessageBox.information(self, "Updating", "Simon will restart to complete the update.")
            QtCore.QCoreApplication.quit()
        except Exception as exc:
            self.status_label.setText(f"Update install error: {exc}")

    def on_update_download_error(self, message: str):
        self.check_update_button.setEnabled(True)
        self.status_label.setText(f"Update download error: {message}")

    def on_level(self, level: float):
        self.level_bar.setValue(int(level * 100))

    def on_listen_clicked(self):
        if not self.recorder.is_running:
            self.start_listening()
        else:
            self.stop_listening()

    def start_listening(self):
        try:
            if self.monitor_check.isChecked():
                self.stop_monitor()
                self._monitor_paused = True
            device_index = self.input_device_combo.currentData()
            samplerate = 16000
            if device_index is not None and device_index != -1:
                try:
                    dev_info = sd.query_devices(int(device_index))
                    samplerate = int(dev_info.get("default_samplerate", 16000))
                except Exception:
                    samplerate = 16000
            self.recorder.start(device=None if device_index == -1 else device_index, samplerate=samplerate)
            self.listen_status.setText("Listening")
            self.listen_button.setText("Stop")
            self.status_label.setText("Listening...")
        except Exception as exc:
            self.status_label.setText(f"Audio error: {exc}")

    def stop_listening(self):
        audio, sample_rate = self.recorder.stop()
        self.listen_status.setText("Not Listening")
        self.listen_button.setText("Listen")
        self.level_bar.setValue(0)
        self.status_label.setText("Transcribing...")
        if self.monitor_check.isChecked() and self._monitor_paused:
            self.start_monitor()
            self._monitor_paused = False

        model_name = self.stt_model_combo.currentData() or "small"
        device = self.stt_device_combo.currentData() or "auto"
        compute_type = self.stt_compute_combo.currentData() or "int8"
        language = self.language_combo.currentData() or "en"

        audio = resample_audio(audio, sample_rate, 16000)
        worker = TranscriptionWorker(self.whisper, audio, language, model_name, device, compute_type)
        worker.signals.finished.connect(self.on_transcription_done)
        worker.signals.error.connect(self.on_worker_error)
        self.threadpool.start(worker)

    def on_transcription_done(self, result: object):
        text = ""
        warning = None
        if isinstance(result, dict):
            text = result.get("text", "") or ""
            warning = result.get("warning")
        elif isinstance(result, str):
            text = result
        else:
            text = str(result)

        if warning:
            self.status_label.setText(warning)
        else:
            self.status_label.setText("Idle")

        if text:
            self.input_edit.setPlainText(text)
            if self.auto_send_check.isChecked():
                self.on_send_clicked()
        elif not warning:
            self.status_label.setText("No speech detected")

    def on_worker_error(self, message: str):
        self.status_label.setText(message)

    def on_send_clicked(self):
        text = self.input_edit.toPlainText().strip()
        if not text:
            return
        self.input_edit.clear()

        self.append_message("You", text)
        self.messages.append(ChatMessage(role="user", content=text))

        system_prompt = self.system_prompt_edit.text().strip()
        full_messages: List[ChatMessage] = []
        if system_prompt:
            full_messages.append(ChatMessage(role="system", content=system_prompt))
        full_messages.extend(self.messages)

        self.status_label.setText("Thinking...")
        self.send_button.setEnabled(False)
        self.listen_button.setEnabled(False)

        worker = LLMWorker(
            provider=self.provider_combo.currentData(),
            model=self.model_edit.text().strip(),
            api_base=self.base_url_edit.text().strip(),
            api_key=self.api_key_edit.text().strip(),
            messages=full_messages,
        )
        worker.signals.finished.connect(self.on_llm_done)
        worker.signals.error.connect(self.on_llm_error)
        self.threadpool.start(worker)

    def on_llm_done(self, reply: str):
        self.status_label.setText("Idle")
        self.send_button.setEnabled(True)
        self.listen_button.setEnabled(True)
        self.messages.append(ChatMessage(role="assistant", content=reply))
        self.append_message("Simon", reply)
        if self.auto_speak_check.isChecked():
            self.speak(reply)

    def on_llm_error(self, message: str):
        self.status_label.setText(message)
        self.send_button.setEnabled(True)
        self.listen_button.setEnabled(True)

    def append_message(self, label: str, content: str):
        safe = escape_html(content)
        self.chat_view.append(f"<b>{label}:</b> {safe}")

    def speak(self, text: str):
        engine = self.tts_combo.currentData()
        voice = self.tts_voice_edit.text().strip()
        piper_endpoint = self.piper_endpoint_edit.text().strip()

        def run_speech():
            if engine == "system":
                cmd = ["say"]
                if voice:
                    cmd.extend(["-v", voice])
                cmd.append(text)
                subprocess.run(cmd)
                return
            if engine == "piper":
                if not piper_endpoint:
                    return
                resp = requests.post(piper_endpoint, json={"text": text, "voice": voice} if voice else {"text": text}, timeout=120)
                if resp.status_code != 200:
                    return
                with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
                    f.write(resp.content)
                    temp_path = f.name
                subprocess.run(["afplay", temp_path])

        threading.Thread(target=run_speech, daemon=True).start()


def main():
    app = QtWidgets.QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
