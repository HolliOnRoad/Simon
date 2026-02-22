import sys
import json
import threading
import subprocess
import tempfile
import shutil
import zipfile
import time
from pathlib import Path
import os
from dataclasses import dataclass
from datetime import datetime, timezone
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


APP_VERSION = "1.0.1"
DEFAULT_UPDATE_URL = "https://HolliOnRoad.github.io/Simon/updates/simon.json"

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


class HistoryStore:
    def __init__(self, app_name: str = "Simon"):
        base_dir = QtCore.QStandardPaths.writableLocation(QtCore.QStandardPaths.AppDataLocation)
        if base_dir:
            base = Path(base_dir)
        else:
            base = Path.home() / f".{app_name.lower()}"
        base.mkdir(parents=True, exist_ok=True)
        self.path = base / "history.jsonl"
        self._lock = threading.Lock()

    def append(self, role: str, content: str) -> None:
        item = {
            "ts": datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z"),
            "role": role,
            "content": content,
        }
        line = json.dumps(item, ensure_ascii=False)
        with self._lock:
            with self.path.open("a", encoding="utf-8") as f:
                f.write(line + "\n")

    def load_recent(self, limit: int = 80) -> List[dict]:
        if not self.path.exists():
            return []
        with self._lock:
            try:
                lines = self.path.read_text(encoding="utf-8").splitlines()
            except Exception:
                return []
        items: List[dict] = []
        for line in lines[-limit:]:
            try:
                items.append(json.loads(line))
            except Exception:
                continue
        return items

    def search(self, query: str, limit: int = 50) -> List[dict]:
        if not self.path.exists():
            return []
        q = query.strip().lower()
        if not q:
            return []
        with self._lock:
            try:
                lines = self.path.read_text(encoding="utf-8").splitlines()
            except Exception:
                return []
        results: List[dict] = []
        for line in reversed(lines):
            try:
                item = json.loads(line)
            except Exception:
                continue
            content = str(item.get("content", ""))
            if q in content.lower():
                results.append(item)
                if len(results) >= limit:
                    break
        return results


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
    spectrumChanged = QtCore.Signal(list)
    error = QtCore.Signal(str)

    def __init__(self, samplerate: int = 16000):
        super().__init__()
        self.samplerate = samplerate
        self._stream: Optional[sd.InputStream] = None
        self.is_running = False
        self.device: Optional[int] = None
        self._spectrum_enabled = False

    def set_spectrum_enabled(self, enabled: bool) -> None:
        self._spectrum_enabled = enabled

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
        if self._spectrum_enabled:
            try:
                samples = indata[:, 0].copy()
                if samples.size < 32:
                    return
                window = np.hanning(samples.size)
                spectrum = np.abs(np.fft.rfft(samples * window))
                if spectrum.size > 1:
                    spectrum = spectrum[1:]
                bands = 20
                chunks = np.array_split(spectrum, bands)
                levels = [float(np.sqrt(np.mean(chunk ** 2))) for chunk in chunks]
                peak = max(levels) if levels else 1.0
                if peak <= 0:
                    peak = 1.0
                levels = [min(1.0, val / peak) for val in levels]
                self.spectrumChanged.emit(levels)
            except Exception:
                pass


class WhisperManager:
    def __init__(self):
        self._model = None
        self._config = None
        self._lock = threading.Lock()

    def transcribe(
        self,
        audio: np.ndarray,
        language: str,
        model_name: str,
        device: str,
        compute_type: str,
        vad_filter: bool = True,
    ) -> tuple[str, Optional[str]]:
        if WhisperModel is None:
            raise RuntimeError("faster-whisper is not installed.")
        if audio.size == 0:
            return "", None

        with self._lock:
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
            transcribe_kwargs: dict = {"language": language, "beam_size": 5, "vad_filter": vad_filter}
            if vad_filter:
                transcribe_kwargs["vad_parameters"] = {"min_silence_duration_ms": 500}
            segments, _ = self._model.transcribe(audio, **transcribe_kwargs)
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


class WakeWordWorker(QtCore.QRunnable):
    def __init__(
        self,
        whisper: WhisperManager,
        device_index: Optional[int],
        samplerate: int,
        wake_word: str,
        language: str,
        model_name: str,
        compute_type: str,
    ):
        super().__init__()
        self.whisper = whisper
        self.device_index = device_index
        self.samplerate = samplerate
        self.wake_word = wake_word.lower().strip()
        self.language = language
        self.model_name = model_name
        self.compute_type = compute_type
        self.signals = WorkerSignals()

    def run(self):
        try:
            frames = int(self.samplerate * 2.0)
            audio = sd.rec(
                frames,
                samplerate=self.samplerate,
                channels=1,
                dtype="float32",
                device=None if self.device_index == -1 else self.device_index,
            )
            sd.wait()
            audio = audio.squeeze()
            audio = resample_audio(audio, self.samplerate, 16000)
            # Skip transcription on silence to avoid hallucinations
            rms = float(np.sqrt(np.mean(audio ** 2)))
            if rms < 0.008:
                self.signals.finished.emit({"detected": False, "text": ""})
                return
            text, _warning = self.whisper.transcribe(
                audio,
                language=self.language,
                model_name=self.model_name,
                device="auto",
                compute_type=self.compute_type,
                vad_filter=False,
            )
            detected = bool(self.wake_word and self.wake_word in text.lower())
            self.signals.finished.emit({"detected": detected, "text": text})
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
                "model": self.model or "mistral:7b-instruct",
                "messages": payload_messages,
                "stream": False,
            }
            resp = requests.post(url, json=payload, timeout=120)
            if resp.status_code != 200:
                try:
                    err = resp.json().get("error", "")
                except Exception:
                    err = ""
                raise RuntimeError(err or f"Ollama HTTP {resp.status_code}")
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


class SpectrumWidget(QtWidgets.QWidget):
    def __init__(self, bars: int = 20):
        super().__init__()
        self._bars = max(8, int(bars))
        self._levels = [0.0 for _ in range(self._bars)]
        self._decay = 0.85
        self.setMinimumHeight(80)
        self.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Fixed)

    def set_levels(self, levels: List[float]) -> None:
        if not levels:
            levels = [0.0 for _ in range(self._bars)]
        if len(levels) < self._bars:
            levels = list(levels) + [0.0] * (self._bars - len(levels))
        if len(levels) > self._bars:
            levels = levels[: self._bars]
        for i, target in enumerate(levels):
            target = max(0.0, min(float(target), 1.0))
            decayed = self._levels[i] * self._decay
            self._levels[i] = max(decayed, target)
        self.update()

    def paintEvent(self, event):
        painter = QtGui.QPainter(self)
        rect = self.rect()
        painter.fillRect(rect, QtGui.QColor(20, 20, 24))
        w = rect.width()
        h = rect.height()
        gap = 3
        bar_w = max(1, int((w - gap * (self._bars + 1)) / self._bars))
        for i, level in enumerate(self._levels):
            bar_h = int(level * (h - 8))
            x = gap + i * (bar_w + gap)
            y = h - bar_h - 4
            color = QtGui.QColor(80, 200, 255)
            if level > 0.7:
                color = QtGui.QColor(255, 140, 80)
            painter.fillRect(QtCore.QRect(x, y, bar_w, bar_h), color)


class HistorySearchDialog(QtWidgets.QDialog):
    def __init__(self, history: HistoryStore, parent=None):
        super().__init__(parent)
        self.history = history
        self.setWindowTitle("Verlauf durchsuchen")
        self.resize(680, 460)

        layout = QtWidgets.QVBoxLayout(self)
        self.search_edit = QtWidgets.QLineEdit()
        self.search_edit.setPlaceholderText("Suchbegriff eingeben...")
        layout.addWidget(self.search_edit)

        self.results_list = QtWidgets.QListWidget()
        layout.addWidget(self.results_list, 1)

        self.preview = QtWidgets.QTextEdit()
        self.preview.setReadOnly(True)
        self.preview.setPlaceholderText("Vorschau...")
        layout.addWidget(self.preview, 1)

        self.search_edit.textChanged.connect(self.on_search_changed)
        self.results_list.currentItemChanged.connect(self.on_item_changed)

    def on_search_changed(self, text: str) -> None:
        query = text.strip()
        self.results_list.clear()
        self.preview.clear()
        if len(query) < 2:
            return
        results = self.history.search(query, limit=80)
        for item in results:
            role = str(item.get("role", ""))
            label = "Du" if role == "user" else "Simon" if role == "assistant" else role
            ts = str(item.get("ts", ""))
            ts_short = ts.replace("T", " ")[:19] if ts else ""
            content = str(item.get("content", ""))
            snippet = content.replace("\n", " ").strip()
            if len(snippet) > 90:
                snippet = snippet[:90] + "..."
            display = f"{ts_short} • {label}: {snippet}"
            lw_item = QtWidgets.QListWidgetItem(display)
            lw_item.setData(QtCore.Qt.UserRole, item)
            self.results_list.addItem(lw_item)

    def on_item_changed(self, current, previous) -> None:
        if current is None:
            self.preview.clear()
            return
        item = current.data(QtCore.Qt.UserRole) or {}
        role = str(item.get("role", ""))
        label = "Du" if role == "user" else "Simon" if role == "assistant" else role
        ts = str(item.get("ts", ""))
        content = str(item.get("content", ""))
        header = f"{label} • {ts}".strip()
        self.preview.setPlainText(f"{header}\n\n{content}")


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
        self.monitor.spectrumChanged.connect(self.on_monitor_spectrum)
        self.monitor.error.connect(self.on_monitor_error)
        self.whisper = WhisperManager()
        self.wake_whisper = WhisperManager()
        self.history = HistoryStore()
        self.messages: List[ChatMessage] = []
        self._monitor_paused = False
        self._monitor_for_visualizer = False
        self._update_check_silent = False
        self._status_before_update: Optional[str] = None
        self._silence_timer = QtCore.QElapsedTimer()
        self._last_voice_ms = 0
        self._waiting_for_voice = False
        self._mic_prompt_shown = False
        self._wake_busy = False
        self._wake_last_trigger = 0.0
        self._monitor_for_wake = False
        self._wake_status_token = 0

        self._build_ui()
        self._build_menu()
        self._load_settings()
        self._load_history()
        QtCore.QTimer.singleShot(1200, self.ensure_mic_access)
        if self.auto_update_check.isChecked() and self.update_url_edit.text().strip():
            QtCore.QTimer.singleShot(1200, self.on_check_updates_silent)

    def _build_menu(self):
        menu = self.menuBar().addMenu("Simon")

        self.action_check_updates = QtGui.QAction("Nach Updates suchen...", self)
        self.action_check_updates.triggered.connect(lambda: self.on_check_updates_clicked(silent=False))
        menu.addAction(self.action_check_updates)

        self.action_auto_updates = QtGui.QAction("Auto-Update-Pruefung", self)
        self.action_auto_updates.setCheckable(True)
        self.action_auto_updates.toggled.connect(self.auto_update_check.setChecked)
        self.auto_update_check.toggled.connect(self.action_auto_updates.setChecked)
        menu.addAction(self.action_auto_updates)

        self.action_history_search = QtGui.QAction("Verlauf durchsuchen...", self)
        self.action_history_search.triggered.connect(self.on_history_search)
        menu.addAction(self.action_history_search)

        menu.addSeparator()

        quit_action = QtGui.QAction("Beenden", self)
        quit_action.triggered.connect(QtWidgets.QApplication.quit)
        menu.addAction(quit_action)

    def on_check_updates_silent(self):
        self.on_check_updates_clicked(silent=True)

    def on_visualizer_toggled(self, checked: bool) -> None:
        self.visualizer_widget.setVisible(checked)
        self.monitor.set_spectrum_enabled(checked)
        if checked and not self.monitor_check.isChecked():
            self._monitor_for_visualizer = True
            self.monitor_check.setChecked(True)
        elif not checked and self._monitor_for_visualizer:
            self._monitor_for_visualizer = False
            self.monitor_check.setChecked(False)

    def on_wake_word_toggled(self, checked: bool) -> None:
        if checked:
            if self.monitor_check.isChecked():
                self._monitor_for_wake = True
                self.monitor_check.setChecked(False)
            self._set_wake_status("Lauscht...")
            self._schedule_wake_listen(300)
        else:
            if self._monitor_for_wake:
                self._monitor_for_wake = False
                self.monitor_check.setChecked(True)
            self._set_wake_status("Inaktiv")

    def _set_wake_status(self, text: str, reset_after_ms: Optional[int] = None) -> None:
        self.wake_status_label.setText(text)
        if reset_after_ms is None:
            return
        self._wake_status_token += 1
        token = self._wake_status_token
        QtCore.QTimer.singleShot(reset_after_ms, lambda: self._reset_wake_status(token))

    def _reset_wake_status(self, token: int) -> None:
        if token != self._wake_status_token:
            return
        if self.wake_word_check.isChecked():
            self.wake_status_label.setText("Lauscht...")
        else:
            self.wake_status_label.setText("Inaktiv")

    def _schedule_wake_listen(self, delay_ms: int = 800) -> None:
        if not self.wake_word_check.isChecked():
            return
        QtCore.QTimer.singleShot(delay_ms, self._run_wake_listen)

    def _run_wake_listen(self) -> None:
        if not self.wake_word_check.isChecked():
            return
        if self._wake_busy or self.recorder.is_running or not self.send_button.isEnabled():
            self._schedule_wake_listen(800)
            return
        device_index = self.input_device_combo.currentData()
        samplerate = 16000
        if device_index is not None and device_index != -1:
            try:
                dev_info = sd.query_devices(int(device_index))
                samplerate = int(dev_info.get("default_samplerate", 16000))
            except Exception:
                samplerate = 16000
        self._wake_busy = True
        language = self.language_combo.currentData() or "de"
        worker = WakeWordWorker(
            self.wake_whisper,
            device_index,
            samplerate,
            "simon",
            language,
            "base",
            "int8",
        )
        worker.signals.finished.connect(self.on_wake_word_done)
        worker.signals.error.connect(self.on_wake_word_error)
        self.threadpool.start(worker)

    def on_wake_word_done(self, result: object) -> None:
        self._wake_busy = False
        if not self.wake_word_check.isChecked():
            return
        detected = False
        if isinstance(result, dict):
            detected = bool(result.get("detected"))
        now = time.monotonic()
        if detected and (now - self._wake_last_trigger) > 2.0:
            self._wake_last_trigger = now
            if not self.recorder.is_running:
                self._set_wake_status("Erkannt", reset_after_ms=2000)
                self.status_label.setText("Wake-Word erkannt")
                self.start_listening()
                self._schedule_wake_listen(1200)
                return
        if isinstance(result, dict):
            snippet = str(result.get("text", "")).strip()
            if snippet:
                show = snippet.replace("\n", " ")
                if len(show) > 40:
                    show = show[:40] + "..."
                self._set_wake_status(f"Hoerte: {show}", reset_after_ms=2000)
        self._schedule_wake_listen(500)

    def on_wake_word_error(self, message: str) -> None:
        self._wake_busy = False
        if not self.wake_word_check.isChecked():
            return
        self._maybe_show_mic_permission_hint(message)
        self._set_wake_status(f"Fehler: {message[:60]}", reset_after_ms=4000)
        self._schedule_wake_listen(1200)

    def on_history_search(self) -> None:
        dialog = HistorySearchDialog(self.history, self)
        dialog.exec()

    def ensure_mic_access(self) -> None:
        if self._mic_prompt_shown:
            return
        device_index = self.input_device_combo.currentData()
        samplerate = 16000
        if device_index is not None and device_index != -1:
            try:
                dev_info = sd.query_devices(int(device_index))
                samplerate = int(dev_info.get("default_samplerate", 16000))
            except Exception:
                samplerate = 16000
        try:
            stream = sd.InputStream(
                samplerate=samplerate,
                channels=1,
                device=None if device_index == -1 else device_index,
                dtype="float32",
            )
            stream.start()
            stream.stop()
            stream.close()
        except Exception as exc:
            self._maybe_show_mic_permission_hint(str(exc))

    def _maybe_show_mic_permission_hint(self, message: str) -> None:
        if self._mic_prompt_shown:
            return
        lower = message.lower()
        if "permission" not in lower and "denied" not in lower and "not authorized" not in lower:
            return
        self._mic_prompt_shown = True
        msg = QtWidgets.QMessageBox(self)
        msg.setWindowTitle("Mikrofon-Zugriff")
        msg.setText(
            "Simon hat keinen Mikrofon-Zugriff. "
            "Bitte aktiviere den Zugriff in den Systemeinstellungen."
        )
        open_button = msg.addButton("Einstellungen oeffnen", QtWidgets.QMessageBox.AcceptRole)
        msg.addButton("OK", QtWidgets.QMessageBox.RejectRole)
        msg.exec()
        if msg.clickedButton() == open_button:
            try:
                subprocess.Popen(
                    [
                        "open",
                        "x-apple.systempreferences:com.apple.preference.security?Privacy_Microphone",
                    ]
                )
            except Exception:
                pass

    def _build_ui(self):
        central = QtWidgets.QWidget()
        self.setCentralWidget(central)

        main_layout = QtWidgets.QVBoxLayout(central)
        main_layout.setSpacing(8)

        settings_box = QtWidgets.QGroupBox("Einstellungen")
        settings_layout = QtWidgets.QGridLayout(settings_box)

        self.language_combo = QtWidgets.QComboBox()
        self.language_combo.addItem("Deutsch (DE)", "de")
        self.language_combo.addItem("Englisch (US)", "en")

        self.provider_combo = QtWidgets.QComboBox()
        self.provider_combo.addItem("Lokal (Ollama)", "ollama")
        self.provider_combo.addItem("API (OpenAI-kompatibel)", "openai")
        self.provider_combo.currentIndexChanged.connect(self.on_provider_changed)

        self.model_combo = QtWidgets.QComboBox()
        self.model_combo.setEditable(True)
        self.model_combo.setInsertPolicy(QtWidgets.QComboBox.NoInsert)
        self.model_combo.setMinimumWidth(160)
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
        self.stt_device_combo.addItem("Auto (GPU falls verfuegbar)", "auto")
        self.stt_device_combo.addItem("CPU", "cpu")

        self.stt_compute_combo = QtWidgets.QComboBox()
        self.stt_compute_combo.addItem("int8 (schnell)", "int8")
        self.stt_compute_combo.addItem("int8_float16 (balanciert)", "int8_float16")
        self.stt_compute_combo.addItem("float16 (genau)", "float16")

        self.stt_preset_combo = QtWidgets.QComboBox()
        self.stt_preset_combo.addItem("Custom", "custom")
        for key, label, _ in STT_PRESETS:
            self.stt_preset_combo.addItem(label, key)

        self.input_device_combo = QtWidgets.QComboBox()
        self.refresh_devices_button = QtWidgets.QPushButton("Geraete suchen")
        self.refresh_devices_button.clicked.connect(lambda: self._check_device_changes(force_rescan=True))
        self.monitor_check = QtWidgets.QCheckBox("Mikrofon-Monitor")
        self.monitor_bar = QtWidgets.QProgressBar()
        self.monitor_bar.setRange(0, 100)
        self.monitor_bar.setValue(0)
        self.monitor_bar.setTextVisible(False)
        self.test_mic_button = QtWidgets.QPushButton("Auto-Test")
        self.test_mic_button.clicked.connect(self.on_test_mic_clicked)

        self.ptt_check = QtWidgets.QCheckBox("Push-to-Talk (Leertaste)")
        self.auto_stop_check = QtWidgets.QCheckBox("Auto-Stopp bei Stille")
        self.silence_duration_spin = QtWidgets.QDoubleSpinBox()
        self.silence_duration_spin.setRange(0.3, 5.0)
        self.silence_duration_spin.setSingleStep(0.1)
        self.silence_duration_spin.setValue(1.2)
        self.silence_duration_spin.setSuffix(" s")
        self.silence_threshold_spin = QtWidgets.QDoubleSpinBox()
        self.silence_threshold_spin.setRange(0.01, 0.2)
        self.silence_threshold_spin.setSingleStep(0.01)
        self.silence_threshold_spin.setValue(0.025)
        self.silence_threshold_spin.setDecimals(2)
        self.wake_word_check = QtWidgets.QCheckBox("Wake-Word (Simon)")
        self.wake_status_label = QtWidgets.QLabel("Inaktiv")
        self.wake_status_label.setStyleSheet("color: #666;")

        self.update_url_edit = QtWidgets.QLineEdit()
        self.update_url_edit.setPlaceholderText(DEFAULT_UPDATE_URL)
        self.auto_update_check = QtWidgets.QCheckBox("Updates automatisch pruefen")
        self.version_label = QtWidgets.QLabel(f"Version {APP_VERSION}")
        self.version_label.setAlignment(QtCore.Qt.AlignRight | QtCore.Qt.AlignVCenter)

        self.tts_combo = QtWidgets.QComboBox()
        self.tts_combo.addItem("Systemstimme", "system")
        self.tts_combo.addItem("Piper (HTTP)", "piper")
        self.tts_voice_edit = QtWidgets.QLineEdit()
        self.piper_endpoint_edit = QtWidgets.QLineEdit()

        self.system_prompt_edit = QtWidgets.QLineEdit()

        self.auto_speak_check = QtWidgets.QCheckBox("Auto-Sprechen")
        self.auto_listen_check = QtWidgets.QCheckBox("Auto-Neustart")
        self.auto_send_check = QtWidgets.QCheckBox("Auto-Senden bei Stopp")

        self.visualizer_check = QtWidgets.QCheckBox("Visualizer aktiv")
        self.visualizer_check.toggled.connect(self.on_visualizer_toggled)
        self.visualizer_widget = SpectrumWidget()
        self.visualizer_widget.setVisible(False)

        self._preset_lock = False
        self.stt_preset_combo.currentIndexChanged.connect(self.on_stt_preset_changed)
        self.stt_model_combo.currentIndexChanged.connect(self.on_stt_settings_changed)
        self.stt_device_combo.currentIndexChanged.connect(self.on_stt_settings_changed)
        self.stt_compute_combo.currentIndexChanged.connect(self.on_stt_settings_changed)
        self.monitor_check.toggled.connect(self.on_monitor_toggled)
        self.input_device_combo.currentIndexChanged.connect(self.on_input_device_changed)
        self.wake_word_check.toggled.connect(self.on_wake_word_toggled)

        row = 0
        settings_layout.addWidget(QtWidgets.QLabel("Sprache"), row, 0)
        settings_layout.addWidget(self.language_combo, row, 1)
        settings_layout.addWidget(QtWidgets.QLabel("LLM"), row, 2)
        settings_layout.addWidget(self.provider_combo, row, 3)
        settings_layout.addWidget(QtWidgets.QLabel("Modell"), row, 4)
        settings_layout.addWidget(self.model_combo, row, 5)
        settings_layout.addWidget(self.auto_speak_check, row, 6)
        settings_layout.addWidget(self.auto_listen_check, row, 7)

        row += 1
        settings_layout.addWidget(QtWidgets.QLabel("STT-Modell"), row, 0)
        settings_layout.addWidget(self.stt_model_combo, row, 1)
        settings_layout.addWidget(QtWidgets.QLabel("Geraet"), row, 2)
        settings_layout.addWidget(self.stt_device_combo, row, 3)
        settings_layout.addWidget(QtWidgets.QLabel("Rechenart"), row, 4)
        settings_layout.addWidget(self.stt_compute_combo, row, 5)
        settings_layout.addWidget(QtWidgets.QLabel("Preset"), row, 6)
        settings_layout.addWidget(self.stt_preset_combo, row, 7)

        row += 1
        settings_layout.addWidget(QtWidgets.QLabel("Eingabegeraet"), row, 0)
        settings_layout.addWidget(self.input_device_combo, row, 1, 1, 5)
        settings_layout.addWidget(self.refresh_devices_button, row, 6, 1, 2)

        row += 1
        settings_layout.addWidget(QtWidgets.QLabel("Mikrofon-Monitor"), row, 0)
        settings_layout.addWidget(self.monitor_check, row, 1)
        settings_layout.addWidget(self.monitor_bar, row, 2, 1, 4)
        settings_layout.addWidget(self.test_mic_button, row, 6, 1, 2)

        row += 1
        settings_layout.addWidget(QtWidgets.QLabel("Aufnahme"), row, 0)
        settings_layout.addWidget(self.ptt_check, row, 1, 1, 2)
        settings_layout.addWidget(self.auto_stop_check, row, 3)
        settings_layout.addWidget(QtWidgets.QLabel("Stille (s)"), row, 4)
        settings_layout.addWidget(self.silence_duration_spin, row, 5)
        settings_layout.addWidget(QtWidgets.QLabel("Schwelle"), row, 6)
        settings_layout.addWidget(self.silence_threshold_spin, row, 7)

        row += 1
        settings_layout.addWidget(QtWidgets.QLabel("Wake-Word"), row, 0)
        settings_layout.addWidget(self.wake_word_check, row, 1, 1, 2)
        settings_layout.addWidget(self.wake_status_label, row, 3, 1, 5)

        row += 1
        settings_layout.addWidget(QtWidgets.QLabel("API Basis-URL"), row, 0)
        settings_layout.addWidget(self.base_url_edit, row, 1, 1, 4)
        settings_layout.addWidget(QtWidgets.QLabel("API Key"), row, 5)
        settings_layout.addWidget(self.api_key_edit, row, 6, 1, 2)

        row += 1
        settings_layout.addWidget(QtWidgets.QLabel("Updates"), row, 0)
        settings_layout.addWidget(self.update_url_edit, row, 1, 1, 5)
        settings_layout.addWidget(self.auto_update_check, row, 6)

        row += 1
        settings_layout.addWidget(QtWidgets.QLabel("TTS"), row, 0)
        settings_layout.addWidget(self.tts_combo, row, 1)
        settings_layout.addWidget(QtWidgets.QLabel("TTS-Stimme"), row, 2)
        settings_layout.addWidget(self.tts_voice_edit, row, 3)
        settings_layout.addWidget(QtWidgets.QLabel("Piper-Endpunkt"), row, 4)
        settings_layout.addWidget(self.piper_endpoint_edit, row, 5, 1, 3)

        row += 1
        settings_layout.addWidget(QtWidgets.QLabel("System-Prompt"), row, 0)
        settings_layout.addWidget(self.system_prompt_edit, row, 1, 1, 7)
        row += 1
        settings_layout.addWidget(self.version_label, row, 6, 1, 2)

        main_layout.addWidget(settings_box)

        visual_box = QtWidgets.QGroupBox("Visualizer")
        visual_layout = QtWidgets.QHBoxLayout(visual_box)
        visual_layout.addWidget(self.visualizer_check)
        visual_layout.addWidget(self.visualizer_widget, 1)
        main_layout.addWidget(visual_box)

        self.chat_view = QtWidgets.QTextEdit()
        self.chat_view.setReadOnly(True)
        self.chat_view.setAcceptRichText(True)
        self.chat_view.setPlaceholderText("Unterhaltung erscheint hier...")
        main_layout.addWidget(self.chat_view, 1)

        level_layout = QtWidgets.QHBoxLayout()
        self.listen_status = QtWidgets.QLabel("Nicht am Zuhoeren")
        self.level_bar = QtWidgets.QProgressBar()
        self.level_bar.setRange(0, 100)
        self.level_bar.setValue(0)
        self.level_bar.setTextVisible(False)
        level_layout.addWidget(self.listen_status)
        level_layout.addWidget(self.level_bar, 1)
        main_layout.addLayout(level_layout)

        input_layout = QtWidgets.QHBoxLayout()
        self.input_edit = QtWidgets.QPlainTextEdit()
        self.input_edit.setPlaceholderText("Nachricht eingeben...")
        self.input_edit.setFixedHeight(70)
        input_layout.addWidget(self.input_edit, 1)

        button_layout = QtWidgets.QVBoxLayout()
        self.listen_button = QtWidgets.QPushButton("Zuhoeren")
        self.listen_button.clicked.connect(self.on_listen_clicked)
        self.send_button = QtWidgets.QPushButton("Senden")
        self.send_button.clicked.connect(self.on_send_clicked)
        button_layout.addWidget(self.listen_button)
        button_layout.addWidget(self.send_button)
        button_layout.addStretch(1)

        input_layout.addLayout(button_layout)
        main_layout.addLayout(input_layout)

        status_layout = QtWidgets.QHBoxLayout()
        self.status_label = QtWidgets.QLabel("Leerlauf")
        status_layout.addWidget(self.status_label)
        status_layout.addStretch(1)
        status_layout.addWidget(self.auto_send_check)
        main_layout.addLayout(status_layout)

        self.populate_input_devices()

        # Auto-refresh audio devices every 5s to detect Bluetooth connections
        self._device_refresh_timer = QtCore.QTimer(self)
        self._device_refresh_timer.timeout.connect(self._auto_refresh_devices)
        self._device_refresh_timer.start(5000)

    def _auto_refresh_devices(self):
        """Periodically refresh device list to detect newly connected Bluetooth devices."""
        if self.recorder.is_running or self.monitor.is_running:
            return
        self._check_device_changes(force_rescan=False)

    def _check_device_changes(self, force_rescan: bool = False):
        try:
            if force_rescan:
                sd._terminate()
                sd._initialize()
            devices = sd.query_devices()
        except Exception as exc:
            print(f"[Devices] refresh error: {exc}", flush=True)
            return
        input_devices = [(idx, dev.get("name", "")) for idx, dev in enumerate(devices)
                         if dev.get("max_input_channels", 0) > 0]
        current_names = set(self.input_device_combo.itemText(i)
                            for i in range(1, self.input_device_combo.count()))
        new_names = {f"{idx}: {name}" for idx, name in input_devices}
        if new_names == current_names:
            return
        print(f"[Devices] changed: +{new_names - current_names} -{current_names - new_names}", flush=True)
        prev_data = self.input_device_combo.currentData()
        self.populate_input_devices()
        # Auto-select newly connected Bluetooth device if on System-Standard
        if prev_data == -1 or prev_data is None:
            for idx, name in input_devices:
                label = f"{idx}: {name}"
                if label not in current_names:
                    bt_keywords = ("airpods", "bluetooth", "headphone", "kopfhörer", "headset")
                    if any(kw in name.lower() for kw in bt_keywords):
                        combo_idx = self.input_device_combo.findData(idx)
                        if combo_idx >= 0:
                            self.input_device_combo.setCurrentIndex(combo_idx)
                            print(f"[Devices] Auto-selected: {name}", flush=True)
                        break

    def _load_settings(self):
        self.language_combo.setCurrentIndex(
            self.language_combo.findData(self.settings.value("language", "en"))
        )
        self.provider_combo.setCurrentIndex(
            self.provider_combo.findData(self.settings.value("provider", "ollama"))
        )
        self.base_url_edit.setText(self.settings.value("api_base", "http://localhost:11434"))
        saved_model = self.settings.value("model", "mistral:7b-instruct")
        if self.provider_combo.currentData() == "ollama":
            self._populate_ollama_models()
        # Select saved model if available, else keep first populated model
        idx = self.model_combo.findText(saved_model)
        if idx >= 0:
            self.model_combo.setCurrentIndex(idx)
        elif self.model_combo.count() == 0:
            self.model_combo.setCurrentText(saved_model)
        self.api_key_edit.setText(self.settings.value("api_key", ""))
        stt_model = self.settings.value("stt_model", "small")
        # Fall back to small if large-v3 is not cached (avoids silent download failures)
        if stt_model == "large-v3":
            from pathlib import Path as _P
            _cache = _P.home() / ".cache" / "huggingface" / "hub"
            _has_large = any(
                (_cache / d / "blobs").exists()
                for d in (_cache.iterdir() if _cache.exists() else [])
                if "large-v3" in str(d)
            )
            if not _has_large:
                stt_model = "small"
        self.settings.setValue("stt_model", stt_model)
        self.stt_model_combo.setCurrentIndex(
            self.stt_model_combo.findData(stt_model)
        )
        self.stt_device_combo.setCurrentIndex(
            self.stt_device_combo.findData(self.settings.value("stt_device", "auto"))
        )
        stt_compute = self.settings.value("stt_compute", "int8")
        if stt_compute in ("float16", "int8_float16"):
            stt_compute = "int8"
        self.settings.setValue("stt_compute", stt_compute)
        self.settings.sync()
        self.stt_compute_combo.setCurrentIndex(
            self.stt_compute_combo.findData(stt_compute)
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
        update_url = self.settings.value("update_url", "")
        if not update_url or "example.com" in str(update_url):
            update_url = DEFAULT_UPDATE_URL
        self.update_url_edit.setText(update_url)
        self.auto_update_check.setChecked(self.settings.value("auto_update", False, bool))
        self.tts_combo.setCurrentIndex(
            self.tts_combo.findData(self.settings.value("tts_engine", "system"))
        )
        self.tts_voice_edit.setText(self.settings.value("tts_voice", ""))
        self.piper_endpoint_edit.setText(self.settings.value("piper_endpoint", "http://localhost:5002/tts"))
        self.system_prompt_edit.setText(
            self.settings.value("system_prompt", "Du bist Simon, ein hilfreicher Sprachassistent. Du kannst Sprache erkennen und verarbeiten. Antworte kurz und klar auf Deutsch.")
        )
        self.auto_speak_check.setChecked(self.settings.value("auto_speak", True, bool))
        self.auto_listen_check.setChecked(self.settings.value("auto_listen", True, bool))
        # Migration: force auto_send and auto_stop to True if not yet migrated
        if not self.settings.value("migrated_defaults_v2", False, bool):
            self.settings.setValue("auto_send", True)
            self.settings.setValue("auto_stop", True)
            self.settings.setValue("migrated_defaults_v2", True)
            self.settings.sync()
        # Migration: update old generic system prompt to voice-aware version
        if not self.settings.value("migrated_system_prompt_v1", False, bool):
            old = self.settings.value("system_prompt", "")
            if old in ("", "Du bist ein hilfreicher KI-Agent. Antworte kurz und klar."):
                self.settings.setValue("system_prompt", "Du bist Simon, ein hilfreicher Sprachassistent. Du kannst Sprache erkennen und verarbeiten. Antworte kurz und klar auf Deutsch.")
            self.settings.setValue("migrated_system_prompt_v1", True)
            self.settings.sync()
        auto_send = self.settings.value("auto_send", True, bool)
        auto_stop = self.settings.value("auto_stop", True, bool)
        print(f"[Settings] auto_send={auto_send} auto_stop={auto_stop}", flush=True)
        self.auto_send_check.setChecked(auto_send)
        self.ptt_check.setChecked(self.settings.value("ptt_enabled", False, bool))
        self.auto_stop_check.setChecked(auto_stop)
        self.silence_duration_spin.setValue(float(self.settings.value("silence_duration", 1.2)))
        saved_threshold = float(self.settings.value("silence_threshold", 0.025))
        # Migrate old default 0.05 → 0.025 (better for quiet mics like AirPods)
        if saved_threshold >= 0.05:
            saved_threshold = 0.025
            self.settings.setValue("silence_threshold", saved_threshold)
            self.settings.sync()
        self.silence_threshold_spin.setValue(saved_threshold)
        self.visualizer_check.setChecked(self.settings.value("visualizer_enabled", False, bool))
        self.wake_word_check.setChecked(self.settings.value("wake_word", False, bool))

    def closeEvent(self, event):
        self._save_settings()
        super().closeEvent(event)

    def _save_settings(self):
        self.settings.setValue("language", self.language_combo.currentData())
        self.settings.setValue("provider", self.provider_combo.currentData())
        self.settings.setValue("model", self.model_combo.currentText().strip())
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
        self.settings.setValue("auto_listen", self.auto_listen_check.isChecked())
        self.settings.setValue("auto_send", self.auto_send_check.isChecked())
        self.settings.setValue("ptt_enabled", self.ptt_check.isChecked())
        self.settings.setValue("auto_stop", self.auto_stop_check.isChecked())
        self.settings.setValue("silence_duration", self.silence_duration_spin.value())
        self.settings.setValue("silence_threshold", self.silence_threshold_spin.value())
        self.settings.setValue("visualizer_enabled", self.visualizer_check.isChecked())
        self.settings.setValue("wake_word", self.wake_word_check.isChecked())

    def _load_history(self):
        items = self.history.load_recent(limit=40)
        if not items:
            return
        for item in items:
            role = str(item.get("role", ""))
            content = str(item.get("content", "")).strip()
            if not content:
                continue
            label = "Du" if role == "user" else "Simon" if role == "assistant" else role
            self.append_message(label, content)
            if role in ("user", "assistant"):
                self.messages.append(ChatMessage(role=role, content=content))

    def _text_input_has_focus(self) -> bool:
        widget = self.focusWidget()
        return isinstance(widget, (QtWidgets.QLineEdit, QtWidgets.QPlainTextEdit, QtWidgets.QTextEdit))

    def on_provider_changed(self):
        provider = self.provider_combo.currentData()
        if provider == "ollama":
            if not self.base_url_edit.text().strip():
                self.base_url_edit.setText("http://localhost:11434")
            self._populate_ollama_models()
        else:
            if self.base_url_edit.text().strip() == "http://localhost:11434":
                self.base_url_edit.setText("")

    def _populate_ollama_models(self):
        base = self.base_url_edit.text().strip() or "http://localhost:11434"
        current = self.model_combo.currentText().strip()
        try:
            resp = requests.get(base.rstrip("/") + "/api/tags", timeout=3)
            if resp.status_code == 200:
                models = [m["name"] for m in resp.json().get("models", [])]
                self.model_combo.blockSignals(True)
                self.model_combo.clear()
                for m in models:
                    self.model_combo.addItem(m)
                self.model_combo.blockSignals(False)
                # restore saved value or pick first
                idx = self.model_combo.findText(current)
                if idx >= 0:
                    self.model_combo.setCurrentIndex(idx)
                elif models:
                    self.model_combo.setCurrentIndex(0)
        except Exception:
            pass  # Ollama not reachable, keep existing text

    def populate_input_devices(self):
        current = self.input_device_combo.currentData()
        self.input_device_combo.clear()
        self.input_device_combo.addItem("System-Standard", -1)
        try:
            devices = sd.query_devices()
        except Exception as exc:
            self.status_label.setText(f"Audio-Geraete Fehler: {exc}")
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
            if self.visualizer_check.isChecked():
                self.visualizer_check.setChecked(False)
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
        self.monitor.set_spectrum_enabled(self.visualizer_check.isChecked())
        self.monitor.start(device=None if device_index == -1 else device_index, samplerate=samplerate)

    def stop_monitor(self):
        self.monitor.stop()
        self.monitor_bar.setValue(0)
        self.visualizer_widget.set_levels([])

    def restart_monitor(self):
        self.stop_monitor()
        self.start_monitor()

    def on_monitor_level(self, level: float):
        self.monitor_bar.setValue(int(level * 100))

    def on_monitor_spectrum(self, levels: List[float]):
        if self.visualizer_check.isChecked():
            self.visualizer_widget.set_levels(levels)

    def on_monitor_error(self, message: str):
        self.status_label.setText(f"Monitor-Fehler: {message}")

    def keyPressEvent(self, event):
        if (
            self.ptt_check.isChecked()
            and event.key() == QtCore.Qt.Key_Space
            and not event.isAutoRepeat()
            and not self._text_input_has_focus()
        ):
            if not self.recorder.is_running:
                self.start_listening()
            event.accept()
            return
        super().keyPressEvent(event)

    def keyReleaseEvent(self, event):
        if (
            self.ptt_check.isChecked()
            and event.key() == QtCore.Qt.Key_Space
            and not event.isAutoRepeat()
            and not self._text_input_has_focus()
        ):
            if self.recorder.is_running:
                self.stop_listening()
            event.accept()
            return
        super().keyReleaseEvent(event)

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
        self.status_label.setText(f"Mikrofon-Test ok (Pegel {level_val:.2f})")
        self.test_mic_button.setEnabled(True)
        if self.monitor_check.isChecked() and self._monitor_paused:
            self.start_monitor()
        self._monitor_paused = False

    def on_test_mic_error(self, message: str):
        self.status_label.setText(f"Mikrofon-Test Fehler: {message}")
        self.test_mic_button.setEnabled(True)
        self._maybe_show_mic_permission_hint(message)
        if self.monitor_check.isChecked() and self._monitor_paused:
            self.start_monitor()
        self._monitor_paused = False

    def set_update_controls_enabled(self, enabled: bool):
        if hasattr(self, "action_check_updates"):
            self.action_check_updates.setEnabled(enabled)

    def _restore_status_after_update_check(self):
        if self._status_before_update is None:
            return
        if self.status_label.text() == "Suche nach Updates...":
            self.status_label.setText(self._status_before_update)
        self._status_before_update = None

    def on_check_updates_clicked(self, silent: bool = False):
        self._update_check_silent = silent
        url = self.update_url_edit.text().strip()
        if not url:
            url = DEFAULT_UPDATE_URL
            self.update_url_edit.setText(url)
        if not url:
            if self._update_check_silent:
                self._restore_status_after_update_check()
            else:
                self.status_label.setText("Update-URL fehlt")
                QtWidgets.QMessageBox.warning(self, "Updates", "Update-URL fehlt.")
            return
        if self._update_check_silent:
            self._status_before_update = None
        else:
            self._status_before_update = self.status_label.text()
            self.status_label.setText("Suche nach Updates...")
        self.set_update_controls_enabled(False)
        worker = UpdateCheckWorker(url, APP_VERSION)
        worker.signals.finished.connect(self.on_update_check_done)
        worker.signals.error.connect(self.on_update_check_error)
        self.threadpool.start(worker)

    def on_update_check_done(self, result: object):
        self.set_update_controls_enabled(True)
        if not isinstance(result, dict):
            if self._update_check_silent:
                self._restore_status_after_update_check()
            else:
                self.status_label.setText("Update-Pruefung fehlgeschlagen")
                QtWidgets.QMessageBox.warning(self, "Updates", "Update-Pruefung fehlgeschlagen.")
            self._status_before_update = None
            self._update_check_silent = False
            return
        status = result.get("status")
        if status == "up_to_date":
            self._restore_status_after_update_check()
            self._update_check_silent = False
            return
        if status == "update":
            latest = result.get("version", "?")
            url = result.get("url", "")
            self._status_before_update = None
            msg = QtWidgets.QMessageBox(self)
            msg.setWindowTitle("Update verfuegbar")
            msg.setText(f"Version {latest} ist verfuegbar. Jetzt herunterladen und installieren?")
            msg.setStandardButtons(QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No)
            if msg.exec() == QtWidgets.QMessageBox.Yes:
                self.status_label.setText("Update wird heruntergeladen...")
                self.set_update_controls_enabled(False)
                worker = UpdateDownloadWorker(url, "Simon")
                worker.signals.finished.connect(self.on_update_download_done)
                worker.signals.error.connect(self.on_update_download_error)
                self.threadpool.start(worker)
            else:
                self.status_label.setText("Update verschoben")
            self._update_check_silent = False
            return
        self.status_label.setText("Unbekannter Update-Status")
        self._status_before_update = None
        self._update_check_silent = False

    def on_update_check_error(self, message: str):
        self.set_update_controls_enabled(True)
        if self._update_check_silent:
            self._restore_status_after_update_check()
        else:
            self.status_label.setText(f"Update-Pruefung Fehler: {message}")
            QtWidgets.QMessageBox.warning(self, "Updates", f"Update-Pruefung Fehler: {message}")
        self._status_before_update = None
        self._update_check_silent = False

    def on_update_download_done(self, result: object):
        self.set_update_controls_enabled(True)
        if not isinstance(result, dict):
            self.status_label.setText("Update-Download fehlgeschlagen")
            return
        zip_path = result.get("zip")
        if not zip_path:
            self.status_label.setText("Update-Download fehlgeschlagen")
            return
        try:
            app_path = get_app_bundle_path()
            target_dir = get_update_target_dir(app_path)
            staged = stage_update_from_zip(Path(zip_path), target_dir, "Simon")
            old_app = app_path or (target_dir / "Simon.app")
            self.status_label.setText("Update wird installiert...")
            launch_update_helper(old_app, staged, os.getpid())
            QtWidgets.QMessageBox.information(self, "Update", "Simon startet neu, um das Update abzuschliessen.")
            QtCore.QCoreApplication.quit()
        except Exception as exc:
            self.status_label.setText(f"Update-Installation Fehler: {exc}")

    def on_update_download_error(self, message: str):
        self.set_update_controls_enabled(True)
        self.status_label.setText(f"Update-Download Fehler: {message}")
        QtWidgets.QMessageBox.warning(self, "Updates", f"Update-Download Fehler: {message}")

    def on_level(self, level: float):
        self.level_bar.setValue(int(level * 100))
        if self.recorder.is_running and self.auto_stop_check.isChecked():
            if not self._silence_timer.isValid():
                self._silence_timer.start()
                self._last_voice_ms = 0
            threshold = float(self.silence_threshold_spin.value())
            elapsed = self._silence_timer.elapsed()
            if level >= threshold:
                # Voice detected
                self._waiting_for_voice = False
                self._last_voice_ms = elapsed
            else:
                # Silence
                if self._waiting_for_voice:
                    # Still waiting for first voice — only stop after 10s of total silence
                    if elapsed >= 10000:
                        if self.recorder.is_running:
                            self.stop_listening()
                    return
                if self._last_voice_ms == 0:
                    self._last_voice_ms = elapsed
                silence_ms = elapsed - self._last_voice_ms
                if silence_ms >= int(self.silence_duration_spin.value() * 1000):
                    if self.recorder.is_running:
                        self.stop_listening()

    def on_listen_clicked(self):
        if not self.recorder.is_running:
            self._waiting_for_voice = False  # manual start: no patience mode
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
            self.listen_status.setText("Zuhoeren")
            self.listen_button.setText("Stopp")
            self.status_label.setText("Hoere zu...")
            if self.auto_stop_check.isChecked():
                self._silence_timer.start()
                self._last_voice_ms = 0
        except Exception as exc:
            message = str(exc)
            lower = message.lower()
            if "permission" in lower or "denied" in lower or "not authorized" in lower:
                message += " (Tipp: Systemeinstellungen → Datenschutz & Sicherheit → Mikrofon)"
                self._maybe_show_mic_permission_hint(message)
            self.status_label.setText(f"Audio-Fehler: {message}")

    def stop_listening(self):
        audio, sample_rate = self.recorder.stop()
        self.listen_status.setText("Nicht am Zuhoeren")
        self.listen_button.setText("Zuhoeren")
        self.level_bar.setValue(0)
        self.status_label.setText("Transkribiere...")
        self._silence_timer.invalidate()
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
            self.status_label.setText("Leerlauf")

        print(f"[Transcription] text={text!r} auto_send={self.auto_send_check.isChecked()}", flush=True)
        if text:
            self.input_edit.setPlainText(text)
            if self.auto_send_check.isChecked():
                self.on_send_clicked()
        elif not warning:
            self.status_label.setText("Keine Sprache erkannt")

    def on_worker_error(self, message: str):
        self.status_label.setText(message)

    def on_send_clicked(self):
        text = self.input_edit.toPlainText().strip()
        print(f"[Send] text={text!r} send_btn_enabled={self.send_button.isEnabled()}", flush=True)
        if not text:
            return
        self.input_edit.clear()

        self.append_message("Du", text)
        print(f"[Send] appended to chat, model={self.model_combo.currentText()}", flush=True)
        self.history.append("user", text)
        self.messages.append(ChatMessage(role="user", content=text))

        system_prompt = self.system_prompt_edit.text().strip()
        full_messages: List[ChatMessage] = []
        if system_prompt:
            full_messages.append(ChatMessage(role="system", content=system_prompt))
        full_messages.extend(self.messages)

        self.status_label.setText("Denke...")
        self.send_button.setEnabled(False)
        self.listen_button.setEnabled(False)

        worker = LLMWorker(
            provider=self.provider_combo.currentData(),
            model=self.model_combo.currentText().strip(),
            api_base=self.base_url_edit.text().strip(),
            api_key=self.api_key_edit.text().strip(),
            messages=full_messages,
        )
        worker.signals.finished.connect(self.on_llm_done)
        worker.signals.error.connect(self.on_llm_error)
        self.threadpool.start(worker)

    def on_llm_done(self, reply: str):
        print(f"[LLM Done] reply={reply[:60]!r} auto_listen={self.auto_listen_check.isChecked()} wake={self.wake_word_check.isChecked()}", flush=True)
        self.status_label.setText("Leerlauf")
        self.send_button.setEnabled(True)
        self.listen_button.setEnabled(True)
        self.messages.append(ChatMessage(role="assistant", content=reply))
        self.append_message("Simon", reply)
        self.history.append("assistant", reply)
        if self.auto_speak_check.isChecked():
            self.speak(reply)
        if self.auto_listen_check.isChecked():
            self._waiting_for_voice = True
            print(f"[AutoRestart] scheduling start_listening in 500ms", flush=True)
            QtCore.QTimer.singleShot(500, self.start_listening)

    def on_llm_error(self, message: str):
        print(f"[LLM Error] {message}", flush=True)
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
    # Single-instance lock: prevent multiple simultaneous Simon processes
    import fcntl
    lock_path = Path(tempfile.gettempdir()) / "simon_instance.lock"
    try:
        lock_fd = open(lock_path, "w")
        fcntl.flock(lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        sys.exit(0)

    app = QtWidgets.QApplication(sys.argv)
    window = MainWindow()
    window.show()
    exit_code = app.exec()
    try:
        fcntl.flock(lock_fd, fcntl.LOCK_UN)
        lock_fd.close()
        lock_path.unlink(missing_ok=True)
    except Exception:
        pass
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
