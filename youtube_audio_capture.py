"""
YouTube / ブラウザ再生の音声をループバックで取り込み（Windows WASAPI Loopback 推奨）、
RMS・簡易オンセットでリズムを推定しノーツ生成用イベントをキューへ渡す。
"""

from __future__ import annotations

import json
import queue
import random
import threading
import time
from collections import deque
from pathlib import Path

import numpy as np

# youtubeotoge と合わせる（ノーツが判定線に来るまでの秒）
NOTE_LEAD_SECONDS = 1.85
MIN_ONSET_GAP = 0.28
SAMPLE_RATE = 44100
BLOCK_SIZE = 1024

JSON_SNAPSHOT = "live_audio_reactive.json"


def _find_loopback_device_index() -> int | None:
    try:
        import sounddevice as sd

        devices = sd.query_devices()
    except Exception:
        return None
    for i, d in enumerate(devices):
        name = str(d.get("name", "")).lower()
        if d.get("max_input_channels", 0) <= 0:
            continue
        if "loopback" in name or "wasapi" in name and "loop" in name:
            return i
    for i, d in enumerate(devices):
        if "ステレオ ミックス" in str(d.get("name", "")) or "stereo mix" in str(
            d.get("name", "")
        ).lower():
            return i
    return None


def _default_input_device_index(sd) -> int:
    """sounddevice 0.5 系では sd.default.device が _InputOutputPair（.input / .output）になる。"""
    d = sd.default.device
    inp = getattr(d, "input", None)
    if inp is not None:
        return int(inp)
    if isinstance(d, (tuple, list)):
        return int(d[0])
    if isinstance(d, (int, float)):
        return int(d)
    try:
        return int(d[0])  # __getitem__ 対応オブジェクト
    except (TypeError, IndexError, ValueError):
        pass
    di = sd.query_devices(kind="input")
    if isinstance(di, dict):
        idx = di.get("index")
        if idx is not None:
            return int(idx)
    raise TypeError(f"既定の入力デバイス番号を取得できません: {type(d).__name__}")


class LiveAudioAnalyzer:
    """バックグラウンドで入力ストリームを解析し、ノーツ候補を queue に積む。"""

    def __init__(self, root: Path, note_lead_sec: float = NOTE_LEAD_SECONDS) -> None:
        self._root = root
        self._note_lead = note_lead_sec
        self._note_queue: queue.Queue[dict] = queue.Queue()
        self._lock = threading.Lock()
        self._rms = 0.0
        self._rms_smooth = 0.0
        self._last_onset_strength = 0.0
        self._estimated_bpm = 0.0
        self._onset_times: deque[float] = deque(maxlen=48)
        self._events: list[dict] = []
        self._max_events_json = 400
        self._error = ""
        self._device_name = ""
        self._running = False
        self._stream = None
        self._t0_perf = 0.0
        self._last_onset_perf = 0.0
        self._env_follower = 0.0
        self._json_path = root / JSON_SNAPSHOT

    @property
    def note_queue(self) -> queue.Queue[dict]:
        return self._note_queue

    def session_now(self) -> float:
        return time.perf_counter() - self._t0_perf

    def get_hud(self) -> dict[str, float | str]:
        with self._lock:
            return {
                "rms": self._rms,
                "rms_smooth": self._rms_smooth,
                "estimated_bpm": self._estimated_bpm,
                "last_onset": self._last_onset_strength,
                "device": self._device_name or "(none)",
                "error": self._error,
            }

    def _flush_json(self) -> None:
        with self._lock:
            payload = {
                "note_lead_seconds": self._note_lead,
                "metrics": {
                    "rms": round(self._rms, 5),
                    "rms_smooth": round(self._rms_smooth, 5),
                    "estimated_bpm": round(self._estimated_bpm, 1),
                    "last_onset_strength": round(self._last_onset_strength, 5),
                },
                "device": self._device_name,
                "note_events": list(self._events[-self._max_events_json :]),
            }
        try:
            self._json_path.write_text(
                json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
        except OSError:
            pass

    def _audio_callback(self, indata, frames, t_info, status) -> None:  # type: ignore[no-untyped-def]
        if status:
            pass
        now = time.perf_counter() - self._t0_perf
        mono = indata[:, 0].astype(np.float64) if indata.ndim > 1 else indata.ravel()
        if mono.size == 0:
            return
        rms = float(np.sqrt(np.mean(mono * mono)))
        with self._lock:
            self._rms = rms
            self._rms_smooth = 0.92 * self._rms_smooth + 0.08 * rms
            env = self._env_follower
            self._env_follower = 0.995 * env + 0.005 * rms
            novelty = max(0.0, rms - max(env * 1.35, 1e-6))
            self._last_onset_strength = novelty

        if novelty > 0.004 + self._rms_smooth * 0.5 and (now - self._last_onset_perf) >= MIN_ONSET_GAP:
            self._last_onset_perf = now
            hit_time = now + self._note_lead
            lane = random.randint(0, 3)
            ev = {
                "hit_time": round(hit_time, 4),
                "lane": lane,
                "spawned_at": round(now, 4),
                "rms": round(rms, 5),
                "onset": round(novelty, 5),
            }
            with self._lock:
                self._onset_times.append(now)
                if len(self._onset_times) >= 4:
                    span = self._onset_times[-1] - self._onset_times[0]
                    if span > 0.5:
                        raw = (len(self._onset_times) - 1) / span * 60.0
                        self._estimated_bpm = float(min(220, max(40, raw)))
                self._events.append(ev)
            self._note_queue.put(ev)

    def start(self, clock_zero_perf: float) -> bool:
        self._t0_perf = clock_zero_perf
        try:
            import sounddevice as sd
        except ImportError:
            self._error = "sounddevice 未インストール (pip install sounddevice)"
            return False

        idx = _find_loopback_device_index()
        if idx is None:
            idx = _default_input_device_index(sd)
            self._error = "ループバック未検出→既定入力。YouTubeを取るには「ステレオ ミックス」等が必要な場合あり"
        else:
            self._error = ""

        try:
            dev_info = sd.query_devices(idx, "input")
            self._device_name = str(dev_info.get("name", ""))
        except Exception:
            self._device_name = str(idx)

        self._last_onset_perf = -1.0
        self._running = True
        try:
            self._stream = sd.InputStream(
                device=idx,
                channels=1,
                samplerate=SAMPLE_RATE,
                blocksize=BLOCK_SIZE,
                callback=self._audio_callback,
            )
            self._stream.start()
        except Exception as e:
            self._running = False
            self._error = f"オーディオ開始失敗: {e}"
            return False
        return True

    def stop(self) -> None:
        self._running = False
        if self._stream is not None:
            try:
                self._stream.stop()
                self._stream.close()
            except Exception:
                pass
            self._stream = None
        self._flush_json()

    def maybe_flush_json(self, now_wall: float) -> None:
        if not hasattr(self, "_last_json_wall"):
            self._last_json_wall = now_wall
        if now_wall - self._last_json_wall >= 1.0:
            self._last_json_wall = now_wall
            self._flush_json()
