"""
ブラウザ（YouTube）の再生音のみをループバック等で取り込む（ローカル MP3 再生は行わない）。

・オンセット（RMS 高いほど間隔を詰める）
・4 帯域スペクトルが閾値超え → 該当レーンにノーツ（色は帯域色）
・RMS > 0.02 付近で追加パルス（サビで密度アップ）
各ブロックは rFFT 帯域（frequency_lanes と同じ Hz 境界）。
"""

from __future__ import annotations

import json
import queue
import threading
import time
from collections import deque
from pathlib import Path

import numpy as np

from frequency_lanes import (
    NOTE_LABELS,
    NOTE_RGB,
    live_block_band_powers,
    live_block_lane_style,
)

# youtubeotoge と合わせる（ノーツが判定線に来るまでの秒）
NOTE_LEAD_SECONDS = 1.85
MIN_ONSET_GAP = 0.28
# オンセット判定 novelty > onset_base + rms_smooth * rms_mul（起動画面で変更可）
DEFAULT_ONSET_BASE = 0.004
DEFAULT_RMS_MUL = 0.5
DEFAULT_MIN_VOLUME = 0.0001
SAMPLE_RATE = 44100
BLOCK_SIZE = 1024

# RMS が高いほどオンセット間隔を詰める（サビで密度アップ）
RMS_LOUD_REF = 0.02
# 帯域正規化がこの値を超えたらそのレーンにノーツ（スペクトラムが高い）
BAND_SPAWN_NORM = 0.36
# レーンごとの最小スポーン間隔（秒）
MIN_LANE_SPAWN_QUIET = 0.24
MIN_LANE_SPAWN_LOUD = 0.11
RMS_FOR_LANE_FAST = 0.038
# RMS が十分高いときの追加パルス（サビ用）
RMS_EXTRA_THRESHOLD = 0.02
# 1 オーディオブロックあたりのスポーン上限（暴発防止）
MAX_NOTES_PER_BLOCK_QUIET = 2
MAX_NOTES_PER_BLOCK_LOUD = 5

JSON_SNAPSHOT = "live_audio_reactive.json"
# 入力デバイスを番号で固定する場合（ループバック名が取れない環境向け）。例: audio_settings.example.json
AUDIO_SETTINGS_FILE = "audio_settings.json"


def load_audio_input_device_index(root: Path) -> int | None:
    """audio_settings.json の input_device_index。無効・未設定は None。"""
    p = root / AUDIO_SETTINGS_FILE
    if not p.is_file():
        return None
    try:
        raw = json.loads(p.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, TypeError):
        return None
    if not isinstance(raw, dict):
        return None
    v = raw.get("input_device_index")
    if v is None:
        return None
    try:
        idx = int(v)
    except (TypeError, ValueError):
        return None
    if idx < 0:
        return None
    return idx


def _input_device_usable(sd, idx: int) -> bool:
    try:
        info = sd.query_devices(idx)
    except Exception:
        return False
    return int(info.get("max_input_channels", 0) or 0) >= 1


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
        if "loopback" in name or ("wasapi" in name and "loop" in name):
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

    def __init__(
        self,
        root: Path,
        note_lead_sec: float = NOTE_LEAD_SECONDS,
        rms_mul: float = DEFAULT_RMS_MUL,
        min_volume: float = DEFAULT_MIN_VOLUME,
    ) -> None:
        self._root = root
        self._note_lead = note_lead_sec
        self._onset_base = float(DEFAULT_ONSET_BASE)
        self._rms_mul = float(rms_mul)
        self._min_volume = float(min_volume)
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
        self._band_power_max = np.ones(4, dtype=np.float64) * 1e-9
        self._spectrum_norm: list[float] = [0.0, 0.0, 0.0, 0.0]
        self._last_lane_spawn: list[float] = [-1e9, -1e9, -1e9, -1e9]
        self._last_rms_extra_spawn = -1e9

    @property
    def note_queue(self) -> queue.Queue[dict]:
        return self._note_queue

    def session_now(self) -> float:
        return time.perf_counter() - self._t0_perf

    def get_hud(self) -> dict[str, float | str | list[float]]:
        with self._lock:
            return {
                "rms": self._rms,
                "rms_smooth": self._rms_smooth,
                "estimated_bpm": self._estimated_bpm,
                "last_onset": self._last_onset_strength,
                "spectrum": list(self._spectrum_norm),
                "device": self._device_name or "(none)",
                "error": self._error,
            }

    def set_reactive_tune(self, rms_mul: float, min_volume: float) -> None:
        """Sensitivity / Min Volume を実行中に更新（コールバックが参照）。"""
        with self._lock:
            self._rms_mul = float(rms_mul)
            self._min_volume = float(min_volume)

    def _flush_json(self) -> None:
        with self._lock:
            payload = {
                "note_lead_seconds": self._note_lead,
                "metrics": {
                    "rms": round(self._rms, 5),
                    "rms_smooth": round(self._rms_smooth, 5),
                    "estimated_bpm": round(self._estimated_bpm, 1),
                    "last_onset_strength": round(self._last_onset_strength, 5),
                    "spectrum_norm": [round(x, 4) for x in self._spectrum_norm],
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

    def _enqueue_note(
        self,
        now: float,
        rms: float,
        novelty: float,
        lane: int,
        rgb: tuple[int, int, int],
        label: str,
        src: str,
    ) -> None:
        hit_time = now + self._note_lead
        r_, g_, b_ = rgb
        ev = {
            "hit_time": round(hit_time, 4),
            "lane": int(lane),
            "spawned_at": round(now, 4),
            "rms": round(rms, 5),
            "onset": round(novelty, 5),
            "r": int(r_),
            "g": int(g_),
            "b": int(b_),
            "label": label,
            "src": src,
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

    def _audio_callback(self, indata, frames, t_info, status) -> None:  # type: ignore[no-untyped-def]
        if status:
            pass
        now = time.perf_counter() - self._t0_perf
        mono = indata[:, 0].astype(np.float64) if indata.ndim > 1 else indata.ravel()
        if mono.size == 0:
            return
        rms = float(np.sqrt(np.mean(mono * mono)))
        raw_b = np.array(live_block_band_powers(mono, float(SAMPLE_RATE)), dtype=np.float64)
        self._band_power_max = 0.992 * self._band_power_max + 0.008 * raw_b
        norm = np.clip(raw_b / np.maximum(self._band_power_max, 1e-18), 0.0, 1.0)
        with self._lock:
            self._rms = rms
            self._rms_smooth = 0.92 * self._rms_smooth + 0.08 * rms
            env = self._env_follower
            self._env_follower = 0.995 * env + 0.005 * rms
            novelty = max(0.0, rms - max(env * 1.35, 1e-6))
            self._last_onset_strength = novelty
            sn = np.array(self._spectrum_norm, dtype=np.float64)
            sn = 0.65 * sn + 0.35 * norm
            self._spectrum_norm = [float(x) for x in sn]
            rms_smooth_now = self._rms_smooth
            ob = self._onset_base
            rm = self._rms_mul
            mv = self._min_volume

        if rms < mv:
            return

        rms_boost = max(0.0, min(1.0, (rms - 0.012) / 0.07))
        eff_onset_gap = MIN_ONSET_GAP * (1.0 - 0.52 * rms_boost)
        thr_band = BAND_SPAWN_NORM * (0.88 if rms > RMS_LOUD_REF else 1.0)
        lane_min_gap = MIN_LANE_SPAWN_LOUD if rms > RMS_FOR_LANE_FAST else MIN_LANE_SPAWN_QUIET
        max_block = MAX_NOTES_PER_BLOCK_LOUD if rms > RMS_LOUD_REF else MAX_NOTES_PER_BLOCK_QUIET

        spawns = 0
        if (
            novelty > ob + rms_smooth_now * rm
            and (now - self._last_onset_perf) >= eff_onset_gap
            and spawns < max_block
        ):
            self._last_onset_perf = now
            lane, rgb, lab = live_block_lane_style(mono, float(SAMPLE_RATE))
            self._enqueue_note(now, rms, novelty, lane, rgb, lab, "onset")
            spawns += 1

        order = list(np.argsort(-norm))
        for b in order:
            if spawns >= max_block:
                break
            nb = int(b)
            if float(norm[nb]) < thr_band:
                continue
            if now - self._last_lane_spawn[nb] < lane_min_gap:
                continue
            self._last_lane_spawn[nb] = now
            rgb = NOTE_RGB[nb]
            self._enqueue_note(now, rms, novelty, nb, rgb, NOTE_LABELS[nb], "band")
            spawns += 1

        if (
            rms > RMS_EXTRA_THRESHOLD
            and spawns < max_block
        ):
            extra_gap = max(0.11, 0.46 - (rms - RMS_EXTRA_THRESHOLD) * 7.5)
            if now - self._last_rms_extra_spawn >= extra_gap:
                self._last_rms_extra_spawn = now
                lane, rgb, lab = live_block_lane_style(mono, float(SAMPLE_RATE))
                self._enqueue_note(now, rms, novelty, lane, rgb, lab, "rms")
                spawns += 1

    def start(self, clock_zero_perf: float) -> bool:
        self._t0_perf = clock_zero_perf
        try:
            import sounddevice as sd
        except ImportError:
            self._error = "sounddevice 未インストール (pip install sounddevice)"
            return False

        forced = load_audio_input_device_index(self._root)
        if forced is not None:
            if not _input_device_usable(sd, forced):
                n_dev = len(sd.query_devices())
                self._error = (
                    f"audio_settings.json の input_device_index={forced} は使えません "
                    f"(0〜{n_dev - 1} の入力ありデバイスを list_audio_devices.py で確認)"
                )
                return False
            idx = forced
            self._error = ""
        else:
            idx = _find_loopback_device_index()
            if idx is None:
                idx = _default_input_device_index(sd)
                self._error = (
                    "ループバック未検出→既定入力。"
                    "Stereo Mix / ループバックを使うには audio_settings.json で input_device_index を指定"
                )
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
