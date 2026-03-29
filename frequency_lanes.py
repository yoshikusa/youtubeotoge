"""
低音→レーン0、中低音→1、中高音→2、高音→3。表示色: キック青・スネア系黄・ハイハット赤。

ライブ: youtube_audio_capture がループバック（YouTube）の各ブロックを rFFT し、
帯域エネルギー最大でレーン・rgb・ラベルを決める。Hz 境界は BAND_EDGES_HZ。
"""

from __future__ import annotations

import numpy as np

# (lo_hz, hi_hz) 半開区間 [lo, hi)
BAND_EDGES_HZ: tuple[tuple[float, float], ...] = (
    (0.0, 250.0),
    (250.0, 1500.0),
    (1500.0, 5000.0),
    (5000.0, 1.0e9),
)

# レーンごとの塗り色（キック青 / スネア黄系 / ハイハット赤）
NOTE_RGB: tuple[tuple[int, int, int], ...] = (
    (70, 130, 240),
    (250, 220, 80),
    (255, 200, 100),
    (240, 70, 90),
)

NOTE_LABELS: tuple[str, ...] = ("キック", "スネア", "スネア", "ハイハット")


def band_energies_for_freqs(power_per_bin: np.ndarray, freqs: np.ndarray) -> list[float]:
    """各ビンにパワー（振幅^2 相当）があるとき、帯域ごとの合計エネルギー。"""
    p = np.asarray(power_per_bin, dtype=np.float64)
    f = np.asarray(freqs, dtype=np.float64)
    out: list[float] = []
    for lo, hi in BAND_EDGES_HZ:
        m = (f >= lo) & (f < hi)
        out.append(float(np.sum(p[m])))
    return out


def lane_and_style_from_energies(energies: list[float]) -> tuple[int, tuple[int, int, int], str]:
    e = np.asarray(energies, dtype=np.float64)
    if e.size != len(BAND_EDGES_HZ):
        raise ValueError("energies length must match band count")
    if float(np.sum(e)) < 1e-20:
        i = 1
    else:
        i = int(np.argmax(e))
    i = max(0, min(len(NOTE_RGB) - 1, i))
    return i, NOTE_RGB[i], NOTE_LABELS[i]


def _rfft_band_powers(mono: np.ndarray, sr: float) -> list[float]:
    """ブロックの 4 帯域パワー（振幅^2 の帯域和）。短すぎる場合はゼロに近い値。"""
    x = np.asarray(mono, dtype=np.float64).ravel()
    if x.size < 32:
        return [1e-12, 1e-12, 1e-12, 1e-12]
    w = np.hanning(x.size)
    spec = np.abs(np.fft.rfft(x * w))
    power = spec * spec
    freqs = np.fft.rfftfreq(x.size, 1.0 / float(sr))
    nyq = 0.5 * float(sr) - 1e-6
    f = np.clip(freqs, 0.0, nyq)
    return band_energies_for_freqs(power, f)


def live_block_band_powers(mono: np.ndarray, sr: float) -> list[float]:
    """ライブ用: レーン0〜3 に対応する 4 帯域のエネルギー（生値）。"""
    return list(_rfft_band_powers(mono, sr))


def live_block_lane_style(mono: np.ndarray, sr: float) -> tuple[int, tuple[int, int, int], str]:
    """短い波形ブロックから rFFT で帯域エネルギーを取り、レーン・色・ラベルを返す。"""
    es = _rfft_band_powers(mono, sr)
    return lane_and_style_from_energies(es)
