"""
音源を解析し、推定BPMとオンセット強度に基づいて chart.json を書き出す。

- 既定（bpm_volume）: 推定BPMから拍間隔を求め、min_gap〜max_gap 秒に収まるグリッド幅
  （拍の整数倍）で曲を区切り、各区間に必ず 1 ノーツ。位置は区間内でオンセットが最大の時刻。
  拍位置に合わせて最初の区間の開始を beat_track に近づける。
- onset: 従来どおりオンセットのみ（本数は検出次第で 1 個のみになることもある）。
- interval: min_gap〜max_gap の乱数間隔、または --note-step 固定間隔。
- レーンは乱数（--seed）
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import librosa
import numpy as np

DEFAULT_AUDIO = "music.mp3"
DEFAULT_OUT = "chart.json"
MIN_NOTE_GAP = 0.5
MAX_NOTE_GAP = 3.0
FIRST_NOTE_T = 0.5


def estimate_bpm(y: np.ndarray, sr: int, hop_length: int) -> float:
    onset_env = librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop_length)
    try:
        tempo_arr = librosa.feature.rhythm.tempo(
            onset_envelope=onset_env, sr=sr, hop_length=hop_length
        )
    except AttributeError:
        # librosa 0.9 系など（feature.rhythm が無い）
        tempo_arr = librosa.beat.tempo(
            onset_envelope=onset_env, sr=sr, hop_length=hop_length
        )
    return float(np.asarray(tempo_arr).ravel()[0])


def onset_times_from_volume(y: np.ndarray, sr: int, hop_length: int) -> np.ndarray:
    """オンセット強度（スペクトル変化＝リズムの立ち上がり）のピーク時刻（秒）。"""
    oenv = librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop_length)
    onset_frames = librosa.onset.onset_detect(
        onset_envelope=oenv,
        sr=sr,
        hop_length=hop_length,
        backtrack=True,
        units="frames",
    )
    return librosa.frames_to_time(onset_frames, sr=sr, hop_length=hop_length)


def thin_min_gap(times: np.ndarray, min_gap: float, first_from: float) -> list[float]:
    times = np.sort(np.unique(np.asarray(times, dtype=float)))
    out: list[float] = []
    for t in times.tolist():
        if t < first_from:
            continue
        if not out or t - out[-1] >= min_gap:
            out.append(t)
    return out


def grid_from_bpm(bpm: float, min_gap: float, max_gap: float) -> float:
    """BPM から 1 スロットの秒幅。拍の整数倍にしつつ [min_gap, max_gap] に収める。"""
    bpm = float(np.clip(bpm, 30.0, 320.0))
    beat = 60.0 / bpm
    if beat >= max_gap:
        return float(max_gap)
    m = max(1, int(np.ceil(min_gap / beat)))
    g = m * beat
    while g > max_gap + 1e-9 and m > 1:
        m -= 1
        g = m * beat
    if g > max_gap:
        return float(max_gap)
    return float(min(max_gap, max(min_gap, g)))


def beat_track_times(
    y: np.ndarray, sr: int, hop_length: int, oenv: np.ndarray
) -> np.ndarray:
    _, bframes = librosa.beat.beat_track(
        onset_envelope=oenv, y=y, sr=sr, hop_length=hop_length
    )
    if bframes.size == 0:
        return np.array([], dtype=float)
    return librosa.frames_to_time(bframes, sr=sr, hop_length=hop_length)


def onset_value_at(oenv: np.ndarray, sr: int, hop_length: int, t: float) -> float:
    fi = int(librosa.time_to_frames(t, sr=sr, hop_length=hop_length))
    fi = max(0, min(fi, len(oenv) - 1))
    return float(oenv[fi])


def resolve_slot_collisions(
    times: list[float],
    oenv: np.ndarray,
    sr: int,
    hop_length: int,
    min_gap: float,
) -> list[float]:
    """隣接スロットで時刻が詰まったとき、オンセットが強い方を残す。"""
    times = sorted(times)
    out: list[float] = []
    for t in times:
        if not out:
            out.append(t)
            continue
        if t - out[-1] < min_gap:
            if onset_value_at(oenv, sr, hop_length, t) >= onset_value_at(
                oenv, sr, hop_length, out[-1]
            ):
                out[-1] = t
        else:
            out.append(t)
    return out


def align_start_to_downbeat(first_from: float, beat_times: np.ndarray) -> float:
    if beat_times.size == 0:
        return first_from
    i = int(np.searchsorted(beat_times, first_from - 1e-4))
    if i >= len(beat_times):
        return first_from
    return float(beat_times[i])


def bpm_volume_slot_times(
    duration: float,
    first_from: float,
    grid: float,
    oenv: np.ndarray,
    sr: int,
    hop_length: int,
    beat_times: np.ndarray,
    min_gap: float,
) -> list[float]:
    """
    各区間 [t, t+grid) にオンセット最大の 1 点を必ず入れる。
    開始は first_from 以降の最初のビート付近に合わせる。
    """
    if grid <= 0:
        return []
    ws = align_start_to_downbeat(first_from, beat_times)
    if ws + 1e-6 < first_from:
        ws = first_from
    out: list[float] = []
    while ws < duration - 0.02:
        we = min(ws + grid, duration)
        hi = we - 1e-7 if we < duration else we
        if hi <= ws + 1e-6:
            break
        ins = best_onset_in_range(oenv, sr, hop_length, ws, hi)
        if ins is None:
            ins = float(0.5 * (ws + we))
        ins = max(ws, min(hi, ins))
        out.append(ins)
        ws += grid
    return resolve_slot_collisions(out, oenv, sr, hop_length, min_gap)


def best_onset_in_range(
    oenv: np.ndarray,
    sr: int,
    hop_length: int,
    t0: float,
    t1: float,
) -> float | None:
    if t1 <= t0:
        return None
    f0 = int(librosa.time_to_frames(t0, sr=sr, hop_length=hop_length))
    f1 = int(librosa.time_to_frames(t1, sr=sr, hop_length=hop_length))
    f0 = max(0, min(f0, len(oenv) - 1))
    f1 = max(f0 + 1, min(f1, len(oenv)))
    sub = oenv[f0:f1]
    if sub.size == 0:
        return None
    peak = f0 + int(np.argmax(sub))
    return float(librosa.frames_to_time(peak, sr=sr, hop_length=hop_length))


def extend_to_duration(
    times: list[float],
    duration: float,
    max_gap: float,
    min_gap: float,
    oenv: np.ndarray,
    sr: int,
    hop_length: int,
    end_margin: float = 0.35,
) -> list[float]:
    """曲の終端付近までノーツが途切れないよう、max_gap ルールで末尾を埋める。"""
    limit = max(0.0, duration - end_margin)
    if not times:
        return times
    out = list(times)
    guard = 0
    while limit - out[-1] > max_gap and guard < 50000:
        guard += 1
        lo = out[-1] + min_gap
        hi = min(limit, out[-1] + max_gap)
        if hi > lo:
            ins = best_onset_in_range(oenv, sr, hop_length, lo, hi)
            if ins is None:
                ins = float(0.5 * (lo + hi))
            ins = max(lo, min(hi, ins))
        else:
            ins = min(lo, limit)
        if ins <= out[-1] + 1e-6:
            break
        out.append(ins)
    if limit - out[-1] >= min_gap:
        lo = out[-1] + min_gap
        if limit >= lo:
            ins = best_onset_in_range(oenv, sr, hop_length, lo, limit)
            if ins is None:
                ins = float(0.5 * (lo + limit))
            ins = max(lo, min(limit, ins))
            if ins > out[-1] + min_gap - 1e-9:
                out.append(ins)
    return out


def enforce_max_gap(
    merged: list[float],
    max_gap: float,
    min_gap: float,
    oenv: np.ndarray,
    sr: int,
    hop_length: int,
) -> list[float]:
    """隣接ノーツの間隔が max_gap を超えないよう、区間内の最強オンセットで補間。"""
    final: list[float] = []
    for t in merged:
        if not final:
            final.append(t)
            continue
        while t - final[-1] > max_gap:
            lo = final[-1] + min_gap
            hi = min(t - min_gap, final[-1] + max_gap)
            if hi > lo:
                ins = best_onset_in_range(oenv, sr, hop_length, lo, hi)
                if ins is None:
                    ins = float(0.5 * (lo + hi))
                ins = max(lo, min(hi, ins))
            else:
                ins = lo
            final.append(ins)
        if t - final[-1] >= min_gap:
            final.append(t)
    return final


def fixed_step_times(duration: float, step: float, first_t: float) -> list[float]:
    """first_t から step 秒おきに曲終端手前までノーツ時刻を並べる。"""
    if step <= 0:
        return []
    out: list[float] = []
    t = first_t
    while t < duration - 0.05:
        out.append(t)
        t += step
    return out


def fallback_uniform_times(
    duration: float,
    min_gap: float,
    max_gap: float,
    seed: int,
    first_t: float = FIRST_NOTE_T,
) -> list[float]:
    """オンセットが取れないとき: min_gap〜max_gap 秒のランダム間隔で配置。"""
    rng = np.random.default_rng(seed)
    out: list[float] = []
    t = first_t
    while t < duration - 0.05:
        out.append(t)
        t += float(rng.uniform(min_gap, max_gap))
    return out


def build_chart(
    audio_path: Path,
    min_gap: float,
    max_gap: float,
    seed: int,
    trim_silence: bool = True,
    trim_top_db: float = 40.0,
    placement: str = "bpm_volume",
    note_step: float | None = None,
) -> tuple[list[dict], float, float, float]:
    """
    戻り値: chart, bpm, duration, audible_start_sec
    audible_start_sec … 先頭無音トリム後の「聴こえ始め」が元ファイルの何秒か（ログ用）
    """
    y, sr = librosa.load(str(audio_path), mono=True, sr=None)
    hop_length = 512
    duration = float(len(y) / sr)

    bpm = estimate_bpm(y, sr, hop_length)
    oenv = librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop_length)

    audible_start = 0.0
    if trim_silence and y.size:
        # trim は (トリム後の y, np.array([開始サンプル, 終了サンプル])) を返す
        _y_trim, idx = librosa.effects.trim(
            y, top_db=trim_top_db, frame_length=2048, hop_length=hop_length
        )
        idx = np.asarray(idx).ravel()
        start_i = int(idx[0])
        end_i = int(idx[1])
        audible_start = float(start_i / sr)
        y_onset = y[start_i:end_i]
    else:
        y_onset = y

    if y_onset.size == 0:
        y_onset = y
        audible_start = 0.0

    first_from = max(FIRST_NOTE_T, audible_start + 0.15)

    if placement == "interval":
        if note_step is not None:
            times = fixed_step_times(duration, note_step, first_from)
        else:
            times = fallback_uniform_times(duration, min_gap, max_gap, seed, first_t=first_from)
        times = extend_to_duration(times, duration, max_gap, min_gap, oenv, sr, hop_length)
    elif placement == "bpm_volume":
        grid = grid_from_bpm(bpm, min_gap, max_gap)
        bt = beat_track_times(y, sr, hop_length, oenv)
        times = bpm_volume_slot_times(
            duration, first_from, grid, oenv, sr, hop_length, bt, min_gap
        )
        times = extend_to_duration(times, duration, max_gap, min_gap, oenv, sr, hop_length)
    else:
        raw_onsets = onset_times_from_volume(y_onset, sr, hop_length) + audible_start
        merged = thin_min_gap(raw_onsets, min_gap, first_from)

        if not merged:
            merged = fallback_uniform_times(duration, min_gap, max_gap, seed, first_t=first_from)

        times = enforce_max_gap(merged, max_gap, min_gap, oenv, sr, hop_length)
        times = extend_to_duration(times, duration, max_gap, min_gap, oenv, sr, hop_length)

    rng = np.random.default_rng(seed + 7919)
    lanes = rng.integers(0, 4, size=len(times))
    chart = [{"time": round(float(tt), 4), "lane": int(ln)} for tt, ln in zip(times, lanes)]
    return chart, bpm, duration, audible_start


def main() -> None:
    root = Path(__file__).resolve().parent
    p = argparse.ArgumentParser(description="音源から chart.json を生成（BPM推定・オンセット）")
    p.add_argument(
        "--audio",
        type=Path,
        default=root / DEFAULT_AUDIO,
        help=f"入力音源（デフォルト: {DEFAULT_AUDIO}）",
    )
    p.add_argument(
        "--out",
        type=Path,
        default=root / DEFAULT_OUT,
        help=f"出力 JSON（デフォルト: {DEFAULT_OUT}）",
    )
    p.add_argument("--min-gap", type=float, default=MIN_NOTE_GAP, help="ノーツ最小間隔（秒）")
    p.add_argument("--max-gap", type=float, default=MAX_NOTE_GAP, help="ノーツ最大間隔（秒）この幅を超えると補間")
    p.add_argument("--seed", type=int, default=42, help="レーン乱数のシード")
    p.add_argument(
        "--no-trim",
        action="store_true",
        help="先頭・末尾の無音トリムをしない（オンセットはファイル先頭から）",
    )
    p.add_argument(
        "--trim-top-db",
        type=float,
        default=40.0,
        help="無音トリムのしきい値（大きいほど残す）",
    )
    p.add_argument(
        "--placement",
        choices=("bpm_volume", "onset", "interval"),
        default="bpm_volume",
        help="bpm_volume=BPMグリッドで各区間に1個＋区間内で最強オンセット（既定） "
        "onset=オンセットのみ interval=min〜max乱数または --note-step",
    )
    p.add_argument(
        "--note-step",
        type=float,
        default=None,
        metavar="SEC",
        help="--placement interval と併用。指定すると乱数ではなく固定 SEC 秒おきに並べる（例: 約200個→ 211/200≈1.055）",
    )
    args = p.parse_args()

    audio_path: Path = args.audio
    if not audio_path.is_file():
        print(f"音源が見つかりません: {audio_path}", file=sys.stderr)
        sys.exit(1)

    if args.note_step is not None and args.placement != "interval":
        print("--note-step は --placement interval との併用のみ有効です。", file=sys.stderr)
        sys.exit(2)

    chart, bpm, duration, audible_start = build_chart(
        audio_path,
        args.min_gap,
        args.max_gap,
        args.seed,
        trim_silence=not args.no_trim,
        trim_top_db=args.trim_top_db,
        placement=args.placement,
        note_step=args.note_step,
    )
    out_path: Path = args.out
    out_path.write_text(json.dumps(chart, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    last_t = chart[-1]["time"] if chart else 0.0
    exp_mean = 0.5 * (args.min_gap + args.max_gap)
    approx_expected = duration / exp_mean if exp_mean > 0 else 0.0
    print(f"推定BPM: {bpm:.2f}")
    print(f"配置モード: {args.placement}")
    if args.placement == "bpm_volume":
        g = grid_from_bpm(bpm, args.min_gap, args.max_gap)
        print(f"BPMスロット幅（秒）: {g:.3f}  （min_gap={args.min_gap} max_gap={args.max_gap} に収めた拍の倍数）")
    print(f"曲の長さ: {duration:.1f}s  先頭無音（参考・トリム後の開始）: {audible_start:.2f}s")
    print(f"ノーツ数: {len(chart)}  最終ノーツ時刻: {last_t:.2f}s")
    if args.placement == "interval":
        if args.note_step:
            ff = max(FIRST_NOTE_T, audible_start + 0.15)
            pred = max(0, int((duration - ff) / args.note_step))
            print(f"（参考）固定間隔 {args.note_step:.3f}s のおおよその本数: 約 {pred}（先頭 {ff:.2f}s から）")
        else:
            print(
                f"（参考）interval 乱数モードの期待本数目安: 約 {approx_expected:.0f} "
                f"(= 曲長 / 平均間隔 {exp_mean:.2f}s)。約200本にしたいなら --note-step を使う"
            )
    print(f"書き出し: {out_path}")


if __name__ == "__main__":
    main()
