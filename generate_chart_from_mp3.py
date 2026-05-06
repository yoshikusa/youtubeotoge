from __future__ import annotations

import argparse
import concurrent.futures
import json
import random
import re
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import librosa
import numpy as np


SONG_DB_FILE = "song_db.json"
CHARTS_DIR = "charts"
YOUTUBE_URL_TXT = "youtube_url.txt"
MP3_DIR = "mp3"
MISMATCH_WARN_SEC = 2.0
OFFSET_ESTIMATE_TIMEOUT_SEC = 10.0
NUMBERED_MP3_PATTERN = r"^\s*(0[1-9]|[1-9][0-9])\s*[-_ ]"

LANES = 4
STAIR_PATTERN = (0, 1, 2, 3, 2, 1)
DEFAULT_HOLD_GAP_SEC = 0.5
DEFAULT_MERGE_SEC = 0.05
LOW_BAND_BINS = 20


def extract_youtube_id(url: str) -> str:
    u = (url or "").strip()
    if not u:
        return ""
    try:
        parsed = urlparse(u)
    except ValueError:
        return ""
    host = parsed.netloc.lower()
    if host.endswith("youtu.be"):
        return parsed.path.strip("/").split("/")[0]
    if "youtube.com" in host or host.endswith("youtube-nocookie.com"):
        q = parse_qs(parsed.query)
        vid = q.get("v")
        if vid and vid[0]:
            return vid[0]
        segs = [s for s in parsed.path.split("/") if s]
        if len(segs) >= 2 and segs[0] in {"embed", "shorts", "v"}:
            return segs[1]
    return ""


def _stft_low_energy(y: np.ndarray, sr: float, hop_length: int) -> np.ndarray:
    S = np.abs(librosa.stft(y, hop_length=hop_length))
    return np.mean(S[:LOW_BAND_BINS], axis=0)


def _frame_at_time(t: float, sr: float, hop_length: int, max_frame: int) -> int:
    return int(np.clip(round(float(t) * sr / hop_length), 0, max_frame))


def _dedupe_times(
    items: list[tuple[float, str]],
    merge_sec: float,
) -> list[tuple[float, str]]:
    """時刻でソートし、近接時刻を1つにまとめる（beat を onset より優先）。"""
    if not items:
        return []
    rank = {"beat": 0, "onset": 1}
    items.sort(key=lambda x: x[0])
    out: list[tuple[float, str]] = []
    cur_t, cur_src = items[0]
    for t, src in items[1:]:
        if t - cur_t <= merge_sec:
            if rank.get(src, 9) < rank.get(cur_src, 9):
                cur_src = src
            cur_t = (cur_t + t) * 0.5
        else:
            out.append((cur_t, cur_src))
            cur_t, cur_src = t, src
    out.append((cur_t, cur_src))
    return out


def _apply_hold_gap(
    taps: list[tuple[float, int]],
    hold_gap_sec: float,
) -> list[dict[str, float | int | str]]:
    """間隔が hold_gap_sec 超なら先頭を hold にし、終端側の tap は折り畳む。"""
    if not taps:
        return []
    out: list[dict[str, float | int | str]] = []
    i = 0
    while i < len(taps):
        t0, lane0 = taps[i]
        if i + 1 < len(taps) and taps[i + 1][0] - t0 > hold_gap_sec:
            t1, _ = taps[i + 1]
            out.append(
                {
                    "time": round(float(t0), 4),
                    "lane": int(lane0),
                    "type": "hold",
                    "end": round(float(t1), 4),
                }
            )
            i += 2
        else:
            out.append(
                {
                    "time": round(float(t0), 4),
                    "lane": int(lane0),
                    "type": "tap",
                }
            )
            i += 1
    return out


def generate_notes(
    mp3_path: Path,
    chart_mode: str = "hybrid",
    seed: int = 42,
    hold_gap_sec: float = DEFAULT_HOLD_GAP_SEC,
    merge_sec: float = DEFAULT_MERGE_SEC,
) -> tuple[list[dict[str, float | int | str]], float, float, np.ndarray, float]:
    """
    音声 → ビート / オンセット → 強度で間引き → キック帯域・階段・レーン流れ → Hold。
    chart_mode: hybrid（ビート＋強オンセット） / beat（ビートのみ） / onset（強オンセットのみ）
    """
    rng = random.Random(seed)
    y, sr = librosa.load(str(mp3_path), sr=None, mono=True)
    if y.size == 0:
        return ([], 0.0, 0.0, y, float(sr))

    duration = float(librosa.get_duration(y=y, sr=sr))
    hop = 512
    onset_env = librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop)

    tempo_v, beat_frames = librosa.beat.beat_track(
        y=y, sr=sr, onset_envelope=onset_env, hop_length=hop
    )
    tempo_arr = np.asarray(tempo_v).flatten()
    tempo = float(tempo_arr[0]) if tempo_arr.size and float(tempo_arr[0]) > 0 else 120.0

    beat_times = librosa.frames_to_time(beat_frames, sr=sr, hop_length=hop)

    onset_frames = librosa.onset.onset_detect(
        onset_envelope=onset_env, sr=sr, hop_length=hop, units="frames"
    )
    onset_times = librosa.frames_to_time(onset_frames, sr=sr, hop_length=hop)

    threshold = float(np.mean(onset_env))

    low_energy = _stft_low_energy(y, float(sr), hop)
    low_mean = float(np.mean(low_energy))
    max_low = low_energy.shape[0] - 1

    candidates: list[tuple[float, str]] = []
    if chart_mode in ("beat", "hybrid"):
        for t in beat_times:
            candidates.append((float(t), "beat"))
    if chart_mode in ("onset", "hybrid"):
        for fr, t in zip(onset_frames, onset_times):
            fi = int(np.clip(int(fr), 0, len(onset_env) - 1))
            if float(onset_env[fi]) > threshold:
                candidates.append((float(t), "onset"))

    merged = _dedupe_times(candidates, merge_sec)
    taps: list[tuple[float, int]] = []
    prev_lane = rng.randint(0, LANES - 1)

    for j, (t, _) in enumerate(merged):
        fi = _frame_at_time(t, float(sr), hop, max_low)
        kick = float(low_energy[fi]) > low_mean
        if kick:
            lane = 0
        else:
            base = STAIR_PATTERN[j % len(STAIR_PATTERN)]
            drift = rng.choice([-1, 0, 1])
            flowed = max(0, min(LANES - 1, prev_lane + drift))
            lane = max(0, min(LANES - 1, (flowed + base + 1) // 2))
        prev_lane = lane
        taps.append((float(t), int(lane)))

    notes = _apply_hold_gap(taps, hold_gap_sec)
    return (notes, tempo, duration, y, float(sr))


def load_song_db(path: Path) -> dict:
    if not path.is_file():
        return {"songs": []}
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, TypeError):
        return {"songs": []}
    if not isinstance(raw, dict):
        return {"songs": []}
    songs = raw.get("songs")
    if not isinstance(songs, list):
        raw["songs"] = []
    return raw


def upsert_song(db_path: Path, entry: dict) -> tuple[float, float]:
    db = load_song_db(db_path)
    songs = db["songs"]
    youtube_duration = 0.0
    existing_offset = 0.0
    for i, row in enumerate(songs):
        if isinstance(row, dict) and str(row.get("youtube_id", "")) == entry["youtube_id"]:
            try:
                youtube_duration = float(row.get("duration_sec", 0.0) or 0.0)
            except (TypeError, ValueError):
                youtube_duration = 0.0
            try:
                existing_offset = float(row.get("offset", 0.0) or 0.0)
            except (TypeError, ValueError):
                existing_offset = 0.0
            songs[i] = entry
            break
    else:
        songs.append(entry)

    db_path.write_text(
        json.dumps(db, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return (youtube_duration, existing_offset)


def _http_urls_from_text(raw: str) -> list[str]:
    out: list[str] = []
    for line in raw.splitlines():
        s = line.strip()
        if not s or s.startswith("#"):
            continue
        if s[0].isdigit():
            s = re.sub(r"^\d{1,2}[\.\):\s|、]+\s*", "", s).strip()
        if s.startswith("http://") or s.startswith("https://"):
            out.append(s.split()[0])
    return out


def load_youtube_url_candidates(root: Path) -> list[str]:
    p = root / YOUTUBE_URL_TXT
    if not p.is_file():
        return []
    try:
        return _http_urls_from_text(p.read_text(encoding="utf-8", errors="ignore"))
    except OSError:
        return []


def numbered_mp3_files(root: Path) -> list[tuple[int, Path]]:
    mp3_dir = root / MP3_DIR
    if not mp3_dir.is_dir():
        return []
    out: list[tuple[int, Path]] = []
    for p in sorted(mp3_dir.glob("*.mp3")):
        m = re.match(NUMBERED_MP3_PATTERN, p.stem)
        if not m:
            continue
        out.append((int(m.group(1)), p))
    return out


def list_auto_pairs(root: Path) -> list[tuple[int, Path, str]]:
    """番号 n と youtube_url.txt の n 行目が対応する (n, mp3_path, url) を昇順で全部返す。"""
    urls = load_youtube_url_candidates(root)
    if not urls:
        raise FileNotFoundError(f"{YOUTUBE_URL_TXT} にURLがありません。")
    mp3s = numbered_mp3_files(root)
    if not mp3s:
        raise FileNotFoundError(f"{MP3_DIR} に番号付きMP3がありません。")
    matched = [(n, p, urls[n - 1]) for n, p in mp3s if 1 <= n <= len(urls)]
    if not matched:
        raise ValueError("番号で一致する URL と MP3 の組み合わせが見つかりません。")
    matched.sort(key=lambda x: x[0])
    return matched


def pick_auto_pair(root: Path, slot: int | None) -> tuple[Path, str]:
    urls = load_youtube_url_candidates(root)
    if not urls:
        raise FileNotFoundError(f"{YOUTUBE_URL_TXT} にURLがありません。")
    mp3s = numbered_mp3_files(root)
    if not mp3s:
        raise FileNotFoundError(f"{MP3_DIR} に番号付きMP3がありません。")

    if slot is not None:
        if slot <= 0 or slot > len(urls):
            raise ValueError(f"--slot は 1〜{len(urls)} を指定してください。")
        slot_mp3 = next((p for n, p in mp3s if n == slot), None)
        if slot_mp3 is None:
            raise FileNotFoundError(f"{MP3_DIR} に {slot:02d} 番のMP3がありません。")
        return (slot_mp3, urls[slot - 1])

    # 引数なし時:
    # 1) 未連携（youtube_id が song_db に無い）を優先
    # 2) なければ最小番号を採用
    matched = list_auto_pairs(root)
    db = load_song_db(root / SONG_DB_FILE)
    songs = db.get("songs", []) if isinstance(db, dict) else []
    linked_ids = {
        str(r.get("youtube_id", ""))
        for r in songs
        if isinstance(r, dict) and str(r.get("youtube_id", "")).strip()
    }
    unlinked = [row for row in matched if extract_youtube_id(row[2]) not in linked_ids]
    n, p, u = (unlinked[0] if unlinked else matched[0])
    print(f"[AUTO] slot={n} url_id={extract_youtube_id(u)} mp3={p.name}")
    return (p, u)


def maybe_confirm_duration_mismatch(mp3_duration: float, youtube_duration: float) -> None:
    if youtube_duration <= 0.0:
        return
    diff = abs(mp3_duration - youtube_duration)
    if diff <= MISMATCH_WARN_SEC:
        return
    ans = input(
        "曲の長さが違いますが、jsonによりnoteを作成しますか？ "
        f"(mp3={mp3_duration:.3f}s youtube={youtube_duration:.3f}s) [y/N]: "
    ).strip().lower()
    if ans not in {"y", "yes"}:
        raise RuntimeError("ユーザーが duration mismatch をキャンセルしました。")


def prompt_offset(default_offset: float) -> float:
    raw = input(
        f"offset を入力してください（Enterで {default_offset:.3f}）: "
    ).strip()
    if not raw:
        return float(default_offset)
    try:
        return float(raw)
    except ValueError as e:
        raise ValueError("offset は数値で入力してください。") from e


def _estimate_offset_core(
    y: np.ndarray,
    sr: float,
    bpm: float,
    youtube_duration: float,
    mp3_duration: float,
) -> float:
    """長さ差がある場合に BPM 単位で offset を推定する。"""
    # 追加の拍追跡（重い場合があるため timeout 対象）
    tempo_track, _ = librosa.beat.beat_track(y=y, sr=sr, units="time")
    if isinstance(tempo_track, np.ndarray):
        tempo_track = float(tempo_track.item()) if tempo_track.size else 0.0
    bpm_ref = float(tempo_track) if float(tempo_track) > 0.0 else float(bpm)

    # 先頭無音（小さいRMS区間）を検出して、開始ずれ候補に使う
    rms = librosa.feature.rms(y=y, frame_length=2048, hop_length=512)[0]
    times = librosa.frames_to_time(np.arange(len(rms)), sr=sr, hop_length=512)
    thr = max(1e-6, float(np.percentile(rms, 25)) * 0.25)
    lead_idx = 0
    for i, v in enumerate(rms):
        if float(v) > thr:
            lead_idx = i
            break
    leading_silence = float(times[min(lead_idx, len(times) - 1)]) if len(times) else 0.0

    # 曲長差を拍の整数倍に寄せて offset 候補化
    beat_sec = 60.0 / max(1e-6, bpm_ref)
    dur_delta = float(mp3_duration - youtube_duration)
    beat_aligned_delta = round(dur_delta / beat_sec) * beat_sec

    # 開始無音と拍整列差分を混ぜて推定（過大シフト防止）
    est = -(0.7 * leading_silence + 0.3 * beat_aligned_delta)
    return float(max(-8.0, min(8.0, est)))


def estimate_offset_with_timeout(
    y: np.ndarray,
    sr: float,
    bpm: float,
    youtube_duration: float,
    mp3_duration: float,
    timeout_sec: float = OFFSET_ESTIMATE_TIMEOUT_SEC,
) -> float | None:
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as ex:
        fut = ex.submit(
            _estimate_offset_core,
            y,
            sr,
            bpm,
            youtube_duration,
            mp3_duration,
        )
        try:
            return float(fut.result(timeout=timeout_sec))
        except concurrent.futures.TimeoutError:
            return None


def generate_chart_for_pair(
    root: Path,
    mp3_path: Path,
    youtube_url: str,
    args: argparse.Namespace,
    *,
    batch: bool = False,
) -> bool:
    """
    1 組の MP3 + YouTube URL について譜面 JSON と song_db を更新する。
    offset 推定がタイムアウトしたとき False（中断）、それ以外 True。
    """
    if not mp3_path.is_file():
        raise FileNotFoundError(f"MP3 not found: {mp3_path}")

    youtube_id = extract_youtube_id(youtube_url)
    if not youtube_id:
        raise ValueError("YouTube ID extraction failed.")

    db_path = root / SONG_DB_FILE
    db = load_song_db(db_path)
    db_songs = db.get("songs", []) if isinstance(db, dict) else []
    existing_row = next(
        (r for r in db_songs if isinstance(r, dict) and str(r.get("youtube_id", "")) == youtube_id),
        None,
    )
    existing_youtube_duration = 0.0
    existing_offset = 0.0
    if isinstance(existing_row, dict):
        try:
            existing_youtube_duration = float(existing_row.get("duration_sec", 0.0) or 0.0)
        except (TypeError, ValueError):
            existing_youtube_duration = 0.0
        try:
            existing_offset = float(existing_row.get("offset", 0.0) or 0.0)
        except (TypeError, ValueError):
            existing_offset = 0.0

    notes, bpm, mp3_duration, y_mono, sr = generate_notes(
        mp3_path,
        chart_mode=str(args.chart_mode),
        seed=int(args.seed),
        hold_gap_sec=float(args.hold_gap),
        merge_sec=float(args.merge_ms) / 1000.0,
    )
    if not args.yes:
        maybe_confirm_duration_mismatch(mp3_duration, existing_youtube_duration)

    auto_offset_enabled = args.auto_offset or not args.manual_offset

    use_cli_offset = args.offset is not None and not batch
    if use_cli_offset:
        offset_value = float(args.offset)
    elif auto_offset_enabled and existing_youtube_duration > 0.0:
        auto_offset = estimate_offset_with_timeout(
            y=y_mono,
            sr=sr,
            bpm=bpm,
            youtube_duration=existing_youtube_duration,
            mp3_duration=mp3_duration,
        )
        if auto_offset is None:
            print("長くかかるので、自分で調べてね")
            return False
        offset_value = auto_offset
        print(f"[AUTO] offset 推定: {offset_value:+.4f}s")
    elif auto_offset_enabled:
        offset_value = float(existing_offset)
        print(f"[AUTO] offset 推定スキップ（duration情報なし）: {offset_value:+.4f}s")
    else:
        offset_value = prompt_offset(existing_offset)

    charts_dir = root / CHARTS_DIR
    charts_dir.mkdir(parents=True, exist_ok=True)
    chart_path = charts_dir / f"{youtube_id}.json"
    chart_payload = {
        "youtube_id": youtube_id,
        "bpm": round(float(bpm), 2),
        "duration_sec": round(float(mp3_duration), 3),
        "notes": notes,
    }
    chart_path.write_text(
        json.dumps(chart_payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    rel_mp3 = mp3_path.name
    entry = {
        "youtube_id": youtube_id,
        "title": mp3_path.stem,
        "youtube_title": mp3_path.stem,
        "duration_sec": round(float(mp3_duration), 3),
        "mp3": rel_mp3,
        "offset": round(float(offset_value), 4),
    }
    youtube_duration, _ = upsert_song(db_path, entry)
    if youtube_duration > 0.0 and abs(mp3_duration - youtube_duration) > MISMATCH_WARN_SEC:
        print(
            "[WARN] MP3/YouTube duration mismatch:",
            f"mp3={mp3_duration:.3f}s youtube={youtube_duration:.3f}s",
        )

    rel_chart = Path(CHARTS_DIR) / f"{youtube_id}.json"
    print(f"generated: {rel_chart}")
    print(f"notes: {len(notes)} bpm:{bpm:.2f} duration:{mp3_duration:.3f}s")
    print(f"youtube_id: {youtube_id}")
    print(f"offset: {offset_value:+.4f}s")
    print(f"updated: {SONG_DB_FILE}")
    return True


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate chart JSON from MP3. 引数なしなら番号付きURL/MP3を自動紐づけ。"
    )
    parser.add_argument("--mp3", required=False, help="MP3 file path")
    parser.add_argument("--youtube", required=False, help="YouTube URL")
    parser.add_argument(
        "--slot",
        required=False,
        type=int,
        help="youtube_url.txt の何番目か (1-based)",
    )
    parser.add_argument(
        "--yes",
        action="store_true",
        help="duration mismatch確認をスキップして続行",
    )
    parser.add_argument(
        "--offset",
        required=False,
        type=float,
        help="offset を直接指定（未指定時は対話入力）",
    )
    parser.add_argument(
        "--auto-offset",
        action="store_true",
        help="長さ差がある時、BPM解析で offset を自動推定",
    )
    parser.add_argument(
        "--manual-offset",
        action="store_true",
        help="offset を対話入力で手動調整（自動判定を使わない）",
    )
    parser.add_argument(
        "--chart-mode",
        choices=("hybrid", "beat", "onset"),
        default="hybrid",
        help="hybrid=ビート＋強オンセット, beat=ビートのみ（叩きやすい）, onset=強オンセットのみ",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="レーンのランダム要素の乱数シード",
    )
    parser.add_argument(
        "--hold-gap",
        type=float,
        default=DEFAULT_HOLD_GAP_SEC,
        help="この秒数より離れた連続ノーツを hold にまとめる（終端の tap は折り畳む）",
    )
    parser.add_argument(
        "--merge-ms",
        type=float,
        default=DEFAULT_MERGE_SEC * 1000.0,
        help="ビートとオンセットの近接時刻をまとめる閾値（ミリ秒）",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help=(
            "youtube_url.txt と mp3/ の番号対応が取れる組をすべて順に生成する（--mp3 未指定時のみ）。"
            "長さ不一致の確認が各曲で出る場合は --yes を併用推奨。"
        ),
    )
    args = parser.parse_args()

    root = Path(__file__).resolve().parent
    work_items: list[tuple[Path, str]] = []

    if args.all and (args.mp3 or args.youtube):
        raise ValueError("--all は --mp3 / --youtube を付けずに使ってください。")
    if args.all and args.slot is not None:
        raise ValueError("--all と --slot は同時に指定できません。")

    if args.mp3 and args.youtube:
        mp3_path = (root / args.mp3).resolve() if not Path(args.mp3).is_absolute() else Path(args.mp3)
        work_items.append((mp3_path, args.youtube))
    elif args.mp3 and not args.youtube:
        m = re.match(r"^\s*(\d{1,2})\s*[-_ ]", Path(args.mp3).stem)
        if not m:
            raise ValueError("--youtube 未指定時は MP3名に番号プレフィックスが必要です。")
        slot = int(m.group(1))
        urls = load_youtube_url_candidates(root)
        if slot <= 0 or slot > len(urls):
            raise ValueError(f"{YOUTUBE_URL_TXT} に {slot} 番のURLがありません。")
        youtube_url = urls[slot - 1]
        mp3_path = (root / args.mp3).resolve() if not Path(args.mp3).is_absolute() else Path(args.mp3)
        print(f"[AUTO] mp3番号 {slot} -> youtube_id={extract_youtube_id(youtube_url)}")
        work_items.append((mp3_path, youtube_url))
    elif args.youtube and not args.mp3:
        raise ValueError("--youtube 指定時は --mp3 も指定してください。")
    else:
        if not numbered_mp3_files(root):
            raise FileNotFoundError(
                f"{MP3_DIR} に 01-〜99- で始まるMP3がないため、自動紐づけできません。"
            )
        if args.all:
            for n, p, u in list_auto_pairs(root):
                print(f"[AUTO] slot={n} url_id={extract_youtube_id(u)} mp3={p.name}")
                work_items.append((p, u))
        else:
            mp3_path, youtube_url = pick_auto_pair(root, args.slot)
            work_items.append((mp3_path, youtube_url))

    batch = len(work_items) > 1
    if batch and args.offset is not None:
        print("[WARN] 複数曲処理では --offset を使わず、曲ごとに offset を推定します。")

    for mp3_path, youtube_url in work_items:
        ok = generate_chart_for_pair(root, mp3_path, youtube_url, args, batch=batch)
        if not ok:
            return
        if batch:
            print()


if __name__ == "__main__":
    main()
