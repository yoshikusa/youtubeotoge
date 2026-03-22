"""
YouTube音ゲー（Python / pygame）
仕様: youtube_otoge_spec.txt

実装段階:
  段階1 — 800x600・4レーン・D/F/J/K・左→右スクロール・右判定ライン
  段階2 — JSON譜面読込・BGMとゲーム時間の同期
  段階3 — Perfect/Great/Good/Miss とスコア
  段階4 — コンボ表示・軽いUI整理（mp3取得は mp3download.py）
"""

from __future__ import annotations

import json
import sys
import webbrowser
from dataclasses import dataclass
from pathlib import Path

import pygame

# --- 段階1: 基本レイアウト・入力 ---
WIDTH, HEIGHT = 800, 600
LANE_COUNT = 4
FPS = 60

HIT_LINE_X = WIDTH - 100  # 左→右スクロール用の判定線（画面右寄り）
LANE_TOP = 120
LANE_HEIGHT = (HEIGHT - LANE_TOP - 80) // LANE_COUNT
NOTE_W, NOTE_H = 44, min(LANE_HEIGHT - 8, 52)
SCROLL_SPEED = 380.0  # px/s（ノーツが判定線に来る位置は時間と同期）

KEY_TO_LANE = {
    pygame.K_d: 0,
    pygame.K_f: 1,
    pygame.K_j: 2,
    pygame.K_k: 3,
}

# --- 段階3: 判定・スコア（秒） ---
PERFECT_WIN = 0.05
GREAT_WIN = 0.10
GOOD_WIN = 0.20

SCORE_TABLE = {"perfect": 100, "great": 70, "good": 50, "miss": 0}

MUSIC_FILE = "music.mp3"
CHART_FILE = "chart.json"
# 任意: {"time_offset_sec": -0.2} でゲーム時間をずらす（負でノーツが遅く＝実音声が遅いとき）
SYNC_FILE = "sync.json"
# フォルダ内の *.url（Internet ショートカット）から読む。無ければこの URL
DEFAULT_YOUTUBE_URL = (
    "https://www.youtube.com/watch?v=5tc14WHUoMw&list=RD5tc14WHUoMw&start_radio=1"
)


@dataclass
class Note:
    time: float
    lane: int
    done: bool = False


def load_chart(path: Path) -> list[Note]:
    """段階2: JSON譜面 [{\"time\", \"lane\"}, ...]"""
    raw = json.loads(path.read_text(encoding="utf-8"))
    out: list[Note] = []
    for row in raw:
        out.append(Note(time=float(row["time"]), lane=int(row["lane"])))
    out.sort(key=lambda n: (n.time, n.lane))
    return out


def load_youtube_url(root: Path) -> str:
    """プロジェクト内の .url から URL= を探す。"""
    for path in sorted(root.glob("*.url")):
        try:
            text = path.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        for line in text.splitlines():
            line = line.strip()
            if line.upper().startswith("URL="):
                u = line[4:].strip()
                if u.startswith("http"):
                    return u
    return DEFAULT_YOUTUBE_URL


def load_time_offset_sec(root: Path) -> float:
    p = root / SYNC_FILE
    if not p.is_file():
        return 0.0
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
        return float(data.get("time_offset_sec", 0.0))
    except (OSError, json.JSONDecodeError, TypeError, ValueError):
        return 0.0


def judge_delta(dt: float) -> str | None:
    """段階3: キー入力時の時間差（秒）から判定名。窓外は None。"""
    a = abs(dt)
    if a <= PERFECT_WIN:
        return "perfect"
    if a <= GREAT_WIN:
        return "great"
    if a <= GOOD_WIN:
        return "good"
    return None


def lane_y(lane: int) -> int:
    return LANE_TOP + lane * LANE_HEIGHT + (LANE_HEIGHT - NOTE_H) // 2


def note_x(note_time: float, now: float) -> float:
    """左から右: まだ来ていないノーツは左側（x が小さい）。"""
    return HIT_LINE_X - (note_time - now) * SCROLL_SPEED


def main() -> None:
    pygame.init()
    pygame.mixer.init()
    pygame.display.set_caption("YouTube音ゲー (pygame)")
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    clock = pygame.time.Clock()
    font = pygame.font.SysFont("consolas", 22)
    font_small = pygame.font.SysFont("consolas", 18)
    font_title = pygame.font.SysFont("consolas", 28)

    chart_path = Path(__file__).resolve().parent / CHART_FILE
    if not chart_path.is_file():
        print(f"譜面が見つかりません: {chart_path}", file=sys.stderr)
        sys.exit(1)

    notes = load_chart(chart_path)
    root = Path(__file__).resolve().parent
    time_offset_sec = load_time_offset_sec(root)
    youtube_url = load_youtube_url(root)

    music_path = root / MUSIC_FILE
    music_started_tick: int | None = None
    session_started = False

    if music_path.is_file():
        pygame.mixer.music.load(str(music_path))
    else:
        print(f"警告: {MUSIC_FILE} がありません。無音でタイマーのみ進行します。")

    def game_time_s() -> float:
        if music_started_tick is None:
            return 0.0
        return (pygame.time.get_ticks() - music_started_tick) / 1000.0 + time_offset_sec

    def start_session() -> None:
        nonlocal session_started, music_started_tick
        if session_started:
            return
        session_started = True
        music_started_tick = pygame.time.get_ticks()
        if music_path.is_file():
            pygame.mixer.music.play(start=0.0)
        webbrowser.open(youtube_url)

    start_btn = pygame.Rect(WIDTH // 2 - 140, HEIGHT // 2 - 28, 280, 56)

    score = 0
    combo = 0
    last_judgment = ""
    running = True

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif not session_started and event.key in (
                    pygame.K_SPACE,
                    pygame.K_RETURN,
                    pygame.K_KP_ENTER,
                ):
                    start_session()
                elif session_started and event.key in KEY_TO_LANE:
                    now = game_time_s()
                    lane = KEY_TO_LANE[event.key]
                    best: Note | None = None
                    best_dt = 999.0
                    for n in notes:
                        if n.done or n.lane != lane:
                            continue
                        dt = now - n.time
                        if abs(dt) < abs(best_dt):
                            best_dt = dt
                            best = n
                    if best is not None:
                        tier = judge_delta(best_dt)
                        if tier:
                            best.done = True
                            score += SCORE_TABLE[tier]
                            combo += 1
                            last_judgment = tier.upper()
                        else:
                            last_judgment = "MISS"
                            combo = 0
            elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                if not session_started and start_btn.collidepoint(event.pos):
                    start_session()

        if not session_started:
            screen.fill((14, 16, 28))
            title = font_title.render("YouTube 音ゲー", True, (230, 230, 240))
            tr = title.get_rect(center=(WIDTH // 2, HEIGHT // 2 - 120))
            screen.blit(title, tr)
            pygame.draw.rect(screen, (80, 120, 220), start_btn, border_radius=10)
            pygame.draw.rect(screen, (200, 220, 255), start_btn, width=2, border_radius=10)
            lbl = font.render("スタート", True, (255, 255, 255))
            lr = lbl.get_rect(center=start_btn.center)
            screen.blit(lbl, lr)
            sub = font_small.render(
                "クリック / Space / Enter ・ ブラウザで YouTube も再生開始",
                True,
                (150, 155, 175),
            )
            sr = sub.get_rect(center=(WIDTH // 2, HEIGHT // 2 + 48))
            screen.blit(sub, sr)
            esc = font_small.render("ESC で終了", True, (120, 125, 140))
            screen.blit(esc, (12, HEIGHT - 28))
            pygame.display.flip()
            clock.tick(FPS)
            continue

        now = game_time_s()

        # 段階3: 通過後も反応なければミス（1回だけ）
        for n in notes:
            if n.done:
                continue
            if now > n.time + GOOD_WIN:
                n.done = True
                score += SCORE_TABLE["miss"]
                combo = 0
                last_judgment = "MISS"

        # 描画
        screen.fill((18, 18, 28))
        # レーン線
        for i in range(LANE_COUNT + 1):
            y = LANE_TOP + i * LANE_HEIGHT
            pygame.draw.line(screen, (50, 50, 70), (0, y), (WIDTH, y), 1)

        pygame.draw.line(screen, (220, 80, 120), (HIT_LINE_X, 0), (HIT_LINE_X, HEIGHT), 3)

        for n in notes:
            if n.done:
                continue
            x = note_x(n.time, now)
            if x < -NOTE_W or x > WIDTH + NOTE_W:
                continue
            rect = pygame.Rect(int(x - NOTE_W // 2), lane_y(n.lane), NOTE_W, NOTE_H)
            pygame.draw.rect(screen, (200, 210, 255), rect, border_radius=6)
            pygame.draw.rect(screen, (255, 255, 255), rect, width=2, border_radius=6)

        # 段階4: UI（スコア・コンボ・最終判定）
        hud = [
            f"Score: {score}",
            f"Combo: {combo}",
        ]
        if last_judgment:
            hud.append(f"Last: {last_judgment}")
        y0 = 8
        for i, line in enumerate(hud):
            surf = font.render(line, True, (240, 240, 245))
            screen.blit(surf, (12, y0 + i * 26))

        hint = font_small.render("D F J K / ESC終了", True, (140, 140, 160))
        screen.blit(hint, (12, HEIGHT - 28))

        pygame.display.flip()
        clock.tick(FPS)

    pygame.quit()


if __name__ == "__main__":
    main()
