"""
YouTube音ゲー（Python / pygame）
仕様: youtube_otoge_spec.txt

3D 透視投影は testRhythmgame.py と同型（FOV・消点・床面 y・z で奥行き）。
譜面の時刻を z に写像し、ステージ・ノーツは半透明。Windows でウィンドウ透過。
"""

from __future__ import annotations

import json
import queue
import sys
import time
import webbrowser
from dataclasses import dataclass
from pathlib import Path

import pygame

from youtube_audio_capture import JSON_SNAPSHOT, LiveAudioAnalyzer

# ウィンドウ・3D・UI の基準解像度 800x600 からの倍率
DISPLAY_SCALE = 1.5


def sounddevice_available() -> bool:
    try:
        import sounddevice  # noqa: F401
    except ImportError:
        return False
    return True


def make_ui_fonts() -> tuple[pygame.font.Font, pygame.font.Font, pygame.font.Font]:
    """日本語表示用。Consolas 等は CJK が tofu になるため OS 標準のゴシックを優先。"""
    candidates = (
        "meiryo",
        "meiryoui",
        "yugothicui",
        "yugothic",
        "yu gothic ui",
        "msgothic",
        "ms gothic",
        "microsoft yahei",
        "notosanscjkjp",
    )
    path: str | None = None
    for name in candidates:
        try:
            path = pygame.font.match_font(name)
        except (OSError, TypeError):
            path = None
        if path:
            break
    st = max(12, int(round(28 * DISPLAY_SCALE)))
    sm = max(10, int(round(22 * DISPLAY_SCALE)))
    ss = max(8, int(round(18 * DISPLAY_SCALE)))
    if path:
        return (
            pygame.font.Font(path, st),
            pygame.font.Font(path, sm),
            pygame.font.Font(path, ss),
        )
    f = pygame.font.SysFont(None, sm)
    return pygame.font.SysFont(None, st), f, pygame.font.SysFont(None, ss)


# --- 表示 ---
WIDTH, HEIGHT = int(800 * DISPLAY_SCALE), int(600 * DISPLAY_SCALE)
LANE_COUNT = 4
FPS = 60

# --- testRhythmgame 由来の 3D パラメータ ---
FOV = int(round(400 * DISPLAY_SCALE))
VANISHING_Y = int(round(200 * DISPLAY_SCALE))
FLOOR_Y = int(round(100 * DISPLAY_SCALE))
Z_START = 20.0
Z_END = 0.5
LANES_X = (-1.5, -0.5, 0.5, 1.5)
# note_time - now がこの秒数で Z_START→Z_END を移動（大きいほど遠くから長く見える）
APPROACH_SECONDS = 1.85

# 半透明（0-255）。ステージ線・ノーツ本体
ALPHA_STAGE_LINE = 100
ALPHA_STAGE_FILL = 45
ALPHA_NOTE_FILL = 160
ALPHA_NOTE_EDGE = 210
ALPHA_HIT_LINE = 180

# Windows ウィンドウ全体の不透明度（255=不透明）。低いほど YouTube が透ける
WINDOW_ALPHA = 200

KEY_TO_LANE = {
    pygame.K_d: 0,
    pygame.K_f: 1,
    pygame.K_j: 2,
    pygame.K_k: 3,
}

# --- 判定 ---
PERFECT_WIN = 0.05
GREAT_WIN = 0.10
GOOD_WIN = 0.20

SCORE_TABLE = {"perfect": 100, "great": 70, "good": 50, "miss": 0}

# --- MP3 再生は停止（YouTube 音声のループバック解析でノーツ生成）---
# MUSIC_FILE = "music.mp3"
CHART_FILE = "chart.json"  # 参照用・未使用可
SYNC_FILE = "sync.json"
DEFAULT_YOUTUBE_URL = (
    "https://www.youtube.com/watch?v=5tc14WHUoMw&list=RD5tc14WHUoMw&start_radio=1"
)


@dataclass
class Note:
    time: float
    lane: int
    done: bool = False


def load_chart(path: Path) -> list[Note]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    out: list[Note] = []
    for row in raw:
        out.append(Note(time=float(row["time"]), lane=int(row["lane"])))
    out.sort(key=lambda n: (n.time, n.lane))
    return out


def load_youtube_url(root: Path) -> str:
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
    a = abs(dt)
    if a <= PERFECT_WIN:
        return "perfect"
    if a <= GREAT_WIN:
        return "great"
    if a <= GOOD_WIN:
        return "good"
    return None


def note_hit_delta_to_z(delta: float) -> float:
    """delta = note_time - now。手前のヒット面 z=Z_END、未来ほど z は大きい。"""
    return Z_END + delta * (Z_START - Z_END) / APPROACH_SECONDS


def note_z_visible(note_time: float, now: float) -> float | None:
    delta = note_time - now
    if delta < -0.35:
        return None
    if delta > APPROACH_SECONDS + 0.55:
        return None
    z = note_hit_delta_to_z(delta)
    return max(0.25, z)


def project_xy(lx: float, z: float) -> tuple[float, float]:
    zz = max(0.2, z)
    sx = (lx / zz) * FOV + WIDTH // 2
    sy = (FLOOR_Y / zz) + VANISHING_Y
    return sx, sy


def note_screen_size(z: float) -> int:
    zz = max(0.2, z)
    return max(int(round(8 * DISPLAY_SCALE)), int(50 * DISPLAY_SCALE / zz))


def try_set_window_see_through() -> None:
    if sys.platform == "win32":
        try:
            import ctypes

            wm = pygame.display.get_wm_info()
            hwnd = wm.get("window")
            if not hwnd:
                return
            user32 = ctypes.windll.user32
            GWL_EXSTYLE = -20
            WS_EX_LAYERED = 0x80000
            ex = user32.GetWindowLongW(hwnd, GWL_EXSTYLE)
            user32.SetWindowLongW(hwnd, GWL_EXSTYLE, ex | WS_EX_LAYERED)
            user32.SetLayeredWindowAttributes(hwnd, 0, WINDOW_ALPHA, 0x2)
            return
        except (OSError, AttributeError, KeyError, TypeError):
            pass
    try:
        win = pygame.display.get_window()
        if win is not None and hasattr(win, "set_opacity"):
            win.set_opacity(WINDOW_ALPHA / 255.0)
    except (AttributeError, pygame.error, TypeError):
        pass


def draw_alpha_round_rect(
    target: pygame.Surface,
    rect: pygame.Rect,
    fill_rgba: tuple[int, int, int, int],
    border_rgba: tuple[int, int, int, int] | None,
    border_w: int,
    radius: int,
) -> None:
    s = pygame.Surface(rect.size, pygame.SRCALPHA)
    pygame.draw.rect(s, fill_rgba, s.get_rect(), border_radius=radius)
    if border_rgba and border_w > 0:
        pygame.draw.rect(s, border_rgba, s.get_rect(), width=border_w, border_radius=radius)
    target.blit(s, rect.topleft)


def draw_3d_stage(overlay: pygame.Surface) -> None:
    """testRhythmgame と同じガイド線。レーン床は z=Z_START〜Z_END の台形を投影。"""
    for lane in range(LANE_COUNT):
        ll = lane - 2
        rr = lane - 1
        poly = [
            project_xy(ll, Z_START),
            project_xy(rr, Z_START),
            project_xy(rr, Z_END),
            project_xy(ll, Z_END),
        ]
        poly_i = [(int(p[0]), int(p[1])) for p in poly]
        lane_surf = pygame.Surface((WIDTH, HEIGHT), pygame.SRCALPHA)
        pygame.draw.polygon(lane_surf, (40, 50, 90, ALPHA_STAGE_FILL), poly_i)
        overlay.blit(lane_surf, (0, 0))

    lw = max(1, int(round(2 * DISPLAY_SCALE)))
    for lx in (-2, -1, 0, 1, 2):
        x_far, y_far = project_xy(lx, Z_START)
        x_near, y_near = project_xy(lx, Z_END)
        pygame.draw.line(
            overlay,
            (160, 170, 220, ALPHA_STAGE_LINE),
            (int(x_far), int(y_far)),
            (int(x_near), int(y_near)),
            lw,
        )

    xl, yl = project_xy(-2, Z_END)
    xr, yr = project_xy(2, Z_END)
    hlw = max(1, int(round(4 * DISPLAY_SCALE)))
    pygame.draw.line(
        overlay,
        (255, 120, 160, ALPHA_HIT_LINE),
        (int(xl), int(yl)),
        (int(xr), int(yr)),
        hlw,
    )


def main() -> None:
    pygame.init()
    # MP3 再生オフ（mixer は未使用のため初期化しない）
    # pygame.mixer.init()
    pygame.display.set_caption("YouTube音ゲー (pygame) — 音声反応 / 重ね表示")
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    try_set_window_see_through()

    clock = pygame.time.Clock()
    font_title, font, font_small = make_ui_fonts()

    root = Path(__file__).resolve().parent
    time_offset_sec = load_time_offset_sec(root)
    youtube_url = load_youtube_url(root)

    notes: list[Note] = []
    session_started = False
    session_t0_perf: float | None = None
    audio_analyzer = LiveAudioAnalyzer(root, note_lead_sec=APPROACH_SECONDS)
    audio_ok = False
    has_sounddevice = sounddevice_available()

    def game_time_s() -> float:
        if session_t0_perf is None:
            return 0.0
        return (time.perf_counter() - session_t0_perf) + time_offset_sec

    def start_session() -> None:
        nonlocal session_started, session_t0_perf, audio_ok
        if session_started:
            return
        session_started = True
        session_t0_perf = time.perf_counter()
        audio_ok = audio_analyzer.start(session_t0_perf)
        if not audio_ok:
            print(audio_analyzer.get_hud().get("error", "audio"), file=sys.stderr)
        webbrowser.open(youtube_url)

    bw = int(round(280 * DISPLAY_SCALE))
    bh = int(round(56 * DISPLAY_SCALE))
    start_btn = pygame.Rect(WIDTH // 2 - bw // 2, HEIGHT // 2 - bh // 2, bw, bh)

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
            screen.fill((12, 14, 22))
            overlay = pygame.Surface((WIDTH, HEIGHT), pygame.SRCALPHA)
            title = font_title.render("YouTube 音ゲー", True, (230, 230, 240))
            tr = title.get_rect(
                center=(WIDTH // 2, HEIGHT // 2 - int(round(120 * DISPLAY_SCALE)))
            )
            overlay.blit(title, tr)
            btn_r = max(4, int(round(10 * DISPLAY_SCALE)))
            btn_w = max(1, int(round(2 * DISPLAY_SCALE)))
            pygame.draw.rect(
                overlay, (80, 120, 220, 200), start_btn, border_radius=btn_r
            )
            pygame.draw.rect(
                overlay, (220, 235, 255, 230), start_btn, width=btn_w, border_radius=btn_r
            )
            lbl = font.render("スタート", True, (255, 255, 255))
            lr = lbl.get_rect(center=start_btn.center)
            overlay.blit(lbl, lr)
            sub = font_small.render(
                "クリック / Space / Enter ・ YouTube 再生→ループバックでノーツ生成",
                True,
                (170, 175, 195),
            )
            sr = sub.get_rect(
                center=(WIDTH // 2, HEIGHT // 2 + int(round(48 * DISPLAY_SCALE)))
            )
            overlay.blit(sub, sr)
            if not has_sounddevice:
                w1 = font_small.render(
                    "! 音声取り込み用パッケージ sounddevice が未インストールです。",
                    True,
                    (255, 140, 120),
                )
                w2 = font_small.render(
                    "  PowerShell で:  pip install sounddevice",
                    True,
                    (255, 180, 140),
                )
                overlay.blit(
                    w1,
                    (
                        WIDTH // 2 - w1.get_width() // 2,
                        HEIGHT // 2 + int(round(78 * DISPLAY_SCALE)),
                    ),
                )
                overlay.blit(
                    w2,
                    (
                        WIDTH // 2 - w2.get_width() // 2,
                        HEIGHT // 2 + int(round(100 * DISPLAY_SCALE)),
                    ),
                )
            esc = font_small.render("ESC で終了", True, (130, 135, 150))
            margin = int(round(12 * DISPLAY_SCALE))
            esc_y = HEIGHT - int(round(28 * DISPLAY_SCALE))
            overlay.blit(esc, (margin, esc_y))
            screen.blit(overlay, (0, 0))
            pygame.display.flip()
            clock.tick(FPS)
            continue

        now = game_time_s()

        while True:
            try:
                ev = audio_analyzer.note_queue.get_nowait()
                notes.append(
                    Note(time=float(ev["hit_time"]), lane=int(ev["lane"]), done=False)
                )
            except queue.Empty:
                break

        audio_analyzer.maybe_flush_json(time.perf_counter())

        for n in notes:
            if n.done:
                continue
            if now > n.time + GOOD_WIN:
                n.done = True
                score += SCORE_TABLE["miss"]
                combo = 0
                last_judgment = "MISS"

        screen.fill((8, 10, 18))
        overlay = pygame.Surface((WIDTH, HEIGHT), pygame.SRCALPHA)
        draw_3d_stage(overlay)

        drawable: list[tuple[float, Note]] = []
        for n in notes:
            if n.done:
                continue
            z = note_z_visible(n.time, now)
            if z is None:
                continue
            drawable.append((z, n))
        drawable.sort(key=lambda t: t[0], reverse=True)

        for z, n in drawable:
            lx = LANES_X[n.lane]
            sx, sy = project_xy(lx, z)
            size = note_screen_size(z)
            rect = pygame.Rect(int(sx - size // 2), int(sy - size // 2), size, size)
            br = max(2, min(8, size // 8))
            draw_alpha_round_rect(
                overlay,
                rect,
                (0, 255, 255, ALPHA_NOTE_FILL),
                (200, 255, 255, ALPHA_NOTE_EDGE),
                max(1, size // 24),
                br,
            )

        screen.blit(overlay, (0, 0))

        hud_h = int(round(90 * DISPLAY_SCALE))
        hud_overlay = pygame.Surface((WIDTH, hud_h), pygame.SRCALPHA)
        hud_overlay.fill((10, 12, 20, 140))
        screen.blit(hud_overlay, (0, 0))
        hud = [f"Score: {score}", f"Combo: {combo}"]
        if last_judgment:
            hud.append(f"Last: {last_judgment}")
        margin = int(round(12 * DISPLAY_SCALE))
        y0 = int(round(8 * DISPLAY_SCALE))
        line_gap = int(round(26 * DISPLAY_SCALE))
        for i, line in enumerate(hud):
            surf = font.render(line, True, (240, 240, 245))
            screen.blit(surf, (margin, y0 + i * line_gap))

        hint = font_small.render("D F J K / ESC終了", True, (160, 165, 185))
        screen.blit(hint, (margin, HEIGHT - int(round(28 * DISPLAY_SCALE))))

        # 右上: 取り込み音量・リズム指標
        hud_r = audio_analyzer.get_hud()
        rms = float(hud_r.get("rms", 0.0))
        rms_s = float(hud_r.get("rms_smooth", 0.0))
        bpm = float(hud_r.get("estimated_bpm", 0.0))
        onset = float(hud_r.get("last_onset", 0.0))
        dev_w = max(8, int(round(28 * DISPLAY_SCALE)))
        err_w = max(20, int(round(45 * DISPLAY_SCALE)))
        dev = str(hud_r.get("device", ""))[:dev_w]
        err = str(hud_r.get("error", ""))
        lines_r = [
            f"RMS {rms:.4f}",
            f"RMS~ {rms_s:.4f}",
            f"BPM~ {bpm:.0f}",
            f"Onset {onset:.4f}",
            f"{JSON_SNAPSHOT}",
        ]
        if dev:
            lines_r.append(dev)
        if err:
            lines_r.append(err[:err_w])
        rx = WIDTH - int(round(280 * DISPLAY_SCALE))
        r_gap = int(round(20 * DISPLAY_SCALE))
        r_top = int(round(8 * DISPLAY_SCALE))
        for i, line in enumerate(lines_r):
            surf = font_small.render(line, True, (200, 210, 230))
            screen.blit(surf, (rx, r_top + i * r_gap))

        pygame.display.flip()
        clock.tick(FPS)

    audio_analyzer.stop()
    pygame.quit()


if __name__ == "__main__":
    main()
