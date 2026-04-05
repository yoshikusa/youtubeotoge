"""
YouTube音ゲー（Python / pygame）
仕様: youtube_otoge_spec.txt

音源はブラウザの YouTube をループバック（sounddevice 入力）で取り込むのみ。
ローカル MP3 再生・ファイル譜面は扱わない。

3D 透視投影（FOV・消点・床面 y・z）。ノーツはオンセット・帯域・RMS に応じて生成（frequency_lanes）。
左上に 4 帯域スペクトラム（レーン色）。ステージ・ノーツは半透明。Windows でウィンドウ透過。
"""

from __future__ import annotations

import io
import json
import os
import queue
import re
import sys
import threading
import time
import webbrowser
from dataclasses import dataclass
from pathlib import Path

import pygame

from frequency_lanes import NOTE_RGB
from youtube_audio_capture import JSON_SNAPSHOT, LiveAudioAnalyzer
from youtube_preview import fetch_youtube_preview

# ウィンドウ・3D・UI の基準解像度 800x600 からの倍率
DISPLAY_SCALE = 1.5
# 起動画面の UI 倍率（1.0=大、0.5=小 の中間付近）
TITLE_SCREEN_UI_SCALE = 0.75


def sounddevice_available() -> bool:
    try:
        import sounddevice  # noqa: F401
    except ImportError:
        return False
    return True


_UI_FONT_CANDIDATES = (
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


def match_ui_gothic_font_path() -> str | None:
    """日本語表示用。Consolas 等は CJK が tofu になるため OS 標準のゴシックを優先。"""
    for name in _UI_FONT_CANDIDATES:
        try:
            path = pygame.font.match_font(name)
        except (OSError, TypeError):
            path = None
        if path:
            return path
    return None


def make_font_at_px(size_px: int) -> pygame.font.Font:
    """UI 用フォント。size_px は pygame のポイント相当（ピクセル高さの目安）。"""
    px = max(6, min(120, int(size_px)))
    path = match_ui_gothic_font_path()
    if path:
        return pygame.font.Font(path, px)
    return pygame.font.SysFont(None, px)


def make_ui_fonts() -> tuple[pygame.font.Font, pygame.font.Font, pygame.font.Font]:
    path = match_ui_gothic_font_path()
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
WIDTH, HEIGHT = int(900 * DISPLAY_SCALE), int(600 * DISPLAY_SCALE)
LANE_COUNT = 4
FPS = 60
# --- testRhythmgame 由来の 3D パラメータ ---
FOV = int(round(300* DISPLAY_SCALE))
VANISHING_Y = int(round(200 * DISPLAY_SCALE))
FLOOR_Y = int(round(100 * DISPLAY_SCALE))
Z_START = 20.0
Z_END = 0.5
LANES_X_BASE = (-0.6, -0.2, 0.2, 0.6)
LANES_X = LANES_X_BASE
LANE_SPREAD_REF = 0.7
# note_time - now がこの秒数で Z_START→Z_END を移動（大きいほど遠くから長く見える）
APPROACH_SECONDS = 1.85


def lanes_x_from_spread(spread: float) -> tuple[float, float, float, float]:
    """レーンの世界 x 中心。UI の Lane Spread（既定 0.7）で横幅を調整。"""
    s = spread / LANE_SPREAD_REF
    return tuple(x * s for x in LANES_X_BASE)

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

# 判定文字・フラッシュ演出（判定線付近）
JUDGE_FLOAT_DURATION = 0.5
JUDGE_RISE_SPEED = 100.0
FLASH_DECAY_PER_S = 3.0
JUDGE_FLASH_RING_BASE = 30

# --- ノーツはループバック音声の解析のみ（時刻=オンセット、レーン=周波数帯）---
SYNC_FILE = "sync.json"
# 表示オプション（任意）。無い場合は既定値。例: display_settings.example.json
DISPLAY_SETTINGS_FILE = "display_settings.json"
# 再生 URL の最終フォールバック（youtube_url.txt / *.url が無いとき）
DEFAULT_YOUTUBE_URL = (
    "https://www.youtube.com/watch?v=5tc14WHUoMw&list=RD5tc14WHUoMw&start_radio=1"
)
# 最大 MAX_YOUTUBE_SLOTS 行（先頭の http 行を 1 番から）。無ければ youtube_url.txt → *.url → 既定
# 数字キー 1〜9・0 は先頭 10 件まで（0=10 番目）。11 件目以降は矢印・▲▼・クリックで選択。
YOUTUBE_URLS_TXT = "youtube_urls.txt"
YOUTUBE_URL_TXT = "youtube_url.txt"
MAX_YOUTUBE_SLOTS = 20
# 起動・ゲーム共通の調整（onset_rms_tune.json に保存）
ONSET_RMS_TUNE_FILE = "onset_rms_tune.json"
DEFAULT_SENSITIVITY = 0.5
DEFAULT_LANE_SPREAD = 0.7
DEFAULT_MIN_VOLUME = 0.0001
TUNE_SENS_MIN = 0.0
TUNE_SENS_MAX = 2.0
TUNE_SENS_STEP = 0.05
TUNE_SPREAD_MIN = 0.25
TUNE_SPREAD_MAX = 1.5
TUNE_SPREAD_STEP = 0.05
TUNE_VOL_MIN = 0.0
TUNE_VOL_MAX = 0.2
TUNE_VOL_STEP = 0.0001  # 通常の ± 刻み
TUNE_VOL_STEP_SHIFT = 0.001  # Shift 押しながら ± のときの刻み


@dataclass
class Note:
    time: float
    lane: int
    done: bool = False
    rgb: tuple[int, int, int] | None = None
    label: str = ""
    judgment: str | None = None
    judge_time: float = 0.0
    flash: float = 0.0


def _http_urls_from_text(raw: str) -> list[str]:
    """コメント・空行を除き、http で始まる URL を最大 MAX_YOUTUBE_SLOTS 件。"""
    out: list[str] = []
    for line in raw.splitlines():
        s = line.strip()
        if not s or s.startswith("#"):
            continue
        if s[0].isdigit():
            s = re.sub(r"^\d{1,2}[\.\):\s|、]+\s*", "", s).strip()
        if s.startswith("http://") or s.startswith("https://"):
            out.append(s.split()[0])
        if len(out) >= MAX_YOUTUBE_SLOTS:
            break
    return out


def load_youtube_url_candidates(root: Path) -> list[str]:
    """タイトルで選ぶ候補（1 件以上保証）。先頭 MAX_YOUTUBE_SLOTS 件まで。"""
    urls: list[str] = []
    multi = root / YOUTUBE_URLS_TXT
    if multi.is_file():
        try:
            urls = _http_urls_from_text(multi.read_text(encoding="utf-8", errors="ignore"))
        except OSError:
            urls = []
    if not urls:
        one = root / YOUTUBE_URL_TXT
        if one.is_file():
            try:
                urls = _http_urls_from_text(one.read_text(encoding="utf-8", errors="ignore"))
            except OSError:
                urls = []
    if not urls:
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
                        urls.append(u)
                        break
            if len(urls) >= MAX_YOUTUBE_SLOTS:
                break
    if not urls:
        urls = [DEFAULT_YOUTUBE_URL]
    return urls[:MAX_YOUTUBE_SLOTS]


@dataclass(frozen=True)
class TitleScreenLayout:
    """起動画面の幾何（ヒットテストと描画で同一値を使う）。"""

    main_title_surf: pygame.Surface
    mt_rect: pygame.Rect
    content_y: int
    prev_px: int
    prev_py: int
    prev_tw: int
    prev_th: int
    prev_pad: int
    bx: int
    panel_w: int
    line_h: int
    url_ch: int
    tx_in: int
    list_hdr_surf: pygame.Surface
    list_hdr_pos: tuple[int, int]
    sel_up_btn: pygame.Rect
    sel_down_btn: pygame.Rect
    url_pick_rects: list[tuple[int, pygame.Rect]]
    Tu: float


@dataclass(frozen=True)
class AudioTuneLayout:
    """TIMING / POSITION / CONTROL の 3 段調整 UI の幾何。"""

    sens_dec: pygame.Rect
    sens_inc: pygame.Rect
    spread_dec: pygame.Rect
    spread_inc: pygame.Rect
    vol_dec: pygame.Rect
    vol_inc: pygame.Rect
    timing_hdr_xy: tuple[int, int]
    sens_lbl_xy: tuple[int, int]
    sens_val_xy: tuple[int, int]
    pos_hdr_xy: tuple[int, int]
    spread_lbl_xy: tuple[int, int]
    spread_val_xy: tuple[int, int]
    ctrl_hdr_xy: tuple[int, int]
    vol_lbl_xy: tuple[int, int]
    vol_val_xy: tuple[int, int]
    panel_back: pygame.Rect


def compute_audio_tune_layout(
    height: int,
    width: int,
    display_scale: float,
    Tu: float,
    font: pygame.font.Font,
    sensitivity: float,
    lane_spread: float,
    min_volume: float,
) -> AudioTuneLayout:
    lm = max(6, int(round(8 * display_scale * Tu)))
    g = max(4, int(round(6 * display_scale * Tu)))
    bw = max(22, int(round(26 * display_scale * Tu)))
    bh = max(20, int(round(24 * display_scale * Tu)))
    sec_gap = int(round(4 * display_scale * Tu))
    row_gap = int(round(3 * display_scale * Tu))
    col_lab = (165, 172, 198)
    col_val = (245, 246, 252)
    hdr_c = (125, 135, 168)

    panel_top = height - int(round(252 * display_scale * Tu))
    y = panel_top

    timing_hdr_xy = (lm, y)
    y += font.get_linesize() + sec_gap

    lab_s = font.render("[ Sensitivity ]", True, col_lab)
    vs = font.render(f"{sensitivity:.2f}", True, col_val)
    sens_lbl_xy = (lm, y + (bh - lab_s.get_height()) // 2)
    x = lm + lab_s.get_width() + g
    sens_dec = pygame.Rect(x, y, bw, bh)
    x = sens_dec.right + g
    sens_val_xy = (x, y + (bh - vs.get_height()) // 2)
    x += vs.get_width() + g
    sens_inc = pygame.Rect(x, y, bw, bh)
    y += max(bh, lab_s.get_height()) + row_gap

    pos_hdr_xy = (lm, y)
    y += font.get_linesize() + sec_gap

    lab_sp = font.render("[ Lane Spread ]", True, col_lab)
    vsp = font.render(f"{lane_spread:.2f}", True, col_val)
    spread_lbl_xy = (lm, y + (bh - lab_sp.get_height()) // 2)
    x = lm + lab_sp.get_width() + g
    spread_dec = pygame.Rect(x, y, bw, bh)
    x = spread_dec.right + g
    spread_val_xy = (x, y + (bh - vsp.get_height()) // 2)
    x += vsp.get_width() + g
    spread_inc = pygame.Rect(x, y, bw, bh)
    y += max(bh, lab_sp.get_height()) + row_gap

    ctrl_hdr_xy = (lm, y)
    y += font.get_linesize() + sec_gap

    lab_v = font.render("[ Min Volume ]", True, col_lab)
    vv = font.render(f"{min_volume:.4f}", True, col_val)
    vol_lbl_xy = (lm, y + (bh - lab_v.get_height()) // 2)
    x = lm + lab_v.get_width() + g
    vol_dec = pygame.Rect(x, y, bw, bh)
    x = vol_dec.right + g
    vol_val_xy = (x, y + (bh - vv.get_height()) // 2)
    x += vv.get_width() + g
    vol_inc = pygame.Rect(x, y, bw, bh)
    y += max(bh, lab_v.get_height())

    max_r = max(sens_inc.right, spread_inc.right, vol_inc.right) + 4
    panel_back = pygame.Rect(
        lm - 6,
        panel_top - 6,
        min(width - lm + 6, max_r - lm + 12),
        y - panel_top + 12,
    )

    return AudioTuneLayout(
        sens_dec=sens_dec,
        sens_inc=sens_inc,
        spread_dec=spread_dec,
        spread_inc=spread_inc,
        vol_dec=vol_dec,
        vol_inc=vol_inc,
        timing_hdr_xy=timing_hdr_xy,
        sens_lbl_xy=sens_lbl_xy,
        sens_val_xy=sens_val_xy,
        pos_hdr_xy=pos_hdr_xy,
        spread_lbl_xy=spread_lbl_xy,
        spread_val_xy=spread_val_xy,
        ctrl_hdr_xy=ctrl_hdr_xy,
        vol_lbl_xy=vol_lbl_xy,
        vol_val_xy=vol_val_xy,
        panel_back=panel_back,
    )


def apply_audio_tune_click(
    pos: tuple[int, int],
    layout: AudioTuneLayout,
    sensitivity: float,
    lane_spread: float,
    min_volume: float,
    *,
    min_vol_shift: bool = False,
) -> tuple[float, float, float, bool]:
    if layout.sens_dec.collidepoint(pos):
        return (
            max(
                TUNE_SENS_MIN,
                min(TUNE_SENS_MAX, round(sensitivity - TUNE_SENS_STEP, 3)),
            ),
            lane_spread,
            min_volume,
            True,
        )
    if layout.sens_inc.collidepoint(pos):
        return (
            max(
                TUNE_SENS_MIN,
                min(TUNE_SENS_MAX, round(sensitivity + TUNE_SENS_STEP, 3)),
            ),
            lane_spread,
            min_volume,
            True,
        )
    if layout.spread_dec.collidepoint(pos):
        return (
            sensitivity,
            max(
                TUNE_SPREAD_MIN,
                min(TUNE_SPREAD_MAX, round(lane_spread - TUNE_SPREAD_STEP, 3)),
            ),
            min_volume,
            True,
        )
    if layout.spread_inc.collidepoint(pos):
        return (
            sensitivity,
            max(
                TUNE_SPREAD_MIN,
                min(TUNE_SPREAD_MAX, round(lane_spread + TUNE_SPREAD_STEP, 3)),
            ),
            min_volume,
            True,
        )
    if layout.vol_dec.collidepoint(pos):
        dv = TUNE_VOL_STEP_SHIFT if min_vol_shift else TUNE_VOL_STEP
        return (
            sensitivity,
            lane_spread,
            max(
                TUNE_VOL_MIN,
                min(TUNE_VOL_MAX, round(min_volume - dv, 6)),
            ),
            True,
        )
    if layout.vol_inc.collidepoint(pos):
        dv = TUNE_VOL_STEP_SHIFT if min_vol_shift else TUNE_VOL_STEP
        return (
            sensitivity,
            lane_spread,
            max(
                TUNE_VOL_MIN,
                min(TUNE_VOL_MAX, round(min_volume + dv, 6)),
            ),
            True,
        )
    return (sensitivity, lane_spread, min_volume, False)


def draw_audio_tune_panel(
    target: pygame.Surface,
    layout: AudioTuneLayout,
    sensitivity: float,
    lane_spread: float,
    min_volume: float,
    font: pygame.font.Font,
    display_scale: float,
    Tu: float,
    *,
    backing: bool = False,
) -> None:
    hdr_c = (125, 135, 168)
    col_lab = (165, 172, 198)
    col_val = (245, 246, 252)
    if backing:
        patch = pygame.Surface(layout.panel_back.size, pygame.SRCALPHA)
        patch.fill((10, 12, 22, 200))
        target.blit(patch, layout.panel_back.topleft)
    br_t = max(2, int(round(4 * display_scale * Tu)))
    ln_t = max(1, int(round(display_scale * Tu)))
    for brect in (
        layout.sens_dec,
        layout.sens_inc,
        layout.spread_dec,
        layout.spread_inc,
        layout.vol_dec,
        layout.vol_inc,
    ):
        pygame.draw.rect(target, (70, 100, 160, 220), brect, border_radius=br_t)
        pygame.draw.rect(
            target,
            (200, 215, 240, 220),
            brect,
            width=ln_t,
            border_radius=br_t,
        )
        sym = "−" if brect in (layout.sens_dec, layout.spread_dec, layout.vol_dec) else "+"
        gs = font.render(sym, True, (255, 255, 255))
        target.blit(gs, gs.get_rect(center=brect.center))
    target.blit(font.render("=== TIMING ===", True, hdr_c), layout.timing_hdr_xy)
    target.blit(font.render("[ Sensitivity ]", True, col_lab), layout.sens_lbl_xy)
    target.blit(
        font.render(f"{sensitivity:.2f}", True, col_val), layout.sens_val_xy
    )
    target.blit(font.render("=== POSITION ===", True, hdr_c), layout.pos_hdr_xy)
    target.blit(font.render("[ Lane Spread ]", True, col_lab), layout.spread_lbl_xy)
    target.blit(
        font.render(f"{lane_spread:.2f}", True, col_val), layout.spread_val_xy
    )
    target.blit(font.render("=== CONTROL ===", True, hdr_c), layout.ctrl_hdr_xy)
    target.blit(font.render("[ Min Volume ]", True, col_lab), layout.vol_lbl_xy)
    target.blit(
        font.render(f"{min_volume:.4f}", True, col_val), layout.vol_val_xy
    )


def compute_title_screen_layout(
    width: int,
    display_scale: float,
    title_scale: float,
    title_scr_title: pygame.font.Font,
    title_scr_small: pygame.font.Font,
    candidates: list[str],
) -> TitleScreenLayout:
    Tu = title_scale
    top_m = int(round(8 * display_scale * Tu)) + 2
    main_t = title_scr_title.render("YouTube 音ゲー", True, (230, 230, 240))
    mt_rect = main_t.get_rect(midtop=(width // 2, top_m))
    content_y = mt_rect.bottom + int(round(5 * display_scale * Tu))

    prev_px = int(round(10 * display_scale * Tu))
    prev_tw = int(round(440 * display_scale * Tu))
    prev_th = int(round(248 * display_scale * Tu))
    prev_pad = max(3, int(round(5 * display_scale * Tu)))
    m_u = int(round(10 * display_scale * Tu))
    panel_w = int(round(360 * display_scale * Tu))
    min_bx = prev_px + prev_tw + 2 * prev_pad + 10
    bx = width - panel_w - m_u
    if bx < min_bx:
        bx = min_bx
        panel_w = max(100, width - bx - m_u)
    prev_py = content_y

    btn_w = max(36, int(round(44 * display_scale * Tu)))
    btn_h = max(30, int(round(36 * display_scale * Tu)))
    gap_btn = max(4, int(round(6 * display_scale * Tu)))
    gap_after_hdr_text = max(6, int(round(10 * display_scale * Tu)))

    list_hdr_surf = title_scr_small.render(
        "↑↓ 選択（▲▼でも可） / 1〜0・列クリック",
        True,
        (188, 194, 218),
    )
    hdr_row_y = content_y
    text_w = list_hdr_surf.get_width()
    buttons_w = btn_w * 2 + gap_btn
    inline_w = text_w + gap_after_hdr_text + buttons_w
    margin_r = max(8, int(round(8 * display_scale * Tu)))
    if bx + inline_w <= width - margin_r:
        hdr_text_y = hdr_row_y + max(0, (btn_h - list_hdr_surf.get_height()) // 2)
        list_hdr_pos = (bx, hdr_text_y)
        sel_up_btn = pygame.Rect(
            bx + text_w + gap_after_hdr_text,
            hdr_row_y,
            btn_w,
            btn_h,
        )
        sel_down_btn = pygame.Rect(
            sel_up_btn.right + gap_btn,
            hdr_row_y,
            btn_w,
            btn_h,
        )
        uy = hdr_row_y + max(btn_h, list_hdr_surf.get_height()) + int(
            round(4 * display_scale * Tu)
        )
    else:
        list_hdr_pos = (bx, hdr_row_y)
        btn_row_y = hdr_row_y + list_hdr_surf.get_height() + int(
            round(3 * display_scale * Tu)
        )
        sel_up_btn = pygame.Rect(bx, btn_row_y, btn_w, btn_h)
        sel_down_btn = pygame.Rect(
            sel_up_btn.right + gap_btn,
            btn_row_y,
            btn_w,
            btn_h,
        )
        uy = btn_row_y + btn_h + int(round(4 * display_scale * Tu))
    row_pad = max(2, int(round(3 * display_scale * Tu)))
    line_h = title_scr_small.get_linesize() + row_pad
    url_ch = 34 + int(display_scale * Tu * 4)
    tx_in = int(round(6 * display_scale * Tu))

    url_pick_rects: list[tuple[int, pygame.Rect]] = []
    for i, _u in enumerate(candidates):
        rr = pygame.Rect(bx - 2, uy - 2, panel_w, line_h + 2)
        url_pick_rects.append((i, rr))
        uy += line_h

    return TitleScreenLayout(
        main_title_surf=main_t,
        mt_rect=mt_rect,
        content_y=content_y,
        prev_px=prev_px,
        prev_py=prev_py,
        prev_tw=prev_tw,
        prev_th=prev_th,
        prev_pad=prev_pad,
        bx=bx,
        panel_w=panel_w,
        line_h=line_h,
        url_ch=url_ch,
        tx_in=tx_in,
        list_hdr_surf=list_hdr_surf,
        list_hdr_pos=list_hdr_pos,
        sel_up_btn=sel_up_btn,
        sel_down_btn=sel_down_btn,
        url_pick_rects=url_pick_rects,
        Tu=Tu,
    )


def _pointer_pos_for_mouse_event(event: pygame.event.Event) -> tuple[int, int]:
    """MOUSEBUTTON* の座標。HiDPI 等で event.pos がずれる環境向けに get_pos も参照。"""
    if hasattr(event, "pos") and event.pos is not None:
        return int(event.pos[0]), int(event.pos[1])
    p = pygame.mouse.get_pos()
    return int(p[0]), int(p[1])


@dataclass
class DisplaySettings:
    """display_settings.json から読む項目。未指定は None（コード側で既定を使う）。"""

    hud_right_font_px: int | None = None
    hud_right_line_gap_px: int | None = None


def load_display_settings(root: Path) -> DisplaySettings:
    p = root / DISPLAY_SETTINGS_FILE
    if not p.is_file():
        return DisplaySettings()
    try:
        raw = json.loads(p.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, TypeError):
        return DisplaySettings()
    if not isinstance(raw, dict):
        return DisplaySettings()

    def _opt_int(key: str) -> int | None:
        v = raw.get(key)
        if v is None:
            return None
        try:
            return int(float(v))
        except (TypeError, ValueError):
            return None

    return DisplaySettings(
        hud_right_font_px=_opt_int("hud_right_font_px"),
        hud_right_line_gap_px=_opt_int("hud_right_line_gap_px"),
    )


def load_time_offset_sec(root: Path) -> float:
    p = root / SYNC_FILE
    if not p.is_file():
        return 0.0
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
        return float(data.get("time_offset_sec", 0.0))
    except (OSError, json.JSONDecodeError, TypeError, ValueError):
        return 0.0


def load_game_tune(root: Path) -> tuple[float, float, float]:
    """sensitivity, lane_spread, min_volume。ファイル無し・不正時は既定値。"""
    sens = DEFAULT_SENSITIVITY
    spr = DEFAULT_LANE_SPREAD
    vol = DEFAULT_MIN_VOLUME
    p = root / ONSET_RMS_TUNE_FILE
    if not p.is_file():
        return (sens, spr, vol)
    try:
        raw = json.loads(p.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, TypeError):
        return (sens, spr, vol)
    if not isinstance(raw, dict):
        return (sens, spr, vol)
    try:
        if "sensitivity" in raw:
            sens = float(raw["sensitivity"])
        elif "rms_mul" in raw:
            sens = float(raw["rms_mul"])
        if "lane_spread" in raw:
            spr = float(raw["lane_spread"])
        if "min_volume" in raw:
            vol = float(raw["min_volume"])
    except (TypeError, ValueError):
        return (DEFAULT_SENSITIVITY, DEFAULT_LANE_SPREAD, DEFAULT_MIN_VOLUME)
    sens = max(TUNE_SENS_MIN, min(TUNE_SENS_MAX, sens))
    spr = max(TUNE_SPREAD_MIN, min(TUNE_SPREAD_MAX, spr))
    vol = max(TUNE_VOL_MIN, min(TUNE_VOL_MAX, vol))
    return (sens, spr, vol)


def save_game_tune(root: Path, sensitivity: float, lane_spread: float, min_volume: float) -> None:
    p = root / ONSET_RMS_TUNE_FILE
    try:
        p.write_text(
            json.dumps(
                {
                    "sensitivity": sensitivity,
                    "lane_spread": lane_spread,
                    "min_volume": min_volume,
                },
                ensure_ascii=False,
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
    except OSError:
        pass


def judge_delta(dt: float) -> str | None:
    a = abs(dt)
    if a <= PERFECT_WIN:
        return "perfect"
    if a <= GREAT_WIN:
        return "great"
    if a <= GOOD_WIN:
        return "good"
    return None


def judgment_text_color(tier: str) -> tuple[int, int, int]:
    t = tier.lower()
    if t == "perfect":
        return (255, 228, 120)
    if t == "great":
        return (255, 255, 255)
    if t == "good":
        return (160, 220, 255)
    return (170, 175, 190)


def load_optional_sound_wav(root: Path, filename: str) -> pygame.mixer.Sound | None:
    p = root / filename
    if not p.is_file():
        return None
    try:
        return pygame.mixer.Sound(str(p))
    except pygame.error:
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


def lane_world_x_edges(xs: tuple[float, ...]) -> tuple[float, ...]:
    """レーン中心 xs の各帯の世界 x 境界（長さ LANE_COUNT+1）。

    隣接レーンの中点を区切りとし、最外側は内側区間と同じ半幅を外へ延ばす（従来の ±2 枠と同型）。
    """
    if len(xs) != LANE_COUNT:
        raise ValueError("レーン中心の要素数は LANE_COUNT と一致させてください")
    edges: list[float] = [xs[0] - (xs[1] - xs[0]) * 0.5]
    for i in range(LANE_COUNT - 1):
        edges.append((xs[i] + xs[i + 1]) * 0.5)
    edges.append(xs[LANE_COUNT - 1] + (xs[LANE_COUNT - 1] - xs[LANE_COUNT - 2]) * 0.5)
    return tuple(edges)


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


def scale_surface_to_fit(surf: pygame.Surface, max_w: int, max_h: int) -> pygame.Surface:
    w, h = surf.get_size()
    if w <= 0 or h <= 0:
        return surf
    sc = min(max_w / w, max_h / h, 1.0)
    nw, nh = max(1, int(w * sc)), max(1, int(h * sc))
    return pygame.transform.smoothscale(surf, (nw, nh))


def truncate_surface_width(
    font: pygame.font.Font, text: str, color: tuple[int, int, int], max_w: int
) -> pygame.Surface:
    ell = "…"
    if font.render(text, True, color).get_width() <= max_w:
        return font.render(text, True, color)
    t = text
    while len(t) > 1:
        t = t[:-1]
        if font.render(t + ell, True, color).get_width() <= max_w:
            return font.render(t + ell, True, color)
    return font.render(ell, True, color)


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


def draw_3d_stage(
    overlay: pygame.Surface, lanes_x: tuple[float, float, float, float]
) -> None:
    """lanes_x に沿ったレーン床と縦グリッド。z=Z_START〜Z_END を台形投影。"""
    x_edges = lane_world_x_edges(lanes_x)
    for lane in range(LANE_COUNT):
        ll = x_edges[lane]
        rr = x_edges[lane + 1]
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
    for lx in x_edges:
        x_far, y_far = project_xy(lx, Z_START)
        x_near, y_near = project_xy(lx, Z_END)
        pygame.draw.line(
            overlay,
            (160, 170, 220, ALPHA_STAGE_LINE),
            (int(x_far), int(y_far)),
            (int(x_near), int(y_near)),
            lw,
        )

    xl, yl = project_xy(x_edges[0], Z_END)
    xr, yr = project_xy(x_edges[-1], Z_END)
    hlw = max(1, int(round(4 * DISPLAY_SCALE)))
    pygame.draw.line(
        overlay,
        (255, 120, 160, ALPHA_HIT_LINE),
        (int(xl), int(yl)),
        (int(xr), int(yr)),
        hlw,
    )


def main() -> tuple[float, float]:
    if sys.platform == "win32":
        # マウス座標と pygame のサーフェス座標のずれ（HiDPI）を抑える
        os.environ.setdefault("SDL_WINDOWS_DPI_AWARENESS", "permonitorv2")
    pygame.init()
    try:
        pygame.mixer.init(frequency=44100, size=-16, channels=2, buffer=512)
    except pygame.error:
        pass
    try:
        pygame.key.stop_text_input()
    except (AttributeError, pygame.error):
        pass
    pygame.display.set_caption("YouTube音ゲー (pygame) — ループバック / 重ね表示")
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    try_set_window_see_through()

    clock = pygame.time.Clock()
    font_title, font, font_small = make_ui_fonts()
    Tu = TITLE_SCREEN_UI_SCALE
    title_scr_title = make_font_at_px(max(14, int(round(26 * DISPLAY_SCALE * Tu))))
    title_scr_font = make_font_at_px(max(12, int(round(19 * DISPLAY_SCALE * Tu))))
    title_scr_small = make_font_at_px(max(9, int(round(15 * DISPLAY_SCALE * Tu))))

    root = Path(__file__).resolve().parent
    se_glass_good = load_optional_sound_wav(root, "glass1.wav")
    se_glass_great = load_optional_sound_wav(root, "glass2.wav")
    display_settings = load_display_settings(root)
    default_hud_right_px = max(8, int(round(18 * DISPLAY_SCALE)))
    hr_px = display_settings.hud_right_font_px
    if hr_px is not None:
        hr_px = max(6, min(120, hr_px))
    else:
        hr_px = default_hud_right_px
    font_hud_right = make_font_at_px(hr_px)
    note_label_sz = max(10, int(round(14 * DISPLAY_SCALE)))
    fp = match_ui_gothic_font_path()
    if fp:
        note_label_font = pygame.font.Font(fp, note_label_sz)
    else:
        note_label_font = pygame.font.SysFont(None, note_label_sz)
    judge_lbl_sz = max(22, int(round(40 * DISPLAY_SCALE)))
    if fp:
        judgment_font = pygame.font.Font(fp, judge_lbl_sz)
    else:
        judgment_font = pygame.font.SysFont(None, judge_lbl_sz)
    if display_settings.hud_right_line_gap_px is not None:
        hud_right_line_gap = max(8, min(100, display_settings.hud_right_line_gap_px))
    else:
        hud_right_line_gap = max(10, font_hud_right.get_linesize())

    time_offset_sec = load_time_offset_sec(root)
    youtube_url_candidates = load_youtube_url_candidates(root)
    selected_youtube_i = 0

    notes: list[Note] = []
    session_started = False
    session_t0_perf: float | None = None
    sensitivity, lane_spread, min_volume = load_game_tune(root)
    audio_analyzer: LiveAudioAnalyzer | None = None
    audio_ok = False
    has_sounddevice = sounddevice_available()

    def game_time_s() -> float:
        if session_t0_perf is None:
            return 0.0
        return (time.perf_counter() - session_t0_perf) + time_offset_sec

    def start_session() -> None:
        nonlocal session_started, session_t0_perf, audio_ok, audio_analyzer
        if session_started:
            return
        session_started = True
        session_t0_perf = time.perf_counter()
        audio_analyzer = LiveAudioAnalyzer(
            root,
            note_lead_sec=APPROACH_SECONDS,
            rms_mul=sensitivity,
            min_volume=min_volume,
        )
        audio_ok = audio_analyzer.start(session_t0_perf)
        if not audio_ok:
            print(audio_analyzer.get_hud().get("error", "audio"), file=sys.stderr)
        webbrowser.open(youtube_url_candidates[selected_youtube_i])

    bw = max(108, int(round(238 * DISPLAY_SCALE * Tu)))
    bh = max(28, int(round(40 * DISPLAY_SCALE * Tu)))
    btn_bot = int(round(56 * DISPLAY_SCALE * Tu))
    start_btn = pygame.Rect(WIDTH // 2 - bw // 2, HEIGHT - btn_bot - bh, bw, bh)

    score = 0
    combo = 0
    last_judgment = ""
    running = True
    title_layout: TitleScreenLayout | None = None
    preview_queue: queue.Queue[tuple[str, str | None, bytes | None]] = queue.Queue()
    preview_cache: dict[str, tuple[str | None, pygame.Surface | None]] = {}
    preview_fetching: set[str] = set()
    last_preview_requested = ""

    def sync_tune_file_and_audio() -> None:
        save_game_tune(root, sensitivity, lane_spread, min_volume)
        if audio_analyzer is not None:
            audio_analyzer.set_reactive_tune(sensitivity, min_volume)

    def kick_preview(url: str) -> None:
        if url in preview_cache or url in preview_fetching:
            return
        preview_fetching.add(url)

        def work() -> None:
            tit, raw = fetch_youtube_preview(url)
            preview_queue.put((url, tit, raw))

        threading.Thread(target=work, daemon=True).start()

    def drain_previews(mw: int, mh: int) -> None:
        while True:
            try:
                u, tit, raw = preview_queue.get_nowait()
            except queue.Empty:
                break
            preview_fetching.discard(u)
            surf = None
            if raw:
                try:
                    surf = pygame.image.load(io.BytesIO(raw)).convert_alpha()
                    surf = scale_surface_to_fit(surf, mw, mh)
                except (pygame.error, ValueError, TypeError):
                    surf = None
            preview_cache[u] = (tit, surf)

    digit_to_slot = {
        pygame.K_1: 0,
        pygame.K_2: 1,
        pygame.K_3: 2,
        pygame.K_4: 3,
        pygame.K_5: 4,
        pygame.K_6: 5,
        pygame.K_7: 6,
        pygame.K_8: 7,
        pygame.K_9: 8,
        pygame.K_0: 9,
        pygame.K_KP1: 0,
        pygame.K_KP2: 1,
        pygame.K_KP3: 2,
        pygame.K_KP4: 3,
        pygame.K_KP5: 4,
        pygame.K_KP6: 5,
        pygame.K_KP7: 6,
        pygame.K_KP8: 7,
        pygame.K_KP9: 8,
        pygame.K_KP0: 9,
    }

    while running:
        Tu_tune = TITLE_SCREEN_UI_SCALE
        audio_tune_layout = compute_audio_tune_layout(
            HEIGHT,
            WIDTH,
            DISPLAY_SCALE,
            Tu_tune,
            title_scr_small,
            sensitivity,
            lane_spread,
            min_volume,
        )
        if not session_started:
            title_layout = compute_title_screen_layout(
                WIDTH,
                DISPLAY_SCALE,
                TITLE_SCREEN_UI_SCALE,
                title_scr_title,
                title_scr_small,
                youtube_url_candidates,
            )
        else:
            title_layout = None
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif not session_started:
                    if event.key in (pygame.K_UP, pygame.K_LEFT):
                        selected_youtube_i = max(0, selected_youtube_i - 1)
                    elif event.key in (pygame.K_DOWN, pygame.K_RIGHT):
                        selected_youtube_i = min(
                            len(youtube_url_candidates) - 1, selected_youtube_i + 1
                        )
                    elif event.key in (
                        pygame.K_SPACE,
                        pygame.K_RETURN,
                        pygame.K_KP_ENTER,
                    ):
                        start_session()
                    else:
                        slot_i = digit_to_slot.get(event.key)
                        if slot_i is not None and slot_i < len(youtube_url_candidates):
                            selected_youtube_i = slot_i
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
                            best.judgment = tier
                            best.judge_time = now
                            best.flash = 1.0
                            score += SCORE_TABLE[tier]
                            combo += 1
                            last_judgment = tier.upper()
                            if tier == "good" and se_glass_good is not None:
                                se_glass_good.play()
                            elif tier in ("great", "perfect") and se_glass_great is not None:
                                se_glass_great.play()
                        else:
                            best.judgment = "miss"
                            best.judge_time = now
                            last_judgment = "MISS"
                            combo = 0
            elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                pos = _pointer_pos_for_mouse_event(event)
                ns, nsp, nv, tune_hit = apply_audio_tune_click(
                    pos,
                    audio_tune_layout,
                    sensitivity,
                    lane_spread,
                    min_volume,
                    min_vol_shift=bool(pygame.key.get_mods() & pygame.KMOD_SHIFT),
                )
                if tune_hit:
                    sensitivity, lane_spread, min_volume = ns, nsp, nv
                    sync_tune_file_and_audio()
                    continue
                if not session_started and title_layout is not None:
                    picked = False
                    for idx, rr in title_layout.url_pick_rects:
                        if rr.collidepoint(pos):
                            selected_youtube_i = idx
                            picked = True
                            break
                    if not picked:
                        n_urls = len(youtube_url_candidates)
                        if title_layout.sel_up_btn.collidepoint(pos):
                            selected_youtube_i = max(0, selected_youtube_i - 1)
                            picked = True
                        elif title_layout.sel_down_btn.collidepoint(pos):
                            selected_youtube_i = min(n_urls - 1, selected_youtube_i + 1)
                            picked = True
                    if not picked and start_btn.collidepoint(pos):
                        start_session()

        if not session_started:
            assert title_layout is not None
            L = title_layout
            Tu = L.Tu
            screen.fill((12, 14, 22))
            overlay = pygame.Surface((WIDTH, HEIGHT), pygame.SRCALPHA)
            overlay.blit(L.main_title_surf, L.mt_rect)
            content_y = L.content_y
            prev_px = L.prev_px
            prev_py = L.prev_py
            prev_tw = L.prev_tw
            prev_th = L.prev_th
            prev_pad = L.prev_pad
            bx = L.bx
            panel_w = L.panel_w

            cur_u = youtube_url_candidates[selected_youtube_i]
            if cur_u != last_preview_requested:
                last_preview_requested = cur_u
                kick_preview(cur_u)
            drain_previews(prev_tw, prev_th)

            pr = pygame.Rect(
                prev_px - prev_pad,
                prev_py - prev_pad,
                prev_tw + 2 * prev_pad,
                prev_th + 2 * prev_pad,
            )
            br_prev = max(3, int(round(6 * DISPLAY_SCALE * Tu)))
            pygame.draw.rect(overlay, (18, 20, 34, 235), pr, border_radius=br_prev)
            pygame.draw.rect(
                overlay,
                (55, 65, 95, 200),
                pr,
                width=max(1, int(round(DISPLAY_SCALE * Tu))),
                border_radius=br_prev,
            )
            pt, psurf = preview_cache.get(cur_u, (None, None))
            prev_loading = cur_u in preview_fetching and cur_u not in preview_cache
            if prev_loading:
                wait_t = title_scr_small.render("読込中…", True, (170, 180, 200))
                overlay.blit(
                    wait_t,
                    (
                        prev_px + (prev_tw - wait_t.get_width()) // 2,
                        prev_py + (prev_th - wait_t.get_height()) // 2,
                    ),
                )
            elif psurf is not None:
                sx = prev_px + (prev_tw - psurf.get_width()) // 2
                sy = prev_py + (prev_th - psurf.get_height()) // 2
                overlay.blit(psurf, (sx, sy))
            elif cur_u in preview_cache:
                na = title_scr_small.render("サムネなし", True, (140, 150, 170))
                overlay.blit(
                    na,
                    (
                        prev_px + (prev_tw - na.get_width()) // 2,
                        prev_py + (prev_th - na.get_height()) // 2,
                    ),
                )

            title_line_y = prev_py + prev_th + prev_pad + int(round(5 * DISPLAY_SCALE * Tu))
            title_max_w = max(prev_tw + int(round(24 * DISPLAY_SCALE * Tu)), 100)
            if pt:
                overlay.blit(
                    truncate_surface_width(
                        title_scr_small, pt, (228, 232, 248), title_max_w
                    ),
                    (prev_px, title_line_y),
                )
            elif cur_u in preview_cache and not prev_loading:
                overlay.blit(
                    title_scr_small.render("（タイトル取得不可）", True, (145, 150, 170)),
                    (prev_px, title_line_y),
                )

            overlay.blit(L.list_hdr_surf, L.list_hdr_pos)
            br_btn = max(2, int(round(5 * DISPLAY_SCALE * Tu)))
            bw_ln = max(1, int(round(DISPLAY_SCALE * Tu)))
            for brect, label, hot in (
                (L.sel_up_btn, "▲", (95, 150, 230)),
                (L.sel_down_btn, "▼", (95, 150, 230)),
            ):
                pygame.draw.rect(overlay, (*hot, 220), brect, border_radius=br_btn)
                pygame.draw.rect(
                    overlay, (220, 230, 255, 200), brect, width=bw_ln, border_radius=br_btn
                )
                glyph = title_scr_small.render(label, True, (255, 255, 255))
                gr = glyph.get_rect(center=brect.center)
                overlay.blit(glyph, gr)
            for i, rr in L.url_pick_rects:
                u = youtube_url_candidates[i]
                uy = rr.top + 2
                num_s = f"{i + 1}."
                tail = u if len(u) <= L.url_ch else u[: L.url_ch - 1] + "…"
                sel = i == selected_youtube_i
                bgc = (55, 80, 140, 210) if sel else (26, 30, 44, 130)
                pygame.draw.rect(
                    overlay,
                    bgc,
                    rr,
                    border_radius=max(2, int(round(4 * DISPLAY_SCALE * Tu))),
                )
                col = (255, 248, 230) if sel else (175, 180, 200)
                row_txt = title_scr_small.render(f"{num_s} {tail}", True, col)
                overlay.blit(row_txt, (bx + L.tx_in, uy))

            btn_r = max(2, int(round(6 * DISPLAY_SCALE * Tu)))
            btn_w = max(1, int(round(DISPLAY_SCALE * Tu)))
            pygame.draw.rect(
                overlay, (80, 120, 220, 200), start_btn, border_radius=btn_r
            )
            pygame.draw.rect(
                overlay, (220, 235, 255, 230), start_btn, width=btn_w, border_radius=btn_r
            )
            lbl = title_scr_font.render("スタート", True, (255, 255, 255))
            lr = lbl.get_rect(center=start_btn.center)
            overlay.blit(lbl, lr)

            hint_y = start_btn.top - int(round(4 * DISPLAY_SCALE * Tu))
            sub = title_scr_small.render(
                "Space / Enter / クリックで開始 ・ ループバック・帯域レーン",
                True,
                (165, 170, 188),
            )
            overlay.blit(sub, sub.get_rect(midbottom=(WIDTH // 2, hint_y)))
            if not has_sounddevice:
                wy = hint_y - sub.get_height() - int(round(4 * DISPLAY_SCALE * Tu))
                w1 = title_scr_small.render(
                    "! sounddevice 未インストール → pip install sounddevice",
                    True,
                    (255, 140, 120),
                )
                overlay.blit(w1, w1.get_rect(midbottom=(WIDTH // 2, wy)))
            draw_audio_tune_panel(
                overlay,
                audio_tune_layout,
                sensitivity,
                lane_spread,
                min_volume,
                title_scr_small,
                DISPLAY_SCALE,
                Tu,
                backing=False,
            )

            esc = title_scr_small.render("ESC 終了", True, (120, 125, 140))
            margin = int(round(8 * DISPLAY_SCALE * Tu))
            esc_y = HEIGHT - int(round(18 * DISPLAY_SCALE * Tu))
            overlay.blit(esc, (margin, esc_y))
            screen.blit(overlay, (0, 0))
            pygame.display.flip()
            clock.tick(FPS)
            continue

        now = game_time_s()
        assert audio_analyzer is not None

        dt_frame = max(1e-4, clock.get_time() / 1000.0)
        for n in notes:
            if n.flash > 0:
                n.flash = max(0.0, n.flash - dt_frame * FLASH_DECAY_PER_S)

        while True:
            try:
                ev = audio_analyzer.note_queue.get_nowait()
                tr: tuple[int, int, int] | None = None
                if all(k in ev for k in ("r", "g", "b")):
                    try:
                        tr = (int(ev["r"]), int(ev["g"]), int(ev["b"]))
                    except (TypeError, ValueError):
                        tr = None
                notes.append(
                    Note(
                        time=float(ev["hit_time"]),
                        lane=int(ev["lane"]),
                        done=False,
                        rgb=tr,
                        label=str(ev.get("label", "") or ""),
                    )
                )
            except queue.Empty:
                break

        audio_analyzer.maybe_flush_json(time.perf_counter())

        for n in notes:
            if n.done:
                continue
            if now > n.time + GOOD_WIN:
                n.done = True
                if n.judgment is None:
                    n.judgment = "miss"
                    n.judge_time = now
                score += SCORE_TABLE["miss"]
                combo = 0
                last_judgment = "MISS"

        screen.fill((8, 10, 18))
        overlay = pygame.Surface((WIDTH, HEIGHT), pygame.SRCALPHA)
        lanes_x = lanes_x_from_spread(lane_spread)
        draw_3d_stage(overlay, lanes_x)

        drawable: list[tuple[float, Note]] = []
        for n in notes:
            if n.done:
                continue
            z = note_z_visible(n.time, now)
            if z is None:
                continue
            drawable.append((z, n))
        drawable.sort(key=lambda t: t[0], reverse=True)

        min_lbl = int(round(22 * DISPLAY_SCALE))
        for z, n in drawable:
            lx = lanes_x[n.lane]
            sx, sy = project_xy(lx, z)
            size = note_screen_size(z)
            rect = pygame.Rect(int(sx - size // 2), int(sy - size // 2), size, size)
            br = max(2, min(8, size // 8))
            if n.rgb:
                fr, fg, fb = n.rgb
                er = min(255, fr + 55)
                eg = min(255, fg + 55)
                eb = min(255, fb + 55)
            else:
                fr, fg, fb = 0, 255, 255
                er, eg, eb = 200, 255, 255
            draw_alpha_round_rect(
                overlay,
                rect,
                (fr, fg, fb, ALPHA_NOTE_FILL),
                (er, eg, eb, ALPHA_NOTE_EDGE),
                max(1, size // 24),
                br,
            )
            if n.label and size >= min_lbl:
                lbl_s = note_label_font.render(n.label, True, (252, 252, 255))
                overlay.blit(lbl_s, lbl_s.get_rect(center=rect.center))

        screen.blit(overlay, (0, 0))

        fx_layer = pygame.Surface((WIDTH, HEIGHT), pygame.SRCALPHA)
        for n in notes:
            if n.flash <= 0:
                continue
            lx_j = lanes_x[n.lane]
            jx, jy = project_xy(lx_j, Z_END)
            ring_r = max(
                3, int(JUDGE_FLASH_RING_BASE * DISPLAY_SCALE * n.flash)
            )
            alpha_ring = int(max(0, min(255, 230 * n.flash)))
            pygame.draw.circle(
                fx_layer,
                (255, 255, 200, alpha_ring),
                (int(jx), int(jy)),
                ring_r,
                width=max(2, int(round(2 * DISPLAY_SCALE))),
            )
        screen.blit(fx_layer, (0, 0))

        for n in notes:
            if not n.judgment:
                continue
            t_j = now - n.judge_time
            if t_j < 0 or t_j >= JUDGE_FLOAT_DURATION:
                continue
            lx_j = lanes_x[n.lane]
            jx, jy = project_xy(lx_j, Z_END)
            y_off = -t_j * JUDGE_RISE_SPEED
            alpha_t = max(0, int(255 * (1.0 - t_j / JUDGE_FLOAT_DURATION)))
            col_j = judgment_text_color(n.judgment)
            label_j = n.judgment.upper()
            txt_j = judgment_font.render(label_j, True, col_j)
            txt_j.set_alpha(alpha_t)
            tr_j = txt_j.get_rect(center=(int(jx), int(jy + y_off)))
            screen.blit(txt_j, tr_j)

        hud_r = audio_analyzer.get_hud()
        hud = [f"Score: {score}", f"Combo: {combo}"]
        if last_judgment:
            hud.append(f"Last: {last_judgment}")
        margin = int(round(12 * DISPLAY_SCALE))
        y0 = int(round(8 * DISPLAY_SCALE))
        line_gap = int(round(26 * DISPLAY_SCALE))
        h_max = int(round(48 * DISPLAY_SCALE))
        spec_gap_top = int(round(6 * DISPLAY_SCALE))
        bar_w = int(round(13 * DISPLAY_SCALE))
        bar_gap = int(round(6 * DISPLAY_SCALE))
        spec_y = y0 + len(hud) * line_gap + spec_gap_top
        hud_h = spec_y + h_max + int(round(12 * DISPLAY_SCALE))

        hud_overlay = pygame.Surface((WIDTH, hud_h), pygame.SRCALPHA)
        hud_overlay.fill((10, 12, 20, 140))
        screen.blit(hud_overlay, (0, 0))
        for i, line in enumerate(hud):
            surf = font.render(line, True, (240, 240, 245))
            screen.blit(surf, (margin, y0 + i * line_gap))

        sl = hud_r.get("spectrum")
        if isinstance(sl, list) and len(sl) >= 4:
            spec4 = [max(0.0, min(1.0, float(sl[j]))) for j in range(4)]
        else:
            spec4 = [0.0, 0.0, 0.0, 0.0]
        for i in range(4):
            lv = spec4[i]
            br, bg, bb = NOTE_RGB[i]
            cr = min(255, int(br + (255 - br) * lv * 0.92))
            cg = min(255, int(bg + (255 - bg) * lv * 0.88))
            cb = min(255, int(bb + (255 - bb) * lv * 0.88))
            bh = max(2, int(h_max * lv))
            xb = margin + i * (bar_w + bar_gap)
            yb = spec_y + (h_max - bh)
            rc = pygame.Rect(xb, yb, bar_w, bh)
            brad = max(2, bar_w // 4)
            pygame.draw.rect(screen, (cr, cg, cb), rc, border_radius=brad)
            pygame.draw.rect(
                screen,
                (min(255, cr + 32), min(255, cg + 32), min(255, cb + 24)),
                rc,
                width=max(1, int(round(DISPLAY_SCALE))),
                border_radius=brad,
            )

        hint = font_small.render(
            "D F J K / ESC終了 ・ 左下: TIMING / POSITION / CONTROL",
            True,
            (160, 165, 185),
        )
        screen.blit(hint, (margin, HEIGHT - int(round(28 * DISPLAY_SCALE))))

        # 右上: 取り込み音量・リズム指標
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
        r_top = int(round(8 * DISPLAY_SCALE))
        for i, line in enumerate(lines_r):
            surf = font_hud_right.render(line, True, (200, 210, 230))
            screen.blit(surf, (rx, r_top + i * hud_right_line_gap))

        tune_layer = pygame.Surface((WIDTH, HEIGHT), pygame.SRCALPHA)
        draw_audio_tune_panel(
            tune_layer,
            audio_tune_layout,
            sensitivity,
            lane_spread,
            min_volume,
            title_scr_small,
            DISPLAY_SCALE,
            TITLE_SCREEN_UI_SCALE,
            backing=True,
        )
        screen.blit(tune_layer, (0, 0))

        pygame.display.flip()
        clock.tick(FPS)

    if audio_analyzer is not None:
        audio_analyzer.stop()
    save_game_tune(root, sensitivity, lane_spread, min_volume)
    pygame.quit()
    return (sensitivity, lane_spread, min_volume)


if __name__ == "__main__":
    main()
