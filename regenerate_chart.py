"""
music.mp3 を基に chart.json を作り直す（generate_chart_from_audio.py のラッパー）。

  python regenerate_chart.py
  python regenerate_chart.py --audio music.mp3
  python regenerate_chart.py --audio my.wav --out chart.json --seed 42

追加の引数はそのまま generate_chart_from_audio.py に渡します。
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def main() -> None:
    root = Path(__file__).resolve().parent
    gen = root / "generate_chart_from_audio.py"
    if not gen.is_file():
        print("generate_chart_from_audio.py が見つかりません。", file=sys.stderr)
        sys.exit(1)
    cmd = [sys.executable, str(gen), *sys.argv[1:]]
    raise SystemExit(subprocess.call(cmd))


if __name__ == "__main__":
    main()
