"""
指定 YouTube URL から音声を取得し mp3 を出力する（yt-dlp + ffmpeg）。
仕様書の動画: https://www.youtube.com/watch?v=5tc14WHUoMw
"""

from pathlib import Path

import yt_dlp

URL = "https://www.youtube.com/watch?v=5tc14WHUoMw&list=RD5tc14WHUoMw&start_radio=1"
OUT_DIR = Path(__file__).resolve().parent
OUT_MP3 = OUT_DIR / "music.mp3"


def main() -> None:
    opts: dict = {
        "format": "bestaudio/best",
        "outtmpl": str(OUT_DIR / "music.%(ext)s"),
        "quiet": False,
        "no_warnings": False,
        "postprocessors": [
            {
                "key": "FFmpegExtractAudio",
                "preferredcodec": "mp3",
                "preferredquality": "192",
            }
        ],
        "overwrite_downloads": True,
    }
    with yt_dlp.YoutubeDL(opts) as ydl:
        ydl.download([URL])
    print(f"完了: {OUT_MP3} （ffmpeg が PATH に必要です）")


if __name__ == "__main__":
    main()
