"""
YouTube oEmbed でタイトル・サムネイル URL を取得し、サムネをバイトで返す（ネットワークのみ）。
画像の pygame 化はメインスレッドで行う。
"""

from __future__ import annotations

import json
import urllib.error
import urllib.parse
import urllib.request

USER_AGENT = "Mozilla/5.0 (Windows NT 10.0; rv:109.0) Gecko/20100101 Firefox/115.0 (youtubeotoge)"


def fetch_youtube_preview(url: str) -> tuple[str | None, bytes | None]:
    """
    同期。戻り値: (title, image_bytes) いずれも失敗時 None。
    YouTube 以外・oEmbed 非対応の URL では (None, None)。
    """
    page = url.strip()
    if not page.startswith("http"):
        return None, None
    try:
        q = urllib.parse.quote(page, safe="")
        oembed = f"https://www.youtube.com/oembed?url={q}&format=json"
        req = urllib.request.Request(oembed, headers={"User-Agent": USER_AGENT})
        with urllib.request.urlopen(req, timeout=12) as r:
            j = json.loads(r.read().decode("utf-8", errors="replace"))
        title = j.get("title")
        if isinstance(title, str):
            t = title.strip() or None
        else:
            t = None
        turl = j.get("thumbnail_url")
        if not turl or not isinstance(turl, str):
            return t, None
        req2 = urllib.request.Request(turl.strip(), headers={"User-Agent": USER_AGENT})
        with urllib.request.urlopen(req2, timeout=15) as r2:
            data = r2.read()
        if not data:
            return t, None
        return t, data
    except (urllib.error.URLError, urllib.error.HTTPError, OSError, ValueError, TypeError):
        return None, None
