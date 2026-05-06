# YouTube 音ゲー (MP3 事前解析版)

ブラウザで YouTube を再生しながら、`pygame` の別ウィンドウで譜面をプレイするリズムゲームです。  
ノーツはリアルタイム音声解析ではなく、**MP3 を事前解析して生成した譜面 JSON** を使用します。

## 構成

```text
project/
  youtubeotoge.py
  generate_chart_from_mp3.py
  song_db.json
  charts/
    <youtube_id>.json
  mp3/
    01 - song.mp3
```

## 必要環境

- Python 3.10+
- 依存インストール:

```powershell
pip install -r requirements.txt
```

## 1) 事前に譜面を作る

### 自動紐づけ（推奨）

`mp3/` に `01-`〜`99-` で始まる MP3 があり、`youtube_url.txt` に URL を行ごとに並べている場合:

```powershell
python generate_chart_from_mp3.py
```

- 例: `13 - はるのとなり.mp3` ⇔ `youtube_url.txt` 13行目
- 未連携（`song_db.json` 未登録）の番号を優先して自動紐づけ

### 番号指定

```powershell
python generate_chart_from_mp3.py --slot 13 --yes
```

### MP3 / URL 明示指定

```powershell
python generate_chart_from_mp3.py --mp3 "mp3/13 - はるのとなり.mp3" --youtube "https://www.youtube.com/watch?v=qJ-Kx7IKYEA"
```

## 2) offset と長さ差チェック

- `duration_sec` 差が 2.0 秒超のとき確認:
  - 「曲の長さが違いますが、jsonによりnoteを作成しますか？」
- `offset` は既定で自動推定（BPM/拍 + 先頭無音ベース）
- 推定が 10 秒超なら中断:
  - 「長くかかるので、自分で調べてね」

主なオプション:

- `--yes` : 長さ差確認をスキップ
- `--manual-offset` : offset を手入力
- `--offset -0.25` : offset を直接指定

## 3) ゲーム起動

```powershell
python youtubeotoge.py
```

起動後の流れ:

1. タイトル画面で YouTube URL を選択
2. URL から `youtube_id` を抽出
3. `song_db.json` から曲情報を取得
4. `charts/<youtube_id>.json` を読み込み
5. `note_time = chart_time + offset` で同期して再生

キー操作:

- レーン: `D F J K`
- 終了: `ESC`

## データ形式

### `song_db.json`

```json
{
  "songs": [
    {
      "youtube_id": "string",
      "title": "string",
      "youtube_title": "string",
      "duration_sec": 123.456,
      "mp3": "filename.mp3",
      "offset": 0.0
    }
  ]
}
```

### `charts/<youtube_id>.json`

- `youtube_id`
- `bpm`
- `duration_sec`
- `notes`: `[{ "time": float, "lane": 0..3 }, ...]`

## トラブルシュート

- `song_db.json に youtube_id=... がありません`
  - 先に `generate_chart_from_mp3.py` でその曲を生成してください
- `charts/<id>.json を読み込めません`
  - 該当ファイルが `charts/` にあるか確認してください
- `ImportError`（`pygame` / `numpy` / `librosa`）
  - `pip install -r requirements.txt`

