# Realtime English Transcriber

Windows デスクトップ上の英語音声をリアルタイムで文字起こし・日本語翻訳するツール。

## 主な機能

- **音声キャプチャ:** WASAPI Loopback でシステム音声（Chrome, Teams, Zoom等）をキャプチャ
- **マイク入力:** マイクデバイスからの音声キャプチャにも対応（出力と同時使用可能）
- **リアルタイム文字起こし:** faster-whisper (CPU, INT8量子化) で英語音声をテキスト化
- **日本語文字起こし:** マルチリンガルモデル使用時、日本語音声の文字起こしを ON/OFF 切替可能
- **日本語翻訳:** DeepL Free API で英語→日本語翻訳（ON/OFF切替可能）
- **ファイル保存:** トランスクリプトを .txt で保存（手動/自動保存）
- **設定リロード:** アプリ再起動なしで config.yaml の変更を反映

## クイックスタート

```bash
# 1. Python 仮想環境を作成
python -m venv .venv
.venv\Scripts\activate

# 2. 依存パッケージをインストール
pip install -r requirements.txt

# 3. 設定ファイルを作成
copy config.yaml.example config.yaml

# 4. config.yaml を編集して DeepL API キーを設定
#    deepl_api_key: "your-api-key-here:fx"
#    API キーは https://www.deepl.com/pro#developer から取得

# 5. アプリ起動
python -m src.main
```

### Whisper モデルの切り替え

`config.yaml` の `whisper_model` で変更できます。

| モデル | 言語検出 | 速度 | 精度 | 用途 |
|---|---|---|---|---|
| `base` | あり（自動） | 普通 | 普通 | 英語以外の音声が混在する環境（デフォルト） |
| `base.en` | なし（英語固定） | 速い | 高い | 英語のみの環境 |
| `small` | あり（自動） | 遅い | 高い | 精度重視（CPU負荷高め） |
| `tiny` | あり（自動） | 最速 | 低い | 軽量・低スペック向け |

```yaml
# 例: 英語専用モデルに切り替え（日本語音声は英語として誤認識される）
whisper_model: "base.en"

# 例: マルチリンガルモデル（英語以外の音声を自動スキップ）
whisper_model: "base"
```

> **Note:** マルチリンガルモデル (`base`, `small` 等) は英語以外の音声を自動検出してスキップします。
> 英語専用モデル (`base.en` 等) は全音声を英語として解釈するため高精度ですが、日本語音声も英語として文字起こしされます。

### マイク入力の使い方

アプリ起動後、GUI 上部に「🔊 出力デバイス」と「🎤 マイクデバイス」の2つのドロップダウンが表示されます。

- **出力デバイスのみ使用:** マイクを「(無効)」に設定
- **マイクのみ使用:** 出力デバイスを「(無効)」に設定
- **両方同時使用:** それぞれデバイスを選択して「開始」

表示エリアでは、出力音声は 🔊 アイコン、マイク音声は 🎤 アイコンで区別され、配色も異なります。

`config.yaml` でデフォルトのマイクデバイスを指定できます:
```yaml
mic_device: "マイクデバイス名"
```

### 日本語文字起こしの切替

マルチリンガルモデル使用時、GUI の「JP文字起こし」チェックボックスで日本語音声の文字起こしを ON/OFF できます。

- **OFF（デフォルト）:** 従来通り英語以外の音声はスキップ
- **ON:** 日本語音声も `JP:` ラベル付きで表示（翻訳は行わず、そのまま表示）

```yaml
# config.yaml でデフォルト値を設定
japanese_transcription_enabled: false
```

> **Note:** 英語専用モデル (`base.en` 等) 使用時は言語検出ができないため、このチェックボックスは無効化されます。

### 設定リロード

`config.yaml` を編集した後、アプリを再起動せずに「⟳ 設定リロード」ボタンで変更を反映できます。

- **録音停止中のみ**リロード可能（録音中はボタンがグレーアウト）
- Whisper モデル変更時は次回録音開始時にモデルが再読み込みされます
- リロード後、変更内容がシステムメッセージとして表示されます

詳細なセットアップ手順は [docs/setup.md](docs/setup.md) を参照。

## 技術スタック

| Component | Technology |
|---|---|
| Language | Python 3.10+ |
| GUI | Tkinter |
| Audio Capture | PyAudioWPatch / sounddevice (WASAPI) |
| VAD | Silero VAD |
| STT | faster-whisper (CTranslate2) |
| Translation | DeepL Free API |
