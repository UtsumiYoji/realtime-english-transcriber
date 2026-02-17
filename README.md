# Realtime English Transcriber

Windows デスクトップ上の英語音声をリアルタイムで文字起こし・日本語翻訳するツール。

## 主な機能

- **音声キャプチャ:** WASAPI Loopback でシステム音声（Chrome, Teams, Zoom等）をキャプチャ
- **リアルタイム文字起こし:** faster-whisper (CPU, INT8量子化) で英語音声をテキスト化
- **日本語翻訳:** DeepL Free API で英語→日本語翻訳（ON/OFF切替可能）
- **ファイル保存:** トランスクリプトを .txt で保存（手動/自動保存）

## クイックスタート

```bash
# 1. Python 仮想環境を作成
python -m venv .venv
.venv\Scripts\activate

# 2. 依存パッケージをインストール
pip install -r requirements.txt

# 3. 設定ファイルを編集 (DeepL API キーを設定)
# config.yaml の deepl_api_key を設定

# 4. アプリ起動
python -m src.main
```

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
