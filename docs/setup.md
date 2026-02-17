# セットアップ手順

## 前提条件

- **OS:** Windows 10 / 11
- **Python:** 3.10 以上
- **CPU:** AVX2 命令セット対応（2015年以降のCPUなら基本対応）
- **メモリ:** 4GB 以上推奨（Whisper モデル + PyTorch で約 2GB 使用）
- **ネットワーク:** DeepL API 利用時に必要（文字起こしのみならオフライン可）

## 1. Python 環境のセットアップ

```bash
# プロジェクトディレクトリに移動
cd realtime-english

# 仮想環境を作成
python -m venv .venv

# 仮想環境を有効化
.venv\Scripts\activate

# pip を最新に更新
pip install --upgrade pip
```

## 2. 依存パッケージのインストール

```bash
pip install -r requirements.txt
```

> **注意:** PyTorch は約 2GB あるため、初回インストールには時間がかかります。
> CPU 版のみで十分です（CUDA 版は不要）。

### インストールに問題がある場合

```bash
# PyAudioWPatch のビルドに失敗する場合
pip install PyAudioWPatch --no-build-isolation

# torch を CPU 版で明示的にインストール
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

## 3. DeepL API キーの取得（翻訳機能を使う場合）

1. [DeepL API](https://www.deepl.com/pro#developer) にアクセス
2. 「無料で登録する」からアカウントを作成
   - **注意:** 無料プランでもクレジットカードの登録が必要です（課金はされません）
3. ログイン後、「アカウント」→「APIキー」からキーをコピー
4. 以下のいずれかの方法でキーを設定:

### 方法A: config.yaml に記載

```yaml
deepl_api_key: "your-api-key-here"
```

### 方法B: 環境変数で設定

```bash
# PowerShell
$env:DEEPL_API_KEY = "your-api-key-here"

# Command Prompt
set DEEPL_API_KEY=your-api-key-here
```

> **翻訳なしでも使えます:** DeepL API キーを設定しなくても、文字起こし機能のみで使用可能です。

## 4. アプリ別音声キャプチャのセットアップ（任意）

デフォルトではシステム全体の音声をキャプチャします。
特定のアプリ（Chrome, Teams, Zoom）のみをキャプチャしたい場合は、
仮想オーディオケーブルを使用します。

### VB-Audio VB-CABLE のインストール

1. [VB-Audio VB-CABLE](https://vb-audio.com/Cable/) にアクセス
2. 「Download」からインストーラをダウンロード・実行
3. PC を再起動

### 使い方

1. キャプチャしたいアプリ（例: Chrome）のオーディオ出力を VB-CABLE に変更
   - **Windows 11:** 設定 → システム → サウンド → 音量ミキサー → アプリごとに出力デバイスを変更
   - **Windows 10:** 設定 → システム → サウンド → アプリの音量とデバイスの設定
2. 本アプリで「CABLE Output (VB-Audio Virtual Cable)」を選択
3. 「Start」でキャプチャ開始

> **ヒント:** VB-CABLE を使用すると、そのアプリの音声はスピーカーから聞こえなくなります。
> 音声を聞きながらキャプチャしたい場合は、[VoiceMeeter](https://vb-audio.com/Voicemeeter/) の
> 使用を検討してください（無料）。

## 5. アプリの起動

```bash
# 仮想環境を有効化
.venv\Scripts\activate

# アプリを起動
python -m src.main
```

### 初回起動時の注意

- **Whisper モデルのダウンロード:** 初回の文字起こし時に Hugging Face からモデルがダウンロードされます（base.en: 約 150MB）。ダウンロード中はコンソールに進捗が表示されます。
- **Silero VAD モデル:** 初回使用時に PyTorch Hub からダウンロードされます。

## 6. 設定 (config.yaml)

```yaml
# DeepL API キー
deepl_api_key: ""

# Whisper モデルサイズ (tiny.en / base.en / small.en)
# tiny.en: 高速・低精度 (75MB) - 低スペックPC向け
# base.en: バランス (150MB) - 推奨
# small.en: 高精度・低速 (500MB) - 高スペックPC向け
whisper_model: "base.en"

# 推論の量子化タイプ (int8 / float32)
# int8: 高速、推奨
# float32: より正確だが遅い
compute_type: "int8"

# デフォルトで翻訳を有効にするか
translation_enabled: true

# 自動保存先パス (空 = 無効)
auto_save_path: ""

# デフォルトのオーディオデバイス名 (部分一致)
default_device: ""

# VAD 感度 (0.0-1.0, 高い = 厳しい)
vad_threshold: 0.5

# 発話の最大長 (秒)
max_speech_duration: 30.0

# 処理する最小発話長 (ミリ秒)
min_speech_ms: 250
```

## トラブルシューティング

### 「No audio devices found」と表示される

- WASAPI 対応のオーディオデバイスが必要です
- Windows のサウンド設定で再生デバイスが有効になっているか確認してください
- PyAudioWPatch が正しくインストールされているか確認: `pip show PyAudioWPatch`

### 文字起こしが遅い

- `config.yaml` の `whisper_model` を `tiny.en` に変更してみてください
- `compute_type` が `int8` になっていることを確認

### 無音なのにテキストが生成される

- `config.yaml` の `vad_threshold` を `0.6` ～ `0.8` に上げてみてください

### DeepL API エラー

- API キーが正しいか確認
- DeepL Free API の月間上限（50万字）に達していないか確認
- ネットワーク接続を確認
