# faster-whisper-pyannote

faster-whisperとpyannoteを使って、文字起こしと話者識別を行うツール

## 実行方法

### uv

リポジトリのルートで依存関係を同期します。

```
uv sync
```

仮想環境を有効化します。

Windows（PowerShell）:

```
.\.venv\Scripts\Activate.ps1
```

macOS:

```
source .venv/bin/activate
```

`src` に移動して `transcript.py` を実行します（相対パスはこのディレクトリ基準です）。

```
cd src
python transcript.py
```

### Docker

まずはDockerイメージをビルドします。

```
docker build -t transcribe-ai .
```

次にコンテナを立ち上げます。

```
docker compose up -d
```

実行
```
docker exec -it transcribe_ai bash
python transcript.py
```

終了
```
exit
docker compose down
```