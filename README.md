# faster-whisper-pyannote

faster-whisperとpyannoteを使って、文字起こしと話者識別を行うツール

## 実行方法

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