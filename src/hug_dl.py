from huggingface_hub import snapshot_download

# ローカルの指定したフォルダに全ファイルをダウンロード
snapshot_download(
    repo_id="deepdml/faster-whisper-large-v3-turbo-ct2",
    local_dir="turbo_offline",
)
