"""
YouTube の動画から音声を抽出し、最初の30秒だけを WAV で保存する。
"""

import subprocess
import sys
import tempfile
from pathlib import Path


def youtube_to_wav(
    url: str = "https://www.youtube.com/watch?v=w_YsDPxo_Qk&t=14s",
    output_path: str = "output.wav",
    duration_sec: int = 30,
) -> str:
    """
    YouTube URL から音声を取得し、指定秒数だけ WAV で保存する。

    Args:
        url: YouTube の動画URL
        output_path: 出力 WAV ファイルのパス
        duration_sec: 抽出する長さ（秒）。デフォルト 30 秒。

    Returns:
        保存した WAV ファイルのパス
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        # 音声のみダウンロード（WAV で取得すると重いので一時的に m4a 等で取得してから変換する場合もある）
        # yt-dlp: 最高品質の音声を取得し、ffmpeg で WAV に変換＋先頭 duration 秒だけ切り出し
        temp_audio = tmp / "audio.%(ext)s"

        subprocess.run(
            [
                sys.executable,
                "-m",
                "yt_dlp",
                "-x",
                "--audio-format",
                "wav",
                "--audio-quality",
                "0",
                "-o",
                str(temp_audio),
                "--no-playlist",
                "--quiet",
                "--no-warnings",
                url,
            ],
            check=True,
            capture_output=True,
        )

        # temp_audio は "audio.%(ext)s" なので実際のファイル名は audio.wav
        downloaded = tmp / "audio.wav"
        if not downloaded.exists():
            # フォールバック: 任意の拡張子
            candidates = list(tmp.glob("audio.*"))
            if not candidates:
                raise FileNotFoundError(
                    f"yt-dlp の出力ファイルが見つかりません: {tmpdir}"
                )
            downloaded = candidates[0]

        # 先頭 duration_sec 秒だけ切り出して出力
        subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-i",
                str(downloaded),
                "-t",
                str(duration_sec),
                "-acodec",
                "pcm_s16le",
                "-ar",
                "16000",
                "-ac",
                "1",
                str(output_path),
            ],
            check=True,
            capture_output=True,
        )

    return str(output_path)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(
            "Usage: python youtube_to_wav.py <YouTube URL> [output.wav] [duration_sec]"
        )
        print(
            "Example: python youtube_to_wav.py "
            "https://www.youtube.com/watch?v=xxxx first30.wav 30"
        )
        sys.exit(1)

    url = sys.argv[1]
    out = sys.argv[2] if len(sys.argv) > 2 else "output.wav"
    duration = int(sys.argv[3]) if len(sys.argv) > 3 else 30

    path = youtube_to_wav(url, output_path=out, duration_sec=duration)
    print(f"Saved: {path} (first {duration} sec)")
