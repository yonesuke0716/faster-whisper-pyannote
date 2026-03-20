# instantiate the pipeline
import json
import os
import tempfile
import time
from pathlib import Path
from types import SimpleNamespace

from faster_whisper import WhisperModel
from pyannote.audio import Pipeline
from pydub import AudioSegment
from pydub.silence import detect_nonsilent
import torchaudio

try:
    from moviepy import VideoFileClip, TextClip, CompositeVideoClip
except ImportError:
    from moviepy.editor import VideoFileClip, TextClip, CompositeVideoClip

# ゲート付きモデル（pyannote 等）用。必須: https://huggingface.co/settings/tokens で発行し、
# 各モデルページで「Agree and access repository」を実施すること
hf_token = os.getenv("HUGGING_FACE_TOKEN") or os.getenv("HF_TOKEN")

# 無音検知で分割するときのパラメータ（pydub detect_nonsilent）
MIN_SILENCE_LEN_MS = 500  # この長さの無音で区切る
SILENCE_THRESH_DB = -30  # これ以下を無音とみなす（dBFS）
WHISPER_SAMPLE_RATE = 16000


def split_audio_on_silence(audio_path: str) -> list[tuple[int, int]]:
    """
    pydub の無音検知で WAV を分割し、各セグメントの (開始ms, 終了ms) のリストを返す。
    """
    sound = AudioSegment.from_file(
        audio_path, format=Path(audio_path).suffix[1:] or "wav"
    )
    # detect_nonsilent には keep_silence 引数はない（split_on_silence のみ）
    segments_ms = detect_nonsilent(
        sound,
        min_silence_len=MIN_SILENCE_LEN_MS,
        silence_thresh=SILENCE_THRESH_DB,
    )
    return segments_ms


def get_speaker_for_interval(diarization, start_sec: float, end_sec: float) -> str:
    """区間 (start_sec, end_sec) で最も発話時間が長い話者を返す。"""
    overlap_by_speaker = {}
    for turn, _, speaker in diarization.itertracks(yield_label=True):
        overlap = max(0, min(end_sec, turn.end) - max(start_sec, turn.start))
        if overlap > 0:
            overlap_by_speaker[speaker] = overlap_by_speaker.get(speaker, 0) + overlap
    if not overlap_by_speaker:
        return "SPEAKER_00"
    return max(overlap_by_speaker, key=overlap_by_speaker.get)


def format_timestamp(sec: float) -> str:
    """秒を HH:MM:SS.mmm に変換。"""
    h = int(sec // 3600)
    m = int((sec % 3600) // 60)
    s = int(sec % 60)
    ms = int((sec - int(sec)) * 1000)
    return f"{h:02}:{m:02}:{s:02}.{ms:03}"


def concat_whisper_pyannote(segments, diarization, duration):
    data = []
    for index, _dict in enumerate(segments):
        start_time = _dict.start
        end_time = _dict.end
        text = _dict.text
        # WAV再生時間より超えたデータを除外
        if start_time > duration:
            break

        # 時、分、秒、ミリ秒に分割
        s_h, s_m, s_s = (
            int(start_time // 3600),
            int((start_time % 3600) // 60),
            int(start_time % 60),
        )
        e_h, e_m, e_s = (
            int(end_time // 3600),
            int((end_time % 3600) // 60),
            int(end_time % 60),
        )

        # ミリ秒を計算
        s_ms = int((start_time - int(start_time)) * 1000)
        e_ms = int((end_time - int(end_time)) * 1000)

        # 話者の割り当て
        for turn, _, speaker in diarization.itertracks(yield_label=True):
            if start_time > turn.end or end_time < turn.start:
                continue
            else:
                current_speaker = speaker

        data.append(
            {
                "id": index + 1,
                "start": f"{s_h:02}:{s_m:02}:{s_s:02}.{s_ms:03}",
                "end": f"{e_h:02}:{e_m:02}:{e_s:02}.{e_ms:03}",
                "text": text,
                "speaker": current_speaker,
            }
        )
    return data


def insert_subtitle(segments: list, mp4_path: str, duration: float):
    mp4_data = VideoFileClip(mp4_path)
    for _, _dict in enumerate(segments):
        start_time = _dict.start
        end_time = _dict.end
        text = _dict.text
        # WAV再生時間より超えたデータを除外
        if start_time > duration:
            break

        # フォントはコンテナに存在する DejaVu-Sans を指定（デフォルトの Courier は slim イメージにない）
        text_clip = TextClip(text, fontsize=70, font="DejaVu-Sans", color="white")
        # テキストの表示位置と時間を指定
        text_clip = (
            text_clip.set_position("bottom")
            .set_duration(end_time - start_time)
            .set_start(start_time)
        )

        # テキストを動画に重ねる
        final_clip = CompositeVideoClip([mp4_data, text_clip])

    # 字幕付き動画を書き出す
    final_clip.write_videofile("output.mp4")


def main(audio_file, mp4_file=None):
    # 無音検知で WAV を分割（pydub）
    segments_ms = split_audio_on_silence(audio_file)
    if not segments_ms:
        raise ValueError("無音検知でセグメントが得られませんでした。")

    sound = AudioSegment.from_file(
        audio_file, format=Path(audio_file).suffix[1:] or "wav"
    )
    waveform, sample_rate = torchaudio.load(audio_file)
    duration = waveform.shape[1] / sample_rate

    # 話者分離は全体に対して1回だけ
    pipeline = Pipeline.from_pretrained("pyannote_config.yaml")
    diarization = pipeline({"waveform": waveform, "sample_rate": sample_rate})

    # 分割ごとに文字起こし（faster_whisper）
    model = WhisperModel("./large-v3", device="cpu", compute_type="int8")

    results = []
    segment_objects = []  # insert_subtitle 用の .start / .end / .text オブジェクト

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        for idx, (start_ms, end_ms) in enumerate(segments_ms):
            start_sec = start_ms / 1000.0
            end_sec = end_ms / 1000.0
            if start_sec > duration:
                break

            # 該当区間を 16kHz モノで書き出し
            chunk = sound[start_ms:end_ms]
            chunk = chunk.set_frame_rate(WHISPER_SAMPLE_RATE).set_channels(1)
            chunk_path = tmpdir / f"chunk_{idx}.wav"
            chunk.export(str(chunk_path), format="wav")

            segs, _ = model.transcribe(str(chunk_path), vad_filter=False)
            text = " ".join(s.text.strip() for s in segs).strip()
            if not text:
                continue

            speaker = get_speaker_for_interval(diarization, start_sec, end_sec)
            results.append(
                {
                    "id": len(results) + 1,
                    "start": format_timestamp(start_sec),
                    "end": format_timestamp(end_sec),
                    "text": text,
                    "speaker": speaker,
                }
            )
            # 字幕用に .start / .end / .text を持つオブジェクト
            segment_objects.append(
                SimpleNamespace(start=start_sec, end=end_sec, text=text)
            )

    if mp4_file:
        insert_subtitle(segment_objects, mp4_file, duration)
    else:
        with open("test.json", "w") as f:
            json.dump(results, f, ensure_ascii=False)


if __name__ == "__main__":
    start_time = time.time()
    results = main("output.wav")
    # main("g_06.wav")
    end_time = time.time()
    # minuteで表示
    print((end_time - start_time) / 60)
