# instantiate the pipeline
import json
import os
import time
import torch
from pathlib import Path
from types import SimpleNamespace

from faster_whisper import WhisperModel
from pyannote.audio import Pipeline
import torchaudio

try:
    from moviepy import VideoFileClip, TextClip, CompositeVideoClip
except ImportError:
    from moviepy.editor import VideoFileClip, TextClip, CompositeVideoClip

# ゲート付きモデル（pyannote 等）用。必須: https://huggingface.co/settings/tokens で発行し、
# 各モデルページで「Agree and access repository」を実施すること
hf_token = os.getenv("HUGGING_FACE_TOKEN") or os.getenv("HF_TOKEN")


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
    # whisper_model_path = "./large-v3"
    whisper_model_path = "./turbo_offline"
    whisper_device = "cuda"
    pyannote_device = "cuda"
    whisper_compute_type = "int8"
    diarization_model_id = "pyannote_config.yaml"

    print(f"[開始] audio_file={audio_file} mp4_file={mp4_file or 'なし'}")
    print(f"[モデル] 話者分離(diarization)={diarization_model_id}")
    print(
        f"[モデル] 文字起こし(whisper)={whisper_model_path} device={whisper_device} compute_type={whisper_compute_type}"
    )

    # 入力音声をロード（話者分離用）
    waveform, sample_rate = torchaudio.load(audio_file)
    duration = waveform.shape[1] / sample_rate

    # 話者分離は全体に対して1回だけ
    print("[話者分離中...]")
    pipeline = Pipeline.from_pretrained(diarization_model_id)
    device = torch.device(pyannote_device if torch.cuda.is_available() else "cpu")
    pipeline.to(device)
    speaker_separation_start = time.perf_counter()
    diarization = pipeline({"waveform": waveform, "sample_rate": sample_rate})
    speaker_separation_end = time.perf_counter()
    speaker_separation_sec = speaker_separation_end - speaker_separation_start
    print("[話者分離完了]")

    # 音声全体を1回だけ文字起こし（faster_whisper）
    print("[文字起こしモデル読み込み中...]")
    model = WhisperModel(
        whisper_model_path, device=whisper_device, compute_type=whisper_compute_type
    )
    # model = WhisperModel(
    #     "turbo", device=whisper_device, compute_type=whisper_compute_type
    # )

    results = []
    segment_objects = []  # insert_subtitle 用の .start / .end / .text オブジェクト

    print("[文字起こし中...]")
    transcription_start = time.perf_counter()
    segs, _ = model.transcribe(audio_file, vad_filter=False)

    for s in segs:
        text = s.text.strip()
        if not text:
            continue

        speaker = get_speaker_for_interval(diarization, s.start, s.end)
        results.append(
            {
                "id": len(results) + 1,
                "start": format_timestamp(s.start),
                "end": format_timestamp(s.end),
                "text": text,
                "speaker": speaker,
            }
        )
        # 字幕用に .start / .end / .text を持つオブジェクト
        segment_objects.append(SimpleNamespace(start=s.start, end=s.end, text=text))

    transcription_end = time.perf_counter()
    transcription_sec = transcription_end - transcription_start
    print("[文字起こし完了]")

    if mp4_file:
        print("[字幕生成中...]")
        insert_subtitle(segment_objects, mp4_file, duration)
    else:
        print("[test.json 書き出し中...]")
        # Windows のデフォルトエンコーディングだと文字化けしやすいので UTF-8 明示
        with open("test.json", "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False)

    timing = {
        "speaker_separation_sec": speaker_separation_sec,
        "transcription_sec": transcription_sec,
    }
    return timing


if __name__ == "__main__":
    timing = main("output.wav")

    print(f"話者分離: {timing['speaker_separation_sec']:.2f}秒")
    print(f"文字起こし: {timing['transcription_sec']:.2f}秒")
