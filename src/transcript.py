# instantiate the pipeline
import json
import os
import time

# import faster_whisper
from faster_whisper import WhisperModel
from pyannote.audio import Pipeline

# import torch
import torchaudio

hf_token = os.getenv("HUGGING_FACE_TOKEN")
start_time = time.time()


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
                "start": f"{s_h:02}:{s_m:02}:{s_s:02},{s_ms:03}",
                "end": f"{e_h:02}:{e_m:02}:{e_s:02},{e_ms:03}",
                "text": text,
                "speaker": current_speaker,
            }
        )
    return data


def main(audio_file):

    # ============= faster_whisper s ================
    # offline（ダウンロード済モデルを使用）
    # faster_whisper.download_model("large-v3", "./models/large-v3")
    # model = WhisperModel("./models/large-v3", device="cpu", compute_type="int8")

    # online（ネットからモデルをダウンロード）
    model = WhisperModel("large-v3", device="cpu", compute_type="int8")

    segments, _ = model.transcribe(audio_file, vad_filter=True)

    # ============= faster_whisper e ================

    # ============= pyannotte s ================
    # offline（ダウンロード済モデルを使用）
    # pipeline = Pipeline.from_pretrained(
    #     "pyannote_config.yaml",
    # )
    # online（ネットからモデルをダウンロード）
    pipeline = Pipeline.from_pretrained(
        "pyannote/speaker-diarization-3.1",
        use_auth_token=str(hf_token),
    )

    # GPUで実行
    # pipeline = pipeline.to(torch.device("cuda"))
    # # cudaが有効かどうかを確認
    # print(torch.cuda.is_available())
    # # cudaのGPU名を確認
    # print(torch.cuda.get_device_name())

    waveform, sample_rate = torchaudio.load(audio_file)
    duration = waveform.shape[1] / sample_rate
    diarization = pipeline({"waveform": waveform, "sample_rate": sample_rate})
    # ============= pyannotte e ================

    results = concat_whisper_pyannote(segments, diarization, duration)

    return results


if __name__ == "__main__":
    results = main("g_06.wav")
    # resultsをjson形式に変換
    with open("g_06.json", "w") as f:
        json.dump(results, f, ensure_ascii=False)

    end_time = time.time()
    # minuteで表示
    print((end_time - start_time) / 60)
