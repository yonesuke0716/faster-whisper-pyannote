import time
from pathlib import Path

from dotenv import load_dotenv
from google import genai
import os

# src から実行してもプロジェクト直下の .env を読む
load_dotenv(Path(__file__).resolve().parent.parent / ".env")
load_dotenv()

gemini_api_key = os.environ["GEMINI_API_KEY"]

# transcript.py の test.json と同一スキーマ（id, start, end, text, speaker）になるよう指示する
TRANSCRIPT_OUTPUT_PROMPT = """\
あなたは音声の話者分離（ダイアライゼーション）と文字起こしを行います。

【出力】
説明文・前置き・マークダウンのコードフェンス（```）は一切付けず、JSON 配列だけを出力してください。先頭が `[`、末尾が `]` となるようにし、パース可能な JSON のみとします。

各要素は次のキーを必ず持つオブジェクトにしてください（transcript.py が書き出す test.json と同じ形）:
- "id": 整数。先頭の発話セグメントから 1 始まりの連番。
- "start": 文字列。開始時刻を "HH:MM:SS.mmm"（24時間制、ミリ秒は3桁ゼロ埋め）。例: "00:01:23.456"
- "end": 文字列。終了時刻。形式は start と同じ。
- "text": 文字列。その区間の発話テキスト（前後の空白は除く）。
- "speaker": 文字列。話者IDは "SPEAKER_00", "SPEAKER_01", ... のように2桁ゼロ埋め。同一話者には常に同じIDを付与する。

【ルール】
- 音声の先頭を 00:00:00.000 とし、start/end は実際の経過時間に合わせる。
- 聞き取れる発話を意味のまとまりでセグメント化する。過度な碎片化は避ける。
- 標準的な JSON の文字列エスケープに従う（改行や引用符を text に含める場合はエスケープ）。

上記のみを満たす JSON 配列を出力し、それ以外は書かないでください。
"""

MODEL = "gemini-3-flash-preview"

# Gemini 3 Flash プレビュー（標準・有料階層）USD / 100 万トークン
# https://ai.google.dev/gemini-api/docs/pricing?hl=ja#gemini-3-flash-preview
PRICE_INPUT_TEXT_IMAGE_VIDEO_PER_1M_USD = 0.50
PRICE_INPUT_AUDIO_PER_1M_USD = 1.00
PRICE_OUTPUT_INCL_THOUGHTS_PER_1M_USD = 3.00


def estimate_cost_usd_gemini_3_flash_preview(
    prompt_token_count: int,
    candidates_token_count: int,
    thoughts_token_count: int | None,
    text_only_prompt_tokens: int | None,
) -> tuple[float, dict[str, float | int]]:
    """usage_metadata とプロンプト単体の count からおおよその USD を推定する。

    入力単価はテキスト/画像/動画と音声で異なるため、プロンプト文字列のトークン数を
    count_tokens で取得し、prompt の残りを音声側として按分する（近似）。
    """
    thoughts = int(thoughts_token_count or 0)
    out_tok = int(candidates_token_count or 0) + thoughts

    if text_only_prompt_tokens is not None and text_only_prompt_tokens >= 0:
        text_part = min(text_only_prompt_tokens, prompt_token_count)
        audio_part = max(0, prompt_token_count - text_part)
    else:
        text_part = 0
        audio_part = prompt_token_count

    input_usd = (
        text_part * PRICE_INPUT_TEXT_IMAGE_VIDEO_PER_1M_USD / 1_000_000
        + audio_part * PRICE_INPUT_AUDIO_PER_1M_USD / 1_000_000
    )
    output_usd = out_tok * PRICE_OUTPUT_INCL_THOUGHTS_PER_1M_USD / 1_000_000
    total = input_usd + output_usd
    breakdown: dict[str, float | int] = {
        "input_text_tokens_est": text_part,
        "input_audio_tokens_est": audio_part,
        "output_tokens_incl_thoughts": out_tok,
        "input_usd": round(input_usd, 6),
        "output_usd": round(output_usd, 6),
        "total_usd": round(total, 6),
    }
    return total, breakdown


client = genai.Client(api_key=gemini_api_key)
myfile = client.files.upload(file="output.wav")

text_only_token_result = client.models.count_tokens(
    model=MODEL, contents=TRANSCRIPT_OUTPUT_PROMPT
)
text_only_tokens = getattr(text_only_token_result, "total_tokens", None)
if text_only_tokens is not None:
    print(f"入力トークン（count_tokens・プロンプト文字列のみ）: {text_only_tokens}")

# 入力のみのトークン数（generate 前の見積）。音声は概ね 32 トークン/秒など:
# https://ai.google.dev/gemini-api/docs/tokens?hl=ja#video-audio
input_token_result = client.models.count_tokens(
    model=MODEL, contents=[TRANSCRIPT_OUTPUT_PROMPT, myfile]
)
input_total = getattr(input_token_result, "total_tokens", None)
if input_total is not None:
    print(f"入力トークン（count_tokens）: {input_total}")
else:
    print(f"入力トークン（count_tokens）: {input_token_result}")

transcription_start = time.perf_counter()
response = client.models.generate_content(
    model=MODEL, contents=[TRANSCRIPT_OUTPUT_PROMPT, myfile]
)
transcription_sec = time.perf_counter() - transcription_start

print(response.text)

um = response.usage_metadata
if um is not None:
    prompt_n = getattr(um, "prompt_token_count", None)
    out_n = getattr(um, "candidates_token_count", None)
    total_n = getattr(um, "total_token_count", None)
    print(
        "usage_metadata: "
        f"prompt_token_count={prompt_n}, "
        f"candidates_token_count={out_n}, "
        f"total_token_count={total_n}"
    )
    thoughts = getattr(um, "thoughts_token_count", None)
    cached = getattr(um, "cached_content_token_count", None)
    if thoughts is not None:
        print(f"  thoughts_token_count={thoughts}")
    if cached is not None:
        print(f"  cached_content_token_count={cached}")

    if prompt_n is not None:
        _, est = estimate_cost_usd_gemini_3_flash_preview(
            prompt_token_count=prompt_n,
            candidates_token_count=out_n or 0,
            thoughts_token_count=thoughts,
            text_only_prompt_tokens=text_only_tokens,
        )
        print(
            "推定料金（gemini-3-flash-preview・標準・有料階層想定・USD・概算）: "
            f"${est['total_usd']:.6f}"
        )
        print(
            f"  入力: テキスト相当 {est['input_text_tokens_est']} tok @ "
            f"${PRICE_INPUT_TEXT_IMAGE_VIDEO_PER_1M_USD}/1M + "
            f"音声相当 {est['input_audio_tokens_est']} tok @ "
            f"${PRICE_INPUT_AUDIO_PER_1M_USD}/1M → ${est['input_usd']:.6f}"
        )
        print(
            f"  出力（候補+思考）: {est['output_tokens_incl_thoughts']} tok @ "
            f"${PRICE_OUTPUT_INCL_THOUGHTS_PER_1M_USD}/1M → ${est['output_usd']:.6f}"
        )
        print(
            "  ※ 無料枠・バッチ・キャッシュ・為替・請求の丸めは未反映。"
            " 入力のテキスト/音声按分は count_tokens（プロンプトのみ）による近似です。"
        )
else:
    print("usage_metadata: なし（API が返さない場合があります）")

print(f"文字起こし: {transcription_sec:.2f}秒")
