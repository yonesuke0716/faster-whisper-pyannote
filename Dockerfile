# ========== Stage 1: Builder（依存関係のインストール） ==========
# ------------ CPU ----------------
FROM python:3.12.6-slim-bullseye AS builder
# --------------------------------

# ------------ GPU ---------------
# FROM nvidia/cuda:12.2.0-devel-ubuntu22.04 AS builder
# RUN apt-get update && \
#     apt-get install -y python3-pip python3-dev && \
#     rm -rf /var/lib/apt/lists/*
# --------------------------------

WORKDIR /app

# NumPy 2.x は torch/ctranslate2 等と不整合のため 1.x に固定（最後に再インストールで上書きを防ぐ）
# pyannote が torchaudio.AudioMetaData を参照するため torch/torchaudio も互換版に固定
# pyannote が use_auth_token を渡すため、huggingface_hub は use_auth_token 未廃止の 0.22.x に固定
RUN pip install -U pip \
    && pip install --no-cache-dir "numpy<2" "torch==2.2.0" "torchaudio==2.2.0" \
    && pip install --no-cache-dir faster-whisper==1.2.1 pyannote-audio==3.3.1 moviepy==1.0.3 matplotlib pydub \
    && pip install --no-cache-dir ImageMagic==0.2.1 \
    && pip install --no-cache-dir --force-reinstall "huggingface_hub>=0.19,<0.23" "numpy<2"

COPY src /app

# ========== Stage 2: Runtime（軽量な本番用イメージ） ==========
# ------------ CPU ----------------
FROM python:3.12.6-slim-bullseye AS runtime
# --------------------------------

# ------------ GPU ---------------
# FROM nvidia/cuda:12.2.0-runtime-ubuntu22.04 AS runtime
# RUN apt-get update && \
#     apt-get install -y python3 python3-distutils && \
#     rm -rf /var/lib/apt/lists/*
# --------------------------------

# ffmpeg: 動画処理 / imagemagick: moviepy の TextClip で字幕描画に必須 / fonts-dejavu: コンテナ内で利用可能なフォント
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg imagemagick fontconfig fonts-dejavu-core \
    && rm -rf /var/lib/apt/lists/*

# ImageMagick のセキュリティポリシー緩和（moviepy が /tmp の @ 経由でテキスト→PNG するため）
RUN POLICY=/etc/ImageMagick-6/policy.xml \
    && if [ -f "$POLICY" ]; then \
      sed -i '/pattern="@\*"/s/rights="none"/rights="read|write"/' "$POLICY"; \
      sed -i '/pattern="PDF"/s/rights="none"/rights="read|write"/' "$POLICY" 2>/dev/null || true; \
    fi

# moviepy TextClip が ImageMagick を参照するためのパス
ENV IMAGEMAGICK_BINARY=/usr/bin/convert

# Builder から Python パッケージのみコピー（pip やビルドキャッシュは含まない）
COPY --from=builder /usr/local/lib/python3.12/site-packages /usr/local/lib/python3.12/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

WORKDIR /app
COPY --from=builder /app /app
