# ------------ CPU ----------------
FROM python:3.12-slim
# --------------------------------

# ------------ GPU ---------------
# FROM nvidia/cuda:12.2.0-devel-ubuntu22.04
# # Install Python
# RUN apt-get update && \
#     apt-get install -y python3-pip python3-dev && \
#     rm -rf /var/lib/apt/lists/*
# --------------------------------
RUN apt-get update

WORKDIR /app

COPY /src /app

RUN pip install -U pip \
    && pip install --no-cache-dir faster-whisper==1.0.2 pyannote-audio==3.3.1

ENV HUGGING_FACE_TOKEN "hf_your_token"