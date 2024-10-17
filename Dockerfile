# ------------ CPU ----------------
FROM python:3.12.6-slim-bullseye AS builder
# --------------------------------

# ------------ GPU ---------------
# FROM nvidia/cuda:12.2.0-devel-ubuntu22.04
# # Install Python
# RUN apt-get update && \
#     apt-get install -y python3-pip python3-dev && \
#     rm -rf /var/lib/apt/lists/*
# --------------------------------
RUN apt-get update

RUN pip install -U pip \
    && pip install --no-cache-dir faster-whisper==1.0.2 pyannote-audio==3.3.1

COPY /src /app

FROM python:3.12.6-slim-bullseye AS dev

WORKDIR /app

COPY --from=builder /usr/local/lib/python3.12/site-packages /root/local/lib/python3.12/site-packages
COPY --from=builder /usr/local/bin usr/local/bin
COPY --from=builder /app /app

ENV HUGGING_FACE_TOKEN "hf_your_token"