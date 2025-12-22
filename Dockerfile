# -------------------------------
# futsal-ai 用 Dockerfile
# -------------------------------

FROM pytorch/pytorch:2.4.1-cuda12.1-cudnn9-runtime

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_NO_CACHE_DIR=1 \
    PYTHONUNBUFFERED=1 \
    TZ=Asia/Tokyo

# unzip を追加（Roboflowの展開で必要になりやすい）
RUN apt-get update && \
    apt-get install -y --no-install-recommends ffmpeg git unzip && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /workspace/futsal-ai

# 依存関係ファイルだけ先にコピーしてインストール（存在するパスを指定）
COPY examples/futsal/requirements.txt examples/futsal/requirements.txt
COPY setup.py setup.py

RUN pip install --upgrade pip && \
    pip install -r examples/futsal/requirements.txt && \
    pip install opencv-python-headless roboflow python-dotenv ultralytics && \
    pip install -e .

WORKDIR /workspace/futsal-ai
