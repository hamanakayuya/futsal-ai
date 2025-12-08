# -------------------------------
# futsal-ai 用 Dockerfile
# -------------------------------

# PyTorch + CUDA12.1 + cuDNN入りの公式イメージ
FROM pytorch/pytorch:2.4.1-cuda12.1-cudnn9-runtime

# 環境変数（ログをバッファせず出す / タイムゾーンなど）
ENV DEBIAN_FRONTEND=noninteractive \
    PIP_NO_CACHE_DIR=1 \
    PYTHONUNBUFFERED=1 \
    TZ=Asia/Tokyo

# Linux パッケージをインストール（ffmpeg, git くらい）
RUN apt-get update && \
    apt-get install -y --no-install-recommends ffmpeg git && \
    rm -rf /var/lib/apt/lists/*

# コンテナ内で作業する場所
WORKDIR /workspace/futsal-ai

# 依存関係ファイルだけ先にコピーしてインストール
# （コード本体は後でホストからマウントする）
COPY examples/soccer/requirements.txt examples/soccer/requirements.txt
COPY setup.py setup.py

RUN pip install --upgrade pip && \
    pip install -r examples/soccer/requirements.txt && \
    pip install opencv-python-headless && \
    pip install -e .

# デフォルトの作業ディレクトリ
WORKDIR /workspace/futsal-ai
