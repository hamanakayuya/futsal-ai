#!/bin/bash
#
# setup.sh  – futsal 版
#  1) data/ ディレクトリを作成
#  2) Google Drive からモデル (.pt) とテスト動画を gdown で取得

# ------------------------------------------------------------
# 0. 変数 – このスクリプト自身があるディレクトリ
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
# 1. data/ が無ければ作る
if [[ ! -e "$DIR/data" ]]; then
    mkdir "$DIR/data"
else
    echo "'data' directory already exists."
fi

# ------------------------------------------------------------
# 2. download the models  – .pt ウェイト
#  ※ <xxx> を Google Drive 共有リンクの ID に置き換えてください
gdown -O "$DIR/data/futsal-ball-detection.pt"     "https://drive.google.com/uc?id=1-F_1yQI5QI3wPNOh0kezGbdgOCKI7KKY"
gdown -O "$DIR/data/futsal-players-detection.pt"  "https://drive.google.com/uc?id=1ddIqEXFAMozxq9a133Yqfr5XJWrshwaH"
gdown -O "$DIR/data/futsal-pitch-detection.pt"    "https://drive.google.com/uc?id=1TSByp3LYr_T61hWZU-FqYRo3EvcdYyAt"

# ------------------------------------------------------------
# 3. download the videos – テスト用動画
gdown -O "$DIR/data/test_video.mp4"               "https://drive.google.com/uc?id=1ZXW93E-1mW0lkSME5_oLp_C_o-7wRF70"

echo "✅  Download finished.  All assets saved to: $DIR/data/"

https://drive.google.com/file/d/1ddIqEXFAMozxq9a133Yqfr5XJWrshwaH/view?usp=drive_link

