# examples/soccer/voronoi_frame.py
import os
import sys
import argparse

import cv2
import numpy as np
import supervision as sv
from ultralytics import YOLO

# === ここが重要：どこから実行しても `sports` を見つけられるようにする ===
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
sys.path.insert(0, ROOT_DIR)

from sports.annotators.soccer import (
    draw_pitch,
    draw_points_on_pitch,
    draw_pitch_voronoi_diagram
)
from sports.common.team import TeamClassifier
from sports.common.view import ViewTransformer
from sports.configs.soccer import SoccerPitchConfiguration

CONFIG = SoccerPitchConfiguration()

# soccer/main.py と同じクラスID（football-player-detection.pt想定）
BALL_CLASS_ID = 0
GOALKEEPER_CLASS_ID = 1
PLAYER_CLASS_ID = 2
REFEREE_CLASS_ID = 3


def get_crops(frame: np.ndarray, detections: sv.Detections):
    return [sv.crop_image(frame, xyxy) for xyxy in detections.xyxy]


def build_transformer_from_keypoints(keypoints: sv.KeyPoints) -> ViewTransformer:
    # keypoints.xy[0] の中で有効点だけ使う
    mask = (keypoints.xy[0][:, 0] > 1) & (keypoints.xy[0][:, 1] > 1)
    src = keypoints.xy[0][mask].astype(np.float32)
    tgt = np.array(CONFIG.vertices)[mask].astype(np.float32)

    # ホモグラフィに最低4点必要
    if src.shape[0] < 4:
        raise RuntimeError(f"Not enough pitch keypoints: {src.shape[0]} (<4). Try another frame_idx.")

    return ViewTransformer(source=src, target=tgt)


def read_frame(video_path: str, frame_idx: int) -> np.ndarray:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ok, frame = cap.read()
    cap.release()

    if not ok or frame is None:
        raise RuntimeError(f"Cannot read frame {frame_idx} from {video_path}")

    return frame


def safe_to_device(model: YOLO, device: str) -> str:
    """
    cuda が詰まっている / OOM の場合に cpu に逃がす
    """
    if device.startswith("cuda"):
        try:
            model.to(device=device)
            return device
        except Exception:
            print("[WARN] CUDA is busy or failed. Falling back to CPU.")
            model.to(device="cpu")
            return "cpu"
    else:
        model.to(device=device)
        return device


def main(
    video: str,
    out_png: str,
    device: str,
    frame_idx: int,
    conf: float,
    kp_th: float,
    imgsz: int,
    opacity: float
):
    script_dir = os.path.dirname(os.path.abspath(__file__))
    player_model_path = os.path.join(script_dir, "data/football-player-detection.pt")
    pitch_model_path = os.path.join(script_dir, "data/football-pitch-detection.pt")

    if not os.path.exists(player_model_path):
        raise FileNotFoundError(f"Player model not found: {player_model_path}")
    if not os.path.exists(pitch_model_path):
        raise FileNotFoundError(f"Pitch model not found: {pitch_model_path}")

    # 1) 指定フレーム取得
    frame = read_frame(video, frame_idx)

    # 2) モデルロード（device割当は安全に）
    pitch_model = YOLO(pitch_model_path)
    player_model = YOLO(player_model_path)
    device = safe_to_device(pitch_model, device)
    device = safe_to_device(player_model, device)

    # 3) ピッチキーポイント検出 → 変換器作成
    pitch_res = pitch_model(frame, verbose=False)[0]
    keypoints = sv.KeyPoints.from_ultralytics(pitch_res)

    # confidenceでフィルタ（推定が荒いとき用）
    if hasattr(keypoints, "confidence") and keypoints.confidence is not None:
        # keypoints.confidence: (1, N)
        keep = keypoints.confidence[0] > kp_th
        keypoints.xy[0][~keep] = 0

    transformer = build_transformer_from_keypoints(keypoints)

    # 4) 選手検出
    det_res = player_model(frame, imgsz=imgsz, conf=conf, verbose=False)[0]
    det = sv.Detections.from_ultralytics(det_res)

    players = det[det.class_id == PLAYER_CLASS_ID]
    if len(players) == 0:
        raise RuntimeError("No players detected in this frame. Try another frame_idx or lower --conf.")

    # 5) チーム分類（1フレーム簡易版：fit→predict）
    crops = get_crops(frame, players)
    team_clf = TeamClassifier(device=device)
    team_clf.fit(crops)
    team_id = team_clf.predict(crops)  # 0/1

    # 6) 画像座標 → ピッチ座標へ
    xy_img = players.get_anchors_coordinates(sv.Position.BOTTOM_CENTER)
    xy_pitch = transformer.transform_points(points=xy_img)

    team1_xy = xy_pitch[team_id == 0]
    team2_xy = xy_pitch[team_id == 1]

    # 7) ボロノイ描画
    pitch_img = draw_pitch(CONFIG)
    vor = draw_pitch_voronoi_diagram(
        config=CONFIG,
        team_1_xy=team1_xy,
        team_2_xy=team2_xy,
        team_1_color=sv.Color.from_hex("00BFFF"),
        team_2_color=sv.Color.from_hex("FF1493"),
        opacity=opacity,
        pitch=pitch_img
    )

    # 8) 位置点も重ねる（見やすく）
    vor = draw_points_on_pitch(CONFIG, team1_xy,
                               face_color=sv.Color.from_hex("00BFFF"),
                               edge_color=sv.Color.WHITE,
                               radius=14, thickness=2, pitch=vor)
    vor = draw_points_on_pitch(CONFIG, team2_xy,
                               face_color=sv.Color.from_hex("FF1493"),
                               edge_color=sv.Color.WHITE,
                               radius=14, thickness=2, pitch=vor)

    os.makedirs(os.path.dirname(out_png) or ".", exist_ok=True)
    cv2.imwrite(out_png, vor)
    print(f"[OK] Saved Voronoi PNG: {out_png}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", required=True, type=str)
    parser.add_argument("--out_png", default="debug_out/voronoi.png", type=str)
    parser.add_argument("--device", default="cpu", type=str)  # "cpu" / "cuda" / "cuda:0"
    parser.add_argument("--frame_idx", default=0, type=int)

    parser.add_argument("--conf", default=0.3, type=float)     # 選手検出conf
    parser.add_argument("--kp_th", default=0.5, type=float)     # ピッチKP信頼度
    parser.add_argument("--imgsz", default=960, type=int)       # 1280は重いのでまず960推奨
    parser.add_argument("--opacity", default=0.55, type=float)  # 0.0〜1.0

    args = parser.parse_args()
    main(
        video=args.video,
        out_png=args.out_png,
        device=args.device,
        frame_idx=args.frame_idx,
        conf=args.conf,
        kp_th=args.kp_th,
        imgsz=args.imgsz,
        opacity=args.opacity
    )