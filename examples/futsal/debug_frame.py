import argparse
import os
import cv2
import numpy as np
import supervision as sv
from tqdm import tqdm
from ultralytics import YOLO

from sports.annotators.futsal import draw_pitch, draw_points_on_pitch
from sports.common.team import TeamClassifier
from sports.common.view import ViewTransformer
from sports.configs.futsal import FutsalPitchConfiguration

PARENT_DIR = os.path.dirname(os.path.abspath(__file__))
PLAYER_DETECTION_MODEL_PATH = os.path.join(PARENT_DIR, "data/best_futsal-players-detection.pt")
PITCH_DETECTION_MODEL_PATH  = os.path.join(PARENT_DIR, "data/best_futsal-pitch-detection.pt")
BALL_DETECTION_MODEL_PATH   = os.path.join(PARENT_DIR, "data/best_futsal-ball-detection.pt")

BALL_CLASS_ID = 0
GOALKEEPER_CLASS_ID = 1
PLAYER_CLASS_ID = 2
REFEREE_CLASS_ID = 3

CONFIG = FutsalPitchConfiguration()
COLORS = ["#FF1493", "#00BFFF", "#FF6347", "#FFD700"]

VERTEX_LABEL_ANNOTATOR = sv.VertexLabelAnnotator(
    color=[sv.Color.from_hex(c) for c in CONFIG.colors],
    text_color=sv.Color.from_hex("#FFFFFF"),
    border_radius=5,
    text_thickness=1,
    text_scale=0.5,
    text_padding=5,
    
)

BOX_ANNOTATOR = sv.BoxAnnotator(color=sv.ColorPalette.from_hex(COLORS), thickness=2)
BOX_LABEL_ANNOTATOR = sv.LabelAnnotator(
    color=sv.ColorPalette.from_hex(COLORS),
    text_color=sv.Color.from_hex("#FFFFFF"),
    text_padding=5,
    text_thickness=1,
)

ELLIPSE_ANNOTATOR = sv.EllipseAnnotator(color=sv.ColorPalette.from_hex(COLORS), thickness=2)
ELLIPSE_LABEL_ANNOTATOR = sv.LabelAnnotator(
    color=sv.ColorPalette.from_hex(COLORS),
    text_color=sv.Color.from_hex("#FFFFFF"),
    text_padding=5,
    text_thickness=1,
    text_position=sv.Position.BOTTOM_CENTER,
)

TRIANGLE_ANNOTATOR = sv.TriangleAnnotator(
    color=sv.Color.from_hex("#FFD700"),
    base=25,
    height=21,
    outline_thickness=1,
)

def get_frame_by_index(video_path: str, frame_idx: int) -> np.ndarray:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise RuntimeError(f"Cannot read frame {frame_idx} from {video_path}")
    return frame

def get_crops(frame: np.ndarray, detections: sv.Detections):
    return [sv.crop_image(frame, xyxy) for xyxy in detections.xyxy]

def resolve_goalkeepers_team_id(players: sv.Detections, players_team_id: np.ndarray, goalkeepers: sv.Detections) -> np.ndarray:
    goalkeepers_xy = goalkeepers.get_anchors_coordinates(sv.Position.BOTTOM_CENTER)
    players_xy = players.get_anchors_coordinates(sv.Position.BOTTOM_CENTER)
    team_0_centroid = players_xy[players_team_id == 0].mean(axis=0)
    team_1_centroid = players_xy[players_team_id == 1].mean(axis=0)

    out = []
    for gxy in goalkeepers_xy:
        d0 = np.linalg.norm(gxy - team_0_centroid)
        d1 = np.linalg.norm(gxy - team_1_centroid)
        out.append(0 if d0 < d1 else 1)
    return np.array(out)

def build_transformer_from_keypoints(keypoints: sv.KeyPoints, conf_th: float = 0.3) -> ViewTransformer:
    # keypoints.xy: (1, N, 2), keypoints.confidence: (1, N)
    conf = keypoints.confidence[0]
    mask = conf > conf_th
    detected_indices = np.where(mask)[0]

    # CONFIG.vertices と index を揃える（超重要）
    valid_indices = detected_indices[detected_indices < len(CONFIG.vertices)]
    if len(valid_indices) < 4:
        raise RuntimeError(f"Not enough keypoints for homography. valid={len(valid_indices)} (need >=4).")

    src = keypoints.xy[0][valid_indices].astype(np.float32)
    tgt = np.array(CONFIG.vertices)[valid_indices].astype(np.float32)
    return ViewTransformer(source=src, target=tgt)

def save_image(path: str, bgr: np.ndarray):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    cv2.imwrite(path, bgr)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source_video_path", type=str, required=True)
    parser.add_argument("--frame_idx", type=int, default=0)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--out_dir", type=str, default="debug_out")
    parser.add_argument("--conf", type=float, default=0.3)
    parser.add_argument("--stride", type=int, default=60)  # team fit用
    args = parser.parse_args()

    out_dir = args.out_dir
    os.makedirs(out_dir, exist_ok=True)

    # 1) 任意フレーム取得
    frame = get_frame_by_index(args.source_video_path, args.frame_idx)
    save_image(os.path.join(out_dir, "00_frame.png"), frame)

    # 2) モデル読み込み
    player_model = YOLO(PLAYER_DETECTION_MODEL_PATH).to(device=args.device)
    pitch_model  = YOLO(PITCH_DETECTION_MODEL_PATH).to(device=args.device)

    # 3) pitch keypoints
    pitch_result = pitch_model(frame, verbose=False)[0]
    keypoints = sv.KeyPoints.from_ultralytics(pitch_result)

    pitch_vis = frame.copy()
    pitch_vis = VERTEX_LABEL_ANNOTATOR.annotate(pitch_vis, keypoints, CONFIG.labels)
    save_image(os.path.join(out_dir, "01_pitch_keypoints.png"), pitch_vis)

    # 4) player/ball/ref detections
    det_result = player_model(frame, imgsz=1280, verbose=False)[0]
    detections = sv.Detections.from_ultralytics(det_result)

    # bboxで確認
    det_bbox = frame.copy()
    det_bbox = BOX_ANNOTATOR.annotate(det_bbox, detections)
    det_bbox = BOX_LABEL_ANNOTATOR.annotate(det_bbox, detections)
    save_image(os.path.join(out_dir, "02_detections_bbox.png"), det_bbox)

    # ellipse + ball triangle（見やすい）
    ball_det = detections[detections.class_id == BALL_CLASS_ID]
    ball_det.xyxy = sv.pad_boxes(ball_det.xyxy, px=10)

    others = detections[detections.class_id != BALL_CLASS_ID]
    others = others.with_nms(threshold=0.5, class_agnostic=True)
    # class_id を main.py 方式で揃えるならここで調整が必要な場合あり（要注意）
    det_ell = frame.copy()
    det_ell = ELLIPSE_ANNOTATOR.annotate(det_ell, others)
    det_ell = TRIANGLE_ANNOTATOR.annotate(det_ell, ball_det)
    save_image(os.path.join(out_dir, "03_detections_ellipse_ball.png"), det_ell)

    # 5) TeamClassifier fit（動画から crops 収集）
    frame_gen = sv.get_video_frames_generator(source_path=args.source_video_path, stride=args.stride)
    crops = []
    for f in tqdm(frame_gen, desc="collecting crops for team fit"):
        r = player_model(f, imgsz=1280, verbose=False)[0]
        d = sv.Detections.from_ultralytics(r)
        players = d[d.class_id == PLAYER_CLASS_ID]
        crops += get_crops(f, players)

    team_classifier = TeamClassifier(device=args.device)
    team_classifier.fit(crops)

    # 6) フレーム上で team assign
    players = detections[detections.class_id == PLAYER_CLASS_ID]
    player_crops = get_crops(frame, players)
    players_team_id = team_classifier.predict(player_crops)

    goalkeepers = detections[detections.class_id == GOALKEEPER_CLASS_ID]
    goalkeepers_team_id = resolve_goalkeepers_team_id(players, players_team_id, goalkeepers)

    referees = detections[detections.class_id == REFEREE_CLASS_ID]

    merged = sv.Detections.merge([players, goalkeepers, referees])
    color_lookup = np.array(
        players_team_id.tolist() +
        goalkeepers_team_id.tolist() +
        [REFEREE_CLASS_ID] * len(referees)
    )

    labels = []
    # tracker_idが無い場合もあるので安全に
    if merged.tracker_id is not None:
        labels = [str(tid) for tid in merged.tracker_id]
    else:
        labels = [""] * len(merged)

    team_vis = frame.copy()
    team_vis = ELLIPSE_ANNOTATOR.annotate(team_vis, merged, custom_color_lookup=color_lookup)
    team_vis = ELLIPSE_LABEL_ANNOTATOR.annotate(team_vis, merged, labels=labels, custom_color_lookup=color_lookup)
    save_image(os.path.join(out_dir, "04_team_classification.png"), team_vis)

    # 7) radar（homography）
    transformer = build_transformer_from_keypoints(keypoints, conf_th=args.conf)

    radar = draw_pitch(config=CONFIG)
    # anchors
    xy = merged.get_anchors_coordinates(sv.Position.BOTTOM_CENTER)
    txy = transformer.transform_points(points=xy)

    radar = draw_points_on_pitch(CONFIG, txy[color_lookup == 0], sv.Color.from_hex(COLORS[0]), radius=18, pitch=radar)
    radar = draw_points_on_pitch(CONFIG, txy[color_lookup == 1], sv.Color.from_hex(COLORS[1]), radius=18, pitch=radar)
    radar = draw_points_on_pitch(CONFIG, txy[color_lookup == 2], sv.Color.from_hex(COLORS[2]), radius=18, pitch=radar)
    radar = draw_points_on_pitch(CONFIG, txy[color_lookup == 3], sv.Color.from_hex(COLORS[3]), radius=18, pitch=radar)
    save_image(os.path.join(out_dir, "05_radar.png"), radar)

    # overlay
    h, w = frame.shape[:2]
    radar_small = sv.resize_image(radar, (w // 2, h // 2))
    rh, rw = radar_small.shape[:2]
    rect = sv.Rect(x=w // 2 - rw // 2, y=h - rh, width=rw, height=rh)
    overlay = sv.draw_image(frame.copy(), radar_small, opacity=0.5, rect=rect)
    save_image(os.path.join(out_dir, "06_radar_overlay.png"), overlay)

    print(f"[OK] saved debug images to: {out_dir}")

if __name__ == "__main__":
    main()
