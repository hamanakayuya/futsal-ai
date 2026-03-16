# examples/futsal/voronoi_frame.py
import argparse
import os
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np
import supervision as sv
from ultralytics import YOLO
from tqdm import tqdm

from sports.annotators.futsal import (
    draw_pitch,
    draw_points_on_pitch,
    draw_pitch_voronoi_diagram,
)
from sports.common.team import TeamClassifier
from sports.configs.futsal import FutsalPitchConfiguration

# ─────────────────────────────────────────────
# paths (main.py / voronoi_video.py と同じ構成)
# ─────────────────────────────────────────────
PARENT_DIR = os.path.dirname(os.path.abspath(__file__))

PLAYER_DETECTION_MODEL_PATH = os.path.join(PARENT_DIR, "data/best_futsal-players-detection.pt")
PITCH_DETECTION_MODEL_PATH  = os.path.join(PARENT_DIR, "data/best_futsal-pitch-detection.pt")
BALL_DETECTION_MODEL_PATH   = os.path.join(PARENT_DIR, "data/best_futsal-ball-detection.pt")

# class ids
BALL_CLASS_ID = 0
GOALKEEPER_CLASS_ID = 1
PLAYER_CLASS_ID = 2
REFEREE_CLASS_ID = 3

CONFIG = FutsalPitchConfiguration()

# チーム色（あなたの運用に合わせてOK）
# ここは「描画色」なので、team_classifier の 0/1 がどちらのチームかで見た目が決まります
COLORS = ["#00BFFF", "#FF1493", "#FF6347", "#FFD700"]  # [team0, team1, referee, other]
REFEREE_COLOR_ID = 2

# radar見た目微調整（必要なら 0 に）
RADAR_Y_OFFSET = -50

# ball on radar
BALL_COLOR = sv.Color.from_hex("#FFFFFF")
BALL_RADIUS = 14

# ─────────────────────────────────────────────
# util
# ─────────────────────────────────────────────
def list_images(frames_dir: Path) -> List[Path]:
    exts = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}
    return [p for p in sorted(frames_dir.iterdir()) if p.is_file() and p.suffix.lower() in exts]


def get_crops(frame: np.ndarray, detections: sv.Detections) -> List[np.ndarray]:
    return [sv.crop_image(frame, xyxy) for xyxy in detections.xyxy]


def resolve_goalkeepers_team_id(players, players_team_id, goalkeepers) -> np.ndarray:
    goalkeepers_xy = goalkeepers.get_anchors_coordinates(sv.Position.BOTTOM_CENTER)
    players_xy = players.get_anchors_coordinates(sv.Position.BOTTOM_CENTER)

    if len(players_team_id) == 0 or len(players_xy) == 0:
        return np.zeros(len(goalkeepers_xy), dtype=int)

    # 片チームしかいない場合もあるのでガード
    if np.sum(players_team_id == 0) == 0 or np.sum(players_team_id == 1) == 0:
        return np.zeros(len(goalkeepers_xy), dtype=int)

    team_0_centroid = players_xy[players_team_id == 0].mean(axis=0)
    team_1_centroid = players_xy[players_team_id == 1].mean(axis=0)

    result = []
    for gk_xy in goalkeepers_xy:
        d0 = np.linalg.norm(gk_xy - team_0_centroid)
        d1 = np.linalg.norm(gk_xy - team_1_centroid)
        result.append(0 if d0 < d1 else 1)
    return np.array(result, dtype=int)


def _infer_ball_class_id_from_model(ball_model: YOLO, fallback: int = 0) -> int:
    """
    モデルnamesに 'ball' が入っていればそのclass_idを使う。
    うまく取れないモデルもあるのでfallbackも用意。
    """
    try:
        names = getattr(ball_model, "names", None)
        if isinstance(names, dict):
            for cid, name in names.items():
                if "ball" in str(name).lower():
                    return int(cid)
        elif isinstance(names, list):
            for cid, name in enumerate(names):
                if "ball" in str(name).lower():
                    return int(cid)
    except Exception:
        pass
    return fallback


def draw_text_box(
    img: np.ndarray,
    text: str,
    pos: Tuple[int, int] = (12, 12),
    font_scale: float = 0.8,
    thickness: int = 2,
    text_color: Tuple[int, int, int] = (255, 255, 255),
    box_color: Tuple[int, int, int] = (0, 0, 0),
    alpha: float = 0.55,
) -> np.ndarray:
    x, y = pos
    font = cv2.FONT_HERSHEY_SIMPLEX
    (tw, th), baseline = cv2.getTextSize(text, font, font_scale, thickness)
    pad = 6
    x0, y0 = x, y
    x1, y1 = x + tw + pad * 2, y + th + pad * 2 + baseline

    overlay = img.copy()
    cv2.rectangle(overlay, (x0, y0), (x1, y1), box_color, -1)
    img = cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0)

    cv2.putText(
        img, text, (x + pad, y + th + pad),
        font, font_scale, text_color, thickness, cv2.LINE_AA
    )
    return img


# ─────────────────────────────────────────────
# Homography (RANSAC)
# ─────────────────────────────────────────────
class HomographyTransformer:
    def __init__(self, H: np.ndarray):
        self.H = H.astype(np.float32)

    def transform_points(self, points: np.ndarray) -> np.ndarray:
        if points is None or len(points) == 0:
            return np.empty((0, 2), dtype=np.float32)
        pts = points.reshape(-1, 1, 2).astype(np.float32)
        out = cv2.perspectiveTransform(pts, self.H)
        return out.reshape(-1, 2).astype(np.float32)


def _keypoints_conf_array(keypoints: sv.KeyPoints) -> Optional[np.ndarray]:
    for attr in ["confidence", "conf", "scores"]:
        if hasattr(keypoints, attr):
            v = getattr(keypoints, attr)
            if v is None:
                continue
            arr = np.array(v)
            if arr.ndim == 2:
                return arr[0]
            if arr.ndim == 1:
                return arr
    return None


def build_transformer_from_keypoints(
    keypoints: sv.KeyPoints,
    kp_conf_th: float = 0.0,
    min_kp: int = 8,
    min_inliers: int = 6,
    ransac_th: float = 8.0,
    min_spread_px: float = 120.0,
) -> Tuple[Optional[HomographyTransformer], dict]:
    info = {"used_kp": 0, "inliers": 0, "spread": 0.0, "ok": False}

    if keypoints is None or keypoints.xy is None or len(keypoints.xy) == 0:
        return None, info

    pts = keypoints.xy[0].astype(np.float32)  # (K,2)
    valid = (pts[:, 0] > 1) & (pts[:, 1] > 1)

    conf = _keypoints_conf_array(keypoints)
    if conf is not None:
        conf = conf.astype(np.float32)
        valid = valid & (conf >= float(kp_conf_th))

    used = int(np.sum(valid))
    info["used_kp"] = used
    if used < int(min_kp):
        return None, info

    src = pts[valid]
    tgt = np.array(CONFIG.vertices, dtype=np.float32)[valid]

    spread = float(max(src[:, 0].max() - src[:, 0].min(), src[:, 1].max() - src[:, 1].min()))
    info["spread"] = spread
    if spread < float(min_spread_px):
        return None, info

    H, inlier_mask = cv2.findHomography(src, tgt, method=cv2.RANSAC, ransacReprojThreshold=float(ransac_th))
    if H is None or inlier_mask is None:
        return None, info

    inliers = int(np.sum(inlier_mask))
    info["inliers"] = inliers
    if inliers < int(min_inliers):
        return None, info

    info["ok"] = True
    return HomographyTransformer(H), info


# ─────────────────────────────────────────────
# Voronoi control ratio (pitch only)
# ─────────────────────────────────────────────
def build_pitch_query_points_cm(
    pitch_img: np.ndarray,
    step_px: int = 4,
    nonzero_th: int = 5,
) -> np.ndarray:
    img = pitch_img
    if img.ndim != 3:
        raise ValueError("pitch_img must be HxWxC")

    if img.shape[2] == 4:
        alpha = img[:, :, 3]
        mask = alpha > 0
    else:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        mask = gray > nonzero_th

    ys, xs = np.where(mask)
    if len(xs) == 0:
        return np.empty((0, 2), np.float32)

    x0, x1 = int(xs.min()), int(xs.max())
    y0, y1 = int(ys.min()), int(ys.max())

    xs_s = np.arange(x0, x1 + 1, step_px, dtype=np.int32)
    ys_s = np.arange(y0, y1 + 1, step_px, dtype=np.int32)
    gx, gy = np.meshgrid(xs_s, ys_s)
    pix = np.stack([gx.reshape(-1), gy.reshape(-1)], axis=1)

    inside = mask[pix[:, 1], pix[:, 0]]
    pix = pix[inside]
    if len(pix) == 0:
        return np.empty((0, 2), np.float32)

    denom_x = max(1, (x1 - x0))
    denom_y = max(1, (y1 - y0))

    x_cm = (pix[:, 0].astype(np.float32) - x0) / denom_x * float(CONFIG.length)
    y_cm = (pix[:, 1].astype(np.float32) - y0) / denom_y * float(CONFIG.width)
    return np.stack([x_cm, y_cm], axis=1).astype(np.float32)


def compute_voronoi_control_ratio(
    query_pts_cm: np.ndarray,
    team0_xy: np.ndarray,
    team1_xy: np.ndarray,
    block: int = 20000,
) -> Tuple[float, float]:
    if query_pts_cm is None or len(query_pts_cm) == 0:
        return 0.0, 0.0
    if team0_xy is None or team1_xy is None or len(team0_xy) == 0 or len(team1_xy) == 0:
        return 0.0, 0.0

    pts = query_pts_cm.astype(np.float32)
    t0 = team0_xy.astype(np.float32)
    t1 = team1_xy.astype(np.float32)

    team0_win = 0
    total = int(pts.shape[0])

    for i in range(0, total, block):
        p = pts[i:i + block]
        d0 = p[:, None, :] - t0[None, :, :]
        d1 = p[:, None, :] - t1[None, :, :]

        min_d0 = np.min(np.sum(d0 * d0, axis=-1), axis=1)
        min_d1 = np.min(np.sum(d1 * d1, axis=-1), axis=1)

        team0_win += int(np.sum(min_d0 < min_d1))

    team0_ratio = team0_win / float(total)
    team1_ratio = 1.0 - team0_ratio
    return float(team0_ratio), float(team1_ratio)


# ─────────────────────────────────────────────
# rendering
# ─────────────────────────────────────────────
def render_radar_pitch_only(
    detections: sv.Detections,
    transformer: Optional[HomographyTransformer],
    color_lookup: np.ndarray,
    ball_xy_img: Optional[np.ndarray],
) -> np.ndarray:
    pitch = draw_pitch(config=CONFIG)
    if transformer is None:
        return pitch

    xy = detections.get_anchors_coordinates(anchor=sv.Position.BOTTOM_CENTER)
    xy_pitch = transformer.transform_points(xy)
    xy_pitch[:, 1] += RADAR_Y_OFFSET

    for cid in range(len(COLORS)):
        pitch = draw_points_on_pitch(
            config=CONFIG,
            xy=xy_pitch[color_lookup == cid],
            face_color=sv.Color.from_hex(COLORS[cid]),
            radius=20,
            pitch=pitch,
        )

    if ball_xy_img is not None and len(ball_xy_img) > 0:
        ball_pitch = transformer.transform_points(ball_xy_img.astype(np.float32))
        ball_pitch[:, 1] += RADAR_Y_OFFSET
        pitch = draw_points_on_pitch(
            config=CONFIG,
            xy=ball_pitch,
            face_color=BALL_COLOR,
            radius=int(BALL_RADIUS),
            pitch=pitch,
        )
    return pitch


def render_voronoi_pitch_only(
    detections: sv.Detections,
    transformer: Optional[HomographyTransformer],
    color_lookup: np.ndarray,
    ball_xy_img: Optional[np.ndarray],
    voronoi_opacity: float,
    control_text: Optional[str],
) -> np.ndarray:
    base = draw_pitch(config=CONFIG)
    if transformer is None:
        if control_text:
            base = draw_text_box(base, control_text)
        return base

    xy = detections.get_anchors_coordinates(anchor=sv.Position.BOTTOM_CENTER)
    xy_pitch = transformer.transform_points(xy)
    xy_pitch[:, 1] += RADAR_Y_OFFSET

    team0_xy = xy_pitch[color_lookup == 0]
    team1_xy = xy_pitch[color_lookup == 1]

    if len(team0_xy) > 0 and len(team1_xy) > 0:
        base = draw_pitch_voronoi_diagram(
            config=CONFIG,
            team_1_xy=team0_xy,
            team_2_xy=team1_xy,
            team_1_color=sv.Color.from_hex(COLORS[0]),
            team_2_color=sv.Color.from_hex(COLORS[1]),
            opacity=float(voronoi_opacity),
            pitch=base,
        )

    for cid in range(len(COLORS)):
        base = draw_points_on_pitch(
            config=CONFIG,
            xy=xy_pitch[color_lookup == cid],
            face_color=sv.Color.from_hex(COLORS[cid]),
            radius=20,
            pitch=base,
        )

    if ball_xy_img is not None and len(ball_xy_img) > 0:
        ball_pitch = transformer.transform_points(ball_xy_img.astype(np.float32))
        ball_pitch[:, 1] += RADAR_Y_OFFSET
        base = draw_points_on_pitch(
            config=CONFIG,
            xy=ball_pitch,
            face_color=BALL_COLOR,
            radius=int(BALL_RADIUS),
            pitch=base,
        )

    if control_text:
        base = draw_text_box(base, control_text)

    return base


def overlay_bottom_center(frame: np.ndarray, overlay_img: np.ndarray, opacity: float) -> np.ndarray:
    h, w = frame.shape[:2]
    ol = sv.resize_image(overlay_img, (w // 2, h // 2))
    rh, rw = ol.shape[:2]
    rect = sv.Rect(x=w // 2 - rw // 2, y=h - rh, width=rw, height=rh)
    return sv.draw_image(frame, ol, opacity=float(opacity), rect=rect)


# ─────────────────────────────────────────────
# fit team classifier from frames (match04)
# ─────────────────────────────────────────────
def fit_team_classifier_from_frames(
    frames: List[Path],
    player_model: YOLO,
    device: str,
    imgsz: int,
    fit_stride: int,
    conf: float,
) -> TeamClassifier:
    crops: List[np.ndarray] = []
    use_frames = frames[::max(1, fit_stride)]

    for p in tqdm(use_frames, desc="fit: collecting player crops"):
        img = cv2.imread(str(p))
        if img is None:
            continue

        res = player_model(img, imgsz=imgsz, verbose=False)[0]
        det = sv.Detections.from_ultralytics(res)

        if hasattr(det, "confidence") and det.confidence is not None:
            det = det[det.confidence >= float(conf)]

        players = det[det.class_id == PLAYER_CLASS_ID]
        if len(players) == 0:
            continue

        crops.extend(get_crops(img, players))

    team = TeamClassifier(device=device)

    if len(crops) < 40:
        print(f"[WARN] crops are few: {len(crops)}  (try smaller --fit_stride or lower --conf)")
    team.fit(crops)
    return team


# ─────────────────────────────────────────────
# ball detection (slice inference for small ball)
# ─────────────────────────────────────────────
def make_ball_slicer(ball_model: YOLO, ball_imgsz: int, slice_wh: int) -> sv.InferenceSlicer:
    def callback(image_slice: np.ndarray) -> sv.Detections:
        res = ball_model(image_slice, imgsz=ball_imgsz, verbose=False)[0]
        return sv.Detections.from_ultralytics(res)

    return sv.InferenceSlicer(
        callback=callback,
        overlap_filter=sv.OverlapFilter.NONE,
        slice_wh=(slice_wh, slice_wh),
    )


def detect_ball_xy_sliced(
    frame: np.ndarray,
    slicer: sv.InferenceSlicer,
    ball_class_id: int,
    ball_conf: float,
    nms_th: float,
) -> Optional[np.ndarray]:
    det = slicer(frame).with_nms(threshold=float(nms_th))

    if hasattr(det, "class_id") and det.class_id is not None:
        det = det[det.class_id == int(ball_class_id)]

    if hasattr(det, "confidence") and det.confidence is not None:
        det = det[det.confidence >= float(ball_conf)]

    if len(det) == 0:
        return None

    if det.confidence is not None and len(det) > 1:
        best = int(np.argmax(det.confidence))
        det = det[[best]]

    xy = det.get_anchors_coordinates(anchor=sv.Position.BOTTOM_CENTER)
    if xy is None or len(xy) == 0:
        return None
    return xy.astype(np.float32)


# ─────────────────────────────────────────────
# main
# ─────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="Generate radar overlay PNG (like video overlay) and voronoi pitch PNG from a specified image. Team fit uses frames folder."
    )
    parser.add_argument("--frames_dir", type=str, required=True, help="静止画フォルダ（match04など）: team fit に使う")
    parser.add_argument("--image_path", type=str, required=True, help="変換したい1枚の画像")
    parser.add_argument("--out_dir", type=str, required=True, help="出力先フォルダ")
    parser.add_argument("--device", type=str, default="cpu")

    # detector settings
    parser.add_argument("--imgsz", type=int, default=1280)
    parser.add_argument("--conf", type=float, default=0.30, help="player conf (team fit & inference)")
    parser.add_argument("--fit_stride", type=int, default=10, help="fit 用に何枚に1枚使うか（小さいほど精度↑だが遅い）")

    # ball settings (重要)
    parser.add_argument("--ball_imgsz", type=int, default=640)
    parser.add_argument("--ball_conf", type=float, default=0.10)
    parser.add_argument("--ball_slice", type=int, default=640, help="InferenceSlicer の tile サイズ")
    parser.add_argument("--ball_nms", type=float, default=0.10)

    # keypoints settings (homography quality gate)
    parser.add_argument("--kp_conf", type=float, default=0.0)
    parser.add_argument("--min_kp", type=int, default=8)
    parser.add_argument("--min_inliers", type=int, default=6)
    parser.add_argument("--ransac_th", type=float, default=8.0)
    parser.add_argument("--min_spread_px", type=float, default=120.0)

    # output settings
    parser.add_argument("--voronoi_opacity", type=float, default=0.55)
    parser.add_argument("--overlay_opacity", type=float, default=0.5)

    args = parser.parse_args()

    frames_dir = Path(args.frames_dir)
    image_path = Path(args.image_path)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not frames_dir.exists():
        raise FileNotFoundError(f"frames_dir not found: {frames_dir}")
    if not image_path.exists():
        raise FileNotFoundError(f"image_path not found: {image_path}")

    frames = list_images(frames_dir)
    if len(frames) == 0:
        raise FileNotFoundError(f"no images in: {frames_dir}")

    # models
    player_model = YOLO(PLAYER_DETECTION_MODEL_PATH).to(device=args.device)
    pitch_model  = YOLO(PITCH_DETECTION_MODEL_PATH).to(device=args.device)
    ball_model   = YOLO(BALL_DETECTION_MODEL_PATH).to(device=args.device)

    ball_class_id = _infer_ball_class_id_from_model(ball_model, fallback=BALL_CLASS_ID)

    # 1) team fit from frames (match04)
    print(f"[1/3] Fit TeamClassifier from: {frames_dir} (fit_stride={args.fit_stride})")
    team_classifier = fit_team_classifier_from_frames(
        frames=frames,
        player_model=player_model,
        device=args.device,
        imgsz=args.imgsz,
        fit_stride=args.fit_stride,
        conf=args.conf,
    )

    # 2) process the specified image only
    print(f"[2/3] Predict & render for image: {image_path.name}")
    frame = cv2.imread(str(image_path))
    if frame is None:
        raise RuntimeError(f"failed to read image: {image_path}")

    # pitch keypoints -> transformer
    pitch_res = pitch_model(frame, verbose=False)[0]
    keypoints = sv.KeyPoints.from_ultralytics(pitch_res)

    transformer, hinfo = build_transformer_from_keypoints(
        keypoints=keypoints,
        kp_conf_th=args.kp_conf,
        min_kp=args.min_kp,
        min_inliers=args.min_inliers,
        ransac_th=args.ransac_th,
        min_spread_px=args.min_spread_px,
    )

    if transformer is None:
        print(f"[WARN] Homography invalid (used_kp={hinfo['used_kp']}, inliers={hinfo['inliers']}, spread={hinfo['spread']:.1f}).")
        print("       Radar/Voronoi will be 'empty pitch' (no points). Consider lowering thresholds or improving pitch model.")

    # player detections
    player_res = player_model(frame, imgsz=args.imgsz, verbose=False)[0]
    det = sv.Detections.from_ultralytics(player_res)

    if hasattr(det, "confidence") and det.confidence is not None:
        det = det[det.confidence >= float(args.conf)]

    players = det[det.class_id == PLAYER_CLASS_ID]
    goalkeepers = det[det.class_id == GOALKEEPER_CLASS_ID]
    referees = det[det.class_id == REFEREE_CLASS_ID]

    # team predict
    crops = get_crops(frame, players)
    players_team_id = team_classifier.predict(crops) if len(crops) > 0 else np.array([], dtype=int)

    goalkeepers_team_id = (
        resolve_goalkeepers_team_id(players, players_team_id, goalkeepers)
        if len(goalkeepers) > 0 and len(players_team_id) > 0 else np.array([], dtype=int)
    )

    merged = sv.Detections.merge([players, goalkeepers, referees])

    color_lookup = np.array(
        (players_team_id.tolist() if len(players_team_id) > 0 else []) +
        (goalkeepers_team_id.tolist() if len(goalkeepers_team_id) > 0 else []) +
        [REFEREE_COLOR_ID] * len(referees),
        dtype=int
    )

    # ball detection (sliced)
    ball_slicer = make_ball_slicer(ball_model, ball_imgsz=args.ball_imgsz, slice_wh=args.ball_slice)
    ball_xy_img = detect_ball_xy_sliced(
        frame=frame,
        slicer=ball_slicer,
        ball_class_id=ball_class_id,
        ball_conf=args.ball_conf,
        nms_th=args.ball_nms,
    )
    if ball_xy_img is None:
        print("[WARN] ball not detected. Try --ball_conf 0.05 or adjust --ball_slice/--ball_imgsz.")

    # control ratio query points (pitch only)
    base_pitch = draw_pitch(config=CONFIG)
    query_pts_cm = build_pitch_query_points_cm(base_pitch, step_px=4, nonzero_th=5)

    control_text = None
    if transformer is not None:
        xy_img = merged.get_anchors_coordinates(anchor=sv.Position.BOTTOM_CENTER)
        xy_pitch = transformer.transform_points(xy_img)
        xy_pitch[:, 1] += RADAR_Y_OFFSET

        team0_xy = xy_pitch[color_lookup == 0]
        team1_xy = xy_pitch[color_lookup == 1]

        t0, t1 = compute_voronoi_control_ratio(query_pts_cm, team0_xy, team1_xy)
        # 表示ラベルは好みで変更OK（ここは team0/team1 をそのまま表示）
        control_text = f"Control (pitch only)  blue:{t0*100:5.1f}%  red:{t1*100:5.1f}%"

    # --- outputs ---
    stem = image_path.stem

    # (A) radar pitch (players+ball) を作って…
    radar_pitch = render_radar_pitch_only(
        detections=merged,
        transformer=transformer,
        color_lookup=color_lookup,
        ball_xy_img=ball_xy_img,
    )

    # (B) それを元画像に重ねた “overlay” を保存（←あなたの希望の出力）
    radar_overlay = overlay_bottom_center(frame.copy(), radar_pitch, opacity=args.overlay_opacity)

    # (C) voronoi pitch は今まで通り（変更なしの方針）
    voronoi_pitch = render_voronoi_pitch_only(
        detections=merged,
        transformer=transformer,
        color_lookup=color_lookup,
        ball_xy_img=ball_xy_img,
        voronoi_opacity=args.voronoi_opacity,
        control_text=control_text,
    )

    radar_overlay_path = out_dir / f"{stem}_radar_overlay.png"
    vor_path          = out_dir / f"{stem}_voronoi.png"

    cv2.imwrite(str(radar_overlay_path), radar_overlay)
    cv2.imwrite(str(vor_path), voronoi_pitch)

    print(f"[3/3] done.")
    print(f"  radar overlay -> {radar_overlay_path}")
    print(f"  voronoi pitch -> {vor_path}")


if __name__ == "__main__":
    main()
