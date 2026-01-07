import argparse
from enum import Enum
from typing import Iterator, List, Optional
from collections import deque

import os
import cv2
import numpy as np
import supervision as sv
from tqdm import tqdm
from ultralytics import YOLO

from sports.annotators.futsal import draw_pitch, draw_points_on_pitch
from sports.common.ball import BallTracker, BallAnnotator
from sports.common.team import TeamClassifier
from sports.common.view import ViewTransformer
from sports.configs.futsal import FutsalPitchConfiguration


PARENT_DIR = os.path.dirname(os.path.abspath(__file__))

PLAYER_DETECTION_MODEL_PATH = os.path.join(PARENT_DIR, 'data/best_futsal-players-detection.pt')
PITCH_DETECTION_MODEL_PATH = os.path.join(PARENT_DIR, 'data/best_futsal-pitch-detection.pt')
BALL_DETECTION_MODEL_PATH = os.path.join(PARENT_DIR, 'data/best_futsal-ball-detection.pt')

BALL_CLASS_ID = 0
GOALKEEPER_CLASS_ID = 1
PLAYER_CLASS_ID = 2
REFEREE_CLASS_ID = 3

STRIDE = 60
CONFIG = FutsalPitchConfiguration()

COLORS = ['#00BFFF', '#FF1493', '#FF6347', '#FFD700']
REFEREE_COLOR_ID = 2

# ★ レーダーY方向オフセット（上に移動）
RADAR_Y_OFFSET = -50

VERTEX_LABEL_ANNOTATOR = sv.VertexLabelAnnotator(
    color=[sv.Color.from_hex(color) for color in CONFIG.colors],
    text_color=sv.Color.from_hex('#FFFFFF'),
    border_radius=5,
    text_thickness=1,
    text_scale=0.5,
    text_padding=5,
)
EDGE_ANNOTATOR = sv.EdgeAnnotator(
    color=sv.Color.from_hex('#FF1493'),
    thickness=2,
    edges=CONFIG.edges,
)
TRIANGLE_ANNOTATOR = sv.TriangleAnnotator(
    color=sv.Color.from_hex('#FF1493'),
    base=20,
    height=15,
)
BOX_ANNOTATOR = sv.BoxAnnotator(
    color=sv.ColorPalette.from_hex(COLORS),
    thickness=2
)
ELLIPSE_ANNOTATOR = sv.EllipseAnnotator(
    color=sv.ColorPalette.from_hex(COLORS),
    thickness=2
)
BOX_LABEL_ANNOTATOR = sv.LabelAnnotator(
    color=sv.ColorPalette.from_hex(COLORS),
    text_color=sv.Color.from_hex('#FFFFFF'),
    text_padding=5,
    text_thickness=1,
)
ELLIPSE_LABEL_ANNOTATOR = sv.LabelAnnotator(
    color=sv.ColorPalette.from_hex(COLORS),
    text_color=sv.Color.from_hex('#FFFFFF'),
    text_padding=5,
    text_thickness=1,
    text_position=sv.Position.BOTTOM_CENTER,
)

BALL_RADAR_COLOR = sv.Color.from_hex('#FFFFFF')
BALL_RADAR_RADIUS = 12
BALL_TRAJ_LENGTH = 20
BALL_TRAJ_RADIUS = 6

BALL_NMS_TH = 0.1
BALL_MIN_CONF = 0.10

PRINT_BALL_DEBUG = False
DRAW_BALL_ON_VIDEO_DEBUG = False
SWAP_TEAMS = False


class Mode(Enum):
    PITCH_DETECTION = 'PITCH_DETECTION'
    PLAYER_DETECTION = 'PLAYER_DETECTION'
    BALL_DETECTION = 'BALL_DETECTION'
    PLAYER_TRACKING = 'PLAYER_TRACKING'
    TEAM_CLASSIFICATION = 'TEAM_CLASSIFICATION'
    RADAR = 'RADAR'


def get_crops(frame: np.ndarray, detections: sv.Detections) -> List[np.ndarray]:
    return [sv.crop_image(frame, xyxy) for xyxy in detections.xyxy]


def resolve_goalkeepers_team_id(players, players_team_id, goalkeepers):
    goalkeepers_xy = goalkeepers.get_anchors_coordinates(sv.Position.BOTTOM_CENTER)
    players_xy = players.get_anchors_coordinates(sv.Position.BOTTOM_CENTER)

    if len(players_team_id) == 0:
        return np.zeros(len(goalkeepers_xy), dtype=int)

    team_0_centroid = players_xy[players_team_id == 0].mean(axis=0)
    team_1_centroid = players_xy[players_team_id == 1].mean(axis=0)

    result = []
    for gk_xy in goalkeepers_xy:
        d0 = np.linalg.norm(gk_xy - team_0_centroid)
        d1 = np.linalg.norm(gk_xy - team_1_centroid)
        result.append(0 if d0 < d1 else 1)
    return np.array(result)


def _infer_ball_class_id_from_model(ball_model: YOLO, fallback: int = 0) -> int:
    try:
        for cid, name in ball_model.names.items():
            if "ball" in name.lower():
                return int(cid)
    except Exception:
        pass
    return fallback


def render_radar(
    detections: sv.Detections,
    keypoints: sv.KeyPoints,
    color_lookup: np.ndarray,
    ball_xy_history: Optional[np.ndarray] = None,
    ball_xy_current: Optional[np.ndarray] = None,
) -> np.ndarray:
    if keypoints is None or keypoints.xy is None or len(keypoints.xy) == 0:
        return draw_pitch(config=CONFIG)

    mask = (keypoints.xy[0][:, 0] > 1) & (keypoints.xy[0][:, 1] > 1)
    if mask.sum() < 4:
        return draw_pitch(config=CONFIG)

    transformer = ViewTransformer(
        source=keypoints.xy[0][mask].astype(np.float32),
        target=np.array(CONFIG.vertices)[mask].astype(np.float32)
    )

    radar = draw_pitch(config=CONFIG)

    # --- players / GK / referee ---
    xy = detections.get_anchors_coordinates(anchor=sv.Position.BOTTOM_CENTER)
    transformed_xy = transformer.transform_points(points=xy)

    # ★ 修正：Y方向にオフセット
    transformed_xy[:, 1] += RADAR_Y_OFFSET

    for cid in range(len(COLORS)):
        radar = draw_points_on_pitch(
            config=CONFIG,
            xy=transformed_xy[color_lookup == cid],
            face_color=sv.Color.from_hex(COLORS[cid]),
            radius=20,
            pitch=radar
        )

    # --- ball trajectory ---
    if ball_xy_history is not None and len(ball_xy_history) > 0:
        hist_t = transformer.transform_points(ball_xy_history.astype(np.float32))
        hist_t[:, 1] += RADAR_Y_OFFSET  # ★ 修正
        radar = draw_points_on_pitch(
            config=CONFIG,
            xy=hist_t,
            face_color=BALL_RADAR_COLOR,
            radius=BALL_TRAJ_RADIUS,
            pitch=radar
        )

    # --- ball current ---
    if ball_xy_current is not None and len(ball_xy_current) > 0:
        cur_t = transformer.transform_points(ball_xy_current.astype(np.float32))
        cur_t[:, 1] += RADAR_Y_OFFSET  # ★ 修正
        radar = draw_points_on_pitch(
            config=CONFIG,
            xy=cur_t,
            face_color=BALL_RADAR_COLOR,
            radius=BALL_RADAR_RADIUS,
            pitch=radar
        )

    return radar


def run_pitch_detection(source_video_path: str, device: str) -> Iterator[np.ndarray]:
    pitch_detection_model = YOLO(PITCH_DETECTION_MODEL_PATH).to(device=device)
    frame_generator = sv.get_video_frames_generator(source_path=source_video_path)

    for frame in frame_generator:
        result = pitch_detection_model(frame, verbose=False)[0]
        keypoints = sv.KeyPoints.from_ultralytics(result)

        annotated_frame = frame.copy()
        annotated_frame = VERTEX_LABEL_ANNOTATOR.annotate(annotated_frame, keypoints, CONFIG.labels)
        yield annotated_frame


def run_player_detection(source_video_path: str, device: str) -> Iterator[np.ndarray]:
    player_detection_model = YOLO(PLAYER_DETECTION_MODEL_PATH).to(device=device)
    frame_generator = sv.get_video_frames_generator(source_path=source_video_path)

    for frame in frame_generator:
        result = player_detection_model(frame, imgsz=1280, verbose=False)[0]
        detections = sv.Detections.from_ultralytics(result)

        annotated_frame = frame.copy()
        annotated_frame = BOX_ANNOTATOR.annotate(annotated_frame, detections)
        annotated_frame = BOX_LABEL_ANNOTATOR.annotate(annotated_frame, detections)
        yield annotated_frame


def run_ball_detection(source_video_path: str, device: str) -> Iterator[np.ndarray]:
    ball_detection_model = YOLO(BALL_DETECTION_MODEL_PATH).to(device=device)
    frame_generator = sv.get_video_frames_generator(source_path=source_video_path)
    ball_tracker = BallTracker(buffer_size=20)
    ball_annotator = BallAnnotator(radius=6, buffer_size=10)

    def callback(image_slice: np.ndarray) -> sv.Detections:
        result = ball_detection_model(image_slice, imgsz=640, verbose=False)[0]
        return sv.Detections.from_ultralytics(result)

    slicer = sv.InferenceSlicer(
        callback=callback,
        overlap_filter=sv.OverlapFilter.NONE,
        slice_wh=(640, 640),
    )

    for frame in frame_generator:
        detections = slicer(frame).with_nms(threshold=BALL_NMS_TH)
        detections = ball_tracker.update(detections)
        annotated_frame = frame.copy()
        annotated_frame = ball_annotator.annotate(annotated_frame, detections)
        yield annotated_frame


def run_player_tracking(source_video_path: str, device: str) -> Iterator[np.ndarray]:
    player_detection_model = YOLO(PLAYER_DETECTION_MODEL_PATH).to(device=device)
    frame_generator = sv.get_video_frames_generator(source_path=source_video_path)
    tracker = sv.ByteTrack(minimum_consecutive_frames=3)

    for frame in frame_generator:
        result = player_detection_model(frame, imgsz=1280, verbose=False)[0]
        detections = sv.Detections.from_ultralytics(result)
        detections = tracker.update_with_detections(detections)

        labels = [str(tracker_id) for tracker_id in detections.tracker_id]
        annotated_frame = frame.copy()
        annotated_frame = ELLIPSE_ANNOTATOR.annotate(annotated_frame, detections)
        annotated_frame = ELLIPSE_LABEL_ANNOTATOR.annotate(annotated_frame, detections, labels=labels)
        yield annotated_frame


def run_team_classification(source_video_path: str, device: str) -> Iterator[np.ndarray]:
    player_detection_model = YOLO(PLAYER_DETECTION_MODEL_PATH).to(device=device)
    frame_generator = sv.get_video_frames_generator(source_path=source_video_path, stride=STRIDE)

    crops = []
    for frame in tqdm(frame_generator, desc='collecting crops'):
        result = player_detection_model(frame, imgsz=1280, verbose=False)[0]
        detections = sv.Detections.from_ultralytics(result)
        crops += get_crops(frame, detections[detections.class_id == PLAYER_CLASS_ID])

    team_classifier = TeamClassifier(device=device)
    team_classifier.fit(crops)

    frame_generator = sv.get_video_frames_generator(source_path=source_video_path)
    tracker = sv.ByteTrack(minimum_consecutive_frames=3)

    for frame in frame_generator:
        result = player_detection_model(frame, imgsz=1280, verbose=False)[0]
        detections = sv.Detections.from_ultralytics(result)
        detections = tracker.update_with_detections(detections)

        players = detections[detections.class_id == PLAYER_CLASS_ID]
        crops = get_crops(frame, players)

        # --- RAW（本来の推定） ---
        players_team_id_raw = team_classifier.predict(crops) if len(crops) > 0 else np.array([])

        # --- GK は RAW の player team で推定 ---
        goalkeepers = detections[detections.class_id == GOALKEEPER_CLASS_ID]
        goalkeepers_team_id_raw = (
            resolve_goalkeepers_team_id(players, players_team_id_raw, goalkeepers)
            if len(goalkeepers) > 0 and len(players_team_id_raw) > 0 else np.array([])
        )

        # ✅ 反転なし：表示も raw のまま
        players_team_id_disp = players_team_id_raw
        goalkeepers_team_id_disp = goalkeepers_team_id_raw

        referees = detections[detections.class_id == REFEREE_CLASS_ID]
        merged = sv.Detections.merge([players, goalkeepers, referees])

        # ✅ color_lookup は「表示用プレイヤー(raw) + GK(raw) + referee固定色」
        color_lookup = np.array(
            (players_team_id_disp.tolist() if len(players_team_id_disp) > 0 else []) +
            (goalkeepers_team_id_disp.tolist() if len(goalkeepers_team_id_disp) > 0 else []) +
            [REFEREE_COLOR_ID] * len(referees)
        )

        labels = [str(tracker_id) for tracker_id in merged.tracker_id]

        annotated_frame = frame.copy()
        annotated_frame = ELLIPSE_ANNOTATOR.annotate(
            annotated_frame, merged, custom_color_lookup=color_lookup
        )
        annotated_frame = ELLIPSE_LABEL_ANNOTATOR.annotate(
            annotated_frame, merged, labels, custom_color_lookup=color_lookup
        )
        yield annotated_frame


def run_radar(source_video_path: str, device: str) -> Iterator[np.ndarray]:
    player_detection_model = YOLO(PLAYER_DETECTION_MODEL_PATH).to(device=device)
    pitch_detection_model = YOLO(PITCH_DETECTION_MODEL_PATH).to(device=device)
    ball_detection_model = YOLO(BALL_DETECTION_MODEL_PATH).to(device=device)

    ball_class_id = _infer_ball_class_id_from_model(ball_detection_model, fallback=BALL_CLASS_ID)

    ball_tracker = BallTracker(buffer_size=20)
    ball_history = deque(maxlen=BALL_TRAJ_LENGTH)

    def ball_callback(image_slice: np.ndarray) -> sv.Detections:
        result = ball_detection_model(image_slice, imgsz=640, verbose=False)[0]
        return sv.Detections.from_ultralytics(result)

    ball_slicer = sv.InferenceSlicer(
        callback=ball_callback,
        overlap_filter=sv.OverlapFilter.NONE,
        slice_wh=(640, 640),
    )

    # --- team classifier fit ---
    crops = []
    for frame in tqdm(
        sv.get_video_frames_generator(source_path=source_video_path, stride=STRIDE),
        desc='collecting crops'
    ):
        result = player_detection_model(frame, imgsz=1280, verbose=False)[0]
        detections = sv.Detections.from_ultralytics(result)
        crops += get_crops(frame, detections[detections.class_id == PLAYER_CLASS_ID])

    team_classifier = TeamClassifier(device=device)
    team_classifier.fit(crops)

    # --- main loop ---
    frame_generator = sv.get_video_frames_generator(source_path=source_video_path)
    tracker = sv.ByteTrack(minimum_consecutive_frames=3)

    for frame in frame_generator:
        # pitch
        pitch_res = pitch_detection_model(frame, verbose=False)[0]
        keypoints = sv.KeyPoints.from_ultralytics(pitch_res)

        # players
        player_res = player_detection_model(frame, imgsz=1280, verbose=False)[0]
        detections = sv.Detections.from_ultralytics(player_res)
        detections = tracker.update_with_detections(detections)

        # ball (slicer)
        ball_dets = ball_slicer(frame).with_nms(threshold=BALL_NMS_TH)
        if hasattr(ball_dets, "class_id") and ball_dets.class_id is not None:
            ball_dets = ball_dets[ball_dets.class_id == ball_class_id]
        if hasattr(ball_dets, "confidence") and ball_dets.confidence is not None and len(ball_dets) > 0:
            ball_dets = ball_dets[ball_dets.confidence >= BALL_MIN_CONF]

        ball_dets = ball_tracker.update(ball_dets)
        if hasattr(ball_dets, "confidence") and ball_dets.confidence is not None and len(ball_dets) > 1:
            best_i = int(np.argmax(ball_dets.confidence))
            ball_dets = ball_dets[[best_i]]

        ball_xy_current = None
        if len(ball_dets) > 0:
            bxy = ball_dets.get_anchors_coordinates(anchor=sv.Position.BOTTOM_CENTER)
            if bxy is not None and len(bxy) > 0:
                ball_xy_current = bxy
                ball_history.append(bxy[0])

        # team classification
        players = detections[detections.class_id == PLAYER_CLASS_ID]
        crops = get_crops(frame, players)

        players_team_id_raw = team_classifier.predict(crops) if len(crops) > 0 else np.array([])

        # GK はRAWで推定
        goalkeepers = detections[detections.class_id == GOALKEEPER_CLASS_ID]
        goalkeepers_team_id_raw = (
            resolve_goalkeepers_team_id(players, players_team_id_raw, goalkeepers)
            if len(goalkeepers) > 0 and len(players_team_id_raw) > 0 else np.array([])
        )

        # ✅ 反転なし：表示も raw のまま
        players_team_id_disp = players_team_id_raw
        goalkeepers_team_id_disp = goalkeepers_team_id_raw

        referees = detections[detections.class_id == REFEREE_CLASS_ID]
        detections_merged = sv.Detections.merge([players, goalkeepers, referees])

        # ✅ color_lookup：プレイヤー(raw) + GK(raw) + referee固定色
        color_lookup = np.array(
            (players_team_id_disp.tolist() if len(players_team_id_disp) > 0 else []) +
            (goalkeepers_team_id_disp.tolist() if len(goalkeepers_team_id_disp) > 0 else []) +
            [REFEREE_COLOR_ID] * len(referees)
        )

        labels = [str(tracker_id) for tracker_id in detections_merged.tracker_id]

        annotated_frame = frame.copy()
        annotated_frame = ELLIPSE_ANNOTATOR.annotate(
            annotated_frame, detections_merged, custom_color_lookup=color_lookup
        )
        annotated_frame = ELLIPSE_LABEL_ANNOTATOR.annotate(
            annotated_frame, detections_merged, labels, custom_color_lookup=color_lookup
        )

        if DRAW_BALL_ON_VIDEO_DEBUG and len(ball_dets) > 0:
            dbg_box = sv.BoxAnnotator(color=sv.ColorPalette.from_hex(["#FFFFFF"]), thickness=2)
            annotated_frame = dbg_box.annotate(annotated_frame, ball_dets)

        ball_hist_np = np.array(ball_history) if len(ball_history) > 0 else None
        radar = render_radar(
            detections_merged,
            keypoints,
            color_lookup,
            ball_xy_history=ball_hist_np,
            ball_xy_current=ball_xy_current
        )

        h, w, _ = frame.shape
        radar = sv.resize_image(radar, (w // 2, h // 2))
        rh, rw, _ = radar.shape
        rect = sv.Rect(x=w // 2 - rw // 2, y=h - rh, width=rw, height=rh)

        annotated_frame = sv.draw_image(annotated_frame, radar, opacity=0.5, rect=rect)
        yield annotated_frame


def main(source_video_path: str, target_video_path: str, device: str, mode: Mode) -> None:
    if mode == Mode.PITCH_DETECTION:
        frame_generator = run_pitch_detection(source_video_path, device)
    elif mode == Mode.PLAYER_DETECTION:
        frame_generator = run_player_detection(source_video_path, device)
    elif mode == Mode.BALL_DETECTION:
        frame_generator = run_ball_detection(source_video_path, device)
    elif mode == Mode.PLAYER_TRACKING:
        frame_generator = run_player_tracking(source_video_path, device)
    elif mode == Mode.TEAM_CLASSIFICATION:
        frame_generator = run_team_classification(source_video_path, device)
    elif mode == Mode.RADAR:
        frame_generator = run_radar(source_video_path, device)
    else:
        raise NotImplementedError(f"Mode {mode} is not implemented.")

    video_info = sv.VideoInfo.from_video_path(source_video_path)
    with sv.VideoSink(target_video_path, video_info) as sink:
        for frame in frame_generator:
            sink.write_frame(frame)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--source_video_path', type=str, required=True)
    parser.add_argument('--target_video_path', type=str, required=True)
    parser.add_argument('--device', type=str, default='cpu')
    parser.add_argument('--mode', type=Mode, default=Mode.PLAYER_DETECTION)
    args = parser.parse_args()
    main(args.source_video_path, args.target_video_path, args.device, args.mode)