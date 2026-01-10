from typing import Optional, List, Tuple

import cv2
import numpy as np
import supervision as sv

from sports.configs.futsal import FutsalPitchConfiguration


# ─────────────────────────────────────────────
#  便利関数: cm 座標 → 画像ピクセル
# ─────────────────────────────────────────────
def to_px(pt: Tuple[float, float], scale: float, pad: int) -> Tuple[int, int]:
    """
    pt: (x_cm, y_cm)
    return: (x_px, y_px)
    """
    x, y = pt
    return int(round(x * scale)) + pad, int(round(y * scale)) + pad


# ─────────────────────────────────────────────
#  ピッチ描画
# ─────────────────────────────────────────────
def draw_pitch(
    config: FutsalPitchConfiguration,
    background_color: sv.Color = sv.Color(34, 139, 34),
    line_color: sv.Color = sv.Color.WHITE,
    padding: int = 50,
    line_thickness: int = 4,
    point_radius: int = 8,
    scale: float = 0.25,
    # 追加: デバッグ表示/ゴール描画
    draw_keypoints: bool = False,
    draw_edges: bool = False,
    draw_goal_frames: bool = True,
    keypoint_radius: int = 10,
    keypoint_fill_color: sv.Color = sv.Color.from_hex("#00BFFF"),  # 水色
    keypoint_text_color: sv.Color = sv.Color.WHITE,
) -> np.ndarray:
    """フットサルピッチを描画する（半円ゴールエリア・PK4点・テンプレ点/線オーバーレイ・ゴール枠）"""

    img_h = int(config.width * scale) + padding * 2
    img_w = int(config.length * scale) + padding * 2
    canvas = np.ones((img_h, img_w, 3), dtype=np.uint8)
    canvas[:] = background_color.as_bgr()

    # ── 外枠（長方形） ─────────────────────────
    tl = (padding, padding)
    br = (padding + int(config.length * scale), padding + int(config.width * scale))
    cv2.rectangle(canvas, tl, br, line_color.as_bgr(), line_thickness)

    # ── センターライン ─────────────────────────
    mid_x = padding + int(config.length * scale / 2)
    cv2.line(
        canvas,
        (mid_x, padding),
        (mid_x, br[1]),
        line_color.as_bgr(),
        line_thickness,
    )

    # ── センターサークル ──────────────────────
    centre = (mid_x, padding + int(config.width * scale / 2))
    cv2.circle(
        canvas,
        centre,
        int(config.centre_circle_radius * scale),
        line_color.as_bgr(),
        line_thickness,
    )

    # ── ゴールエリア半円（左右） ───────────────
    goal_r_px = int(config.penalty_spot_distance * scale)  # 半径 6m (=600cm) 想定
    mid_y = centre[1]

    # 左側（角度 -90〜+90°相当の見た目にする）
    left_center = (padding, mid_y)
    cv2.ellipse(
        canvas,
        left_center,
        (goal_r_px, goal_r_px),
        180,
        90,
        270,
        line_color.as_bgr(),
        line_thickness,
    )

    # 右側
    right_center = (padding + int(config.length * scale), mid_y)
    cv2.ellipse(
        canvas,
        right_center,
        (goal_r_px, goal_r_px),
        0,
        90,
        270,
        line_color.as_bgr(),
        line_thickness,
    )

    # ── PK スポット４点（既存ロジック） ──────────────────────
    spots_cm = [
        (config.penalty_spot_distance, config.width / 2),                       # 左第１
        (config.length - config.penalty_spot_distance, config.width / 2),       # 右第１
        (config.goal_area_length + 700, config.width / 2),                      # 左第２（既存仕様）
        (config.length - config.goal_area_length - 700, config.width / 2),      # 右第２（既存仕様）
    ]
    for pt in spots_cm:
        cv2.circle(
            canvas,
            to_px(pt, scale, padding),
            point_radius,
            line_color.as_bgr(),
            -1,
        )

    # ─────────────────────────────────────────────
    #  ゴール枠（会場のゴールポストを描画）
    #  ※ キーポイント 03,04 / 35,36 がゴールポストと仮定
    # ─────────────────────────────────────────────
    if draw_goal_frames:
        verts = config.vertices

        left_post_top = verts[3 - 1]
        left_post_bottom = verts[4 - 1]
        right_post_top = verts[35 - 1]
        right_post_bottom = verts[36 - 1]

        goal_depth_cm = 100.0  # 見た目用の奥行き

        left_back_top = (left_post_top[0] - goal_depth_cm, left_post_top[1])
        left_back_bottom = (left_post_bottom[0] - goal_depth_cm, left_post_bottom[1])
        right_back_top = (right_post_top[0] + goal_depth_cm, right_post_top[1])
        right_back_bottom = (right_post_bottom[0] + goal_depth_cm, right_post_bottom[1])

        def draw_goal(post_top, post_bottom, back_top, back_bottom):
            p1 = to_px(post_top, scale, padding)
            p2 = to_px(post_bottom, scale, padding)
            b1 = to_px(back_top, scale, padding)
            b2 = to_px(back_bottom, scale, padding)

            cv2.line(canvas, p1, p2, line_color.as_bgr(), line_thickness)
            cv2.line(canvas, p1, b1, line_color.as_bgr(), line_thickness)
            cv2.line(canvas, p2, b2, line_color.as_bgr(), line_thickness)
            cv2.line(canvas, b1, b2, line_color.as_bgr(), line_thickness)

        draw_goal(left_post_top, left_post_bottom, left_back_top, left_back_bottom)
        draw_goal(right_post_top, right_post_bottom, right_back_top, right_back_bottom)

    # ─────────────────────────────────────────────
    #  edges（テンプレ線）をオーバーレイ
    # ─────────────────────────────────────────────
    if draw_edges and hasattr(config, "edges"):
        for a, b in config.edges:
            p1 = config.vertices[a - 1]
            p2 = config.vertices[b - 1]
            cv2.line(
                canvas,
                to_px(p1, scale, padding),
                to_px(p2, scale, padding),
                line_color.as_bgr(),
                max(1, int(round(line_thickness * 0.75))),
            )

    # ─────────────────────────────────────────────
    #  キーポイント(01-38) + ラベル描画
    # ─────────────────────────────────────────────
    if draw_keypoints and hasattr(config, "vertices"):
        labels = getattr(config, "labels", None)
        if labels is None:
            labels = [f"{i+1:02}" for i in range(len(config.vertices))]

        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5
        font_th = 1

        for i, pt in enumerate(config.vertices):
            p = to_px(pt, scale, padding)

            cv2.circle(canvas, p, keypoint_radius, keypoint_fill_color.as_bgr(), -1)
            cv2.circle(canvas, p, keypoint_radius, sv.Color.BLACK.as_bgr(), 2)

            text = labels[i] if i < len(labels) else f"{i+1:02}"
            tx, ty = p[0] + 6, p[1] - 6

            cv2.putText(canvas, text, (tx, ty), font, font_scale, (0, 0, 0), 3, cv2.LINE_AA)
            cv2.putText(canvas, text, (tx, ty), font, font_scale, keypoint_text_color.as_bgr(), font_th, cv2.LINE_AA)

    return canvas


# ─────────────────────────────────────────────
#  点描画
# ─────────────────────────────────────────────
def draw_points_on_pitch(
    config: FutsalPitchConfiguration,
    xy: np.ndarray,
    face_color: sv.Color = sv.Color.RED,
    edge_color: sv.Color = sv.Color.BLACK,
    radius: int = 10,
    thickness: int = 2,
    padding: int = 50,
    scale: float = 0.25,
    pitch: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    xy: (N,2) in cm
    """
    if pitch is None:
        pitch = draw_pitch(config, padding=padding, scale=scale)

    xy = np.asarray(xy)
    if xy.size == 0:
        return pitch
    xy = xy.reshape(-1, 2)

    for point in xy:
        p = to_px((float(point[0]), float(point[1])), scale, padding)
        cv2.circle(pitch, p, radius, face_color.as_bgr(), -1)
        cv2.circle(pitch, p, radius, edge_color.as_bgr(), thickness)
    return pitch


# ─────────────────────────────────────────────
#  経路描画
# ─────────────────────────────────────────────
def draw_paths_on_pitch(
    config: FutsalPitchConfiguration,
    paths: List[np.ndarray],
    color: sv.Color = sv.Color.WHITE,
    thickness: int = 2,
    padding: int = 50,
    scale: float = 0.25,
    pitch: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    paths: List[(Ti,2)] in cm
    """
    if pitch is None:
        pitch = draw_pitch(config, padding=padding, scale=scale)

    for path in paths:
        if path is None:
            continue
        path = np.asarray(path)
        if path.size == 0:
            continue
        path = path.reshape(-1, 2)

        pts = [to_px((float(p[0]), float(p[1])), scale, padding) for p in path]
        if len(pts) < 2:
            continue
        for i in range(len(pts) - 1):
            cv2.line(pitch, pts[i], pts[i + 1], color.as_bgr(), thickness)

    return pitch


# ─────────────────────────────────────────────
#  Voronoi 図（soccer.py から完全移植 + futsal座標(cm)対応）
# ─────────────────────────────────────────────
def draw_pitch_voronoi_diagram(
    config: FutsalPitchConfiguration,
    team_1_xy: np.ndarray,
    team_2_xy: np.ndarray,
    team_1_color: sv.Color = sv.Color.RED,
    team_2_color: sv.Color = sv.Color.WHITE,
    opacity: float = 0.5,
    padding: int = 50,
    scale: float = 0.25,
    pitch: Optional[np.ndarray] = None
) -> np.ndarray:
    """
    フットサルピッチ上に Voronoi 図（支配領域）を描画する。

    IMPORTANT:
      - team_1_xy / team_2_xy は「cm座標系」
      - pitch を draw_pitch() で作った場合は、同じ padding/scale を必ず渡す
    """

    # ベースピッチ生成（scale/padding を揃える）
    if pitch is None:
        pitch = draw_pitch(
            config=config,
            padding=padding,
            scale=scale
        )

    # (N,2) float32 に揃える（空でもOK）
    def _normalize_xy(xy: np.ndarray) -> np.ndarray:
        if xy is None:
            return np.empty((0, 2), dtype=np.float32)
        xy = np.asarray(xy)
        if xy.size == 0:
            return np.empty((0, 2), dtype=np.float32)
        return xy.reshape(-1, 2).astype(np.float32)

    team_1_xy = _normalize_xy(team_1_xy)
    team_2_xy = _normalize_xy(team_2_xy)

    # 期待される画像サイズ（pitch 生成時の理想サイズ）
    scaled_width = int(config.width * scale)
    scaled_length = int(config.length * scale)
    expected_h = scaled_width + 2 * padding
    expected_w = scaled_length + 2 * padding

    # 実際の pitch サイズ（差があっても破綻しないように、実サイズに合わせる）
    img_h, img_w = pitch.shape[:2]
    if (img_h, img_w) != (expected_h, expected_w):
        # pitch が別設定で作られている可能性があるので、ここでは警告は出さず
        # “実サイズ”を優先して Voronoi グリッドを作る。
        pass

    # Voronoi 塗り
    voronoi = np.zeros_like(pitch, dtype=np.uint8)
    c1 = np.array(team_1_color.as_bgr(), dtype=np.uint8)
    c2 = np.array(team_2_color.as_bgr(), dtype=np.uint8)

    # ピクセル座標グリッド（(0,0) をピッチ左上に合わせるため padding を引く）
    y_coords, x_coords = np.indices((img_h, img_w), dtype=np.float32)
    x_coords -= float(padding)
    y_coords -= float(padding)

    # 最近傍距離（二乗距離）を計算
    def _min_dist2(xy_cm: np.ndarray) -> np.ndarray:
        if xy_cm.shape[0] == 0:
            return np.full((img_h, img_w), np.inf, dtype=np.float32)

        xy_px = xy_cm * float(scale)  # cm -> px

        dx = xy_px[:, 0][:, None, None] - x_coords[None, :, :]
        dy = xy_px[:, 1][:, None, None] - y_coords[None, :, :]
        dist2 = dx * dx + dy * dy
        return np.min(dist2, axis=0)

    min_d2_t1 = _min_dist2(team_1_xy)
    min_d2_t2 = _min_dist2(team_2_xy)

    control_mask = (min_d2_t1 < min_d2_t2)

    voronoi[control_mask] = c1
    voronoi[~control_mask] = c2

    overlay = cv2.addWeighted(voronoi, float(opacity), pitch, 1.0 - float(opacity), 0.0)
    return overlay


# ─────────────────────────────────────────────
#  手元テスト（静止画）
# ─────────────────────────────────────────────
if __name__ == "__main__":
    cfg = FutsalPitchConfiguration()

    SCALE = 0.25
    PAD = 50

    # ベースピッチ（scale/padding を固定して作る）
    base = draw_pitch(
        cfg,
        draw_keypoints=False,
        draw_edges=False,
        draw_goal_frames=True,
        scale=SCALE,
        padding=PAD,
    )

    # 仮の選手座標（cm）
    team1 = np.array([
        [400, 500], [800, 900], [1200, 300], [1500, 1000], [1700, 1400]
    ], dtype=np.float32)
    team2 = np.array([
        [3000, 500], [2800, 900], [2400, 300], [2200, 1000], [2000, 1400]
    ], dtype=np.float32)

    # Voronoi 図（同じ scale/padding を渡す）
    img = draw_pitch_voronoi_diagram(
        config=cfg,
        team_1_xy=team1,
        team_2_xy=team2,
        team_1_color=sv.Color.from_hex("#FF1493"),
        team_2_color=sv.Color.from_hex("#00BFFF"),
        opacity=0.45,
        padding=PAD,
        scale=SCALE,
        pitch=base
    )

    cv2.imwrite("futsal_voronoi_debug.png", img)
    print("saved: futsal_voronoi_debug.png")
