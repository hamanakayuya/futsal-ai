from typing import Optional, List, Tuple
import cv2
import numpy as np
import supervision as sv

from sports.configs.futsal import FutsalPitchConfiguration


# ─────────────────────────────────────────────
#  便利関数: cm 座標 → 画像ピクセル
# ─────────────────────────────────────────────
def to_px(pt: Tuple[float, float], scale: float, pad: int) -> Tuple[int, int]:
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

    #   左側（角度 -90〜+90°）
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

    #   右側（角度 +90〜+270°）
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
    #  追加: ゴール枠（ゴールポストを持つ会場向け）
    #  ※ キーポイントの 03,04 / 35,36 がゴールポストとのことなので、
    #     それを使って「ゴールライン外側へ少し張り出した」ゴール枠を描画します。
    # ─────────────────────────────────────────────
    if draw_goal_frames:
        verts = config.vertices
        # 03,04,35,36 は 1-indexed。Python list は 0-indexed。
        left_post_top = verts[3 - 1]
        left_post_bottom = verts[4 - 1]
        right_post_top = verts[35 - 1]
        right_post_bottom = verts[36 - 1]

        # ゴールの奥行き（外側へ何cm張り出すか）: ここは見た目用（会場の縮尺に依存しない）
        # 必要なら 80〜150cm の間で調整してください。
        goal_depth_cm = 100.0

        # 左ゴールは x が負側へ、右ゴールは x が正側へ張り出し
        left_back_top = (left_post_top[0] - goal_depth_cm, left_post_top[1])
        left_back_bottom = (left_post_bottom[0] - goal_depth_cm, left_post_bottom[1])
        right_back_top = (right_post_top[0] + goal_depth_cm, right_post_top[1])
        right_back_bottom = (right_post_bottom[0] + goal_depth_cm, right_post_bottom[1])

        # 線を描画
        def draw_goal(post_top, post_bottom, back_top, back_bottom):
            p1 = to_px(post_top, scale, padding)
            p2 = to_px(post_bottom, scale, padding)
            b1 = to_px(back_top, scale, padding)
            b2 = to_px(back_bottom, scale, padding)

            # ゴールライン上の縦棒（ポスト間）
            cv2.line(canvas, p1, p2, line_color.as_bgr(), line_thickness)

            # 奥行き（上・下）
            cv2.line(canvas, p1, b1, line_color.as_bgr(), line_thickness)
            cv2.line(canvas, p2, b2, line_color.as_bgr(), line_thickness)

            # 奥の縦棒
            cv2.line(canvas, b1, b2, line_color.as_bgr(), line_thickness)

        draw_goal(left_post_top, left_post_bottom, left_back_top, left_back_bottom)
        draw_goal(right_post_top, right_post_bottom, right_back_top, right_back_bottom)

    # ─────────────────────────────────────────────
    #  追加: edges（テンプレ線）をオーバーレイ
    # ─────────────────────────────────────────────
    if draw_edges and hasattr(config, "edges"):
        for a, b in config.edges:
            # edges は 1-indexed の前提
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
    #  追加: キーポイント(01-38) + ラベル描画
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

            # 点
            cv2.circle(canvas, p, keypoint_radius, keypoint_fill_color.as_bgr(), -1)
            cv2.circle(canvas, p, keypoint_radius, sv.Color.BLACK.as_bgr(), 2)

            # ラベル文字（点の右上に出す）
            text = labels[i] if i < len(labels) else f"{i+1:02}"
            tx, ty = p[0] + 6, p[1] - 6

            # 視認性のために黒縁取り → 白文字
            cv2.putText(canvas, text, (tx, ty), font, font_scale, (0, 0, 0), 3, cv2.LINE_AA)
            cv2.putText(canvas, text, (tx, ty), font, font_scale, keypoint_text_color.as_bgr(), font_th, cv2.LINE_AA)

    return canvas


# ─────────────────────────────────────────────
#  その他ユーティリティ（最小限のまま）
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
    if pitch is None:
        pitch = draw_pitch(config, padding=padding, scale=scale)
    for point in xy:
        p = to_px(tuple(point), scale, padding)
        cv2.circle(pitch, p, radius, face_color.as_bgr(), -1)
        cv2.circle(pitch, p, radius, edge_color.as_bgr(), thickness)
    return pitch


def draw_paths_on_pitch(
    config: FutsalPitchConfiguration,
    paths: List[np.ndarray],
    color: sv.Color = sv.Color.WHITE,
    thickness: int = 2,
    padding: int = 50,
    scale: float = 0.25,
    pitch: Optional[np.ndarray] = None,
) -> np.ndarray:
    if pitch is None:
        pitch = draw_pitch(config, padding=padding, scale=scale)
    for path in paths:
        pts = [to_px(tuple(p), scale, padding) for p in path if p.size > 0]
        for i in range(len(pts) - 1):
            cv2.line(pitch, pts[i], pts[i + 1], color.as_bgr(), thickness)
    return pitch


def draw_pitch_voronoi_diagram(*args, **kwargs):
    # 詳細実装は省略（元コード互換のダミー）
    return draw_pitch(*args, **kwargs)


# ─────────────────────────────────────────────
#  手元テスト
# ─────────────────────────────────────────────
if __name__ == "__main__":
    cfg = FutsalPitchConfiguration()

    # 01-38 + edges + goal を重ねた「デバッグ版」
    img = draw_pitch(
        cfg,
        draw_keypoints=False,
        draw_edges=False,
        draw_goal_frames=True,
        scale=0.25,
        padding=50,
    )

    cv2.imwrite("futsal_pitch_debug.png", img)
    print("saved: futsal_pitch_debug.png")
