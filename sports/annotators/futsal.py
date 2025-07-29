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
    return int(x * scale) + pad, int(y * scale) + pad


# ─────────────────────────────────────────────
#  ピッチ描画
# ─────────────────────────────────────────────
def draw_pitch(
    config: FutsalPitchConfiguration,
    background_color: sv.Color = sv.Color(193, 138, 58),
    line_color: sv.Color = sv.Color.WHITE,
    padding: int = 50,
    line_thickness: int = 4,
    point_radius: int = 8,
    scale: float = 0.25,
) -> np.ndarray:
    """フットサルピッチを描画する（半円ゴールエリア・PK4点・三角除去）"""

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
    goal_r_px = int(config.penalty_spot_distance * scale)  # 半径 6 m
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

    # ── PK スポット４点 ──────────────────────
    spots_cm = [
        (config.penalty_spot_distance, config.width / 2),                       # 左第１
        (config.length - config.penalty_spot_distance, config.width / 2),       # 右第１ 
        (config.goal_area_length + 700, config.width / 2),                            # 左第２
        (config.length - config.goal_area_length - 700, config.width / 2),            # 右第２
    ]
    for pt in spots_cm:
        cv2.circle(
            canvas,
            to_px(pt, scale, padding),
            point_radius,
            line_color.as_bgr(),
            -1,
        )

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
    img = draw_pitch(cfg)
    cv2.imshow("Futsal Pitch", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
