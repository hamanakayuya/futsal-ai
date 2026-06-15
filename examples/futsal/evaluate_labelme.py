# examples/futsal/evaluate_labelme.py
import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import cv2
import numpy as np
from ultralytics import YOLO

# Hungarian（scipyがあれば最適割当、無ければgreedy）
try:
    from scipy.optimize import linear_sum_assignment
    SCIPY_OK = True
except Exception:
    SCIPY_OK = False

from sports.configs.futsal import FutsalPitchConfiguration

# ============================================================
# Full-court only mode: 反面コート由来のキーポイントを除外
# ============================================================
# 反面コートの線上にあるkp(物理座標が反面30x16基準で定義困難)を除外
HALF_COURT_KP = {9, 11, 12, 14, 15, 17, 22, 24, 25, 27, 28, 30}  # 反面コート由来12点
# 全38点から反面kp(13個)を除いた25点を使う
FULL_COURT_KP = set(range(1, 39)) - HALF_COURT_KP


# ============================================================
# Full-court only mode: 反面コート由来のキーポイントを除外
# ============================================================
# 反面コートの線上にあるkp(物理座標が反面30x16基準で定義困難)を除外
HALF_COURT_KP = {9, 11, 12, 14, 15, 17, 22, 24, 25, 27, 28, 30}  # 反面コート由来12点
# 全38点から反面kp(13個)を除いた25点を使う
FULL_COURT_KP = set(range(1, 39)) - HALF_COURT_KP



# ============================================================
# 1) Labelme reader
# ============================================================
def load_labelme_points(json_path: Path):
    """
    Labelme json から shape_type="point" の点だけ読む想定。
    label (case/space tolerant):
      - "player" (複数)
      - "goalkeeper" / "gk" (あれば) → player扱い
      - "ball" (1個)
      - "01".."38" or "1".."38" (ピッチ基準点)
    """
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    players_px: List[Tuple[float, float]] = []
    ball_px: Optional[Tuple[float, float]] = None
    pitch_kp_px: Dict[int, Tuple[float, float]] = {}

    for sh in data.get("shapes", []):
        if sh.get("shape_type") != "point":
            continue

        label_raw = sh.get("label", "")
        label = (label_raw or "").strip().lower()

        pts = sh.get("points", None)
        if not pts or len(pts) == 0:
            continue

        x, y = float(pts[0][0]), float(pts[0][1])

        if label in ("player", "goalkeeper", "gk"):
            players_px.append((x, y))
        elif label == "ball":
            ball_px = (x, y)
        else:
            # pitch keypoint (full-court only)
            try:
                kid = int(label)  # "01" -> 1
                if kid in FULL_COURT_KP:
                    pitch_kp_px[kid] = (x, y)
            except Exception:
                pass

    return players_px, ball_px, pitch_kp_px


# ============================================================
# 2) Homography
# ============================================================
def to_homography(
    img_pts: List[Tuple[float, float]],
    world_pts: List[Tuple[float, float]],
    ransac_th_px: float = 6.0
) -> Optional[np.ndarray]:
    """image(px) -> world(cm) のホモグラフィ"""
    if len(img_pts) < 4:
        return None
    src = np.array(img_pts, dtype=np.float32)
    dst = np.array(world_pts, dtype=np.float32)
    H, _ = cv2.findHomography(src, dst, method=cv2.RANSAC, ransacReprojThreshold=ransac_th_px)
    return H


def px_to_world(H: np.ndarray, pts_px: List[Tuple[float, float]]) -> np.ndarray:
    if len(pts_px) == 0:
        return np.zeros((0, 2), dtype=np.float32)
    pts = np.array(pts_px, dtype=np.float32).reshape(-1, 1, 2)
    w = cv2.perspectiveTransform(pts, H).reshape(-1, 2)
    return w


# ============================================================
# 3) Matching (players)
# ============================================================
def match_players(gt_xy: np.ndarray, pr_xy: np.ndarray, max_dist_cm: float):
    """
    gt_xy: (G,2), pr_xy: (P,2)
    return: list of (gi, pi, dist_cm)
    """
    G, P = gt_xy.shape[0], pr_xy.shape[0]
    if G == 0 or P == 0:
        return []

    dmat = np.linalg.norm(gt_xy[:, None, :] - pr_xy[None, :, :], axis=2)  # (G,P)

    pairs = []
    if SCIPY_OK:
        gi, pi = linear_sum_assignment(dmat)
        for g, p in zip(gi, pi):
            dist = float(dmat[g, p])
            if dist <= max_dist_cm:
                pairs.append((int(g), int(p), dist))
    else:
        # greedy fallback
        used_g, used_p = set(), set()
        flat = [(dmat[g, p], g, p) for g in range(G) for p in range(P)]
        flat.sort(key=lambda x: x[0])
        for dist, g, p in flat:
            if g in used_g or p in used_p:
                continue
            if dist > max_dist_cm:
                break
            used_g.add(g)
            used_p.add(p)
            pairs.append((int(g), int(p), float(dist)))

    return pairs


def summarize(arr: List[float], name: str):
    if len(arr) == 0:
        print(f"{name}: no data")
        return
    a = np.array(arr, dtype=np.float32)
    mean = float(a.mean())
    med = float(np.median(a))
    p95 = float(np.percentile(a, 95))
    within50 = float((a <= 50.0).mean() * 100.0)
    within100 = float((a <= 100.0).mean() * 100.0)
    print(
        f"{name}: n={len(a)} mean={mean:.1f}cm med={med:.1f}cm p95={p95:.1f}cm  "
        f"<=50cm:{within50:.1f}%  <=100cm:{within100:.1f}%"
    )


# ============================================================
# 4) Inference
# ============================================================
def infer_pitch_keypoints_pose(model: YOLO, image_bgr: np.ndarray, kp_conf: float) -> Dict[int, Tuple[float, float]]:
    """
    pitch pose:
      res.keypoints.xy: (n, K, 2)  K=38
      res.keypoints.conf: (n, K)
    1..38 の dict で返す
    """
    res = model.predict(image_bgr, verbose=False)[0]
    if getattr(res, "keypoints", None) is None or res.keypoints is None:
        return {}

    kps = res.keypoints
    xy = kps.xy  # (n,K,2)
    conf = kps.conf if getattr(kps, "conf", None) is not None else None

    if xy is None or len(xy) == 0:
        return {}

    # インスタンスが複数出たら、平均conf最大を採用
    idx = 0
    if conf is not None and len(conf) > 0:
        mean_conf = conf.mean(axis=1).cpu().numpy()
        idx = int(np.argmax(mean_conf))

    xy_np = xy[idx].cpu().numpy()  # (K,2)
    if conf is not None:
        cf_np = conf[idx].cpu().numpy()
    else:
        cf_np = np.ones((xy_np.shape[0],), dtype=np.float32)

    out: Dict[int, Tuple[float, float]] = {}
    K = xy_np.shape[0]
    for i in range(K):
        if float(cf_np[i]) >= kp_conf:
            kid = i + 1
            if kid in FULL_COURT_KP:  # 反面コートkpは除外
                out[kid] = (float(xy_np[i, 0]), float(xy_np[i, 1]))
    return out


def infer_players_detect(model: YOLO, image_bgr: np.ndarray, det_conf: float) -> List[Tuple[float, float]]:
    """
    detect player:
      bbox bottom-center を足元点として返す
    """
    res = model.predict(image_bgr, verbose=False, conf=det_conf)[0]
    if getattr(res, "boxes", None) is None or res.boxes is None:
        return []
    xyxy = res.boxes.xyxy.cpu().numpy()
    confs = res.boxes.conf.cpu().numpy()

    pts = []
    for bb, cf in zip(xyxy, confs):
        if float(cf) < det_conf:
            continue
        x1, y1, x2, y2 = bb
        cx = (x1 + x2) / 2.0
        fy = y2  # bottom
        pts.append((float(cx), float(fy)))
    return pts


def infer_ball_detect(model: YOLO, image_bgr: np.ndarray, det_conf: float) -> Optional[Tuple[float, float]]:
    """
    detect ball:
      bbox center。複数なら conf 最大を採用
    """
    res = model.predict(image_bgr, verbose=False, conf=det_conf)[0]
    if getattr(res, "boxes", None) is None or res.boxes is None:
        return None

    xyxy = res.boxes.xyxy.cpu().numpy()
    confs = res.boxes.conf.cpu().numpy()
    if len(xyxy) == 0:
        return None

    idx = int(np.argmax(confs))
    x1, y1, x2, y2 = xyxy[idx]
    cx = (x1 + x2) / 2.0
    cy = (y1 + y2) / 2.0
    return (float(cx), float(cy))


# ============================================================
# 5) File pairing
# ============================================================
def collect_pairs(d: Path) -> List[Tuple[Path, Path]]:
    """dir内の *.png を拾い、同名 *.json があるものだけをペアにして返す"""
    pngs = sorted(d.glob("*.png"))
    pairs: List[Tuple[Path, Path]] = []
    for p in pngs:
        j = p.with_suffix(".json")
        if j.exists():
            pairs.append((p, j))
    return pairs


# ============================================================
# 6) Main evaluation
# ============================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True, help="labelme png/json directory (any names ok)")
    ap.add_argument("--players_model", required=True)
    ap.add_argument("--ball_model", required=True)
    ap.add_argument("--pitch_model", required=True)
    ap.add_argument("--out_csv", default="eval_result.csv")

    ap.add_argument("--kp_conf", type=float, default=0.30, help="pitch pose keypoint conf threshold")
    ap.add_argument("--det_conf", type=float, default=0.25, help="player/ball detect conf threshold")
    ap.add_argument("--ransac_th", type=float, default=6.0, help="RANSAC reproj threshold (px)")
    ap.add_argument("--max_match_cm", type=float, default=300.0, help="max distance for player matching (cm)")
    ap.add_argument(
        "--frames",
        type=int,
        default=None,
        help="max number of PNG/JSON pairs to evaluate (default: all pairs)",
    )
    args = ap.parse_args()

    d = Path(args.dir)
    assert d.exists(), f"dir not found: {d}"

    pairs = collect_pairs(d)
    if len(pairs) == 0:
        raise RuntimeError(f"No PNG/JSON pairs found in: {d}")

    if args.frames is not None:
        pairs = pairs[: int(args.frames)]

    cfg = FutsalPitchConfiguration()
    vertices = cfg.vertices  # index 0 -> kp1

    # models
    m_players = YOLO(args.players_model)
    m_ball = YOLO(args.ball_model)
    m_pitch = YOLO(args.pitch_model)

    # stats
    player_err_A: List[float] = []  # end-to-end
    player_err_B: List[float] = []  # object-only
    ball_err_A: List[float] = []
    ball_err_B: List[float] = []

    total_gt_players = 0
    matched_A = 0
    matched_B = 0

    gtH_fail = 0
    prH_fail = 0
    gt_ball_exist = 0
    pr_ball_miss = 0

    rows = []

    for idx, (png, js) in enumerate(pairs, start=1):
        img = cv2.imread(str(png))
        if img is None:
            rows.append([png.name, "READ_FAIL", 0, 0, 0, 0, "", "", 0, 0])
            continue

        # ---- GT from labelme ----
        gt_players_px, gt_ball_px, gt_kp_px = load_labelme_points(js)

        # build GT H
        gt_img_pts, gt_w_pts = [], []
        for kid, (x, y) in gt_kp_px.items():
            gt_img_pts.append((x, y))
            gt_w_pts.append(vertices[kid - 1])  # kp -> vertices
        H_gt = to_homography(gt_img_pts, gt_w_pts, ransac_th_px=args.ransac_th)

        if H_gt is None:
            gtH_fail += 1
            rows.append([png.name, "SKIP_NO_GT_H", len(gt_players_px), 0, 0, 0, "", "", len(gt_img_pts), 0])
            continue

        # ---- Pred from models ----
        pr_kp_px = infer_pitch_keypoints_pose(m_pitch, img, kp_conf=args.kp_conf)

        pr_img_pts, pr_w_pts = [], []
        for kid, (x, y) in pr_kp_px.items():
            pr_img_pts.append((x, y))
            pr_w_pts.append(vertices[kid - 1])
        H_pr = to_homography(pr_img_pts, pr_w_pts, ransac_th_px=args.ransac_th)
        if H_pr is None:
            prH_fail += 1

        pr_players_px = infer_players_detect(m_players, img, det_conf=args.det_conf)
        pr_ball_px = infer_ball_detect(m_ball, img, det_conf=args.det_conf)

        # ---- Convert GT objects to world(cm) using H_gt ----
        gt_players_w = px_to_world(H_gt, gt_players_px)
        total_gt_players += gt_players_w.shape[0]

        gt_ball_w = None
        if gt_ball_px is not None:
            gt_ball_exist += 1
            gt_ball_w = px_to_world(H_gt, [gt_ball_px])[0]

        # ---- A) End-to-end: pred objects via H_pr, GT via H_gt ----
        pr_players_w_A = np.zeros((0, 2), dtype=np.float32)
        pr_ball_w_A = None
        if H_pr is not None:
            pr_players_w_A = px_to_world(H_pr, pr_players_px)
            if pr_ball_px is not None:
                pr_ball_w_A = px_to_world(H_pr, [pr_ball_px])[0]

        # ---- B) Object-only: pred objects via H_gt ----
        pr_players_w_B = px_to_world(H_gt, pr_players_px)
        pr_ball_w_B = None
        if pr_ball_px is not None:
            pr_ball_w_B = px_to_world(H_gt, [pr_ball_px])[0]

        # ---- Ball errors ----
        if gt_ball_w is not None and pr_ball_px is None:
            pr_ball_miss += 1

        if gt_ball_w is not None and pr_ball_w_A is not None:
            ball_err_A.append(float(np.linalg.norm(gt_ball_w - pr_ball_w_A)))
        if gt_ball_w is not None and pr_ball_w_B is not None:
            ball_err_B.append(float(np.linalg.norm(gt_ball_w - pr_ball_w_B)))

        # ---- Player matching ----
        pairsA = match_players(gt_players_w, pr_players_w_A, max_dist_cm=args.max_match_cm)
        pairsB = match_players(gt_players_w, pr_players_w_B, max_dist_cm=args.max_match_cm)

        matched_A += len(pairsA)
        matched_B += len(pairsB)

        for _, _, dist in pairsA:
            player_err_A.append(dist)
        for _, _, dist in pairsB:
            player_err_B.append(dist)

        rows.append([
            png.name,
            "OK",
            len(gt_players_px),
            len(pr_players_px),
            len(pairsA),
            len(pairsB),
            f"{ball_err_A[-1]:.1f}" if (gt_ball_w is not None and pr_ball_w_A is not None) else "",
            f"{ball_err_B[-1]:.1f}" if (gt_ball_w is not None and pr_ball_w_B is not None) else "",
            len(gt_img_pts),   # GT keypoints used
            len(pr_img_pts),   # Pred keypoints used
        ])

    # ---- save CSV ----
    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(out_csv, "w", encoding="utf-8") as f:
        f.write("frame,status,gt_players,pred_players,matched_A,matched_B,ball_err_A_cm,ball_err_B_cm,gt_kp_used,pred_kp_used\n")
        for r in rows:
            f.write(",".join(map(str, r)) + "\n")

    # ---- print summary ----
    print("\n================= SUMMARY =================")
    print(f"Saved CSV: {out_csv}")
    print(f"Pairs evaluated: {len(pairs)}")
    print(f"GT-H fail frames: {gtH_fail}")
    print(f"Pred-H fail frames: {prH_fail}")
    print(f"GT players total: {total_gt_players}")
    print(f"Matched players A(end-to-end): {matched_A} ({(matched_A / max(total_gt_players, 1))*100:.1f}%)")
    print(f"Matched players B(object-only): {matched_B} ({(matched_B / max(total_gt_players, 1))*100:.1f}%)")
    print(f"GT ball frames: {gt_ball_exist}")
    print(f"Ball missed frames (GT exists, pred none): {pr_ball_miss}")

    summarize(player_err_A, "Player error A(end-to-end)")
    summarize(player_err_B, "Player error B(object-only)")
    summarize(ball_err_A,   "Ball error A(end-to-end)")
    summarize(ball_err_B,   "Ball error B(object-only)")
    print("==========================================\n")


if __name__ == "__main__":
    main()
