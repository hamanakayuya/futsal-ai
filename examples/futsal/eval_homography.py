# examples/futsal/eval_homography.py
"""
目的:
  2Dレーダービュー（ホモグラフィ変換）の「推定がうまくできているか」を
  30秒動画など“video単位”で定量評価するためのスクリプト。

やっていること（GTなしでできる範囲の定量評価）:
  - ピッチキーポイント検出 → RANSACでホモグラフィ推定
  - 推定ホモグラフィの品質を、キーポイントの再投影誤差で評価
    1) pitch誤差（cm）： H(image->pitch) を使って src_kp を pitchへ写像し、target(=CONFIG.vertices) と比較
    2) image誤差（px）： invH(pitch->image) を使って target を imageへ写像し、src_kp と比較
  - さらに、使用KP数、inlier数、spread(画面内での広がり) などもログ化

出力:
  - out_csv にフレームごとの評価値を保存
  - 任意で debug_dir にデバッグ画像（検出点と再投影点の可視化）を保存
"""

import argparse
import os
from pathlib import Path
from typing import Optional, Tuple, Dict

import cv2
import numpy as np
import supervision as sv
from ultralytics import YOLO
from tqdm import tqdm

from sports.configs.futsal import FutsalPitchConfiguration


# -----------------------------
# utils
# -----------------------------
def _keypoints_conf_array(keypoints: sv.KeyPoints) -> Optional[np.ndarray]:
    """
    sv.KeyPoints から confidence 配列を取り出す（環境差対策）。
    shape: (K,)
    """
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


def _compute_spread_px(src_xy: np.ndarray) -> float:
    """点群が画面内でどれだけ広がっているか（最大レンジ）"""
    if src_xy is None or len(src_xy) == 0:
        return 0.0
    dx = float(src_xy[:, 0].max() - src_xy[:, 0].min())
    dy = float(src_xy[:, 1].max() - src_xy[:, 1].min())
    return max(dx, dy)


def _project_points(H: np.ndarray, pts: np.ndarray) -> np.ndarray:
    """
    Hでpts(N,2)を射影
    """
    if pts is None or len(pts) == 0:
        return np.empty((0, 2), dtype=np.float32)
    pts_ = pts.reshape(-1, 1, 2).astype(np.float32)
    out = cv2.perspectiveTransform(pts_, H.astype(np.float32))
    return out.reshape(-1, 2).astype(np.float32)


def estimate_homography_from_keypoints(
    keypoints: sv.KeyPoints,
    config: FutsalPitchConfiguration,
    kp_conf_th: float = 0.0,
    min_kp: int = 8,
    min_inliers: int = 6,
    ransac_th: float = 8.0,
    min_spread_px: float = 120.0,
) -> Tuple[Optional[np.ndarray], Dict]:
    """
    keypoints.xy[0] (image) と config.vertices (pitch, cm) を対応させ、
    cv2.findHomography(src=image -> tgt=pitch) を推定する。
    """
    info = {"used_kp": 0, "inliers": 0, "spread": 0.0, "ok": False}

    if keypoints is None or keypoints.xy is None or len(keypoints.xy) == 0:
        return None, info

    pts_img_all = keypoints.xy[0].astype(np.float32)          # (K,2)
    pts_pitch_all = np.array(config.vertices, np.float32)     # (K,2)

    if pts_img_all.shape[0] != pts_pitch_all.shape[0]:
        # ラベル数がズレていたら評価不能（ここは環境依存）
        return None, info

    valid = (pts_img_all[:, 0] > 1) & (pts_img_all[:, 1] > 1)

    conf = _keypoints_conf_array(keypoints)
    if conf is not None:
        conf = conf.astype(np.float32)
        valid = valid & (conf >= float(kp_conf_th))

    used = int(np.sum(valid))
    info["used_kp"] = used
    if used < int(min_kp):
        return None, info

    src = pts_img_all[valid]
    tgt = pts_pitch_all[valid]

    spread = _compute_spread_px(src)
    info["spread"] = float(spread)
    if spread < float(min_spread_px):
        return None, info

    H, inlier_mask = cv2.findHomography(
        src, tgt,
        method=cv2.RANSAC,
        ransacReprojThreshold=float(ransac_th),
    )
    if H is None or inlier_mask is None:
        return None, info

    inliers = int(np.sum(inlier_mask))
    info["inliers"] = inliers
    if inliers < int(min_inliers):
        return None, info

    info["ok"] = True
    return H.astype(np.float32), info


def compute_reprojection_errors(
    H_img2pitch: np.ndarray,
    src_img: np.ndarray,
    tgt_pitch: np.ndarray,
) -> Tuple[float, float]:
    """
    - pitch RMSE (cm):  H(image->pitch)でsrc_img→pitchに写像し、tgt_pitchと比較
    - image RMSE (px): invH(pitch->image)でtgt_pitch→imageに写像し、src_imgと比較
    """
    pred_pitch = _project_points(H_img2pitch, src_img)
    diff_pitch = pred_pitch - tgt_pitch
    rmse_cm = float(np.sqrt(np.mean(np.sum(diff_pitch * diff_pitch, axis=1))))

    # 逆変換
    H_pitch2img = np.linalg.inv(H_img2pitch).astype(np.float32)
    pred_img = _project_points(H_pitch2img, tgt_pitch)
    diff_img = pred_img - src_img
    rmse_px = float(np.sqrt(np.mean(np.sum(diff_img * diff_img, axis=1))))

    return rmse_cm, rmse_px


def draw_debug(
    frame: np.ndarray,
    src_img: np.ndarray,
    tgt_pitch: np.ndarray,
    H_img2pitch: np.ndarray,
) -> np.ndarray:
    """
    デバッグ可視化:
      - 緑: 検出キーポイント（image）
      - 赤: pitch頂点をinvHでimageへ戻した点（=再投影点）
    """
    out = frame.copy()

    H_pitch2img = np.linalg.inv(H_img2pitch).astype(np.float32)
    reproj_img = _project_points(H_pitch2img, tgt_pitch)

    for (x, y) in src_img.astype(int):
        cv2.circle(out, (int(x), int(y)), 4, (0, 255, 0), -1)

    for (x, y) in reproj_img.astype(int):
        cv2.circle(out, (int(x), int(y)), 4, (0, 0, 255), -1)

    return out


# -----------------------------
# main
# -----------------------------
def main():
    parser = argparse.ArgumentParser(description="Evaluate homography quality on a video by keypoint reprojection errors.")
    parser.add_argument("--source_video_path", type=str, required=True)
    parser.add_argument("--pitch_model_path", type=str, default=None,
                        help="省略時は examples/futsal/data/best_futsal-pitch-detection.pt を使用")
    parser.add_argument("--device", type=str, default="cpu")

    # sampling
    parser.add_argument("--stride", type=int, default=1, help="何フレームに1回評価するか")
    parser.add_argument("--imgsz", type=int, default=1280)

    # homography filters
    parser.add_argument("--kp_conf", type=float, default=0.0)
    parser.add_argument("--min_kp", type=int, default=8)
    parser.add_argument("--min_inliers", type=int, default=6)
    parser.add_argument("--ransac_th", type=float, default=8.0)
    parser.add_argument("--min_spread_px", type=float, default=120.0)

    # outputs
    parser.add_argument("--out_csv", type=str, required=True)
    parser.add_argument("--debug_dir", type=str, default=None, help="指定するとデバッグ画像を書き出す")
    parser.add_argument("--debug_stride", type=int, default=60, help="デバッグ画像を何フレームに1回保存するか")

    args = parser.parse_args()

    src = Path(args.source_video_path)
    if not src.exists():
        raise FileNotFoundError(f"source_video_path not found: {src}")

    # default pitch model path
    if args.pitch_model_path is None:
        parent = Path(__file__).resolve().parent
        args.pitch_model_path = str(parent / "data" / "best_futsal-pitch-detection.pt")

    pitch_model_path = Path(args.pitch_model_path)
    if not pitch_model_path.exists():
        raise FileNotFoundError(f"pitch_model_path not found: {pitch_model_path}")

    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    debug_dir = Path(args.debug_dir) if args.debug_dir else None
    if debug_dir:
        debug_dir.mkdir(parents=True, exist_ok=True)

    config = FutsalPitchConfiguration()
    tgt_all = np.array(config.vertices, dtype=np.float32)  # (K,2)

    pitch_model = YOLO(str(pitch_model_path)).to(device=args.device)

    cap = cv2.VideoCapture(str(src))
    if not cap.isOpened():
        raise RuntimeError(f"failed to open video: {src}")

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) if cap.get(cv2.CAP_PROP_FRAME_COUNT) > 0 else None

    rows = []
    frame_idx = 0
    pbar = tqdm(total=total, desc="eval") if total is not None else tqdm(desc="eval")

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        if args.stride > 1 and (frame_idx % args.stride != 0):
            frame_idx += 1
            pbar.update(1)
            continue

        # pitch keypoints
        res = pitch_model(frame, imgsz=args.imgsz, verbose=False)[0]
        keypoints = sv.KeyPoints.from_ultralytics(res)

        H, info = estimate_homography_from_keypoints(
            keypoints=keypoints,
            config=config,
            kp_conf_th=args.kp_conf,
            min_kp=args.min_kp,
            min_inliers=args.min_inliers,
            ransac_th=args.ransac_th,
            min_spread_px=args.min_spread_px,
        )

        rmse_cm = None
        rmse_px = None

        if H is not None and info.get("ok", False):
            pts_img_all = keypoints.xy[0].astype(np.float32)

            # valid maskを再現
            valid = (pts_img_all[:, 0] > 1) & (pts_img_all[:, 1] > 1)
            conf = _keypoints_conf_array(keypoints)
            if conf is not None:
                valid = valid & (conf.astype(np.float32) >= float(args.kp_conf))

            src_used = pts_img_all[valid]
            tgt_used = tgt_all[valid]

            if len(src_used) >= 4:
                rmse_cm, rmse_px = compute_reprojection_errors(H, src_used, tgt_used)

                if debug_dir and (frame_idx % max(1, args.debug_stride) == 0):
                    dbg = draw_debug(frame, src_used, tgt_used, H)
                    cv2.imwrite(str(debug_dir / f"dbg_{frame_idx:06d}.jpg"), dbg)

        rows.append({
            "frame_idx": frame_idx,
            "used_kp": info.get("used_kp", 0),
            "inliers": info.get("inliers", 0),
            "spread_px": info.get("spread", 0.0),
            "ok": int(bool(info.get("ok", False))),
            "rmse_pitch_cm": "" if rmse_cm is None else f"{rmse_cm:.3f}",
            "rmse_image_px": "" if rmse_px is None else f"{rmse_px:.3f}",
        })

        frame_idx += 1
        pbar.update(1)

    pbar.close()
    cap.release()

    # write CSV
    header = ["frame_idx", "used_kp", "inliers", "spread_px", "ok", "rmse_pitch_cm", "rmse_image_px"]
    with out_csv.open("w", encoding="utf-8") as f:
        f.write(",".join(header) + "\n")
        for r in rows:
            f.write(",".join(str(r[h]) for h in header) + "\n")

    print("[done]")
    print(f"  video : {src}")
    print(f"  csv   : {out_csv}")
    if debug_dir:
        print(f"  debug : {debug_dir}")


if __name__ == "__main__":
    main()
