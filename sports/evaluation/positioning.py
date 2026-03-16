import csv
import os
import numpy as np
from collections import defaultdict
from typing import Dict, List, Optional


class PositioningEvaluator:
    """
    フットサルにおけるチームポジショニング評価クラス

    30秒ウィンドウごとに以下を算出：
    - チーム重心（centroid）
    - コンパクトさ（dispersion）
    - 横幅（width）
    - 縦幅（depth）
    - 攻守フェーズ（phase）
    """

    def __init__(
        self,
        fps: float,
        window_sec: int = 30,
        pitch_center_x: float = 0.0,
        out_dir: str = "outputs"
    ):
        self.fps = fps
        self.window_sec = window_sec
        self.frames_per_window = int(fps * window_sec)
        self.pitch_center_x = pitch_center_x

        self.out_dir = out_dir
        os.makedirs(out_dir, exist_ok=True)

        # window -> team -> [positions]
        self._buffer = defaultdict(lambda: defaultdict(list))
        self._ball_buffer = defaultdict(list)

        self._results: List[Dict] = []
        self._current_window = 0

    # =========================================================
    # 毎フレーム更新
    # =========================================================
    def update(
        self,
        frame_idx: int,
        team_id: int,
        positions: np.ndarray,
        ball_xy: Optional[np.ndarray],
        match_id: str
    ):
        if positions is None or len(positions) == 0:
            return

        window_id = frame_idx // self.frames_per_window

        if window_id != self._current_window:
            self._flush_window(match_id)
            self._current_window = window_id

        self._buffer[window_id][team_id].append(positions)

        if ball_xy is not None and len(ball_xy) > 0:
            self._ball_buffer[window_id].append(ball_xy[0][0])

    # =========================================================
    # window確定処理
    # =========================================================
    def _flush_window(self, match_id: str):
        window_id = self._current_window

        if window_id not in self._buffer:
            return

        start_sec = window_id * self.window_sec
        end_sec = start_sec + self.window_sec

        ball_x_mean = None
        if len(self._ball_buffer[window_id]) > 0:
            ball_x_mean = float(np.mean(self._ball_buffer[window_id]))

        for team_id, frames_positions in self._buffer[window_id].items():
            all_positions = np.vstack(frames_positions)

            # --- 重心 ---
            centroid = all_positions.mean(axis=0)

            # --- コンパクトさ ---
            dispersion = float(
                np.mean(np.linalg.norm(all_positions - centroid, axis=1))
            )

            # --- 横幅・縦幅 ---
            width = float(all_positions[:, 0].max() - all_positions[:, 0].min())
            depth = float(all_positions[:, 1].max() - all_positions[:, 1].min())

            # --- 攻守判定 ---
            phase = self._judge_phase(team_id, ball_x_mean)

            self._results.append({
                "match_id": match_id,
                "window_id": window_id,
                "start_sec": start_sec,
                "end_sec": end_sec,
                "team_id": team_id,
                "phase": phase,
                "centroid_x": centroid[0],
                "centroid_y": centroid[1],
                "dispersion": dispersion,
                "width": width,
                "depth": depth
            })

        del self._buffer[window_id]
        if window_id in self._ball_buffer:
            del self._ball_buffer[window_id]

    # =========================================================
    # 攻守判定
    # =========================================================
    def _judge_phase(self, team_id: int, ball_x: Optional[float]) -> str:
        if ball_x is None:
            return "unknown"

        if team_id == 0:
            return "attack" if ball_x > self.pitch_center_x else "defense"
        else:
            return "attack" if ball_x < self.pitch_center_x else "defense"

    # =========================================================
    # 結果取得
    # =========================================================
    def get_results(self) -> List[Dict]:
        return self._results

    # =========================================================
    # CSV出力
    # =========================================================
    def export_csv(self, filename: str):
        path = os.path.join(self.out_dir, filename)

        with open(path, "w", newline="") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=[
                    "match_id",
                    "window_id",
                    "start_sec",
                    "end_sec",
                    "team_id",
                    "phase",
                    "centroid_x",
                    "centroid_y",
                    "dispersion",
                    "width",
                    "depth"
                ]
            )
            writer.writeheader()
            for r in self._results:
                writer.writerow(r)

        print(f"[INFO] positioning csv saved -> {path}")
