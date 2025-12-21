from dataclasses import dataclass, field
from typing import List, Tuple
import math


@dataclass
class FutsalPitchConfiguration:
    """
    Roboflowのフィールドキーポイント(01-38)に対応する、会場固有フットサル設定。

    - 通常コート: 34m × 20m (3400cm × 2000cm)
    - 反面コート: 30m × 16m を左右に2面
      - 反面Aと反面Bの「内側タッチライン間」が 1m (=100cm)
      - よって反面コート中心xは 850cm と 2550cm
    - 反面センターサークル半径: 3m (=300cm)
    - 反面ゴールエリア半円半径: 6m (=600cm)
    - 通常ゴールエリア半円半径: 6m (=600cm) ※描画側の仕様と整合
    - 第1PK: 6m (=600cm)
    - 第2PK: 10m (=1000cm)
    - ゴール幅: 3m (=300cm) → ゴールポストy = 1000±150
    """

    # ── 通常コート寸法（cm） ─────────────────────────────
    length: float = 3400
    width: float = 2000

    # ── 通常コート: ゴールエリア半円半径（cm） ───────────
    # annotators/futsal.py 側で「penalty_spot_distance(=600cm)を半円半径として使っている」前提に合わせる
    goal_area_radius: float = 600

    # ── PK（cm） ───────────────────────────────────────
    penalty_mark_distance: float = 600    # 第1PK 6m
    second_penalty_distance: float = 1000 # 第2PK 10m

    # ── センターサークル半径（cm） ─────────────────────
    centre_circle_radius: float = 300

    # ── 反面コート（30×16）設定（cm） ─────────────────
    half_length: float = 3000  # 30m（縦方向に延びる想定）
    half_width: float = 1600   # 16m（横方向）
    half_centre_circle_radius: float = 300
    half_goal_area_radius: float = 600
    gap_between_inner_touchlines: float = 100  # 1m

    # ─────────────────────────────────────────────
    #  幾何ユーティリティ
    # ─────────────────────────────────────────────
    @staticmethod
    def _circle_circle_intersections(
        c1: Tuple[float, float], r1: float,
        c2: Tuple[float, float], r2: float
    ) -> List[Tuple[float, float]]:
        """
        2円の交点（最大2点）を返す。交点なしの場合は空リスト。
        """
        x0, y0 = c1
        x1, y1 = c2
        dx = x1 - x0
        dy = y1 - y0
        d = math.hypot(dx, dy)
        if d == 0:
            return []
        # 交点なし
        if d > r1 + r2 or d < abs(r1 - r2):
            return []
        # 交点が1つ（接する）
        if d == r1 + r2 or d == abs(r1 - r2):
            # 接点
            a = (r1**2 - r2**2 + d**2) / (2*d)
            xm = x0 + a * dx / d
            ym = y0 + a * dy / d
            return [(xm, ym)]

        a = (r1**2 - r2**2 + d**2) / (2*d)
        h_sq = r1**2 - a**2
        h = math.sqrt(max(h_sq, 0.0))

        xm = x0 + a * dx / d
        ym = y0 + a * dy / d

        rx = -dy * (h / d)
        ry =  dx * (h / d)

        p1 = (xm + rx, ym + ry)
        p2 = (xm - rx, ym - ry)
        return [p1, p2]

    def _half_centres_x(self) -> Tuple[float, float]:
        """
        反面A/Bの中心x(cm)を「内側タッチライン間が1m」から算出。
        half_width=1600 なので半幅=800。
        内側タッチライン間: (xB-800) - (xA+800) = 100 → xB-xA=1700
        左右対称: xA=1700-d, xB=1700+d → 2d=1700 → d=850
        よって xA=850, xB=2550
        """
        mid = self.length / 2  # 1700
        d = (self.half_width / 2) + (self.gap_between_inner_touchlines / 2)  # 800 + 50 = 850
        return (mid - d, mid + d)  # (850, 2550)

    # ─────────────────────────────────────────────
    #  38点の座標（cm）
    # ─────────────────────────────────────────────
    @property
    def vertices(self) -> List[Tuple[float, float]]:
        """
        Roboflowのラベル番号(01-38)と一致する順序で返す。
        返却座標は cm 単位 (x:左→右, y:上→下)。
        """
        L = float(self.length)
        W = float(self.width)
        mid_x = L / 2
        mid_y = W / 2

        # 通常ゴールポスト（幅3m → ±1.5m）
        goal_half = 150.0
        y_post_top = mid_y - goal_half   # 850
        y_post_bottom = mid_y + goal_half # 1150

        # 通常ゴールエリア半円（中心はゴールライン中央）
        left_goal_c = (0.0, mid_y)
        right_goal_c = (L, mid_y)
        Rg = float(self.goal_area_radius)  # 600

        # 反面中心（左右）
        halfA_x, halfB_x = self._half_centres_x()

        # 反面センターサークル（中心は通常のハーフライン上 y=1000 とする）
        # ※あなたの会場図の薄い線は通常センターラインの左右に並ぶ想定で、yは同じ中心帯
        halfA_cc = (halfA_x, mid_y)
        halfB_cc = (halfB_x, mid_y)
        Rhc = float(self.half_centre_circle_radius)  # 300

        # 9,11（左）= 反面Aセンターサークル × 通常左ゴールエリア半円
        inter_L = self._circle_circle_intersections(halfA_cc, Rhc, left_goal_c, Rg)
        # 28,30（右）= 反面Bセンターサークル × 通常右ゴールエリア半円
        inter_R = self._circle_circle_intersections(halfB_cc, Rhc, right_goal_c, Rg)

        # 交点2つ前提（上/下）
        # yが小さい方が上
        def sort_ud(pts: List[Tuple[float, float]]) -> Tuple[Tuple[float, float], Tuple[float, float]]:
            pts2 = sorted(pts, key=lambda p: p[1])
            return pts2[0], pts2[1]

        # 安全策：交点が取れない場合は近似（実運用では必ず取れるはず）
        if len(inter_L) >= 2:
            pL_up, pL_dn = sort_ud(inter_L)
        else:
            pL_up = (583.8235, 861.6162)
            pL_dn = (583.8235, 1138.3838)

        if len(inter_R) >= 2:
            pR_up, pR_dn = sort_ud(inter_R)
        else:
            pR_up = (2816.1765, 861.6162)
            pR_dn = (2816.1765, 1138.3838)

        # 反面ゴールエリア半円 × 通常タッチライン（y=0, y=2000）
        # 反面ゴール中心は「反面長さ30m」なので y中心=1000 から ±1500 → -500 / 2500
        # y=0 と交わるとき: (x-cx)^2 + (0 - (-500))^2 = 600^2 → (x-cx)^2 = 600^2 - 500^2 = 110000
        # √110000 ≈ 331.662479...
        dy = 500.0
        r = float(self.half_goal_area_radius)  # 600
        dx_sq = r*r - dy*dy  # 360000 - 250000 = 110000
        dx = math.sqrt(max(dx_sq, 0.0))  # 331.662...

        # 上(y=0)
        top_y = 0.0
        bottom_y = W

        # 反面A：上交点2つ
        A_top1 = (halfA_x - dx, top_y)
        A_top2 = (halfA_x + dx, top_y)
        # 反面B
        B_top1 = (halfB_x - dx, top_y)
        B_top2 = (halfB_x + dx, top_y)

        # 下(y=2000)
        A_bot1 = (halfA_x - dx, bottom_y)
        A_bot2 = (halfA_x + dx, bottom_y)
        B_bot1 = (halfB_x - dx, bottom_y)
        B_bot2 = (halfB_x + dx, bottom_y)

        # 通常センターサークル交点（上下左右）
        # 18/21: センターラインの上下端
        # 19/20: センターラインとセンターサークルの上下交点
        # 16/23: センターサークル左右交点（y=1000）
        C_up = (mid_x, mid_y - self.centre_circle_radius)  # (1700, 700)
        C_dn = (mid_x, mid_y + self.centre_circle_radius)  # (1700, 1300)
        C_left = (mid_x - self.centre_circle_radius, mid_y) # (1400, 1000)
        C_right = (mid_x + self.centre_circle_radius, mid_y) # (2000, 1000)

        # 通常ゴールエリア半円上端/下端が外周(左境界)と交わる点（y=400/1600）
        # 02/05 などで使う
        y_arc_top = mid_y - Rg  # 400
        y_arc_bot = mid_y + Rg  # 1600

        # 07/08,31/32：通常ゴールエリア半円上で y=700/1300 の点（テンプレ形状に合わせる）
        # y=700/1300 は中心から ±300 なので x=√(600^2-300^2)=519.615...
        y1 = mid_y - 300.0  # 700
        y2 = mid_y + 300.0  # 1300
        x_on_goal_arc = math.sqrt(max(Rg*Rg - 300.0*300.0, 0.0))  # 519.615...
        # 左側（中心(0,1000)から右方向）
        L_arc_700 = (x_on_goal_arc, y1)
        L_arc_1300 = (x_on_goal_arc, y2)
        # 右側（中心(L,1000)から左方向）
        R_arc_700 = (L - x_on_goal_arc, y1)
        R_arc_1300 = (L - x_on_goal_arc, y2)

        # ここから38点をRoboflow番号順で並べる
        # ※あなたが確定させた番号の意味に合わせて配置する
        #   - 10/29: 第1PK
        #   - 13/26: 第2PK
        #   - 03/04,35/36: ゴールポスト
        #   - 09/11,28/30: 重なり点（反面センターサークル×通常ゴールエリア半円）
        #   - 12/15/22/25, 14/17/24/27: 重なり点（反面ゴールエリア半円×通常タッチライン）
        return [
            # 01-06: 左外周
            (0.0, 0.0),          # 01 左上コーナー
            (0.0, y_arc_top),    # 02 左外周（ゴールエリア半円上端: y=400）
            (0.0, y_post_top),   # 03 左ゴールポスト（上）
            (0.0, y_post_bottom),# 04 左ゴールポスト（下）
            (0.0, y_arc_bot),    # 05 左外周（ゴールエリア半円下端: y=1600）
            (0.0, W),            # 06 左下コーナー

            # 07-11: 左側（通常ゴールエリア半円＋会場固有点＋PK）
            L_arc_700,           # 07 左ゴールエリア半円上側（y=700付近）
            L_arc_1300,          # 08 左ゴールエリア半円下側（y=1300付近）
            (pL_up[0], pL_up[1]),# 09 重なり点（上）: 反面センターサークル×通常ゴールエリア半円
            (self.penalty_mark_distance, mid_y),  # 10 左 第1PK (6m)
            (pL_dn[0], pL_dn[1]),# 11 重なり点（下）

            # 12-17: 反面ゴールエリア半円×通常タッチライン（左側）
            (A_top1[0], A_top1[1]), # 12 上タッチライン交点（反面A）
            (self.second_penalty_distance, mid_y),# 13 左 第2PK (10m)
            (A_bot1[0], A_bot1[1]), # 14 下タッチライン交点（反面A）
            (A_top2[0], A_top2[1]), # 15 上タッチライン交点（反面A）
            C_left,                  # 16 センターサークル左端
            (A_bot2[0], A_bot2[1]),  # 17 下タッチライン交点（反面A）

            # 18-21: センターライン（上→下）
            (mid_x, 0.0),  # 18 センターライン×上タッチライン
            C_up,          # 19 センターライン×センターサークル（上）
            C_dn,          # 20 センターライン×センターサークル（下）
            (mid_x, W),    # 21 センターライン×下タッチライン

            # 22-27: 反面ゴールエリア半円×通常タッチライン（右側）
            (B_top1[0], B_top1[1]),              # 22 上タッチライン交点（反面B）
            C_right,                              # 23 センターサークル右端
            (B_bot1[0], B_bot1[1]),              # 24 下タッチライン交点（反面B）
            (B_top2[0], B_top2[1]),              # 25 上タッチライン交点（反面B）
            (L - self.second_penalty_distance, mid_y),  # 26 右 第2PK (10m)
            (B_bot2[0], B_bot2[1]),              # 27 下タッチライン交点（反面B）

            # 28-32: 右側（会場固有点＋PK＋通常ゴールエリア半円上/下）
            (pR_up[0], pR_up[1]),                # 28 重なり点（上）: 反面センターサークル×通常ゴールエリア半円
            (L - self.penalty_mark_distance, mid_y),    # 29 右 第1PK (6m)
            (pR_dn[0], pR_dn[1]),                # 30 重なり点（下）
            R_arc_700,                            # 31 右ゴールエリア半円上側（y=700付近）
            R_arc_1300,                           # 32 右ゴールエリア半円下側（y=1300付近）

            # 33-38: 右外周
            (L, 0.0),          # 33 右上コーナー
            (L, y_arc_top),    # 34 右外周（ゴールエリア半円上端: y=400）
            (L, y_post_top),   # 35 右ゴールポスト（上）
            (L, y_post_bottom),# 36 右ゴールポスト（下）
            (L, y_arc_bot),    # 37 右外周（ゴールエリア半円下端: y=1600）
            (L, W),            # 38 右下コーナー
        ]
    # ─────────────────────────────────────────────
    #  互換性のためのエイリアス（旧コード対応）
    # ─────────────────────────────────────────────
    @property
    def penalty_spot_distance(self) -> float:
        """
        旧コード互換: 第1PK(6m)の距離。
        annotators/futsal.py がこの名前を参照しているため残す。
        """
        return float(self.penalty_mark_distance)

    @property
    def goal_area_length(self) -> float:
        """
        旧コード互換: 'ゴールエリア半円の中心までの距離=3m' として扱っていた値。
        annotators/futsal.py の 2PK計算で使用されているため残す。
        """
        return 300.0  # 3m = 300cm

    # ─────────────────────────────────────────────
    #  エッジ（テンプレ線の近似）
    #  ※ホモグラフィ自体は対応点が命なので、エッジは主に描画用です
    # ─────────────────────────────────────────────
    edges: List[Tuple[int, int]] = field(default_factory=lambda: [
        # 外枠（左）
        (1, 2), (2, 3), (3, 4), (4, 5), (5, 6),
        # 外枠（右）
        (33, 34), (34, 35), (35, 36), (36, 37), (37, 38),

        # 上タッチライン（テンプレの並び）
        (1, 12), (12, 15), (15, 18), (18, 22), (22, 25), (25, 33),

        # 下タッチライン
        (6, 14), (14, 17), (17, 21), (21, 24), (24, 27), (27, 38),

        # センターライン
        (18, 19), (19, 20), (20, 21),

        # センターサークル（菱形＋横棒）
        (16, 19), (19, 23), (23, 20), (20, 16), (16, 23),

        # 左側の形（概形）
        (2, 7), (7, 9), (9, 10), (10, 11), (11, 8), (8, 5),

        # 右側の形（概形）
        (34, 31), (31, 28), (28, 29), (29, 30), (30, 32), (32, 37),
    ])

    labels: List[str] = field(default_factory=lambda: [
        f"{i:02}" for i in range(1, 39)  # 01-38
    ])

    colors: List[str] = field(default_factory=lambda: [
        # Roboflowテンプレの色分けに寄せる（左=オレンジ / 中央=ピンク / 右=水色）
        # 01-17: 左側（オレンジ系）
        *["#FF8C00"] * 17,
        # 18-23: 中央（ピンク系）
        *["#FF1493"] * 6,
        # 24-38: 右側（水色系）
        *["#00BFFF"] * 15,
    ])
