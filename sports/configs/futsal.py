from dataclasses import dataclass, field
from typing import List, Tuple

@dataclass
class FutsalPitchConfiguration:
    length: float = 4000
    width: float = 2000
    goal_area_length: float = 300          # 半円中心まで 3 m
    penalty_spot_distance: float = 600     # 2 PK (=6 m)
    centre_circle_radius: float = 300

    @property
    def vertices(self) -> List[Tuple[float, float]]:
        return [
            (0, 0),                                  # 01 左上コーナー
            (0, self.width * 0.2),                   # 02
            (0, self.width * 0.35),                  # 03
            (0, self.width * 0.65),                  # 04
            (0, self.width * 0.8),                   # 05
            (0, self.width),                         # 06 左下コーナー
            (self.goal_area_length, self.width * 0.35),  # 07 左半円端①
            (self.goal_area_length, self.width * 0.65),  # 08 左半円端②
            (self.length * 0.25, self.width * 0.4),      # 09 左アーク
            (self.length / 2, 0),                        # 10 中央上
            (self.length / 2, self.width * 0.45),        # 11 アーク上
            (self.length / 2, self.width * 0.55),        # 12 アーク下
            (self.length / 2, self.width),               # 13 中央下
            (self.length * 0.75, self.width * 0.4),      # 14 右アーク
            (self.length - self.goal_area_length, self.width * 0.35),  # 15 右半円端①
            (self.length - self.goal_area_length, self.width * 0.65),  # 16 右半円端②
            (self.length, 0),                            # 17 右上
            (self.length, self.width * 0.2),             # 18
            (self.length, self.width * 0.35),            # 19
            (self.length, self.width * 0.65),            # 20
            (self.length, self.width * 0.8),             # 21
            (self.length, self.width),                   # 22 右下
            (self.penalty_spot_distance, self.width / 2),               # 23 左2PK (10 m)
            (self.length - self.penalty_spot_distance, self.width / 2), # 24 右2PK (10 m)
            (self.goal_area_length, self.width / 2),                    # 25 左1PK (6 m)
            (self.length - self.goal_area_length, self.width / 2)       # 26 右1PK (6 m)
        ]

    # ── 外枠・センターライン・アーク端同士など、既存 24 点用の線はそのまま ──
    #   追加した 25,26 は孤立点 (PK スポット) なので edge には含めない
    edges: List[Tuple[int, int]] = field(default_factory=lambda: [
        (1, 2), (2, 3), (3, 4), (4, 5), (5, 6), (7, 8),
        (10, 11), (11, 12), (12, 13), (9, 11), (9, 12),
        (11, 14), (12, 14), (15, 16), (17, 18), (18, 19),
        (19, 20), (20, 21), (21, 22),
        (1, 10), (2, 7), (3, 7), (4, 8), (5, 8), (6, 13),
        (10, 17), (15, 19), (16, 20), (13, 22)
    ])

    labels: List[str] = field(default_factory=lambda: [
        f"{i:02}" for i in range(1, 27)   # ← 26 個へ拡張
    ])

    colors: List[str] = field(default_factory=lambda: [
        # 左側: 01–08 → オレンジ
        "#FF8C00", "#FF8C00", "#FF8C00", "#FF8C00",
        "#FF8C00", "#FF8C00", "#FF8C00", "#FF8C00",
        # 中央: 09–14 → ピンク
        "#FF1493", "#FF1493", "#FF1493", "#FF1493",
        "#FF1493", "#FF1493",
        # 右側: 15–22 → 青
        "#00BFFF", "#00BFFF", "#00BFFF", "#00BFFF",
        "#00BFFF", "#00BFFF", "#00BFFF", "#00BFFF",
        # 2PK (23,24) → 赤
        "#FF0000", "#FF0000",
        # 1PK (25,26) → 赤
        "#FF0000", "#FF0000"
    ])
