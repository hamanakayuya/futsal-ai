# /home/hamanaka/futsal-ai/examples/futsal/plot_error_hist.py
import argparse
from pathlib import Path
import re
import math

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# -----------------------------
# utils
# -----------------------------
def ensure_outdir(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    return p


def find_col(cols, patterns):
    """Return the first column name matching any regex pattern."""
    for pat in patterns:
        rx = re.compile(pat, re.IGNORECASE)
        for c in cols:
            if rx.search(c):
                return c
    return None


def clean_series_to_cm(s: pd.Series) -> np.ndarray:
    """Convert a pandas Series to float ndarray and drop NaNs."""
    return pd.to_numeric(s, errors="coerce").dropna().astype(float).values


def summarize_cm(arr: np.ndarray) -> dict:
    """
    Return basic stats + threshold rates (<=10cm, <=20cm) in percent.
    All plots share the same thresholds by design.
    """
    if arr is None or len(arr) == 0:
        return dict(
            n=0, mean=np.nan, median=np.nan, p95=np.nan, p99=np.nan,
            le10=np.nan, le20=np.nan, vmin=np.nan, vmax=np.nan
        )
    return dict(
        n=int(len(arr)),
        mean=float(np.mean(arr)),
        median=float(np.median(arr)),
        p95=float(np.percentile(arr, 95)),
        p99=float(np.percentile(arr, 99)),
        le10=float(np.mean(arr <= 10.0) * 100.0),
        le20=float(np.mean(arr <= 20.0) * 100.0),
        vmin=float(np.min(arr)),
        vmax=float(np.max(arr)),
    )


def clip_by_max(arr: np.ndarray, max_cm: float | None) -> np.ndarray:
    if arr is None:
        return arr
    if max_cm is None:
        return arr
    return arr[arr <= float(max_cm)]


def nice_ceil(x: float, step: float) -> float:
    return math.ceil(x / step) * step


def choose_xlim_and_binwidth(
    arr: np.ndarray,
    user_max_cm: float | None,
    user_bin_width: float | None,
):
    """
    Pick a readable x-axis upper bound and bin width per-plot.
    - If user_max_cm given -> fixed.
    - Else use p99*1.2 as base, then round up to nice ticks.
    - If errors are small, use small xmax and narrow bins (1cm).
    """
    if arr is None or len(arr) == 0:
        return 100.0, 5.0

    st = summarize_cm(arr)
    p99 = st["p99"]

    # xmax
    if user_max_cm is not None:
        xmax = float(user_max_cm)
    else:
        xmax = float(max(10.0, p99 * 1.2))

        if xmax <= 30:
            xmax = nice_ceil(max(xmax, 20.0), 5)   # 20/25/30
        elif xmax <= 80:
            xmax = nice_ceil(xmax, 10)             # 40/50/60/70/80
        else:
            xmax = nice_ceil(xmax, 25)             # 100/125/150/...

    # bin width
    if user_bin_width is not None:
        bw = float(user_bin_width)
    else:
        if xmax <= 30:
            bw = 1.0
        elif xmax <= 80:
            bw = 2.0
        else:
            bw = 5.0

    # Degenerate range protection
    if np.isfinite(st["vmin"]) and np.isfinite(st["vmax"]) and (st["vmax"] - st["vmin"] < 1e-6):
        xmax = max(xmax, st["vmax"] + bw * 2)

    return xmax, bw


def build_nbins(xmax: float, bin_width: float, max_bins: int = 120) -> int:
    nb = int(max(5, round(xmax / bin_width)))
    return min(nb, max_bins)


def stats_text(st: dict) -> str:
    """
    NOTE: thresholds are fixed to 10/20cm for ALL plots.
    """
    return (
        f"n={st['n']}\n"
        f"mean={st['mean']:.1f} cm\n"
        f"med={st['median']:.1f} cm\n"
        f"p95={st['p95']:.1f} cm\n"
        f"<=10cm={st['le10']:.1f}%\n"
        f"<=20cm={st['le20']:.1f}%"
    )


def add_stat_lines(ax, st: dict):
    """Draw vertical lines for mean/median/p95 (no color specified)."""
    for x, label in [(st["mean"], "mean"), (st["median"], "median"), (st["p95"], "p95")]:
        if np.isfinite(x):
            ax.axvline(x, linestyle="--", linewidth=1)
            ax.text(
                x, 0.98, label,
                transform=ax.get_xaxis_transform(),
                rotation=90, va="top", ha="right"
            )


# -----------------------------
# plotting
# -----------------------------
def plot_hist(
    arr: np.ndarray,
    title: str,
    out_path: Path,
    user_max_cm: float | None,
    user_bin_width: float | None,
):
    if arr is None or len(arr) == 0:
        print(f"[SKIP] no data: {title}")
        return

    st = summarize_cm(arr)
    xmax, bw = choose_xlim_and_binwidth(arr, user_max_cm, user_bin_width)
    nbins = build_nbins(xmax, bw)

    fig = plt.figure(figsize=(7.2, 4.6))
    ax = fig.add_subplot(111)

    # IMPORTANT: specify range to avoid "all mass in tiny region" looking empty
    ax.hist(arr, bins=nbins, range=(0, xmax))

    ax.set_title(title)
    ax.set_xlabel("Error [cm] (Euclidean distance between GT and prediction)")
    ax.set_ylabel("Count (number of matched samples)")
    ax.grid(True, alpha=0.25)
    ax.set_xlim(0, xmax)

    add_stat_lines(ax, st)
    ax.text(0.98, 0.98, stats_text(st), transform=ax.transAxes, ha="right", va="top")

    fig.tight_layout()
    fig.savefig(out_path, dpi=220)
    plt.close(fig)
    print(f"[OK] saved: {out_path}")


def plot_overlay(
    a: np.ndarray,
    b: np.ndarray,
    title: str,
    out_path: Path,
    user_max_cm: float | None,
    user_bin_width: float | None,
):
    if (a is None or len(a) == 0) and (b is None or len(b) == 0):
        print(f"[SKIP] no data: {title}")
        return

    comb = []
    if a is not None and len(a) > 0:
        comb.append(a)
    if b is not None and len(b) > 0:
        comb.append(b)
    comb = np.concatenate(comb) if len(comb) else np.array([1.0])

    xmax, bw = choose_xlim_and_binwidth(comb, user_max_cm, user_bin_width)
    nbins = build_nbins(xmax, bw)

    fig = plt.figure(figsize=(7.2, 4.6))
    ax = fig.add_subplot(111)

    # density=True makes shape comparison easier (counts can differ).
    if a is not None and len(a) > 0:
        ax.hist(a, bins=nbins, range=(0, xmax), alpha=0.55, label="End-to-End (A)", density=True)
    if b is not None and len(b) > 0:
        ax.hist(b, bins=nbins, range=(0, xmax), alpha=0.55, label="Object-only (B)", density=True)

    ax.set_title(title)
    ax.set_xlabel("Error [cm] (Euclidean distance between GT and prediction)")
    ax.set_ylabel("Density (distribution shape)")
    ax.grid(True, alpha=0.25)
    ax.set_xlim(0, xmax)
    ax.legend()

    fig.tight_layout()
    fig.savefig(out_path, dpi=220)
    plt.close(fig)
    print(f"[OK] saved: {out_path}")


# -----------------------------
# main: load CSV (wide/long), plot, export stats
# -----------------------------
def main():
    ap = argparse.ArgumentParser(description="Plot error histograms from evaluate_labelme CSV (wide/long supported)")
    ap.add_argument("--csv", type=str, required=True)
    ap.add_argument("--out_dir", type=str, required=True)
    ap.add_argument("--max_cm", type=float, default=None, help="Fix x-axis max for ALL plots (optional)")
    ap.add_argument("--bin_width", type=float, default=None, help="Fix bin width [cm] for ALL plots (optional)")
    ap.add_argument("--title_prefix", type=str, default="")
    args = ap.parse_args()

    csv_path = Path(args.csv)
    out_dir = ensure_outdir(Path(args.out_dir))
    df = pd.read_csv(csv_path)

    prefix = (args.title_prefix + " ") if args.title_prefix else ""
    cols = list(df.columns)

    # --- detect WIDE format (evaluate_labelme CSV) ---
    is_wide = (
        ("matched_A" in df.columns) or ("matched_B" in df.columns) or
        ("ball_err_A_cm" in df.columns) or ("ball_err_B_cm" in df.columns)
    )

    player_A = player_B = ball_A = ball_B = None

    if is_wide:
        print("[INFO] Detected WIDE CSV format.")
        if "matched_A" in df.columns:
            player_A = clip_by_max(clean_series_to_cm(df["matched_A"]), args.max_cm)
        if "matched_B" in df.columns:
            player_B = clip_by_max(clean_series_to_cm(df["matched_B"]), args.max_cm)
        if "ball_err_A_cm" in df.columns:
            ball_A = clip_by_max(clean_series_to_cm(df["ball_err_A_cm"]), args.max_cm)
        if "ball_err_B_cm" in df.columns:
            ball_B = clip_by_max(clean_series_to_cm(df["ball_err_B_cm"]), args.max_cm)

    else:
        # --- LONG format fallback ---
        print("[INFO] Detected LONG CSV format.")
        obj_col = find_col(cols, [r"^(obj|object|type|kind)$", r"(obj|object|type|kind)", r"(class|label)"])
        mode_col = find_col(cols, [r"^(mode)$", r"(mode)"])
        err_col  = find_col(cols, [r"^(err_cm|error_cm)$", r"(err|error).*cm"])

        if obj_col is None or mode_col is None or err_col is None:
            raise RuntimeError(
                "Cannot detect CSV format.\n"
                "Need either WIDE columns: matched_A/matched_B/ball_err_A_cm/ball_err_B_cm\n"
                "or LONG columns: object + mode + error_cm.\n"
                f"columns={cols}"
            )

        sub_player = df[df[obj_col].astype(str).str.lower() == "player"].copy()
        sub_ball   = df[df[obj_col].astype(str).str.lower() == "ball"].copy()

        def pick(sub, which):
            m = sub[mode_col].astype(str).str.lower()
            if which == "A":
                sub2 = sub[m.str.contains("a") | m.str.contains("end")]
            else:
                sub2 = sub[m.str.contains("b") | m.str.contains("obj") | m.str.contains("image")]
            arr = clean_series_to_cm(sub2[err_col])
            return clip_by_max(arr, args.max_cm)

        player_A = pick(sub_player, "A")
        player_B = pick(sub_player, "B")
        ball_A   = pick(sub_ball, "A")
        ball_B   = pick(sub_ball, "B")

    # --- plots ---
    plot_hist(player_A, f"{prefix}Player Error (End-to-End / A)", out_dir / "hist_player_A_end_to_end.png", args.max_cm, args.bin_width)
    plot_hist(player_B, f"{prefix}Player Error (Object-only / B)", out_dir / "hist_player_B_object_only.png", args.max_cm, args.bin_width)
    plot_hist(ball_A,   f"{prefix}Ball Error (End-to-End / A)",   out_dir / "hist_ball_A_end_to_end.png", args.max_cm, args.bin_width)
    plot_hist(ball_B,   f"{prefix}Ball Error (Object-only / B)",  out_dir / "hist_ball_B_object_only.png", args.max_cm, args.bin_width)

    plot_overlay(player_A, player_B, f"{prefix}Player Error Overlay (A vs B)", out_dir / "overlay_player_A_vs_B.png", args.max_cm, args.bin_width)
    plot_overlay(ball_A,   ball_B,   f"{prefix}Ball Error Overlay (A vs B)",   out_dir / "overlay_ball_A_vs_B.png", args.max_cm, args.bin_width)

    # --- stats summary CSV (for paper) ---
    rows = []
    for obj, mode, arr in [
        ("player", "A_end_to_end", player_A),
        ("player", "B_object_only", player_B),
        ("ball",   "A_end_to_end", ball_A),
        ("ball",   "B_object_only", ball_B),
    ]:
        st = summarize_cm(arr if arr is not None else np.array([]))
        rows.append({
            "object": obj,
            "mode": mode,
            "n": st["n"],
            "mean_cm": st["mean"],
            "median_cm": st["median"],
            "p95_cm": st["p95"],
            "p99_cm": st["p99"],
            "le_10cm_pct": st["le10"],
            "le_20cm_pct": st["le20"],
            "max_cm_fixed": args.max_cm if args.max_cm is not None else "",
            "bin_width_fixed_cm": args.bin_width if args.bin_width is not None else "",
        })

    stats_df = pd.DataFrame(rows)
    stats_path = out_dir / "stats_summary.csv"
    stats_df.to_csv(stats_path, index=False)
    print(f"[OK] saved: {stats_path}")


if __name__ == "__main__":
    main()
