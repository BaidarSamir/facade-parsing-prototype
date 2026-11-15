"""
Window-to-Wall Ratio (WWR) Estimation – Façade Parsing Prototype
================================================================
Relevant for building energy performance analysis (Eurac Research).

FORMULA
-------
  WWR  =  window_pixels  /  (building_pixels + window_pixels + door_pixels)

  Class indices (CFP dataset):
    0 = background   1 = building   2 = window
    3 = sky           4 = roof       5 = door
    6 = tree          7 = people     8 = car
    9 = sign

EXCLUSION RULE
--------------
  An image is excluded from the pred-vs-GT comparison if
  EITHER the ground-truth gross-facade area OR the predicted gross-facade area
  is below MIN_GROSS_PX pixels (default 1 000 px on the inference-resolution
  image).  This avoids division by near-zero denominators in both masks.
  Excluded images are counted and reported; they are NOT counted in MAE / bias /
  correlation.

TWO SECTIONS
------------
  Section A — GT-only dataset statistics (all 602 images, all splits).
              Computed directly from the processed mask PNGs in data/cfp_processed/.
              Exclusion rule: gt_gross_px < MIN_GROSS_PX (GT mask only).
              Reported per split (train / val / test) and overall.

  Section B — Pred vs GT comparison (61 test images only).
              Requires results/wwr_predictions.json produced by the Kaggle
              inference snippet.  Each record supplies gt_wwr, pred_wwr,
              gt_gross_px, pred_gross_px.  A pair is excluded if either
              gross_px field is None or < MIN_GROSS_PX.

USAGE
-----
  python src/compute_wwr.py
"""

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from scipy import stats

# ── Paths ────────────────────────────────────────────────────────────────────
SRC_DIR     = Path(__file__).parent
ROOT        = SRC_DIR.parent
DATA_PROC   = ROOT / "data" / "cfp_processed"
RESULTS_DIR = ROOT / "results"
FIGURES_DIR = RESULTS_DIR / "figures"
WWR_PRED_JSON = RESULTS_DIR / "wwr_predictions.json"
OUT_JSON      = RESULTS_DIR / "wwr_metrics.json"
OUT_FIG       = FIGURES_DIR / "wwr_analysis.png"

FIGURES_DIR.mkdir(parents=True, exist_ok=True)

MIN_GROSS_PX = 1_000   # minimum gross-facade pixels to include an image


# ── Helpers ──────────────────────────────────────────────────────────────────

def wwr_from_mask(mask: np.ndarray, min_px: int = MIN_GROSS_PX):
    """Return (wwr, gross_px) or (None, gross_px) if below threshold."""
    win   = int((mask == 2).sum())
    bld   = int((mask == 1).sum())
    door  = int((mask == 5).sum())
    gross = win + bld + door
    wwr   = float(win / gross) if gross >= min_px else None
    return wwr, gross


def split_stats(wwr_list: list) -> dict:
    arr = np.array(wwr_list)
    return {
        "n":      len(arr),
        "mean":   round(float(np.mean(arr)),              4),
        "std":    round(float(np.std(arr)),               4),
        "median": round(float(np.median(arr)),            4),
        "p25":    round(float(np.percentile(arr, 25)),    4),
        "p75":    round(float(np.percentile(arr, 75)),    4),
        "min":    round(float(np.min(arr)),               4),
        "max":    round(float(np.max(arr)),               4),
    }


# ══════════════════════════════════════════════════════════════════════════════
# SECTION A — GT-only statistics (all splits)
# ══════════════════════════════════════════════════════════════════════════════

def section_a():
    print("\n" + "="*60)
    print("SECTION A — GT-only WWR Statistics (Ground Truth Masks)")
    print("="*60)

    splits = ["train", "val", "test"]
    all_gt = []
    split_data   = {}   # split → list of wwr values
    split_excl   = {}   # split → excluded count

    for split in splits:
        msk_dir = DATA_PROC / split / "masks"
        if not msk_dir.exists():
            print(f"  [WARN] {msk_dir} not found — skipping {split}")
            continue

        paths = sorted(msk_dir.glob("*.png"))
        wwrs, excluded = [], 0
        for p in paths:
            m = np.array(Image.open(p))
            val, gross = wwr_from_mask(m)
            if val is None:
                excluded += 1
            else:
                wwrs.append(val)
                all_gt.append(val)

        split_data[split] = wwrs
        split_excl[split] = excluded
        st = split_stats(wwrs) if wwrs else {}
        print(f"  {split.upper():<5}: {len(wwrs)}/{len(paths)} valid  "
              f"(excluded: {excluded})  |  "
              f"Mean {st.get('mean',0)*100:.2f}% ± {st.get('std',0)*100:.2f}%  "
              f"Median {st.get('median',0)*100:.2f}%")

    overall = split_stats(all_gt) if all_gt else {}
    print(f"\n  OVERALL: {overall.get('n',0)} facades  "
          f"Mean {overall.get('mean',0)*100:.2f}%  "
          f"Std {overall.get('std',0)*100:.2f}%  "
          f"Median {overall.get('median',0)*100:.2f}%  "
          f"P25–P75 [{overall.get('p25',0)*100:.2f}%, {overall.get('p75',0)*100:.2f}%]")

    return {
        "description": "Ground-truth WWR statistics — do NOT interpret as model accuracy",
        "min_gross_px_threshold": MIN_GROSS_PX,
        "overall":  overall,
        "by_split": {s: {**split_stats(split_data[s]),
                         "excluded": split_excl[s]}
                     for s in split_data if split_data[s]},
        "_gt_by_split_raw": split_data,   # kept for plotting, not written to JSON
    }


# ══════════════════════════════════════════════════════════════════════════════
# SECTION B — Pred vs GT comparison (test set, 61 images)
# ══════════════════════════════════════════════════════════════════════════════

def section_b():
    print("\n" + "="*60)
    print("SECTION B — B2 Predicted WWR vs Ground-Truth WWR (Test Set)")
    print("="*60)

    if not WWR_PRED_JSON.exists():
        print(f"\n  [ERROR] {WWR_PRED_JSON} not found.")
        print("  → Run the Kaggle inference snippet first, download the JSON,")
        print(f"    and place it at: {WWR_PRED_JSON}")
        return None

    with open(WWR_PRED_JSON) as f:
        records = json.load(f)

    total = len(records)
    pairs_gt, pairs_pred, excluded = [], [], []

    for r in records:
        gt_px   = r.get("gt_gross_px",   0) or 0
        pred_px = r.get("pred_gross_px", 0) or 0
        gt_w    = r.get("gt_wwr")
        pred_w  = r.get("pred_wwr")

        # Exclude if EITHER side has insufficient facade area or missing value
        if gt_w is None or pred_w is None or gt_px < MIN_GROSS_PX or pred_px < MIN_GROSS_PX:
            excluded.append(r["stem"])
        else:
            pairs_gt.append(gt_w)
            pairs_pred.append(pred_w)

    n_included = len(pairs_gt)
    n_excluded = len(excluded)

    if n_included < 2:
        print(f"  [ERROR] Only {n_included} valid pairs — cannot compute stats.")
        return None

    gt_arr   = np.array(pairs_gt)
    pred_arr = np.array(pairs_pred)

    mae   = float(np.mean(np.abs(pred_arr - gt_arr)))
    bias  = float(np.mean(pred_arr - gt_arr))          # positive = over-predicts
    r_val, p_val = stats.pearsonr(gt_arr, pred_arr)
    ss_res = float(np.sum((gt_arr - pred_arr)**2))
    ss_tot = float(np.sum((gt_arr - gt_arr.mean())**2))
    r2     = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")

    print(f"\n  Included pairs : {n_included} / {total}")
    print(f"  Excluded pairs : {n_excluded}"
          + (f"  ({', '.join(excluded[:5])}{'…' if n_excluded > 5 else ''})"
             if excluded else ""))
    print(f"\n  MAE   (mean |pred−GT|)   : {mae*100:.2f}  pp")
    print(f"  Bias  (mean  pred−GT)    : {bias*100:+.2f} pp  "
          f"({'over-predicts' if bias > 0 else 'under-predicts'} WWR on average)")
    print(f"  Pearson r                : {r_val:.3f}  (p = {p_val:.3g})")
    print(f"  R²                       : {r2:.3f}")

    return {
        "description": "B2 predicted WWR vs ground-truth WWR — test set only (61 images)",
        "formula": "WWR = window_px / (building_px + window_px + door_px)",
        "exclusion_rule": f"Pair excluded if gt_gross_px < {MIN_GROSS_PX} OR pred_gross_px < {MIN_GROSS_PX}",
        "min_gross_px_threshold": MIN_GROSS_PX,
        "total_test_images": total,
        "included_pairs":    n_included,
        "excluded_pairs":    n_excluded,
        "excluded_stems":    excluded,
        "mae_pp":   round(mae  * 100, 3),
        "bias_pp":  round(bias * 100, 3),
        "pearson_r": round(float(r_val), 4),
        "pearson_p": round(float(p_val), 6),
        "r2":        round(float(r2),    4),
        "gt_mean_pp":   round(float(gt_arr.mean()   * 100), 3),
        "gt_std_pp":    round(float(gt_arr.std()    * 100), 3),
        "pred_mean_pp": round(float(pred_arr.mean() * 100), 3),
        "pred_std_pp":  round(float(pred_arr.std()  * 100), 3),
        "_gt_arr":   pairs_gt,    # kept for plotting
        "_pred_arr": pairs_pred,
    }


# ══════════════════════════════════════════════════════════════════════════════
# FIGURES
# ══════════════════════════════════════════════════════════════════════════════

def make_figures(sec_a: dict, sec_b):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # ── Left: GT WWR histogram (all splits, labeled "Ground Truth Only") ──────
    ax = axes[0]
    colors = {"train": "#abd9e9", "val": "#fdae61", "test": "#f46d43"}
    raw = sec_a.get("_gt_by_split_raw", {})
    for split, wwrs in raw.items():
        if wwrs:
            ax.hist(np.array(wwrs) * 100,
                    bins=np.linspace(0, 70, 29),
                    alpha=0.65, color=colors.get(split, "gray"),
                    edgecolor="white", linewidth=0.5, label=split.capitalize())

    overall = sec_a.get("overall", {})
    ax.axvline(overall.get("mean",   0) * 100, color="red",    ls="--", lw=1.5,
               label=f"Mean {overall.get('mean',0)*100:.1f}%")
    ax.axvline(overall.get("median", 0) * 100, color="black",  ls=":",  lw=1.5,
               label=f"Median {overall.get('median',0)*100:.1f}%")
    ax.set_xlabel("Window-to-Wall Ratio (%)", fontsize=11)
    ax.set_ylabel("Number of Façades",        fontsize=11)
    ax.set_title("WWR Distribution — Ground Truth Only\n"
                 f"(N = {overall.get('n',0)} façades, all splits)", fontsize=11)
    ax.legend(fontsize=9)
    ax.grid(True, ls="--", alpha=0.4)

    # ── Right: Pred vs GT scatter (B2, test set) ──────────────────────────────
    ax2 = axes[1]
    if sec_b is not None:
        gt_arr   = np.array(sec_b["_gt_arr"])   * 100
        pred_arr = np.array(sec_b["_pred_arr"]) * 100
        ax2.scatter(gt_arr, pred_arr, s=40, alpha=0.75,
                    color="#2b5c8f", edgecolors="white", linewidths=0.4)
        lims = [0, max(gt_arr.max(), pred_arr.max()) * 1.08]
        ax2.plot(lims, lims, "k--", lw=1.2, label="Identity (perfect)")
        ax2.set_xlim(lims); ax2.set_ylim(lims)
        ax2.set_xlabel("Ground-Truth WWR (%)",  fontsize=11)
        ax2.set_ylabel("Predicted WWR (%)",     fontsize=11)
        ax2.set_title("B2 Predicted vs Ground-Truth WWR\n"
                      f"(test set, N = {sec_b['included_pairs']} pairs)", fontsize=11)
        txt = (f"MAE  = {sec_b['mae_pp']:.2f} pp\n"
               f"Bias = {sec_b['bias_pp']:+.2f} pp\n"
               f"R²   = {sec_b['r2']:.3f}\n"
               f"r    = {sec_b['pearson_r']:.3f}")
        ax2.text(0.05, 0.97, txt, transform=ax2.transAxes,
                 va="top", ha="left", fontsize=9.5,
                 bbox=dict(boxstyle="round,pad=0.4", fc="white", ec="gray", alpha=0.9))
        ax2.legend(fontsize=9)
        ax2.grid(True, ls="--", alpha=0.4)
    else:
        ax2.text(0.5, 0.5,
                 "No wwr_predictions.json\n(run Kaggle snippet first)",
                 ha="center", va="center", transform=ax2.transAxes,
                 fontsize=11, color="gray")
        ax2.set_title("B2 Predicted vs GT WWR — data missing", fontsize=11)

    plt.tight_layout()
    plt.savefig(OUT_FIG, dpi=150)
    plt.close()
    print(f"\n✓ Figure saved → {OUT_FIG}")


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    print("Window-to-Wall Ratio Analysis — Façade Parsing Prototype")
    print(f"MIN_GROSS_PX threshold : {MIN_GROSS_PX} px")

    sec_a = section_a()
    sec_b = section_b()

    # ── Write JSON ────────────────────────────────────────────────────────────
    # Strip private keys before serializing
    out_a = {k: v for k, v in sec_a.items() if not k.startswith("_")}
    out_b = ({k: v for k, v in sec_b.items() if not k.startswith("_")}
             if sec_b else None)

    with open(OUT_JSON, "w") as f:
        json.dump({"section_a_gt_statistics": out_a,
                   "section_b_pred_vs_gt":    out_b}, f, indent=2)
    print(f"\n✓ Metrics saved → {OUT_JSON}")

    make_figures(sec_a, sec_b)

    if sec_b is None:
        print("\n[ACTION REQUIRED] Generate wwr_predictions.json from Kaggle, then re-run.")
        sys.exit(0)


if __name__ == "__main__":
    main()
