import os
import time
import cv2
import numpy as np
import pandas as pd
from scipy import stats
from pipeline_utils import (
    strip_chrome, compute_iqa_metrics,
    apply_condition_a_raw, apply_condition_b_static, apply_condition_c_adaptive,
)

# ==========================================
# CONFIG — edit these
# ==========================================
INPUT_ROOT = "/home/aaush/cardium_native/cleaned_cardium"
OUTPUT_BASE = "/home/aaush/cardium_native"
LOG_PATH = os.path.join(OUTPUT_BASE, "pipeline_log.xlsx")
CHECKPOINT_EVERY = 200
POSITIVE_CLASS = "chd"  # must match one of your class folder names

CONDITIONS = {
    "A_raw": os.path.join(OUTPUT_BASE, "processed_A_raw"),
    "B_static": os.path.join(OUTPUT_BASE, "processed_B_static"),
    "C_adaptive": os.path.join(OUTPUT_BASE, "processed_C_adaptive"),
}
for d in CONDITIONS.values():
    os.makedirs(d, exist_ok=True)

# ==========================================
# GATHER IMAGES
# ==========================================
image_records = []
for class_name in sorted(os.listdir(INPUT_ROOT)):
    class_dir = os.path.join(INPUT_ROOT, class_name)
    if not os.path.isdir(class_dir):
        continue
    for filename in os.listdir(class_dir):
        if filename.lower().endswith((".png", ".jpg", ".jpeg")):
            image_records.append((class_name, filename))

total_images = len(image_records)
classes_found = sorted(set(c for c, _ in image_records))
print(f"Found {total_images} images across classes: {classes_found}")
if len(classes_found) != 2:
    print(f"WARNING: expected 2 classes, found {len(classes_found)}. The binary "
          f"metrics in scripts 2/3 assume exactly 2 classes ('{POSITIVE_CLASS}' vs one other).")

# ==========================================
# PROCESS ALL THREE CONDITIONS + LOG
# ==========================================
log_rows = []
start_time = time.time()

for count, (class_name, filename) in enumerate(image_records, 1):
    src_path = os.path.join(INPUT_ROOT, class_name, filename)
    img = cv2.imread(src_path)
    if img is None:
        print(f"[skip] Could not read {src_path}")
        continue

    raw_metrics = compute_iqa_metrics(strip_chrome(img))  # post-chrome-strip, pre-enhancement

    try:
        out_a, _ = apply_condition_a_raw(img)
        out_b, params_b = apply_condition_b_static(img)
        out_c, params_c, _ = apply_condition_c_adaptive(img)
    except Exception as e:
        print(f"[error] {filename}: {e}")
        continue

    rel_paths = {}
    for cond_name, out_img in [("A_raw", out_a), ("B_static", out_b), ("C_adaptive", out_c)]:
        dst_dir = os.path.join(CONDITIONS[cond_name], class_name)
        os.makedirs(dst_dir, exist_ok=True)
        cv2.imwrite(os.path.join(dst_dir, filename), out_img)
        # relative path only -- no local machine paths in the shared log
        rel_paths[cond_name] = os.path.join(f"processed_{cond_name}", class_name, filename)

    log_rows.append({
        "filename": filename,
        "class": class_name,
        "path_condition_A": rel_paths["A_raw"],
        "path_condition_B": rel_paths["B_static"],
        "path_condition_C": rel_paths["C_adaptive"],
        "raw_brightness": raw_metrics["brightness"],
        "raw_contrast_std": raw_metrics["contrast_std"],
        "raw_sharpness": raw_metrics["sharpness_laplacian_var"],
        "raw_entropy": raw_metrics["entropy"],
        "adaptive_gamma": params_c["gamma"],
        "adaptive_clahe_clip": params_c["clahe_clip_limit"],
        "adaptive_denoise_applied": params_c["denoise_applied"],
        "adaptive_sharpen_applied": params_c["sharpen_applied"],
        "gamma_applied": params_c["gamma"] is not None,
        "clahe_applied": params_c["clahe_clip_limit"] is not None,
    })

    if count % 200 == 0 or count == total_images:
        print(f"[{count}/{total_images}] {class_name}/{filename}")
    if count % CHECKPOINT_EVERY == 0:
        pd.DataFrame(log_rows).to_csv(os.path.join(OUTPUT_BASE, "_checkpoint.csv"), index=False)

log_df = pd.DataFrame(log_rows)
log_df.to_excel(LOG_PATH, index=False)
elapsed = (time.time() - start_time) / 60
print(f"\nDone. Processed {len(log_rows)}/{total_images} images in {elapsed:.2f} min.")
print(f"Log written to {LOG_PATH}")

# ==========================================
# THRESHOLD CALIBRATION HINT
# ==========================================
print("\nRaw metric percentiles (use these to sanity-check IQA_THRESHOLDS in "
      "pipeline_utils.py -- they were not tuned on your actual data):")
for metric in ["raw_brightness", "raw_contrast_std", "raw_sharpness", "raw_entropy"]:
    p10, p50, p90 = np.percentile(log_df[metric], [10, 50, 90])
    print(f"  {metric:18s} p10={p10:8.2f}  p50={p50:8.2f}  p90={p90:8.2f}")

# ==========================================
# CONFOUND DIAGNOSTIC
# ==========================================
print("\n" + "=" * 64)
print("ACQUISITION CONFOUND DIAGNOSTIC (raw, pre-enhancement metrics)")
print("=" * 64)
if len(classes_found) == 2:
    a, b = classes_found
    flagged = []
    for metric in ["raw_brightness", "raw_contrast_std", "raw_entropy", "raw_sharpness"]:
        ga = log_df.loc[log_df["class"] == a, metric]
        gb = log_df.loc[log_df["class"] == b, metric]
        t, p = stats.ttest_ind(ga, gb, equal_var=False)
        flag = "  <-- SIGNIFICANT (p<0.05)" if p < 0.05 else ""
        if p < 0.05:
            flagged.append(metric)
        print(f"{metric:18s} {a}: mean={ga.mean():7.2f}   {b}: mean={gb.mean():7.2f}   p={p:.4g}{flag}")

    print("\nAdaptive decision rates by class (should look similar unless the "
          "pathology itself genuinely correlates with raw image quality):")
    for flag_col in ["gamma_applied", "clahe_applied", "adaptive_denoise_applied", "adaptive_sharpen_applied"]:
        ct = pd.crosstab(log_df["class"], log_df[flag_col])
        print(f"\n{flag_col}:")
        print(ct.to_string())
        if ct.shape == (2, 2) and ct.values.min() > 0:
            chi2, p, _, _ = stats.chi2_contingency(ct)
            if p < 0.05:
                flagged.append(flag_col)
                print(f"  <-- SIGNIFICANT (p={p:.4g})")

    if flagged:
        print(f"\nWARNING: {', '.join(flagged)} differ significantly between classes. "
              "That's consistent with an acquisition confound (different machines, "
              "presets, or sources per class) rather than the pathology itself -- a "
              "classifier can shortcut on this no matter how good scripts 2/3 are. "
              "Investigate (compare a handful of raw chd vs non_chd images by eye, "
              "check acquisition metadata) before trusting downstream benchmark numbers.")
    else:
        print("\nNo significant difference detected on these metrics. Reassuring, but "
              "not conclusive -- it can't see confounds outside what these four numbers "
              "measure (e.g. a consistent watermark, cropping style, or metadata field).")
else:
    print("Diagnostic assumes exactly 2 classes; skipped.")
print("=" * 64)
