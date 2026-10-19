import os
import json
import cv2
import numpy as np
import pandas as pd
import torch
import streamlit as st
from PIL import Image
from torchvision import transforms

from pipeline_utils import (
    ARCH_LIST,
    IMG_SIZE,
    IMAGENET_MEAN,
    IMAGENET_STD,
    IQA_THRESHOLDS,
    get_model,
    get_target_layer,
    apply_condition_a_raw,
    apply_condition_b_static,
    apply_condition_c_adaptive,
    GradCAM,
    describe_focus_region,
    overlay_heatmap,
)

# ==========================================
# PATHS & CONFIG
# ==========================================
REPORT_DIR = "/home/aaush/cardium_native"
RAW_DIR = os.path.join(REPORT_DIR, "cleaned_cardium")
MODEL_DIR = os.path.join(REPORT_DIR, "models")
MASTER_REPORT_PATH = os.path.join(REPORT_DIR, "ieee_master_report.xlsx")
PIPELINE_LOG_PATH = os.path.join(REPORT_DIR, "pipeline_log.xlsx")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

st.set_page_config(
    page_title="CARDIUM-AI | CHD Diagnostic & Ablation Suite",
    layout="wide",
)

val_tf = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])


# ==========================================
# MODEL & METADATA LOADER
# ==========================================
def load_metadata():
    class_names_path = os.path.join(MODEL_DIR, "class_names.json")
    best_arch_path = os.path.join(MODEL_DIR, "best_arch.json")

    class_names = ["chd", "non_chd"]
    best_arch = "efficientnet_b0"
    positive_class = "chd"

    if os.path.exists(class_names_path):
        with open(class_names_path) as f:
            class_names = json.load(f)
    if os.path.exists(best_arch_path):
        with open(best_arch_path) as f:
            info = json.load(f)
            best_arch = info.get("architecture", best_arch)
            positive_class = info.get("positive_class", positive_class)

    return class_names, best_arch, positive_class


@st.cache_resource
def load_checkpoint_model(arch: str, num_classes: int, weights_path: str):
    model = get_model(arch, num_classes, pretrained=False)
    loaded = False
    if os.path.exists(weights_path):
        state = torch.load(weights_path, map_location=DEVICE)
        model.load_state_dict(state)
        loaded = True
    model.to(DEVICE)
    model.eval()
    return model, loaded


class_names, default_arch, positive_class = load_metadata()

# ==========================================
# SIDEBAR
# ==========================================
with st.sidebar:
    st.title("CARDIUM-AI Control")
    st.caption(f"Compute Device: `{DEVICE}`")

    selected_arch = st.selectbox(
        "Model Architecture",
        ARCH_LIST,
        index=ARCH_LIST.index(default_arch) if default_arch in ARCH_LIST else 0,
    )
    weights_path = st.text_input(
        "Checkpoint Path",
        value=os.path.join(MODEL_DIR, "best_model.pt"),
    )

    st.divider()
    st.subheader("Grad-CAM Controls")
    alpha_val = st.slider("Heatmap Opacity (Alpha)", 0.10, 0.85, 0.45, 0.05)
    cam_threshold = st.slider("Activation Threshold", 0.0, 0.60, 0.25, 0.05)

    st.divider()
    st.subheader("Adaptive IQA Thresholds")
    st.json(IQA_THRESHOLDS)

# ==========================================
# MAIN TABS
# ==========================================
st.title("Congenital Heart Disease (CHD) Ultrasound Pipeline")
st.markdown(
    "**Adaptive Image Quality Assessment (IQA), Multi-Architecture Benchmarking & Grad-CAM Interpretability**"
)

tab_studio, tab_reports = st.tabs([
    "Live Diagnostic & Preprocessing Studio",
    "Benchmark & Ablation Explorer",
])

# ------------------------------------------
# TAB 1: LIVE DIAGNOSTIC STUDIO
# ------------------------------------------
with tab_studio:
    input_mode = st.radio(
        "Select Image Source",
        ["Upload Ultrasound Scan", "Pick from Local Dataset (cleaned_cardium)"],
        horizontal=True,
    )

    img_bgr = None
    source_label = ""

    if input_mode == "Upload Ultrasound Scan":
        uploaded = st.file_uploader(
            "Upload Fetal Echocardiogram (.png, .jpg, .jpeg)",
            type=["png", "jpg", "jpeg"],
        )
        if uploaded is not None:
            file_bytes = np.asarray(bytearray(uploaded.read()), dtype=np.uint8)
            img_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
            source_label = uploaded.name
    else:
        if os.path.isdir(RAW_DIR):
            col_c, col_f = st.columns(2)
            available_classes = [
                d for d in sorted(os.listdir(RAW_DIR))
                if os.path.isdir(os.path.join(RAW_DIR, d))
            ]
            if available_classes:
                chosen_class = col_c.selectbox("Folder (Ground Truth)", available_classes)
                class_folder = os.path.join(RAW_DIR, chosen_class)
                files = [
                    f for f in sorted(os.listdir(class_folder))
                    if f.lower().endswith((".png", ".jpg", ".jpeg"))
                ]
                if files:
                    chosen_file = col_f.selectbox("Select Scan", files)
                    full_p = os.path.join(class_folder, chosen_file)
                    img_bgr = cv2.imread(full_p)
                    source_label = f"{chosen_class}/{chosen_file}"
        else:
            st.info(f"Local directory `{RAW_DIR}` not found. Use file upload instead.")

    if img_bgr is not None:
        # Run Conditions A, B, C from pipeline_utils
        out_a, _ = apply_condition_a_raw(img_bgr)
        out_b, params_b = apply_condition_b_static(img_bgr)
        out_c, params_c, raw_metrics = apply_condition_c_adaptive(img_bgr)

        # Model Inference & Grad-CAM on Condition C
        model, is_loaded = load_checkpoint_model(selected_arch, len(class_names), weights_path)
        if not is_loaded:
            st.warning(
                f"Checkpoint `{weights_path}` not found or mismatched with `{selected_arch}`. "
                "Using base architecture weights."
            )

        out_c_rgb = cv2.cvtColor(out_c, cv2.COLOR_BGR2RGB)
        input_tensor = val_tf(Image.fromarray(out_c_rgb)).unsqueeze(0).to(DEVICE)

        target_layer = get_target_layer(model, selected_arch)
        gradcam = GradCAM(model, target_layer)
        cam, pred_idx, confidence = gradcam.generate(input_tensor)
        if hasattr(gradcam, "remove_hooks"):
            gradcam.remove_hooks()

        region, (cx, cy) = describe_focus_region(cam)
        in_center = (1 / 3 <= cx <= 2 / 3) and (1 / 3 <= cy <= 2 / 3)

        try:
            overlay_bgr = overlay_heatmap(out_c, cam, alpha=alpha_val, threshold=cam_threshold)
        except TypeError:
            overlay_bgr = overlay_heatmap(out_c, cam, alpha=alpha_val)

        # --- TOP METRICS BANNER ---
        st.subheader(f"Diagnostic Result: `{source_label}`")
        m1, m2, m3, m4 = st.columns(4)
        pred_name = class_names[pred_idx].upper()
        m1.metric("Predicted Class", pred_name)
        m2.metric("Model Confidence", f"{confidence * 100:.2f}%")
        m3.metric("Grad-CAM Focus Region", f"{region} ({cx:.2f}, {cy:.2f})")
        m4.metric("Centroid in Center Third", "Yes (Cardiac ROI)" if in_center else "No (Peripheral)")

        st.divider()

        # --- PREPROCESSING ABLATION + GRAD-CAM VISUALS ---
        st.subheader("Preprocessing Conditions (A vs B vs C) & Grad-CAM")
        v1, v2, v3, v4 = st.columns(4)
        with v1:
            st.markdown("**Condition A: Chrome-Stripped Raw**")
            st.image(cv2.cvtColor(out_a, cv2.COLOR_BGR2RGB), use_container_width=True)
        with v2:
            st.markdown("**Condition B: Static Enhancement**")
            st.image(cv2.cvtColor(out_b, cv2.COLOR_BGR2RGB), use_container_width=True)
        with v3:
            st.markdown("**Condition C: Adaptive IQA**")
            st.image(out_c_rgb, use_container_width=True)
        with v4:
            st.markdown("**Condition C + Grad-CAM Overlay**")
            st.image(cv2.cvtColor(overlay_bgr, cv2.COLOR_BGR2RGB), use_container_width=True)

        st.divider()

        # --- IQA TELEMETRY ---
        col_iqa, col_rules = st.columns(2)
        with col_iqa:
            st.subheader("Measured Raw IQA Statistics (Post-Chrome-Strip)")
            iqa_df = pd.DataFrame([raw_metrics]).T.rename(columns={0: "Measured Value"})
            st.dataframe(iqa_df, use_container_width=True)
        with col_rules:
            st.subheader("Condition C Adaptive Decisions Triggered")
            rules_df = pd.DataFrame([params_c]).T.rename(columns={0: "Applied Parameter"})
            st.dataframe(rules_df, use_container_width=True)

# ------------------------------------------
# TAB 2: BENCHMARK & ABLATION EXPLORER
# ------------------------------------------
with tab_reports:
    col_r1, col_r2 = st.columns(2)

    with col_r1:
        st.subheader("5-Fold Cross-Validation Summary")
        if os.path.exists(PIPELINE_LOG_PATH):
            try:
                cv_df = pd.read_excel(PIPELINE_LOG_PATH, sheet_name="cv_summary")
                st.dataframe(cv_df.round(4), use_container_width=True)
            except Exception as e:
                st.info(f"`cv_summary` sheet not found in `{PIPELINE_LOG_PATH}` yet ({e}).")
        else:
            st.info(f"Run `train_classifier.py` to generate `{PIPELINE_LOG_PATH}`.")

    with col_r2:
        st.subheader("Preprocessing Ablation Summary (A vs B vs C)")
        if os.path.exists(MASTER_REPORT_PATH):
            try:
                ab_df = pd.read_excel(MASTER_REPORT_PATH, sheet_name="ablation_summary")
                st.dataframe(ab_df.round(4), use_container_width=True)
            except Exception as e:
                st.info(f"`ablation_summary` sheet not found in `{MASTER_REPORT_PATH}` yet ({e}).")
        else:
            st.info(f"Run `explain_and_report.py` to generate `{MASTER_REPORT_PATH}`.")

    st.divider()
    st.subheader("Held-Out Validation Grad-CAM Audit Log")
    if os.path.exists(MASTER_REPORT_PATH):
        try:
            gcam_df = pd.read_excel(MASTER_REPORT_PATH, sheet_name="gradcam_results")
            st.dataframe(gcam_df, use_container_width=True)
        except Exception as e:
            st.info(f"Could not load `gradcam_results` sheet: {e}")
