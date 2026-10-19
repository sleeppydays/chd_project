# CARDIUM-AI: Congenital Heart Disease (CHD) Fetal Ultrasound Diagnostic & Explainability Suite

An end-to-end clinical computer vision and deep learning framework for detecting **Congenital Heart Disease (CHD)** from fetal echocardiogram scans (`chd` vs. `non_chd`).

Rather than treating medical image classification as a naive black-box training loop, **CARDIUM-AI** integrates automated scanner chrome removal, an **Adaptive Image Quality Assessment (IQA)** preprocessing engine, **acquisition confound statistical auditing**, **5-fold Stratified Group Cross-Validation** across three convolutional architectures, and **sector-masked Grad-CAM** spatial interpretability.

## Key Engineering Highlights

* **Adaptive Image Quality Assessment (IQA) & Chrome Stripping:** Automatically crops ultrasound scanner UI overlays and applies an elliptical sector mask (`strip_chrome`) before measuring four pre-enhancement telemetry metrics: mean brightness, contrast standard deviation, Laplacian variance sharpness, and Shannon entropy. Enhancement primitives (Gamma LUT, LAB-space CLAHE, edge-preserving bilateral filtering, and Gaussian unsharp masking) are triggered dynamically per scan only when measured statistics cross calibrated thresholds.

* **Acquisition Confound Statistical Auditing:** Runs automated two-sample Welch's $t$-tests ($\alpha = 0.05$) on raw pre-enhancement IQA metrics and $\chi^2$ contingency tests on adaptive decision rates across classes to detect scanner/preset shortcuts before downstream model evaluation.

* **Multi-Architecture 5-Fold Stratified Group Benchmark:** Evaluates **ResNet-18**, **MobileNetV2**, and **EfficientNet-B0** using `StratifiedGroupKFold` to prevent patient/study-level data leakage, applying anatomically valid augmentations (no horizontal/vertical flips that violate cardiac situs orientation).

* **High-Throughput WSL2 / CUDA Data Pipeline:** Engineered with multi-worker data prefetching (`num_workers=4/8`), page-locked host memory (`pin_memory=True`), non-blocking H2D tensor transfers, and CuDNN auto-tuning (`torch.backends.cudnn.benchmark = True`), accelerating training throughput from **2.85 it/s to 61.69 it/s (>21x speedup)**.

* **3-Condition Preprocessing Ablation Study:** Evaluates fixed train/validation splits across **Condition A** (Chrome-Stripped Raw), **Condition B** (Static Enhancement Pipeline), and **Condition C** (Proposed Adaptive IQA Pipeline).

* **Vectorized Sector-Masked Grad-CAM & Interactive Studio:** Computes vectorized gradient-weighted class activation maps at the final convolutional feature layer, calculates normalized $(c_x, c_y)$ activation centroids to verify focus within the central cardiac region of interest (ROI), and suppresses background bleed via thresholded bicubic overlays inside an interactive **Streamlit** web application.

## Repository Structure

```
chd_project/
├── pipeline_utils.py        # Core primitives: chrome stripping, IQA metrics, A/B/C conditions, Model Factory, Grad-CAM
├── preprocess_and_log.py    # Step 1: Generates Condition A/B/C datasets, logs IQA telemetry, runs Welch's t-test & Chi2 audit
├── train_classifier.py      # Step 2: 5-fold StratifiedGroupKFold benchmark across 3 CNNs + full-dataset deployment refit
├── explain_and_report.py    # Step 3: Aligned A/B/C ablation study, held-out Grad-CAM centroid audit, and formatted Excel report
├── app.py                   # Interactive Streamlit GUI for real-time A/B/C preprocessing inspection & Grad-CAM inference
└── .gitignore               # Excludes raw patient datasets, model checkpoints (.pt), and generated heatmaps
```

## Methodology & Pipeline Architecture

### 1. Scanner Chrome Removal & Adaptive IQA (`preprocess_and_log.py`)

Raw ultrasound frames frequently contain vendor text, depth rulers, and acoustic fan borders that deep neural networks can exploit as shortcut features. Every image first passes through `strip_chrome()`:

1. **Top & Right UI Crop:** Removes top $6\%$ banner metadata and right $22\%$ scanner parameter text.
2. **Elliptical Sector Mask:** Masks the acoustic cone centered at $(0.50w, 0.42h)$ with axes $(0.75w, 0.85h)$.

Next, `compute_iqa_metrics()` evaluates grayscale brightness ($\mu$), contrast ($\sigma$), sharpness ($\text{Var}(\nabla^2 I)$), and Shannon entropy ($H = -\sum p_i \log_2 p_i$) to generate three parallel datasets:

| Condition | Description | Enhancement Logic |
| :--- | :--- | :--- |
| **Condition A (`A_raw`)** | Chrome-Stripped Baseline | `strip_chrome(img)` only; zero pixel intensity modification. |
| **Condition B (`B_static`)** | Fixed Enhancement | Unconditionally applies $\gamma = 1.2$, CLAHE ($\text{clip}=2.0$), bilateral filtering ($d=9, \sigma_c=75, \sigma_s=75$), and unsharp masking ($\text{amount}=1.5, \sigma=1.0$). |
| **Condition C (`C_adaptive`)** | **Adaptive IQA (Proposed)** | Dynamically triggers $\gamma \in \{1.3, 0.8\}$ if $\mu \notin [80, 170]$, adaptive LAB-CLAHE ($\text{clip} = \text{clip}(4.0 - \sigma/15, 1.5, 4.0)$) if $\sigma < 35$, bilateral denoising if $\text{Var}(\nabla^2 I) > 800$, or unsharp masking if $\text{Var}(\nabla^2 I) < 150$. |

### 2. Multi-Architecture Cross-Validation (`train_classifier.py`)

* **Architectures:** `resnet18`, `mobilenet_v2`, and `efficientnet_b0` initialized with ImageNet-1K pretrained weights and custom linear classification heads.
* **Class Imbalance Handling:** Inverse class-frequency weighting passed to `nn.CrossEntropyLoss`.
* **Anatomically Safe Augmentation:** Random rotation ($\pm 5^\circ$), color jitter ($\pm 15\%$ brightness/contrast), and random erasing ($p=0.5$), deliberately excluding flips to preserve anatomical chamber orientation.
* **Evaluation Metrics:** Accuracy, Precision, Recall, F1-Score, ROC-AUC, Sensitivity, and Specificity. Selects the top-performing architecture by mean cross-validation F1-score and refits a deployable checkpoint (`models/best_model.pt`).

### 3. Ablation & Spatial Explainability (`explain_and_report.py`)

* Holds a single grouped, stratified train/validation split fixed across `A_raw`, `B_static`, and `C_adaptive` to isolate the exact contribution of adaptive preprocessing.
* Generates Grad-CAM overlays on unseen validation scans and computes the normalized activation centroid:
  $$
  (c_x, c_y) = \left( \frac{\sum_{x,y} x \cdot \text{CAM}(x,y)}{W \sum_{x,y} \text{CAM}(x,y)}, \frac{\sum_{x,y} y \cdot \text{CAM}(x,y)}{H \sum_{x,y} \text{CAM}(x,y)} \right)
  $$
* Audits what percentage of validation predictions focus within the central third of the frame ($1/3 \le c_x, c_y \le 2/3$) where the fetal cardiac chambers reside, exporting formatted tables to `ieee_master_report.xlsx` and `ablation_summary.md`.

## Installation & Quickstart

### 1. Clone the Repository & Set Up Environment
```bash
git clone https://github.com/sleeppydays/chd_project.git
cd chd_project

conda create -n chd_ai python=3.10 -y
conda activate chd_ai

# Install PyTorch with CUDA support + pipeline dependencies
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install opencv-python-headless numpy pandas scipy scikit-learn openpyxl tabulate tqdm pillow streamlit
```

### 2. Run the 3-Stage Pipeline
Ensure your dataset is organized under `/home/aaush/cardium_native/cleaned_cardium/{chd,non_chd}` (or update `INPUT_ROOT` / `REPORT_DIR` at the top of each script):
```bash
# Stage 1: Build Condition A/B/C datasets, log IQA metrics, and run confound diagnostics
python preprocess_and_log.py

# Stage 2: Run 5-fold StratifiedGroupKFold across ResNet-18, MobileNetV2, and EfficientNet-B0
python train_classifier.py

# Stage 3: Run A vs. B vs. C ablation study, generate Grad-CAMs, and compile master Excel report
python explain_and_report.py
```

### 3. Launch the Interactive Diagnostic GUI
```bash
streamlit run app.py
```
Open `http://localhost:8501` in your browser to upload fetal echocardiogram scans, compare Conditions A, B, and C side by side, inspect live IQA telemetry, and adjust Grad-CAM opacity and activation thresholds in real time.

## Author

**Ayush Singh**  
B.Tech Computer Science & Engineering, SASTRA University | BS Data Science, IIT Madras  
[GitHub (@sleeppydays)](https://github.com/sleeppydays) • [Email](mailto:aaush.singh8307@gmail.com)