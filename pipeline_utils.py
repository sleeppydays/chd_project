import cv2
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support, roc_auc_score, confusion_matrix
)
from torchvision.models import (
    resnet18, ResNet18_Weights,
    mobilenet_v2, MobileNet_V2_Weights,
    efficientnet_b0, EfficientNet_B0_Weights,
)
 
ARCH_LIST = ["resnet18", "mobilenet_v2", "efficientnet_b0"]
IMG_SIZE = 224
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
 
 
def get_group_id(filename):
    """EDIT ME if your filenames encode a patient/study ID, e.g.
    'P00231_frame04.png' -> 'P00231'. As written, every image is its own
    group (no real grouping) -- fine only if each image is genuinely an
    independent patient/study, which is unlikely for ultrasound. Shared
    by scripts 2 and 3 so the CV benchmark and the ablation study use
    identical, aligned splits."""
    return filename
 
# ==========================================
# CHROME REMOVAL (mandatory first step, per spec)
# ==========================================
def crop_scanner_chrome(image, top_frac=0.06, right_text_frac=0.22):
    h, w = image.shape[:2]
    top = int(h * top_frac)
    right = int(w * (1 - right_text_frac))
    return image[top:, :right]
 
 
def mask_to_sector(image, center_frac=(0.50, 0.42), axes_frac=(0.75, 0.85)):
    h, w = image.shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)
    center = (int(w * center_frac[0]), int(h * center_frac[1]))
    axes = (int(w * axes_frac[0] / 2), int(h * axes_frac[1] / 2))
    cv2.ellipse(mask, center, axes, 0, 0, 360, 255, -1)
    if len(image.shape) == 3:
        mask = cv2.merge([mask, mask, mask])
    return cv2.bitwise_and(image, mask)
 
 
def strip_chrome(image):
    return mask_to_sector(crop_scanner_chrome(image))
 
 
# ==========================================
# IMAGE QUALITY METRICS
# ==========================================
def compute_iqa_metrics(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
    hist = cv2.calcHist([gray], [0], None, [256], [0, 256]).ravel()
    hist_norm = hist / max(hist.sum(), 1)
    hist_norm = hist_norm[hist_norm > 0]
    entropy = float(-np.sum(hist_norm * np.log2(hist_norm)))
    return {
        "brightness": float(np.mean(gray)),
        "contrast_std": float(np.std(gray)),
        "sharpness_laplacian_var": float(cv2.Laplacian(gray, cv2.CV_64F).var()),
        "entropy": entropy,
    }
 
 
# ==========================================
# ENHANCEMENT PRIMITIVES
# ==========================================
def enhance_brightness_gamma(image, gamma):
    inv_gamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in range(256)]).astype("uint8")
    return cv2.LUT(image, table)
 
 
def enhance_contrast_clahe(image, clip_limit, tile_grid=(8, 8)):
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid)
    if len(image.shape) == 3:
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        l2 = clahe.apply(l)
        return cv2.cvtColor(cv2.merge((l2, a, b)), cv2.COLOR_LAB2BGR)
    return clahe.apply(image)
 
 
def denoise_edge_preserving(image):
    return cv2.bilateralFilter(image, d=9, sigmaColor=75, sigmaSpace=75)
 
 
def enhance_sharpness_unsharp(image, amount=1.5, sigma=1.0):
    blurred = cv2.GaussianBlur(image, (0, 0), sigma)
    return cv2.addWeighted(image, 1 + amount, blurred, -amount, 0)
 
 
# ==========================================
# THREE PREPROCESSING CONDITIONS (for the ablation study)
# ==========================================
def apply_condition_a_raw(image):
    """Condition A: chrome-stripped only, zero enhancement."""
    return strip_chrome(image), {}
 
 
def apply_condition_b_static(image, gamma=1.2, clahe_clip=2.0):
    """Condition B: the old fixed pipeline, applied to every image regardless
    of whether it needs it."""
    out = strip_chrome(image)
    out = enhance_brightness_gamma(out, gamma)
    out = enhance_contrast_clahe(out, clahe_clip)
    out = denoise_edge_preserving(out)
    out = enhance_sharpness_unsharp(out)
    params = {"gamma": gamma, "clahe_clip_limit": clahe_clip,
              "denoise_applied": True, "sharpen_applied": True}
    return out, params
 
 
IQA_THRESHOLDS = {
    "brightness_low": 80.0, "brightness_high": 170.0,
    "contrast_low_std": 35.0,
    "sharpness_high": 800.0,   # above this: likely speckle-dominated -> denoise
    "sharpness_low": 150.0,    # below this: reads flat/soft -> sharpen
}
 
 
def apply_condition_c_adaptive(image, thresholds=IQA_THRESHOLDS):
    """Condition C (proposed): chrome-stripped, then only the corrections
    each image's own measured statistics indicate it needs. Returns the
    raw (pre-correction, post-chrome-strip) metrics alongside the applied
    params so callers can log both and run confound diagnostics on the
    raw numbers specifically.
 
    NOTE on the sharpness metric: Laplacian variance is a standard but
    imperfect proxy here -- both real anatomical edges and speckle noise
    raise it, so this heuristic can't fully tell them apart. A wavelet- or
    local-variance-based noise estimator would be a stronger choice for a
    paper; documenting that as a limitation is more honest than pretending
    this discriminates perfectly.
    """
    out = strip_chrome(image)
    raw_metrics = compute_iqa_metrics(out)
    params = {"gamma": None, "clahe_clip_limit": None,
              "denoise_applied": False, "sharpen_applied": False}
 
    if raw_metrics["brightness"] < thresholds["brightness_low"]:
        params["gamma"] = 1.3
    elif raw_metrics["brightness"] > thresholds["brightness_high"]:
        params["gamma"] = 0.8
    if params["gamma"] is not None:
        out = enhance_brightness_gamma(out, params["gamma"])
 
    if raw_metrics["contrast_std"] < thresholds["contrast_low_std"]:
        clip = float(np.clip(4.0 - raw_metrics["contrast_std"] / 15.0, 1.5, 4.0))
        out = enhance_contrast_clahe(out, clip)
        params["clahe_clip_limit"] = clip
 
    if raw_metrics["sharpness_laplacian_var"] > thresholds["sharpness_high"]:
        out = denoise_edge_preserving(out)
        params["denoise_applied"] = True
    elif raw_metrics["sharpness_laplacian_var"] < thresholds["sharpness_low"]:
        out = enhance_sharpness_unsharp(out)
        params["sharpen_applied"] = True
 
    return out, params, raw_metrics
 
 
# ==========================================
# MODEL FACTORY (3 architectures, shared by CV benchmark + ablation + Grad-CAM)
# ==========================================
def get_model(arch, num_classes, pretrained=True):
    if arch == "resnet18":
        m = resnet18(weights=ResNet18_Weights.DEFAULT if pretrained else None)
        m.fc = nn.Linear(m.fc.in_features, num_classes)
    elif arch == "mobilenet_v2":
        m = mobilenet_v2(weights=MobileNet_V2_Weights.DEFAULT if pretrained else None)
        m.classifier[1] = nn.Linear(m.classifier[1].in_features, num_classes)
    elif arch == "efficientnet_b0":
        m = efficientnet_b0(weights=EfficientNet_B0_Weights.DEFAULT if pretrained else None)
        m.classifier[1] = nn.Linear(m.classifier[1].in_features, num_classes)
    else:
        raise ValueError(f"Unknown architecture: {arch}. Choose from {ARCH_LIST}.")
    return m
 
 
def get_target_layer(model, arch):
    if arch == "resnet18":
        return model.layer4[-1]
    elif arch in ("mobilenet_v2", "efficientnet_b0"):
        return model.features[-1]
    raise ValueError(f"Unknown architecture: {arch}. Choose from {ARCH_LIST}.")
 
 
# ==========================================
# GRAD-CAM (Upgraded in pipeline_utils.py)
# ==========================================
class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.activations = None
        self.gradients = None
        self.handles = [
            target_layer.register_forward_hook(self._save_activation),
            target_layer.register_full_backward_hook(self._save_gradient),
        ]

    def _save_activation(self, module, inp, out):
        self.activations = out.detach()

    def _save_gradient(self, module, grad_in, grad_out):
        self.gradients = grad_out[0].detach()

    def generate(self, input_tensor, class_idx=None):
        self.model.zero_grad()
        output = self.model(input_tensor)
        probs = torch.softmax(output, dim=1)[0]
        if class_idx is None:
            class_idx = int(output.argmax(dim=1).item())
        output[:, class_idx].sum().backward()

        # Vectorized channel weighting (replaces slow Python loop)
        weights = self.gradients[0].mean(dim=(1, 2), keepdim=True)
        cam = (weights * self.activations[0]).sum(dim=0)
        cam = torch.relu(cam)
        cam -= cam.min()
        if cam.max() > 0:
            cam /= cam.max()
        return cam.cpu().numpy(), class_idx, float(probs[class_idx].item())

    def remove_hooks(self):
        for h in self.handles:
            h.remove()


def overlay_heatmap(image_bgr, cam, alpha=0.45, threshold=0.25):
    h, w = image_bgr.shape[:2]
    # Smooth bicubic upscaling + Gaussian smoothing to eliminate 7x7 grid blockiness
    cam_resized = cv2.resize(cam, (w, h), interpolation=cv2.INTER_CUBIC)
    cam_resized = cv2.GaussianBlur(cam_resized, (0, 0), sigmaX=5, sigmaY=5)
    if cam_resized.max() > 0:
        cam_resized = (cam_resized - cam_resized.min()) / (cam_resized.max() - cam_resized.min())

    heatmap = cv2.applyColorMap(np.uint8(255 * cam_resized), cv2.COLORMAP_JET)
    blended = cv2.addWeighted(heatmap, alpha, image_bgr, 1 - alpha, 0)

    # Only apply heatmap inside the non-black ultrasound sector where activation > threshold
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY) if len(image_bgr.shape) == 3 else image_bgr
    active_mask = (gray > 5) & (cam_resized >= threshold)
    out = image_bgr.copy()
    out[active_mask] = blended[active_mask]
    return out
 
 
# ==========================================
# CLINICAL METRICS (binary: positive_class vs the other)
# ==========================================
def compute_binary_metrics(y_true, y_pred, y_score_positive, class_names, positive_class="chd"):
    """y_score_positive: predicted probability of `positive_class` for each
    sample. Assumes exactly 2 classes."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    pos_idx = class_names.index(positive_class)
    neg_idx = [i for i in range(len(class_names)) if i != pos_idx][0]
 
    acc = accuracy_score(y_true, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="binary", pos_label=pos_idx, zero_division=0
    )
    try:
        roc_auc = roc_auc_score((y_true == pos_idx).astype(int), y_score_positive)
    except ValueError:
        roc_auc = float("nan")  # only one class present, e.g. a degenerate tiny fold
 
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(class_names))))
    tp = cm[pos_idx, pos_idx]
    fn = cm[pos_idx].sum() - tp
    tn = cm[neg_idx, neg_idx]
    fp = cm[neg_idx].sum() - tn
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else float("nan")
    specificity = tn / (tn + fp) if (tn + fp) > 0 else float("nan")
 
    return {
        "accuracy": acc, "precision": precision, "recall": recall, "f1": f1,
        "roc_auc": roc_auc, "sensitivity": sensitivity, "specificity": specificity,
    }


def describe_focus_region(cam):
    h, w = cam.shape
    ys, xs = np.mgrid[0:h, 0:w]
    total = cam.sum()
    if total == 0:
        return "diffuse / no strong focus", (0.5, 0.5)
    cy = (ys * cam).sum() / total
    cx = (xs * cam).sum() / total
    row = "upper" if cy < h / 3 else ("lower" if cy > 2 * h / 3 else "middle")
    col = "left" if cx < w / 3 else ("right" if cx > 2 * w / 3 else "center")
    region = "center" if (row, col) == ("middle", "center") else f"{row}-{col}"
    return region, (cx / w, cy / h)
