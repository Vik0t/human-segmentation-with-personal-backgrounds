import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F_torch
from PIL import Image
from torchvision.transforms import functional as F_vision
import numpy as np
import time
import os
ZERO_DCE_MAX_SIDE = 256  # very small for fast enhancement
MODEL_PATH = 'segmentation/RobustVideoMatting/rvm_mobilenetv3.pth'
def segment_and_composite(
    frame_bgr: np.ndarray,
    bg_rgb: np.ndarray,
    model_rvm,
    rec_states,
    device: str,
    model_dce=None,
    enhance_method: str = 'none',
    brightness_threshold: int = 60,
    rvm_input_scale: float = 0.5,
    downsample_ratio: float = 0.25,
    zero_dce_max_side: int = 256
):
    """
    Segments a human from a camera frame and composites it onto a background.

    Args:
        frame_bgr (np.ndarray): Input frame from camera (H, W, 3) in BGR (uint8, 0-255).
        bg_rgb (np.ndarray): Background image (H, W, 3) in RGB (uint8, 0-255). Must match output resolution.
        model_rvm: Loaded RobustVideoMatting model (in eval mode).
        rec_states: List of 4 recurrent states from previous RVM inference (or [None]*4 initially).
        device (str): 'cuda', 'mps', or 'cpu'.
        model_dce: Optional Zero-DCE++ model for low-light enhancement.
        enhance_method (str): 'none', 'clahe_gamma', or 'zero_dce'.
        brightness_threshold (int): Mean brightness below which enhancement is triggered.
        rvm_input_scale (float): Scale factor for RVM input (e.g., 0.5 for half resolution).
        downsample_ratio (float): RVM internal downsample ratio.
        zero_dce_max_side (int): Max side length for Zero-DCE++ input (for speed).

    Returns:
        out_bgr (np.ndarray): Composited output frame in BGR (uint8, same shape as input).
        new_rec_states: Updated recurrent states for next frame.
        alpha_map (np.ndarray, optional): Alpha matte (H, W, float32, 0–1) if needed for debugging.
    """
    H, W = frame_bgr.shape[:2]
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

    # --- Decide if enhancement is needed ---
    needs_enh = (frame_bgr.mean() < brightness_threshold) if enhance_method != 'none' else False
    matting_rgb = frame_rgb.copy()

    if needs_enh:
        if enhance_method == 'zero_dce' and model_dce is not None:
            # Resize for speed
            scale = min(1.0, zero_dce_max_side / max(H, W))
            if scale < 1.0:
                small = cv2.resize(frame_rgb, (int(W * scale), int(H * scale)), interpolation=cv2.INTER_AREA)
            else:
                small = frame_rgb
            t = torch.as_tensor(small, device='cpu').permute(2, 0, 1).float().div_(255.0).unsqueeze(0).to(device)
            with torch.inference_mode():
                enhanced = model_dce(t)
            enhanced = (enhanced[0].permute(1, 2, 0).cpu().mul(255.0).clamp(0, 255).byte().numpy())
            if scale < 1.0:
                enhanced = cv2.resize(enhanced, (W, H), interpolation=cv2.INTER_LINEAR)
            matting_rgb = enhanced
        elif enhance_method == 'clahe_gamma':
            gamma = 1.5
            inv_gamma = 1.0 / gamma
            table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in np.arange(256)]).astype("uint8")
            adjusted = cv2.LUT(frame_bgr, table)
            lab = cv2.cvtColor(adjusted, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
            l = clahe.apply(l)
            lab = cv2.merge((l, a, b))
            matting_rgb = cv2.cvtColor(cv2.cvtColor(lab, cv2.COLOR_LAB2BGR), cv2.COLOR_BGR2RGB)

    # --- RVM Inference ---
    h_in, w_in = int(H * rvm_input_scale), int(W * rvm_input_scale)
    small_rgb = cv2.resize(matting_rgb, (w_in, h_in), interpolation=cv2.INTER_AREA)
    src_tensor = torch.from_numpy(small_rgb).to(device, non_blocking=True).permute(2, 0, 1).float().div_(255.0).unsqueeze(0)

    with torch.inference_mode():
        fgr, pha, *new_rec = model_rvm(src_tensor, *rec_states, downsample_ratio=downsample_ratio)

    # --- Upsample & smooth alpha ---
    pha_up = F_torch.interpolate(pha, size=(H, W), mode='bilinear', align_corners=False)
    kernel = torch.ones(1, 1, 3, 3, device=device, dtype=pha_up.dtype) / 9.0
    pha_smooth = F_torch.conv2d(pha_up, kernel, padding=1)
    alpha = pha_smooth.clamp(0, 1)

    # --- Composite ---
    orig_tensor = torch.from_numpy(frame_rgb).to(device, non_blocking=True).permute(2, 0, 1).float().div_(255.0).unsqueeze(0)
    bg_tensor = torch.from_numpy(bg_rgb).to(device, non_blocking=True).permute(2, 0, 1).float().div_(255.0).unsqueeze(0)
    composited = orig_tensor * alpha + bg_tensor * (1 - alpha)
    out_rgb = (composited[0].permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
    out_bgr = cv2.cvtColor(out_rgb, cv2.COLOR_RGB2BGR)

    # Optional: return alpha for debugging or post-processing
    alpha_np = alpha[0, 0].cpu().numpy()  # (H, W)

    return out_bgr, new_rec, alpha_np


class ZeroDCEEnhancer(nn.Module):
    def __init__(self):
        super().__init__()
        self.relu = nn.ReLU(inplace=True)
        self.e_conv1 = nn.Conv2d(3, 32, 3, 1, 1)
        self.e_conv2 = nn.Conv2d(32, 32, 3, 1, 1)
        self.e_conv3 = nn.Conv2d(32, 32, 3, 1, 1)
        self.e_conv4 = nn.Conv2d(32, 32, 3, 1, 1)
        self.e_conv5 = nn.Conv2d(64, 32, 3, 1, 1)
        self.e_conv6 = nn.Conv2d(64, 32, 3, 1, 1)
        self.e_conv7 = nn.Conv2d(64, 24, 3, 1, 1)

    def enhance(self, x, curves):
        r1, r2, r3, r4, r5, r6, r7, r8 = torch.split(curves, 3, dim=1)
        for r in [r1, r2, r3, r4, r5, r6, r7, r8]:
            x = x + r * (x * x - x)
        return torch.clamp(x, 0, 1)

    def forward(self, x):
        x1 = self.relu(self.e_conv1(x))
        x2 = self.relu(self.e_conv2(x1))
        x3 = self.relu(self.e_conv3(x2))
        x4 = self.relu(self.e_conv4(x3))
        x5 = self.relu(self.e_conv5(torch.cat([x3, x4], dim=1)))
        x6 = self.relu(self.e_conv6(torch.cat([x2, x5], dim=1)))
        curves = self.e_conv7(torch.cat([x1, x6], dim=1))
        return self.enhance(x, curves)
def load_zero_dce(path, device):
    model = ZeroDCEEnhancer().eval().to(device)
    if os.path.exists(path):
        sd = torch.load(path, map_location=device)
        if any(k.startswith('module.') for k in sd.keys()):
            sd = {k.replace('module.', ''): v for k, v in sd.items()}
        model.load_state_dict(sd)
        print("Zero-DCE++ loaded successfully.")
        return model
    else:
        print(f"Zero-DCE++ weights not found at {path}. Enhancement disabled.")
        return None

def enhance_zero_dce(frame_rgb, model, device, max_side=ZERO_DCE_MAX_SIDE):
    h, w = frame_rgb.shape[:2]
    scale = min(1.0, max_side / max(h, w))
    if scale < 1.0:
        frame_small = cv2.resize(frame_rgb, (int(w*scale), int(h*scale)), interpolation=cv2.INTER_AREA)
    else:
        frame_small = frame_rgb
    t = torch.as_tensor(frame_small, device='cpu').permute(2,0,1).float().div_(255.0).unsqueeze(0).to(device)
    with torch.inference_mode():
        enhanced = model(t)
    enhanced = (enhanced[0].permute(1,2,0).cpu().mul(255.0).clamp(0,255).byte().numpy())
    if scale < 1.0:
        enhanced = cv2.resize(enhanced, (w,h), interpolation=cv2.INTER_LINEAR)
    return enhanced


DEVICE = 'mps' if torch.backends.mps.is_available() else 'cuda' if torch.cuda.is_available() else 'cpu'

# Load RVM
from RobustVideoMatting.model import MattingNetwork
model_rvm = MattingNetwork('mobilenetv3').eval().to(DEVICE)
state = torch.load(MODEL_PATH, map_location=DEVICE)
if any(k.startswith('module.') for k in state.keys()):
    state = {k.replace('module.',''):v for k,v in state.items()}
model_rvm.load_state_dict(state)

# Optionally load Zero-DCE++
model_dce = load_zero_dce('models/zero_dce++.pth', DEVICE)  # reuse your function

# Load background
bg_rgb = np.array(Image.open("image.png").convert("RGB").resize((1280, 720)))

rec = [None] * 4
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

while True:
    ret, frame_bgr = cap.read()
    if not ret: break

    out_bgr, rec, alpha = segment_and_composite(
        frame_bgr=frame_bgr,
        bg_rgb=bg_rgb,
        model_rvm=model_rvm,
        rec_states=rec,
        device=DEVICE,
        model_dce=model_dce,
        enhance_method='zero_dce',  # or 'clahe_gamma' or 'none'
        brightness_threshold=60,
        rvm_input_scale=0.5,
        downsample_ratio=0.25
    )

    cv2.imshow('Result', out_bgr)
    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
