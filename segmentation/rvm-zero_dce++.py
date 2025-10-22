import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F_torch
from PIL import Image
from torchvision.transforms import functional as F_vision
import numpy as np
import time
import os

# ======================
# CONFIGURATION
# ======================
DEVICE = 'mps' if torch.mps.is_available() else 'cuda' if torch.cuda.is_available() else 'cpu'
MODEL_PATH = '/Users/vik0t/hackatons/human-segment/segmentation/RobustVideoMatting/rvm_mobilenetv3.pth'
BG_PATH = "segmentation/image.png"
ZERO_DCE_PATH = "models/zero_dce++.pth"
OUTPUT_WIDTH, OUTPUT_HEIGHT = 1280, 720
DOWNSAMPLE_RATIO = 0.25
ENHANCE_BRIGHTNESS_THRESHOLD = 60
ENHANCE_METHOD = 'zero_dce'  # 'zero_dce', 'clahe_gamma', 'none'
ENABLE_DEBUG = False

print(f"Using device: {DEVICE}")

# ======================
# ZERO-DCE++ MODEL
# ======================
class ZeroDCEEnhancer(nn.Module):
    def __init__(self):
        super().__init__()
        self.relu = nn.ReLU(inplace=True)
        
        self.e_conv1 = nn.Conv2d(3, 32, 3, 1, 1, bias=True)
        self.e_conv2 = nn.Conv2d(32, 32, 3, 1, 1, bias=True)
        self.e_conv3 = nn.Conv2d(32, 32, 3, 1, 1, bias=True)
        self.e_conv4 = nn.Conv2d(32, 32, 3, 1, 1, bias=True)
        self.e_conv5 = nn.Conv2d(64, 32, 3, 1, 1, bias=True)  # ← 64 in!
        self.e_conv6 = nn.Conv2d(64, 32, 3, 1, 1, bias=True)  # ← 64 in!
        self.e_conv7 = nn.Conv2d(64, 24, 3, 1, 1, bias=True)  # ← 64 in!

    def enhance(self, x, curves):
        r1, r2, r3, r4, r5, r6, r7, r8 = torch.split(curves, 3, dim=1)
        for r in [r1, r2, r3, r4, r5, r6, r7, r8]:
            x = x + r * (torch.pow(x, 2) - x)
        return torch.clamp(x, 0, 1)

    def forward(self, x):
        x1 = self.relu(self.e_conv1(x))
        x2 = self.relu(self.e_conv2(x1))
        x3 = self.relu(self.e_conv3(x2))
        x4 = self.relu(self.e_conv4(x3))
        
        # Concatenate features (skip connections)
        x5 = self.relu(self.e_conv5(torch.cat([x3, x4], dim=1)))  # 32+32=64
        x6 = self.relu(self.e_conv6(torch.cat([x2, x5], dim=1)))  # 32+32=64
        curves = self.e_conv7(torch.cat([x1, x6], dim=1))        # 32+32=64
        
        return self.enhance(x, curves)

def load_zero_dce(path, device):
    model = ZeroDCEEnhancer().eval().to(device)
    if os.path.exists(path):
        state_dict = torch.load(path, map_location=device)
        # Original weights use DataParallel keys
        if 'module.' in list(state_dict.keys())[0]:
            state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
        model.load_state_dict(state_dict, strict=True)
        print("Zero-DCE++ loaded successfully.")
    else:
        print(f"Warning: Zero-DCE++ weights not found at {path}. Enhancement will be disabled.")
        return None
    return model

def enhance_zero_dce(frame_rgb, model, device, max_side=640):
    h, w = frame_rgb.shape[:2]
    scale = min(1.0, max_side / max(h, w))
    if scale < 1:
        new_w, new_h = int(w * scale), int(h * scale)
        input_rgb = cv2.resize(frame_rgb, (new_w, new_h), interpolation=cv2.INTER_AREA)
    else:
        input_rgb = frame_rgb

    tensor = torch.from_numpy(input_rgb).to(device, non_blocking=True).permute(2, 0, 1).float().div_(255.0).unsqueeze_(0)
    with torch.no_grad():
        enhanced = model(tensor)
    enhanced = (enhanced.squeeze(0).permute(1, 2, 0) * 255).byte().cpu().numpy()
    if scale < 1:
        enhanced = cv2.resize(enhanced, (w, h), interpolation=cv2.INTER_LINEAR)
    return enhanced

# ======================
# OTHER HELPERS (CLAHE, etc.)
# ======================
def enhance_clahe_gamma(frame_bgr):
    gamma = 1.5
    inv_gamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
    frame_bgr = cv2.LUT(frame_bgr, table)
    lab = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    l = clahe.apply(l)
    lab = cv2.merge((l, a, b))
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

def needs_enhancement(frame_bgr, threshold=60):
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    return gray.mean() < threshold

def np_rgb_to_tensor(np_rgb, device):
    return torch.from_numpy(np_rgb).to(device, non_blocking=True).permute(2, 0, 1).float().div_(255.0).unsqueeze_(0)

def load_background(bg_path, width, height, device):
    if not os.path.exists(bg_path):
        bg_pil = Image.new("RGB", (width, height), (0, 255, 0))
    else:
        bg_pil = Image.open(bg_path).convert("RGB")
    bg_resized = bg_pil.resize((width, height), Image.LANCZOS)
    return F_vision.to_tensor(bg_resized).to(device, non_blocking=True).unsqueeze(0)

# ======================
# SETUP
# ======================
bg_tensor = load_background(BG_PATH, OUTPUT_WIDTH, OUTPUT_HEIGHT, DEVICE)

# Load RVM
from RobustVideoMatting.model import MattingNetwork
model_rvm = MattingNetwork('mobilenetv3').eval().to(DEVICE)
model_rvm.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))

# Load Zero-DCE++
model_dce = None
if ENHANCE_METHOD == 'zero_dce':
    model_dce = load_zero_dce(ZERO_DCE_PATH, DEVICE)

# Compile only on CUDA
if DEVICE == 'cuda' and hasattr(torch, 'compile'):
    model_rvm = torch.compile(model_rvm, mode="reduce-overhead")
    if model_dce:
        model_dce = torch.compile(model_dce, mode="reduce-overhead")

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

rec = [None] * 4
fps = 0.0
fps_alpha = 0.1

# ======================
# MAIN LOOP
# ======================
with torch.no_grad():
    while True:
        start_time = time.time()
        ret, frame_bgr = cap.read()
        if not ret:
            break

        frame_bgr = cv2.resize(frame_bgr, (OUTPUT_WIDTH, OUTPUT_HEIGHT))
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

        should_enhance = needs_enhancement(frame_bgr, ENHANCE_BRIGHTNESS_THRESHOLD)
        matting_input_rgb = frame_rgb

        if should_enhance and ENHANCE_METHOD != 'none':
            if ENHANCE_METHOD == 'zero_dce' and model_dce is not None:
                matting_input_rgb = enhance_zero_dce(frame_rgb, model_dce, DEVICE, max_side=640)
            elif ENHANCE_METHOD == 'clahe_gamma':
                enhanced_bgr = enhance_clahe_gamma(frame_bgr)
                matting_input_rgb = cv2.cvtColor(enhanced_bgr, cv2.COLOR_BGR2RGB)

        src_tensor = np_rgb_to_tensor(matting_input_rgb, DEVICE)
        _, pha, *rec = model_rvm(src_tensor, *rec, downsample_ratio=DOWNSAMPLE_RATIO)

        orig_tensor = np_rgb_to_tensor(frame_rgb, DEVICE)
        pha_up = F_torch.interpolate(pha, size=(OUTPUT_HEIGHT, OUTPUT_WIDTH), mode='bilinear', align_corners=False)
        com = orig_tensor * pha_up + bg_tensor * (1 - pha_up)
        output_rgb = (com.squeeze(0).permute(1, 2, 0) * 255).byte().cpu().numpy()
        output_bgr = cv2.cvtColor(output_rgb, cv2.COLOR_RGB2BGR)

        # --- DEBUG / DISPLAY ---
        if ENABLE_DEBUG:
            src_orig_tensor = np_rgb_to_tensor(frame_rgb, DEVICE)
            rec_debug = [r.clone() if r is not None else None for r in rec]
            _, pha_orig, *_ = model_rvm(src_orig_tensor, *rec_debug, downsample_ratio=DOWNSAMPLE_RATIO)

            grid_h, grid_w = 360, 640
            orig_small = cv2.resize(frame_bgr, (grid_w, grid_h))
            enh_small = cv2.resize(cv2.cvtColor(matting_input_rgb, cv2.COLOR_RGB2BGR), (grid_w, grid_h))
            alpha_orig_disp = cv2.resize((pha_orig.squeeze().cpu().numpy() * 255).astype(np.uint8), (grid_w, grid_h))
            alpha_enh_disp = cv2.resize((pha.squeeze().cpu().numpy() * 255).astype(np.uint8), (grid_w, grid_h))

            top_row = np.hstack((orig_small, enh_small))
            bottom_row = np.hstack((cv2.cvtColor(alpha_orig_disp, cv2.COLOR_GRAY2BGR),
                                    cv2.cvtColor(alpha_enh_disp, cv2.COLOR_GRAY2BGR)))
            grid = np.vstack((top_row, bottom_row))

            cv2.putText(grid, f"FPS: {fps:.1f} | DEBUG | {ENHANCE_METHOD}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
            cv2.imshow('Debug Grid', grid)
            cv2.destroyWindow('Output')
        else:
            tag = ENHANCE_METHOD.upper() if should_enhance else "NONE"
            cv2.putText(output_bgr, f"FPS: {fps:.1f} | Enh: {tag}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.imshow('Output', output_bgr)
            cv2.destroyWindow('Debug Grid')

        # --- Controls ---
        elapsed = time.time() - start_time
        current_fps = 1.0 / max(elapsed, 1e-6)
        fps = fps_alpha * current_fps + (1 - fps_alpha) * fps

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('t'):
            ENABLE_DEBUG = not ENABLE_DEBUG
            print(f"Debug mode: {'ON' if ENABLE_DEBUG else 'OFF'}")
        elif key == ord('1'):
            ENHANCE_METHOD = 'none'
            print("Enhancement: NONE")
        elif key == ord('2'):
            ENHANCE_METHOD = 'clahe_gamma'
            print("Enhancement: CLAHE+Gamma")
        elif key == ord('3'):
            if model_dce:
                ENHANCE_METHOD = 'zero_dce'
                print("Enhancement: Zero-DCE++")
            else:
                print("Zero-DCE++ not loaded!")

cap.release()
cv2.destroyAllWindows()