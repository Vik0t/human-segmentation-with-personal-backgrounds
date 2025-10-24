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
DEVICE = 'mps' if torch.backends.mps.is_available() else 'cuda' if torch.cuda.is_available() else 'cpu'
MODEL_PATH = '/Users/vik0t/hackatons/human-segment/segmentation/RobustVideoMatting/rvm_mobilenetv3.pth'
BG_PATH = "segmentation/image.png"
ZERO_DCE_PATH = "models/zero_dce++.pth"
OUTPUT_WIDTH, OUTPUT_HEIGHT = 1280, 720
DOWNSAMPLE_RATIO = 0.25  # keep low for speed
ENHANCE_BRIGHTNESS_THRESHOLD = 60
ENHANCE_METHOD = 'none'  # 'zero_dce', 'clahe_gamma', 'none'
ENABLE_DEBUG = False
RVM_INPUT_SCALE = 0.5  # internal downsample for speed
ZERO_DCE_MAX_SIDE = 256  # very small for fast enhancement

print(f"Using device: {DEVICE}")

# ======================
# ZERO-DCE++ MODEL
# ======================
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

# ======================
# OTHER HELPERS
# ======================
def enhance_clahe_gamma(frame_bgr):
    gamma = 1.5
    inv_gamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in np.arange(256)]).astype("uint8")
    frame_bgr = cv2.LUT(frame_bgr, table)
    lab = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    l = clahe.apply(l)
    lab = cv2.merge((l,a,b))
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

def needs_enhancement(frame_bgr, threshold=ENHANCE_BRIGHTNESS_THRESHOLD):
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    return float(gray.mean()) < float(threshold)

def np_rgb_to_tensor(np_rgb, device):
    return torch.from_numpy(np_rgb).to(device, non_blocking=True).permute(2,0,1).float().div_(255.0).unsqueeze(0)

def load_background(bg_path, width, height, device):
    if not os.path.exists(bg_path):
        bg = Image.new("RGB", (width, height), (0,255,0))
    else:
        bg = Image.open(bg_path).convert("RGB")
    bg = bg.resize((width,height), Image.LANCZOS)
    return F_vision.to_tensor(bg).to(device).unsqueeze(0)

def smooth_alpha(alpha):
    # alpha: [1,1,H,W] tensor on device
    kernel = torch.tensor([[1,1,1],[1,1,1],[1,1,1]], device=alpha.device, dtype=alpha.dtype).unsqueeze(0).unsqueeze(0)/9.0
    return F_torch.conv2d(alpha, kernel, padding=1)

# ======================
# SETUP
# ======================
bg_tensor = load_background(BG_PATH, OUTPUT_WIDTH, OUTPUT_HEIGHT, DEVICE)
from RobustVideoMatting.model import MattingNetwork
model_rvm = MattingNetwork('mobilenetv3').eval().to(DEVICE)
state = torch.load(MODEL_PATH, map_location=DEVICE)
if any(k.startswith('module.') for k in state.keys()):
    state = {k.replace('module.',''):v for k,v in state.items()}
model_rvm.load_state_dict(state)

model_dce = None
if ENHANCE_METHOD == 'zero_dce':
    model_dce = load_zero_dce(ZERO_DCE_PATH, DEVICE)

rec = [None]*4
fps = 0.0
fps_alpha = 0.1

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_BUFFERSIZE,1)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, OUTPUT_WIDTH)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, OUTPUT_HEIGHT)
if not cap.isOpened():
    print("Error: cannot open webcam")
    exit()

prev_alpha = None

# ======================
# MAIN LOOP
# ======================
with torch.inference_mode():
    while True:
        t0 = time.time()
        ret, frame_bgr = cap.read()
        if not ret:
            break
        frame_bgr = cv2.resize(frame_bgr, (OUTPUT_WIDTH, OUTPUT_HEIGHT))
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

        # --- Enhancement ---
        matting_rgb = frame_rgb
        if needs_enhancement(frame_bgr) and ENHANCE_METHOD != 'none':
            if ENHANCE_METHOD == 'zero_dce' and model_dce:
                matting_rgb = enhance_zero_dce(frame_rgb, model_dce, DEVICE)
            elif ENHANCE_METHOD == 'clahe_gamma':
                matting_rgb = cv2.cvtColor(enhance_clahe_gamma(frame_bgr), cv2.COLOR_BGR2RGB)

        # --- RVM ---
        # downscale input for speed
        h_in, w_in = int(OUTPUT_HEIGHT*RVM_INPUT_SCALE), int(OUTPUT_WIDTH*RVM_INPUT_SCALE)
        small_rgb = cv2.resize(matting_rgb, (w_in, h_in), interpolation=cv2.INTER_AREA)
        src_tensor = np_rgb_to_tensor(small_rgb, DEVICE)
        _, pha, *rec = model_rvm(src_tensor, *rec, downsample_ratio=DOWNSAMPLE_RATIO)

        # --- Alpha upsample + simple smoothing ---
        pha_up = F_torch.interpolate(pha, size=(OUTPUT_HEIGHT, OUTPUT_WIDTH), mode='bilinear', align_corners=False)
        pha_up = smooth_alpha(pha_up)
        alpha = pha_up.clamp(0,1)

        # --- Composite ---
        orig_tensor = np_rgb_to_tensor(frame_rgb, DEVICE)
        com = orig_tensor * alpha + bg_tensor * (1-alpha)
        out_rgb = (com[0].permute(1,2,0).cpu().numpy()*255).astype(np.uint8)
        out_bgr = cv2.cvtColor(out_rgb, cv2.COLOR_RGB2BGR)

        # --- FPS overlay ---
        elapsed = time.time()-t0
        fps_cur = 1.0 / max(elapsed,1e-6)
        fps = fps_alpha*fps_cur + (1-fps_alpha)*fps
        tag = ENHANCE_METHOD.upper() if needs_enhancement(frame_bgr) else "NONE"
        cv2.putText(out_bgr, f"FPS:{fps:.1f} | Enh:{tag}", (10,30), cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2)

        cv2.imshow('Output', out_bgr)
        key = cv2.waitKey(1) & 0xFF
        if key==ord('q'): break
        elif key==ord('t'): ENABLE_DEBUG = not ENABLE_DEBUG
        elif key==ord('1'): ENHANCE_METHOD='none'
        elif key==ord('2'): ENHANCE_METHOD='clahe_gamma'
        elif key==ord('3'): ENHANCE_METHOD='zero_dce' if model_dce else 'none'

cap.release()
cv2.destroyAllWindows()
