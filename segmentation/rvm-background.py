import cv2
import torch
from PIL import Image
from torchvision.transforms import functional as F
from RobustVideoMatting.model import MattingNetwork
import numpy as np
import time
import os

# ======================
# CONFIGURATION
# ======================
DEVICE = 'mps' if torch.mps.is_available() else 'cuda' if torch.cuda.is_available() else 'cpu'
MODEL_PATH = f'{os.path.dirname(os.path.abspath(__file__))}/RobustVideoMatting/rvm_mobilenetv3.pth'
BG_PATH = "segmentation/image.png"
OUTPUT_WIDTH, OUTPUT_HEIGHT = 1280, 720
DOWNSAMPLE_RATIO = 0.25
ENHANCE_BRIGHTNESS_THRESHOLD = 60
ENABLE_DEBUG = False  # Toggle this or use 't' key

print(f"Using device: {DEVICE}")

# ======================
# HELPERS
# ======================
def enhance_low_light(frame_bgr):
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
        print(f"Warning: Background {bg_path} not found. Using green screen.")
        bg_pil = Image.new("RGB", (width, height), (0, 255, 0))
    else:
        bg_pil = Image.open(bg_path).convert("RGB")
    bg_resized = bg_pil.resize((width, height), Image.LANCZOS)
    return F.to_tensor(bg_resized).to(device, non_blocking=True).unsqueeze(0)

# ======================
# SETUP
# ======================
bg_tensor = load_background(BG_PATH, OUTPUT_WIDTH, OUTPUT_HEIGHT, DEVICE)

model = MattingNetwork('mobilenetv3').eval().to(DEVICE)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))

if DEVICE == 'cuda' and hasattr(torch, 'compile'):
    model = torch.compile(model, mode="reduce-overhead")
    print("Model compiled with torch.compile() (CUDA only)")

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

# Single recurrent state for temporal consistency
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

        # Resize early
        frame_bgr = cv2.resize(frame_bgr, (OUTPUT_WIDTH, OUTPUT_HEIGHT))
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

        # Decide enhancement
        should_enhance = needs_enhancement(frame_bgr, ENHANCE_BRIGHTNESS_THRESHOLD)
        matting_input_rgb = enhance_low_light(frame_bgr) if should_enhance else frame_rgb
        if should_enhance:
            matting_input_rgb = cv2.cvtColor(matting_input_rgb, cv2.COLOR_BGR2RGB)

        # Run matting ONCE
        src_tensor = np_rgb_to_tensor(matting_input_rgb, DEVICE)
        _, pha, *rec = model(src_tensor, *rec, downsample_ratio=DOWNSAMPLE_RATIO)

        # Composite with ORIGINAL RGB
        orig_tensor = np_rgb_to_tensor(frame_rgb, DEVICE)
        pha_up = torch.nn.functional.interpolate(
            pha, size=(OUTPUT_HEIGHT, OUTPUT_WIDTH), mode='bilinear', align_corners=False
        )
        com = orig_tensor * pha_up + bg_tensor * (1 - pha_up)
        output_rgb = (com.squeeze(0).permute(1, 2, 0) * 255).byte().cpu().numpy()
        output_bgr = cv2.cvtColor(output_rgb, cv2.COLOR_RGB2BGR)

        # --- DEBUG MODE (only if enabled) ---
        if ENABLE_DEBUG:
            # Re-run matting on ORIGINAL input for comparison
            src_orig_tensor = np_rgb_to_tensor(frame_rgb, DEVICE)
            rec_debug = [r.clone() if r is not None else None for r in rec]  # preserve main state
            _, pha_orig, *_ = model(src_orig_tensor, *rec_debug, downsample_ratio=DOWNSAMPLE_RATIO)

            # Build debug grid
            grid_h, grid_w = 360, 640
            orig_small = cv2.resize(frame_bgr, (grid_w, grid_h))
            enh_small = cv2.resize(
                cv2.cvtColor(matting_input_rgb, cv2.COLOR_RGB2BGR) if should_enhance else frame_bgr,
                (grid_w, grid_h)
            )

            alpha_orig_disp = cv2.resize((pha_orig.squeeze().cpu().numpy() * 255).astype(np.uint8), (grid_w, grid_h))
            alpha_enh_disp = cv2.resize((pha.squeeze().cpu().numpy() * 255).astype(np.uint8), (grid_w, grid_h))

            top_row = np.hstack((orig_small, enh_small))
            bottom_row = np.hstack((cv2.cvtColor(alpha_orig_disp, cv2.COLOR_GRAY2BGR),
                                    cv2.cvtColor(alpha_enh_disp, cv2.COLOR_GRAY2BGR)))
            grid = np.vstack((top_row, bottom_row))

            cv2.putText(grid, f"FPS: {fps:.1f} | DEBUG MODE", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
            cv2.imshow('Debug Grid', grid)
            try: # this is to avoid errors when the window is not open
                cv2.destroyWindow('Output')
            except:
                pass
        else:
            mode_text = "Enhanced Matting" if should_enhance else "Direct Matting"
            cv2.putText(output_bgr, f"FPS: {fps:.1f} | {mode_text}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.imshow('Output', output_bgr)
            try:
                cv2.destroyWindow('Debug Grid')
            except:
                pass

        # --- FPS & Controls ---
        elapsed = time.time() - start_time
        current_fps = 1.0 / max(elapsed, 1e-6)
        fps = fps_alpha * current_fps + (1 - fps_alpha) * fps

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('t'):
            ENABLE_DEBUG = not ENABLE_DEBUG
            print(f"Debug mode: {'ON' if ENABLE_DEBUG else 'OFF'}")
            time.sleep(0.2)  # debounce
        elif key == ord('b'):
            try:
                bg_tensor = load_background(BG_PATH, OUTPUT_WIDTH, OUTPUT_HEIGHT, DEVICE)
                print("Background reloaded.")
            except Exception as e:
                print(f"Failed to reload background: {e}")

cap.release()
cv2.destroyAllWindows()