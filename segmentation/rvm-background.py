import cv2
import torch
from PIL import Image
from torchvision.transforms import functional as F
from RobustVideoMatting.model import MattingNetwork
import numpy as np
import time

# ======================
# CONFIGURATION
# ======================
DEVICE = 'mps' if torch.mps.is_available() else 'cuda' if torch.cuda.is_available() else 'cpu'
MODEL_PATH = '/Users/vik0t/hackatons/human-segment/module4/RobustVideoMatting/rvm_mobilenetv3.pth'
BG_PATH = "module4/image.png"
OUTPUT_WIDTH, OUTPUT_HEIGHT = 1280, 720  # Fixed output resolution
DOWNSAMPLE_RATIO = 0.25
ENHANCE_BRIGHTNESS_THRESHOLD = 60  # Only enhance if avg brightness < this

print(f"Using device: {DEVICE}")

# ======================
# HELPERS
# ======================
def enhance_low_light(frame_bgr, method='both'):
    if method == 'clahe':
        lab = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        l = clahe.apply(l)
        lab = cv2.merge((l, a, b))
        return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    elif method == 'gamma':
        gamma = 1.5
        inv_gamma = 1.0 / gamma
        table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
        return cv2.LUT(frame_bgr, table)
    elif method == 'both':
        frame_bgr = enhance_low_light(frame_bgr, 'gamma')
        frame_bgr = enhance_low_light(frame_bgr, 'clahe')
    return frame_bgr

def needs_enhancement(frame_bgr, threshold=60):
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    return gray.mean() < threshold

def np_rgb_to_tensor(np_rgb, device):
    # np_rgb: (H, W, 3), uint8, RGB
    return torch.from_numpy(np_rgb).to(device, non_blocking=True).permute(2, 0, 1).float().div_(255.0).unsqueeze_(0)

# ======================
# SETUP
# ======================
# Load background (fixed size)
bg_pil = Image.open(BG_PATH).convert("RGB")
bg_resized = bg_pil.resize((OUTPUT_WIDTH, OUTPUT_HEIGHT), Image.LANCZOS)
bg_tensor = F.to_tensor(bg_resized).to(DEVICE).unsqueeze(0)  # (1, 3, H, W)

# Load model
model = MattingNetwork('mobilenetv3').eval().to(DEVICE)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))

# Use torch.compile if available (PyTorch >= 2.0)
if hasattr(torch, 'compile'):
    model = torch.compile(model)
    print("Model compiled with torch.compile()")

# Camera setup
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Reduce latency
if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

# Recurrent states
rec_orig = [None] * 4
rec_enh = [None] * 4

# Mode & FPS
view_mode = 0  # 0 = final, 1 = debug
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

        # --- Preprocessing ---
        # Convert to RGB once
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        orig_resized_rgb = cv2.resize(frame_rgb, (OUTPUT_WIDTH, OUTPUT_HEIGHT))

        # Always define enhanced_bgr for display
        if True:
            enhanced_bgr = enhance_low_light(frame_bgr, method='both')
        else:
            enhanced_bgr = frame_bgr.copy()  # ← critical: always assign
        
        # Then convert for model
        enhanced_rgb = cv2.cvtColor(enhanced_bgr, cv2.COLOR_BGR2RGB)
        enh_resized_rgb = cv2.resize(enhanced_rgb, (OUTPUT_WIDTH, OUTPUT_HEIGHT))

        # Convert to tensors
        src_orig_tensor = np_rgb_to_tensor(orig_resized_rgb, DEVICE)
        src_enh_tensor = np_rgb_to_tensor(enh_resized_rgb, DEVICE)

        # --- Matting ---
        if view_mode == 0:
            # Only run enhanced matting for final output
            _, pha_enh, *rec_enh = model(src_enh_tensor, *rec_enh, downsample_ratio=DOWNSAMPLE_RATIO)

            # Composite: original RGB + enhanced alpha + background
            orig_tensor = np_rgb_to_tensor(orig_resized_rgb, DEVICE)
            pha_up = torch.nn.functional.interpolate(pha_enh, size=(OUTPUT_HEIGHT, OUTPUT_WIDTH), mode='bilinear', align_corners=False)
            com = orig_tensor * pha_up + bg_tensor * (1 - pha_up)

            output_rgb = (com.squeeze(0).permute(1, 2, 0) * 255).byte().cpu().numpy()
            output_bgr = cv2.cvtColor(output_rgb, cv2.COLOR_RGB2BGR)

            cv2.putText(output_bgr, f"FPS: {fps:.1f} | Mode: Final", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.imshow('Output', output_bgr)
            cv2.destroyWindow('Debug Grid')

        else:
            # Run both for debug
            _, pha_orig, *rec_orig = model(src_orig_tensor, *rec_orig, downsample_ratio=DOWNSAMPLE_RATIO)
            _, pha_enh, *rec_enh = model(src_enh_tensor, *rec_enh, downsample_ratio=DOWNSAMPLE_RATIO)

            # Debug grid
            grid_h, grid_w = 360, 640
            orig_small = cv2.resize(frame_bgr, (grid_w, grid_h))
            enh_small = cv2.resize(enhanced_bgr if 'enhanced_bgr' in locals() else frame_bgr, (grid_w, grid_h))

            alpha_orig_disp = (pha_orig.squeeze().cpu().numpy() * 255).astype(np.uint8)
            alpha_enh_disp = (pha_enh.squeeze().cpu().numpy() * 255).astype(np.uint8)
            alpha_orig_small = cv2.resize(alpha_orig_disp, (grid_w, grid_h))
            alpha_enh_small = cv2.resize(alpha_enh_disp, (grid_w, grid_h))

            alpha_orig_color = cv2.cvtColor(alpha_orig_small, cv2.COLOR_GRAY2BGR)
            alpha_enh_color = cv2.cvtColor(alpha_enh_small, cv2.COLOR_GRAY2BGR)

            top_row = np.hstack((orig_small, enh_small))
            bottom_row = np.hstack((alpha_orig_color, alpha_enh_color))
            grid = np.vstack((top_row, bottom_row))

            cv2.putText(grid, f"FPS: {fps:.1f} | Mode: Debug Grid", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
            cv2.putText(grid, "Original", (10, grid_h + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(grid, "Enhanced", (grid_w + 10, grid_h + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            cv2.imshow('Debug Grid', grid)
            cv2.destroyWindow('Output')

        # --- FPS & Controls ---
        elapsed = time.time() - start_time
        current_fps = 1.0 / elapsed if elapsed > 0 else 0
        fps = fps_alpha * current_fps + (1 - fps_alpha) * fps

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('t'):
            view_mode = 1 - view_mode
            print(f"Toggled to: {'Final Output' if view_mode == 0 else '2x2 Debug Grid'}")
            time.sleep(0.2)  # debounce
        elif key == ord('b'):
            try:
                bg_pil = Image.open(BG_PATH).convert("RGB")
                bg_resized = bg_pil.resize((OUTPUT_WIDTH, OUTPUT_HEIGHT), Image.LANCZOS)
                bg_tensor = F.to_tensor(bg_resized).to(DEVICE).unsqueeze(0)
                print("Background reloaded.")
            except Exception as e:
                print(f"Failed to reload background: {e}")

# ======================
# CLEANUP
# ======================
cap.release()
cv2.destroyAllWindows()