import cv2
import torch
import numpy as np
from PIL import Image
from ultralytics import YOLO
from torchvision.transforms import functional as F
from RobustVideoMatting.model import MattingNetwork
import time

# ======================
# CONFIGURATION
# ======================
DEVICE = 'mps' if torch.backends.mps.is_available() else 'cuda' if torch.cuda.is_available() else 'cpu'
MODEL_PATH = '/Users/vik0t/hackatons/human-segment/module4/RobustVideoMatting/rvm_mobilenetv3.pth'
BG_PATH = "module4/image.png"
OUTPUT_WIDTH, OUTPUT_HEIGHT = 1280, 720  # Fixed output resolution for full frame
DOWNSAMPLE_RATIO = 0.25
ENHANCE_BRIGHTNESS_THRESHOLD = 60  # Only enhance if avg brightness < this

print(f"Using device: {DEVICE}")

# ======================
# LOW-LIGHT ENHANCEMENT HELPERS
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

def needs_enhancement(frame_bgr, threshold=ENHANCE_BRIGHTNESS_THRESHOLD):
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    return gray.mean() < threshold

def np_rgb_to_tensor(np_rgb, device):
    # np_rgb: (H, W, 3), uint8, RGB
    return torch.from_numpy(np_rgb).to(device, non_blocking=True).permute(2, 0, 1).float().div_(255.0).unsqueeze_(0)

# ======================
# MODELS SETUP
# ======================
# YOLO detector (for bounding boxes or segmentation masks)
detector = YOLO("yolov8n.pt")  # Make sure you have this model file

# Matting model
matting_model = MattingNetwork('mobilenetv3').eval().to(DEVICE)
matting_model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))

# For macOS MPS: set mixed precision etc.
if DEVICE == 'mps':
    torch.backends.mps.allow_tf32 = True
    torch.backends.mps.flush_on_exit = True
    torch.set_float32_matmul_precision('medium')

# Only compile when not on MPS
if hasattr(torch, 'compile') and DEVICE != 'mps':
    matting_model = torch.compile(matting_model)
    print("Matting model compiled with torch.compile()")
else:
    print("Skipping torch.compile() for MPS (Metal backend)")

# ======================
# BACKGROUND LOAD
# ======================
bg_pil = Image.open(BG_PATH).convert("RGB")
bg_bgr_full = cv2.cvtColor(np.array(bg_pil), cv2.COLOR_RGB2BGR)
bg_bgr_full = cv2.resize(bg_bgr_full, (OUTPUT_WIDTH, OUTPUT_HEIGHT))

# ======================
# VIDEO CAPTURE SETUP
# ======================
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

# ======================
# TRACKING & RECURRENCE STATE SETUP
# ======================
# Placeholder tracker logic: you must plug in a tracker (SORT / Norfair / any) to get stable IDs
# For now we will assign a dummy ID to each person (single-person case)
# If multi-person, use tracker.update(...) etc.

recs = {}         # track_id → [r1,r2,r3,r4]
last_shapes = {}  # track_id → (H, W)

# Dummy single‐person ID
DUMMY_ID = 0
recs[DUMMY_ID] = [None] * 4
last_shapes[DUMMY_ID] = None

# FPS measurement
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

        # Resize full frame to OUTPUT size for consistent output
        frame_bgr = cv2.resize(frame_bgr, (OUTPUT_WIDTH, OUTPUT_HEIGHT))
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

        # --- DETECTION ---
        results = detector(frame_rgb)
        det = results[0]

        # Extract bounding boxes for class “person” (class 0 in COCO)
        person_boxes = []
        if hasattr(det, 'boxes'):
            for box, cls in zip(det.boxes, det.boxes.cls):
                if int(cls.item()) == 0:
                    xyxy = box.xyxy[0].cpu().numpy().astype(int)
                    x1, y1, x2, y2 = xyxy
                    # Clip
                    x1, y1 = max(0, x1), max(0, y1)
                    x2, y2 = min(frame_bgr.shape[1], x2), min(frame_bgr.shape[0], y2)
                    person_boxes.append((x1, y1, x2, y2))

        if len(person_boxes) == 0:
            # No person detected — just show original frame
            cv2.putText(frame_bgr, f"FPS: {fps:.1f}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
            cv2.imshow('Output', frame_bgr)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            continue

        # For demo: process only first detected person
        x1, y1, x2, y2 = person_boxes[0]
        track_id = DUMMY_ID

        # Crop the region of interest
        crop_bgr = frame_bgr[y1:y2, x1:x2]
        if crop_bgr.size == 0:
            # Invalid crop, skip
            cv2.imshow('Output', frame_bgr)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            continue

        # Low-light enhancement only if needed
        if needs_enhancement(crop_bgr):
            crop_bgr = enhance_low_light(crop_bgr, method='both')

        # Prepare matting input (resize crop to fixed size or keep aspect)
        # Option: Keep original crop size → dynamic shape handling below
        crop_rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
        crop_resized_rgb = cv2.resize(crop_rgb, (OUTPUT_WIDTH, OUTPUT_HEIGHT))
        src_tensor = np_rgb_to_tensor(crop_resized_rgb, DEVICE)

        # Reset recurrence if shape changed for this track_id
        cur_shape = tuple(src_tensor.shape[-2:])  # (H, W)
        if last_shapes.get(track_id) != cur_shape:
            recs[track_id] = [None] * 4
            last_shapes[track_id] = cur_shape

        # Run matting
        _, pha, *new_rec = matting_model(src_tensor, *recs[track_id], downsample_ratio=DOWNSAMPLE_RATIO)

        # Save new recurrence
        recs[track_id] = new_rec

        # Convert alpha to numpy & resize to crop size
        pha_np = (pha.squeeze(0).squeeze(0).cpu().numpy())
        alpha_crop = cv2.resize(pha_np, (x2 - x1, y2 - y1))
        # Make 3-channel copy for blending
        alpha_3ch = np.repeat(alpha_crop[:, :, None], 3, axis=2)

        # Prepare background region
        bg_region = bg_bgr_full[y1:y2, x1:x2]

        # Composite (crop region)
        crop_out = (crop_bgr.astype(np.float32) * alpha_3ch +
                    bg_region.astype(np.float32) * (1.0 - alpha_3ch)).astype(np.uint8)

        # Place back into output frame
        output_bgr = frame_bgr.copy()
        output_bgr[y1:y2, x1:x2] = crop_out

        # Display FPS
        fps = fps_alpha * (1.0 / (time.time() - start_time)) + (1 - fps_alpha) * fps
        cv2.putText(output_bgr, f"FPS: {fps:.1f}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)

        cv2.imshow('Output', output_bgr)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# === CLEANUP ===
cap.release()
cv2.destroyAllWindows()
