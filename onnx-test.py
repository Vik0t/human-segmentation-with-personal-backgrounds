"""
RVM ONNX real-time demo optimized for Apple Silicon and CUDA (onnxruntime).
- Reads ONNX model, picks best EP (CUDA / CoreML / CPU).
- Keeps recurrent state for temporal consistency.
- Uses a capture thread + single-slot queue to minimize latency.
- Composites with a background image (or green if missing).
"""
ENABLE_DEBUG = False  # global debug flag
import os
import sys
import time
import queue
import threading
import platform
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

try:
    import onnxruntime as ort
except Exception as e:
    raise RuntimeError("onnxruntime is required. Install onnxruntime or onnxruntime-gpu (for CUDA).") from e

# ================
# CONFIG
# ================
MODEL_ONNX_PATH = "rvm_mobilenetv3_fp32.onnx"  # place your ONNX RVM model here
BG_PATH = "segmentation/image.png"
OUTPUT_WIDTH, OUTPUT_HEIGHT = 1280, 720
DOWNSAMPLE_RATIO = 0.25  # keep same as export-time / expected by model (if model expects)
ENHANCE_BRIGHTNESS_THRESHOLD = 60
ENABLE_DEBUG = False

# Camera
CAM_IDX = 0
CAM_WIDTH, CAM_HEIGHT = OUTPUT_WIDTH, OUTPUT_HEIGHT

# Capture queue (size 1 to drop stale frames)
FRAME_QUEUE = queue.Queue(maxsize=1)

# ================
# HELPERS
# ================
def choose_providers():
    avail = ort.get_available_providers()
    # prefer CUDA -> CoreML (mac) -> CPU
    providers = []
    if 'CUDAExecutionProvider' in avail:
        providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
    elif 'CoreMLExecutionProvider' in avail:
        providers = ['CoreMLExecutionProvider', 'CPUExecutionProvider']
    elif 'MetalExecutionProvider' in avail:
        providers = ['MetalExecutionProvider', 'CPUExecutionProvider']
    else:
        providers = ['CPUExecutionProvider']
    return providers

def enhance_low_light(frame_bgr):
    # gamma + CLAHE
    gamma = 1.5
    inv_gamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in np.arange(256)]).astype("uint8")
    frame = cv2.LUT(frame_bgr, table)
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    l = clahe.apply(l)
    lab = cv2.merge((l, a, b))
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

def needs_enhancement(frame_bgr, threshold=ENHANCE_BRIGHTNESS_THRESHOLD):
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    return float(gray.mean()) < threshold

def load_background(bg_path, width, height):
    if not os.path.exists(bg_path):
        # green screen fallback
        bg = Image.new("RGB", (width, height), (0, 255, 0))
    else:
        bg = Image.open(bg_path).convert("RGB")
    bg = bg.resize((width, height), Image.LANCZOS)
    bg_np = np.asarray(bg).astype(np.float32) / 255.0  # HWC RGB [0,1]
    return bg_np

def to_src_input_rgb(frame_rgb):
    """
    Convert HWC uint8 RGB to NCHW float32 [0,1]
    """
    # frame_rgb: H x W x 3 (uint8)
    x = frame_rgb.astype(np.float32) / 255.0
    # transpose to NCHW
    x = x.transpose(2, 0, 1)[None, ...].astype(np.float32)
    return x

def find_image_input_name(session):
    """
    Find an input that is likely the image tensor (4-D, NCHW).
    """
    for inp in session.get_inputs():
        shape = inp.shape  # may contain None
        # pick a 4-d input
        if len(shape) == 4:
            return inp.name
    # fallback to first input
    return session.get_inputs()[0].name

def find_downsample_input_name(session):
    # if model exported with downsample_ratio as input, find a scalar float input
    for inp in session.get_inputs():
        if inp.name.lower().find('down') >= 0 or inp.name.lower().find('ratio') >= 0:
            return inp.name
    # none found
    return None

def find_recurrent_input_names(session, exclude_img_name):
    # recurrent states typically have shapes like [1, C, H', W'] or similar and names r1..r4 or res*
    rec_names = []
    for inp in session.get_inputs():
        if inp.name == exclude_img_name:
            continue
        # heuristics: name contains 'r' or 'rec' or 'res' or has 4 dims (but not the image)
        if any(k in inp.name.lower() for k in ('r', 'rec', 'res', 'hidden', 'state')):
            rec_names.append(inp.name)
        else:
            # if 4-d and not img (for some exports), also consider as rec
            if len(inp.shape) == 4:
                rec_names.append(inp.name)
    return rec_names

def find_alpha_output_name(session):
    # alpha/pha output usually has 4-d shape with 1 channel in C
    for out in session.get_outputs():
        if any(k in out.name.lower() for k in ('pha', 'alpha', 'a_out', 'alpha_out')):
            return out.name
    # fallback: choose first output that has 4 dims and channel==1 or smallest channel
    candidate = None
    for out in session.get_outputs():
        shape = out.shape
        if len(shape) == 4:
            candidate = out.name
            break
    # if none, take first
    if candidate is None:
        candidate = session.get_outputs()[0].name
    return candidate

# ================
# CAPTURE THREAD
# ================
def capture_thread_fn(cap, queue_stop_event):
    while not queue_stop_event.is_set():
        ret, frame = cap.read()
        if not ret:
            time.sleep(0.01)
            continue
        # Try to push latest frame, drop previous if exists
        try:
            if FRAME_QUEUE.full():
                _ = FRAME_QUEUE.get_nowait()
        except queue.Empty:
            pass
        try:
            FRAME_QUEUE.put_nowait(frame)
        except queue.Full:
            pass

# ================
# MAIN
# ================
def main():
    global ENABLE_DEBUG
    print("ONNX Runtime version:", ort.__version__)
    if not os.path.exists(MODEL_ONNX_PATH):
        raise FileNotFoundError(f"ONNX model not found: {MODEL_ONNX_PATH}. Export one from the PyTorch repo first.")

    # Session options
    so = ort.SessionOptions()
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    # reasonable thread counts
    cpu_count = os.cpu_count() or 4
    so.intra_op_num_threads = max(1, cpu_count // 2)
    so.inter_op_num_threads = 1

    providers = choose_providers()
    print("Available ONNX execution providers:", ort.get_available_providers())
    print("Using providers (in order):", providers)

    # create session
    sess = ort.InferenceSession(MODEL_ONNX_PATH, sess_options=so, providers=providers)

    # Inspect inputs/outputs and pick names
    img_in_name = find_image_input_name(sess)
    down_in_name = find_downsample_input_name(sess)
    rec_in_names = find_recurrent_input_names(sess, img_in_name)
    alpha_out_name = find_alpha_output_name(sess)

    print("Detected model inputs:")
    for i in sess.get_inputs():
        print("  ", i.name, i.shape, i.type)
    print("Detected model outputs:")
    for o in sess.get_outputs():
        print("  ", o.name, o.shape, o.type)
    print(f"Image input => {img_in_name}")
    print(f"downsample input => {down_in_name}")
    print(f"recurrent inputs => {rec_in_names}")
    print(f"alpha output => {alpha_out_name}")

    # Initialize recurrent states to zeros based on input shapes (if present)
    recs = {}
    for rname in rec_in_names:
        inp = next((i for i in sess.get_inputs() if i.name == rname), None)
        shape = inp.shape  # may contain None
        # replace dynamic dims with concrete minimal dims (1)
        concrete_shape = [1 if (s is None or isinstance(s, str)) else int(s) for s in shape]
        recs[rname] = np.zeros(concrete_shape, dtype=np.float32)

    # Load background
    bg_np = load_background(BG_PATH, OUTPUT_WIDTH, OUTPUT_HEIGHT)  # HWC RGB [0,1]

    # Setup camera + thread
    cap = cv2.VideoCapture(CAM_IDX, cv2.CAP_ANY)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAM_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAM_HEIGHT)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    if not cap.isOpened():
        raise RuntimeError("Could not open camera")

    stop_event = threading.Event()
    t = threading.Thread(target=capture_thread_fn, args=(cap, stop_event), daemon=True)
    t.start()

    fps = 0.0
    fps_alpha = 0.1

    try:
        print("Starting inference loop. Press 'q' to quit, 't' to toggle debug, 'b' to reload bg.")
        while True:
            t0 = time.time()
            try:
                frame_bgr = FRAME_QUEUE.get(timeout=1.0)  # BGR
            except queue.Empty:
                # no frame, continue
                continue

            # Resize early (keeps consistent dims)
            frame_bgr = cv2.resize(frame_bgr, (OUTPUT_WIDTH, OUTPUT_HEIGHT))
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

            should_enhance = needs_enhancement(frame_bgr)
            matting_input = enhance_low_light(frame_bgr) if should_enhance else frame_bgr
            if should_enhance:
                matting_input = cv2.cvtColor(matting_input, cv2.COLOR_BGR2RGB)

            # Prepare model inputs
            src_np = to_src_input_rgb(matting_input)  # NCHW float32
            ort_inputs = {img_in_name: src_np}

            # add recurrent states (reuse previous)
            for rname, rval in recs.items():
                if rname not in ['fgr', 'r1o', 'r2o', 'r3o', 'r4o']:
                    ort_inputs[rname] = rval

            # add downsample_ratio if model expects it
            if down_in_name is not None:
                # some exports expect float32 or int scalar
                ort_inputs[down_in_name] = np.array([DOWNSAMPLE_RATIO], dtype=np.float32)

            # Run ONNX session
            out = sess.run(None, ort_inputs)
            # Determine outputs mapping: find alpha output by name index
            out_map = {o.name: out[idx] for idx, o in enumerate(sess.get_outputs())}
            pha = out_map.get(alpha_out_name)  # shape N x 1 x h x w

            # Update recurrent states: replace recs with outputs that match rec names
            # Heuristic: output names that look like rec names or known patterns
            for o in sess.get_outputs():
                oname = o.name
                if any(k in oname.lower() for k in ('r', 'rec', 'res', 'hidden', 'state')):
                    recs.setdefault(oname, out_map[oname])
                # some exports may simply have names matching input recurrent names+suffix; update if shapes match
                if oname in recs:
                    recs[oname] = out_map[oname]

            # Fallback: if outputs include items with same shape as rec inputs, try to map them by order
            # (only if recs still zeros)
            if any(np.all(recs[rn] == 0.0) for rn in recs) and len(recs) > 0:
                # match by shapes
                for out_name, arr in out_map.items():
                    for in_name, in_arr in list(recs.items()):
                        if arr.shape == in_arr.shape:
                            recs[in_name] = arr

            # Composite
            # pha: N x 1 x ph x pw
            pha_np = pha.astype(np.float32)
            # resize alpha to output dim HxW
            alpha = pha_np[0, 0]
            alpha_up = cv2.resize(alpha, (OUTPUT_WIDTH, OUTPUT_HEIGHT), interpolation=cv2.INTER_LINEAR)
            alpha_up = np.clip(alpha_up[..., None], 0.0, 1.0)  # H x W x 1

            # orig float rgb [0,1]
            orig_rgb = frame_rgb.astype(np.float32) / 255.0
            comp = orig_rgb * alpha_up + bg_np * (1.0 - alpha_up)
            # convert to BGR uint8 for display
            out_bgr = cv2.cvtColor((comp * 255.0).astype(np.uint8), cv2.COLOR_RGB2BGR)

            # debug overlay
            if ENABLE_DEBUG:
                # show alpha as grayscale overlay
                alpha_vis = (alpha_up[..., 0] * 255).astype(np.uint8)
                alpha_color = cv2.cvtColor(alpha_vis, cv2.COLOR_GRAY2BGR)
                top = np.hstack((cv2.resize(frame_bgr, (640, 360)), cv2.resize(cv2.cvtColor(matting_input if not should_enhance else cv2.cvtColor(matting_input, cv2.COLOR_BGR2RGB), cv2.COLOR_RGB2BGR), (640, 360))))
                bot = np.hstack((cv2.resize(alpha_color, (640, 360)), cv2.resize(out_bgr, (640, 360))))
                grid = np.vstack((top, bot))
                cv2.putText(grid, f"FPS: {fps:.1f} | DEBUG", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)
                cv2.imshow("Debug Grid", grid)
                try:
                    cv2.destroyWindow("Output")
                except:
                    pass
            else:
                mode_text = "Enhanced Matting" if should_enhance else "Direct Matting"
                cv2.putText(out_bgr, f"FPS: {fps:.1f} | {mode_text}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
                cv2.imshow("Output", out_bgr)
                try:
                    cv2.destroyWindow("Debug Grid")
                except:
                    pass

            # FPS smoothing
            elapsed = time.time() - t0
            current_fps = 1.0 / max(elapsed, 1e-6)
            fps = fps_alpha * current_fps + (1 - fps_alpha) * fps

            k = cv2.waitKey(1) & 0xFF
            if k == ord('q'):
                break
            elif k == ord('t'):
               
                ENABLE_DEBUG = not ENABLE_DEBUG
                print("Debug =", ENABLE_DEBUG)
            elif k == ord('b'):
                bg_np = load_background(BG_PATH, OUTPUT_WIDTH, OUTPUT_HEIGHT)
                print("Background reloaded.")

    finally:
        stop_event.set()
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
