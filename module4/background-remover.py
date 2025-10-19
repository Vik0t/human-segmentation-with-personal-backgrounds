import argparse
import cv2
import mediapipe as mp
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import time
import os
import threading
import queue
import coremltools
import PIL.Image

# Simple config
DEFAULT_TEMPLATE = None  # path to default template image if any
FONT_PATH = None  # if you have a TTF font, put path here to render names nicer


def enhance_low_light(frame):
    """Enhance low-light using CLAHE on the luminance channel."""
    img_y_cr_cb = cv2.cvtColor(frame, cv2.COLOR_BGR2YCrCb)
    y, cr, cb = cv2.split(img_y_cr_cb)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    y = clahe.apply(y)
    img_y_cr_cb = cv2.merge((y, cr, cb))
    enhanced = cv2.cvtColor(img_y_cr_cb, cv2.COLOR_YCrCb2BGR)
    return frame


def ensure_contrast_color(bg_region):
    """Return 'black' or 'white' to contrast with the bg_region average luminance."""
    # bg_region: BGR numpy region
    if bg_region.size == 0:
        return (255, 255, 255)
    b, g, r = cv2.split(bg_region)
    lum = 0.2126 * r.mean() + 0.7152 * g.mean() + 0.0722 * b.mean()
    return (0, 0, 0) if lum > 128 else (255, 255, 255)


def render_text_on_template(template_bgr, text, position):
    """Render text (name/role) on template image using PIL for nicer fonts."""
    img_pil = Image.fromarray(cv2.cvtColor(template_bgr, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img_pil)
    try:
        font = ImageFont.truetype(FONT_PATH or "/Library/Fonts/Arial.ttf", size=28)
    except Exception:
        font = ImageFont.load_default()
    x, y = position
    h, w = template_bgr.shape[:2]
    sample_x1 = int(max(0, x - 50))
    sample_y1 = int(max(0, y - 30))
    sample_x2 = int(min(w, x + 50))
    sample_y2 = int(min(h, y + 30))
    bg_region = template_bgr[sample_y1:sample_y2, sample_x1:sample_x2]
    color = ensure_contrast_color(bg_region)
    draw.text((x, y), text, fill=tuple(int(c) for c in color[::-1]), font=font)  # PIL uses RGB
    return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)


def apply_template(frame_bgr, mask, template_bgr, position_text=None, text=None):
    """Compose person (frame) over template using mask (float 0..1)."""
    # Resize template to frame size if needed
    h, w = frame_bgr.shape[:2]
    th, tw = template_bgr.shape[:2]
    if (th, tw) != (h, w):
        template_bgr = cv2.resize(template_bgr, (w, h), interpolation=cv2.INTER_AREA)

    # Smooth mask and make 3 channels
    # Use a smaller kernel for speed on Apple Silicon while keeping quality
    mask_blur = cv2.GaussianBlur((mask * 255).astype(np.uint8), (11, 11), 0)
    alpha = (mask_blur.astype(np.float32) / 255.0)[:, :, None]

    # Composite: person over template
    composed = (frame_bgr.astype(np.float32) * alpha + template_bgr.astype(np.float32) * (1 - alpha)).astype(np.uint8)

    # Add text if provided
    if text and position_text:
        composed = render_text_on_template(composed, text, position_text)
    return composed


def blur_background(frame, mask, ksize=41):
    """Blur background outside the person mask for privacy or depth effect."""
    # Reduce blur kernel by default to improve performance; keep parameter
    k = ksize if ksize % 2 == 1 else ksize + 1
    blurred = cv2.GaussianBlur(frame, (k, k), 0)
    alpha = mask[:, :, None]
    out = (frame * alpha + blurred * (1 - alpha)).astype(np.uint8)
    return out


def anonymize_faces(frame, face_detector, blur_kernel=(99, 99)):
    """Detect faces and blur them (privacy). face_detector is a mediapipe face detection object."""
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_detector.process(rgb)
    if not results.detections:
        return frame
    out = frame.copy()
    h, w = frame.shape[:2]
    for det in results.detections:
        bbox = det.location_data.relative_bounding_box
        x1 = int(max(0, bbox.xmin * w))
        y1 = int(max(0, bbox.ymin * h))
        x2 = int(min(w, (bbox.xmin + bbox.width) * w))
        y2 = int(min(h, (bbox.ymin + bbox.height) * h))
        if x2 - x1 > 0 and y2 - y1 > 0:
            roi = out[y1:y2, x1:x2]
            roi_blur = cv2.GaussianBlur(roi, blur_kernel, 0)
            out[y1:y2, x1:x2] = roi_blur
    return out


def run_video_segmentation(template_path=None, name_text=None, blur_faces=False, cam_index=0, fast=False, no_enhance=False, process_scale=None):
    # дефолтные оптимизации для apple silicon
    PROCESS_SCALE = 0.5 if process_scale is None else float(process_scale)
    MAX_FPS = None
    # toggle быстрого режима
    SEGMENT_SKIP = 1
    if fast:
        PROCESS_SCALE = min(PROCESS_SCALE, 0.4)
        SEGMENT_SKIP = 2
    mp_selfie = mp.solutions.selfie_segmentation
    mp_face = mp.solutions.face_detection
    selfie = mp_selfie.SelfieSegmentation(model_selection=1)  # 0 or 1; 1 is generally better for full-body
    face_det = mp_face.FaceDetection(min_detection_confidence=0.6) if blur_faces else None

    # считыаем в потоке, чтобы не блокировать основной цикл
    class ThreadedCapture:
        def __init__(self, src=0, queue_size=4):
            self.cap = cv2.VideoCapture(src)
            self.q = queue.Queue(maxsize=queue_size)
            self.stopped = False
            self.thread = threading.Thread(target=self._reader, daemon=True)
            if not self.cap.isOpened():
                raise RuntimeError("Cannot open camera")
            self.thread.start()

        def _reader(self):
            while not self.stopped:
                ret, frame = self.cap.read()
                if not ret:
                    self.stopped = True
                    return
                try:
                    if self.q.full():
                        try:
                            self.q.get_nowait()
                        except Exception:
                            pass
                    self.q.put_nowait(frame)
                except Exception:
                    pass

        def read(self, timeout=0.01):
            try:
                return True, self.q.get(timeout=timeout)
            except Exception:
                return False, None

        def release(self):
            self.stopped = True
            try:
                self.thread.join(timeout=0.2)
            except Exception:
                pass
            self.cap.release()

    try:
        cap = ThreadedCapture(cam_index)
    except RuntimeError as e:
        print(e)
        return

    template_bgr = None
    if template_path and os.path.exists(template_path):
        template_bgr = cv2.imread(template_path)
    else:
        # загружаем дефолтный бэкграунд если картинки нет
        _, sample = cap.read()
        if sample is None:
            sample = np.full((480, 640, 3), 180, dtype=np.uint8)
        template_bgr = cv2.GaussianBlur(sample, (101, 101), 0)

    # для сглаживания маски между кадрами
    prev_mask = None
    smooth_alpha = 0.6
  

    cached_template = None
    cached_template_size = (0, 0)

    try:
        frame_skip = 0
        frame_idx = 0
        do_enhance = not no_enhance and (not fast)
        while True:
            ret, frame = cap.read()
            if not ret or frame is None:
                time.sleep(0.005)
                continue
            start = time.time()
            # опциональная оптимизация для слабого света
            if do_enhance:
                frame_enh = enhance_low_light(frame)
            else:
                frame_enh = frame

            # даунскейлим сегментацию для скорости
            if PROCESS_SCALE != 1.0:
                seg_h = int(frame_enh.shape[0] * PROCESS_SCALE)
                seg_w = int(frame_enh.shape[1] * PROCESS_SCALE)
                seg_frame = cv2.resize(frame_enh, (seg_w, seg_h), interpolation=cv2.INTER_AREA)
            else:
                seg_frame = frame_enh

            # запускаем сегментацию не на каждом кадре а на каждом SEGMENT_SKIP
            results = None
            if frame_idx % SEGMENT_SKIP == 0:
                rgb = cv2.cvtColor(seg_frame, cv2.COLOR_BGR2RGB)
                results = selfie.process(rgb)
            frame_idx += 1
            # используем предыдущую маску если пропустили сегментацию
            if results is None:
                mask = prev_mask if prev_mask is not None else np.zeros((frame.shape[0], frame.shape[1]), dtype=np.float32)
            else:
                rgb = cv2.cvtColor(seg_frame, cv2.COLOR_BGR2RGB)
                results = selfie.process(rgb)
            if results.segmentation_mask is None:
                mask = np.zeros((frame.shape[0], frame.shape[1]), dtype=np.float32)
            else:
                # ресайзим маску сегментации обратно к размеру кадра
                mask = cv2.resize(results.segmentation_mask, (frame.shape[1], frame.shape[0]))
                # refine mask: threshold + morphological ops
                mask = cv2.GaussianBlur(mask, (11, 11), 0)
                _, mask = cv2.threshold((mask * 255).astype(np.uint8), 127, 255, cv2.THRESH_BINARY)
                mask = mask.astype(np.float32) / 255.0
                kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
                mask = cv2.morphologyEx((mask * 255).astype(np.uint8), cv2.MORPH_CLOSE, kernel).astype(np.float32) / 255.0

            # временное сглаживание маски
            if prev_mask is None:
                prev_mask = mask
            mask = smooth_alpha * prev_mask + (1 - smooth_alpha) * mask
            prev_mask = mask

            # блюр для красоты
            bg_blur = blur_background(frame_enh, mask, ksize=31)

            # делаем композит с шаблоном
            if template_bgr is not None:
                if cached_template is None or cached_template_size != (frame_enh.shape[1], frame_enh.shape[0]):
                    cached_template = cv2.resize(template_bgr, (frame_enh.shape[1], frame_enh.shape[0]), interpolation=cv2.INTER_AREA)
                    cached_template_size = (frame_enh.shape[1], frame_enh.shape[0])
            else:
                cached_template = None

            use_template = cached_template if cached_template is not None else template_bgr
            composed = apply_template(frame_enh, mask, use_template, position_text=(20, 20), text=name_text)

            # блюр лица для анонимности (как вариант)
            if blur_faces and face_det is not None:
                composed = anonymize_faces(composed, face_det, blur_kernel=(99, 99))

            fps = 1.0 / (time.time() - start + 1e-6)
            cv2.putText(composed, f"FPS: {int(fps)}", (10, composed.shape[0] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            cv2.imshow("Personalized Background", composed)
            key = cv2.waitKey(1) & 0xFF
            if key == 27 or key == ord('q'):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()
        selfie.close()
        if face_det:
            face_det.close()


def parse_args():
    parser = argparse.ArgumentParser(description="локальная сегментация человека с заменой фона")
    parser.add_argument("--template", type=str, help="путь к изображению шаблона фона", default=DEFAULT_TEMPLATE)
    parser.add_argument("--name", type=str, help="название", default=None)
    parser.add_argument("--blur-faces", action="store_true", help="блюрить лица для приватности")
    parser.add_argument("--camera", type=int, default=0, help="номер камеры (по умолчанию 0)")
    parser.add_argument("--fast", action="store_true", help="использовать быстрый режим (меньше деталей, но быстрее)")
    parser.add_argument("--no-enhance", action="store_true", help="выключить улучшение для слабого света (ускоряет работу)")
    parser.add_argument("--process-scale", type=float, default=None, help="кастомный масштаб обработки (по умолчанию 0.5 или 0.4 в fast режиме)")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_video_segmentation(template_path=args.template, name_text=args.name, blur_faces=args.blur_faces,
                           cam_index=args.camera, fast=args.fast, no_enhance=args.no_enhance, process_scale=args.process_scale)