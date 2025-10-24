import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F_torch
from PIL import Image, ImageDraw, ImageFont
from torchvision.transforms import functional as F_vision
import numpy as np
import time
import os
import base64

from employee_data_module import employee_data_module, PrivacyLevel

# ======================
# CONFIGURATION
# ======================
DEVICE = 'mps' if torch.mps.is_available() else 'cuda' if torch.cuda.is_available() else 'cpu'
MODEL_PATH = '/Users/shilov_n_jr/PycharmProjects/human-segmentation-with-personal-backgrounds/rvm_mobilenetv3.pth'
BG_PATH = "/Users/shilov_n_jr/PycharmProjects/human-segmentation-with-personal-backgrounds/1920х1080.png"
ZERO_DCE_PATH = "models/zero_dce++.pth"
OUTPUT_WIDTH, OUTPUT_HEIGHT = 1280, 720
DOWNSAMPLE_RATIO = 0.25
ENHANCE_BRIGHTNESS_THRESHOLD = 60
ENHANCE_METHOD = 'zero_dce'  # 'zero_dce', 'clahe_gamma', 'none'
ENABLE_DEBUG = False
CYRILLIC_FONT = None

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


def load_employee_data():
    """
    Функция загрузки данных сотрудника
    В реальной реализации здесь может быть загрузка из файла, API и т.д.
    Сейчас используется заглушка с фиксированными данными
    """
    # Заглушка с данными сотрудника
    employee_json = {
        "employee": {
            "full_name": "Петрова Анна Владимировна",
            "position": "Руководитель проекта",
            "company": "ООО «Рога и Копыта»",
            "department": "Департамент компьютерного зрения",
            "office_location": "Новосибирск, техноларк «Идея»",
            "contact": {
                "email": "anna.petrova@company.ru",
                "telegram": "@anna_teamlead"
            },
            "branding": {
                "logo_url": "https://example.com/logo.png",
                "corporate_colors": {
                    "primary": "#0052CC",
                    "secondary": "#0088D9"
                },
                "slogan": "Инновации в каждый кадр"
            },
            "privacy_level": "high"
        }
    }

    try:
        # Загружаем данные в модуль
        employee_data_module.load_employee_data(employee_json)

        # Устанавливаем высокий уровень конфиденциальности для генерации QR-кодов
        employee_data_module.set_privacy_level(PrivacyLevel.HIGH)

        print("Данные сотрудника успешно загружены")

        # Тестируем генерацию данных для отображения
        rendering_data = employee_data_module.get_data_for_rendering()
        print("QR-коды сгенерированы:", 'qr_codes' in rendering_data)

        return True

    except Exception as e:
        print(f"Ошибка загрузки данных сотрудника: {e}")
        return False

# ===================
# Вспомогательные функции для добавления оверлеев
# ===================

def init_cyrillic_font():
    """Инициализирует шрифт один раз при первом вызове"""
    global CYRILLIC_FONT
    if CYRILLIC_FONT is not None:
        return CYRILLIC_FONT

    font_paths = [
        "C:/Windows/Fonts/arial.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/System/Library/Fonts/Arial.ttf",
        "arial.ttf",
        "DejaVuSans.ttf"
    ]

    for font_path in font_paths:
        try:
            CYRILLIC_FONT = ImageFont.truetype(font_path, 16)
            print(f"Успешно загружен шрифт: {font_path}")
            return CYRILLIC_FONT
        except:
            continue

    try:
        CYRILLIC_FONT = ImageFont.load_default()
        print("Используется стандартный шрифт")
        return CYRILLIC_FONT
    except:
        print("Не удалось загрузить шрифт")
        return None

def add_text_overlay_pil(image, employee_data, position='bottom_left'):
    """Добавляет текстовую информацию на изображение используя PIL"""
    # Конвертируем OpenCV image (BGR) в PIL Image (RGB)
    pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil_image)

    height, width = image.shape[:2]

    # Определяем позицию для текста
    if position == 'bottom_left':
        x_start = 20
        y_start = height - 200  # Увеличиваем отступ
    else:
        x_start = width - 350
        y_start = height - 200

    # Собираем текстовые строки
    text_lines = []

    if 'full_name' in employee_data:
        text_lines.append(f"Имя: {employee_data['full_name']}")
    if 'position' in employee_data:
        text_lines.append(f"Должность: {employee_data['position']}")
    if 'company' in employee_data:
        text_lines.append(f"Компания: {employee_data['company']}")
    if 'department' in employee_data:
        text_lines.append(f"Отдел: {employee_data['department']}")
    if 'office_location' in employee_data:
        text_lines.append(f"Локация: {employee_data['office_location']}")
    if 'contact' in employee_data and 'email' in employee_data['contact']:
        text_lines.append(f"Email: {employee_data['contact']['email']}")
    if 'contact' in employee_data and 'telegram' in employee_data['contact']:
        text_lines.append(f"Telegram: {employee_data['contact']['telegram']}")

    font = init_cyrillic_font() if CYRILLIC_FONT is None else CYRILLIC_FONT

    if font is None:
        # Если не нашли шрифты, пробуем загрузить стандартный
        try:
            font = ImageFont.load_default()
            print("Используется стандартный шрифт")
        except:
            print("Не удалось загрузить шрифт")
            return cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)

    text_color = (255, 255, 255)  # Белый цвет
    line_height = 25

    # Добавляем текст на изображение
    for i, line in enumerate(text_lines):
        y_pos = y_start + i * line_height

        try:
            # Получаем размеры текста
            bbox = draw.textbbox((0, 0), line, font=font)
            text_width = bbox[2] - bbox[0]
            text_height = bbox[3] - bbox[1]

            # Рисуем полупрозрачный фон
            draw.rectangle([x_start - 5, y_pos - 5, x_start + text_width + 5, y_pos + text_height + 5],
                           fill=(0, 0, 0))

            # Добавляем текст
            draw.text((x_start, y_pos), line, fill=text_color, font=font)
        except Exception as e:
            print(f"Ошибка при отрисовке текста: {e}")
            continue

    # Конвертируем обратно в OpenCV format (BGR)
    return cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)

def add_qr_overlay(image, qr_data, position='bottom_right'):
    """Добавляет QR-коды на изображение"""
    height, width = image.shape[:2]

    # Размер QR-кода
    qr_size = 100

    if position == 'bottom_right':
        x_start = width - qr_size - 20
        y_start = height - qr_size - 20
    else:
        x_start = 20
        y_start = height - qr_size - 20

    qr_count = 0

    # Добавляем Telegram QR-код
    if 'telegram_qr' in qr_data and qr_data['telegram_qr']:
        try:
            # Декодируем base64 в изображение
            qr_bytes = base64.b64decode(qr_data['telegram_qr'])
            qr_np = np.frombuffer(qr_bytes, np.uint8)
            qr_img = cv2.imdecode(qr_np, cv2.IMREAD_COLOR)

            if qr_img is not None:
                # Масштабируем QR-код до нужного размера
                qr_img = cv2.resize(qr_img, (qr_size, qr_size))

                # Смещаем позицию в зависимости от количества QR-кодов
                x_pos = x_start - (qr_count * (qr_size + 10))
                y_pos = y_start

                # Накладываем QR-код на изображение
                y_end = y_pos + qr_size
                x_end = x_pos + qr_size

                # Убеждаемся, что координаты в пределах изображения
                if (y_pos >= 0 and y_end <= height and
                        x_pos >= 0 and x_end <= width):
                    image[y_pos:y_end, x_pos:x_end] = qr_img

                    # Добавляем подпись под QR-кодом (английский текст)
                    label_y = y_end + 15
                    if label_y < height - 5:
                        cv2.putText(image, "Telegram",
                                    (x_pos, label_y),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                                    (255, 255, 255), 1)

                    qr_count += 1

        except Exception as e:
            print(f"Ошибка при добавлении Telegram QR: {e}")

    # Добавляем Contact QR-код
    if 'contact_qr' in qr_data and qr_data['contact_qr']:
        try:
            qr_bytes = base64.b64decode(qr_data['contact_qr'])
            qr_np = np.frombuffer(qr_bytes, np.uint8)
            qr_img = cv2.imdecode(qr_np, cv2.IMREAD_COLOR)

            if qr_img is not None:
                qr_img = cv2.resize(qr_img, (qr_size, qr_size))

                # Смещаем позицию для второго QR-кода
                x_pos = x_start - (qr_count * (qr_size + 10))
                y_pos = y_start

                x_end = x_pos + qr_size
                y_end = y_pos + qr_size

                if (y_pos >= 0 and y_end <= height and
                        x_pos >= 0 and x_end <= width):
                    image[y_pos:y_end, x_pos:x_end] = qr_img

                    # Добавляем подпись (английский текст)
                    label_y = y_end + 15
                    if label_y < height - 5:
                        cv2.putText(image, "Contact",
                                    (x_pos, label_y),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                                    (255, 255, 255), 1)

                    qr_count += 1

        except Exception as e:
            print(f"Ошибка при добавлении Contact QR: {e}")


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

# Модифицированный основной цикл
with torch.no_grad():
    # Загружаем данные сотрудника перед началом цикла
    print("Загрузка данных сотрудника...")
    if load_employee_data():
        print("✅ Данные сотрудника загружены успешно")
    else:
        print("❌ Ошибка загрузки данных сотрудника")

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

        # --- ДОБАВЛЕНИЕ НАДПИСЕЙ И QR-КОДОВ ---
        try:
            # Получаем данные для отображения из модуля сотрудника
            employee_data = employee_data_module.get_data_for_rendering()

            # Добавляем текстовую информацию в левый нижний угол (используем PIL версию)
            output_bgr = add_text_overlay_pil(output_bgr, employee_data, 'bottom_left')

            # Добавляем QR-коды в правый нижний угол
            if 'qr_codes' in employee_data:
                add_qr_overlay(output_bgr, employee_data['qr_codes'], 'bottom_right')

        except Exception as e:
            # В случае ошибки просто продолжаем работу без оверлеев
            if ENABLE_DEBUG:
                print(f"Ошибка добавления оверлеев: {e}")

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
            # Добавляем информацию о сотруднике в верхний левый угол (английский текст)
            if employee_data_module.current_data:
                # Используем только английский текст для FPS строки
                employee_name = "Anna"  # Простое английское имя для теста
                cv2.putText(output_bgr, f"{employee_name} | FPS: {fps:.1f} | Enh: {tag}", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            else:
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
        elif key == ord('p'):
            # Переключение уровня конфиденциальности
            current_level = employee_data_module.current_privacy_level
            if current_level == PrivacyLevel.LOW:
                employee_data_module.set_privacy_level(PrivacyLevel.MEDIUM)
                print("Privacy: MEDIUM - добавлены компания и отдел")
            elif current_level == PrivacyLevel.MEDIUM:
                employee_data_module.set_privacy_level(PrivacyLevel.HIGH)
                print("Privacy: HIGH - добавлены контакты и QR-коды")
            else:
                employee_data_module.set_privacy_level(PrivacyLevel.LOW)
                print("Privacy: LOW - только имя и должность")

cap.release()
cv2.destroyAllWindows()