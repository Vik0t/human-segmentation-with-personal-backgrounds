import cv2
import torch
from PIL import Image
from RobustVideoMatting.model import MattingNetwork
from torchvision.transforms import functional as F

# чуваки с видюхами от nvidia используем cuda, Яна Юрьевна - mps
device = 'mps' if torch.mps.is_available() else 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")

model = MattingNetwork('mobilenetv3').eval().to(device)
model.load_state_dict(torch.load('/Users/vik0t/hackatons/human-segment/RobustVideoMatting/rvm_mobilenetv3.pth', map_location=device))

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

rec = [None] * 4
downsample_ratio = 0.25  # менььше - быстрее но хуже качество


from torchvision.io import read_image

image_path = "image.png"
img = read_image(image_path)  # Returns (C, H, W) tensor with values in [0, 255]
pink_rgb = img.float().unsqueeze(0).to(device) / 255.0  # Normalize to [0,1]
# pink_rgb = torch.tensor([255, 192, 203], dtype=torch.float32, device=device).view(1, 3, 1, 1) / 255.0


def enhance_low_light(frame):
    """Enhance low-light using CLAHE on the luminance channel."""
    img_y_cr_cb = cv2.cvtColor(frame, cv2.COLOR_BGR2YCrCb)
    y, cr, cb = cv2.split(img_y_cr_cb)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    y = clahe.apply(y)
    img_y_cr_cb = cv2.merge((y, cr, cb))
    enhanced = cv2.cvtColor(img_y_cr_cb, cv2.COLOR_YCrCb2BGR)
    return frame



with torch.no_grad():
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = enhance_low_light(frame)
        # меняем размер кадра для ускорения
        src = cv2.resize(frame, (1280, 720))
        src_rgb = cv2.cvtColor(src, cv2.COLOR_BGR2RGB)
        # в тензор [1, 3, H, W], нормализованный в [0,1]
        src_tensor = torch.from_numpy(src_rgb).to(device).float().div(255).permute(2, 0, 1).unsqueeze(0)

        fgr, pha, *rec = model(src_tensor, *rec, downsample_ratio=downsample_ratio)
        bg_img = Image.open("image.png").convert("RGB")

        # Get target size from pha (assuming pha is [1,1,H,W])
        _, _, H, W = pha.shape

        # Resize background to match
        bg_img = bg_img.resize((W, H), Image.LANCZOS)  # (width, height)

        # Convert to tensor
        bg_tensor = F.to_tensor(bg_img).to(device)  # Shape: (3, H, W)
        pink_rgb = bg_tensor.unsqueeze(0)  # Shape: (1, 3, H, W)
        # композитный кадр: foreground * alpha + pink * (1 - alpha)
        com = fgr * pha + pink_rgb * (1 - pha)

        com_rgb = (com.squeeze(0).permute(1, 2, 0) * 255).byte().cpu().numpy()
        com_bgr = cv2.cvtColor(com_rgb, cv2.COLOR_RGB2BGR)

        cv2.imshow('human segmenting', com_bgr)
        if cv2.waitKey(1) == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()