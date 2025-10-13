import cv2
import torch
from RobustVideoMatting.model import MattingNetwork

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

pink_rgb = torch.tensor([255, 192, 203], dtype=torch.float32, device=device).view(1, 3, 1, 1) / 255.0

with torch.no_grad():
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # меняем размер кадра для ускорения
        src = cv2.resize(frame, (1280, 720))
        src_rgb = cv2.cvtColor(src, cv2.COLOR_BGR2RGB)
        # в тензор [1, 3, H, W], нормализованный в [0,1]
        src_tensor = torch.from_numpy(src_rgb).to(device).float().div(255).permute(2, 0, 1).unsqueeze(0)

        fgr, pha, *rec = model(src_tensor, *rec, downsample_ratio=downsample_ratio)

        # композитный кадр: foreground * alpha + pink * (1 - alpha)
        com = fgr * pha + pink_rgb * (1 - pha)

        com_rgb = (com.squeeze(0).permute(1, 2, 0) * 255).byte().cpu().numpy()
        com_bgr = cv2.cvtColor(com_rgb, cv2.COLOR_RGB2BGR)

        cv2.imshow('Сегментация чувака', com_bgr)
        if cv2.waitKey(1) == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()