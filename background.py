# server.py
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import base64
from io import BytesIO
from PIL import Image
from pathlib import Path

from background_gen.api import BackgroundGenerator

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
)

class GenerateRequest(BaseModel):
    employee: dict
    background_base64: str  # фон в base64

@app.post("/generate_background")
def generate_background(request: GenerateRequest):
    try:
        # Декодируем фон
        bg_data = base64.b64decode(request.background_base64.split(",")[-1])
        bg_image = Image.open(BytesIO(bg_data)).convert("RGB")
        
        # Сохраняем временно
        temp_bg = Path("temp_bg.jpg")
        bg_image.save(temp_bg, "JPEG")

        # Генерируем
        generator = BackgroundGenerator(request.employee, background_path=temp_bg)
        result_img = generator.generate()

        output_path = "generated_bg.jpg"
        result_img.save(output_path, "JPEG", quality=95)

        # Удаляем временный файл
        temp_bg.unlink(missing_ok=True)

        return FileResponse(output_path, media_type="image/jpeg")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))