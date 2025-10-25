import json
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont, ImageColor
import requests
from io import BytesIO
import qrcode
from qrcode.image.styledpil import StyledPilImage
from qrcode.image.styles.moduledrawers import RoundedModuleDrawer
import cairosvg

class BackgroundGenerator:
    FONT_PATH = Path("background_gen/Sansation-Bold.ttf")
    
    ICONS_DIR = Path("background_gen")

    def __init__(self, employee_data: dict, background_path: Path = None):
        self.employee = employee_data
        self.bg_path = background_path or Path("background_gen/default_bg.png")
        self.level = self.employee.get("privacy_level", "low")
        self.display_data = self._build_display_data()
        self.img = Image.open(self.bg_path).resize((1920, 1080), Image.LANCZOS)
        self.draw = ImageDraw.Draw(self.img, "RGBA")
        self.bg_brightness = self._get_background_brightness()
        self.is_light_bg = self.bg_brightness > 100
        self.primary_color, self.secondary_color, self.has_branding_colors = self._resolve_colors()

    def _build_display_data(self) -> dict:
        data = {}
        if self.level in ["low", "medium", "high"]:
            data.update({
                "full_name": self.employee.get("full_name"),
                "position": self.employee.get("position")
            })
        if self.level in ["medium", "high"]:
            data.update({
                "company": self.employee.get("company"),
                "department": self.employee.get("department"),
                "office_location": self.employee.get("office_location")
            })
        if self.level == "high":
            contact = self.employee.get("contact", {})
            branding = self.employee.get("branding", {})
            data.update({
                "email": contact.get("email"),
                "telegram": contact.get("telegram"),
                "logo_url": branding.get("logo_url"),
                "slogan": branding.get("slogan")
            })
        return data

    def _get_background_brightness(self, sample_size=512) -> float:
        img = Image.open(self.bg_path).convert("RGB")
        img = img.resize((sample_size, sample_size), Image.LANCZOS)
        pixels = list(img.getdata())
        return sum(sum(p) for p in pixels) / (len(pixels) * 3)

    def _resolve_colors(self):
        primary = (0, 0, 0) if self.is_light_bg else (255, 255, 255)
        secondary = (80, 80, 80) if self.is_light_bg else (200, 200, 200)
        has_branding = False

        branding = self.employee.get("branding", {})
        corporate_colors = branding.get("corporate_colors", {})
        if corporate_colors and "primary" in corporate_colors and "secondary" in corporate_colors:
            try:
                primary = ImageColor.getrgb(corporate_colors["primary"])
                secondary = ImageColor.getrgb(corporate_colors["secondary"])
                has_branding = True
            except Exception:
                pass
        return primary, secondary, has_branding

    @staticmethod
    def _darken_color(color, factor=0.6):
        return tuple(int(c * factor) for c in color)

    @staticmethod
    def _load_font(size):
        try:
            return ImageFont.truetype(str(BackgroundGenerator.FONT_PATH), int(size))
        except OSError:
            return ImageFont.load_default()

    @staticmethod
    def _load_image_from_url(url):
        headers = {'User-Agent': 'Mozilla/5.0'}
        try:
            response = requests.get(url.strip(), headers=headers)
            response.raise_for_status()
            return Image.open(BytesIO(response.content))
        except Exception as e:
            print(f"Ошибка загрузки изображения {url}: {e}")
            return None

    @staticmethod
    def _recolor_icon(icon_path: Path, color):
        if icon_path.suffix.lower() == '.svg':
            png_data = cairosvg.svg2png(url=str(icon_path))
            icon = Image.open(BytesIO(png_data)).convert("RGBA")
        else:
            icon = Image.open(icon_path).convert("RGBA")
        
        # Recolor
        data = icon.getdata()
        new_data = []
        for item in data:
            if item[3] > 0:
                new_data.append((color[0], color[1], color[2], item[3]))
            else:
                new_data.append(item)
        icon.putdata(new_data)
        return icon

    @staticmethod
    def _qr_round(url: str, primary=(0, 82, 204), secondary=(0, 184, 217), use_gradient=True):
        qr = qrcode.QRCode(
            error_correction=qrcode.constants.ERROR_CORRECT_H,
            version=3,
            border=1,
            box_size=15
        )
        qr.add_data(url)
        qr.make(fit=True)

        if not use_gradient:
            return qr.make_image(fill_color="black", back_color="white").convert("RGBA")

        base_img = qr.make_image(fill_color="black", back_color="white")
        w, h = base_img.size
        gradient = Image.new("RGBA", (w, h))
        for y in range(h):
            ratio = y / h
            r = int(primary[0] + (secondary[0] - primary[0]) * ratio)
            g = int(primary[1] + (secondary[1] - primary[1]) * ratio)
            b = int(primary[2] + (secondary[2] - primary[2]) * ratio)
            for x in range(w):
                gradient.putpixel((x, y), (r, g, b, 255))

        mask = base_img.convert("L").point(lambda x: 255 if x == 0 else 0)
        result = Image.new("RGBA", (w, h))
        result.paste(gradient, (0, 0), mask)
        return result

    def _draw_text_in_box(self, text, x, y, max_width, max_height, fill_color):
        size = 48
        best_font = None
        best_bbox = None

        while size >= 10:
            font = self._load_font(size)
            bbox = self.draw.textbbox((0, 0), text, font=font)
            w = bbox[2] - bbox[0]
            h = bbox[3] - bbox[1]
            if w <= max_width and h <= max_height:
                best_font = font
                best_bbox = bbox
                break
            size -= 1

        if best_font is None:
            best_font = self._load_font(10)
            best_bbox = self.draw.textbbox((0, 0), text, font=best_font)

        offset_x = -best_bbox[0]
        offset_y = -best_bbox[1]
        self.draw.text((x + offset_x, y + offset_y), text, fill=fill_color, font=best_font)

    def _draw_name_and_position(self):
        if self.display_data.get("full_name"):
            self.draw.rounded_rectangle([38, 36, 38 + 575, 36 + 100], fill=(255, 255, 255) if self.is_light_bg else (0, 0, 0), radius=15)
            self._draw_text_in_box(self.display_data["full_name"], 64, 56, 530, 60, self.primary_color)
        if self.display_data.get("position"):
            self._draw_text_in_box(self.display_data["position"], 64, 56 + 40, 530, 40, self.secondary_color)

    def _draw_company_and_department(self):
        if self.display_data.get("company"):
            self.draw.rounded_rectangle([38, 144, 38 + 370, 144 + 425], fill=(255, 255, 255) if self.is_light_bg else (0, 0, 0), radius=15)
            self._draw_text_in_box(self.display_data["company"], 64, 163, 325, 40, self.primary_color)
        if self.display_data.get("department"):
            self._draw_text_in_box(self.display_data["department"], 64, 163 + 35, 325, 35, self.secondary_color)

    def _draw_email(self):
        if self.level == "high" and self.display_data.get("email"):
            card_fill = (255, 255, 255) if self.is_light_bg else (0, 0, 0)
            at_bg = self._darken_color(self.secondary_color)
            self.draw.rounded_rectangle([1500, 41, 1500 + 384, 41 + 50], fill=card_fill, radius=15)
            self.draw.rounded_rectangle([1503, 44, 1503 + 44, 44 + 44], fill=self.secondary_color, radius=15)
            self._draw_text_in_box(self.display_data["email"], 1551, 50, 370 - 50, 50, self.primary_color)
            self.draw.text((1508, 46), "@", fill=at_bg, font=self._load_font(32))

    def _draw_location(self):
        if self.display_data.get("office_location"):
            card_fill = (255, 255, 255) if self.is_light_bg else (0, 0, 0)
            at_bg = self._darken_color(self.secondary_color)
            self.draw.rounded_rectangle([38, 577, 38 + 384, 577 + 50], fill=card_fill, radius=15)
            self.draw.rounded_rectangle([40, 580, 40 + 44, 580 + 44], fill=self.secondary_color, radius=15)
            self._draw_text_in_box(self.display_data["office_location"], 90, 592, 375 - 50, 40, self.primary_color)
            try:
                icon = self._recolor_icon(self.ICONS_DIR / "location.svg", at_bg)
                icon = icon.resize((32, 32), Image.LANCZOS)
                self.img.paste(icon, (47, 586), icon)
            except Exception as e:
                print(f"Иконка локации: {e}")

    def _draw_logo(self):
        if self.level == "high" and self.display_data.get("logo_url"):
            logo_img = self._load_image_from_url(self.display_data["logo_url"])
            if logo_img:
                if logo_img.mode != "RGBA":
                    logo_img = logo_img.convert("RGBA")
                block_w, block_h = 310, 310
                max_logo = min(block_w - 40, block_h - 40)
                logo_img.thumbnail((max_logo, max_logo), Image.LANCZOS)
                container = Image.new("RGBA", (block_w, block_h), (0, 0, 0, 0))
                d_cont = ImageDraw.Draw(container)
                d_cont.rounded_rectangle([0, 0, block_w - 1, block_h - 1], outline=self.primary_color, width=6, radius=30)
                x = (block_w - logo_img.width) // 2
                y = (block_h - logo_img.height) // 2
                container.paste(logo_img, (x, y), logo_img)
                self.img.paste(container, (64, 235), container)

    def _draw_telegram_qr(self):
        if self.level == "high" and self.display_data.get("telegram"):
            card_fill = (255, 255, 255) if self.is_light_bg else (0, 0, 0)
            self.draw.rounded_rectangle([1508, 100, 1508 + 375, 100 + 425], fill=card_fill, radius=15)
            url = self.display_data["telegram"].replace("@", "https://t.me/").strip()
            qr_img = self._qr_round(url, self.primary_color, self.secondary_color, self.has_branding_colors)
            qr_img = qr_img.convert("RGBA").resize((335, 335), Image.LANCZOS)
            self.img.paste(qr_img, (1530, 120), qr_img)
            self._draw_text_in_box(self.display_data["telegram"], 1535, 470, 325, 50, self.primary_color)

    def _draw_slogan(self):
        if self.level == "high" and self.display_data.get("slogan"):
            card_fill = (255, 255, 255) if self.is_light_bg else (0, 0, 0)
            slogan = f"«{self.display_data['slogan']}»"
            self.draw.rounded_rectangle([1508, 530, 1508 + 375, 530 + 50], fill=card_fill, radius=15)
            self._draw_text_in_box(slogan, 1523, 543, 355, 50, self.primary_color)

    def generate(self) -> Image.Image:
        self._draw_name_and_position()
        self._draw_company_and_department()
        self._draw_email()
        self._draw_location()
        self._draw_logo()
        self._draw_telegram_qr()
        self._draw_slogan()
        return self.img.convert("RGB")

# --- Пример использования ---
if __name__ == "__main__":
    sample_json_str = """
    {
      "employee": {
        "full_name": "Иванов Сергей Петрович",
        "position": "Ведущий инженер по компьютерному зрению",
        "company": "ООО «Рога и Копыта»",
        "department": "Департамент компьютерного зрения",
        "office_location": "Новосибирск, технопарк «Идея»",
        "contact": {
          "email": "sergey.ivanov@t1dp.ru",
          "telegram": "@sergey_vision"
        },
        "branding": {
          "logo_url": "https://upload.wikimedia.org/wikipedia/commons/thumb/2/2f/Google_2015_logo.svg/250px-Google_2015_logo.svg.png",
          "corporate_colors": {
            "primary": "#4285F4",
            "secondary": "#34A853"
          },
          "slogan": "Инновации в каждый кадр"
        },
        "privacy_level": "high"
      }
    }
    """

    data = json.loads(sample_json_str)
    generator = BackgroundGenerator(data["employee"])
    img = generator.generate()
    img.save("personalized_background_high.jpg", "JPEG", quality=100)
    print("✅ Сохранено!")