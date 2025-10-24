import json
import re
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass
from enum import Enum
import urllib.parse
from pathlib import Path
import base64
from io import BytesIO
import qrcode
from PIL import Image, ImageDraw


class PrivacyLevel(Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


@dataclass
class ContactInfo:
    email: str
    telegram: Optional[str] = None


@dataclass
class CorporateColors:
    primary: str
    secondary: str


@dataclass
class Branding:
    logo_url: str
    corporate_colors: CorporateColors
    slogan: str


@dataclass
class QRCodeData:
    telegram_url: Optional[str] = None
    telegram_qr_base64: Optional[str] = None
    contact_qr_base64: Optional[str] = None


@dataclass
class DiffusionModelRequest:
    prompt: str
    negative_prompt: str
    colors: Dict[str, str]
    text_elements: Dict[str, str]
    qr_codes_required: bool
    style: str = "corporate professional"
    width: int = 1024
    height: int = 768


@dataclass
class Employee:
    full_name: str
    position: str
    company: str
    department: str
    office_location: str
    contact: ContactInfo
    branding: Branding
    privacy_level: PrivacyLevel


class EmployeeDataModule:
    """
    Модуль для загрузки и обработки данных сотрудника из JSON
    """

    def __init__(self):
        self.current_data: Optional[Employee] = None
        self.current_privacy_level: PrivacyLevel = PrivacyLevel.MEDIUM
        self.valid_privacy_levels = [level.value for level in PrivacyLevel]

    def load_employee_data(self, data_source: str | Path | Dict[str, Any]) -> Employee:
        """
        Загружает данные сотрудника из различных источников
        """
        try:
            if isinstance(data_source, (str, Path)):
                if str(data_source).startswith(('http://', 'https://')):
                    json_data = self._load_from_url(data_source)
                else:
                    json_data = self._load_from_file(data_source)
            elif isinstance(data_source, dict):
                json_data = data_source
            else:
                raise ValueError("Неподдерживаемый источник данных")

            self._validate_data(json_data)
            self.current_data = self._parse_employee_data(json_data['employee'])

            print("Данные сотрудника успешно загружены")
            return self.current_data

        except Exception as e:
            print(f"Ошибка загрузки данных сотрудника: {e}")
            raise

    def _load_from_file(self, file_path: str | Path) -> Dict[str, Any]:
        """Загружает JSON из файла"""
        with open(file_path, 'r', encoding='utf-8') as file:
            return json.load(file)

    def _load_from_url(self, url: str) -> Dict[str, Any]:
        """Загружает JSON по URL"""
        try:
            import requests
            response = requests.get(url)
            response.raise_for_status()
            return response.json()
        except ImportError:
            raise ImportError("Для загрузки по URL требуется установить библиотеку requests")

    def _parse_employee_data(self, employee_dict: Dict[str, Any]) -> Employee:
        """Парсит словарь в объект Employee"""
        contact_dict = employee_dict.get('contact', {})
        contact = ContactInfo(
            email=contact_dict.get('email', ''),
            telegram=contact_dict.get('telegram')
        )

        branding_dict = employee_dict.get('branding', {})
        colors_dict = branding_dict.get('corporate_colors', {})
        corporate_colors = CorporateColors(
            primary=colors_dict.get('primary', '#0052CC'),
            secondary=colors_dict.get('secondary', '#0088D9')
        )

        branding = Branding(
            logo_url=branding_dict.get('logo_url', ''),
            corporate_colors=corporate_colors,
            slogan=branding_dict.get('slogan', '')
        )

        privacy_level = PrivacyLevel(employee_dict.get('privacy_level', 'medium'))

        return Employee(
            full_name=employee_dict['full_name'],
            position=employee_dict['position'],
            company=employee_dict['company'],
            department=employee_dict['department'],
            office_location=employee_dict.get('office_location', ''),
            contact=contact,
            branding=branding,
            privacy_level=privacy_level
        )

    def _validate_data(self, data: Dict[str, Any]) -> None:
        """Валидация структуры данных"""
        if not data or not isinstance(data, dict):
            raise ValueError("Данные должны быть словарем")

        if 'employee' not in data:
            raise ValueError("Отсутствует корневой объект employee")

        emp = data['employee']

        required_fields = ['full_name', 'position', 'company', 'department']
        for field in required_fields:
            if field not in emp or not emp[field]:
                raise ValueError(f"Обязательное поле отсутствует: {field}")

        if not self._is_valid_name_format(emp['full_name']):
            print("Предупреждение: полное имя может быть в нестандартном формате")

        if emp.get('contact', {}).get('email'):
            if not self._is_valid_email(emp['contact']['email']):
                raise ValueError("Неверный формат email")

        if emp.get('branding', {}).get('logo_url'):
            if not self._is_valid_url(emp['branding']['logo_url']):
                raise ValueError("Неверный формат URL логотипа")

        if emp.get('branding', {}).get('corporate_colors'):
            colors = emp['branding']['corporate_colors']
            if colors.get('primary') and not self._is_valid_hex_color(colors['primary']):
                raise ValueError("Неверный формат основного цвета")
            if colors.get('secondary') and not self._is_valid_hex_color(colors['secondary']):
                raise ValueError("Неверный формат дополнительного цвета")

        if emp.get('privacy_level'):
            if emp['privacy_level'] not in self.valid_privacy_levels:
                raise ValueError(
                    f"Неверный уровень конфиденциальности. Допустимые значения: {', '.join(self.valid_privacy_levels)}")

    def set_privacy_level(self, level: str | PrivacyLevel) -> None:
        """Устанавливает уровень конфиденциальности"""
        if isinstance(level, str):
            level = PrivacyLevel(level)
        self.current_privacy_level = level
        print(f"Уровень конфиденциальности установлен: {level.value}")

    def get_data_for_rendering(self) -> Dict[str, Any]:
        """
        Получает отфильтрованные данные для рендеринга с учетом уровня конфиденциальности
        """
        if not self.current_data:
            raise ValueError("Данные не загружены")

        emp = self.current_data
        filtered_data = {
            'branding': {
                'logo_url': emp.branding.logo_url,
                'corporate_colors': {
                    'primary': emp.branding.corporate_colors.primary,
                    'secondary': emp.branding.corporate_colors.secondary
                },
                'slogan': emp.branding.slogan
            }
        }

        # Применяем фильтрацию по уровню конфиденциальности
        if self.current_privacy_level == PrivacyLevel.LOW:
            filtered_data['full_name'] = emp.full_name
            filtered_data['position'] = emp.position

        elif self.current_privacy_level == PrivacyLevel.MEDIUM:
            filtered_data['full_name'] = emp.full_name
            filtered_data['position'] = emp.position
            filtered_data['company'] = emp.company
            filtered_data['department'] = emp.department
            filtered_data['office_location'] = emp.office_location

        elif self.current_privacy_level == PrivacyLevel.HIGH:
            filtered_data['full_name'] = emp.full_name
            filtered_data['position'] = emp.position
            filtered_data['company'] = emp.company
            filtered_data['department'] = emp.department
            filtered_data['office_location'] = emp.office_location
            filtered_data['contact'] = {
                'email': emp.contact.email,
                'telegram': emp.contact.telegram
            }

            # Генерируем QR-коды в корпоративных цветах для высокого уровня конфиденциальности
            qr_data = self._generate_colored_qr_codes()
            if qr_data:
                filtered_data['qr_codes'] = {
                    'telegram_qr': qr_data.telegram_qr_base64,
                    'contact_qr': qr_data.contact_qr_base64,
                    'telegram_url': qr_data.telegram_url
                }

        # Добавляем запрос к диффузионной модели для всех уровней
        diffusion_request = self._create_diffusion_model_request(filtered_data)
        filtered_data['diffusion_model_request'] = diffusion_request.__dict__

        return filtered_data

    def _generate_colored_qr_codes(self) -> Optional[QRCodeData]:
        """
        Генерирует QR-коды в корпоративных цветах
        """
        if not self.current_data:
            return None

        emp = self.current_data
        primary_color = emp.branding.corporate_colors.primary
        secondary_color = emp.branding.corporate_colors.secondary

        qr_data = QRCodeData()

        # Генерация QR-кода для Telegram
        if emp.contact.telegram:
            telegram_username = emp.contact.telegram.lstrip('@')
            telegram_url = f"https://t.me/{telegram_username}"
            qr_data.telegram_url = telegram_url

            # Создаем цветной QR-код для Telegram
            telegram_qr = qrcode.QRCode(
                version=1,
                error_correction=qrcode.constants.ERROR_CORRECT_H,
                box_size=10,
                border=4,
            )
            telegram_qr.add_data(telegram_url)
            telegram_qr.make(fit=True)

            # Создаем изображение с корпоративными цветами
            telegram_img = telegram_qr.make_image(
                fill_color=primary_color,
                back_color=secondary_color
            ).convert('RGB')

            buffered = BytesIO()
            telegram_img.save(buffered, format="PNG", optimize=True)
            qr_data.telegram_qr_base64 = base64.b64encode(buffered.getvalue()).decode()

        # Генерация QR-кода для контактной информации
        contact_text = self._generate_vcard_text()
        if contact_text:
            contact_qr = qrcode.QRCode(
                version=1,
                error_correction=qrcode.constants.ERROR_CORRECT_H,
                box_size=10,
                border=4,
            )
            contact_qr.add_data(contact_text)
            contact_qr.make(fit=True)

            # Создаем изображение с инвертированными корпоративными цветами
            contact_img = contact_qr.make_image(
                fill_color=secondary_color,
                back_color=primary_color
            ).convert('RGB')

            buffered = BytesIO()
            contact_img.save(buffered, format="PNG", optimize=True)
            qr_data.contact_qr_base64 = base64.b64encode(buffered.getvalue()).decode()

        return qr_data

    def _create_diffusion_model_request(self, rendering_data: Dict[str, Any]) -> DiffusionModelRequest:
        """
        Создает запрос к диффузионной модели для генерации фона
        """
        emp = self.current_data
        primary_color = emp.branding.corporate_colors.primary
        secondary_color = emp.branding.corporate_colors.secondary

        # Формируем текстовые элементы для отображения
        text_elements = {}

        if 'full_name' in rendering_data:
            text_elements['name'] = rendering_data['full_name']
        if 'position' in rendering_data:
            text_elements['position'] = rendering_data['position']
        if 'company' in rendering_data:
            text_elements['company'] = rendering_data['company']
        if 'department' in rendering_data:
            text_elements['department'] = rendering_data['department']
        if 'office_location' in rendering_data:
            text_elements['location'] = rendering_data['office_location']
        if emp.branding.slogan:
            text_elements['slogan'] = emp.branding.slogan

        # Создаем промпт для генеративной модели
        prompt_parts = [
            "Professional corporate background for employee identification",
            f"Company: {emp.company}",
            f"Colors: primary {primary_color}, secondary {secondary_color}",
            "Clean, modern design with geometric elements",
            "Professional business style",
            "High resolution, suitable for video conferencing",
            "Elegant typography areas for text placement"
        ]

        if emp.branding.slogan:
            prompt_parts.append(f"Incorporate slogan: {emp.branding.slogan}")

        negative_prompt_parts = [
            "blurry, low quality, pixelated",
            "cluttered, messy, chaotic",
            "childish, cartoonish, informal",
            "dark, depressing, gloomy",
            "text directly on image",
            "watermarks, logos, signatures"
        ]

        return DiffusionModelRequest(
            prompt=", ".join(prompt_parts),
            negative_prompt=", ".join(negative_prompt_parts),
            colors={
                "primary": primary_color,
                "secondary": secondary_color,
                "text_color": self._get_contrast_color(primary_color),
                "qr_color": primary_color,
                "background_color": secondary_color
            },
            text_elements=text_elements,
            qr_codes_required='qr_codes' in rendering_data,
            style="corporate professional modern business"
        )

    def _generate_vcard_text(self) -> Optional[str]:
        """
        Генерирует текст vCard для QR-кода контактов
        """
        if not self.current_data:
            return None

        emp = self.current_data

        # Разбиваем полное имя на компоненты
        name_parts = emp.full_name.split()
        if len(name_parts) >= 3:
            last_name, first_name, middle_name = name_parts[0], name_parts[1], name_parts[2]
        elif len(name_parts) == 2:
            last_name, first_name, middle_name = name_parts[0], name_parts[1], ""
        else:
            last_name, first_name, middle_name = emp.full_name, "", ""

        vcard_lines = [
            "BEGIN:VCARD",
            "VERSION:3.0",
            f"N:{last_name};{first_name};{middle_name};;",
            f"FN:{emp.full_name}",
            f"ORG:{emp.company}",
            f"TITLE:{emp.position}",
            f"EMAIL:{emp.contact.email}",
        ]

        # Добавляем Telegram как социальную сеть
        if emp.contact.telegram:
            telegram_username = emp.contact.telegram.lstrip('@')
            vcard_lines.append(f"X-SOCIALPROFILE;type=telegram:https://t.me/{telegram_username}")

        # Добавляем заметку с дополнительной информацией
        note_lines = []
        if emp.department:
            note_lines.append(f"Отдел: {emp.department}")
        if emp.office_location:
            note_lines.append(f"Локация: {emp.office_location}")
        if emp.branding.slogan:
            note_lines.append(f"Слоган: {emp.branding.slogan}")

        if note_lines:
            vcard_lines.append(f"NOTE:{' | '.join(note_lines)}")

        vcard_lines.append("END:VCARD")

        return "\n".join(vcard_lines)

    def _get_contrast_color(self, hex_color: str) -> str:
        """
        Определяет контрастный цвет для текста на основе фона
        """
        # Убираем # из начала
        hex_color = hex_color.lstrip('#')

        # Конвертируем в RGB
        r, g, b = int(hex_color[0:2], 16), int(hex_color[2:4], 16), int(hex_color[4:6], 16)

        # Вычисляем яркость (формула из WCAG)
        luminance = (0.299 * r + 0.587 * g + 0.114 * b) / 255

        # Возвращаем белый для темных цветов, черный для светлых
        return "#FFFFFF" if luminance < 0.5 else "#000000"

    def generate_contact_qr_data(self) -> Optional[Dict[str, Any]]:
        """
        Генерирует данные для QR-кода контактов (публичный метод)
        """
        if not self.current_data:
            return None

        qr_data = self._generate_colored_qr_codes()
        if not qr_data:
            return None

        result = {
            'contact_text': self._generate_vcard_text(),
            'has_telegram': bool(qr_data.telegram_url),
        }

        if qr_data.telegram_url:
            result['telegram_url'] = qr_data.telegram_url
            result['telegram_qr_base64'] = qr_data.telegram_qr_base64

        if qr_data.contact_qr_base64:
            result['contact_qr_base64'] = qr_data.contact_qr_base64

        # Добавляем запрос к диффузионной модели
        rendering_data = self.get_data_for_rendering()
        diffusion_request = self._create_diffusion_model_request(rendering_data)
        result['diffusion_model_request'] = diffusion_request.__dict__

        return result

    def update_data(self, new_fields: Dict[str, Any]) -> None:
        """Обновляет данные сотрудника"""
        if not self.current_data:
            raise ValueError("Данные не загружены")

        temp_data = self._employee_to_dict(self.current_data)
        temp_data['employee'].update(new_fields)

        self._validate_data(temp_data)
        self.current_data = self._parse_employee_data(temp_data['employee'])
        print("Данные успешно обновлены")

    def get_full_data(self) -> Employee:
        """Получает полные данные сотрудника"""
        if not self.current_data:
            raise ValueError("Данные не загружены")
        return self.current_data

    def export_to_json(self, file_path: str | Path) -> None:
        """Экспортирует текущие данные в JSON файл"""
        if not self.current_data:
            raise ValueError("Данные не загружены")

        data_dict = self._employee_to_dict(self.current_data)
        with open(file_path, 'w', encoding='utf-8') as file:
            json.dump(data_dict, file, ensure_ascii=False, indent=2)

        print(f"Данные экспортированы в {file_path}")

    def _employee_to_dict(self, employee: Employee) -> Dict[str, Any]:
        """Конвертирует объект Employee в словарь"""
        return {
            'employee': {
                'full_name': employee.full_name,
                'position': employee.position,
                'company': employee.company,
                'department': employee.department,
                'office_location': employee.office_location,
                'contact': {
                    'email': employee.contact.email,
                    'telegram': employee.contact.telegram
                },
                'branding': {
                    'logo_url': employee.branding.logo_url,
                    'corporate_colors': {
                        'primary': employee.branding.corporate_colors.primary,
                        'secondary': employee.branding.corporate_colors.secondary
                    },
                    'slogan': employee.branding.slogan
                },
                'privacy_level': employee.privacy_level.value
            }
        }

    # Вспомогательные методы валидации
    def _is_valid_name_format(self, name: str) -> bool:
        return isinstance(name, str) and len(name.split()) >= 2

    def _is_valid_email(self, email: str) -> bool:
        pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
        return bool(re.match(pattern, email))

    def _is_valid_url(self, url: str) -> bool:
        try:
            result = urllib.parse.urlparse(url)
            return all([result.scheme, result.netloc])
        except:
            return False

    def _is_valid_hex_color(self, color: str) -> bool:
        pattern = r'^#([A-Fa-f0-9]{6}|[A-Fa-f0-9]{3})$'
        return bool(re.match(pattern, color))

    def reset(self) -> None:
        """Сбрасывает текущие данные"""
        self.current_data = None
        self.current_privacy_level = PrivacyLevel.MEDIUM


employee_data_module = EmployeeDataModule()