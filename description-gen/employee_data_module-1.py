import json
import re
from typing import Dict, Any, Optional, List
from dataclasses import dataclass
from enum import Enum
import urllib.parse
from pathlib import Path


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

        Args:
            data_source: Путь к JSON файлу, URL или готовый словарь с данными

        Returns:
            Employee: Объект с данными сотрудника
        """
        try:
            if isinstance(data_source, (str, Path)):
                # Загрузка из файла
                if str(data_source).startswith(('http://', 'https://')):
                    json_data = self._load_from_url(data_source)
                else:
                    json_data = self._load_from_file(data_source)
            elif isinstance(data_source, dict):
                # Готовый словарь
                json_data = data_source
            else:
                raise ValueError("Неподдерживаемый источник данных")

            # Валидация данных
            self._validate_data(json_data)

            # Преобразование в объект Employee
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

        # Парсинг контактной информации
        contact_dict = employee_dict.get('contact', {})
        contact = ContactInfo(
            email=contact_dict.get('email', ''),
            telegram=contact_dict.get('telegram')
        )

        # Парсинг брендинга
        branding_dict = employee_dict.get('branding', {})
        colors_dict = branding_dict.get('corporate_colors', {})
        corporate_colors = CorporateColors(
            primary=colors_dict.get('primary', '#000000'),
            secondary=colors_dict.get('secondary', '#FFFFFF')
        )

        branding = Branding(
            logo_url=branding_dict.get('logo_url', ''),
            corporate_colors=corporate_colors,
            slogan=branding_dict.get('slogan', '')
        )

        # Парсинг уровня конфиденциальности
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

        # Обязательные поля
        required_fields = ['full_name', 'position', 'company', 'department']
        for field in required_fields:
            if field not in emp or not emp[field]:
                raise ValueError(f"Обязательное поле отсутствует: {field}")

        # Валидация формата имени
        if not self._is_valid_name_format(emp['full_name']):
            print("Предупреждение: полное имя может быть в нестандартном формате")

        # Валидация email
        # if emp.get('contact', {}).get('email'):
        #     if not self._is_valid_email(emp['contact']['email']):
        #         raise ValueError("Неверный формат email")

        # Валидация URL логотипа
        if emp.get('branding', {}).get('logo_url'):
            if not self._is_valid_url(emp['branding']['logo_url']):
                raise ValueError("Неверный формат URL логотипа")

        # Валидация цветов
        if emp.get('branding', {}).get('corporate_colors'):
            colors = emp['branding']['corporate_colors']
            if colors.get('primary') and not self._is_valid_hex_color(colors['primary']):
                raise ValueError("Неверный формат основного цвета")
            if colors.get('secondary') and not self._is_valid_hex_color(colors['secondary']):
                raise ValueError("Неверный формат дополнительного цвета")

        # Валидация уровня конфиденциальности
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

        Returns:
            Dict: Данные для передачи в модуль рендеринга фона
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

        return filtered_data

    def update_data(self, new_fields: Dict[str, Any]) -> None:
        """Обновляет данные сотрудника"""
        if not self.current_data:
            raise ValueError("Данные не загружены")

        # Создаем временную копию для валидации
        temp_data = self._employee_to_dict(self.current_data)
        temp_data['employee'].update(new_fields)

        # Валидируем обновленные данные
        self._validate_data(temp_data)

        # Обновляем текущие данные
        self.current_data = self._parse_employee_data(temp_data['employee'])
        print("Данные успешно обновлены")

    def get_full_data(self) -> Employee:
        """Получает полные данные сотрудника"""
        if not self.current_data:
            raise ValueError("Данные не загружены")
        return self.current_data

    def generate_contact_qr_data(self) -> Optional[Dict[str, str]]:
        """Генерирует данные для QR-кода контактов"""
        if not self.current_data or not self.current_data.contact.email:
            return None

        contact = self.current_data.contact
        contact_text = f"Email: {contact.email}"
        if contact.telegram:
            contact_text += f"\nTelegram: {contact.telegram}"

        # В реальной реализации здесь будет генерация QR-кода
        # Например, с помощью библиотеки qrcode
        print(f"Генерация QR-кода для: {contact_text}")

        return {
            'text': contact_text,
            # 'qr_image': ...  # Здесь будет PIL Image или bytes с QR-кодом
        }

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