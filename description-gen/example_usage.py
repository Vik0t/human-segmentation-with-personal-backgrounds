# example_usage_with_qr.py
from employee_data_module import employee_data_module, PrivacyLevel
import json
import base64

# Пример данных с Telegram
example_data_with_telegram = {
    "employee": {
        "full_name": "Иванов Сергей Петрович",
        "position": "Ведущий инженер по компьютерному зрению",
        "company": "ООО «Рога и Копыта»",
        "department": "Департамент компьютерного зрения",
        "office_location": "Новосибирск, техноларк «Идея»",
        "contact": {
            "email": "sergey.ivanov@tldp.ru",
            "telegram": "@sergey_vision"
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


def demonstrate_module():
    """Демонстрация работы модуля"""
    try:
        # 1. Загрузка данных
        employee = employee_data_module.load_employee_data(example_data_with_telegram)

        # 2. Тестирование разных уровней конфиденциальности
        print("=== Low privacy level ===")
        employee_data_module.set_privacy_level(PrivacyLevel.LOW)
        low_data = employee_data_module.get_data_for_rendering()
        print(json.dumps(low_data, ensure_ascii=False, indent=2))

        print("\n=== Medium privacy level ===")
        employee_data_module.set_privacy_level(PrivacyLevel.MEDIUM)
        medium_data = employee_data_module.get_data_for_rendering()
        print(json.dumps(medium_data, ensure_ascii=False, indent=2))

        print("\n=== High privacy level ===")
        employee_data_module.set_privacy_level(PrivacyLevel.HIGH)
        high_data = employee_data_module.get_data_for_rendering()
        print(json.dumps(high_data, ensure_ascii=False, indent=2))

        # 3. Обновление данных
        print("\n=== Updating employee data ===")
        employee_data_module.update_data({
            "full_name": "Петрова Анна Владимировна",
            "position": "Руководитель проекта",
            "contact": {
                "email": "anna.petrova@company.ru",
                "telegram": "@anna_teamlead"
            }
        })

        # 4. Получение обновленных данных
        print("\n=== Updated data (High privacy) ===")
        updated_data = employee_data_module.get_data_for_rendering()
        print(json.dumps(updated_data, ensure_ascii=False, indent=2))

        # 5. Полные данные
        print("\n=== Full employee data ===")
        full_employee = employee_data_module.get_full_data()
        full_data_dict = {
            "full_name": full_employee.full_name,
            "position": full_employee.position,
            "company": full_employee.company,
            "department": full_employee.department,
            "office_location": full_employee.office_location,
            "contact": {
                "email": full_employee.contact.email,
                "telegram": full_employee.contact.telegram
            },
            "branding": {
                "logo_url": full_employee.branding.logo_url,
                "corporate_colors": {
                    "primary": full_employee.branding.corporate_colors.primary,
                    "secondary": full_employee.branding.corporate_colors.secondary
                },
                "slogan": full_employee.branding.slogan
            },
            "privacy_level": full_employee.privacy_level.value
        }
        print(json.dumps(full_data_dict, ensure_ascii=False, indent=2))

        # 6. QR-коды отдельно
        print("\n=== QR codes data ===")
        qr_data = employee_data_module.generate_contact_qr_data()
        if qr_data:
            # Убираем base64 данные для читаемости вывода
            qr_display_data = {
                "has_telegram": qr_data['has_telegram'],
                "telegram_url": qr_data.get('telegram_url'),
                "contact_text_preview": qr_data.get('contact_text', '')[:100] + "..." if qr_data.get(
                    'contact_text') else None,
                "has_telegram_qr": bool(qr_data.get('telegram_qr_base64')),
                "has_contact_qr": bool(qr_data.get('contact_qr_base64'))
            }
            print(json.dumps(qr_display_data, ensure_ascii=False, indent=2))

            # Сохранение QR-кодов в файлы
            if qr_data.get('telegram_qr_base64'):
                with open('telegram_qr.png', 'wb') as f:
                    f.write(base64.b64decode(qr_data['telegram_qr_base64']))

            if qr_data.get('contact_qr_base64'):
                with open('contact_qr.png', 'wb') as f:
                    f.write(base64.b64decode(qr_data['contact_qr_base64']))

        # 7. Экспорт в файл
        employee_data_module.export_to_json("updated_employee_data.json")

    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    demonstrate_module()
