# Цифровой дресс-код: фон, который выделяет вас

Создайте локальный ML-модуль сегментации видео и генератор персонализированных фонов.

Требования
- ОС: Linux / macOS / Windows
- Python: 3.12.7
- Node.JS: 10.9.4
- CUDA 11.x/12.x + драйверы (опционально для GPU). CPU-поддержка есть.
- Объём RAM: 8+ GB (рекомендуется 16+ GB)

Быстрый чеклист для судей (минимум для проверки)
1. Клонировать репозиторий:
   ```bash
   git clone https://github.com/Vik0t/human-segmentation-with-personal-backgrounds.git
   cd human-segmentation-with-personal-backgrounds
   ```

3. Создать и активировать виртуальное окружение:
   ```bash
   python -m venv .venv
   ```
   # Linux / macOS
   ```bash
   source .venv/bin/activate
   ```
   # Windows (PowerShell)
   ```bash
   .venv\Scripts\Activate.ps1
   ```

4. Установить зависимости:
   ```bash
   pip install -r requirements.txt
   cd frontend/
   npm i
   ```

5. Запуск:
   ```bash
   npm run build
   npm run start
   ```
