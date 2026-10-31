"""Grand Mobile Support Copilot package."""
import os

__version__ = "1.0.0"

# .env читается один раз при импорте пакета (override=False: реальные переменные
# окружения имеют приоритет). Файл полностью опционален — сервис работает без него.
try:  # pragma: no cover - зависит от наличия .env рядом с проектом
    from dotenv import load_dotenv

    load_dotenv(dotenv_path=os.getenv("ENV_FILE", ".env"), override=False)
except Exception:  # pragma: no cover - python-dotenv может отсутствовать в окружении
    pass
