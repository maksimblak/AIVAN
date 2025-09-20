#!/usr/bin/env python3
"""
🤖 Запуск ИИ-Иван (максимально простая версия)
Только /start и вопросы - никаких кнопок и меню
"""

import os
import sys
import asyncio
from pathlib import Path

# Добавляем путь к проекту
sys.path.insert(0, str(Path(__file__).parent))

def check_requirements():
    """Проверяет наличие необходимых библиотек"""
    required_packages = [
        'aiogram',
        'openai', 
        'httpx',
        'python-dotenv'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package.replace('-', '_'))
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        print("❌ Отсутствуют необходимые пакеты:")
        for package in missing_packages:
            print(f"   • {package}")
        print("\n📦 Установите их командой:")
        print(f"   pip install {' '.join(missing_packages)}")
        return False
    
    return True

def check_env_variables():
    """Проверяет переменные окружения"""
    from dotenv import load_dotenv
    load_dotenv()
    
    required_vars = {
        'TELEGRAM_BOT_TOKEN': 'Токен Telegram бота',
        'OPENAI_API_KEY': 'API ключ OpenAI'
    }
    
    missing_vars = []
    
    for var, description in required_vars.items():
        if not os.getenv(var):
            missing_vars.append((var, description))
    
    if missing_vars:
        print("❌ Отсутствуют обязательные переменные окружения:")
        for var, desc in missing_vars:
            print(f"   • {var} - {desc}")
        
        print("\n📝 Создайте .env файл с содержимым:")
        print("TELEGRAM_BOT_TOKEN=your_bot_token_here")
        print("OPENAI_API_KEY=your_openai_key_here")
        return False
    
    return True

def show_banner():
    """Показывает баннер"""
    banner = """
╔════════════════════════════════════════╗
║      🤖 ИИ-Иван - Простая версия       ║
╠════════════════════════════════════════╣
║  Максимально простой интерфейс:        ║
║  /start → отправить вопрос → получить  ║
║  ответ с анализом судебной практики    ║
╚════════════════════════════════════════╝
"""
    print(banner)

def show_interface():
    """Показывает описание интерфейса"""
    print("💡 Как пользоваться:")
    print("   1. Отправьте боту /start")
    print("   2. Напишите ваш юридический вопрос")
    print("   3. Получите структурированный ответ с практикой")
    print("")
    print("🎯 Особенности:")
    print("   • Никаких кнопок - только текст и команды")  
    print("   • Красивый дизайн с рамками и эмодзи")
    print("   • Анимированные статусы загрузки")
    print("   • Структурированные ответы с примерами")
    print("   • Анализ российского права и судебной практики")
    print("   • Автоматический поиск релевантных дел")

def show_config():
    """Показывает конфигурацию"""
    from dotenv import load_dotenv
    load_dotenv()
    
    print("\n⚙️ Конфигурация:")
    
    model = os.getenv('OPENAI_MODEL', 'gpt-5-mini')
    print(f"   🤖 OpenAI модель: {model}")
    
    animation = os.getenv('USE_STATUS_ANIMATION', '1')
    status_type = "анимированные" if animation == '1' else "прогресс-бар"
    print(f"   ✨ Статусы: {status_type}")
    
    web_search = "отключен" if os.getenv('DISABLE_WEB', '0') == '1' else "включен"
    print(f"   🔍 Веб-поиск: {web_search}")
    
    proxy = os.getenv('TELEGRAM_PROXY_URL')
    if proxy:
        clean_proxy = proxy.split('@')[-1] if '@' in proxy else proxy
        print(f"   🌐 Прокси: {clean_proxy}")
    else:
        print("   🌐 Прокси: не используется")

async def run_bot():
    """Запускает простого бота"""
    try:
        print("\n🚀 Запускаю простого бота...")
        
        from src.core.main_simple import main
        await main()
        
    except KeyboardInterrupt:
        print("\n👋 Бот остановлен пользователем")
    except Exception as e:
        print(f"\n💥 Ошибка запуска: {e}")
        return False
    
    return True

def main():
    """Основная функция"""
    show_banner()
    show_interface()
    
    print("\n🔍 Проверяю зависимости...")
    
    if not check_requirements():
        print("\n❌ Установите недостающие пакеты и повторите запуск")
        return 1
    
    print("✅ Все пакеты установлены")
    
    if not check_env_variables():
        print("\n❌ Настройте переменные окружения и повторите запуск")
        return 1
    
    print("✅ Переменные окружения настроены")
    
    show_config()
    
    try:
        # Пытаемся использовать uvloop для производительности
        try:
            import uvloop
            uvloop.install()
            print("🚀 Использую uvloop для повышенной производительности")
        except ImportError:
            print("⚡ uvloop не найден, использую стандартный event loop")
        
        asyncio.run(run_bot())
        
    except KeyboardInterrupt:
        print("\n👋 До свидания!")
        return 0
    except Exception as e:
        print(f"\n💥 Критическая ошибка: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())
