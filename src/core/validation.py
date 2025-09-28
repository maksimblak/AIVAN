"""
Модуль валидации входных данных для безопасности и корректности обработки
"""

from __future__ import annotations

import html
import re
from dataclasses import dataclass
from enum import Enum
from typing import Any


class ValidationError(Exception):
    """Исключение валидации входных данных"""

    pass


class ValidationSeverity(Enum):
    """Уровень критичности ошибки валидации"""

    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


@dataclass
class ValidationResult:
    """Результат валидации"""

    is_valid: bool
    cleaned_data: str | None = None
    errors: list[str] | None = None
    warnings: list[str] | None = None
    severity: ValidationSeverity = ValidationSeverity.ERROR

    def __post_init__(self):
        if self.errors is None:
            self.errors = []
        if self.warnings is None:
            self.warnings = []


class InputValidator:
    """Валидатор пользовательского ввода"""

    # Константы валидации
    MAX_MESSAGE_LENGTH = 8000
    MAX_QUESTION_LENGTH = 4000
    MIN_QUESTION_LENGTH = 10

    # Паттерны безопасности
    SUSPICIOUS_PATTERNS = [
        r"(?i)<script.*?>",  # JavaScript
        r"(?i)javascript:",  # JavaScript протокол
        r"(?i)data:text/html",  # Data URLs
        r"(?i)vbscript:",  # VBScript
        r"(?i)on\w+\s*=",  # Event handlers
        r"(?i)<iframe.*?>",  # iframes
        r"(?i)<object.*?>",  # objects
        r"(?i)<embed.*?>",  # embeds
        r"(?i)<link.*?>",  # external links
        r"(?i)<meta.*?>",  # meta tags
    ]

    # Паттерны для SQL инъекций (на всякий случай)
    SQL_INJECTION_PATTERNS = [
        r"(?i)\b(union|select|insert|update|delete|drop|create|alter|exec|execute)\b",
        r'(?i)[\'"]\s*;\s*--',
        r'(?i)[\'"]\s*or\s+[\'"]\w*[\'"]\s*=\s*[\'"]\w*[\'"]',
        r'(?i)[\'"]\s*or\s+\d+\s*=\s*\d+',
        r'(?i)[\'"]\s*;\s*(drop|delete|truncate)',
    ]

    # Паттерны чрезмерно длинных повторений
    SPAM_PATTERNS = [
        r"(.)\1{50,}",  # Один символ повторяется 50+ раз
        r"(\w{1,10})\1{10,}",  # Слово повторяется 10+ раз
        r"[!?]{10,}",  # Много восклицательных/вопросительных знаков
    ]

    # Запрещенный контент
    FORBIDDEN_PATTERNS = [
        r"(?i)\b(password|пароль|token|токен|api[_\s]?key|ключ[_\s]?апи)\b",
        r"(?i)\b(credit[_\s]?card|кредитн\w*[_\s]?карт\w*)\b",
        r"(?i)\b(социальн\w*[_\s]?безопасност\w*|social[_\s]?security)\b",
    ]

    @classmethod
    def validate_question(cls, text: str | None, user_id: int | None = None) -> ValidationResult:
        """
        Основная валидация юридического вопроса
        """
        if not text:
            return ValidationResult(is_valid=False, errors=["Текст вопроса не может быть пустым"])

        # Базовые проверки
        text = text.strip()

        # Проверка длины
        if len(text) < cls.MIN_QUESTION_LENGTH:
            return ValidationResult(
                is_valid=False,
                errors=[f"Вопрос слишком короткий (минимум {cls.MIN_QUESTION_LENGTH} символов)"],
            )

        if len(text) > cls.MAX_QUESTION_LENGTH:
            return ValidationResult(
                is_valid=False,
                errors=[f"Вопрос слишком длинный (максимум {cls.MAX_QUESTION_LENGTH} символов)"],
            )

        result = ValidationResult(is_valid=True, cleaned_data=text)

        # Проверка на подозрительный контент
        cls._check_security_patterns(text, result)

        # Проверка на спам
        cls._check_spam_patterns(text, result)

        # Проверка на запрещенный контент
        cls._check_forbidden_content(text, result)

        # Санитизация
        result.cleaned_data = cls._sanitize_text(text)

        # Если есть критические ошибки, отклоняем
        if any(error for error in result.errors if "критично" in error.lower()):
            result.is_valid = False
            result.severity = ValidationSeverity.CRITICAL

        return result

    @classmethod
    def validate_user_id(cls, user_id: Any) -> ValidationResult:
        """Валидация user_id"""
        if user_id in (None, ""):
            return ValidationResult(is_valid=False, errors=["User ID не может быть пустым"])

        try:
            uid = int(user_id)
        except (ValueError, TypeError):
            return ValidationResult(is_valid=False, errors=["User ID должен быть числом"])

        if uid <= 0:
            return ValidationResult(is_valid=False, errors=["User ID должен быть положительным числом"])

        max_allowed = 9_223_372_036_854_775_807  # 2**63 - 1
        if uid > max_allowed:
            return ValidationResult(is_valid=False, errors=["User ID слишком велик"])

        return ValidationResult(is_valid=True, cleaned_data=str(uid))

    @classmethod
    def validate_payment_amount(cls, amount: Any, currency: str = "RUB") -> ValidationResult:
        """Валидация суммы платежа"""
        try:
            amt = float(amount)

            if amt <= 0:
                return ValidationResult(
                    is_valid=False, errors=["Сумма платежа должна быть больше нуля"]
                )

            # Лимиты по валютам
            limits = {
                "RUB": {"min": 1, "max": 100000},
                "XTR": {"min": 1, "max": 10000},
                "USDT": {"min": 0.01, "max": 1000},
            }

            if currency in limits:
                min_amt = limits[currency]["min"]
                max_amt = limits[currency]["max"]

                if amt < min_amt or amt > max_amt:
                    return ValidationResult(
                        is_valid=False,
                        errors=[f"Сумма должна быть между {min_amt} и {max_amt} {currency}"],
                    )

            return ValidationResult(is_valid=True, cleaned_data=str(amt))

        except (ValueError, TypeError):
            return ValidationResult(is_valid=False, errors=["Сумма должна быть числом"])

    @classmethod
    def _check_security_patterns(cls, text: str, result: ValidationResult) -> None:
        """Проверка на подозрительные паттерны безопасности"""
        for pattern in cls.SUSPICIOUS_PATTERNS:
            if re.search(pattern, text, re.IGNORECASE | re.MULTILINE):
                result.errors.append("Обнаружен подозрительный контент (критично)")
                result.severity = ValidationSeverity.CRITICAL
                return

        # Проверка SQL инъекций
        for pattern in cls.SQL_INJECTION_PATTERNS:
            if re.search(pattern, text):
                result.warnings.append("Обнаружены подозрительные SQL-паттерны")
                break

    @classmethod
    def _check_spam_patterns(cls, text: str, result: ValidationResult) -> None:
        """Проверка на спам-паттерны"""
        for pattern in cls.SPAM_PATTERNS:
            if re.search(pattern, text):
                result.warnings.append("Обнаружены признаки спама")
                break

    @classmethod
    def _check_forbidden_content(cls, text: str, result: ValidationResult) -> None:
        """Проверка на запрещенный контент"""
        for pattern in cls.FORBIDDEN_PATTERNS:
            if re.search(pattern, text):
                result.errors.append("Обнаружен запрещенный контент (персональные данные)")
                result.severity = ValidationSeverity.ERROR
                break

    @classmethod
    def _sanitize_text(cls, text: str) -> str:
        """Санитизация текста"""
        # HTML escape
        text = html.escape(text)

        # Удаление лишних пробелов
        text = re.sub(r"\s+", " ", text).strip()

        # Удаление невидимых символов
        text = re.sub(r"[\u200b-\u200f\u2028-\u202f\u205f-\u206f\ufeff]", "", text)

        return text

    @classmethod
    def validate_config_value(cls, key: str, value: Any, expected_type: type) -> ValidationResult:
        """Валидация конфигурационных значений"""
        try:
            if expected_type == bool:
                if isinstance(value, str):
                    cleaned_bool = value.lower() in ("true", "1", "yes", "on")
                else:
                    cleaned_bool = bool(value)
            elif expected_type == int:
                cleaned_int = int(value)
                if cleaned_int < 0:
                    return ValidationResult(
                        is_valid=False, errors=[f"Значение {key} должно быть неотрицательным"]
                    )
            elif expected_type == float:
                cleaned_float = float(value)
                if cleaned_float < 0:
                    return ValidationResult(
                        is_valid=False, errors=[f"Значение {key} должно быть неотрицательным"]
                    )
            elif expected_type == str:
                cleaned_str = str(value).strip()
                if not cleaned_str:
                    return ValidationResult(
                        is_valid=False, errors=[f"Значение {key} не может быть пустым"]
                    )
            else:
                cleaned_value = expected_type(value)

            # Возвращаем правильное значение в зависимости от типа
            if expected_type == bool:
                return ValidationResult(is_valid=True, cleaned_data=str(cleaned_bool))
            elif expected_type == int:
                return ValidationResult(is_valid=True, cleaned_data=str(cleaned_int))
            elif expected_type == float:
                return ValidationResult(is_valid=True, cleaned_data=str(cleaned_float))
            elif expected_type == str:
                return ValidationResult(is_valid=True, cleaned_data=cleaned_str)
            else:
                return ValidationResult(is_valid=True, cleaned_data=str(cleaned_value))

        except (ValueError, TypeError) as e:
            return ValidationResult(
                is_valid=False, errors=[f"Неверный тип значения для {key}: {str(e)}"]
            )


# Декоратор для автоматической валидации
def validate_input(validation_func):
    """Декоратор для автоматической валидации входных данных"""

    def decorator(func):
        async def wrapper(*args, **kwargs):
            # Предполагаем что первый аргумент - это данные для валидации
            if args:
                validation_result = validation_func(args[0])
                if not validation_result.is_valid:
                    raise ValidationError(
                        f"Ошибка валидации: {', '.join(validation_result.errors)}"
                    )
                # Заменяем данные на очищенные
                args = (validation_result.cleaned_data,) + args[1:]
            return await func(*args, **kwargs)

        return wrapper

    return decorator
