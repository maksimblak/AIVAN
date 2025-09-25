"""
Модуль обезличивания (анонимизации) документов
Удаление персональных данных из документов для безопасного обмена
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Any

from .base import DocumentProcessor, DocumentResult, ProcessingError
from .utils import FileFormatHandler, TextProcessor

logger = logging.getLogger(__name__)


class DocumentAnonymizer(DocumentProcessor):
    """Класс для обезличивания персональных данных в документах"""

    def __init__(self):
        super().__init__(name="DocumentAnonymizer", max_file_size=50 * 1024 * 1024)
        self.supported_formats = [".pdf", ".docx", ".doc", ".txt"]
        self.anonymization_map: dict[str, str] = {}

    async def process(
        self,
        file_path: str | Path,
        anonymization_mode: str = "replace",
        exclude_types: list[str] | None = None,
        **kwargs,
    ) -> DocumentResult:
        """
        Обезличивание документа

        Args:
            file_path: путь к файлу
            anonymization_mode: режим обезличивания ("replace", "mask", "remove")
        """

        # Извлекаем текст из файла
        success, text = await FileFormatHandler.extract_text_from_file(file_path)
        if not success:
            raise ProcessingError(f"Не удалось извлечь текст: {text}", "EXTRACTION_ERROR")

        cleaned_text = TextProcessor.clean_text(text)
        if not cleaned_text.strip():
            raise ProcessingError("Документ не содержит текста", "EMPTY_DOCUMENT")

        self.anonymization_map = {}
        exclude_set = {item.lower() for item in (exclude_types or [])}

        # Обезличиваем текст
        anonymized_text, report = self._anonymize_text(
            cleaned_text, anonymization_mode, exclude_set
        )
        report["excluded_types"] = sorted(exclude_set) if exclude_set else []

        result_data = {
            "anonymized_text": anonymized_text,
            "anonymization_report": report,
            "anonymization_map": self.anonymization_map.copy(),
            "original_file": str(file_path),
            "mode": anonymization_mode,
        }

        return DocumentResult.success_result(
            data=result_data, message="Обезличивание документа успешно завершено"
        )

    def _anonymize_text(
        self, text: str, mode: str, exclude: set[str] | None = None
    ) -> tuple[str, dict[str, Any]]:
        """Основная функция обезличивания"""
        anonymized_text = text
        report: dict[str, Any] = {"processed_items": [], "statistics": {}}
        exclude_set = {item.lower() for item in (exclude or set())}

        def _should_process(kind: str) -> bool:
            return kind not in exclude_set

        names_count = phones_count = emails_count = addresses_count = documents_count = (
            bank_details_count
        ) = 0

        if _should_process("names"):
            anonymized_text, names_count = self._anonymize_names(anonymized_text, mode)
        if _should_process("phones"):
            anonymized_text, phones_count = self._anonymize_phones(anonymized_text, mode)
        if _should_process("emails"):
            anonymized_text, emails_count = self._anonymize_emails(anonymized_text, mode)
        if _should_process("addresses"):
            anonymized_text, addresses_count = self._anonymize_addresses(anonymized_text, mode)
        if _should_process("documents"):
            anonymized_text, documents_count = self._anonymize_documents(anonymized_text, mode)
        if _should_process("bank_details"):
            anonymized_text, bank_details_count = self._anonymize_bank_details(
                anonymized_text, mode
            )

        report["statistics"] = {
            "names": names_count,
            "phones": phones_count,
            "emails": emails_count,
            "addresses": addresses_count,
            "documents": documents_count,
            "bank_details": bank_details_count,
        }

        return anonymized_text, report

    def _anonymize_names(self, text: str, mode: str) -> tuple[str, int]:
        """Обезличивание ФИО"""
        # Простой паттерн для русских имен
        name_pattern = r"\b[А-ЯЁ][а-яё]+\s+[А-ЯЁ][а-яё]+(?:\s+[А-ЯЁ][а-яё]+)?\b"
        matches = re.findall(name_pattern, text)

        count = 0
        for match in matches:
            replacement = self._get_replacement(match, "Лицо", mode)
            text = text.replace(match, replacement)
            count += 1

        return text, count

    def _anonymize_phones(self, text: str, mode: str) -> tuple[str, int]:
        """Обезличивание номеров телефонов"""
        phone_patterns = [
            r"\+7\s*\(\d{3}\)\s*\d{3}-\d{2}-\d{2}",
            r"\+7\s*\d{3}\s*\d{3}\s*\d{2}\s*\d{2}",
            r"8\s*\(\d{3}\)\s*\d{3}-\d{2}-\d{2}",
            r"8\s*\d{3}\s*\d{3}\s*\d{2}\s*\d{2}",
        ]

        count = 0
        for pattern in phone_patterns:
            matches = re.findall(pattern, text)
            for match in matches:
                replacement = self._get_replacement(match, "Телефон", mode)
                text = text.replace(match, replacement)
                count += 1

        return text, count

    def _anonymize_emails(self, text: str, mode: str) -> tuple[str, int]:
        """Обезличивание email адресов"""
        email_pattern = r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b"
        matches = re.findall(email_pattern, text)

        count = 0
        for match in matches:
            replacement = self._get_replacement(match, "Email", mode)
            text = text.replace(match, replacement)
            count += 1

        return text, count

    def _anonymize_addresses(self, text: str, mode: str) -> tuple[str, int]:
        """Обезличивание адресов"""
        # Простые паттерны для адресов
        address_patterns = [
            r"г\.\s*[А-ЯЁ][а-яё]+",  # г. Москва
            r"ул\.\s*[А-ЯЁ][а-яё\s]+",  # ул. Ленина
            r"д\.\s*\d+",  # д. 10
            r"\d{6}",  # почтовый индекс
        ]

        count = 0
        for pattern in address_patterns:
            matches = re.findall(pattern, text)
            for match in matches:
                replacement = self._get_replacement(match, "Адрес", mode)
                text = text.replace(match, replacement)
                count += 1

        return text, count

    def _anonymize_documents(self, text: str, mode: str) -> tuple[str, int]:
        """Обезличивание номеров документов"""
        doc_patterns = [
            r"\b\d{4}\s*\d{6}\b",  # паспорт
            r"\b\d{11}\b",  # СНИЛС
            r"\b\d{10,12}\b",  # ИНН
        ]

        count = 0
        for pattern in doc_patterns:
            matches = re.findall(pattern, text)
            for match in matches:
                replacement = self._get_replacement(match, "Документ", mode)
                text = text.replace(match, replacement)
                count += 1

        return text, count

    def _anonymize_bank_details(self, text: str, mode: str) -> tuple[str, int]:
        """Обезличивание банковских реквизитов"""
        bank_patterns = [
            r"\b\d{20}\b",  # расчетный счет
            r"\b\d{9}\b",  # БИК
        ]

        count = 0
        for pattern in bank_patterns:
            matches = re.findall(pattern, text)
            for match in matches:
                replacement = self._get_replacement(match, "БанкРеквизит", mode)
                text = text.replace(match, replacement)
                count += 1

        return text, count

    def _get_replacement(self, original: str, data_type: str, mode: str) -> str:
        """Получить замену для персональных данных"""
        if original in self.anonymization_map:
            return self.anonymization_map[original]

        if mode == "remove":
            replacement = "[УДАЛЕНО]"
        elif mode == "mask":
            replacement = "*" * len(original)
        else:  # replace mode
            if data_type not in [
                "Лицо",
                "Организация",
                "Email",
                "Телефон",
                "Адрес",
                "Документ",
                "БанкРеквизит",
            ]:
                data_type = "Данные"

            # Считаем количество уже замененных элементов этого типа
            existing_count = sum(
                1 for v in self.anonymization_map.values() if v.startswith(data_type)
            )
            replacement = f"[{data_type}-{existing_count + 1}]"

        self.anonymization_map[original] = replacement
        return replacement

    def restore_data(
        self, anonymized_text: str, restoration_key: dict[str, str] | None = None
    ) -> str:
        """Восстановление данных из обезличенного текста"""
        if not restoration_key:
            restoration_key = {v: k for k, v in self.anonymization_map.items()}

        restored_text = anonymized_text
        for anonymized, original in restoration_key.items():
            restored_text = restored_text.replace(anonymized, original)

        return restored_text
