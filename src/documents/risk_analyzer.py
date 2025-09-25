"""
Модуль анализа рисков и проверки договоров
Выявление ошибок, несоответствий законодательству и скрытых рисков в документах
"""

from __future__ import annotations

import logging
import re
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any

from .base import DocumentProcessor, DocumentResult, ProcessingError
from .utils import FileFormatHandler, TextProcessor

logger = logging.getLogger(__name__)


class RiskLevel(Enum):
    """Уровни критичности рисков"""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


RISK_ANALYSIS_PROMPT = """
Ты — эксперт по анализу рисков в юридических документах и договорах.

Твоя задача — провести комплексную проверку документа и выявить все потенциальные риски.

ОБЛАСТИ АНАЛИЗА:

1. **Соответствие законодательству РФ**
   - Проверка на соответствие ГК РФ
   - Соответствие отраслевому законодательству
   - Актуальность правовых норм

2. **Анализ условий договора**
   - Баланс прав и обязанностей сторон
   - Корректность расчета неустойки
   - Условия расторжения договора
   - Автоматическое продление

3. **Финансовые риски**
   - Скрытые платежи и комиссии
   - Некорректные расчеты
   - Валютные риски
   - Индексация цен

4. **Процедурные риски**
   - Неточные сроки
   - Некорректные процедуры уведомления
   - Спорные формулировки
   - Отсутствие важных пунктов

СТРУКТУРА ОТВЕТА:

**Критичные риски (ВЫСОКИЙ/КРИТИЧЕСКИЙ уровень):**
- [Описание риска]
- Потенциальные последствия: [последствия]
- Рекомендация: [как устранить]
- Ссылка на закон: [если применимо]

**Средние риски:**
- [аналогично]

**Низкие риски:**
- [аналогично]

**Рекомендуемые корректировки:**
1. [Конкретная рекомендация с исправленной формулировкой]
2. [Следующая рекомендация]

**Общая оценка документа:**
- Уровень риска: [НИЗКИЙ/СРЕДНИЙ/ВЫСОКИЙ/КРИТИЧЕСКИЙ]
- Рекомендация к подписанию: [ДА/НЕТ/С ДОРАБОТКАМИ]

ВАЖНО:
- Используй только информацию из документа
- Указывай конкретные пункты с проблемами
- Предлагай конкретные исправления
- Ссылайся на релевантные статьи ГК РФ
"""


class RiskAnalyzer(DocumentProcessor):
    """Класс для анализа рисков в договорах и документах"""

    def __init__(self, openai_service=None):
        super().__init__(name="RiskAnalyzer", max_file_size=50 * 1024 * 1024)  # 50MB
        self.supported_formats = [".pdf", ".docx", ".doc", ".txt"]
        self.openai_service = openai_service

        # Предопределенные паттерны рисков
        self.risk_patterns = self._initialize_risk_patterns()

    def _initialize_risk_patterns(self) -> dict[str, list[dict[str, Any]]]:
        """Инициализация паттернов рисков"""
        return {
            "automatic_renewal": [
                {
                    "pattern": r"автоматическ(?:и|ое)\s+(?:продлен|возобновлен)",
                    "risk_level": RiskLevel.HIGH,
                    "description": "Автоматическое продление договора без уведомления",
                    "recommendation": "Добавить четкий механизм уведомления о продлении",
                }
            ],
            "hidden_fees": [
                {
                    "pattern": r"дополнительн(?:ые|ая)\s+(?:плат|комис|сбор)",
                    "risk_level": RiskLevel.MEDIUM,
                    "description": "Возможные скрытые платежи или комиссии",
                    "recommendation": "Точно определить все возможные доплаты",
                }
            ],
            "penalty_issues": [
                {
                    "pattern": r"неустойка.*?(?:\d+\s*%|\d+\s*руб)",
                    "risk_level": RiskLevel.MEDIUM,
                    "description": "Условия неустойки требуют проверки",
                    "recommendation": "Проверить соответствие размера неустойки законодательству",
                }
            ],
            "unclear_terms": [
                {
                    "pattern": r"в\s+разумн(?:ые|ый)\s+срок|в\s+кратчайш(?:ие|ий)\s+срок",
                    "risk_level": RiskLevel.MEDIUM,
                    "description": "Неопределенные временные рамки",
                    "recommendation": "Установить конкретные сроки исполнения",
                }
            ],
            "unbalanced_rights": [
                {
                    "pattern": r"(?:заказчик|клиент|покупатель).*?(?:не\s+несет\s+ответственност|освобождается)",
                    "risk_level": RiskLevel.HIGH,
                    "description": "Дисбаланс прав и обязанностей сторон",
                    "recommendation": "Пересмотреть распределение ответственности",
                }
            ],
        }

    async def process(
        self, file_path: str | Path, custom_criteria: list[str] | None = None, **kwargs
    ) -> DocumentResult:
        """
        Основной метод анализа рисков

        Args:
            file_path: путь к файлу
            custom_criteria: пользовательские критерии недопустимых рисков
            **kwargs: дополнительные параметры
        """

        if not self.openai_service:
            raise ProcessingError("OpenAI сервис не инициализирован", "SERVICE_ERROR")

        # Извлекаем текст из файла
        success, text = await FileFormatHandler.extract_text_from_file(file_path)
        if not success:
            raise ProcessingError(f"Не удалось извлечь текст: {text}", "EXTRACTION_ERROR")

        # Очищаем и обрабатываем текст
        cleaned_text = TextProcessor.clean_text(text)
        if not cleaned_text.strip():
            raise ProcessingError("Документ не содержит текста", "EMPTY_DOCUMENT")

        # Проводим автоматическую проверку по паттернам
        pattern_risks = self._analyze_by_patterns(cleaned_text)

        # Проводим AI-анализ
        ai_analysis = await self._ai_risk_analysis(cleaned_text, custom_criteria)

        # Проверяем соответствие законодательству
        legal_compliance = await self._check_legal_compliance(cleaned_text)

        # Объединяем результаты
        ai_risks = ai_analysis.get("risks", [])
        combined_risks = pattern_risks + ai_risks
        highlighted_text = self.highlight_problematic_clauses(cleaned_text, combined_risks)

        result_data = {
            "overall_risk_level": self._calculate_overall_risk(combined_risks),
            "pattern_risks": pattern_risks,
            "ai_analysis": ai_analysis,
            "legal_compliance": legal_compliance,
            "recommendations": self._generate_recommendations(pattern_risks, ai_analysis),
            "highlighted_text": highlighted_text,
            "original_file": str(file_path),
            "analysis_timestamp": datetime.now().isoformat(),
        }

        return DocumentResult.success_result(
            data=result_data, message="Анализ рисков успешно завершен"
        )

    def _analyze_by_patterns(self, text: str) -> list[dict[str, Any]]:
        """Анализ документа по предопределенным паттернам рисков"""
        found_risks = []

        for risk_category, patterns in self.risk_patterns.items():
            for pattern_info in patterns:
                matches = re.finditer(pattern_info["pattern"], text, re.IGNORECASE | re.DOTALL)

                for match in matches:
                    context_start = max(0, match.start() - 100)
                    context_end = min(len(text), match.end() + 100)
                    context = text[context_start:context_end].strip()

                    risk = {
                        "category": risk_category,
                        "risk_level": pattern_info["risk_level"].value,
                        "description": pattern_info["description"],
                        "recommendation": pattern_info["recommendation"],
                        "context": context,
                        "position": match.start(),
                        "matched_text": match.group(),
                    }

                    found_risks.append(risk)

        return found_risks

    async def _ai_risk_analysis(
        self, text: str, custom_criteria: list[str] | None = None
    ) -> dict[str, Any]:
        """AI-анализ рисков с помощью OpenAI"""

        try:
            # Добавляем пользовательские критерии в промпт
            prompt = RISK_ANALYSIS_PROMPT
            if custom_criteria:
                criteria_text = "\n".join(f"- {criterion}" for criterion in custom_criteria)
                prompt += f"\n\nДОПОЛНИТЕЛЬНЫЕ КРИТЕРИИ ПРОВЕРКИ:\n{criteria_text}"

            # Если документ слишком длинный, обрабатываем частями
            if len(text) > 8000:
                chunks = TextProcessor.split_into_chunks(text, max_chunk_size=6000)
                chunk_analyses = []

                for i, chunk in enumerate(chunks):
                    logger.info(f"Анализирую часть {i+1}/{len(chunks)}")

                    result = await self.openai_service.ask_legal(
                        system_prompt=prompt,
                        user_message=f"Часть {i+1} из {len(chunks)} документа:\n\n{chunk}",
                    )

                    if result.get("ok"):
                        chunk_analyses.append(
                            {
                                "chunk_number": i + 1,
                                "analysis": result.get("text", ""),
                                "chunk_text": chunk[:200] + "..." if len(chunk) > 200 else chunk,
                            }
                        )

                if chunk_analyses:
                    # Объединяем анализы частей
                    combined_analysis = "\n\n".join([ca["analysis"] for ca in chunk_analyses])

                    # Создаем итоговый анализ
                    final_prompt = (
                        """
                    Объедини следующие анализы рисков частей документа в единый структурированный отчет:

                    """
                        + combined_analysis
                    )

                    final_result = await self.openai_service.ask_legal(
                        system_prompt=RISK_ANALYSIS_PROMPT, user_message=final_prompt
                    )

                    if final_result.get("ok"):
                        return {
                            "analysis": final_result.get("text", ""),
                            "method": "chunked_analysis",
                            "chunks_analyzed": len(chunks),
                            "chunk_details": chunk_analyses,
                            "risks": self._parse_risks_from_ai_text(final_result.get("text", "")),
                        }
            else:
                # Анализируем документ целиком
                result = await self.openai_service.ask_legal(
                    system_prompt=prompt, user_message=text
                )

                if result.get("ok"):
                    analysis_text = result.get("text", "")
                    return {
                        "analysis": analysis_text,
                        "method": "single_analysis",
                        "chunks_analyzed": 1,
                        "risks": self._parse_risks_from_ai_text(analysis_text),
                    }

            raise ProcessingError("Не удалось провести AI-анализ", "AI_ANALYSIS_ERROR")

        except Exception as e:
            logger.error(f"Ошибка AI-анализа: {e}")
            return {
                "analysis": f"Ошибка анализа: {str(e)}",
                "method": "error",
                "chunks_analyzed": 0,
                "risks": [],
            }

    def _parse_risks_from_ai_text(self, analysis_text: str) -> list[dict[str, Any]]:
        """Парсинг рисков из текста AI-анализа"""
        risks = []

        # Простой парсинг - можно улучшить регулярными выражениями
        sections = ["критичные риски", "средние риски", "низкие риски"]

        for section in sections:
            section_pattern = rf"{section}.*?(?=(?:{'|'.join(sections)}|рекомендуемые корректировки|общая оценка|$))"
            section_match = re.search(section_pattern, analysis_text, re.IGNORECASE | re.DOTALL)

            if section_match:
                section_text = section_match.group()
                # Извлекаем отдельные риски из секции
                risk_items = re.findall(r"-\s*([^-\n]+(?:\n(?!\s*-)[^\n]*)*)", section_text)

                risk_level = RiskLevel.MEDIUM  # по умолчанию
                if "критичные" in section:
                    risk_level = RiskLevel.CRITICAL
                elif "низкие" in section:
                    risk_level = RiskLevel.LOW

                for risk_text in risk_items:
                    if risk_text.strip():
                        risks.append(
                            {
                                "risk_level": risk_level.value,
                                "description": risk_text.strip(),
                                "source": "ai_analysis",
                            }
                        )

        return risks

    def _extract_legal_violations(self, analysis_text: str) -> list[dict[str, Any]]:
        """Простое извлечение нарушений из текста анализа AI."""
        if not analysis_text:
            return []

        violations: list[dict[str, Any]] = []
        seen: set[str] = set()
        for raw_line in analysis_text.splitlines():
            line = raw_line.strip()
            if not line:
                continue
            if re.match(r"^(?:[-•*—]|\d+[.)])\s*", line):
                normalized = re.sub(r"^(?:[-•*—]|\d+[.)])\s*", "", line).strip()
                norm_key = normalized.lower()
                if not normalized or norm_key in seen:
                    continue
                seen.add(norm_key)
                ref_match = re.search(
                    r"(ст\.?\s*\d+[\w\-]*|article\s*\d+)", normalized, re.IGNORECASE
                )
                violations.append(
                    {"text": normalized, "reference": ref_match.group(1) if ref_match else None}
                )
        return violations

    async def _check_legal_compliance(self, text: str) -> dict[str, Any]:
        """Проверка соответствия законодательству"""

        compliance_prompt = """
        Проверь документ на соответствие основным требованиям российского законодательства:

        1. Гражданский кодекс РФ (общие положения о договорах)
        2. Специальные требования для данного типа договора
        3. Защита прав потребителей (если применимо)
        4. Антимонопольное законодательство
        5. Налоговое законодательство

        Укажи конкретные нарушения со ссылками на статьи законов.
        """

        try:
            result = await self.openai_service.ask_legal(
                system_prompt=compliance_prompt,
                user_message=text[:8000],  # Ограничиваем длину для проверки соответствия
            )

            if result.get("ok"):
                analysis_text = result.get("text", "")
                return {
                    "status": "completed",
                    "analysis": analysis_text,
                    "violations": self._extract_legal_violations(analysis_text),
                }
            else:
                return {
                    "status": "failed",
                    "analysis": "Не удалось провести проверку соответствия законодательству",
                    "violations": [],
                }

        except Exception as e:
            logger.error(f"Ошибка проверки законодательства: {e}")
            return {"status": "error", "analysis": f"Ошибка: {str(e)}", "violations": []}

    def _calculate_overall_risk(self, risks: list[dict[str, Any]]) -> str:
        """Рассчитать общий уровень риска"""
        if not risks:
            return RiskLevel.LOW.value

        risk_scores = {
            RiskLevel.LOW.value: 1,
            RiskLevel.MEDIUM.value: 2,
            RiskLevel.HIGH.value: 3,
            RiskLevel.CRITICAL.value: 4,
        }

        max_risk_score = max(risk_scores.get(risk.get("risk_level", "low"), 1) for risk in risks)

        if max_risk_score >= 4:
            return RiskLevel.CRITICAL.value
        elif max_risk_score >= 3:
            return RiskLevel.HIGH.value
        elif max_risk_score >= 2:
            return RiskLevel.MEDIUM.value
        else:
            return RiskLevel.LOW.value

    def _generate_recommendations(
        self, pattern_risks: list[dict[str, Any]], ai_analysis: dict[str, Any]
    ) -> list[str]:
        """Генерация рекомендаций по устранению рисков"""
        recommendations = []

        # Рекомендации на основе найденных паттернов
        for risk in pattern_risks:
            if risk.get("recommendation"):
                recommendations.append(risk["recommendation"])

        # Базовые рекомендации
        recommendations.extend(
            [
                "Обязательно проконсультироваться с юристом перед подписанием",
                "Проверить все реквизиты и полномочия контрагента",
                "Убедиться в актуальности всех ссылок на законодательство",
            ]
        )

        return list(set(recommendations))  # Убираем дубликаты

    def highlight_problematic_clauses(self, text: str, risks: list[dict[str, Any]]) -> str:
        """Подсветка проблемных пунктов в тексте"""
        highlighted_text = text

        for risk in risks:
            if "position" in risk and "matched_text" in risk:
                matched_text = risk["matched_text"]
                highlighted_text = highlighted_text.replace(
                    matched_text,
                    f"**[РИСК: {risk['risk_level'].upper()}]** {matched_text} **[/{risk['risk_level'].upper()}]**",
                )

        return highlighted_text
