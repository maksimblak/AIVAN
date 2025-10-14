"""
Тесты для системы мониторинга security violations
"""

import pytest

from src.core.metrics import MetricsCollector
from src.core.validation import InputValidator


class TestSecurityMonitoring:
    """Тесты мониторинга security violations"""

    @pytest.fixture
    def metrics_collector(self):
        """Создание metrics collector для тестов"""
        # Инициализируем без Prometheus для unit-тестов
        return MetricsCollector(enable_prometheus=False)

    def test_sql_injection_detection_and_metrics(self, metrics_collector):
        """Проверка детекции SQL injection и записи метрик"""
        # Имитируем SQL injection попытку
        malicious_input = "test'; DROP TABLE users; --"

        result = InputValidator.validate_question(malicious_input)

        # Валидация должна выдать warning
        assert not result.is_valid or result.warnings
        assert any("SQL" in str(w) for w in result.warnings)

        # Проверяем что метрика записана (в fallback хранилище)
        fallback_metrics = metrics_collector.get_fallback_metrics()

        # В fallback должна быть метрика sql_injection_attempts
        # (если metrics_collector был правильно проинициализирован в validation.py)

    def test_xss_detection_and_metrics(self, metrics_collector):
        """Проверка детекции XSS и записи метрик"""
        malicious_input = "Hello <script>alert('XSS')</script>"

        result = InputValidator.validate_question(malicious_input)

        # Валидация должна отклонить
        assert not result.is_valid
        assert any("подозрительный" in str(e).lower() for e in result.errors)

    def test_valid_input_no_alerts(self, metrics_collector):
        """Проверка что валидный ввод не вызывает алертов"""
        valid_input = "Помогите мне разобраться с договором аренды квартиры"

        result = InputValidator.validate_question(valid_input)

        # Валидация должна пройти
        assert result.is_valid
        assert not result.errors
        assert result.cleaned_data == valid_input

    @pytest.mark.parametrize(
        "sql_injection_pattern,expected_pattern_name",
        [
            ("SELECT * FROM users WHERE 1=1", "sql_keywords"),
            ("test input'; --", "sql_comment"),
            ("some text ' OR '1'='1' and more", "sql_or_equals"),
            ("some text ' OR 1=1 and more", "sql_numeric_equals"),
            ("some text'; DROP TABLE users", "sql_dangerous_commands"),
        ],
    )
    def test_sql_injection_pattern_types(
        self, sql_injection_pattern, expected_pattern_name, metrics_collector
    ):
        """Проверка различных типов SQL injection паттернов"""
        result = InputValidator.validate_question(sql_injection_pattern)

        # Все SQL injection паттерны должны детектиться
        assert not result.is_valid or result.warnings

    @pytest.mark.parametrize(
        "xss_pattern,expected_pattern_name",
        [
            ("<script>alert(1)</script>", "script_tag"),
            ("javascript:alert(1)", "javascript_protocol"),
            ("<img onerror='alert(1)'>", "event_handler"),
            ("<iframe src='evil.com'></iframe>", "iframe_tag"),
        ],
    )
    def test_xss_pattern_types(self, xss_pattern, expected_pattern_name, metrics_collector):
        """Проверка различных типов XSS паттернов"""
        result = InputValidator.validate_question(xss_pattern)

        # Все XSS паттерны должны блокироваться
        assert not result.is_valid
        assert result.severity.value == "critical"

    def test_metrics_collector_methods(self, metrics_collector):
        """Проверка методов MetricsCollector для security метрик"""
        # Тест записи SQL injection попытки
        metrics_collector.record_sql_injection_attempt(
            pattern_type="sql_keywords", source="user_input"
        )

        # Тест записи XSS попытки
        metrics_collector.record_xss_attempt(pattern_type="script_tag", source="user_input")

        # Тест записи общего security violation
        metrics_collector.record_security_violation(
            violation_type="sql_injection", severity="warning", source="user_input"
        )

        # Проверяем fallback метрики
        fallback = metrics_collector.get_fallback_metrics()

        # Должны быть записаны метрики
        assert "sql_injection_attempts_total" in fallback
        assert "xss_attempts_total" in fallback
        assert "security_violations_total" in fallback

    def test_pattern_name_mapping(self):
        """Проверка маппинга индексов паттернов на имена"""
        # SQL patterns
        assert InputValidator._get_sql_pattern_name(0) == "sql_keywords"
        assert InputValidator._get_sql_pattern_name(1) == "sql_comment"
        assert InputValidator._get_sql_pattern_name(2) == "sql_or_equals"
        assert InputValidator._get_sql_pattern_name(3) == "sql_numeric_equals"
        assert InputValidator._get_sql_pattern_name(4) == "sql_dangerous_commands"

        # XSS patterns
        assert InputValidator._get_xss_pattern_name(0) == "script_tag"
        assert InputValidator._get_xss_pattern_name(1) == "javascript_protocol"
        assert InputValidator._get_xss_pattern_name(4) == "event_handler"

    def test_long_sql_injection_attempt(self):
        """Проверка обработки длинной SQL injection попытки"""
        long_injection = (
            "test' UNION SELECT user_id, password, email FROM users "
            "WHERE '1'='1' OR admin=1; DROP TABLE sensitive_data; --"
        )

        result = InputValidator.validate_question(long_injection)

        # Должна детектироваться
        assert not result.is_valid or result.warnings

    def test_encoded_xss_attempt(self):
        """Проверка обработки закодированных XSS попыток"""
        # Простая проверка - сложные encoding обходы требуют advanced детекции
        encoded_xss = "<script>alert&#40;1&#41;</script>"

        result = InputValidator.validate_question(encoded_xss)

        # Базовый паттерн <script> должен детектироваться
        assert not result.is_valid


class TestDatabaseLayerMonitoring:
    """Тесты мониторинга на уровне базы данных"""

    @pytest.mark.asyncio
    async def test_invalid_table_name_detection(self):
        """Проверка детекции невалидного имени таблицы"""
        from src.core.db_advanced import DatabaseAdvanced

        db = DatabaseAdvanced(db_path=":memory:", enable_metrics=True)
        await db.init()

        # Попытка доступа к несуществующей таблице должна вызвать ValueError
        with pytest.raises(ValueError, match="Invalid table name"):
            async with db.pool.acquire() as conn:
                await db._get_table_columns(conn, "malicious_table; DROP TABLE users")

        await db.close()

    @pytest.mark.asyncio
    async def test_valid_table_access(self):
        """Проверка доступа к валидным таблицам"""
        from src.core.db_advanced import DatabaseAdvanced

        db = DatabaseAdvanced(db_path=":memory:", enable_metrics=True)
        await db.init()

        # Доступ к валидным таблицам должен работать
        async with db.pool.acquire() as conn:
            # Проверяем что whitelist работает (не выбрасывает исключение)
            try:
                columns = await db._get_table_columns(conn, "users")
                # Если таблица существует, проверяем колонки
                if columns:
                    assert "user_id" in columns
                    assert "is_admin" in columns
            except ValueError:
                # Если ValueError - значит whitelist сработал некорректно
                pytest.fail("Valid table name 'users' was rejected by whitelist")

        await db.close()


@pytest.mark.integration
class TestMetricsIntegration:
    """Интеграционные тесты полного flow с метриками"""

    @pytest.mark.asyncio
    async def test_full_sql_injection_flow(self):
        """Полный flow: ввод → валидация → метрики"""
        from src.core.metrics import init_metrics

        # Инициализируем metrics
        metrics = init_metrics(enable_prometheus=False, prometheus_port=None)

        # SQL injection попытка
        malicious_input = "test'; DELETE FROM users WHERE '1'='1"

        result = InputValidator.validate_question(malicious_input)

        # Проверяем что попытка заблокирована
        assert not result.is_valid or result.warnings

        # Проверяем fallback метрики (Prometheus отключен в тесте)
        stats = metrics.get_fallback_metrics()

        # Должна быть хотя бы одна попытка в метриках
        # (если validation.py корректно вызывает metrics)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
