from openai import OpenAI
import httpx, os
from dotenv import load_dotenv
from urllib.parse import urlparse

load_dotenv()

def build_proxy():
    url = os.getenv("TELEGRAM_PROXY_URL")
    user = os.getenv("TELEGRAM_PROXY_USER")
    pwd  = os.getenv("TELEGRAM_PROXY_PASS")
    if not url:
        return None
    if "://" not in url:
        url = "http://" + url
    u = urlparse(url)
    if user and pwd and not u.username:
        return f"{u.scheme}://{user}:{pwd}@{u.hostname}:{u.port}"
    return url

PROXY = build_proxy()

client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
    http_client=httpx.Client(
        proxy=PROXY,          # httpx >= 0.28
        timeout=30.0,
        verify=True,
        trust_env=False,
    ),
)

resp = client.responses.create(
    model=os.getenv("OPENAI_MODEL", "gpt-5"),
    input="расторгнуть договор аренды?",
    max_output_tokens=16,      # ⬅️ минимум 16
)
print(resp.output_text.strip())





# #!/usr/bin/env python3
# # -*- coding: utf-8 -*-
#
# """
# OpenAI Proxy Checker — проверка прокси на доступ к OpenAI (web + API).
#
# Особенности:
# - Минимально реалистичные браузерные заголовки (без подозрительных sec-ch-*).
# - TLS-проверка ВКЛ по умолчанию (можно отключить флагом --insecure).
# - Сессия requests с ретраями и коннект-пулом.
# - Корректная трактовка API-доступности: 401 на /v1/models = reachable без ключа.
# - Дешёвый API-пинг с ключом (max_tokens=1), Chat или Responses API.
# - Параллельная проверка прокси (ThreadPoolExecutor).
# - Поддержка http/https/socks5/socks5h, аутентификация в URL или отдельными полями.
# - Метрики: статус, время отклика, редиректы.
# - JSON-отчёт и сохранение рабочих прокси.
#
# Автор стиля: «коротко, ясно, по-взрослому». (И да, без лишних "секси"-хедеров в raw requests 😎)
# """
#
# from __future__ import annotations
#
# import argparse
# import json
# import os
# import random
# import sys
# import time
# from concurrent.futures import ThreadPoolExecutor, as_completed
# from dataclasses import dataclass, field
# from typing import Dict, List, Optional, Tuple
# from urllib.parse import urlparse
#
# import requests
# import urllib3
# from requests.adapters import HTTPAdapter
# from urllib3.util.retry import Retry
#
# # По умолчанию не глушим warning'и. Будем делать это только при --insecure.
# # urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
#
#
# # ──────────────────────────────── Модели данных ────────────────────────────────
# @dataclass
# class URLCheckResult:
#     url: str
#     status: str
#     success: bool
#     latency_s: Optional[float] = None
#     redirects: int = 0
#
#
# @dataclass
# class ProxyResult:
#     proxy_display: str
#     working: bool
#     best_url: Optional[str]
#     url_results: List[URLCheckResult] = field(default_factory=list)
#     working_urls: List[str] = field(default_factory=list)
#     api_result: Optional[str] = None
#
#
# # ──────────────────────────────── Класс Checker ────────────────────────────────
# class OpenAIProxyChecker:
#     WEB_URLS = [
#         "https://openai.com",
#         "https://help.openai.com",
#         "https://platform.openai.com",
#         "https://platform.openai.com/docs/overview",
#         "https://chat.openai.com",  # исторический домен, всё ещё полезно тестить
#     ]
#     API_MODELS_URL = "https://api.openai.com/v1/models"
#
#     def __init__(
#         self,
#         timeout: int = 15,
#         max_workers: int = 5,
#         check_all_urls: bool = False,
#         api_key: Optional[str] = None,
#         api_type: str = "chat",  # "chat" | "responses"
#         model: Optional[str] = None,
#         insecure: bool = False,
#     ):
#         """
#         :param timeout: таймаут на запрос (секунды)
#         :param max_workers: параллельность по числу прокси
#         :param check_all_urls: если False — останавливаемся на первом успехе для данного прокси
#         :param api_key: ключ OpenAI для реального API-пинга (опционально)
#         :param api_type: "chat" или "responses" — какой API пинговать с ключом
#         :param model: модель для пинга (если None — читаем из env OPENAI_MODEL или берём "gpt-4.1")
#         :param insecure: если True — verify=False и выключаем SSL warning'и (НЕ РЕКОМЕНДУЕТСЯ)
#         """
#         self.timeout = timeout
#         self.max_workers = max_workers
#         self.check_all_urls = check_all_urls
#         self.api_key = api_key
#         self.api_type = api_type.lower().strip()
#         self.model = model or os.getenv("OPENAI_MODEL", "gpt-5")
#         self.insecure = insecure
#
#         self.openai_urls: List[str] = [*self.WEB_URLS, self.API_MODELS_URL]
#
#         self.working_proxies: List[ProxyResult] = []
#         self.failed_proxies: List[ProxyResult] = []
#
#         self.session = self._build_session(insecure=self.insecure)
#
#     # ──────────────────────────────── Вспомогательные ──────────────────────────
#     def _build_session(self, insecure: bool = False) -> requests.Session:
#         """Создаёт requests.Session с ретраями и коннект-пулом."""
#         s = requests.Session()
#         retry = Retry(
#             total=3,
#             backoff_factor=0.3,
#             status_forcelist=[429, 500, 502, 503, 504],
#             allowed_methods={"HEAD", "GET", "POST", "OPTIONS"},
#             raise_on_status=False,
#         )
#         adapter = HTTPAdapter(max_retries=retry, pool_connections=64, pool_maxsize=64)
#         s.mount("http://", adapter)
#         s.mount("https://", adapter)
#         # TLS проверка по умолчанию включена
#         s.verify = False if insecure else True
#         if insecure:
#             urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
#         return s
#
#     def get_realistic_headers(self) -> Dict[str, str]:
#         """
#         Возвращает минимально правдоподобные заголовки браузера.
#         Без sec-ch-ua*, sec-fetch-*, чтобы не триггерить антибот на сыром HTTP-клиенте.
#         """
#         user_agents = [
#             "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/138.0.0.0 Safari/537.36",
#             "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/138.0.0.0 Safari/537.36",
#             "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/138.0.0.0 Safari/537.36",
#             "Mozilla/5.0 (Linux; Android 10; K) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/138.0.0.0 Mobile Safari/537.36",
#         ]
#         return {
#             "User-Agent": random.choice(user_agents),
#             "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
#             "Accept-Language": "ru,en;q=0.9",
#             "Cache-Control": "no-cache",
#             "Pragma": "no-cache",
#         }
#
#     # ──────────────────────────────── Прокси парсер ────────────────────────────
#     def parse_proxy(self, proxy_string: str, username: Optional[str] = None, password: Optional[str] = None) -> Optional[Dict[str, str]]:
#         """
#         Поддерживаемые форматы:
#           - ip:port
#           - scheme://ip:port  (http|https|socks5|socks5h)
#           - user:pass@ip:port
#           - scheme://user:pass@ip:port
#         Также можно передать username/password отдельными аргументами.
#         """
#         p = (proxy_string or "").strip()
#         if not p:
#             return None
#
#         def build(u: str) -> Dict[str, str]:
#             return {"http": u, "https": u}
#
#         # username/password отдельно
#         if username and password:
#             u = urlparse(p if "://" in p else f"http://{p}")
#             if u.scheme and u.hostname and u.port:
#                 return build(f"{u.scheme}://{username}:{password}@{u.hostname}:{u.port}")
#
#         # без схемы (ip:port)
#         if "://" not in p and ":" in p and "@" not in p:
#             ip, port = p.split(":", 1)
#             return build(f"http://{ip}:{port}")
#
#         # есть user:pass без схемы
#         if "@" in p and "://" not in p:
#             p = "http://" + p
#
#         # нормальная схема
#         if p.startswith(("http://", "https://", "socks5://", "socks5h://")):
#             return build(p)
#
#         return None
#
#     # ──────────────────────────────── API пинг ─────────────────────────────────
#     def test_openai_api(self, proxy_dict: Dict[str, str]) -> Optional[str]:
#         """
#         Дешёвый API-пинг:
#         - Если ключа нет — не делаем пинг.
#         - Если ключ есть — делаем лёгкий вызов (max_output_tokens=1) через выбранный API.
#         Возвращаем короткий статус + текст ошибки при не-200.
#         """
#         if not self.api_key:
#             return None
#
#         headers = {
#             "Authorization": f"Bearer {self.api_key}",
#             "Content-Type": "application/json",
#             "User-Agent": "ProxyChecker/1.0",
#         }
#
#         try:
#             if self.api_type == "responses":
#                 # ✅ Responses API: самый простой валидный payload
#                 payload = {
#                     "model": self.model,
#                     "input": "ping",  # <-- ключевая правка: строка, а не массив сообщений
#                     "max_output_tokens": 1,  # для responses — именно max_output_tokens
#                 }
#                 r = self.session.post(
#                     "https://api.openai.com/v1/responses",
#                     json=payload,
#                     headers=headers,
#                     proxies=proxy_dict,
#                     timeout=self.timeout,
#                 )
#             else:  # "chat"
#                 payload = {
#                     "model": self.model,
#                     "messages": [{"role": "user", "content": "ping"}],
#                     "max_tokens": 1,  # для chat — max_tokens
#                 }
#                 r = self.session.post(
#                     "https://api.openai.com/v1/chat/completions",
#                     json=payload,
#                     headers=headers,
#                     proxies=proxy_dict,
#                     timeout=self.timeout,
#                 )
#
#             sc = r.status_code
#             if sc == 200:
#                 return "🤖 API SUCCESS (200)"
#             # Вернём текст ошибки, чтобы сразу видеть причину 400/403/…
#             try:
#                 err = r.json().get("error", {}).get("message")
#             except Exception:
#                 err = None
#             if sc == 401:
#                 return "🔑 API UNAUTHORIZED (401)"
#             if sc == 403:
#                 return f"🚫 API FORBIDDEN (403){' - ' + err if err else ''}"
#             if sc == 429:
#                 return "⏰ API RATE LIMITED (429)"
#             return f"❌ API HTTP {sc}{' - ' + err if err else ''}"
#
#         except requests.exceptions.Timeout:
#             return "⏱️ API TIMEOUT"
#         except requests.exceptions.ProxyError:
#             return "🔌 API PROXY ERROR"
#         except requests.exceptions.ConnectionError:
#             return "🌐 API CONNECTION ERROR"
#         except Exception as e:
#             return f"💥 API ERROR: {str(e)[:60]}"
#
#     # ──────────────────────────────── Проверка URL ─────────────────────────────
#     def _check_single_url(self, url: str, proxy_dict: Dict[str, str]) -> URLCheckResult:
#         headers = self.get_realistic_headers()
#         t0 = time.perf_counter()
#         try:
#             resp = self.session.get(
#                 url,
#                 headers=headers,
#                 proxies=proxy_dict,
#                 timeout=self.timeout,
#                 allow_redirects=True,
#                 stream=True,  # быстрее получить headers и закрыть
#             )
#             latency = time.perf_counter() - t0
#             sc = resp.status_code
#             redirects = len(resp.history)
#
#             is_api = url.startswith("https://api.openai.com/")
#             # Правила успеха:
#             # - WEB: 2xx/3xx
#             # - API: 2xx/3xx или 401/404/405 (reachability без ключа/метода)
#             ok = (200 <= sc < 400) or (is_api and sc in (401, 404, 405))
#
#             status = f"{'✅' if ok else '❌'} HTTP {sc}"
#             return URLCheckResult(url=url, status=status, success=ok, latency_s=latency, redirects=redirects)
#
#         except requests.exceptions.Timeout:
#             return URLCheckResult(url=url, status="⏱️ TIMEOUT", success=False)
#         except requests.exceptions.ProxyError:
#             return URLCheckResult(url=url, status="🔌 PROXY ERROR", success=False)
#         except requests.exceptions.ConnectionError:
#             return URLCheckResult(url=url, status="🌐 CONNECTION ERROR", success=False)
#         except requests.exceptions.SSLError:
#             return URLCheckResult(url=url, status="🔒 SSL ERROR", success=False)
#         except Exception as e:
#             return URLCheckResult(url=url, status=f"💥 ERROR: {str(e)[:60]}", success=False)
#
#     # ──────────────────────────────── Проверка прокси ──────────────────────────
#     def check_openai_proxy(
#         self,
#         proxy: str,
#         username: Optional[str] = None,
#         password: Optional[str] = None,
#     ) -> Optional[ProxyResult]:
#         """Проверяет один прокси по веб-URL и API-модели."""
#         proxy_dict = self.parse_proxy(proxy, username, password)
#         if not proxy_dict:
#             print(f"⚠️ Не удалось распарсить прокси: {proxy}")
#             return None
#
#         proxy_display = proxy if not (username and password) else f"{proxy} (auth: {username}:***)"
#         print(f"🔍 Проверяю прокси: {proxy_display}")
#
#         res = ProxyResult(proxy_display=proxy_display, working=False, best_url=None)
#
#         # API-пинг (если есть ключ)
#         api_status = self.test_openai_api(proxy_dict)
#         res.api_result = api_status
#         if api_status and "SUCCESS" in api_status:
#             res.working = True
#
#         # Прогон URL'ов
#         for i, url in enumerate(self.openai_urls, 1):
#             if i > 1:
#                 # лёгкая стохастическая пауза, чтобы не спамить
#                 time.sleep(random.uniform(0.3, 0.9))
#
#             print(f"   🌐 [{i}/{len(self.openai_urls)}] {url}")
#             ures = self._check_single_url(url, proxy_dict)
#             res.url_results.append(ures)
#
#             if ures.success:
#                 res.working = True
#                 res.working_urls.append(url)
#                 if not res.best_url:
#                     res.best_url = url
#                 print(f"   ✅ OK {ures.status} • {ures.latency_s:.3f}s • redirects={ures.redirects}")
#                 if not self.check_all_urls:
#                     break
#             else:
#                 print(f"   {ures.status}")
#
#         total_ok = len(res.working_urls) + (1 if (api_status and "SUCCESS" in api_status) else 0)
#         if res.working:
#             print(f"   🎉 Итого работает: {total_ok} сервис(а/ов)")
#         else:
#             print("   💔 Ни один сервис OpenAI не заработал")
#
#         return res
#
#     # ──────────────────────────────── Загрузка прокси ──────────────────────────
#     @staticmethod
#     def load_proxies_from_file(filename: str) -> List[str]:
#         """Загружает прокси из файла, игнорируя пустые строки и комментарии (#)."""
#         proxies: List[str] = []
#         try:
#             with open(filename, "r", encoding="utf-8") as f:
#                 for line in f:
#                     line = line.strip()
#                     if not line or line.startswith("#"):
#                         continue
#                     proxies.append(line)
#         except FileNotFoundError:
#             print(f"❌ Файл не найден: {filename}")
#         except Exception as e:
#             print(f"❌ Ошибка чтения {filename}: {e}")
#         return proxies
#
#     @staticmethod
#     def load_from_env() -> Tuple[List[str], Optional[Dict[str, str]], Optional[str]]:
#         """
#         Загружает прокси и ключ из env / .env:
#           TELEGRAM_PROXY_URL, TELEGRAM_PROXY_USER, TELEGRAM_PROXY_PASS, OPENAI_API_KEY
#         """
#         env_file = ".env"
#         if os.path.exists(env_file):
#             try:
#                 from dotenv import load_dotenv
#
#                 load_dotenv(env_file)
#                 print(f"📄 Загружен .env")
#             except Exception:
#                 pass
#
#         proxy_url = os.getenv("TELEGRAM_PROXY_URL")
#         proxy_user = os.getenv("TELEGRAM_PROXY_USER")
#         proxy_pass = os.getenv("TELEGRAM_PROXY_PASS")
#         api_key = os.getenv("OPENAI_API_KEY")
#
#         if proxy_url:
#             auth_data = None
#             if proxy_user and proxy_pass:
#                 auth_data = {"username": proxy_user, "password": proxy_pass}
#             return [proxy_url], auth_data, api_key
#
#         return [], None, api_key
#
#     # ──────────────────────────────── Массовая проверка ────────────────────────
#     def check_proxies(
#         self,
#         proxies: List[str],
#         auth_data: Optional[Dict[str, str]] = None,
#     ) -> None:
#         """Параллельно проверяет список прокси и печатает summary."""
#         print(f"🚀 Начинаю проверку {len(proxies)} прокси на OpenAI…")
#         print("🎯 URL для теста:")
#         for i, u in enumerate(self.openai_urls, 1):
#             print(f"   {i}. {u}")
#         print("=" * 80)
#
#         def run_one(px: str) -> Optional[ProxyResult]:
#             if auth_data:
#                 return self.check_openai_proxy(px, auth_data.get("username"), auth_data.get("password"))
#             return self.check_openai_proxy(px)
#
#         futures = []
#         with ThreadPoolExecutor(max_workers=self.max_workers) as ex:
#             for p in proxies:
#                 futures.append(ex.submit(run_one, p))
#
#             for fut in as_completed(futures):
#                 result = fut.result()
#                 if not result:
#                     continue
#                 if result.working:
#                     self.working_proxies.append(result)
#                     print(f"✅ ПРОКСИ РАБОТАЕТ • лучший URL: {result.best_url}")
#                 else:
#                     self.failed_proxies.append(result)
#                     print("❌ Прокси не прошёл тесты")
#                 print("-" * 50)
#
#         self.print_summary()
#
#     # ──────────────────────────────── Отчёты ───────────────────────────────────
#     def print_summary(self) -> None:
#         print("\n" + "=" * 80)
#         total = len(self.working_proxies) + len(self.failed_proxies)
#         print("📊 ИТОГИ ПРОВЕРКИ OpenAI:")
#         print(f"Всего проверено: {total}")
#         print(f"✅ Работающих: {len(self.working_proxies)}")
#         print(f"❌ Не работающих: {len(self.failed_proxies)}")
#         print("=" * 80)
#
#         if self.working_proxies:
#             print("\n🎉 РАБОЧИЕ ПРОКСИ:")
#             for r in self.working_proxies:
#                 ok_urls = [u.url for u in r.url_results if u.success]
#                 print(f"\n🔗 {r.proxy_display}")
#                 if r.api_result:
#                     print(f"   🤖 API: {r.api_result}")
#                 print(f"   🏆 Лучший URL: {r.best_url}")
#                 print(f"   📈 Работают: {len(ok_urls)}/{len(self.openai_urls)}")
#                 print("   ✅ URL:")
#                 for u in ok_urls:
#                     print(f"      • {u}")
#
#         if self.failed_proxies:
#             print(f"\n💔 НЕ РАБОЧИЕ ПРОКСИ ({len(self.failed_proxies)}):")
#             for r in self.failed_proxies:
#                 print(f"\n❌ {r.proxy_display}")
#                 print("   📋 Результаты:")
#                 for u in r.url_results:
#                     print(f"      • {u.url}: {u.status}")
#
#     def save_working_proxies(self, filename: str = "working_openai_proxies.txt") -> None:
#         if not self.working_proxies:
#             print("ℹ️ Нет рабочих прокси для сохранения.")
#             return
#         with open(filename, "w", encoding="utf-8") as f:
#             for r in self.working_proxies:
#                 f.write(f"{r.proxy_display} # best: {r.best_url}\n")
#         print(f"💾 Сохранено: {filename}")
#
#     def export_json(self, filename: str) -> None:
#         data = []
#         for r in self.working_proxies + self.failed_proxies:
#             data.append(
#                 {
#                     "proxy": r.proxy_display,
#                     "working": r.working,
#                     "best_url": r.best_url,
#                     "api_result": r.api_result,
#                     "urls": [
#                         {
#                             "url": u.url,
#                             "status": u.status,
#                             "success": u.success,
#                             "latency_s": u.latency_s,
#                             "redirects": u.redirects,
#                         }
#                         for u in r.url_results
#                     ],
#                 }
#             )
#         with open(filename, "w", encoding="utf-8") as fh:
#             json.dump(data, fh, ensure_ascii=False, indent=2)
#         print(f"🧾 JSON-отчёт сохранён: {filename}")
#
#
# # ──────────────────────────────── CLI ─────────────────────────────────────────
# def parse_args(argv: List[str]) -> argparse.Namespace:
#     p = argparse.ArgumentParser(
#         description="Проверка прокси на доступ к OpenAI (web + api). TLS включён по умолчанию.",
#         formatter_class=argparse.ArgumentDefaultsHelpFormatter,
#     )
#     src = p.add_mutually_exclusive_group()
#     src.add_argument("--file", help="Файл со списком прокси")
#     src.add_argument("--proxies", nargs="+", help="Прокси списком через пробел")
#     src.add_argument("--env", action="store_true", help="Загрузить прокси из .env/ENV")
#
#     p.add_argument("--username", help="Логин для прокси (если не в URL)")
#     p.add_argument("--password", help="Пароль для прокси (если не в URL)")
#     p.add_argument("--api-key", help="OpenAI API ключ (иначе берётся из OPENAI_API_KEY)")
#     p.add_argument("--model", help="Модель для API-пинга (по умолчанию OPENAI_MODEL или gpt-4.1)")
#     p.add_argument("--api", choices=["chat", "responses"], default="chat", help="Каким API пинговать")
#     p.add_argument("--timeout", type=int, default=20, help="Таймаут запроса (сек)")
#     p.add_argument("--concurrency", type=int, default=5, help="Число потоков")
#     p.add_argument("--full", action="store_true", help="Проверять все URL (иначе стоп на первом успешном)")
#     p.add_argument("--insecure", action="store_true", help="Отключить TLS-проверку (verify=False) — НЕ РЕКОМЕНДУЕТСЯ")
#     p.add_argument("--save", help="Сохранить рабочие прокси в файл")
#     p.add_argument("--json", help="Сохранить подробный JSON-отчёт")
#     return p.parse_args(argv)
#
#
# def main(argv: Optional[List[str]] = None) -> None:
#     args = parse_args(argv or sys.argv[1:])
#
#     # Источник прокси
#     proxies: List[str] = []
#     auth_data: Optional[Dict[str, str]] = None
#
#     if args.env:
#         proxies, auth_data, env_key = OpenAIProxyChecker.load_from_env()
#         api_key = args.api_key or env_key
#     elif args.file:
#         proxies = OpenAIProxyChecker.load_proxies_from_file(args.file)
#         api_key = args.api_key or os.getenv("OPENAI_API_KEY")
#     elif args.proxies:
#         proxies = [p for p in args.proxies if p.strip()]
#         api_key = args.api_key or os.getenv("OPENAI_API_KEY")
#     else:
#         # Фоллбек к лёгкой интерактивщине — как в твоём оригинале
#         print("📥 Источник прокси не указан. Введите прокси по одному (пусто — завершить):")
#         while True:
#             line = input("Прокси: ").strip()
#             if not line:
#                 break
#             proxies.append(line)
#         api_key = args.api_key or os.getenv("OPENAI_API_KEY")
#
#     if args.username and args.password:
#         auth_data = {"username": args.username, "password": args.password}
#
#     if not proxies:
#         print("❌ Прокси не найдены.")
#         return
#
#     checker = OpenAIProxyChecker(
#         timeout=args.timeout,
#         max_workers=args.concurrency,
#         check_all_urls=args.full,
#         api_key=api_key,
#         api_type=args.api,
#         model=args.model,
#         insecure=args.insecure,
#     )
#
#     if api_key:
#         masked = f"{api_key[:6]}...{api_key[-4:]}" if len(api_key) >= 12 else "***"
#         print(f"🔐 API ключ найден: {masked}")
#     else:
#         print("ℹ️ API ключ не найден — реальный API-пинг будет пропущен.")
#
#     t0 = time.time()
#     checker.check_proxies(proxies, auth_data)
#     dt = time.time() - t0
#     print(f"\n⏱️ Время выполнения: {dt:.2f} сек")
#
#     if args.save:
#         checker.save_working_proxies(args.save)
#     if args.json:
#         checker.export_json(args.json)
#
#
# if __name__ == "__main__":
#     main()
