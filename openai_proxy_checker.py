from __future__ import annotations

import os
from urllib.parse import urlparse
from dotenv import load_dotenv
import httpx
from openai import OpenAI

# ---------- ENV / .env ----------
load_dotenv()

def _require_env(name: str) -> str:
    val = os.getenv(name, "").strip()
    if not val:
        raise RuntimeError(f"{name} не задан. Укажи его в переменных окружения или .env")
    return val

# ---------- PROXY (опционально) ----------
def build_proxy() -> str | None:
    url = os.getenv("TELEGRAM_PROXY_URL", "").strip()
    if not url:
        return None
    if "://" not in url:
        url = "http://" + url
    u = urlparse(url)

    user = os.getenv("TELEGRAM_PROXY_USER", "").strip()
    pwd  = os.getenv("TELEGRAM_PROXY_PASS", "").strip()

    if user and pwd and (not u.username):
        host = u.hostname or ""
        port = f":{u.port}" if u.port else ""
        return f"{u.scheme}://{user}:{pwd}@{host}{port}"
    return url

def make_httpx_client(proxy: str | None) -> httpx.Client:
    return httpx.Client(
        proxy=proxy,                 # NB: именно proxies=, а не proxy=
        timeout=httpx.Timeout(30.0),
        verify=True,
        trust_env=False,
        http2=True,
    )

# ---------- helpers ----------
def _extract_text(resp) -> str:
    text = getattr(resp, "output_text", None)
    if text:
        return text.strip()
    try:
        return (resp.output[0].content[0].text or "").strip()
    except Exception:
        return str(resp)

def _reasoning_ate_all_tokens(resp) -> bool:
    try:
        out = int(getattr(resp.usage, "output_tokens", 0) or 0)
        rdet = getattr(resp.usage, "output_tokens_details", None)
        rtok = int(getattr(rdet, "reasoning_tokens", 0) or 0)
        return out > 0 and out == rtok  # весь бюджет вывода ушёл в рассуждения
    except Exception:
        return False

# ---------- MAIN ----------
def main() -> None:
    api_key = _require_env("OPENAI_API_KEY")
    model   = _require_env("OPENAI_MODEL")

    client = OpenAI(
        api_key=api_key,
        http_client=make_httpx_client(build_proxy()),
    )

    # fail-fast: проверим доступ к модели
    client.models.retrieve(model)

    prompt = "Ответь одним коротким предложением: как расторгнуть договор аренды?"

    # первый вызов — без temperature/top_p
    resp = client.responses.create(
        model=model,
        input=prompt,
        text={"verbosity": "low"},      # допустимые значения: low | medium | high
        reasoning={"effort": "low"},    # «думай короче», чтобы хватило токенов на текст
        max_output_tokens=256,          # запас под текст
    )

    # если текста нет (всё съел reasoning) — повторим с большим лимитом
    if _reasoning_ate_all_tokens(resp) or not _extract_text(resp):
        resp = client.responses.create(
            model=model,
            input=prompt,
            text={"verbosity": "low"},
            reasoning={"effort": "low"},
            max_output_tokens=512,
        )

    print(_extract_text(resp))

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        import sys
        sys.stderr.write(f"[ERROR] {e}\n")
        sys.exit(1)
