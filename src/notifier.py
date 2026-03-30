"""Telegram notification helper for Gold Regime X.

Reads credentials from environment variables (or .env file if python-dotenv
is installed).  All public functions degrade gracefully when credentials are
missing — the rest of the system continues to operate without Telegram.

Environment variables required in .env:
    TELEGRAM_BOT_TOKEN   — BotFather token (e.g. 123456789:AAF-xxx…)
    TELEGRAM_CHAT_ID     — Your personal chat ID (see README_REMOTE.md)

Install:
    pip install requests python-dotenv
"""

import os
import requests

from src.logger import setup_logger

logger = setup_logger(__name__)

# Load .env on import if python-dotenv is available (silently ignored otherwise)
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

_TELEGRAM_API = "https://api.telegram.org/bot{token}/sendMessage"


def get_credentials() -> tuple:
    """Return (bot_token, chat_id) from environment variables."""
    return os.getenv("TELEGRAM_BOT_TOKEN"), os.getenv("TELEGRAM_CHAT_ID")


def send_telegram_msg(message: str, parse_mode: str = "HTML") -> bool:
    """Send a message to your configured Telegram chat.

    Returns True on success, False on any failure (never raises).
    The ``parse_mode`` parameter accepts 'HTML' or 'Markdown'.
    Use HTML tags like <b>bold</b>, <code>monospace</code>, <i>italic</i>.
    """
    token, chat_id = get_credentials()
    if not token or not chat_id:
        logger.debug("Telegram credentials not set — skipping notification.")
        return False

    payload = {
        "chat_id":    chat_id,
        "text":       message,
        "parse_mode": parse_mode,
    }
    try:
        r = requests.post(
            _TELEGRAM_API.format(token=token),
            data=payload,
            timeout=10,
        )
        if not r.ok:
            logger.warning("Telegram API error %d: %s", r.status_code, r.text[:200])
            return False
        return True
    except Exception as exc:
        logger.warning("Telegram send failed: %s", exc)
        return False
