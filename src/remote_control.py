"""Telegram remote control panel for Gold Regime X.

Runs a long-polling Telegram bot that accepts keyboard button commands from
the authorised user and dispatches them as subprocesses so the listener
thread is never blocked.

Keyboard layout:
    ┌──────────────────────┬─────────────────────┐
    │  🚀 START TRADING    │  🛑 STOP TRADING     │
    ├──────────────────────┼─────────────────────┤
    │  📉 START OPTIMIZE   │  📊 BOT STATUS       │
    │     (M5)             │                     │
    └──────────────────────┴─────────────────────┘

Required env vars (see .env.example):
    TELEGRAM_BOT_TOKEN   — BotFather token
    TELEGRAM_CHAT_ID     — Your chat ID (used for outbound heartbeat messages)
    ALLOWED_USER_ID      — Your numeric Telegram user ID (security gate)
    LIVE_TF              — Default TF for START TRADING (default: H1)
    LIVE_BROKER          — Default broker  (default: headway_cent)
    LIVE_BALANCE         — Default balance (default: 15)

Usage:
    python main.py --mode listen
"""

import os
import subprocess
import time

import requests
from requests.exceptions import ReadTimeout

from src.logger import setup_logger
from src.notifier import get_credentials, send_telegram_msg

logger = setup_logger(__name__)

# ── Telegram keyboard sent with every reply ────────────────────────────────────
_KEYBOARD = {
    "keyboard": [
        ["🚀 START TRADING",       "🛑 STOP TRADING"],
        ["📉 START OPTIMIZE (M5)", "📊 BOT STATUS"],
    ],
    "resize_keyboard":   True,
    "one_time_keyboard": False,
}

# Tracks subprocesses launched by this session so we can terminate them
_procs: dict[str, subprocess.Popen] = {}


def _api(token: str, method: str, _req_timeout=(10, 15), **params) -> dict:
    """Call a Telegram Bot API method; returns the parsed JSON response.

    ``_req_timeout`` is passed to requests as (connect_timeout, read_timeout)
    and is intentionally NOT forwarded to Telegram — hence the underscore prefix.
    Long-polling calls should supply a read timeout > the Telegram ``timeout``
    parameter to avoid spurious ReadTimeout exceptions.
    """
    try:
        r = requests.post(
            f"https://api.telegram.org/bot{token}/{method}",
            json=params,
            timeout=_req_timeout,
        )
        return r.json()
    except ReadTimeout:
        # Expected for getUpdates when no new messages arrive within the poll
        # window — Telegram returns an empty result list, which is normal.
        logger.debug("Telegram API (%s): long-poll returned empty (no new messages).", method)
        return {}
    except Exception as exc:
        logger.warning("Telegram API (%s) failed: %s", method, exc)
        return {}


def _reply(token: str, chat_id, text: str) -> None:
    """Send a message with the persistent control keyboard attached."""
    _api(
        token, "sendMessage",
        chat_id=chat_id,
        text=text,
        reply_markup=_KEYBOARD,
        parse_mode="HTML",
    )


def _proc_alive(key: str) -> bool:
    """Return True if the subprocess stored under *key* is still running."""
    proc = _procs.get(key)
    return proc is not None and proc.poll() is None


def _handle(token: str, chat_id, user_id: str, text: str) -> None:
    """Authorise and dispatch a single inbound command."""
    allowed = os.getenv("ALLOWED_USER_ID", "")
    if user_id != allowed:
        _api(token, "sendMessage", chat_id=chat_id, text="Unauthorized.")
        return

    tf      = os.getenv("LIVE_TF",      "H1")
    broker  = os.getenv("LIVE_BROKER",  "headway_cent")
    balance = os.getenv("LIVE_BALANCE", "15")

    if text == "🚀 START TRADING":
        if _proc_alive("trading"):
            _reply(token, chat_id, "Trading is already running.")
            return
        _procs["trading"] = subprocess.Popen([
            "python", "main.py",
            "--mode", "live", "--account", "live",
            "--tf", tf, "--broker", broker, "--balance", balance,
        ])
        _reply(token, chat_id,
               f"<b>Trading started</b>\nTF={tf}  broker={broker}  balance=${balance}")

    elif text == "🛑 STOP TRADING":
        if _proc_alive("trading"):
            _procs["trading"].terminate()
            _reply(token, chat_id, "<b>Trading stopped.</b>")
        else:
            _reply(token, chat_id, "No active trading process found.")

    elif text == "📉 START OPTIMIZE (M5)":
        if _proc_alive("optimizer"):
            _reply(token, chat_id, "Optimization is already running.")
            return
        _procs["optimizer"] = subprocess.Popen([
            "python", "main.py",
            "--mode", "optimize", "--tf", "M5",
            "--broker", broker, "--balance", balance, "--trials", "500",
        ])
        _reply(token, chat_id,
               "M5 optimization started.\n"
               "(Resumes from study.db if previous progress exists.)")

    elif text == "📊 BOT STATUS":
        try:
            from src.auditor import get_daily_report
            report = get_daily_report(broker=broker)
        except Exception as exc:
            report = f"Status unavailable: {exc}"
        _reply(token, chat_id, report)

    else:
        _reply(token, chat_id, "Use the keyboard buttons below to control the bot.")


def run_listener() -> None:
    """Poll Telegram for updates and dispatch commands until KeyboardInterrupt.

    Uses the getUpdates long-polling method (timeout=30s) so the bot reacts
    within seconds without holding a persistent WebSocket connection.
    """
    token, _ = get_credentials()
    if not token:
        raise RuntimeError(
            "TELEGRAM_BOT_TOKEN not set.  "
            "Copy .env.example to .env and fill in your credentials."
        )

    logger.info("Remote control listener started.  Waiting for commands...")
    send_telegram_msg("<b>Gold Regime X</b> remote control is <b>online</b>.")

    offset = 0
    while True:
        try:
            # timeout=25: Telegram holds the connection up to 25s for new msgs.
            # _req_timeout read must exceed that so requests doesn't give up first.
            data = _api(token, "getUpdates", offset=offset,
                        timeout=25, _req_timeout=(10, 35))
            for update in data.get("result", []):
                offset = update["update_id"] + 1
                msg     = update.get("message", {})
                chat_id = msg.get("chat", {}).get("id")
                user_id = str(msg.get("from", {}).get("id", ""))
                text    = msg.get("text", "").strip()
                if chat_id and text:
                    logger.info(
                        "Inbound: user_id=%s  chat_id=%s  text=%r",
                        user_id, chat_id, text,
                    )
                    _handle(token, chat_id, user_id, text)

        except KeyboardInterrupt:
            logger.info("Remote control stopped.")
            send_telegram_msg("Gold Regime X remote control is <b>offline</b>.")
            break
        except Exception as exc:
            logger.error("Listener error: %s — retrying in 10s", exc)
            time.sleep(10)
