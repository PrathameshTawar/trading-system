from __future__ import annotations

import json
import logging
import os
import urllib.request
from pathlib import Path
from typing import Any

logger = logging.getLogger("signalforge.alerts")


def send_alert(
    level: str,
    message: str,
    details: dict[str, Any] | None = None,
    log_file: str | Path = "paper/alerts.log",
) -> dict[str, Any]:
    """Send structured alert to console, log file, and optional external Webhooks (Telegram / Slack)."""
    alert_record = {
        "timestamp": str(os.environ.get("ALERT_TIMESTAMP") or ""),
        "level": level.upper(),
        "message": message,
        "details": details or {},
    }

    # 1. Console Logging
    msg = f"[{alert_record['level']}] {message} | Details: {details or {}}"
    if level.upper() in ("CRITICAL", "HIGH", "ERROR"):
        logger.error(msg)
    else:
        logger.info(msg)

    # 2. Append to Local Log File
    try:
        p = Path(log_file)
        p.parent.mkdir(parents=True, exist_ok=True)
        with p.open("a", encoding="utf-8") as f:
            f.write(json.dumps(alert_record, default=str) + "\n")
    except Exception as exc:
        logger.warning(f"Failed to append alert to file {log_file}: {exc}")

    # 3. Optional Slack Webhook
    slack_url = os.environ.get("SLACK_WEBHOOK_URL")
    if slack_url:
        try:
            payload = json.dumps({"text": f":warning: *SignalForge Alert [{level.upper()}]*: {message}"}).encode("utf-8")
            req = urllib.request.Request(slack_url, data=payload, headers={"Content-Type": "application/json"})
            urllib.request.urlopen(req, timeout=3)
        except Exception as exc:
            logger.warning(f"Slack alert webhook failed: {exc}")

    # 4. Optional Telegram Bot Notification
    bot_token = os.environ.get("TELEGRAM_BOT_TOKEN")
    chat_id = os.environ.get("TELEGRAM_CHAT_ID")
    if bot_token and chat_id:
        try:
            tg_url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
            payload = json.dumps({"chat_id": chat_id, "text": f"🚨 SignalForge Alert [{level.upper()}]:\n{message}"}).encode("utf-8")
            req = urllib.request.Request(tg_url, data=payload, headers={"Content-Type": "application/json"})
            urllib.request.urlopen(req, timeout=3)
        except Exception as exc:
            logger.warning(f"Telegram alert send failed: {exc}")

    return alert_record
