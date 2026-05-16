"""Multi-timeframe health monitor for Gold Regime X.

Periodically re-validates HMM+XGBoost signal quality on the most recently
synced MT5 data.  If rolling Sharpe for any timeframe drops below the alert
threshold a Telegram notification is fired immediately.

Usage (via main.py):
    python main.py --mode guardian --tf M5,M15,H1 --period 3m --interval 3600
    (defaults: all three TFs, last 3 months of data, check every hour)

The check re-uses the same validator.run_validation() that gates live trading,
so the health score is directly comparable to sync_validate output.
"""

import sys
import time

from src.logger import setup_logger
from src.notifier import send_telegram_msg

logger = setup_logger(__name__)

SHARPE_ALERT_THRESHOLD = 0.6   # Telegram alert fires below this
SHARPE_CRITICAL        = 0.4   # escalated "CRITICAL" label


# ── Main guardian loop ────────────────────────────────────────────────────────

def run_guardian(
    tfs: list,
    broker: str,
    account_size: float,
    period: str = "3m",
    interval_sec: int = 3600,
) -> None:
    """Monitor timeframe health continuously; alert via Telegram on degradation.

    Args:
        tfs:           List of timeframe strings, e.g. ["M5", "M15", "H1"].
        broker:        Broker key used for backtest cost simulation.
        account_size:  USD balance for lot-sizing in the backtest.
        period:        MT5 lookback period for synced data ('3m', '6m', etc.).
        interval_sec:  Seconds between health checks.  Default 3600 (1 hour).
    """
    from src.validator import run_validation

    logger.info(
        "Guardian started — TFs=%s  interval=%ds  alert_threshold=%.1f",
        tfs, interval_sec, SHARPE_ALERT_THRESHOLD,
    )
    send_telegram_msg(
        f"<b>Guardian online</b>\n"
        f"Monitoring: <b>{', '.join(tfs)}</b>\n"
        f"Check interval: every {interval_sec // 60} min"
    )

    cycle = 0

    while True:
        cycle += 1
        lines   = [f"<b>Guardian Health Check #{cycle}</b>"]
        alerts  = []

        # ── HMM/XGBoost health check (every cycle) ────────────────────────
        for tf in tfs:
            try:
                result = run_validation(
                    tf=tf,
                    broker=broker,
                    account_size=account_size,
                    period=period,
                )
                sharpe = result["sharpe"]
                status = result["status"]
                trades = result["n_trades"]

                if sharpe < SHARPE_CRITICAL:
                    icon = "CRITICAL"
                elif sharpe < SHARPE_ALERT_THRESHOLD:
                    icon = "WARN"
                elif status == "pass":
                    icon = "OK"
                else:
                    icon = "WARN"

                lines.append(
                    f"  [{tf}] Sharpe={sharpe:.3f}  "
                    f"Trades={trades}  [{icon}]"
                )
                logger.info(
                    "Guardian [%s]: sharpe=%.3f  status=%s  trades=%d",
                    tf, sharpe, status, trades,
                )

                if sharpe < SHARPE_ALERT_THRESHOLD:
                    level = "CRITICAL" if sharpe < SHARPE_CRITICAL else "WARNING"
                    alerts.append(
                        f"<b>GUARDIAN {level} — [{tf}]</b>\n"
                        f"Rolling Sharpe: <b>{sharpe:.3f}</b> "
                        f"(threshold: {SHARPE_ALERT_THRESHOLD})\n"
                        f"Action: python main.py --mode optimize --tf {tf}"
                    )
            except Exception as exc:
                msg = f"  [{tf}] Check error: {exc}"
                lines.append(msg)
                logger.error("Guardian [%s] check failed: %s", tf, exc)

        # ── Send periodic digest and any alerts ───────────────────────────
        send_telegram_msg("\n".join(lines))
        for alert in alerts:
            send_telegram_msg(alert)

        logger.info(
            "Guardian cycle #%d complete — next in %ds.", cycle, interval_sec
        )
        time.sleep(interval_sec)
