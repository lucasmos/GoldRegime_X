# Gold Regime X — Live Trading Guide

This guide explains how to operate the Python MT5 Live Bridge for Gold Regime X.
The bridge connects directly to your running MetaTrader5 terminal, validates the
model against recent market data, and executes orders automatically on XAUUSD.

---

## Prerequisites

| Requirement | Notes |
|---|---|
| Windows OS | MetaTrader5 Python package is Windows-only |
| MT5 terminal running | Must be open and logged into your Headway account |
| XAUUSD in Market Watch | Right-click Market Watch → Show All, or search XAUUSD |
| Algorithmic trading enabled | MT5 → Tools → Options → Expert Advisors → Allow Algorithmic Trading |
| Models trained | `models/hmm_model.pkl` and `models/xgb_model.pkl` must exist |
| Python package | `pip install MetaTrader5>=5.0.45` |
| **EA removed from chart** | The GoldRegimeX.mq5 EA and the Python bridge share `MAGIC_NUMBER = 123456`. Running both simultaneously will double-count daily trades. Remove the EA from the XAUUSD chart before starting the Python bridge. |

---

## MT5 Terminal Setup

1. Open MetaTrader5.
2. Go to **Tools → Options → Expert Advisors**.
3. Check **Allow Algorithmic Trading**.
4. Check **Allow WebRequest for listed URL** and add `http://localhost` if using a local bridge.
5. Log into your Headway account (Cent or Live).
6. Add XAUUSD to the Market Watch if it is not already visible.

---

## Quick-Start Workflow

### Step 1 — Train the models (if not done already)

```
python main.py --mode process --tf H1
python main.py --mode optimize --trials 250 --broker headway_cent --balance 15 --tf H1
python main.py --mode train --broker headway_cent --balance 15 --tf H1
```

### Step 2 — Sync recent data and validate the model

Run this daily before going live to ensure the model is still healthy.

```
python main.py --mode sync_validate --period 3m --broker headway_cent --balance 15 --tf H1
```

| Validation Status | Meaning | Action |
|---|---|---|
| **PASS** (Sharpe ≥ 0.8) | Model is stable | Proceed to Step 3 |
| **WARN** (Sharpe 0.5–0.8) | Borderline performance | Proceed with caution or re-optimise |
| **FAIL** (Sharpe < 0.5) | Market regime drift | Run optimize + train before going live |

A FAIL status will exit with code 1 and block the script from continuing.

### Step 3 — Paper trade on demo (recommended first)

```
python main.py --mode live --account demo --broker headway_cent --balance 15 --tf H1
```

Demo mode logs every signal and simulates the full session-limit logic, but
sends no real orders to MT5.  Use this to verify signals appear as expected.

### Step 4 — Switch to live account

```
python main.py --mode live --account live --broker headway_cent --balance 15 --tf H1
```

You will be prompted to type `YES` to confirm before the loop starts.

---

## Command Reference

```
python main.py --mode sync_validate [OPTIONS]
    --period 3m          Lookback window  (e.g. 3m, 6m, 12m)
    --tf H1              Timeframe        (H1 or M15)
    --broker headway_cent
    --balance 15

python main.py --mode live [OPTIONS]
    --account demo|live  demo = dry-run (no orders)  live = real orders
    --tf H1              Must match the timeframe the models were trained on
    --broker headway_cent
    --balance 15         USD balance used for lot-sizing
    --period             (not used in live mode)
```

---

## How to Monitor Live Trading

### Terminal View (MT5)

Open the **Trade** tab at the bottom of the MT5 terminal to see all open
positions.  Each position placed by the Python bridge will show:

- **Symbol**: XAUUSD
- **Comment**: `GRX_BUY_p1of1_s0_p0.73` (direction / position index / HMM state / probability)
- **Magic**: 123456

### Expert Logs

Go to the **Journal** or **Experts** tab in MT5.  Errors from API calls
(e.g. "Invalid S/L", "Not enough money") appear in red here.

### Python Log File

All signals, orders, and errors are written to `logs/goldregimex.log`.

```
# Key log lines to watch for:
SIGNAL BUY    — signal fired, orders being placed
Order filled  — order confirmed by broker
Session limit — daily cap enforced (no more trades today)
Order failed  — execution error (check MT5 Journal tab)
Insufficient margin — reduce lot size or top up account
High-vol deviation  — elevated deviation used due to regime instability
```

To follow the log in real time:

```
# PowerShell
Get-Content logs/goldregimex.log -Wait -Tail 30
```

---

## Risk Management Reference

| Account Balance | Max Trades/Day | Positions per Signal | Total Positions |
|---|---|---|---|
| ≤ $50 USD | 2 | 1 | 2 maximum |
| > $50 USD (Bull/Bear) | 3 | 2 | 6 maximum |
| > $50 USD (Chop) | 2 | 2 | 4 maximum |

**All trades use the 1% risk rule:**

```
lot_size = (1% of USD balance) / (ATR(14) × 2.0)
```

The stop-loss is placed at `entry ± ATR(14) × 2.0`.  No take-profit is set —
positions are managed by the next opposing signal or manual intervention.

**Cent Account Note**: On Headway Cent, the terminal displays $15 USD as
`1500.00` in the balance field.  The Python bridge detects `--broker headway_cent`
and automatically divides by 100 for all internal calculations.  If you see a
balance of `1500.00` in MT5 and pass `--balance 15`, both will agree.

---

## Order Specifications

| Parameter | Value | Notes |
|---|---|---|
| Filling type | IOC (Immediate or Cancel) | Standard for Headway ECN |
| Deviation (normal) | 20 points | $0.20 on XAUUSD |
| Deviation (high-vol) | 50 points | When HMM self-transition prob < 0.70 |
| Magic number | 123456 | Matches GoldRegimeX.mq5 |
| SL | ATR × 2.0 below/above entry | Hard stop on every trade |
| TP | None | No fixed take-profit |

IOC means: if the broker cannot fill the order within the deviation window,
the order is cancelled entirely (never partially filled).  This prevents
unexpected partial exposure during gold news spikes.

---

## Troubleshooting

| Problem | Likely Cause | Fix |
|---|---|---|
| `ConnectionError: Could not connect to MT5` | Terminal not running | Open MT5 and log in first |
| `FileNotFoundError: models/hmm_model.pkl` | Models not trained | Run `--mode train` |
| Validation status FAIL every day | Model too old for current market | Run `--mode optimize` then `--mode train` |
| `Order failed: retcode=10006` | No connection to broker | Check MT5 connection indicator (bottom-right) |
| `Order failed: retcode=10015` | Price changed faster than deviation allows | Will retry on next bar; consider longer TF |
| `Insufficient margin` repeated | Account below minimum for lot size | Top up account or reduce `--balance` value |
| Signals appear but no trades fired | `dry_run=True` (demo mode) | Switch to `--account live` after confirming signals look correct |
| Double positions appearing | EA still on chart | Remove GoldRegimeX.mq5 EA from XAUUSD chart |

---

## Emergency Stop

Press **Ctrl+C** in the terminal window.  The loop handles `KeyboardInterrupt`
gracefully: it logs the shutdown message and calls `mt5.shutdown()` to cleanly
disconnect the Python package.

All open positions remain open in MT5 after the script exits.  You must close
them manually from the Trade tab or by placing an opposing order.

---

## Complete Workflow Example (Headway Cent, $15 balance, H1)

```bash
# 1. First time setup
python main.py --mode process --tf H1
python main.py --mode optimize --trials 250 --broker headway_cent --balance 15 --tf H1
python main.py --mode train --broker headway_cent --balance 15 --tf H1

# 2. Daily routine (before market open)
python main.py --mode sync_validate --period 3m --broker headway_cent --balance 15 --tf H1
#   → PASS: continue
#   → WARN: continue with reduced size or check market conditions
#   → FAIL: run optimize + train again before going live

# 3. Paper trade for several days to verify signals
python main.py --mode live --account demo --broker headway_cent --balance 15 --tf H1

# 4. Go live when satisfied
python main.py --mode live --account live --broker headway_cent --balance 15 --tf H1
```
