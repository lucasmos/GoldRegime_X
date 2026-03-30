//+------------------------------------------------------------------+
//|                                                  GoldRegimeX.mq5 |
//|                              Gold Regime X — Hybrid ML EA        |
//+------------------------------------------------------------------+
#property copyright "Gold Regime X"
#property version   "2.00"
#property strict

#resource "\\Files\\xgb_model.onnx" as uchar OnnxModel[]

//--- Trading inputs
input double RiskPercent     = 1.0;    // Risk per trade (%)
input double ProbThreshold   = 0.65;  // XGB probability threshold for Long
input double ShortThreshold  = 0.35;  // XGB probability threshold for Short
input int    ATRPeriod       = 14;    // ATR period
input double ATRMultiplier   = 2.0;   // ATR stop loss multiplier
input int    RSIPeriod       = 14;    // RSI period
input int    ChopState       = 2;     // HMM Chop state index (no trade)
input int    MagicNumber     = 123456;
input int    NStates        = 3;     // Classes in trained model — must match n_states from Python (default 3)

//--- Timeframe input (M15 or H1)
input ENUM_TIMEFRAMES SignalTimeframe = PERIOD_H1;  // Timeframe for ONNX signals

//--- Account / session inputs
//    These are OVERRIDDEN at runtime by AdaptiveScaling logic below.
//    Set only as fallback defaults.
input bool IsCentAccount     = true;  // Headway Cent micro-lot floor

//--- HMM proxy thresholds (update from Python training output)
input double BullReturnThreshold =  0.0001;
input double BearReturnThreshold = -0.0001;
input double VolThreshold        =  0.005;

//--- Handles
long onnx_handle = INVALID_HANDLE;
int  atr_handle  = INVALID_HANDLE;
int  rsi_handle  = INVALID_HANDLE;

//--- Session state
int      session_trade_count = 0;
datetime last_session_day    = 0;

//+------------------------------------------------------------------+
//| Adaptive scaling from Python spec                                |
//|  ≤ $50 USD : max_trades=2, pos_per_trade=1                       |
//|  > $50 USD : max_trades=2|3 (Chop vs Bull/Bear), pos_per_trade=2 |
//+------------------------------------------------------------------+
int GetMaxTrades(double balance, int hmm_state)
{
    if(balance <= 50.0) return 2;
    return (hmm_state == ChopState) ? 2 : 3;
}

int GetPosPerTrade(double balance)
{
    return (balance <= 50.0) ? 1 : 2;
}

//+------------------------------------------------------------------+
void CheckSessionReset()
{
    MqlDateTime dt;
    TimeToStruct(TimeCurrent(), dt);
    datetime today = StringToTime(StringFormat("%04d.%02d.%02d 00:00",
                                               dt.year, dt.mon, dt.day));
    if(today != last_session_day)
    {
        if(last_session_day != 0)
            PrintFormat("New session [%s]: reset count (was %d)",
                        TimeToString(today, TIME_DATE), session_trade_count);
        session_trade_count = 0;
        last_session_day    = today;
    }
}

//+------------------------------------------------------------------+
int OnInit()
{
    onnx_handle = OnnxCreateFromBuffer(OnnxModel, ONNX_DEFAULT);
    if(onnx_handle == INVALID_HANDLE)
    {
        Print("ONNX load failed: ", GetLastError());
        return INIT_FAILED;
    }
    long input_shape[]  = {1, 4};
    long label_shape[]  = {1};
    long prob_shape[]   = {1, NStates};
    if(!OnnxSetInputShape(onnx_handle, 0, input_shape))
    {
        PrintFormat("ONNX input shape error (code %d)", GetLastError());
        return INIT_FAILED;
    }
    if(!OnnxSetOutputShape(onnx_handle, 0, label_shape))
    {
        PrintFormat("ONNX label output[0] shape error (code %d)", GetLastError());
        return INIT_FAILED;
    }
    if(!OnnxSetOutputShape(onnx_handle, 1, prob_shape))
    {
        PrintFormat("ONNX prob output[1] shape error — NStates=%d may not match trained model (code %d)",
                    NStates, GetLastError());
        return INIT_FAILED;
    }

    atr_handle = iATR(_Symbol, SignalTimeframe, ATRPeriod);
    rsi_handle = iRSI(_Symbol, SignalTimeframe, RSIPeriod, PRICE_CLOSE);
    if(atr_handle == INVALID_HANDLE || rsi_handle == INVALID_HANDLE)
    {
        Print("Indicator handle error");
        return INIT_FAILED;
    }

    double bal = AccountInfoDouble(ACCOUNT_BALANCE);
    PrintFormat("GoldRegimeX v2.00 | TF=%s | Balance=%.2f | MaxTrades=%d | PosPerTrade=%d | Cent=%s",
                EnumToString(SignalTimeframe), bal,
                GetMaxTrades(bal, 0), GetPosPerTrade(bal),
                IsCentAccount ? "YES" : "NO");
    return INIT_SUCCEEDED;
}

//+------------------------------------------------------------------+
void OnDeinit(const int reason)
{
    if(onnx_handle != INVALID_HANDLE) OnnxRelease(onnx_handle);
    if(atr_handle  != INVALID_HANDLE) IndicatorRelease(atr_handle);
    if(rsi_handle  != INVALID_HANDLE) IndicatorRelease(rsi_handle);
}

//+------------------------------------------------------------------+
double StdDev(double &arr[], int size)
{
    double sum = 0, sq = 0;
    for(int i = 0; i < size; i++) { sum += arr[i]; sq += arr[i]*arr[i]; }
    double mean = sum / size;
    return MathSqrt(sq/size - mean*mean);
}

int GetHMMState(double kalman_ret, double vol)
{
    if(kalman_ret > BullReturnThreshold && vol < VolThreshold) return 0;
    if(kalman_ret < BearReturnThreshold)                       return 1;
    return 2;
}

bool ComputeFeatures(float &features[], int &hmm_state_out)
{
    ArrayResize(features, 4);
    double cl[];
    if(CopyClose(_Symbol, SignalTimeframe, 0, 22, cl) < 22) return false;

    double rets[];
    ArrayResize(rets, 20);
    for(int i = 0; i < 20; i++) rets[i] = MathLog(cl[i+1] / cl[i]);

    double vol = StdDev(rets, 20);
    double lr  = MathLog(cl[21] / cl[20]);
    hmm_state_out = GetHMMState(lr, vol);
    features[0]   = (float)hmm_state_out;

    double rsi_buf[];
    if(CopyBuffer(rsi_handle, 0, 0, 2, rsi_buf) < 2) return false;
    features[1] = (float)(rsi_buf[1] - rsi_buf[0]);

    double atr_buf[];
    if(CopyBuffer(atr_handle, 0, 0, 1, atr_buf) < 1) return false;
    features[2] = (float)(atr_buf[0] / cl[21]);
    features[3] = (float)MathLog(cl[20] / cl[19]);
    return true;
}

//+------------------------------------------------------------------+
//| Lot sizing with micro-lot floor for Headway Cent                 |
//+------------------------------------------------------------------+
double CalculateLotSize(double stop_pts, int pos_per_trade)
{
    double balance  = AccountInfoDouble(ACCOUNT_BALANCE);
    double risk_amt = balance * RiskPercent / 100.0 / pos_per_trade;  // split across positions
    double tv       = SymbolInfoDouble(_Symbol, SYMBOL_TRADE_TICK_VALUE);
    double ts       = SymbolInfoDouble(_Symbol, SYMBOL_TRADE_TICK_SIZE);
    double min_lot  = SymbolInfoDouble(_Symbol, SYMBOL_VOLUME_MIN);
    double max_lot  = SymbolInfoDouble(_Symbol, SYMBOL_VOLUME_MAX);
    double lot_step = SymbolInfoDouble(_Symbol, SYMBOL_VOLUME_STEP);

    if(IsCentAccount) min_lot = MathMax(min_lot, 0.01);
    if(tv == 0 || ts == 0 || stop_pts == 0) return min_lot;

    double lot = risk_amt / (stop_pts / ts * tv);
    lot = MathFloor(lot / lot_step) * lot_step;
    return MathMax(min_lot, MathMin(max_lot, lot));
}

bool HasOpenPosition()
{
    for(int i = 0; i < PositionsTotal(); i++)
        if(PositionGetSymbol(i) == _Symbol &&
           PositionGetInteger(POSITION_MAGIC) == MagicNumber) return true;
    return false;
}

bool SendOrder(ENUM_ORDER_TYPE type, double price, double sl, double lot, string comment)
{
    MqlTradeRequest req = {};
    MqlTradeResult  res = {};
    req.action  = TRADE_ACTION_DEAL;
    req.symbol  = _Symbol;
    req.volume  = lot;
    req.type    = type;
    req.price   = price;
    req.sl      = sl;
    req.magic   = MagicNumber;
    req.comment = comment;
    return OrderSend(req, res);
}

//+------------------------------------------------------------------+
void OnTick()
{
    CheckSessionReset();

    double balance     = AccountInfoDouble(ACCOUNT_BALANCE);
    float  features[];
    int    hmm_state   = 0;

    // Compute features first so we know the HMM state for adaptive limits
    if(!ComputeFeatures(features, hmm_state)) return;

    int max_trades    = GetMaxTrades(balance, hmm_state);
    int pos_per_trade = GetPosPerTrade(balance);

    // Enforce daily session limit
    if(session_trade_count >= max_trades) return;

    // Only trade on new bar for the configured timeframe
    static datetime last_bar_time = 0;
    datetime cur_bar = iTime(_Symbol, SignalTimeframe, 0);
    if(cur_bar == last_bar_time) return;
    last_bar_time = cur_bar;

    if(HasOpenPosition()) return;

    // ONNX inference
    long  labels[];
    float probs[];
    if(!OnnxRun(onnx_handle, ONNX_DEFAULT, features, labels, probs))
    {
        Print("ONNX run failed: ", GetLastError());
        return;
    }

    // probs[0] = P(Bull/class-0), probs[1] = P(Bear/class-1)
    float bull_prob = probs[0];
    float bear_prob = probs[1];

    double atr_buf[];
    CopyBuffer(atr_handle, 0, 0, 1, atr_buf);
    double stop_dist = atr_buf[0] * ATRMultiplier;
    double lot       = CalculateLotSize(stop_dist, pos_per_trade);

    double ask = SymbolInfoDouble(_Symbol, SYMBOL_ASK);
    double bid = SymbolInfoDouble(_Symbol, SYMBOL_BID);

    // Long signal: high Bull probability
    if(bull_prob > ProbThreshold && hmm_state != ChopState)
    {
        double sl = NormalizeDouble(ask - stop_dist, _Digits);
        bool ok = false;
        for(int p = 0; p < pos_per_trade && session_trade_count < max_trades; p++)
        {
            if(SendOrder(ORDER_TYPE_BUY, ask, sl, lot,
                         StringFormat("GRX Long p%d", p+1)))
            {
                session_trade_count++;
                ok = true;
            }
        }
        if(ok)
            PrintFormat("LONG ×%d | count=%d/%d | lot=%.2f | sl=%.2f | bull_prob=%.3f | state=%d | tf=%s",
                        pos_per_trade, session_trade_count, max_trades,
                        lot, sl, bull_prob, hmm_state, EnumToString(SignalTimeframe));
    }

    // Short signal: high Bear probability (bear_prob > 1 - ShortThreshold, default 0.65)
    if(bear_prob > (1.0f - (float)ShortThreshold) && hmm_state != ChopState &&
       session_trade_count < max_trades)
    {
        double sl = NormalizeDouble(bid + stop_dist, _Digits);
        for(int p = 0; p < pos_per_trade && session_trade_count < max_trades; p++)
        {
            if(SendOrder(ORDER_TYPE_SELL, bid, sl, lot,
                         StringFormat("GRX Short p%d", p+1)))
                session_trade_count++;
        }
        PrintFormat("SHORT ×%d | count=%d/%d | lot=%.2f | bear_prob=%.3f | state=%d | tf=%s",
                    pos_per_trade, session_trade_count, max_trades,
                    lot, bear_prob, hmm_state, EnumToString(SignalTimeframe));
    }
}
//+------------------------------------------------------------------+
