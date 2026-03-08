"""
233740 종목 분봉 기술적 분석 및 전략 백테스트
"""

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

# ── 데이터 로드 ────────────────────────────────────────────────────────────────
df = pd.read_csv('minute_chart_233740.csv', encoding='utf-8-sig')
df = df.sort_values(['일자', '시간']).reset_index(drop=True)

# datetime 인덱스 생성
df['dt'] = pd.to_datetime(df['일자'].astype(str) + df['시간'].astype(str).str.zfill(4),
                          format='%Y%m%d%H%M')
df.set_index('dt', inplace=True)

print("=" * 65)
print("  233740 분봉 데이터 기본 정보")
print("=" * 65)
print(f"  기간       : {df['일자'].min()} ~ {df['일자'].max()}")
print(f"  거래일 수  : {df['일자'].nunique()} 일")
print(f"  분봉 수    : {len(df):,} 개")
print(f"  시작 종가  : {df['종가'].iloc[0]:,} 원")
print(f"  마지막 종가: {df['종가'].iloc[-1]:,} 원")
hold_ret = (df['종가'].iloc[-1] / df['종가'].iloc[0] - 1) * 100
print(f"  단순 보유  : {hold_ret:+.2f}%")
print()

# ── 지표 계산 함수 ─────────────────────────────────────────────────────────────

def ema(series, n):
    return series.ewm(span=n, adjust=False).mean()

def rsi(series, n=14):
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/n, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/n, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))

def macd(series, fast=12, slow=26, signal=9):
    macd_line = ema(series, fast) - ema(series, slow)
    signal_line = ema(macd_line, signal)
    hist = macd_line - signal_line
    return macd_line, signal_line, hist

def bollinger(series, n=20, k=2):
    mid = series.rolling(n).mean()
    std = series.rolling(n).std()
    return mid + k * std, mid, mid - k * std

def atr(high, low, close, n=14):
    tr = pd.concat([
        high - low,
        (high - close.shift()).abs(),
        (low  - close.shift()).abs()
    ], axis=1).max(axis=1)
    return tr.ewm(alpha=1/n, adjust=False).mean()

# ── 지표 계산 ─────────────────────────────────────────────────────────────────
close = df['종가'].astype(float)
high  = df['고가'].astype(float)
low   = df['저가'].astype(float)
vol   = df['거래량'].astype(float)

df['rsi14']      = rsi(close, 14)
df['ema5']       = ema(close, 5)
df['ema20']      = ema(close, 20)
df['ema60']      = ema(close, 60)
df['macd_line'], df['macd_sig'], df['macd_hist'] = macd(close)
df['bb_upper'], df['bb_mid'], df['bb_lower']     = bollinger(close, 20, 2)
df['atr14']      = atr(high, low, close, 14)
df['vol_ma20']   = vol.rolling(20).mean()
df['vol_ratio']  = vol / df['vol_ma20']

# ── 백테스트 엔진 ──────────────────────────────────────────────────────────────

def backtest(signals, close, slippage=0.001, fee=0.0003):
    """
    signals: +1 매수, -1 매도(청산), 0 유지
    단순 롱 전략: 매수 → 익절/손절 없이 다음 신호 매도 시 청산
    수수료·슬리피지 양방향 적용
    """
    equity   = 1.0
    position = 0
    buy_price = 0.0
    trades = []

    prices = close.values
    sigs   = signals.values

    for i in range(1, len(prices)):
        price = prices[i]
        sig   = sigs[i - 1]  # 전봉 신호로 현재봉 체결

        if sig == 1 and position == 0:
            buy_price = price * (1 + slippage)
            equity   -= equity * fee
            position  = 1

        elif sig == -1 and position == 1:
            sell_price = price * (1 - slippage)
            ret        = sell_price / buy_price - 1
            equity    *= (1 + ret)
            equity    -= equity * fee
            trades.append(ret)
            position   = 0

    # 미청산 포지션 마지막 종가로 강제 청산
    if position == 1:
        sell_price = prices[-1] * (1 - slippage)
        ret        = sell_price / buy_price - 1
        equity    *= (1 + ret)
        trades.append(ret)

    return equity - 1, trades


def metrics(total_ret, trades, close):
    n   = len(trades)
    if n == 0:
        return dict(total=0, trades=0, win_rate=0, avg_ret=0, mdd=0, sharpe=0)

    win     = sum(1 for r in trades if r > 0)
    win_r   = win / n * 100
    avg_r   = np.mean(trades) * 100
    # MDD: 누적 equity curve로 계산
    equity_curve = np.cumprod([1 + r for r in trades])
    peak = np.maximum.accumulate(equity_curve)
    dd   = (equity_curve - peak) / peak
    mdd  = dd.min() * 100
    # Sharpe (일별 수익률 기준 근사)
    if np.std(trades) > 0:
        sharpe = np.mean(trades) / np.std(trades) * np.sqrt(252 * 6.5 * 60)
    else:
        sharpe = 0

    return dict(
        total    = total_ret * 100,
        trades   = n,
        win_rate = win_r,
        avg_ret  = avg_r,
        mdd      = mdd,
        sharpe   = sharpe,
    )

# ── 전략 정의 ─────────────────────────────────────────────────────────────────

results = {}

# 1. EMA 골든/데드크로스 (5 × 20)
sig1 = pd.Series(0, index=df.index)
sig1[df['ema5'] > df['ema20']]  = 1   # 매수 유지
sig1[df['ema5'] <= df['ema20']] = -1  # 매도/관망

# 크로스 시점만 신호
buy1  = ((df['ema5'] > df['ema20']) & (df['ema5'].shift() <= df['ema20'].shift())).astype(int)
sell1 = ((df['ema5'] < df['ema20']) & (df['ema5'].shift() >= df['ema20'].shift())).astype(int)
sig1  = buy1.replace(0, np.nan) * 1 - sell1.replace(0, np.nan)
sig1  = sig1.ffill().fillna(0).clip(-1, 1)
# 정확히 크로스 시그널만
entry1 = buy1.astype(float)
exit1  = sell1.astype(float)
combined1 = entry1 - exit1
combined1 = combined1.where(combined1 != 0).ffill()
# 간단하게: 골든크로스=1, 데드크로스=-1 forward-fill
state1 = pd.Series(0.0, index=df.index)
for i in range(1, len(df)):
    if buy1.iloc[i]:
        state1.iloc[i] = 1
    elif sell1.iloc[i]:
        state1.iloc[i] = -1
    else:
        state1.iloc[i] = state1.iloc[i-1]

tot1, tr1 = backtest(state1, close)
results['EMA 5×20 크로스'] = metrics(tot1, tr1, close)

# 2. RSI 역추세 (과매도 매수, 과매수 매도)
state2 = pd.Series(0.0, index=df.index)
pos = 0
for i in range(1, len(df)):
    r = df['rsi14'].iloc[i-1]
    if r < 30 and pos == 0:
        state2.iloc[i] = 1
        pos = 1
    elif r > 70 and pos == 1:
        state2.iloc[i] = -1
        pos = 0
    else:
        state2.iloc[i] = state2.iloc[i-1] if pos == 1 else 0

tot2, tr2 = backtest(state2, close)
results['RSI 역추세 (30/70)'] = metrics(tot2, tr2, close)

# 3. RSI 역추세 완화 (35/65)
state3 = pd.Series(0.0, index=df.index)
pos = 0
for i in range(1, len(df)):
    r = df['rsi14'].iloc[i-1]
    if r < 35 and pos == 0:
        state3.iloc[i] = 1
        pos = 1
    elif r > 65 and pos == 1:
        state3.iloc[i] = -1
        pos = 0
    else:
        state3.iloc[i] = state3.iloc[i-1] if pos == 1 else 0

tot3, tr3 = backtest(state3, close)
results['RSI 역추세 (35/65)'] = metrics(tot3, tr3, close)

# 4. MACD 크로스
state4 = pd.Series(0.0, index=df.index)
pos = 0
macd_buy  = (df['macd_hist'] > 0) & (df['macd_hist'].shift() <= 0)
macd_sell = (df['macd_hist'] < 0) & (df['macd_hist'].shift() >= 0)
for i in range(1, len(df)):
    if macd_buy.iloc[i-1] and pos == 0:
        state4.iloc[i] = 1
        pos = 1
    elif macd_sell.iloc[i-1] and pos == 1:
        state4.iloc[i] = -1
        pos = 0
    else:
        state4.iloc[i] = 1.0 if pos == 1 else 0.0

tot4, tr4 = backtest(state4, close)
results['MACD 히스토그램 크로스'] = metrics(tot4, tr4, close)

# 5. 볼린저밴드 하단 터치 매수 → 중심선 매도
state5 = pd.Series(0.0, index=df.index)
pos = 0
for i in range(1, len(df)):
    c  = close.iloc[i-1]
    bl = df['bb_lower'].iloc[i-1]
    bm = df['bb_mid'].iloc[i-1]
    if c <= bl and pos == 0:
        state5.iloc[i] = 1
        pos = 1
    elif c >= bm and pos == 1:
        state5.iloc[i] = -1
        pos = 0
    else:
        state5.iloc[i] = 1.0 if pos == 1 else 0.0

tot5, tr5 = backtest(state5, close)
results['볼린저밴드 하단↔중심'] = metrics(tot5, tr5, close)

# 6. 볼린저밴드 돌파 매수 (상단 돌파 → 추세 추종)
state6 = pd.Series(0.0, index=df.index)
pos = 0
for i in range(1, len(df)):
    c  = close.iloc[i-1]
    bu = df['bb_upper'].iloc[i-1]
    bm = df['bb_mid'].iloc[i-1]
    if c >= bu and pos == 0:
        state6.iloc[i] = 1
        pos = 1
    elif c <= bm and pos == 1:
        state6.iloc[i] = -1
        pos = 0
    else:
        state6.iloc[i] = 1.0 if pos == 1 else 0.0

tot6, tr6 = backtest(state6, close)
results['볼린저밴드 상단 돌파 추세'] = metrics(tot6, tr6, close)

# 7. 복합: EMA 추세 필터 + RSI 과매도 진입
#    EMA60 위에서 RSI < 40 → 매수, RSI > 60 → 매도
state7 = pd.Series(0.0, index=df.index)
pos = 0
for i in range(1, len(df)):
    r   = df['rsi14'].iloc[i-1]
    c   = close.iloc[i-1]
    e60 = df['ema60'].iloc[i-1]
    if r < 40 and c > e60 and pos == 0:
        state7.iloc[i] = 1
        pos = 1
    elif r > 60 and pos == 1:
        state7.iloc[i] = -1
        pos = 0
    else:
        state7.iloc[i] = 1.0 if pos == 1 else 0.0

tot7, tr7 = backtest(state7, close)
results['EMA60↑ + RSI 40/60'] = metrics(tot7, tr7, close)

# 8. 거래량 급증 + 양봉 돌파 모멘텀
#    전봉 대비 거래량 2배 이상 + 양봉 → 매수, 3봉 후 청산
state8 = pd.Series(0.0, index=df.index)
pos = 0
hold_cnt = 0
HOLD_N = 5  # 5분 보유 후 청산
for i in range(1, len(df)):
    vr = df['vol_ratio'].iloc[i-1]
    up = close.iloc[i-1] > df['시가'].iloc[i-1]
    if vr > 2.0 and up and pos == 0:
        state8.iloc[i] = 1
        pos = 1
        hold_cnt = 0
    elif pos == 1:
        hold_cnt += 1
        if hold_cnt >= HOLD_N:
            state8.iloc[i] = -1
            pos = 0
            hold_cnt = 0
        else:
            state8.iloc[i] = 1.0
    else:
        state8.iloc[i] = 0.0

tot8, tr8 = backtest(state8, close)
results['거래량 급증 모멘텀 (5봉)'] = metrics(tot8, tr8, close)

# 9. 장 초반 모멘텀 (09:01~09:10 방향으로 09:10~15:20 추세 추종)
#    당일 09:10 가격이 09:01 시가보다 높으면 매수, 장 마감 청산
daily_groups = df.groupby('일자')

open_dir_trades = []
for date, grp in daily_groups:
    grp = grp.sort_index()
    early = grp.between_time('09:01', '09:10')
    if len(early) < 2:
        continue
    open_px  = early['종가'].iloc[0]
    break_px = early['종가'].iloc[-1]
    rest     = grp.between_time('09:11', '15:20')
    if len(rest) == 0:
        continue
    entry_px = rest['종가'].iloc[0]
    exit_px  = rest['종가'].iloc[-1]
    if break_px > open_px:      # 상승 방향
        ret = exit_px / entry_px - 1
    else:                        # 하락 방향 (매수 전략이므로 관망)
        continue
    open_dir_trades.append(ret)

if open_dir_trades:
    equity9 = np.prod([1 + r for r in open_dir_trades]) - 1
    results['장초반 모멘텀 (일별)'] = metrics(equity9, open_dir_trades, close)

# ── 결과 출력 ─────────────────────────────────────────────────────────────────

print("=" * 65)
print("  전략별 백테스트 결과 (수수료 0.03% + 슬리피지 0.1% 양방향)")
print("=" * 65)
hdr = f"{'전략':<24} {'총수익':>8} {'거래':>6} {'승률':>7} {'평균':>7} {'MDD':>8} {'Sharpe':>7}"
print(hdr)
print("-" * 65)

buy_and_hold = hold_ret

ranked = sorted(results.items(), key=lambda x: x[1]['total'], reverse=True)
for name, m in ranked:
    print(f"{name:<24} {m['total']:>+7.1f}% {m['trades']:>6} "
          f"{m['win_rate']:>6.1f}% {m['avg_ret']:>+6.2f}% "
          f"{m['mdd']:>+7.1f}% {m['sharpe']:>7.2f}")

print("-" * 65)
print(f"{'바이앤홀드 (기준선)':<24} {buy_and_hold:>+7.1f}%")
print()

# ── 연도별 성과 (최우수 전략) ─────────────────────────────────────────────────
best_name, best_m = ranked[0]
print(f"  ▶ 최우수 전략: [{best_name}]  총수익 {best_m['total']:+.1f}%")
print()

# ── 추가 통계: 시간대별 평균 수익률 ──────────────────────────────────────────
df['hour'] = df.index.hour
df['minute'] = df.index.minute
df['time_slot'] = df['hour'] * 100 + df['minute']
df['ret_1m'] = close.pct_change() * 100

print("=" * 65)
print("  시간대별 1분 평균 수익률 (상위 10 / 하위 10)")
print("=" * 65)
slot_ret = df.groupby('time_slot')['ret_1m'].mean()
print("  [상위 - 많이 오르는 시간대]")
for ts, r in slot_ret.nlargest(10).items():
    h, m = divmod(ts, 100)
    print(f"    {h:02d}:{m:02d}   {r:+.4f}%")
print("  [하위 - 많이 내리는 시간대]")
for ts, r in slot_ret.nsmallest(10).items():
    h, m = divmod(ts, 100)
    print(f"    {h:02d}:{m:02d}   {r:+.4f}%")
print()

# ── 요일별 평균 수익률 ────────────────────────────────────────────────────────
df['weekday'] = df.index.weekday  # 0=월 4=금
day_map = {0:'월',1:'화',2:'수',3:'목',4:'금'}
day_ret = df.groupby(['일자','weekday'])['ret_1m'].sum().reset_index()
day_ret2 = day_ret.groupby('weekday')['ret_1m'].mean()
print("=" * 65)
print("  요일별 일중 평균 총수익률")
print("=" * 65)
for d, r in day_ret2.items():
    print(f"    {day_map.get(d,'?')}   {r:+.4f}%")
print()

# ── 분봉 변동성 분석 ──────────────────────────────────────────────────────────
df['range_pct'] = (high - low) / low * 100
print("=" * 65)
print("  분봉 가격 범위 통계 (고가-저가 / 저가)")
print("=" * 65)
print(f"    평균  변동 폭: {df['range_pct'].mean():.4f}%")
print(f"    중앙값 변동 폭: {df['range_pct'].median():.4f}%")
print(f"    90 분위  : {df['range_pct'].quantile(0.9):.4f}%")
print(f"    99 분위  : {df['range_pct'].quantile(0.99):.4f}%")
print()

# ── 종합 의견 ─────────────────────────────────────────────────────────────────
print("=" * 65)
print("  종합 분석 의견")
print("=" * 65)
best_vs_bah = best_m['total'] - buy_and_hold
print(f"  · 바이앤홀드 대비 최우수 전략 초과 수익: {best_vs_bah:+.1f}%p")
profitable = [(n, m) for n, m in results.items() if m['total'] > 0]
print(f"  · 수익 플러스 전략 수: {len(profitable)} / {len(results)}")
beat_bah = [(n, m) for n, m in results.items() if m['total'] > buy_and_hold]
print(f"  · 바이앤홀드 초과 전략 수: {len(beat_bah)} / {len(results)}")
print()
