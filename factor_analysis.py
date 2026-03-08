"""
233740 VB+RP 전략 — 추가 개선 Factor 탐색
기준: RP < 0.2 + gap-down 충족일 전수를 대상으로
      추가 조건이 신호 품질을 얼마나 향상시키는지 검증
"""
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

FEE = 0.0013  # 편도

# ── 데이터 로드 & 일봉 생성 ───────────────────────────────────────────────────
df_min = pd.read_csv('minute_chart_233740.csv', encoding='utf-8-sig')
df_min = df_min.sort_values(['일자', '시간']).reset_index(drop=True)
df_min['일자'] = df_min['일자'].astype(int)
df_min['dt']  = pd.to_datetime(
    df_min['일자'].astype(str) + df_min['시간'].astype(str).str.zfill(4),
    format='%Y%m%d%H%M'
)

daily = (df_min.groupby('일자')
         .agg(open=('시가','first'), high=('고가','max'),
              low=('저가','min'), close=('종가','last'), vol=('거래량','sum'))
         .reset_index().sort_values('일자').reset_index(drop=True))
daily['date']    = pd.to_datetime(daily['일자'].astype(str), format='%Y%m%d')
daily['weekday'] = daily['date'].dt.weekday

# ── 기술적 지표 계산 ─────────────────────────────────────────────────────────
def add_rp(df, w=5):
    h = df['high'].rolling(w).max()
    l = df['low'].rolling(w).min()
    return (df['close'] - l) / (h - l).replace(0, np.nan)

def rsi(s, n=14):
    d   = s.diff()
    g   = d.clip(lower=0).ewm(alpha=1/n, adjust=False).mean()
    l   = (-d.clip(upper=0)).ewm(alpha=1/n, adjust=False).mean()
    return 100 - 100 / (1 + g / l.replace(0, np.nan))

c = daily['close']
daily['rp5']          = add_rp(daily, 5)
daily['prev_close']   = c.shift(1)
daily['prev_open']    = daily['open'].shift(1)
daily['prev_high']    = daily['high'].shift(1)
daily['prev_low']     = daily['low'].shift(1)
daily['prev_range']   = (daily['high'] - daily['low']).shift(1)
daily['next_open']    = daily['open'].shift(-1)
daily['gap_pct']      = (daily['open'] - daily['prev_close']) / daily['prev_close'] * 100
daily['gap_down']     = daily['gap_pct'] < 0

# 수익률 시리즈
daily['ret_1d']  = c.pct_change() * 100                     # 당일 수익률
daily['ret_prev']= c.pct_change().shift(1) * 100            # 전일 수익률
daily['ret_2d']  = c.pct_change(2).shift(1) * 100           # 전2일 수익률
daily['ret_5d']  = c.pct_change(5).shift(1) * 100           # 전5일 누적
daily['ret_10d'] = c.pct_change(10).shift(1) * 100          # 전10일 누적

# 연속 하락일 수 (당일 기준 이전)
consec = []
cnt = 0
for r in daily['ret_prev']:
    if np.isnan(r):
        consec.append(0)
    elif r < 0:
        cnt += 1
        consec.append(cnt)
    else:
        cnt = 0
        consec.append(0)
daily['consec_down'] = consec

# MA 괴리율 & RSI
for n in [5, 10, 20, 60]:
    daily[f'ma{n}']  = c.rolling(n).mean()
daily['dist_ma5']  = (c / daily['ma5']  - 1) * 100
daily['dist_ma20'] = (c / daily['ma20'] - 1) * 100
daily['dist_ma60'] = (c / daily['ma60'] - 1) * 100
daily['rsi14']     = rsi(c, 14)

# 볼린저밴드 위치
bb_mid = c.rolling(20).mean()
bb_std = c.rolling(20).std()
bb_up  = bb_mid + 2 * bb_std
bb_lo  = bb_mid - 2 * bb_std
daily['bb_pct']  = (c - bb_lo) / (bb_up - bb_lo).replace(0, np.nan) * 100  # 0=하단, 100=상단
daily['below_bb'] = c < bb_lo   # 하단 이탈

# ATR
tr = pd.concat([daily['high'] - daily['low'],
                (daily['high'] - c.shift()).abs(),
                (daily['low']  - c.shift()).abs()], axis=1).max(axis=1)
daily['atr14']   = tr.ewm(alpha=1/14, adjust=False).mean()
daily['atr_pct'] = daily['atr14'] / c * 100

# 거래량
daily['vol_ma20']    = daily['vol'].rolling(20).mean()
daily['vol_ratio']   = daily['vol'] / daily['vol_ma20']       # 당일
daily['prev_vol_r']  = daily['vol_ratio'].shift(1)             # 전일
daily['vol_5d_avg']  = daily['vol'].rolling(5).mean().shift(1)
daily['vol_surge']   = daily['vol_ratio'] > 1.5               # 당일 거래량 급증

# 당일 장중 위치 (종가가 오늘 range 어디에?)
daily['close_in_day'] = ((daily['close'] - daily['low']) /
                          (daily['high'] - daily['low']).replace(0, np.nan) * 100)  # 0=저점, 100=고점

# 전일 캔들 타입
daily['prev_bearish']  = daily['prev_close'] < daily['prev_open']   # 전일 음봉
daily['prev_hammer']   = (                                            # 전일 해머
    ((daily['prev_close'] - daily['prev_low']) /
     (daily['prev_high']  - daily['prev_low'] + 1).replace(0,np.nan)) > 0.6
)
daily['gap_vs_range']  = daily['gap_pct'].abs() / daily['prev_range'] * daily['prev_close']  # 갭/전일range 비율
daily['prev_rp5']      = daily['rp5'].shift(1)                       # 전일 RP 값

# 월중 주차
daily['week_of_month'] = daily['date'].apply(lambda d: (d.day - 1) // 7 + 1)
daily['month_end']     = (daily['date'] + pd.Timedelta(days=1)).dt.month != daily['date'].dt.month

# ── RP + gap-down 기준일 추출 ────────────────────────────────────────────────
universe = daily[
    (daily['rp5'] < 0.2) &
    (daily['gap_down']) &
    (~daily['rp5'].isna()) &
    (~daily['next_open'].isna()) &
    (~daily['prev_close'].isna()) &
    (daily['weekday'] != 3) &    # 목요일 스킵
    (daily.index >= 65)
].copy()

universe['net_ret'] = (universe['next_open'] / universe['close'] - 1 - FEE * 2) * 100
universe['win']     = universe['net_ret'] > 0

print("=" * 72)
print("  Factor 탐색 — RP<0.2 + gap-down 기준일 전수 분석")
print(f"  대상: {len(universe)}건  "
      f"기본 승률:{universe['win'].mean()*100:.1f}%  "
      f"기본 평균:{universe['net_ret'].mean():+.3f}%  "
      f"기본 총수익:{((1+universe['net_ret']/100).prod()-1)*100:+.1f}%")
print("=" * 72)

# ── Factor 분석 공통 함수 ─────────────────────────────────────────────────────

def split_analysis(label, mask, df=universe, show_complement=True):
    """mask=True 그룹 / False 그룹 성과 비교. 항상 (sp, sn, diff) 반환."""
    pos = df[mask]
    neg = df[~mask]

    def grp_stats(g):
        if len(g) == 0:
            return None
        tot = (1 + g['net_ret']/100).prod() - 1
        return dict(n=len(g), win=g['win'].mean()*100,
                    avg=g['net_ret'].mean(), total=tot*100,
                    sharpe=g['net_ret'].mean()/g['net_ret'].std()*np.sqrt(252)
                           if g['net_ret'].std()>0 else 0)

    sp = grp_stats(pos)
    sn = grp_stats(neg)
    diff = (sp['avg'] - sn['avg']) if (sp and sn) else 0
    return sp, sn, diff

def print_split(label, mask, df=universe):
    sp, sn, diff = split_analysis(label, mask, df)
    if sp is None:
        return
    symbol = '▲' if diff > 0.2 else ('▽' if diff < -0.2 else '─')
    sn_str = f"{sn['n']:>3}건/{sn['win']:>5.1f}%/{sn['avg']:>+6.3f}%" if sn else "  없음"
    print(f"  {symbol} {label:<32}  "
          f"O:{sp['n']:>3}건/{sp['win']:>5.1f}%/{sp['avg']:>+6.3f}%  "
          f"X:{sn_str}  "
          f"Δavg={diff:>+6.3f}%")

# ── SECTION 1: 가격·추세 기반 팩터 ──────────────────────────────────────────
print()
print("  [1] 가격·추세 기반 팩터")
print(f"  {'팩터':<34} {'조건O':>4} 건/승률/평균  {'조건X':>4} 건/승률/평균  Δavg")
print("  " + "-" * 70)

print_split("close > MA20 (상승추세)",     universe['dist_ma20'] > 0)
print_split("close > MA60 (중기 상승추세)", universe['dist_ma60'] > 0)
print_split("MA20 괴리 -3%↓ (과도낙폭)",  universe['dist_ma20'] < -3)
print_split("MA20 괴리 -5%↓ (심각낙폭)",  universe['dist_ma20'] < -5)
print_split("MA5 괴리 -2%↓",              universe['dist_ma5']  < -2)
print_split("볼린저 하단 이탈",            universe['below_bb'])
print_split("BB 위치 < 20%",              universe['bb_pct'] < 20)
print_split("RSI < 30 (과매도)",          universe['rsi14'] < 30)
print_split("RSI < 40",                  universe['rsi14'] < 40)
print_split("RSI < 50",                  universe['rsi14'] < 50)

# ── SECTION 2: 낙폭·모멘텀 팩터 ─────────────────────────────────────────────
print()
print("  [2] 낙폭·모멘텀 팩터")
print("  " + "-" * 70)

print_split("전일 수익률 < 0 (전일 음봉)",  universe['ret_prev'] < 0)
print_split("전일 수익률 < -1%",            universe['ret_prev'] < -1)
print_split("전일 수익률 < -2%",            universe['ret_prev'] < -2)
print_split("전5일 누적 < -5%",             universe['ret_5d']  < -5)
print_split("전5일 누적 < -3%",             universe['ret_5d']  < -3)
print_split("전10일 누적 < -10%",           universe['ret_10d'] < -10)
print_split("연속 하락 ≥ 2일",             universe['consec_down'] >= 2)
print_split("연속 하락 ≥ 3일",             universe['consec_down'] >= 3)
print_split("RP5 값 < 0.10 (극단 과매도)", universe['rp5'] < 0.10)
print_split("RP5 값 < 0.05",              universe['rp5'] < 0.05)
print_split("전일 RP5 < 0.2 (연속 RP)",   universe['prev_rp5'] < 0.2)

# ── SECTION 3: 갭·시가 팩터 ─────────────────────────────────────────────────
print()
print("  [3] 갭·시가 팩터")
print("  " + "-" * 70)

print_split("갭하락 > 0.3% (의미있는 갭)", universe['gap_pct'] < -0.3)
print_split("갭하락 > 1%",                universe['gap_pct'] < -1.0)
print_split("갭하락 > 2%",                universe['gap_pct'] < -2.0)
print_split("갭/전일Range > 50%",
            universe['gap_pct'].abs() / (universe['prev_range']/universe['prev_close']*100) > 0.5)

# 시가 vs MA 위치
print_split("시가 < MA20 (MA 아래 갭다운)", universe['open'] < universe['ma20'])
print_split("시가 < MA5",                  universe['open'] < universe['ma5'])

# ── SECTION 4: 거래량 팩터 ───────────────────────────────────────────────────
print()
print("  [4] 거래량 팩터")
print("  " + "-" * 70)

print_split("당일 거래량 > MA20×1.5",     universe['vol_ratio'] > 1.5)
print_split("당일 거래량 > MA20×2.0",     universe['vol_ratio'] > 2.0)
print_split("전일 거래량 > MA20×1.5",     universe['prev_vol_r'] > 1.5)
print_split("전일 거래량 > MA20×2.0",     universe['prev_vol_r'] > 2.0)
print_split("전일 거래량 < MA20×0.8",     universe['prev_vol_r'] < 0.8)   # 전일 저거래량

# ── SECTION 5: 캔들·장중 패턴 팩터 ─────────────────────────────────────────
print()
print("  [5] 캔들·장중 패턴 팩터")
print("  " + "-" * 70)

print_split("전일 음봉",                   universe['prev_bearish'])
print_split("ATR < 3% (저변동성)",         universe['atr_pct'] < 3.0)
print_split("ATR < 2%",                    universe['atr_pct'] < 2.0)
print_split("ATR > 4% (고변동성)",         universe['atr_pct'] > 4.0)
print_split("종가 장중 하위20% (저점 마감)",universe['close_in_day'] < 20)
print_split("종가 장중 하위40%",            universe['close_in_day'] < 40)
print_split("종가 장중 상위20% (고점 마감)",universe['close_in_day'] > 80)

# ── SECTION 6: 계절·캘린더 팩터 ─────────────────────────────────────────────
print()
print("  [6] 캘린더 팩터")
print("  " + "-" * 70)

print_split("월요일 발생",                  universe['weekday'] == 0)
print_split("화요일 발생",                  universe['weekday'] == 1)
print_split("수요일 발생",                  universe['weekday'] == 2)
print_split("금요일 발생",                  universe['weekday'] == 4)
print_split("월말 5거래일 이내",            universe['month_end'])
print_split("월초 (1주차)",                universe['week_of_month'] == 1)
print_split("월말 (4~5주차)",              universe['week_of_month'] >= 4)

# ── SECTION 7: 복합 팩터 상위 조합 탐색 ─────────────────────────────────────
print()
print("=" * 72)
print("  [7] 유망 팩터 2중 조합 스크리닝")
print("  (기준: Δavg > +0.5%p 또는 승률 > 80%)")
print("=" * 72)

# 단일 팩터 중 Δavg 상위 선별
candidates = {
    'gap < -1%'        : universe['gap_pct'] < -1.0,
    'gap < -2%'        : universe['gap_pct'] < -2.0,
    'rp5 < 0.10'       : universe['rp5'] < 0.10,
    'ret_prev < -1%'   : universe['ret_prev'] < -1,
    'ret_5d < -3%'     : universe['ret_5d'] < -3,
    'consec ≥2'        : universe['consec_down'] >= 2,
    'MA20 괴리<-3%'    : universe['dist_ma20'] < -3,
    'below_bb'         : universe['below_bb'],
    'RSI<40'           : universe['rsi14'] < 40,
    'vol_ratio>1.5'    : universe['vol_ratio'] > 1.5,
    'close_in_day<40%' : universe['close_in_day'] < 40,
    'open<MA20'        : universe['open'] < universe['ma20'],
    'prev_bearish'     : universe['prev_bearish'],
    'atr<3%'           : universe['atr_pct'] < 3.0,
}

combo_results = []
keys = list(candidates.keys())
for i in range(len(keys)):
    for j in range(i+1, len(keys)):
        ka, kb = keys[i], keys[j]
        mask = candidates[ka] & candidates[kb]
        sub  = universe[mask]
        if len(sub) < 5:
            continue
        avg = sub['net_ret'].mean()
        win = sub['win'].mean() * 100
        tot = (1 + sub['net_ret']/100).prod() - 1
        shr = sub['net_ret'].mean()/sub['net_ret'].std()*np.sqrt(252) if sub['net_ret'].std()>0 else 0
        combo_results.append(dict(a=ka, b=kb, n=len(sub), win=win, avg=avg, total=tot*100, sharpe=shr))

combo_df = pd.DataFrame(combo_results).sort_values('avg', ascending=False)

print(f"\n  {'조합':<42} {'건수':>4} {'승률':>6} {'평균':>8} {'총수익':>8} {'Sharpe':>7}")
print("  " + "-" * 72)
for _, r in combo_df.head(20).iterrows():
    print(f"  {r['a']+' & '+r['b']:<42} {r['n']:>4}건  {r['win']:>5.1f}%  "
          f"{r['avg']:>+7.3f}%  {r['total']:>+7.1f}%  {r['sharpe']:>6.2f}")

# ── SECTION 8: 연속성 분석 — factor가 모든 연도에서 작동하는가 ───────────────
print()
print("=" * 72)
print("  [8] 유망 팩터 연도별 일관성 검증")
print("=" * 72)

top_factors = {
    'gap < -2%'         : universe['gap_pct'] < -2.0,
    'rp5 < 0.10'        : universe['rp5'] < 0.10,
    'RSI < 40'          : universe['rsi14'] < 40,
    'dist_ma20 < -3%'   : universe['dist_ma20'] < -3,
    'consec_down ≥ 2'   : universe['consec_down'] >= 2,
    'close_in_day < 40%': universe['close_in_day'] < 40,
    'below_bb'          : universe['below_bb'],
}

universe['year'] = universe['date'].dt.year
years = sorted(universe['year'].unique())

for fname, mask in top_factors.items():
    sub = universe[mask]
    print(f"\n  ▸ {fname}  (전체: {len(sub)}건  승률{sub['win'].mean()*100:.1f}%  평균{sub['net_ret'].mean():+.3f}%)")
    for yr in years:
        g = sub[sub['year'] == yr]
        if g.empty:
            continue
        print(f"    {yr}: {len(g):>3}건  승률{g['win'].mean()*100:>5.1f}%  평균{g['net_ret'].mean():>+6.3f}%")

# ── SECTION 9: 백테스트 수준 검증 — 유망 조합을 실제 전략에 적용 ─────────────
print()
print("=" * 72)
print("  [9] 백테스트 수준 검증 — RP 조건에 추가 filter 적용")
print("=" * 72)

# 기존 backtest 함수를 재사용하기 위해 임포트
import importlib, sys, os
sys.path.insert(0, os.getcwd())

# 직접 간단 백테스트 구현 (filter 조건 주입)
def bt_with_rp_filter(extra_mask_col, threshold, direction='<', rp_gap_down=True):
    """extra_mask_col: daily의 컬럼명, direction: '<' or '>'"""
    d = daily.copy()
    d['rp5']       = add_rp(d, 5)
    d['gap_down']  = d['gap_pct'] < 0
    trades = []
    for i in range(65, len(d)-1):
        row = d.iloc[i]
        if np.isnan(row['rp5']) or np.isnan(row['next_open']):
            continue
        if int(row['weekday']) == 3:
            continue
        # RP + gap-down 기본 조건
        if row['rp5'] >= 0.2:
            continue
        if rp_gap_down and not row['gap_down']:
            continue
        # 추가 filter
        val = row.get(extra_mask_col, np.nan)
        if np.isnan(val):
            continue
        if direction == '<' and val >= threshold:
            continue
        if direction == '>' and val <= threshold:
            continue
        entry = float(row['close'])
        exit_ = float(row['next_open'])
        net   = (exit_/entry - 1 - FEE*2) * 100
        trades.append(net)
    if not trades:
        return dict(total=0, n=0, win=0, avg=0, sharpe=0)
    arr = np.array(trades)
    eq  = np.cumprod(1 + arr/100)
    return dict(total=(eq[-1]-1)*100, n=len(arr),
                win=(arr>0).mean()*100, avg=arr.mean(),
                sharpe=arr.mean()/arr.std()*np.sqrt(252) if arr.std()>0 else 0)

# 기준선: RP+gap-down만
base_trades = []
for i in range(65, len(daily)-1):
    row = daily.iloc[i]
    if np.isnan(row['rp5']) or np.isnan(row['next_open']): continue
    if int(row['weekday']) == 3: continue
    if row['rp5'] >= 0.2: continue
    if not row['gap_down']: continue
    net = (row['next_open']/row['close'] - 1 - FEE*2) * 100
    base_trades.append(net)

arr_b = np.array(base_trades)
eq_b  = np.cumprod(1 + arr_b/100)
base_bt = dict(total=(eq_b[-1]-1)*100, n=len(arr_b),
               win=(arr_b>0).mean()*100, avg=arr_b.mean(),
               sharpe=arr_b.mean()/arr_b.std()*np.sqrt(252))

print(f"\n  {'조건':<40} {'건수':>4} {'승률':>6} {'평균':>8} {'총수익':>8} {'Sharpe':>7}")
print("  " + "-" * 70)
print(f"  {'기준선 (RP+gap-down)':<40} {base_bt['n']:>4}건  {base_bt['win']:>5.1f}%  "
      f"{base_bt['avg']:>+7.3f}%  {base_bt['total']:>+7.1f}%  {base_bt['sharpe']:>6.2f}")

filters = [
    ('+ gap_pct < -2%',        'gap_pct',       -2.0, '<'),
    ('+ gap_pct < -1%',        'gap_pct',       -1.0, '<'),
    ('+ gap_pct < -0.5%',      'gap_pct',       -0.5, '<'),
    ('+ rp5 < 0.10',           'rp5',            0.10, '<'),
    ('+ RSI14 < 40',           'rsi14',          40,   '<'),
    ('+ RSI14 < 30',           'rsi14',          30,   '<'),
    ('+ dist_ma20 < -3%',      'dist_ma20',      -3.0, '<'),
    ('+ dist_ma20 < -5%',      'dist_ma20',      -5.0, '<'),
    ('+ below_bb = True',      'below_bb',        0.5,  '>'),  # bool > 0.5 = True
    ('+ close_in_day < 40%',   'close_in_day',   40,   '<'),
    ('+ consec_down ≥ 2',      'consec_down',     1.5,  '>'),
    ('+ ret_prev < -1%',       'ret_prev',        -1.0, '<'),
    ('+ vol_ratio > 1.5',      'vol_ratio',       1.5,  '>'),
    ('+ atr_pct < 3%',         'atr_pct',         3.0,  '<'),
]

for label, col, thr, direction in filters:
    m = bt_with_rp_filter(col, thr, direction)
    if m['n'] == 0:
        continue
    imp = m['avg'] - base_bt['avg']
    mark = '★' if imp > 0.3 and m['n'] >= 10 else '  '
    print(f"  {mark}{label:<40} {m['n']:>4}건  {m['win']:>5.1f}%  "
          f"{m['avg']:>+7.3f}%  {m['total']:>+7.1f}%  {m['sharpe']:>6.2f}  Δ{imp:>+.3f}%")

# ── SECTION 10: VB 신호 factor 분석 ─────────────────────────────────────────
print()
print("=" * 72)
print("  [10] VB 신호 factor 분석 (09:14 이전 early VB 기준)")
print("=" * 72)

# VB 발생일 추출 (분봉 필요)
min_by_date = {}
for dk, g in df_min.groupby('일자'):
    min_by_date[dk] = g.sort_values('dt').reset_index(drop=True)

day_vb_data = {}
for dk, bars in min_by_date.items():
    pr     = daily.loc[daily['일자']==dk, 'prev_range'].values
    d_open = daily.loc[daily['일자']==dk, 'open'].values
    if len(pr)==0 or np.isnan(pr[0]) or pr[0]<=0:
        continue
    highs  = bars['고가'].values.astype(float)
    b_open = bars['시가'].values.astype(float)
    times  = (bars['dt'].dt.hour*60 + bars['dt'].dt.minute).values.astype(int)
    day_vb_data[dk] = dict(norm=(highs-d_open[0])/pr[0],
                            b_open=b_open, times=times,
                            d_open=d_open[0], pr=pr[0])

ALPHA = 0.25
vb_rows = []
for i in range(65, len(daily)-1):
    row  = daily.iloc[i]
    dk   = int(row['일자'])
    if int(row['weekday']) == 3: continue
    if np.isnan(row['prev_range']) or np.isnan(row['next_open']): continue
    data = day_vb_data.get(dk)
    if data is None: continue
    mask = data['norm'] >= ALPHA
    if not mask.any(): continue
    idx  = mask.argmax()
    t    = data['times'][idx]
    if t > 9*60+14: continue   # early VB only
    entry_px = max(data['b_open'][idx], data['d_open'] + ALPHA*data['pr'])
    exit_px  = float(row['next_open'])
    net = (exit_px/entry_px - 1 - FEE*2) * 100
    vb_rows.append({**row.to_dict(), 'net_ret': net, 'win': net>0, 'entry': entry_px})

vb_df = pd.DataFrame(vb_rows)
print(f"\n  Early VB 기준선: {len(vb_df)}건  "
      f"승률{vb_df['win'].mean()*100:.1f}%  평균{vb_df['net_ret'].mean():+.3f}%")
print(f"\n  {'팩터':<38} {'O 건/승/평균':<30} {'X 건/승/평균'}")
print("  " + "-" * 72)

vb_factors = {
    'gap-down 발생일':          vb_df['gap_pct'] < 0,
    'gap-up 발생일':            vb_df['gap_pct'] > 0,
    'gap-up > 1%':              vb_df['gap_pct'] > 1.0,
    'close > MA20 (상승추세)':  vb_df['dist_ma20'] > 0,
    'RSI > 50':                 vb_df['rsi14'] > 50,
    'RSI < 50':                 vb_df['rsi14'] < 50,
    'ret_prev > 0 (전일 양봉)': vb_df['ret_prev'] > 0,
    'ret_5d > 0 (5일 상승중)':  vb_df['ret_5d'] > 0,
    'vol_ratio > 2.0 (거래량)': vb_df['vol_ratio'] > 2.0,
    'atr_pct < 3%':             vb_df['atr_pct'] < 3.0,
    '전일 양봉':                vb_df['prev_bearish'] == False,
}

for label, mask in vb_factors.items():
    pos = vb_df[mask]
    neg = vb_df[~mask]
    if len(pos) < 5 or len(neg) < 5:
        continue
    diff = pos['net_ret'].mean() - neg['net_ret'].mean()
    sym  = '▲' if diff > 0.15 else ('▽' if diff < -0.15 else '─')
    print(f"  {sym} {label:<38}  "
          f"{len(pos):>3}건/{pos['win'].mean()*100:>5.1f}%/{pos['net_ret'].mean():>+6.3f}%   "
          f"{len(neg):>3}건/{neg['win'].mean()*100:>5.1f}%/{neg['net_ret'].mean():>+6.3f}%  "
          f"Δ{diff:>+.3f}%")

# ── SECTION 11: 최종 권고 요약 ────────────────────────────────────────────────
print()
print("=" * 72)
print("  [11] 종합 Factor 권고 요약 (RP 신호 개선)")
print("=" * 72)

# 자동 요약 계산
factor_summary = []
for label, col, thr, direction in filters:
    m = bt_with_rp_filter(col, thr, direction)
    if m['n'] >= 8:
        factor_summary.append((label.replace('+ ',''), m['avg'], m['win'], m['n'], m['sharpe']))

factor_summary.sort(key=lambda x: x[1], reverse=True)
print(f"\n  ★ 추가 필터 평균수익 기준 Top 5:")
for i, (lbl, avg, win, n, shr) in enumerate(factor_summary[:5], 1):
    delta = avg - base_bt['avg']
    print(f"    {i}. {lbl:<38}  {n:>3}건  승률{win:>5.1f}%  평균{avg:>+.3f}%  Δ{delta:>+.3f}%  S={shr:.2f}")

print(f"""
  RP+gap-down 기준선: {base_bt['n']}건  승률{base_bt['win']:.1f}%  평균{base_bt['avg']:+.3f}%

  권고 방향:
  ① 갭 크기 필터   : gap_pct < -1% 이상 갭다운만 진입 → 거래 선별
  ② RSI 확인      : RSI < 40 추가 시 과매도 확인 강화
  ③ RP 극단값     : rp5 < 0.10 (극단 과매도) 특히 우수
  ④ 캔들 위치     : 종가가 당일 range 하위 40% 이내 (저점 마감)
  ⑤ 낙폭 확인     : 전일 수익률 < -1% (연속 매도압력 확인)
  ⑥ 조합 권고     : gap < -1% & rp5 < 0.10  또는  gap < -2% & RSI < 40
""")
