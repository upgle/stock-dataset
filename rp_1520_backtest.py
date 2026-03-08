"""
233740 RP 전략 — 동시호가(15:20) 기준 백테스트
─────────────────────────────────────────────────
• 진입가  : 15:20 바의 시가 (동시호가 직전 마지막 정규매매 가격)
• 청산가  : 익일 시초가 (기존과 동일)
• RP 조건 : 15:20까지의 고가·저가·현재가로 RP5 재산출 (미래참조 없음)
  - 전4일 : 일봉 고가·저가·종가 그대로 사용
  - 당일  : 15:20까지의 장중 고가·저가 + 15:20 시가를 현재가로 사용

비교 기준:
  A) 기존  — 15:30 공식 종가 기준 RP + 종가 진입
  B) 신규  — 15:20 기준 RP + 15:20 시가 진입
"""
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

FEE = 0.0013  # 편도

# ── 데이터 로드 ─────────────────────────────────────────────────────────────
df_min = pd.read_csv('minute_chart_233740.csv', encoding='utf-8-sig')
df_min = df_min.sort_values(['일자', '시간']).reset_index(drop=True)
df_min['일자'] = df_min['일자'].astype(int)
df_min['dt']  = pd.to_datetime(
    df_min['일자'].astype(str) + df_min['시간'].astype(str).str.zfill(4),
    format='%Y%m%d%H%M')

# ── 일봉 (공식 종가 기준) ────────────────────────────────────────────────────
daily = (df_min.groupby('일자')
         .agg(open=('시가','first'), high=('고가','max'),
              low=('저가','min'), close=('종가','last'), vol=('거래량','sum'))
         .reset_index().sort_values('일자').reset_index(drop=True))
daily['date']      = pd.to_datetime(daily['일자'].astype(str), format='%Y%m%d')
daily['weekday']   = daily['date'].dt.weekday
daily['prev_close']= daily['close'].shift(1)
daily['next_open'] = daily['open'].shift(-1)
daily['gap_pct']   = (daily['open'] - daily['prev_close']) / daily['prev_close'] * 100
daily['gap_down']  = daily['gap_pct'] < 0

# ── 15:20 기준 일봉 생성 ─────────────────────────────────────────────────────
# 15:20 바의 시가 = 진입가
bar1520 = (df_min[df_min['시간'] == 1520]
           .set_index('일자')[['시가','고가','저가','종가']]
           .rename(columns={'시가':'px1520','고가':'high1520',
                             '저가':'low1520','종가':'close1520'}))

# 15:20까지 장중 고가·저가 (09:00~15:20 포함)
intra = df_min[df_min['시간'] <= 1520].groupby('일자').agg(
    high_1520=('고가','max'),
    low_1520 =('저가','min')).reset_index()

daily = (daily
         .merge(bar1520.reset_index(), on='일자', how='left')
         .merge(intra, on='일자', how='left'))

# 15:20 현재가: 시가(시초) — 동시호가 직전 첫 체결 가능 가격
daily['entry_1520'] = daily['px1520']

# ── 15:20 기준 RP5 산출 ──────────────────────────────────────────────────────
# rolling 5일창 사용: 전4일은 일봉 고·저, 당일은 장중 고·저(15:20 기준)
def rp5_1520(df):
    """당일 고저를 high_1520/low_1520으로 대체한 후 RP5 계산"""
    h5 = df['high_1520'].rolling(5).max()   # 당일 포함 5일 고가
    l5 = df['low_1520'].rolling(5).min()    # 당일 포함 5일 저가
    return (df['px1520'] - l5) / (h5 - l5).replace(0, np.nan)

daily['rp5_1520'] = rp5_1520(daily)

# 기존 RP5 (공식 종가 기준)
def rp5_close(df, w=5):
    h = df['high'].rolling(w).max()
    l = df['low'].rolling(w).min()
    return (df['close'] - l) / (h - l).replace(0, np.nan)

daily['rp5_close'] = rp5_close(daily)

# ── 보조 지표 (당일 종가 기준 — 15:30 이후 확정이지만 참고용) ─────────────
c = daily['close']
for n in [5, 10, 20, 60]:
    daily[f'ma{n}'] = c.rolling(n).mean()
daily['dist_ma20'] = (c / daily['ma20'] - 1) * 100
daily['dist_ma5']  = (c / daily['ma5']  - 1) * 100

def rsi_fn(s, n=14):
    d = s.diff()
    g = d.clip(lower=0).ewm(alpha=1/n, adjust=False).mean()
    l = (-d.clip(upper=0)).ewm(alpha=1/n, adjust=False).mean()
    return 100 - 100 / (1 + g / l.replace(0, np.nan))
daily['rsi14'] = rsi_fn(c, 14)

bb_mid = c.rolling(20).mean()
bb_std = c.rolling(20).std()
bb_lo  = bb_mid - 2 * bb_std
daily['bb_pct']   = (c - bb_lo) / (4 * bb_std).replace(0, np.nan) * 100
daily['below_bb'] = c < bb_lo

tr = pd.concat([daily['high']-daily['low'],
                (daily['high']-c.shift()).abs(),
                (daily['low'] -c.shift()).abs()], axis=1).max(axis=1)
daily['atr14']   = tr.ewm(alpha=1/14, adjust=False).mean()
daily['atr_pct'] = daily['atr14'] / c * 100

daily['vol_ma20']  = daily['vol'].rolling(20).mean()
daily['vol_ratio'] = daily['vol'] / daily['vol_ma20']
daily['prev_vol_r']= daily['vol_ratio'].shift(1)
daily['ret_prev']  = c.pct_change().shift(1) * 100
daily['ret_5d']    = c.pct_change(5).shift(1) * 100
daily['close_in_day'] = ((daily['close'] - daily['low']) /
                          (daily['high'] - daily['low']).replace(0, np.nan) * 100)

# 15:20 기준 장중 위치
daily['px1520_in_day'] = ((daily['px1520'] - daily['low_1520']) /
                           (daily['high_1520'] - daily['low_1520']).replace(0, np.nan) * 100)

# ── 공통 조건 (두 방식 공통 적용) ────────────────────────────────────────────
WARM   = 65
RP_THR = 0.20

def build_universe(df, rp_col, entry_col, label):
    rows = []
    for i in range(WARM, len(df) - 1):
        row = df.iloc[i]
        if int(row['weekday']) == 3: continue
        if np.isnan(row[rp_col]) or np.isnan(row['next_open']): continue
        if np.isnan(row[entry_col]) or row[entry_col] <= 0: continue
        if row[rp_col] >= RP_THR: continue
        if not row['gap_down']: continue
        entry  = float(row[entry_col])
        exit_  = float(row['next_open'])
        net    = (exit_ / entry - 1 - FEE * 2) * 100
        rows.append({**row.to_dict(), 'net_ret': net, 'win': net > 0,
                     'entry_px': entry, 'rp_val': row[rp_col]})
    return pd.DataFrame(rows)

uni_close = build_universe(daily, 'rp5_close', 'close',     '기존(종가)')
uni_1520  = build_universe(daily, 'rp5_1520',  'entry_1520','15:20')

# ── 통계 함수 ────────────────────────────────────────────────────────────────
def stats(g):
    if len(g) == 0: return None
    eq = (1 + g['net_ret'] / 100).cumprod()
    mdd = ((eq - eq.cummax()) / eq.cummax()).min() * 100
    return dict(n=len(g), win=g['win'].mean()*100, avg=g['net_ret'].mean(),
                total=(eq.iloc[-1]-1)*100, mdd=mdd,
                med=g['net_ret'].median(),
                sharpe=g['net_ret'].mean()/g['net_ret'].std()*np.sqrt(252)
                       if g['net_ret'].std()>0 else 0)

def row_str(label, m):
    return (f"  {label:<38} {m['n']:>4}건  승률{m['win']:>5.1f}%  "
            f"평균{m['avg']:>+6.3f}%  중앙{m['med']:>+6.3f}%  "
            f"총{m['total']:>+7.1f}%  MDD{m['mdd']:>+6.1f}%  S={m['sharpe']:>5.2f}")

sc = stats(uni_close)
s1 = stats(uni_1520)

print("=" * 84)
print("  RP 전략 — 동시호가(15:20) vs 공식종가(15:30) 비교")
print("=" * 84)
print()
print(row_str("A. 기존  (종가 기준 RP + 종가 진입)",  sc))
print(row_str("B. 신규  (15:20 기준 RP + 15:20 진입)", s1))
diff_avg = s1['avg'] - sc['avg']
diff_tot = s1['total'] - sc['total']
print(f"\n  진입가 차이 영향: 평균수익 {diff_avg:>+.4f}%  총수익 {diff_tot:>+.2f}%p")

# ── 진입가 차이 분포 ──────────────────────────────────────────────────────────
common = daily[daily['일자'].isin(uni_close['일자'].values) &
               daily['일자'].isin(uni_1520['일자'].values)].copy()
common = common.dropna(subset=['entry_1520','close'])
price_diff = (common['entry_1520'] / common['close'] - 1) * 100
print()
print("  ── 진입가 차이 분포 (15:20 시가 vs 15:30 종가) ──")
print(f"    평균차이: {price_diff.mean():>+.4f}%  중앙값: {price_diff.median():>+.4f}%")
print(f"    std: {price_diff.std():.4f}%  min: {price_diff.min():>+.3f}%  max: {price_diff.max():>+.3f}%")
bins = [(-99,-1),(-1,-0.5),(-0.5,-0.2),(-0.2,0),(0,0.2),(0.2,0.5),(0.5,1),(1,99)]
for lo, hi in bins:
    cnt = ((price_diff >= lo) & (price_diff < hi)).sum()
    pct = cnt / len(price_diff) * 100
    if cnt > 0:
        hi_s = f"{hi:+.1f}%" if abs(hi)<10 else "∞"
        print(f"    {lo:>+.1f}%~{hi_s:<6}  {cnt:>3}건  {pct:>5.1f}%")

# ── RP 값 비교 (동일 유니버스에서 RP_close vs RP_1520) ─────────────────────
print()
print("  ── RP 값 차이 (15:20 기준 vs 종가 기준, 동일 신호 발생일) ──")
# 두 유니버스 공통 날짜
common_dates = set(uni_close['일자'].values) & set(uni_1520['일자'].values)
only_close   = set(uni_close['일자'].values) - set(uni_1520['일자'].values)
only_1520    = set(uni_1520['일자'].values)  - set(uni_close['일자'].values)
print(f"    공통 신호일: {len(common_dates)}건  종가만: {len(only_close)}건  15:20만: {len(only_1520)}건")

# ── 연도별 비교 ──────────────────────────────────────────────────────────────
print()
print("  ── 연도별 성과 비교 ──")
print(f"  {'연도':<8} {'기존_건수':>8} {'기존_승률':>9} {'기존_평균':>9} | "
      f"{'15:20_건수':>10} {'15:20_승률':>10} {'15:20_평균':>10}")
print("  " + "-" * 78)
for yr in sorted(uni_close['date'].dt.year.unique()):
    gc = uni_close[uni_close['date'].dt.year == yr]
    g1 = uni_1520[uni_1520['date'].dt.year == yr]
    sc_y = stats(gc)
    s1_y = stats(g1)
    if sc_y and s1_y:
        print(f"  {yr:<8} {sc_y['n']:>8}건  {sc_y['win']:>7.1f}%  {sc_y['avg']:>+7.3f}%  | "
              f"{s1_y['n']:>8}건  {s1_y['win']:>9.1f}%  {s1_y['avg']:>+9.3f}%")

# ── Factor 분석 (15:20 기준 유니버스) ────────────────────────────────────────
print()
print("=" * 84)
print("  [Factor 분석] — 15:20 기준 유니버스 (당일 종가 기반 지표 사용 가능)")
print("=" * 84)

def prt(label, mask, df=uni_1520):
    pos = df[mask]
    neg = df[~mask]
    if len(pos) < 5: return
    sp = stats(pos)
    sn = stats(neg)
    if sp is None: return
    diff = sp['avg'] - sn['avg'] if sn else 0
    sym  = '▲' if diff > 0.2 else ('▽' if diff < -0.2 else '─')
    sn_s = f"{sn['n']:>3}건/{sn['win']:>5.1f}%/{sn['avg']:>+6.3f}%" if sn else "  없음"
    print(f"  {sym} {label:<36}  "
          f"O:{sp['n']:>3}건/{sp['win']:>5.1f}%/{sp['avg']:>+6.3f}%  "
          f"X:{sn_s}  Δ{diff:>+.3f}%")

base_1520 = s1

print(f"\n  기준선: {base_1520['n']}건  승률{base_1520['win']:.1f}%  "
      f"평균{base_1520['avg']:+.3f}%  Sharpe{base_1520['sharpe']:.2f}")
print()
print("  [1] 가격·추세")
print("  " + "-" * 74)
prt("close > MA20",                    uni_1520['dist_ma20'] > 0)
prt("close > MA5",                     uni_1520['dist_ma5']  > 0)
prt("MA20 괴리 < -3%",                 uni_1520['dist_ma20'] < -3)
prt("MA20 괴리 < -5%",                 uni_1520['dist_ma20'] < -5)
prt("볼린저 하단 이탈",                uni_1520['below_bb'])
prt("BB 위치 < 20%",                   uni_1520['bb_pct'] < 20)
prt("RSI < 30",                        uni_1520['rsi14'] < 30)
prt("RSI < 40",                        uni_1520['rsi14'] < 40)
prt("RSI < 50",                        uni_1520['rsi14'] < 50)

print()
print("  [2] 낙폭·모멘텀")
print("  " + "-" * 74)
prt("전일 수익률 < -1%",               uni_1520['ret_prev'] < -1)
prt("전일 수익률 < -2%",               uni_1520['ret_prev'] < -2)
prt("전5일 누적 < -3%",                uni_1520['ret_5d'] < -3)
prt("RP5(15:20) < 0.10",              uni_1520['rp_val'] < 0.10)
prt("RP5(15:20) < 0.05",              uni_1520['rp_val'] < 0.05)

print()
print("  [3] 갭·시가")
print("  " + "-" * 74)
prt("갭하락 > 0.3%",                   uni_1520['gap_pct'] < -0.3)
prt("갭하락 > 1%",                     uni_1520['gap_pct'] < -1.0)
prt("갭하락 > 2%",                     uni_1520['gap_pct'] < -2.0)

print()
print("  [4] 거래량 (당일 종가 시점 기준)")
print("  " + "-" * 74)
prt("당일 거래량 > MA20×1.5",          uni_1520['vol_ratio'] > 1.5)
prt("당일 거래량 > MA20×2.0",          uni_1520['vol_ratio'] > 2.0)
prt("전일 거래량 > MA20×1.5",          uni_1520['prev_vol_r'] > 1.5)

print()
print("  [5] 장중 위치")
print("  " + "-" * 74)
prt("15:20가 장중 하위20%",            uni_1520['px1520_in_day'] < 20)
prt("15:20가 장중 하위40%",            uni_1520['px1520_in_day'] < 40)
prt("종가 장중 하위20% (저점마감)",     uni_1520['close_in_day'] < 20)
prt("종가 장중 하위40%",               uni_1520['close_in_day'] < 40)
prt("ATR < 3%",                        uni_1520['atr_pct'] < 3.0)

print()
print("  [6] 캘린더")
print("  " + "-" * 74)
for wd, nm in [(0,'월'),(1,'화'),(2,'수'),(4,'금')]:
    prt(f"{nm}요일", uni_1520['weekday'] == wd)

# ── 2중 조합 ──────────────────────────────────────────────────────────────────
print()
print("=" * 84)
print("  [7] 2중 조합 스크리닝 (15:20 유니버스)")
print("=" * 84)

cands = {
    'gap<-1%':          uni_1520['gap_pct'] < -1.0,
    'gap<-2%':          uni_1520['gap_pct'] < -2.0,
    'rp<0.10':          uni_1520['rp_val'] < 0.10,
    'MA20괴리<-3%':     uni_1520['dist_ma20'] < -3,
    'below_bb':         uni_1520['below_bb'],
    'RSI<40':           uni_1520['rsi14'] < 40,
    'RSI<30':           uni_1520['rsi14'] < 30,
    'ret_prev<-1%':     uni_1520['ret_prev'] < -1,
    'ret_5d<-3%':       uni_1520['ret_5d'] < -3,
    'vol>1.5x':         uni_1520['vol_ratio'] > 1.5,
    '1520_하위40%':     uni_1520['px1520_in_day'] < 40,
    '종가_하위40%':     uni_1520['close_in_day'] < 40,
}

combos = []
keys = list(cands.keys())
for i in range(len(keys)):
    for j in range(i+1, len(keys)):
        ka, kb = keys[i], keys[j]
        sub = uni_1520[cands[ka] & cands[kb]]
        if len(sub) < 5: continue
        sp = stats(sub)
        combos.append(dict(combo=f"{ka} & {kb}", **sp))

cdf = pd.DataFrame(combos).sort_values('avg', ascending=False)
print(f"\n  {'조합':<38} {'건수':>4}  {'승률':>6}  {'평균':>8}  {'총수익':>8}  Sharpe")
print("  " + "-" * 72)
for _, r in cdf.head(20).iterrows():
    print(f"  {r['combo']:<38} {r['n']:>4}건  {r['win']:>5.1f}%  "
          f"{r['avg']:>+7.3f}%  {r['total']:>+7.1f}%  {r['sharpe']:>6.2f}")

# ── 최종 요약 ─────────────────────────────────────────────────────────────────
print()
print("=" * 84)
print("  [종합 요약]")
print("=" * 84)
print(f"""
  ┌────────────────────────────────────────────────────────────────────┐
  │  구분           건수    승률     평균수익   총수익   MDD     Sharpe │
  ├────────────────────────────────────────────────────────────────────┤
  │  기존 (종가)   {sc['n']:>4}건  {sc['win']:>5.1f}%  {sc['avg']:>+7.3f}%   {sc['total']:>+7.1f}%  {sc['mdd']:>+6.1f}%  {sc['sharpe']:>5.2f} │
  │  15:20 진입    {s1['n']:>4}건  {s1['win']:>5.1f}%  {s1['avg']:>+7.3f}%   {s1['total']:>+7.1f}%  {s1['mdd']:>+6.1f}%  {s1['sharpe']:>5.2f} │
  └────────────────────────────────────────────────────────────────────┘

  ※ 진입 시점 변경에 따른 차이: 평균 {diff_avg:>+.4f}% / 총수익 {diff_tot:>+.2f}%p
""")
