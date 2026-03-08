"""
233740 VB(Volume Breakout) 신호 — Factor 탐색
★ 미래참조 없음: VB 진입 시점(09:00~09:14)에 알 수 있는 데이터만 사용

사용 가능 데이터:
  - 전일 종가·시가·고가·저가·거래량, MA, RSI, BB, ATR (모두 shift(1))
  - 당일 시가(gap_pct 계산 가능)
  - VB 발생 시각 & 강도 (분봉 실시간)
  - VB 발생까지 누적 거래량 (분봉 실시간)

사용 불가 데이터:
  - 당일 종가·고가·저가
  - 당일 일간 거래량 합계
"""
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

FEE   = 0.0013
ALPHA = 0.25

# ── 일봉 생성 ─────────────────────────────────────────────────────────────────
df_min = pd.read_csv('minute_chart_233740.csv', encoding='utf-8-sig')
df_min = df_min.sort_values(['일자','시간']).reset_index(drop=True)
df_min['일자'] = df_min['일자'].astype(int)
df_min['dt']  = pd.to_datetime(
    df_min['일자'].astype(str) + df_min['시간'].astype(str).str.zfill(4),
    format='%Y%m%d%H%M')

daily = (df_min.groupby('일자')
         .agg(open=('시가','first'), high=('고가','max'),
              low=('저가','min'), close=('종가','last'), vol=('거래량','sum'))
         .reset_index().sort_values('일자').reset_index(drop=True))
daily['date']    = pd.to_datetime(daily['일자'].astype(str), format='%Y%m%d')
daily['weekday'] = daily['date'].dt.weekday
c = daily['close']

# ── 기술적 지표 (모두 당일 종가 기반) ────────────────────────────────────────
for n in [5, 10, 20, 60]:
    daily[f'ma{n}'] = c.rolling(n).mean()

tr = pd.concat([daily['high'] - daily['low'],
                (daily['high'] - c.shift()).abs(),
                (daily['low']  - c.shift()).abs()], axis=1).max(axis=1)
daily['atr14'] = tr.ewm(alpha=1/14, adjust=False).mean()

bb_mid = c.rolling(20).mean()
bb_std = c.rolling(20).std()
bb_up  = bb_mid + 2*bb_std
bb_lo  = bb_mid - 2*bb_std
daily['bb_pct']   = (c - bb_lo) / (bb_up - bb_lo).replace(0, np.nan) * 100
daily['above_bb'] = c > bb_up

def rsi_fn(s, n=14):
    d = s.diff()
    g = d.clip(lower=0).ewm(alpha=1/n, adjust=False).mean()
    l = (-d.clip(upper=0)).ewm(alpha=1/n, adjust=False).mean()
    return 100 - 100 / (1 + g / l.replace(0, np.nan))
daily['rsi14'] = rsi_fn(c, 14)

daily['vol_ma20']  = daily['vol'].rolling(20).mean()
daily['vol_ratio'] = daily['vol'] / daily['vol_ma20']

# ── 전일(shift 1) 지표 — VB 진입 시점에 알 수 있는 값들 ────────────────────
daily['prev_close']    = c.shift(1)
daily['prev_open']     = daily['open'].shift(1)
daily['prev_high']     = daily['high'].shift(1)
daily['prev_low']      = daily['low'].shift(1)
daily['prev_range']    = (daily['high'] - daily['low']).shift(1)
daily['next_open']     = daily['open'].shift(-1)

daily['gap_pct']       = (daily['open'] - daily['prev_close']) / daily['prev_close'] * 100
daily['gap_up']        = daily['gap_pct'] > 0

# 전일 MA (shift(1) = 어제 장 마감 시점 MA)
for n in [5, 10, 20, 60]:
    daily[f'prev_ma{n}'] = daily[f'ma{n}'].shift(1)

# 전일 종가 vs 전일 MA 괴리
daily['prev_dist_ma5']  = (daily['prev_close'] / daily['prev_ma5']  - 1) * 100
daily['prev_dist_ma10'] = (daily['prev_close'] / daily['prev_ma10'] - 1) * 100
daily['prev_dist_ma20'] = (daily['prev_close'] / daily['prev_ma20'] - 1) * 100
daily['prev_dist_ma60'] = (daily['prev_close'] / daily['prev_ma60'] - 1) * 100

# 당일 시가 vs 전일 MA (시가는 장 시작에 알 수 있음)
daily['open_vs_pma5']  = (daily['open'] / daily['prev_ma5']  - 1) * 100
daily['open_vs_pma10'] = (daily['open'] / daily['prev_ma10'] - 1) * 100
daily['open_vs_pma20'] = (daily['open'] / daily['prev_ma20'] - 1) * 100

# 전일 정배열
daily['prev_ma5_gt_ma20'] = daily['prev_ma5'] > daily['prev_ma20']
daily['prev_ma20_gt_ma60']= daily['prev_ma20'] > daily['prev_ma60']

# 전일 RSI·BB·ATR
daily['prev_rsi14']    = daily['rsi14'].shift(1)
daily['prev_bb_pct']   = daily['bb_pct'].shift(1)
daily['prev_above_bb'] = daily['above_bb'].shift(1)
daily['prev_atr_pct']  = (daily['atr14'] / c * 100).shift(1)

# 전일 수익률 & 누적
daily['ret_prev']  = c.pct_change().shift(1) * 100       # 어제 수익률
daily['ret_5d']    = c.pct_change(5).shift(1) * 100      # 전5일 누적 (어제 기준)
daily['ret_10d']   = c.pct_change(10).shift(1) * 100

# 전일 거래량 비율
daily['prev_vol_r']  = daily['vol_ratio'].shift(1)
daily['prev_vol_r2'] = daily['vol_ratio'].shift(2)  # 이틀 전

# 전일 캔들 유형
daily['prev_bearish'] = daily['prev_close'] < daily['prev_open']

daily['week_of_month'] = daily['date'].apply(lambda d: (d.day - 1) // 7 + 1)
daily['year']          = daily['date'].dt.year

# ── 분봉 날짜별 사전 ─────────────────────────────────────────────────────────
min_by_date = {}
for dk, g in df_min.groupby('일자'):
    min_by_date[dk] = g.sort_values('dt').reset_index(drop=True)

# VB 사전계산
day_vb_data = {}
for dk, bars in min_by_date.items():
    dr = daily.loc[daily['일자'] == dk]
    if dr.empty: continue
    pr = float(dr['prev_range'].values[0])
    if np.isnan(pr) or pr <= 0: continue
    d_open = float(dr['open'].values[0])
    highs  = bars['고가'].values.astype(float)
    b_open = bars['시가'].values.astype(float)
    b_vol  = bars['거래량'].values.astype(float)
    times  = (bars['dt'].dt.hour * 60 + bars['dt'].dt.minute).values.astype(int)
    norm   = (highs - d_open) / pr
    day_vb_data[dk] = dict(norm=norm, b_open=b_open, b_vol=b_vol,
                            times=times, d_open=d_open, pr=pr)

# ── VB 발생일 추출 (진입 시점 정보만 포함) ───────────────────────────────────
vb_rows = []
for i in range(65, len(daily) - 1):
    row = daily.iloc[i]
    dk  = int(row['일자'])
    if int(row['weekday']) == 3: continue
    if np.isnan(row['prev_range']) or np.isnan(row['next_open']): continue
    data = day_vb_data.get(dk)
    if data is None: continue

    mask = data['norm'] >= ALPHA
    if not mask.any(): continue
    idx      = mask.argmax()
    t        = data['times'][idx]
    trig_px  = data['d_open'] + ALPHA * data['pr']
    entry_px = max(data['b_open'][idx], trig_px)
    vb_str   = data['norm'][idx]
    exit_px  = float(row['next_open'])
    net      = (exit_px / entry_px - 1 - FEE * 2) * 100

    # VB 발생까지 누적 거래량 (진입 시점 정보)
    vol_to_trig = data['b_vol'][:idx + 1].sum()
    prev_daily_vol = float(row['prev_vol_r'] * row['vol_ma20']) if not np.isnan(row['prev_vol_r']) else np.nan
    # 분당 평균 대비 VB 시점 누적 (전일 일간거래량 / 390분으로 정규화)
    vol_trig_norm = vol_to_trig / (prev_daily_vol / 390 * (t - 540 + 1)) if (
        not np.isnan(prev_daily_vol) and prev_daily_vol > 0 and t > 540) else np.nan

    vb_rows.append({
        **row.to_dict(),
        'net_ret'       : net,
        'win'           : net > 0,
        'entry'         : entry_px,
        'trig_t'        : t,
        't_min'         : t - 540,
        'vb_str'        : vb_str,
        'vb_excess'     : vb_str - ALPHA,
        'vol_trig_norm' : vol_trig_norm,  # VB까지 누적거래량 (전일 분당 대비)
    })

vb = pd.DataFrame(vb_rows)
early = vb[vb['trig_t'] <= 554].copy()

print("=" * 72)
print("  VB Factor 탐색 — 미래참조 없음 (전일 지표 기준)")
print(f"  전체VB: {len(vb)}건  승률:{vb['win'].mean()*100:.1f}%  평균:{vb['net_ret'].mean():+.3f}%")
print(f"  Early(≤09:14): {len(early)}건  승률:{early['win'].mean()*100:.1f}%  평균:{early['net_ret'].mean():+.3f}%")
print("=" * 72)

# ── 공통 함수 ─────────────────────────────────────────────────────────────────
def stats(g):
    if len(g) == 0: return None
    return dict(n=len(g), win=g['win'].mean()*100, avg=g['net_ret'].mean(),
                total=((1+g['net_ret']/100).prod()-1)*100,
                sharpe=g['net_ret'].mean()/g['net_ret'].std()*np.sqrt(252)
                       if g['net_ret'].std()>0 else 0)

def prt(label, mask, df=early):
    pos, neg = df[mask], df[~mask]
    sp, sn = stats(pos), stats(neg)
    if sp is None: return
    diff = sp['avg'] - sn['avg'] if sn else 0
    sym  = '▲' if diff > 0.15 else ('▽' if diff < -0.15 else '─')
    sn_s = f"{sn['n']:>3}건/{sn['win']:>5.1f}%/{sn['avg']:>+6.3f}%" if sn else " —"
    print(f"  {sym} {label:<38}  "
          f"O:{sp['n']:>3}건/{sp['win']:>5.1f}%/{sp['avg']:>+6.3f}%  "
          f"X:{sn_s}  Δ{diff:>+.3f}%")

base = stats(early)

# ══════════════════════════════════════════════════════════════════════════════
print()
print("  [0] VB 시각대별 성과 (전체 기준)")
print("  " + "-"*70)
time_bins = [
    ("09:00~09:04",  (vb['trig_t']>=540)&(vb['trig_t']<=544)),
    ("09:05~09:09",  (vb['trig_t']>=545)&(vb['trig_t']<=549)),
    ("09:10~09:14",  (vb['trig_t']>=550)&(vb['trig_t']<=554)),
    ("09:15~09:29",  (vb['trig_t']>=555)&(vb['trig_t']<=569)),
    ("09:30~09:59",  (vb['trig_t']>=570)&(vb['trig_t']<=599)),
    ("10:00~10:59",  (vb['trig_t']>=600)&(vb['trig_t']<=659)),
    ("11:00~",       vb['trig_t']>=660),
]
for label, mask in time_bins:
    g = vb[mask]
    if len(g) < 3: continue
    print(f"    {label:<16} {len(g):>4}건  승률{g['win'].mean()*100:>5.1f}%  "
          f"평균{g['net_ret'].mean():>+6.3f}%  "
          f"총수익{((1+g['net_ret']/100).prod()-1)*100:>+7.1f}%")

print(f"\n  ※ 이하 Early VB (≤09:14) {len(early)}건 기준 / 기준선 승률{base['win']:.1f}% 평균{base['avg']:+.3f}%")

# ══════════════════════════════════════════════════════════════════════════════
print()
print("  [1] 갭 방향 & 크기  (당일 시가 vs 전일 종가 — 진입 전 확인 가능)")
print("  " + "-"*70)
prt("gap-up",                     early['gap_up'])
prt("gap-up > 0.5%",              early['gap_pct'] > 0.5)
prt("gap-up > 1%",                early['gap_pct'] > 1.0)
prt("gap-up > 2%",                early['gap_pct'] > 2.0)
prt("gap-up > 3%",                early['gap_pct'] > 3.0)
prt("gap-down",                   ~early['gap_up'])
prt("gap-down < -1%",             early['gap_pct'] < -1.0)
prt("gap 중립 (±0.5%)",            early['gap_pct'].abs() <= 0.5)

# ══════════════════════════════════════════════════════════════════════════════
print()
print("  [2] 추세 & MA (★전일 종가 기준)")
print("  " + "-"*70)
prt("전일종가 > 전일MA5",           early['prev_dist_ma5']  > 0)
prt("전일종가 > 전일MA10",          early['prev_dist_ma10'] > 0)
prt("전일종가 > 전일MA20",          early['prev_dist_ma20'] > 0)
prt("전일종가 > 전일MA60",          early['prev_dist_ma60'] > 0)
prt("전일MA5 > 전일MA20 (정배열)",  early['prev_ma5_gt_ma20'])
prt("전일MA20 > 전일MA60",          early['prev_ma20_gt_ma60'])
prt("시가 > 전일MA5",              early['open_vs_pma5']  > 0)
prt("시가 > 전일MA10",             early['open_vs_pma10'] > 0)
prt("시가 > 전일MA20",             early['open_vs_pma20'] > 0)
prt("전일MA20 괴리 > +3%",         early['prev_dist_ma20'] > 3)
prt("전일MA20 괴리 +0~+3%",        (early['prev_dist_ma20']>0)&(early['prev_dist_ma20']<=3))
prt("전일MA5 괴리 > +2%",          early['prev_dist_ma5'] > 2)

# ══════════════════════════════════════════════════════════════════════════════
print()
print("  [3] 모멘텀 (★전일 수익률 기준)")
print("  " + "-"*70)
prt("전일 수익률 > 0 (전일 양봉)",   early['ret_prev'] > 0)
prt("전일 수익률 > +1%",            early['ret_prev'] > 1.0)
prt("전일 수익률 > +2%",            early['ret_prev'] > 2.0)
prt("전일 수익률 > +3%",            early['ret_prev'] > 3.0)
prt("전5일 누적 > 0%",              early['ret_5d']  > 0)
prt("전5일 누적 > +3%",             early['ret_5d']  > 3.0)
prt("전5일 누적 > +5%",             early['ret_5d']  > 5.0)
prt("전10일 누적 > 0%",             early['ret_10d'] > 0)
prt("전일 음봉",                    early['prev_bearish'])
prt("전일 수익률 < -1%",            early['ret_prev'] < -1.0)

# ══════════════════════════════════════════════════════════════════════════════
print()
print("  [4] RSI & 볼린저밴드 (★전일 종가 기준)")
print("  " + "-"*70)
prt("전일 RSI > 70 (과매수)",       early['prev_rsi14'] > 70)
prt("전일 RSI > 60",               early['prev_rsi14'] > 60)
prt("전일 RSI > 50",               early['prev_rsi14'] > 50)
prt("전일 RSI 50~70",              (early['prev_rsi14']>=50)&(early['prev_rsi14']<70))
prt("전일 RSI < 50",               early['prev_rsi14'] < 50)
prt("전일 RSI < 40",               early['prev_rsi14'] < 40)
prt("전일 BB 상단 이탈",            early['prev_above_bb'])
prt("전일 BB 위치 > 80%",          early['prev_bb_pct'] > 80)
prt("전일 BB 위치 > 60%",          early['prev_bb_pct'] > 60)
prt("전일 BB 위치 50~80%",         (early['prev_bb_pct']>=50)&(early['prev_bb_pct']<80))
prt("전일 BB 위치 < 50%",          early['prev_bb_pct'] < 50)
prt("전일 BB 위치 < 20%",          early['prev_bb_pct'] < 20)

# ══════════════════════════════════════════════════════════════════════════════
print()
print("  [5] 거래량 (★전일 거래량만 사용 / 당일 일간합계는 미래참조)")
print("  " + "-"*70)
prt("전일 거래량 > MA20×1.5",       early['prev_vol_r'] > 1.5)
prt("전일 거래량 > MA20×2.0",       early['prev_vol_r'] > 2.0)
prt("전일 거래량 > MA20×1.2",       early['prev_vol_r'] > 1.2)
prt("전일 거래량 < MA20×1.0",       early['prev_vol_r'] < 1.0)
prt("전일 거래량 < MA20×0.8",       early['prev_vol_r'] < 0.8)
prt("이틀 전 거래량 > MA20×1.5",    early['prev_vol_r2'] > 1.5)

# ── VB까지 누적 거래량 (분봉 실시간 — 진입 시점 확인 가능)
print()
print("  [5b] VB 발생까지 누적 거래량 (전일 분당평균 대비, 실시간 사용 가능)")
print("  " + "-"*70)
prt("누적거래량 배율 > 2×",         early['vol_trig_norm'] > 2.0)
prt("누적거래량 배율 > 3×",         early['vol_trig_norm'] > 3.0)
prt("누적거래량 배율 > 5×",         early['vol_trig_norm'] > 5.0)
prt("누적거래량 배율 < 1×",         early['vol_trig_norm'] < 1.0)
# 분위수
for lo, hi in [(0,1),(1,2),(2,3),(3,5),(5,10),(10,99)]:
    g = early[(early['vol_trig_norm']>=lo)&(early['vol_trig_norm']<hi)]
    if len(g) < 3: continue
    hi_s = f"{hi}×" if hi < 99 else "∞"
    print(f"    누적거래량 {lo}×~{hi_s:<4}  "
          f"{len(g):>4}건  승률{g['win'].mean()*100:>5.1f}%  "
          f"평균{g['net_ret'].mean():>+6.3f}%")

# ══════════════════════════════════════════════════════════════════════════════
print()
print("  [6] VB 강도 (분봉 실시간 — 진입 시점 확인 가능)")
print("  " + "-"*70)
prt("VB 강도 ≥ 0.30",              early['vb_str'] >= 0.30)
prt("VB 강도 ≥ 0.35",              early['vb_str'] >= 0.35)
prt("VB 강도 ≥ 0.40",              early['vb_str'] >= 0.40)
prt("VB 강도 ≥ 0.50",              early['vb_str'] >= 0.50)
prt("VB 강도 0.25~0.30 (겨우 돌파)", early['vb_str'] < 0.30)
for lo, hi in [(0.25,0.28),(0.28,0.32),(0.32,0.38),(0.38,0.45),(0.45,0.55),(0.55,9)]:
    g = early[(early['vb_str']>=lo)&(early['vb_str']<hi)]
    if len(g) < 3: continue
    hi_s = f"{hi:.2f}" if hi < 9 else "∞   "
    print(f"    VB강도 {lo:.2f}~{hi_s}  "
          f"{len(g):>3}건  승률{g['win'].mean()*100:>5.1f}%  "
          f"평균{g['net_ret'].mean():>+6.3f}%")

# ══════════════════════════════════════════════════════════════════════════════
print()
print("  [7] 변동성 ATR (★전일 기준)")
print("  " + "-"*70)
prt("전일 ATR < 2%",               early['prev_atr_pct'] < 2.0)
prt("전일 ATR < 3%",               early['prev_atr_pct'] < 3.0)
prt("전일 ATR 2~4%",               (early['prev_atr_pct']>=2)&(early['prev_atr_pct']<4))
prt("전일 ATR > 4%",               early['prev_atr_pct'] > 4.0)
prt("전일 ATR > 5%",               early['prev_atr_pct'] > 5.0)

# ══════════════════════════════════════════════════════════════════════════════
print()
print("  [8] 캘린더 팩터")
print("  " + "-"*70)
for wd, nm in [(0,'월'),(1,'화'),(2,'수'),(4,'금')]:
    prt(f"{nm}요일", early['weekday'] == wd)
prt("월초 (1주차)",                early['week_of_month'] == 1)
prt("월말 (4~5주차)",              early['week_of_month'] >= 4)

# ══════════════════════════════════════════════════════════════════════════════
print()
print("=" * 72)
print("  [9] 2중 조합 스크리닝")
print("=" * 72)

candidates = {
    'gap-up':               early['gap_pct'] > 0,
    'gap-up>1%':            early['gap_pct'] > 1.0,
    'gap-up>2%':            early['gap_pct'] > 2.0,
    '전일종가>MA5':          early['prev_dist_ma5']  > 0,
    '전일종가>MA20':         early['prev_dist_ma20'] > 0,
    '전일MA5>MA20(정배열)':  early['prev_ma5_gt_ma20'],
    '시가>전일MA20':         early['open_vs_pma20'] > 0,
    '전일RSI>50':            early['prev_rsi14'] > 50,
    '전일RSI>60':            early['prev_rsi14'] > 60,
    '전일RSI>70':            early['prev_rsi14'] > 70,
    '전일BB>80%':            early['prev_bb_pct'] > 80,
    '전일BB상단이탈':         early['prev_above_bb'],
    '전일양봉':               ~early['prev_bearish'],
    'ret_prev>1%':           early['ret_prev'] > 1.0,
    'ret_prev>2%':           early['ret_prev'] > 2.0,
    'ret_5d>0%':             early['ret_5d'] > 0,
    '전일vol>1.5x':          early['prev_vol_r'] > 1.5,
    '누적vol>3x':            early['vol_trig_norm'] > 3.0,
    'VB강도≥0.35':           early['vb_str'] >= 0.35,
    'VB강도≥0.40':           early['vb_str'] >= 0.40,
    '전일ATR<4%':            early['prev_atr_pct'] < 4.0,
}

combo_res = []
keys = list(candidates.keys())
for i in range(len(keys)):
    for j in range(i+1, len(keys)):
        ka, kb = keys[i], keys[j]
        sub = early[candidates[ka] & candidates[kb]]
        if len(sub) < 8: continue
        sp = stats(sub)
        combo_res.append(dict(combo=f"{ka} & {kb}", **sp))

cdf = pd.DataFrame(combo_res).sort_values('avg', ascending=False)
print(f"\n  {'조합':<44} {'건수':>4}  {'승률':>6}  {'평균':>8}  {'총수익':>8}  Sharpe")
print("  " + "-"*75)
for _, r in cdf.head(25).iterrows():
    print(f"  {r['combo']:<44} {r['n']:>4}건  {r['win']:>5.1f}%  "
          f"{r['avg']:>+7.3f}%  {r['total']:>+7.1f}%  {r['sharpe']:>6.2f}")

# ══════════════════════════════════════════════════════════════════════════════
print()
print("=" * 72)
print("  [10] 연도별 일관성 — 상위 단일 팩터")
print("=" * 72)

top_factors = {
    'gap-up':               early['gap_pct'] > 0,
    'gap-up > 1%':          early['gap_pct'] > 1.0,
    '전일종가 > 전일MA5':    early['prev_dist_ma5'] > 0,
    '전일종가 > 전일MA20':   early['prev_dist_ma20'] > 0,
    '전일MA5 > MA20(정배열)': early['prev_ma5_gt_ma20'],
    '전일RSI > 50':          early['prev_rsi14'] > 50,
    '전일BB 상단 이탈':       early['prev_above_bb'],
    '전일BB > 80%':          early['prev_bb_pct'] > 80,
    '전일 양봉':              ~early['prev_bearish'],
    '누적거래량 > 3×':        early['vol_trig_norm'] > 3.0,
}

for fname, mask in top_factors.items():
    sub = early[mask]
    if len(sub) < 5: continue
    delta = sub['net_ret'].mean() - base['avg']
    print(f"\n  ▸ {fname}  (전체: {len(sub)}건  "
          f"승률{sub['win'].mean()*100:.1f}%  "
          f"평균{sub['net_ret'].mean():+.3f}%  Δ{delta:>+.3f}%)")
    for yr in sorted(early['year'].unique()):
        g = sub[sub['year'] == yr]
        if len(g) < 2: continue
        print(f"    {yr}: {len(g):>3}건  승률{g['win'].mean()*100:>5.1f}%  "
              f"평균{g['net_ret'].mean():>+6.3f}%")

# ══════════════════════════════════════════════════════════════════════════════
print()
print("=" * 72)
print("  [11] 전략별 백테스트 요약")
print("=" * 72)

strategies = [
    ("Early VB 기준선",                        early['trig_t'] >= 540),
    ("gap-up만",                              early['gap_pct'] > 0),
    ("gap-up & 전일종가>MA5",                  (early['gap_pct']>0)&(early['prev_dist_ma5']>0)),
    ("gap-up & 전일종가>MA20",                 (early['gap_pct']>0)&(early['prev_dist_ma20']>0)),
    ("gap-up & 전일MA5>MA20(정배열)",           (early['gap_pct']>0)&(early['prev_ma5_gt_ma20'])),
    ("gap-up & 전일RSI>50",                   (early['gap_pct']>0)&(early['prev_rsi14']>50)),
    ("gap-up & 전일BB>80%",                   (early['gap_pct']>0)&(early['prev_bb_pct']>80)),
    ("gap-up & 전일BB상단이탈",                 (early['gap_pct']>0)&(early['prev_above_bb'])),
    ("gap-up>1% & 전일RSI>50",               (early['gap_pct']>1)&(early['prev_rsi14']>50)),
    ("gap-up>1% & 전일MA5>MA20",             (early['gap_pct']>1)&(early['prev_ma5_gt_ma20'])),
    ("gap-up>2% & 전일RSI>50",               (early['gap_pct']>2)&(early['prev_rsi14']>50)),
    ("전일종가>MA5 & 전일RSI>50",              (early['prev_dist_ma5']>0)&(early['prev_rsi14']>50)),
    ("전일종가>MA5 & 전일BB>80%",              (early['prev_dist_ma5']>0)&(early['prev_bb_pct']>80)),
    ("전일BB>80% & 전일RSI>50",              (early['prev_bb_pct']>80)&(early['prev_rsi14']>50)),
    ("전일BB상단이탈 & 전일종가>MA20",           (early['prev_above_bb'])&(early['prev_dist_ma20']>0)),
    ("gap-up & 전일종가>MA5 & 전일RSI>50",   (early['gap_pct']>0)&(early['prev_dist_ma5']>0)&(early['prev_rsi14']>50)),
    ("gap-up>1% & 전일종가>MA5 & BB>80%",   (early['gap_pct']>1)&(early['prev_dist_ma5']>0)&(early['prev_bb_pct']>80)),
    ("누적vol>3× & gap-up",                  (early['vol_trig_norm']>3)&(early['gap_pct']>0)),
    ("누적vol>3× & 전일종가>MA5",             (early['vol_trig_norm']>3)&(early['prev_dist_ma5']>0)),
]

print(f"\n  {'전략':<44} {'건수':>4}  {'승률':>6}  {'평균':>8}  {'총수익':>8}  Sharpe  Δavg")
print("  " + "-"*82)
for label, mask in strategies:
    g = early[mask]
    sp = stats(g)
    if sp is None: continue
    diff = sp['avg'] - base['avg']
    mark = '★' if diff > 0.3 and sp['n'] >= 10 else '  '
    print(f"  {mark}{label:<44} {sp['n']:>4}건  {sp['win']:>5.1f}%  "
          f"{sp['avg']:>+7.3f}%  {sp['total']:>+7.1f}%  {sp['sharpe']:>6.2f}  Δ{diff:>+.3f}%")

# ══════════════════════════════════════════════════════════════════════════════
print()
print("=" * 72)
print("  [12] 종합 권고")
print("=" * 72)

all_st = []
for label, mask in strategies[1:]:
    g = early[mask]
    sp = stats(g)
    if sp and sp['n'] >= 10:
        all_st.append((label, sp['avg'], sp['win'], sp['n'], sp['sharpe'], sp['avg']-base['avg']))
all_st.sort(key=lambda x: x[1], reverse=True)

print(f"\n  Early VB 기준선: {base['n']}건  승률{base['win']:.1f}%  평균{base['avg']:+.3f}%\n")
print(f"  ★ 평균수익 기준 Top 전략 (≥10건):")
for i, (lbl, avg, win, n, shr, d) in enumerate(all_st[:8], 1):
    print(f"    {i}. {lbl:<44} {n:>3}건  승률{win:>5.1f}%  평균{avg:>+.3f}%  Δ{d:>+.3f}%  S={shr:.2f}")

print(f"""
  VB 개선 핵심 (미래참조 없는 실용 조건):
  ① gap-up 필터     : 시가 갭업 발생일만 (gap-down VB 배제)
  ② 전일 MA 위치    : 전일 종가 > 전일 MA5 (단기 상승추세)
  ③ 전일 RSI       : 전일 RSI > 50 (모멘텀 유지)
  ④ 전일 BB 위치    : 전일 BB 위치 > 80% or 상단 이탈 (강한 모멘텀)
  ⑤ 누적 거래량     : VB 발생까지 누적 거래량 > 전일 분당평균×3
  ⑥ 황금 조합       : gap-up & 전일종가>MA5 & 전일RSI>50
""")
