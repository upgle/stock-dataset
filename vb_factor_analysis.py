"""
233740 VB(Volume Breakout) 신호 — Factor 탐색
대상: ALPHA=0.25 기반 VB 발생일 전수 (시간대 구분 포함)
"""
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

FEE    = 0.0013
ALPHA  = 0.25

# ── 데이터 로드 & 일봉 ────────────────────────────────────────────────────────
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
daily['date']     = pd.to_datetime(daily['일자'].astype(str), format='%Y%m%d')
daily['weekday']  = daily['date'].dt.weekday
c = daily['close']

# 기술적 지표
for n in [5,10,20,60]:
    daily[f'ma{n}'] = c.rolling(n).mean()
tr = pd.concat([daily['high']-daily['low'],
                (daily['high']-c.shift()).abs(),
                (daily['low'] -c.shift()).abs()], axis=1).max(axis=1)
daily['atr14']       = tr.ewm(alpha=1/14,adjust=False).mean()
daily['atr_pct']     = daily['atr14']/c*100
daily['prev_close']  = c.shift(1)
daily['prev_open']   = daily['open'].shift(1)
daily['prev_high']   = daily['high'].shift(1)
daily['prev_low']    = daily['low'].shift(1)
daily['prev_range']  = (daily['high']-daily['low']).shift(1)
daily['next_open']   = daily['open'].shift(-1)
daily['gap_pct']     = (daily['open']-daily['prev_close'])/daily['prev_close']*100
daily['gap_up']      = daily['gap_pct'] > 0
daily['ret_1d']      = c.pct_change()*100
daily['ret_prev']    = c.pct_change().shift(1)*100
daily['ret_5d']      = c.pct_change(5).shift(1)*100
daily['ret_10d']     = c.pct_change(10).shift(1)*100
daily['dist_ma5']    = (c/daily['ma5'] -1)*100
daily['dist_ma10']   = (c/daily['ma10']-1)*100
daily['dist_ma20']   = (c/daily['ma20']-1)*100
daily['dist_ma60']   = (c/daily['ma60']-1)*100
daily['vol_ma20']    = daily['vol'].rolling(20).mean()
daily['vol_ratio']   = daily['vol']/daily['vol_ma20']
daily['prev_vol_r']  = daily['vol_ratio'].shift(1)
bb_mid = c.rolling(20).mean()
bb_std = c.rolling(20).std()
bb_up  = bb_mid + 2*bb_std
bb_lo  = bb_mid - 2*bb_std
daily['bb_pct']      = (c-bb_lo)/(bb_up-bb_lo).replace(0,np.nan)*100
daily['above_bb']    = c > bb_up

def rsi_fn(s, n=14):
    d = s.diff()
    g = d.clip(lower=0).ewm(alpha=1/n,adjust=False).mean()
    l = (-d.clip(upper=0)).ewm(alpha=1/n,adjust=False).mean()
    return 100 - 100/(1+g/l.replace(0,np.nan))
daily['rsi14']       = rsi_fn(c,14)
daily['prev_bearish']= daily['prev_close'] < daily['prev_open']
daily['open_vs_ma20']= (daily['open']/daily['ma20']-1)*100  # 시가의 MA20 괴리
daily['week_of_month']= daily['date'].apply(lambda d:(d.day-1)//7+1)
daily['year']        = daily['date'].dt.year

# 분봉 날짜별 딕셔너리
min_by_date = {}
for dk, g in df_min.groupby('일자'):
    min_by_date[dk] = g.sort_values('dt').reset_index(drop=True)

# VB 사전계산
day_vb_data = {}
for dk, bars in min_by_date.items():
    dr = daily.loc[daily['일자']==dk]
    if dr.empty: continue
    pr = float(dr['prev_range'].values[0])
    if np.isnan(pr) or pr <= 0: continue
    d_open = float(dr['open'].values[0])
    highs  = bars['고가'].values.astype(float)
    b_open = bars['시가'].values.astype(float)
    b_close= bars['종가'].values.astype(float)
    b_vol  = bars['거래량'].values.astype(float)
    times  = (bars['dt'].dt.hour*60 + bars['dt'].dt.minute).values.astype(int)
    norm   = (highs - d_open) / pr
    day_vb_data[dk] = dict(norm=norm, b_open=b_open, b_close=b_close,
                            b_vol=b_vol, times=times, d_open=d_open, pr=pr)

# ── VB 발생일 전수 추출 ────────────────────────────────────────────────────────
vb_rows = []
for i in range(65, len(daily)-1):
    row = daily.iloc[i]
    dk  = int(row['일자'])
    if int(row['weekday']) == 3: continue
    if np.isnan(row.get('prev_range',np.nan)) or np.isnan(row['next_open']): continue
    data = day_vb_data.get(dk)
    if data is None: continue
    mask = data['norm'] >= ALPHA
    if not mask.any(): continue

    idx      = mask.argmax()
    t        = data['times'][idx]
    trig_px  = data['d_open'] + ALPHA * data['pr']
    entry_px = max(data['b_open'][idx], trig_px)
    vb_str   = data['norm'][idx]          # VB 강도 (ALPHA 초과분)
    exit_px  = float(row['next_open'])
    net      = (exit_px/entry_px - 1 - FEE*2) * 100

    # 분봉 추가 지표
    t_min = t - 9*60           # 장 시작 후 경과 분 (0=09:00)
    vb_rows.append({
        **row.to_dict(),
        'net_ret' : net,
        'win'     : net > 0,
        'entry'   : entry_px,
        'trig_t'  : t,           # 분봉 시각 (분)
        't_min'   : t_min,       # 장 시작 후 경과
        'vb_str'  : vb_str,      # 정규화 고가 (ALPHA=0.25)
        'vb_excess': vb_str - ALPHA,  # ALPHA 초과분
    })

vb = pd.DataFrame(vb_rows)
N_ALL = len(vb)
WR_ALL = vb['win'].mean()*100
AVG_ALL = vb['net_ret'].mean()

print("="*72)
print("  VB Factor 탐색 — 전수 분석")
print(f"  전체: {N_ALL}건  승률:{WR_ALL:.1f}%  평균:{AVG_ALL:+.3f}%  "
      f"총수익:{((1+vb['net_ret']/100).prod()-1)*100:+.1f}%")
print("="*72)

# ── 공통 함수 ─────────────────────────────────────────────────────────────────
def stats(g):
    if len(g) == 0: return None
    return dict(n=len(g), win=g['win'].mean()*100, avg=g['net_ret'].mean(),
                total=((1+g['net_ret']/100).prod()-1)*100,
                sharpe=g['net_ret'].mean()/g['net_ret'].std()*np.sqrt(252)
                       if g['net_ret'].std()>0 else 0)

def prt(label, mask, df=vb):
    pos, neg = df[mask], df[~mask]
    sp, sn = stats(pos), stats(neg)
    if sp is None: return
    diff = sp['avg'] - sn['avg'] if sn else 0
    sym  = '▲' if diff>0.15 else ('▽' if diff<-0.15 else '─')
    sn_s = f"{sn['n']:>3}건/{sn['win']:>5.1f}%/{sn['avg']:>+6.3f}%" if sn else " —"
    print(f"  {sym} {label:<35}  "
          f"O:{sp['n']:>3}건/{sp['win']:>5.1f}%/{sp['avg']:>+6.3f}%  "
          f"X:{sn_s}  Δ{diff:>+.3f}%")

# ══════════════════════════════════════════════════════════════════════════════
print()
print("  [0] VB 발생 시각대별 성과")
print("  " + "-"*70)
time_bins = [
    ("09:00~09:04 (1분봉)", (vb['trig_t'] >= 540) & (vb['trig_t'] <= 544)),
    ("09:05~09:09",         (vb['trig_t'] >= 545) & (vb['trig_t'] <= 549)),
    ("09:10~09:14",         (vb['trig_t'] >= 550) & (vb['trig_t'] <= 554)),
    ("09:15~09:19",         (vb['trig_t'] >= 555) & (vb['trig_t'] <= 559)),
    ("09:20~09:29",         (vb['trig_t'] >= 560) & (vb['trig_t'] <= 569)),
    ("09:30~09:59",         (vb['trig_t'] >= 570) & (vb['trig_t'] <= 599)),
    ("10:00~10:59",         (vb['trig_t'] >= 600) & (vb['trig_t'] <= 659)),
    ("11:00~13:00",         (vb['trig_t'] >= 660) & (vb['trig_t'] <= 780)),
    ("13:01~15:30",         vb['trig_t'] > 780),
]
for label, mask in time_bins:
    g = vb[mask]
    if len(g) < 3: continue
    tot = (1+g['net_ret']/100).prod()-1
    print(f"    {label:<22} {len(g):>4}건  승률{g['win'].mean()*100:>5.1f}%  "
          f"평균{g['net_ret'].mean():>+6.3f}%  총수익{tot*100:>+7.1f}%")

# 이하 early VB (≤09:14) 만 분석 - 기존 전략과 동일
early = vb[vb['trig_t'] <= 554].copy()
print(f"\n  ※ 이하 분석은 Early VB (≤09:14) 기준")
print(f"     Early: {len(early)}건  승률{early['win'].mean()*100:.1f}%  평균{early['net_ret'].mean():+.3f}%")

def prt_e(label, mask):
    prt(label, mask, df=early)

# ══════════════════════════════════════════════════════════════════════════════
print()
print("  [1] 시가 갭 방향 & 크기")
print("  " + "-"*70)
prt_e("gap-up (시가 > 전일종가)",    early['gap_up'])
prt_e("gap-up > 0.5%",              early['gap_pct'] > 0.5)
prt_e("gap-up > 1%",                early['gap_pct'] > 1.0)
prt_e("gap-up > 2%",                early['gap_pct'] > 2.0)
prt_e("gap-up > 3%",                early['gap_pct'] > 3.0)
prt_e("gap-down (시가 < 전일종가)",  ~early['gap_up'])
prt_e("gap-down < -1%",             early['gap_pct'] < -1.0)
prt_e("gap 중립 (±0.5%)",           early['gap_pct'].abs() <= 0.5)

# ══════════════════════════════════════════════════════════════════════════════
print()
print("  [2] 추세 & MA 위치")
print("  " + "-"*70)
prt_e("close > MA5",                early['dist_ma5']  > 0)
prt_e("close > MA10",               early['dist_ma10'] > 0)
prt_e("close > MA20 (중기상승추세)", early['dist_ma20'] > 0)
prt_e("close > MA60 (장기상승추세)", early['dist_ma60'] > 0)
prt_e("MA5 > MA20 (정배열)",        early['ma5'] > early['ma20'])
prt_e("MA20 > MA60 (중장기 정배열)", early['ma20'] > early['ma60'])
prt_e("시가 > MA20",                early['open_vs_ma20'] > 0)
prt_e("시가 > MA5",                 early['open'] > early['ma5'])
prt_e("dist_ma20 > +3% (MA 위 크게)", early['dist_ma20'] > 3)
prt_e("dist_ma20 +0~+3% (MA 위 소)",  (early['dist_ma20'] > 0) & (early['dist_ma20'] <= 3))

# ══════════════════════════════════════════════════════════════════════════════
print()
print("  [3] 모멘텀 & 낙폭")
print("  " + "-"*70)
prt_e("전일 수익률 > 0 (전일 양봉)",  early['ret_prev'] > 0)
prt_e("전일 수익률 > +1%",           early['ret_prev'] > 1.0)
prt_e("전일 수익률 > +2%",           early['ret_prev'] > 2.0)
prt_e("전5일 누적 > 0%",             early['ret_5d']   > 0)
prt_e("전5일 누적 > +3%",            early['ret_5d']   > 3.0)
prt_e("전5일 누적 > +5%",            early['ret_5d']   > 5.0)
prt_e("전10일 누적 > 0%",            early['ret_10d']  > 0)
prt_e("전일 음봉",                   early['prev_bearish'])
prt_e("전일 수익률 < -1% (전일 하락)", early['ret_prev'] < -1.0)

# ══════════════════════════════════════════════════════════════════════════════
print()
print("  [4] RSI & 볼린저밴드")
print("  " + "-"*70)
prt_e("RSI > 70 (과매수)",           early['rsi14'] > 70)
prt_e("RSI 50~70 (강세 적정)",       (early['rsi14'] >= 50) & (early['rsi14'] < 70))
prt_e("RSI > 50",                   early['rsi14'] > 50)
prt_e("RSI 40~60 (중립)",           (early['rsi14'] >= 40) & (early['rsi14'] < 60))
prt_e("RSI < 50",                   early['rsi14'] < 50)
prt_e("RSI < 40",                   early['rsi14'] < 40)
prt_e("BB 상단 이탈 (강세 돌파)",    early['above_bb'])
prt_e("BB 위치 > 80% (상단 근처)",   early['bb_pct'] > 80)
prt_e("BB 위치 50~80%",             (early['bb_pct'] >= 50) & (early['bb_pct'] < 80))
prt_e("BB 위치 < 50%",              early['bb_pct'] < 50)
prt_e("BB 위치 < 20%",              early['bb_pct'] < 20)

# ══════════════════════════════════════════════════════════════════════════════
print()
print("  [5] 거래량")
print("  " + "-"*70)
prt_e("당일 거래량 > MA20×1.5",     early['vol_ratio'] > 1.5)
prt_e("당일 거래량 > MA20×2.0",     early['vol_ratio'] > 2.0)
prt_e("당일 거래량 > MA20×3.0",     early['vol_ratio'] > 3.0)
prt_e("전일 거래량 > MA20×1.5",     early['prev_vol_r'] > 1.5)
prt_e("전일 거래량 > MA20×2.0",     early['prev_vol_r'] > 2.0)
prt_e("전일 거래량 < MA20×0.8",     early['prev_vol_r'] < 0.8)
prt_e("전일 거래량 < MA20×1.0",     early['prev_vol_r'] < 1.0)

# ══════════════════════════════════════════════════════════════════════════════
print()
print("  [6] VB 강도 (얼마나 크게 돌파했는가)")
print("  " + "-"*70)
prt_e("VB 강도 ≥ 0.30 (초과0.05)",   early['vb_str'] >= 0.30)
prt_e("VB 강도 ≥ 0.35",             early['vb_str'] >= 0.35)
prt_e("VB 강도 ≥ 0.40",             early['vb_str'] >= 0.40)
prt_e("VB 강도 0.25~0.30 (겨우 돌파)", early['vb_str'] < 0.30)
prt_e("VB 초과분 (excess) > 0.10",   early['vb_excess'] > 0.10)
# VB 강도 분위수
for lo, hi in [(0.25,0.27),(0.27,0.30),(0.30,0.35),(0.35,0.40),(0.40,0.50),(0.50,9)]:
    g = early[(early['vb_str']>=lo) & (early['vb_str']<hi)]
    if len(g) < 3: continue
    hi_s = f"{hi:.2f}" if hi < 9 else "∞   "
    print(f"    VB강도 {lo:.2f}~{hi_s}  "
          f"{len(g):>3}건  승률{g['win'].mean()*100:>5.1f}%  "
          f"평균{g['net_ret'].mean():>+6.3f}%")

# ══════════════════════════════════════════════════════════════════════════════
print()
print("  [7] 변동성 (ATR)")
print("  " + "-"*70)
prt_e("ATR < 2% (저변동성)",         early['atr_pct'] < 2.0)
prt_e("ATR < 3%",                   early['atr_pct'] < 3.0)
prt_e("ATR 2~4% (적정 변동성)",     (early['atr_pct'] >= 2) & (early['atr_pct'] < 4))
prt_e("ATR > 4%",                   early['atr_pct'] > 4.0)
prt_e("ATR > 5% (고변동성)",         early['atr_pct'] > 5.0)

# ══════════════════════════════════════════════════════════════════════════════
print()
print("  [8] 캘린더 팩터")
print("  " + "-"*70)
for wd, nm in [(0,'월'),(1,'화'),(2,'수'),(4,'금')]:
    prt_e(f"{nm}요일", early['weekday']==wd)
prt_e("월초 (1주차)",               early['week_of_month']==1)
prt_e("월말 (4~5주차)",             early['week_of_month']>=4)

# ══════════════════════════════════════════════════════════════════════════════
print()
print("="*72)
print("  [9] 2중 조합 스크리닝 — Early VB")
print("="*72)

candidates = {
    'gap-up':             early['gap_pct'] > 0,
    'gap-up>1%':          early['gap_pct'] > 1.0,
    'gap-up>2%':          early['gap_pct'] > 2.0,
    'close>MA20':         early['dist_ma20'] > 0,
    'close>MA60':         early['dist_ma60'] > 0,
    'MA5>MA20(정배열)':    early['ma5'] > early['ma20'],
    'RSI>50':             early['rsi14'] > 50,
    'RSI 50~70':          (early['rsi14']>=50)&(early['rsi14']<70),
    'BB>80%':             early['bb_pct'] > 80,
    'above_bb':           early['above_bb'],
    '전일양봉':            ~early['prev_bearish'],
    'ret_prev>1%':        early['ret_prev'] > 1.0,
    'ret_5d>0%':          early['ret_5d'] > 0,
    'ret_5d>3%':          early['ret_5d'] > 3.0,
    'vol_ratio>1.5':      early['vol_ratio'] > 1.5,
    'prev_vol<1.0':       early['prev_vol_r'] < 1.0,
    'VB강도≥0.35':        early['vb_str'] >= 0.35,
    'VB강도≥0.40':        early['vb_str'] >= 0.40,
    'atr<3%':             early['atr_pct'] < 3.0,
    '월요일':              early['weekday']==0,
}

combo_res = []
keys = list(candidates.keys())
for i in range(len(keys)):
    for j in range(i+1,len(keys)):
        ka,kb = keys[i],keys[j]
        sub = early[candidates[ka]&candidates[kb]]
        if len(sub)<8: continue
        sp = stats(sub)
        combo_res.append(dict(combo=f"{ka} & {kb}", **sp))

cdf = pd.DataFrame(combo_res).sort_values('avg',ascending=False)
print(f"\n  {'조합':<42} {'건수':>4}  {'승률':>6}  {'평균':>8}  {'총수익':>8}  {'Sharpe':>7}")
print("  "+"-"*75)
for _,r in cdf.head(25).iterrows():
    print(f"  {r['combo']:<42} {r['n']:>4}건  {r['win']:>5.1f}%  "
          f"{r['avg']:>+7.3f}%  {r['total']:>+7.1f}%  {r['sharpe']:>6.2f}")

# ══════════════════════════════════════════════════════════════════════════════
print()
print("="*72)
print("  [10] 연도별 일관성 — 상위 단일 팩터")
print("="*72)

top_factors_vb = {
    'gap-up':              early['gap_pct'] > 0,
    'gap-up > 1%':         early['gap_pct'] > 1.0,
    'close > MA20':        early['dist_ma20'] > 0,
    'MA5 > MA20 (정배열)': early['ma5'] > early['ma20'],
    'RSI > 50':            early['rsi14'] > 50,
    'above_bb':            early['above_bb'],
    'VB강도 ≥ 0.35':       early['vb_str'] >= 0.35,
    'vol_ratio > 1.5':     early['vol_ratio'] > 1.5,
    '전일 양봉':            ~early['prev_bearish'],
    'ret_5d > 0':          early['ret_5d'] > 0,
}

for fname, mask in top_factors_vb.items():
    sub = early[mask]
    if len(sub) < 5: continue
    base_avg = early['net_ret'].mean()
    delta = sub['net_ret'].mean() - base_avg
    print(f"\n  ▸ {fname}  (전체: {len(sub)}건  "
          f"승률{sub['win'].mean()*100:.1f}%  "
          f"평균{sub['net_ret'].mean():+.3f}%  Δ{delta:>+.3f}%)")
    for yr in sorted(early['year'].unique()):
        g = sub[sub['year']==yr]
        if len(g)<2: continue
        print(f"    {yr}: {len(g):>3}건  승률{g['win'].mean()*100:>5.1f}%  평균{g['net_ret'].mean():>+6.3f}%")

# ══════════════════════════════════════════════════════════════════════════════
print()
print("="*72)
print("  [11] 잠재적 개선 전략별 백테스트 요약")
print("="*72)

strategies = [
    ("Early VB 기준선 (≤09:14)",    (early['trig_t']>=540)),
    ("gap-up만",                    early['gap_pct']>0),
    ("gap-up & close>MA20",        (early['gap_pct']>0)&(early['dist_ma20']>0)),
    ("gap-up & MA5>MA20(정배열)",   (early['gap_pct']>0)&(early['ma5']>early['ma20'])),
    ("gap-up & RSI>50",            (early['gap_pct']>0)&(early['rsi14']>50)),
    ("gap-up & above_bb",          (early['gap_pct']>0)&(early['above_bb'])),
    ("gap-up & ret_5d>0",          (early['gap_pct']>0)&(early['ret_5d']>0)),
    ("gap-up & VB강도≥0.35",       (early['gap_pct']>0)&(early['vb_str']>=0.35)),
    ("gap-up & vol>1.5x",          (early['gap_pct']>0)&(early['vol_ratio']>1.5)),
    ("gap-up>1% & RSI>50",         (early['gap_pct']>1.0)&(early['rsi14']>50)),
    ("gap-up>1% & MA5>MA20",       (early['gap_pct']>1.0)&(early['ma5']>early['ma20'])),
    ("gap-up>2% & RSI>50",         (early['gap_pct']>2.0)&(early['rsi14']>50)),
    ("close>MA20 & RSI>50",        (early['dist_ma20']>0)&(early['rsi14']>50)),
    ("MA5>MA20 & RSI 50~70",       (early['ma5']>early['ma20'])&(early['rsi14']>=50)&(early['rsi14']<70)),
    ("MA5>MA20 & BB>80%",          (early['ma5']>early['ma20'])&(early['bb_pct']>80)),
    ("gap-up & MA5>MA20 & RSI>50",(early['gap_pct']>0)&(early['ma5']>early['ma20'])&(early['rsi14']>50)),
]

base = stats(early)
print(f"\n  {'전략':<42} {'건수':>4}  {'승률':>6}  {'평균':>8}  {'총수익':>8}  {'Sharpe':>7}  Δavg")
print("  "+"-"*80)
for label, mask in strategies:
    g = early[mask]
    sp = stats(g)
    if sp is None: continue
    diff = sp['avg'] - base['avg']
    mark = '★' if diff>0.3 and sp['n']>=10 else '  '
    print(f"  {mark}{label:<42} {sp['n']:>4}건  {sp['win']:>5.1f}%  "
          f"{sp['avg']:>+7.3f}%  {sp['total']:>+7.1f}%  {sp['sharpe']:>6.2f}  "
          f"Δ{diff:>+.3f}%")

# ══════════════════════════════════════════════════════════════════════════════
print()
print("="*72)
print("  [12] 종합 권고 요약 (VB 신호 개선)")
print("="*72)

# 자동 선별
all_strategies = []
for label, mask in strategies[1:]:
    g = early[mask]
    sp = stats(g)
    if sp and sp['n'] >= 10:
        all_strategies.append((label, sp['avg'], sp['win'], sp['n'], sp['sharpe'],
                                sp['avg']-base['avg']))

all_strategies.sort(key=lambda x: x[1], reverse=True)

print(f"""
  Early VB 기준선: {base['n']}건  승률{base['win']:.1f}%  평균{base['avg']:+.3f}%

  ★ 평균수익 기준 Top 개선 전략:""")
for i,(lbl,avg,win,n,shr,d) in enumerate(all_strategies[:8],1):
    print(f"    {i}. {lbl:<40}  {n:>3}건  승률{win:>5.1f}%  평균{avg:>+.3f}%  Δ{d:>+.3f}%  S={shr:.2f}")

print(f"""
  VB 개선 핵심 방향:
  ① 방향성 필터  : gap-up 발생일만 진입 (gap-down VB는 역추세라 불리)
  ② 추세 필터    : close > MA20 or MA5 > MA20 (정배열) → 상승추세 확인
  ③ 모멘텀 필터  : RSI > 50, 5일 누적 수익 > 0 (모멘텀 살아있는 날)
  ④ 돌파 강도    : VB 강도 ≥ 0.35 (0.25 겨우 터치 X, 강한 돌파만)
  ⑤ BB 위치      : BB 상위권(>80%) or 상단 이탈 = 강력 모멘텀 신호
  ⑥ 3중 조합     : gap-up & MA5>MA20 & RSI>50 → 핵심 황금 조합
""")
