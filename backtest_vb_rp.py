"""
233740 (KODEX 코스닥150 레버리지) — VB + RP 전략 백테스트 및 개선 분석
"""
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

FEE = 0.0013  # 수수료 0.03% + 슬리피지 0.1% (편도)

# ── 데이터 로드 ────────────────────────────────────────────────────────────────
df_min = pd.read_csv('minute_chart_233740.csv', encoding='utf-8-sig')
df_min = df_min.sort_values(['일자', '시간']).reset_index(drop=True)
df_min['일자'] = df_min['일자'].astype(int)
df_min['dt']  = pd.to_datetime(
    df_min['일자'].astype(str) + df_min['시간'].astype(str).str.zfill(4),
    format='%Y%m%d%H%M'
)

# ── 일봉 생성 ─────────────────────────────────────────────────────────────────
daily = (df_min.groupby('일자')
         .agg(open=('시가','first'), high=('고가','max'),
              low=('저가','min'),  close=('종가','last'), vol=('거래량','sum'))
         .reset_index()
         .sort_values('일자').reset_index(drop=True))
daily['date']       = pd.to_datetime(daily['일자'].astype(str), format='%Y%m%d')
daily['weekday']    = daily['date'].dt.weekday         # 0=월 … 4=금
daily['prev_range'] = (daily['high'] - daily['low']).shift(1)
daily['prev_close'] = daily['close'].shift(1)
daily['gap_down']   = daily['open'] < daily['prev_close']   # 당일 시가 < 전일 종가
daily['gap_pct']    = (daily['open'] - daily['prev_close']) / daily['prev_close'] * 100
daily['next_open']  = daily['open'].shift(-1)

# 이동평균 · ATR
for n in [5, 10, 20, 60]:
    daily[f'ma{n}'] = daily['close'].rolling(n).mean()
tr = pd.concat([daily['high'] - daily['low'],
                (daily['high'] - daily['close'].shift()).abs(),
                (daily['low']  - daily['close'].shift()).abs()], axis=1).max(axis=1)
daily['atr14']   = tr.ewm(alpha=1/14, adjust=False).mean()
daily['atr_pct'] = daily['atr14'] / daily['close'] * 100
ddict = daily.set_index('일자').to_dict('index')   # 빠른 row 조회

# ── 분봉 → 날짜별 딕셔너리 ────────────────────────────────────────────────────
min_by_date = {}
for dk, g in df_min.groupby('일자'):
    min_by_date[dk] = g.sort_values('dt').reset_index(drop=True)

# ── VB 사전계산 (정규화 고가 배열) ────────────────────────────────────────────
#   trigger = open + alpha * prev_range
#   bar 돌파 조건: bar_high >= trigger  →  (bar_high - open) / prev_range >= alpha
day_vb = {}
for dk, bars in min_by_date.items():
    dr = ddict.get(dk)
    if dr is None:
        continue
    pr = dr['prev_range']
    if np.isnan(pr) or pr <= 0:
        continue
    d_open = float(dr['open'])
    highs  = bars['고가'].values.astype(float)
    b_open = bars['시가'].values.astype(float)
    times  = (bars['dt'].dt.hour * 60 + bars['dt'].dt.minute).values.astype(int)
    day_vb[dk] = dict(
        norm   = (highs - d_open) / pr,
        b_open = b_open,
        times  = times,
        d_open = d_open,
        pr     = pr,
    )

def find_vb(dk, alpha, cutoff=9*60+14):
    """numpy 배열로 VB 진입 탐색. O(n) 벡터 연산."""
    data = day_vb.get(dk)
    if data is None:
        return None
    mask = data['norm'] >= alpha
    if not mask.any():
        return None
    idx      = mask.argmax()
    trig_px  = data['d_open'] + alpha * data['pr']
    entry_px = max(data['b_open'][idx], trig_px)
    t        = data['times'][idx]
    return {'price': entry_px, 'weight': 1.0 if t <= cutoff else 0.1}

# ── RP 계산 ───────────────────────────────────────────────────────────────────
def add_rp(df, window):
    h = df['high'].rolling(window).max()
    l = df['low'].rolling(window).min()
    return (df['close'] - l) / (h - l).replace(0, np.nan)

# ── 백테스트 엔진 ─────────────────────────────────────────────────────────────
def backtest(alpha=0.25, rp_thresh=0.2, rp_win=5,
             skip_thu=True, skip_wed=False, skip_fri=False,
             monday_aux=True, trend_ma=None, atr_max=None,
             vb_cutoff=9*60+14, rp_gap_down=False):
    d = daily.copy()
    d['rp'] = add_rp(d, rp_win)
    skip = {w for w, f in [(3,skip_thu),(2,skip_wed),(4,skip_fri)] if f}
    trades = []

    for i in range(max(rp_win, 60), len(d) - 1):
        row = d.iloc[i]
        if np.isnan(row['prev_range']) or np.isnan(row['next_open']):
            continue

        wd       = int(row['weekday'])
        dk       = int(row['일자'])
        exit_px  = float(row['next_open'])
        close_px = float(row['close'])

        # ATR 필터
        if atr_max and not np.isnan(row['atr_pct']) and row['atr_pct'] > atr_max:
            continue

        # 추세 필터 (월보조는 필터 미적용)
        if trend_ma:
            mv = row.get(f'ma{trend_ma}', np.nan)
            if not np.isnan(mv) and close_px < mv:
                if not (monday_aux and wd == 0):
                    continue

        signal, entry_px, weight = None, None, 1.0

        if wd not in skip:
            trigger = float(row['open']) + alpha * float(row['prev_range'])
            vb = find_vb(dk, alpha, vb_cutoff)
            if vb:
                signal, entry_px, weight = 'vb', vb['price'], vb['weight']

            if signal is None:
                rp_val = row['rp']
                gap_ok = (not rp_gap_down) or bool(row['gap_down'])
                if not np.isnan(rp_val) and rp_val < rp_thresh and gap_ok:
                    signal, entry_px = 'rp', close_px

        if signal is None and monday_aux and wd == 0:
            signal, entry_px = 'mon', close_px

        if signal and entry_px and entry_px > 0 and exit_px > 0:
            raw = exit_px / entry_px - 1
            net = (raw - FEE * 2) * weight
            trades.append(dict(date=row['date'], signal=signal,
                               weight=weight, entry=entry_px, exit=exit_px,
                               raw=raw*100, net=net*100))

    return pd.DataFrame(trades)

def metrics(df):
    if df.empty:
        return dict(total=0, n=0, win=0, avg=0, mdd=0, sharpe=0)
    eq   = (1 + df['net']/100).cumprod()
    peak = eq.cummax()
    mdd  = ((eq - peak)/peak).min() * 100
    win  = (df['net'] > 0).mean() * 100
    avg  = df['net'].mean()
    std  = df['net'].std()
    shr  = avg/std * np.sqrt(252) if std > 0 else 0
    return dict(total=(eq.iloc[-1]-1)*100, n=len(df), win=win, avg=avg, mdd=mdd, sharpe=shr)

def row_str(name, m, mark=''):
    return (f"  {mark}{name:<28} {m['total']:>+7.1f}%  {m['n']:>4}건  "
            f"승률{m['win']:>5.1f}%  평균{m['avg']:>+6.3f}%  "
            f"MDD{m['mdd']:>+6.1f}%  S={m['sharpe']:>5.2f}")

# ══════════════════════════════════════════════════════════════════════════════
# 1. 기준 전략
# ══════════════════════════════════════════════════════════════════════════════
print("=" * 72)
print("  [기준 전략]  VB α=0.25 + RP<0.2/5일 + 월보조 + 목스킵")
print("=" * 72)

base = backtest()
bm   = metrics(base)
print(row_str("전체", bm))

# 바이앤홀드 (같은 기간)
bah_start = daily[daily['일자'] >= int(base['date'].min().strftime('%Y%m%d'))]['close'].iloc[0]
bah_end   = daily['close'].iloc[-1]
bah_ret   = (bah_end / bah_start - 1) * 100
print(f"  {'바이앤홀드 (기준선)':<29} {bah_ret:>+7.1f}%")

print()
print("  ── 신호별 분해 ──")
for sig in ['vb', 'rp', 'mon']:
    sub = base[base['signal'] == sig]
    if sub.empty:
        continue
    m = metrics(sub)
    print(row_str(f"  {sig.upper()}", m))

# VB 시간컷 분해
vb_df = base[base['signal'] == 'vb']
if not vb_df.empty:
    print()
    print("  ── VB 시간 컷오프 분해 ──")
    full_vb = vb_df[vb_df['weight'] == 1.0]
    part_vb = vb_df[vb_df['weight'] == 0.1]
    for label, sub in [('09:14 이전 (100%)', full_vb), ('09:15 이후 (10%)', part_vb)]:
        if sub.empty:
            continue
        m2 = metrics(sub)
        # 자본 기준 raw 평균
        raw_avg = sub['raw'].mean()
        print(f"    {label:<22} {m2['n']:>4}건  승률{m2['win']:>5.1f}%  "
              f"raw평균{raw_avg:>+6.3f}%  net평균{m2['avg']:>+6.3f}%")

# 연도별
print()
print("  ── 연도별 성과 ──")
base['year'] = base['date'].dt.year
for yr, g in base.groupby('year'):
    m = metrics(g)
    print(f"    {yr}: 총수익{m['total']:>+7.1f}%  {m['n']:>3}건  승률{m['win']:>5.1f}%  Sharpe{m['sharpe']:>5.2f}")

# ══════════════════════════════════════════════════════════════════════════════
# 2. 파라미터 민감도 분석
# ══════════════════════════════════════════════════════════════════════════════
print()
print("=" * 72)
print("  [파라미터 민감도]")
print("=" * 72)

def sweep(label, param_list, bt_kwargs_fn):
    print(f"\n  ▸ {label}")
    results = []
    for p in param_list:
        kw = bt_kwargs_fn(p)
        t  = backtest(**kw)
        m  = metrics(t)
        results.append((p, m))
    best = max(results, key=lambda x: x[1]['sharpe'])
    for p, m in results:
        star = '★' if p == best[0] else ' '
        print(f"    {star} {str(p):<10} 총{m['total']:>+7.1f}%  {m['n']:>4}건  "
              f"승률{m['win']:>5.1f}%  평균{m['avg']:>+6.3f}%  "
              f"MDD{m['mdd']:>+6.1f}%  S={m['sharpe']:>5.2f}")
    return best

sweep("Alpha (VB 진입 민감도)",
      [0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50],
      lambda a: dict(alpha=a))

sweep("RP 임계값",
      [0.10, 0.15, 0.20, 0.25, 0.30, 0.35],
      lambda v: dict(rp_thresh=v))

sweep("RP 룩백 윈도우 (일)",
      [3, 5, 7, 10, 15, 20],
      lambda w: dict(rp_win=w))

sweep("VB 시간 컷오프 (분)",
      [9*60+5, 9*60+10, 9*60+14, 9*60+20, 9*60+30, 10*60],
      lambda c: dict(vb_cutoff=c))

# ══════════════════════════════════════════════════════════════════════════════
# 3. 필터 효과
# ══════════════════════════════════════════════════════════════════════════════
print()
print("=" * 72)
print("  [필터 효과 분석]")
print("=" * 72)

# 추세 필터
print("\n  ▸ 추세 필터 (MA 기준 — close > MA 일 때만 진입)")
for tf in [None, 20, 60]:
    t = backtest(trend_ma=tf)
    m = metrics(t)
    lbl = f"MA{tf} 필터" if tf else "필터없음(기준)"
    print(row_str(lbl, m, '★' if tf == 20 else '  '))

# 요일 필터 조합
print("\n  ▸ 요일 스킵 조합")
day_combos = [
    ("목만 스킵 (기준)",     dict(skip_thu=True,  skip_wed=False, skip_fri=False)),
    ("목+수 스킵",           dict(skip_thu=True,  skip_wed=True,  skip_fri=False)),
    ("목+금 스킵",           dict(skip_thu=True,  skip_wed=False, skip_fri=True)),
    ("목+수+금 스킵",        dict(skip_thu=True,  skip_wed=True,  skip_fri=True)),
    ("스킵 없음",            dict(skip_thu=False, skip_wed=False, skip_fri=False)),
    ("월+화만 진입",         dict(skip_thu=True,  skip_wed=True,  skip_fri=True, monday_aux=False)),
]
for lbl, kw in day_combos:
    t = backtest(**kw)
    m = metrics(t)
    print(row_str(lbl, m))

# ATR 필터
print("\n  ▸ ATR% 상한 필터 (고변동성 날 제외)")
for av in [None, 2.0, 3.0, 4.0, 5.0]:
    t = backtest(atr_max=av)
    m = metrics(t)
    lbl = f"ATR<{av}%" if av else "ATR필터없음(기준)"
    print(row_str(lbl, m))

# 월요일 보조매수
print("\n  ▸ 월요일 보조매수 온/오프")
for mon in [True, False]:
    t = backtest(monday_aux=mon)
    m = metrics(t)
    print(row_str(f"월보조={mon}", m))

# ══════════════════════════════════════════════════════════════════════════════
# 4. 최적 조합 그리드 서치
# ══════════════════════════════════════════════════════════════════════════════
print()
print("=" * 72)
print("  [그리드 서치 — 최적 파라미터 조합]")
print("=" * 72)
print("  (Sharpe 기준 정렬, 상위 15개)")
print()

combos = []
for alpha in [0.20, 0.25, 0.30]:
    for rp_t in [0.15, 0.20, 0.25]:
        for rp_w in [5, 7, 10]:
            for tf in [None, 20]:
                for wed in [False, True]:
                    for cutoff in [9*60+14, 9*60+20]:
                        t = backtest(alpha=alpha, rp_thresh=rp_t, rp_win=rp_w,
                                     trend_ma=tf, skip_wed=wed, vb_cutoff=cutoff)
                        m = metrics(t)
                        combos.append(dict(alpha=alpha, rp_t=rp_t, rp_w=rp_w,
                                           tf=tf, wed=wed, cutoff=cutoff, **m))

combo_df = pd.DataFrame(combos).sort_values('sharpe', ascending=False)
print(f"  {'α':>5} {'RP임계':>6} {'RPwin':>5} {'MA':>5} {'수스킵':>5} {'컷오프':>6} "
      f"{'총수익':>8} {'Sharpe':>6} {'MDD':>7} {'거래':>5}")
print("  " + "-" * 70)
for _, r in combo_df.head(15).iterrows():
    cutoff_str = f"{int(r['cutoff'])//60:02d}:{int(r['cutoff'])%60:02d}"
    print(f"  {r['alpha']:>5.2f} {r['rp_t']:>6.2f} {r['rp_w']:>5.0f} "
          f"{str(r['tf']):>5} {str(r['wed']):>5} {cutoff_str:>6} "
          f"  {r['total']:>+6.1f}% {r['sharpe']:>6.2f} {r['mdd']:>+6.1f}% {r['n']:>5}")

# ── 최우수 조합 상세 분석 ──────────────────────────────────────────────────────
best = combo_df.iloc[0]
print()
print("=" * 72)
print("  [최우수 조합 상세 분석]")
print("=" * 72)
tf_val = None if str(best['tf']) == 'None' else int(best['tf'])
t_best = backtest(alpha=float(best['alpha']), rp_thresh=float(best['rp_t']),
                  rp_win=int(best['rp_w']), trend_ma=tf_val,
                  skip_wed=bool(best['wed']), vb_cutoff=int(best['cutoff']))

print(f"  파라미터: α={best['alpha']:.2f}  RP<{best['rp_t']:.2f}  "
      f"RPwin={int(best['rp_w'])}일  MA={best['tf']}필터  "
      f"수스킵={best['wed']}  컷오프={int(best['cutoff'])//60:02d}:{int(best['cutoff'])%60:02d}")
print()

mb = metrics(t_best)
print(row_str("최우수 조합", mb, '★ '))
print(f"  {'기준 전략':<29} {bm['total']:>+7.1f}%  {bm['n']:>4}건  "
      f"승률{bm['win']:>5.1f}%  평균{bm['avg']:>+6.3f}%  "
      f"MDD{bm['mdd']:>+6.1f}%  S={bm['sharpe']:>5.2f}")
print(f"  {'바이앤홀드':<29} {bah_ret:>+7.1f}%")
print()

# 신호별 분해
print("  ── 신호별 분해 ──")
for sig in ['vb', 'rp', 'mon']:
    sub = t_best[t_best['signal'] == sig]
    if sub.empty:
        continue
    m2 = metrics(sub)
    print(row_str(f"  {sig.upper()}", m2))

# 연도별
print()
print("  ── 연도별 성과 ──")
t_best['year'] = t_best['date'].dt.year
for yr, g in t_best.groupby('year'):
    m2 = metrics(g)
    print(f"    {yr}: 총수익{m2['total']:>+7.1f}%  {m2['n']:>3}건  승률{m2['win']:>5.1f}%  Sharpe{m2['sharpe']:>5.2f}")

# ── 훈련/검증 분리 ────────────────────────────────────────────────────────────
print()
print("=" * 72)
print("  [과적합 방지 검증] — 전반기(훈련) vs 후반기(검증)")
print("=" * 72)

mid_date = daily['date'].median()
mid_str  = mid_date.strftime('%Y%m%d')
print(f"  분할 기준일: {mid_str}  (전반기 250일 / 후반기 250일)")
print()

for period, is_first in [("전반기(훈련)", True), ("후반기(검증)", False)]:
    print(f"  [{period}]")
    if is_first:
        sub_d    = t_best[t_best['date'] < mid_date]
        base_sub = base[base['date'] < mid_date]
    else:
        sub_d    = t_best[t_best['date'] >= mid_date]
        base_sub = base[base['date'] >= mid_date]
    if sub_d.empty or base_sub.empty:
        print("    (데이터 없음)")
        continue
    m_best_p = metrics(sub_d)
    m_base_p = metrics(base_sub)
    print(f"    최우수 조합: 총{m_best_p['total']:>+7.1f}%  {m_best_p['n']:>3}건  "
          f"승률{m_best_p['win']:>5.1f}%  Sharpe{m_best_p['sharpe']:>5.2f}")
    print(f"    기준 전략  : 총{m_base_p['total']:>+7.1f}%  {m_base_p['n']:>3}건  "
          f"승률{m_base_p['win']:>5.1f}%  Sharpe{m_base_p['sharpe']:>5.2f}")
    print()

# ══════════════════════════════════════════════════════════════════════════════
# 5. 종합 요약
# ══════════════════════════════════════════════════════════════════════════════
print("=" * 72)
print("  [종합 개선 권고]")
print("=" * 72)

# 각 파라미터 그룹 최우수
alpha_best   = combo_df.sort_values('sharpe',ascending=False)['alpha'].iloc[0]
rp_t_best    = combo_df.sort_values('sharpe',ascending=False)['rp_t'].iloc[0]
rp_w_best    = combo_df.sort_values('sharpe',ascending=False)['rp_w'].iloc[0]
tf_best      = combo_df.sort_values('sharpe',ascending=False)['tf'].iloc[0]
wed_best     = combo_df.sort_values('sharpe',ascending=False)['wed'].iloc[0]
cutoff_best  = int(combo_df.sort_values('sharpe',ascending=False)['cutoff'].iloc[0])

print(f"""
  ① VB Alpha  :  0.25(현행) → {alpha_best:.2f}  권고
     - 값이 작을수록 진입 빠름/거래 多, 클수록 신호 강도 ↑

  ② RP 임계값 :  0.20(현행) → {rp_t_best:.2f}  권고
     - 낮출수록 극단적 과매도만 선별, 거래 수 감소 & 평균 수익 개선

  ③ RP 윈도우 :  5일(현행) → {int(rp_w_best)}일  권고
     - 기간 늘릴수록 더 넓은 시야의 단기 과매도 탐지

  ④ 추세 필터 :  없음(현행) → MA{tf_best} 필터  권고
     - close > MA{tf_best} 조건 추가 시 하락 추세 진입 억제

  ⑤ 수요일 스킵: False(현행) → {wed_best}  권고
     - 수→목 오버나이트 회피 (목요일 시초가 변동성 ↑)

  ⑥ VB 시간컷  :  09:14:30(현행) → {cutoff_best//60:02d}:{cutoff_best%60:02d}  권고
     - 컷오프 이후 10% 비중 축소 시점 조정

  ※ 단, 모든 개선안은 동일 데이터 내 최적화이므로 과적합 위험 존재.
    후반기(검증) 성과가 전반기(훈련)와 유사한지 반드시 확인할 것.
""")

# ══════════════════════════════════════════════════════════════════════════════
# 추가 분석: RP + Gap-Down 조건
# ══════════════════════════════════════════════════════════════════════════════
print()
print("=" * 72)
print("  [RP Gap-Down 조건 분석]")
print("  조건: 5일 RP < 0.2  +  당일 시가 < 전일 종가 (gap-down)")
print("=" * 72)

# ── 1. 전체 전략 수준 비교 ──────────────────────────────────────────────────
t_rp_base   = backtest(rp_gap_down=False)
t_rp_gapdn  = backtest(rp_gap_down=True)
m_base_all  = metrics(t_rp_base)
m_gapdn_all = metrics(t_rp_gapdn)

print()
print("  ▸ 전체 전략 비교 (VB+RP+월보조 포함)")
print(row_str("기준 (gap-down 조건 없음)", m_base_all))
print(row_str("RP gap-down 조건 추가",    m_gapdn_all))

# ── 2. RP 신호만 분리 비교 ──────────────────────────────────────────────────
print()
print("  ▸ RP 신호만 분리 비교")

rp_only_base  = t_rp_base[t_rp_base['signal'] == 'rp'].copy()
rp_only_gapdn = t_rp_gapdn[t_rp_gapdn['signal'] == 'rp'].copy()

m_rp_base  = metrics(rp_only_base)
m_rp_gapdn = metrics(rp_only_gapdn)

print(row_str("  RP (gap-down 없음)", m_rp_base))
print(row_str("  RP (gap-down 있음)", m_rp_gapdn))

# ── 3. RP 발생일 전수 분석 (gap-down 유무별) ──────────────────────────────
print()
print("  ▸ RP 조건(RP<0.2) 충족일의 gap-down 유무별 수익 분포")

d_analysis = daily.copy()
d_analysis['rp'] = add_rp(d_analysis, 5)

# RP 조건 충족일 전체 추출 (목요일 스킵, warm-up 제외)
rp_days = d_analysis[
    (d_analysis['rp'] < 0.2) &
    (~d_analysis['rp'].isna()) &
    (~d_analysis['prev_close'].isna()) &
    (~d_analysis['next_open'].isna()) &
    (d_analysis['weekday'] != 3) &
    (d_analysis.index >= 60)
].copy()

rp_days['raw_ret'] = (rp_days['next_open'] / rp_days['close'] - 1) * 100
rp_days['net_ret'] = rp_days['raw_ret'] - FEE * 2 * 100

gap_groups = rp_days.groupby('gap_down')
print(f"\n  {'구분':<22} {'건수':>5} {'승률':>7} {'평균수익':>9} {'중앙값':>8} {'std':>7}")
print("  " + "-" * 58)
for gd, g in sorted(gap_groups, key=lambda x: x[0]):
    label = "gap-down (시가<전종)" if gd else "gap-up/flat (시가≥전종)"
    win   = (g['net_ret'] > 0).mean() * 100
    avg   = g['net_ret'].mean()
    med   = g['net_ret'].median()
    std   = g['net_ret'].std()
    print(f"  {label:<22} {len(g):>5}건  {win:>6.1f}%  {avg:>+8.3f}%  {med:>+7.3f}%  {std:>6.3f}%")

# ── 4. gap 크기별 수익 분포 ─────────────────────────────────────────────────
print()
print("  ▸ RP 조건 충족일 — gap 크기별 평균 수익 (순수 RP 발생일)")
gap_bins = [(-99, -2, "갭하락 -2%↓"),
            (-2,  -1, "갭하락 -1~-2%"),
            (-1, -0.3,"갭하락 -0.3~-1%"),
            (-0.3, 0, "갭하락 0~-0.3%"),
            (0,   99, "갭상승/보합 0%↑")]
print(f"\n  {'구분':<22} {'건수':>5} {'승률':>7} {'평균수익':>9}")
print("  " + "-" * 47)
for lo, hi, label in gap_bins:
    sub = rp_days[(rp_days['gap_pct'] >= lo) & (rp_days['gap_pct'] < hi)]
    if sub.empty:
        continue
    win = (sub['net_ret'] > 0).mean() * 100
    avg = sub['net_ret'].mean()
    print(f"  {label:<22} {len(sub):>5}건  {win:>6.1f}%  {avg:>+8.3f}%")

# ── 5. 연도별 gap-down 효과 ─────────────────────────────────────────────────
print()
print("  ▸ 연도별 RP 신호 성과 (gap-down 유무)")
rp_days['year'] = rp_days['date'].dt.year
for yr, g in rp_days.groupby('year'):
    gd  = g[g['gap_down']]
    nongd = g[~g['gap_down']]
    gd_avg  = gd['net_ret'].mean()  if len(gd)  > 0 else float('nan')
    ng_avg  = nongd['net_ret'].mean() if len(nongd) > 0 else float('nan')
    print(f"    {yr}:  gap-down {len(gd):>3}건 평균{gd_avg:>+6.3f}%  |  "
          f"gap-up/flat {len(nongd):>3}건 평균{ng_avg:>+6.3f}%")

# ── 6. 임계값별 gap-down 효과 비교 ─────────────────────────────────────────
print()
print("  ▸ RP 임계값별 gap-down 조건 효과")
print(f"  {'RP임계':>6} {'조건':>20} {'건수':>5} {'승률':>7} {'평균':>8} {'총수익':>8} {'Sharpe':>7}")
print("  " + "-" * 60)
for rp_t in [0.10, 0.15, 0.20, 0.25, 0.30]:
    for gd in [False, True]:
        t = backtest(rp_thresh=rp_t, rp_gap_down=gd)
        rp_sub = t[t['signal'] == 'rp']
        if rp_sub.empty:
            continue
        m = metrics(rp_sub)
        gd_label = "RP+gap-down" if gd else "RP 기준"
        print(f"  {rp_t:>6.2f} {gd_label:>20} {m['n']:>5}건  {m['win']:>6.1f}%  "
              f"{m['avg']:>+7.3f}%  {m['total']:>+7.1f}%  {m['sharpe']:>6.2f}")
    print()

# ── 7. 최종 권고 ────────────────────────────────────────────────────────────
print("=" * 72)
print("  [gap-down 조건 종합 결론]")
print("=" * 72)

# 수치 기반 결론
gd_sub  = rp_days[rp_days['gap_down']]
ngd_sub = rp_days[~rp_days['gap_down']]
gd_win  = (gd_sub['net_ret'] > 0).mean() * 100
ngd_win = (ngd_sub['net_ret'] > 0).mean() * 100
gd_avg  = gd_sub['net_ret'].mean()
ngd_avg = ngd_sub['net_ret'].mean()
effect  = gd_avg - ngd_avg

print(f"""
  gap-down 있을 때  : {len(gd_sub):>3}건  승률{gd_win:.1f}%  평균{gd_avg:>+.3f}%
  gap-down 없을 때  : {len(ngd_sub):>3}건  승률{ngd_win:.1f}%  평균{ngd_avg:>+.3f}%
  gap-down 효과     : {effect:>+.3f}%p (양수=gap-down이 유리)

  전략 수준 영향:
    기준 전략         : 총수익{m_base_all['total']:>+7.1f}%  Sharpe{m_base_all['sharpe']:>5.2f}
    RP gap-down 추가  : 총수익{m_gapdn_all['total']:>+7.1f}%  Sharpe{m_gapdn_all['sharpe']:>5.2f}
    차이              : {m_gapdn_all['total']-m_base_all['total']:>+.1f}%p
""")
if effect > 0.1:
    print("  → gap-down 조건이 RP 신호 품질을 유의미하게 개선합니다. 추가 권고.")
elif effect > 0:
    print("  → gap-down 조건이 약간 유리하나 거래 수 감소를 감안해 선택적 적용.")
else:
    print("  → gap-down 조건이 RP 신호를 개선하지 못합니다. 추가 불필요.")
