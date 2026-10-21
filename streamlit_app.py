"""Mobile-friendly Korean swing-trading research app; no trading execution."""
from datetime import date, timedelta, datetime
from zoneinfo import ZoneInfo
import math
import zipfile
from pathlib import Path

import FinanceDataReader as fdr
import pandas as pd
import streamlit as st

st.set_page_config(page_title="스윙 단일종목 분석", page_icon="📈", layout="centered")
st.title("📈 단 하나의 스윙 후보")
st.caption("국내 주식 · 며칠~몇 주 보유 · 휴대폰 최적화 · 자동매매 아님")
st.warning("수익을 보장하지 않습니다. 가격 신호는 참고용이며, 주문 전 최신 호가·공시·거래정지 여부를 확인하세요.")

CAPITAL_DEFAULT = 200_000
RISK_FRACTION_DEFAULT = 0.01  # 1% of capital
MAX_DATA_AGE_DAYS = 5

@st.cache_data(ttl=600, show_spinner=False)
def get_listing(market: str) -> pd.DataFrame:
    df = fdr.StockListing(market)
    if df is None or df.empty or not {'Symbol', 'Name'}.issubset(df.columns):
        raise RuntimeError("상장 종목 목록을 가져오지 못했습니다.")
    df = df.copy()
    df['Symbol'] = df['Symbol'].astype(str).str.strip()
    df = df[df['Symbol'].str.fullmatch(r'\d{6}', na=False)].drop_duplicates('Symbol')
    if 'Marcap' in df:
        df['Marcap'] = pd.to_numeric(df['Marcap'], errors='coerce')
        df = df.sort_values('Marcap', ascending=False, na_position='last')
    return df

@st.cache_data(ttl=600, show_spinner=False)
def get_history(code: str, start: str, end: str) -> pd.DataFrame:
    df = fdr.DataReader(code, start, end)
    if df is None or df.empty:
        return pd.DataFrame()
    df = df.copy().sort_index()
    for c in ['Open', 'High', 'Low', 'Close', 'Volume']:
        if c in df:
            df[c] = pd.to_numeric(df[c], errors='coerce')
    required = {'Open', 'High', 'Low', 'Close', 'Volume'}
    if not required.issubset(df.columns):
        return pd.DataFrame()
    return df.dropna(subset=list(required))

def calculate(code: str, name: str, hist: pd.DataFrame):
    if len(hist) < 60:
        return None
    d = hist.copy().sort_index()
    d['SMA20'] = d.Close.rolling(20).mean()
    d['SMA50'] = d.Close.rolling(50).mean()
    prev_close = d.Close.shift(1)
    tr = pd.concat([(d.High-d.Low), (d.High-prev_close).abs(), (d.Low-prev_close).abs()], axis=1).max(axis=1)
    d['ATR14'] = tr.rolling(14).mean()
    d['Turnover'] = d.Close * d.Volume
    d['TurnoverAvg20'] = d.Turnover.shift(1).rolling(20).mean()
    d['VolAvg20'] = d.Volume.shift(1).rolling(20).mean()
    last = d.iloc[-1]
    last_date = pd.Timestamp(d.index[-1]).date()
    today_kst = datetime.now(ZoneInfo('Asia/Seoul')).date()
    age_days = (today_kst - last_date).days
    close = float(last.Close)
    atr = float(last.ATR14) if pd.notna(last.ATR14) else float('nan')
    sma20 = float(last.SMA20) if pd.notna(last.SMA20) else float('nan')
    sma50 = float(last.SMA50) if pd.notna(last.SMA50) else float('nan')
    if not all(math.isfinite(x) and x > 0 for x in [close, atr, sma20, sma50]):
        return None
    avg_turn = float(last.TurnoverAvg20) if pd.notna(last.TurnoverAvg20) else 0
    avg_vol = float(last.VolAvg20) if pd.notna(last.VolAvg20) else 0
    turnover = float(last.Turnover)
    turnover_ratio = turnover / avg_turn if avg_turn > 0 else 0
    volume_ratio = float(last.Volume) / avg_vol if avg_vol > 0 else 0
    daily_pct = float((close / float(d.Close.iloc[-2]) - 1) * 100)
    prev20high = float(d.High.iloc[-21:-1].max())
    trend = close > sma20 > sma50
    breakout = close > prev20high
    liquidity = turnover >= 1_000_000_000
    momentum = turnover_ratio >= 1.5 and volume_ratio >= 1.2 and daily_pct < 8
    fresh = 0 <= age_days <= MAX_DATA_AGE_DAYS
    signal = bool(fresh and trend and breakout and liquidity and momentum)
    # Entry is the prior 20-session high plus one tick-like buffer (0.1%); no live order book available.
    trigger = max(prev20high * 1.001, close) if signal else prev20high * 1.001
    stop = max(1, trigger - 2 * atr)
    target = trigger + 3 * atr
    return {
        'code': code, 'name': name, 'date': last_date.isoformat(), 'age_days': age_days,
        'fresh': fresh, 'close': close, 'sma20': sma20, 'sma50': sma50, 'atr': atr,
        'prev20high': prev20high, 'turnover_ratio': turnover_ratio, 'volume_ratio': volume_ratio,
        'daily_pct': daily_pct, 'signal': signal, 'trigger': trigger, 'stop': stop, 'target': target,
        'turnover': turnover, 'trend': trend, 'breakout': breakout, 'liquidity': liquidity,
        'momentum': momentum,
    }

def won(v):
    return f"{v:,.0f}원"

with st.sidebar:
    st.header("분석 설정")
    market = st.selectbox("시장", ["KRX", "KOSPI", "KOSDAQ"], index=0)
    universe = st.select_slider("분석할 대형주 수", options=[50, 100, 150, 200], value=100)
    capital = st.number_input("투자 예정 금액 (원)", min_value=10_000, max_value=100_000_000, value=CAPITAL_DEFAULT, step=10_000)
    risk_pct = st.select_slider("거래당 허용 손실률", options=[0.5, 1.0, 1.5, 2.0], value=1.0, format_func=lambda x: f"{x:.1f}%")
    st.caption("기본값: 20만 원 · 거래당 최대 계획 손실 1% (2,000원)")
    mode = st.radio("분석 대상", ["시스템이 후보 하나 선정", "종목 직접 선택"], index=0)

st.markdown("### 분석 준비 완료")
st.caption("설정을 확인한 뒤 아래 버튼을 한 번 누르면 종목 분석을 시작합니다.")
run_analysis = st.button("알려줘", type="primary", use_container_width=True)

if run_analysis:
    end = date.today()
    start = end - timedelta(days=220)

    try:
        with st.spinner("상장 종목 목록을 불러오는 중..."):
            listing = get_listing(market).head(universe)
        if mode == "종목 직접 선택":
            options = {f"{r['Name']} ({r['Symbol']})": (str(r['Symbol']), str(r['Name'])) for _, r in listing.iterrows()}
            chosen = st.selectbox("분석할 종목", list(options.keys()))
            code, name = options[chosen]
            with st.spinner(f"{name}의 가격 데이터를 분석 중..."):
                hist = get_history(code, start.isoformat(), (end + timedelta(days=1)).isoformat())
            r = calculate(code, name, hist) if not hist.empty else None
            results = [r] if r else []
        else:
            results = []
            progress = st.progress(0, text="후보 종목을 분석하고 있습니다...")
            total = len(listing)
            for i, (_, row) in enumerate(listing.iterrows(), start=1):
                code, name = str(row['Symbol']), str(row['Name'])
                try:
                    hist = get_history(code, start.isoformat(), (end + timedelta(days=1)).isoformat())
                    r = calculate(code, name, hist) if not hist.empty else None
                    if r:
                        results.append(r)
                except Exception:
                    pass
                if i % 5 == 0 or i == total:
                    progress.progress(i / max(total, 1), text=f"분석 중: {i}/{total}")
            progress.empty()
            # Fresh data first; qualifying signals first; then activity/volume/trend.
            results.sort(key=lambda x: (x['fresh'], x['signal'], x['turnover_ratio'], x['volume_ratio'], x['daily_pct']), reverse=True)
            results = results[:1]

        if not results:
            st.error("분석할 유효한 데이터가 없습니다. 연결 상태를 확인하거나 종목을 직접 선택해 주세요.")
        else:
            r = results[0]
            st.caption(f"종목코드 {r['code']} · 데이터 기준일 {r['date']}")
            st.subheader(f"{r['name']} ({r['code']})")
            if not r['fresh']:
                st.error(f"데이터가 오래됐습니다 ({r['age_days']}일 경과). 신호·매수 수량을 제공하지 않습니다. 데이터 갱신 후 다시 분석하세요.")
            else:
                st.metric("최근 종가", won(r['close']), f"{r['daily_pct']:+.2f}%")
                if r['signal']:
                    st.success("조건 충족: 진입 후보. 다음 거래일에 실시간 가격과 돌파 유지 여부를 다시 확인하세요.")
                else:
                    st.info("현재는 대기. 모든 조건이 동시에 충족되지 않아 신규 진입 신호를 내지 않습니다.")

                st.subheader("진입·청산 기준")
                c1, c2 = st.columns(2)
                c1.metric("진입 확인 가격", won(r['trigger']))
                c2.metric("손절 기준", won(r['stop']), f"-{(r['trigger']-r['stop'])/r['trigger']*100:.1f}%")
                c3, c4 = st.columns(2)
                c3.metric("목표가 기준", won(r['target']), f"+{(r['target']-r['trigger'])/r['trigger']*100:.1f}%")
                c4.metric("손익비", "1 : 1.5")

                risk_budget = float(capital) * float(risk_pct) / 100
                per_share_risk = max(0, r['trigger'] - r['stop'])
                qty_by_risk = math.floor(risk_budget / per_share_risk) if per_share_risk > 0 else 0
                qty_by_cash = math.floor(float(capital) / r['trigger']) if r['trigger'] > 0 else 0
                qty = max(0, min(qty_by_risk, qty_by_cash)) if r['signal'] else 0
                est_cost = qty * r['trigger']
                est_risk = qty * per_share_risk
                est_profit = qty * (r['target'] - r['trigger'])
                st.subheader("20만 원 기준 매수 수량")
                a, b = st.columns(2)
                a.metric("계산 매수 수량", f"{qty:,}주" if r['signal'] else "0주 · 대기")
                b.metric("예상 매수 금액", won(est_cost))
                st.caption(f"허용 손실 예산 {won(risk_budget)} · 계획상 손절 손실 약 {won(est_risk)} · 목표 도달 시 예상 이익 약 {won(est_profit)} (수수료·세금·슬리피지 제외). 실제 체결가는 달라질 수 있습니다.")

                st.subheader("진입 조건 확인")
                checks = [
                    ("데이터 최신성 (5일 이내)", r['fresh']),
                    ("상승 추세: 종가 > 20일선 > 50일선", r['trend']),
                    ("20거래일 고점 돌파", r['breakout']),
                    ("추정 거래대금 10억 원 이상", r['liquidity']),
                    ("거래대금 20일 평균 대비 1.5배 이상", r['turnover_ratio'] >= 1.5),
                    ("거래량 20일 평균 대비 1.2배 이상", r['volume_ratio'] >= 1.2),
                    ("당일 상승률 8% 미만", r['daily_pct'] < 8),
                ]
                for label, passed in checks:
                    st.write(("✅ " if passed else "⬜ ") + label)
                st.subheader("나오는 규칙")
                st.write("- **손절:** 일봉 종가가 손절 기준 아래로 내려가면 다음 거래 시 청산을 검토합니다. 갭 하락 시 손실은 계산보다 커질 수 있습니다.")
                st.write("- **목표가:** 목표가에 도달하면 계획대로 일부/전량 청산을 검토합니다. 목표가 도달을 보장하지 않습니다.")
                st.write("- **보유 기간:** 스윙 전략이므로 매일 종가 후 재평가합니다. 신호가 사라지거나 데이터가 오래되면 신규 진입하지 않습니다.")
                st.write(f"- **거래대금 증가:** {r['turnover_ratio']:.2f}배 · **거래량 증가:** {r['volume_ratio']:.2f}배 · **ATR(14):** {won(r['atr'])}")

                out = pd.DataFrame([{
                    '기준일': r['date'], '종목명': r['name'], '종목코드': r['code'], '최근종가': round(r['close']),
                    '진입확인가격': round(r['trigger']), '손절기준': round(r['stop']), '목표가기준': round(r['target']),
                    '신호': '진입 후보' if r['signal'] else '대기', '투자금': int(capital), '허용손실률(%)': risk_pct,
                    '매수수량(주)': qty, '예상매수금액': round(est_cost), '계획손실액': round(est_risk),
                    '목표이익(비용제외)': round(est_profit), '데이터경과일': r['age_days'],
                    '거래대금증가배수': round(r['turnover_ratio'], 2), '거래량증가배수': round(r['volume_ratio'], 2),
                }])
                st.download_button("분석 결과 CSV 다운로드", out.to_csv(index=False).encode('utf-8-sig'), file_name=f"스윙_단일종목_{r['date']}.csv", mime="text/csv", use_container_width=True)
    except Exception as e:
        st.error("데이터 제공처에 연결하지 못했습니다. 인터넷 연결을 확인하거나 잠시 후 다시 실행해 주세요.")
        st.caption(f"오류 유형: {type(e).__name__}. 자세한 오류에는 제공처 내부 메시지가 포함될 수 있어 일부만 표시합니다.")

else:
    st.info("분석을 시작하려면 **알려줘** 버튼을 누르세요.")

st.divider()
st.caption("교육·연구용 도구입니다. 투자자문이나 매매 주문 기능이 아닙니다. 실제 투자 전 최신 데이터, 공시, 거래비용과 본인의 위험 감수 수준을 확인하세요.")
