"""Korean long-term stock research screener. Research only; no trading execution."""
from datetime import date, timedelta, datetime
from zoneinfo import ZoneInfo
import math
import pandas as pd
import FinanceDataReader as fdr
import streamlit as st

st.set_page_config(page_title="장기투자 종목 분석", page_icon="🌱", layout="centered")
st.title("🌱 장기투자 분석기")
st.caption("국내 주식 · 1년 이상 관점 · 장기 추세와 위험 지표 · 자동매매 아님")
st.warning("분석 점수는 과거 가격 데이터를 요약한 참고 지표입니다. 미래 수익을 예측하거나 매수를 보장하지 않습니다. 투자 전 사업·재무제표·공시·수수료를 별도로 확인하세요.")

MAX_DATA_AGE_DAYS = 7

@st.cache_data(ttl=1800, show_spinner=False)
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

@st.cache_data(ttl=1800, show_spinner=False)
def get_history(code: str, start: str, end: str) -> pd.DataFrame:
    df = fdr.DataReader(code, start, end)
    if df is None or df.empty or 'Close' not in df.columns:
        return pd.DataFrame()
    df = df.copy().sort_index()
    for c in ['Open', 'High', 'Low', 'Close', 'Volume']:
        if c in df:
            df[c] = pd.to_numeric(df[c], errors='coerce')
    return df.dropna(subset=['Close'])

def safe_num(row, key):
    try:
        v = pd.to_numeric(row.get(key), errors='coerce')
        return float(v) if pd.notna(v) else None
    except Exception:
        return None

def analyze(code: str, name: str, hist: pd.DataFrame, listing_row=None):
    if hist is None or len(hist) < 220:
        return None
    d = hist.copy().sort_index()
    d['SMA50'] = d['Close'].rolling(50).mean()
    d['SMA200'] = d['Close'].rolling(200).mean()
    d['DailyReturn'] = d['Close'].pct_change()
    d['Volatility252'] = d['DailyReturn'].rolling(252).std() * math.sqrt(252) * 100
    last = d.iloc[-1]
    asof = pd.Timestamp(d.index[-1]).date()
    today = datetime.now(ZoneInfo('Asia/Seoul')).date()
    age = (today - asof).days
    close = float(last['Close'])
    sma50 = float(last['SMA50']) if pd.notna(last['SMA50']) else float('nan')
    sma200 = float(last['SMA200']) if pd.notna(last['SMA200']) else float('nan')
    if not all(math.isfinite(x) and x > 0 for x in [close, sma50, sma200]):
        return None
    ret1y = (close / float(d['Close'].iloc[-253]) - 1) * 100 if len(d) >= 253 else float('nan')
    ret3y = (close / float(d['Close'].iloc[-756]) - 1) * 100 if len(d) >= 756 else None
    period = d['Close'].tail(min(756, len(d)))
    peak = period.cummax()
    max_dd = float(((period / peak) - 1).min() * 100)
    vol = float(last['Volatility252']) if pd.notna(last['Volatility252']) else None
    fresh = 0 <= age <= MAX_DATA_AGE_DAYS
    trend = close > sma200 and sma50 > sma200
    positive_1y = math.isfinite(ret1y) and ret1y > 0
    positive_3y = ret3y is not None and ret3y > 0
    moderate_vol = vol is not None and vol <= 45
    # Optional valuation data: source columns vary by market and provider.
    per = pbr = eps = dps = None
    if listing_row is not None:
        per = safe_num(listing_row, 'PER')
        pbr = safe_num(listing_row, 'PBR')
        eps = safe_num(listing_row, 'EPS')
        dps = safe_num(listing_row, 'DPS')
    score = 0
    score += 30 if trend else 0
    score += 20 if positive_1y else 0
    score += 15 if positive_3y else 0
    score += 15 if max_dd > -35 else 0
    score += 10 if moderate_vol else 0
    score += 5 if eps is not None and eps > 0 else 0
    score += 5 if per is not None and 0 < per <= 25 else 0
    return {
        'code': code, 'name': name, 'date': asof.isoformat(), 'age': age, 'fresh': fresh,
        'close': close, 'sma50': sma50, 'sma200': sma200, 'ret1y': ret1y, 'ret3y': ret3y,
        'max_dd': max_dd, 'vol': vol, 'trend': trend, 'positive_1y': positive_1y,
        'positive_3y': positive_3y, 'moderate_vol': moderate_vol, 'score': score,
        'per': per, 'pbr': pbr, 'eps': eps, 'dps': dps,
    }

def won(v): return f"{v:,.0f}원"
def pct(v): return f"{v:+.1f}%" if v is not None and math.isfinite(v) else "자료 없음"

with st.sidebar:
    st.header("분석 설정")
    market = st.selectbox("시장", ["KRX", "KOSPI", "KOSDAQ"], index=0)
    universe = st.select_slider("시가총액 상위 분석 종목 수", options=[30, 50, 100, 150], value=50)
    capital = st.number_input("투자 예정 금액 (원)", min_value=10_000, max_value=100_000_000, value=300_000, step=10_000)
    mode = st.radio("분석 대상", ["시스템이 관찰 후보 선정", "종목 직접 선택"], index=0)
    st.caption("장기 관점 예시: 1년 이상. 분산투자와 정기적인 재검토를 고려하세요.")

st.write("**분석 기준:** 50일·200일 이동평균, 1년/3년 가격 변화, 최대 낙폭, 연환산 변동성, 데이터 최신성. PER·EPS 등 재무 지표는 제공처에서 받을 수 있을 때만 표시합니다.")
if st.button("알려줘", type="primary", use_container_width=True):
    try:
        with st.spinner("상장 종목 목록을 불러오는 중..."):
            listing = get_listing(market)
        # Exclude tiny/no-cap entries when possible, then scan the chosen universe.
        listing = listing.head(universe)
        rows = []
        if mode == "종목 직접 선택":
            opts = {f"{r['Name']} ({r['Symbol']})": r for _, r in listing.iterrows()}
            if not opts:
                st.error("선택할 종목 목록이 없습니다.")
                st.stop()
            selected_label = st.selectbox("분석할 종목", list(opts.keys()), key="long_selected")
            selected = opts[selected_label]
            with st.spinner(f"{selected['Name']}의 장기 데이터를 분석 중..."):
                hist = get_history(str(selected['Symbol']), (date.today() - timedelta(days=1500)).isoformat(), (date.today() + timedelta(days=1)).isoformat())
            r = analyze(str(selected['Symbol']), str(selected['Name']), hist, selected) if not hist.empty else None
            if r: rows.append(r)
        else:
            progress = st.progress(0, text="장기 지표를 계산하고 있습니다...")
            for i, (_, row) in enumerate(listing.iterrows(), start=1):
                try:
                    code, name = str(row['Symbol']), str(row['Name'])
                    hist = get_history(code, (date.today() - timedelta(days=1500)).isoformat(), (date.today() + timedelta(days=1)).isoformat())
                    r = analyze(code, name, hist, row) if not hist.empty else None
                    if r: rows.append(r)
                except Exception:
                    pass
                if i % 5 == 0 or i == len(listing):
                    progress.progress(i / max(len(listing), 1), text=f"분석 중: {i}/{len(listing)}")
            progress.empty()
            rows.sort(key=lambda x: (x['fresh'], x['score'], x['ret1y'] if math.isfinite(x['ret1y']) else -999), reverse=True)
            rows = rows[:1]
        if not rows:
            st.error("유효한 장기 가격 데이터가 없습니다. 종목 수를 줄이거나 직접 선택을 확인해 주세요.")
        else:
            r = rows[0]
            st.caption(f"종목코드 {r['code']} · 데이터 기준일 {r['date']}")
            st.subheader(f"관찰 후보: {r['name']} ({r['code']})")
            if not r['fresh']:
                st.error(f"데이터가 오래됐습니다({r['age']}일 경과). 최신 데이터로 확인할 때까지 분석 점수를 신뢰하지 마세요.")
            else:
                st.metric("최근 종가", won(r['close']))
            st.metric("장기 지표 점수", f"{r['score']}/100", help="과거 가격 추세·변동성·일부 제공 가능한 재무 지표를 단순 합산한 연구용 점수이며, 기대수익률이나 매수 확률이 아닙니다.")
            a, b = st.columns(2)
            a.metric("1년 가격 변화", pct(r['ret1y']))
            b.metric("3년 가격 변화", pct(r['ret3y']))
            c, d2 = st.columns(2)
            c.metric("50일 이동평균", won(r['sma50']))
            d2.metric("200일 이동평균", won(r['sma200']))
            st.metric("최근 최대 낙폭(최대 약 3년)", pct(r['max_dd']))
            st.metric("연환산 변동성", pct(r['vol']))
            st.subheader("장기 점검 항목")
            checks = [
                ("데이터 최신성(7일 이내)", r['fresh']),
                ("장기 상승 추세: 종가 > 200일선 및 50일선 > 200일선", r['trend']),
                ("1년 가격 변화가 양수", r['positive_1y']),
                ("3년 가격 변화가 양수(데이터 충분 시)", r['positive_3y'] if r['ret3y'] is not None else None),
                ("최대 낙폭이 -35%보다 얕음", r['max_dd'] > -35),
                ("연환산 변동성 45% 이하", r['moderate_vol'] if r['moderate_vol'] is not None else None),
                ("EPS가 양수(제공될 때만)", r['eps'] > 0 if r['eps'] is not None else None),
                ("PER 0~25배(제공될 때만)", 0 < r['per'] <= 25 if r['per'] is not None else None),
            ]
            for label, passed in checks:
                st.write(("✅ " if passed is True else "⬜ " if passed is False else "➖ 자료 없음: ") + label)
            st.subheader("투자금 예시")
            qty = math.floor(float(capital) / r['close']) if r['close'] > 0 else 0
            st.metric("예산 내 최대 정수 주식 수", f"{qty:,}주")
            st.metric("예상 투입 금액", won(qty * r['close']))
            st.caption("계산은 투자금 전액을 한 종목에 넣는 단순 예시입니다. 실제로는 한 종목 집중을 피하고, 현금 여유·거래비용·분할매수 계획을 고려하세요. 매수 권유가 아닙니다.")
            st.subheader("장기 투자에서 별도로 확인할 것")
            st.write("- 최근 사업보고서의 매출·영업이익·순이익·영업현금흐름 추세")
            st.write("- 부채 수준, 주식 수 증가/희석, 경쟁력, 산업 전망, 배당 지속 가능성")
            st.write("- 공시·감사의견·관리종목/거래정지 여부와 현재 가격의 가치평가")
            out = pd.DataFrame([{
                '기준일': r['date'], '종목명': r['name'], '종목코드': r['code'], '최근종가': round(r['close']),
                '장기지표점수': r['score'], '1년변화(%)': round(r['ret1y'], 2) if math.isfinite(r['ret1y']) else None,
                '3년변화(%)': round(r['ret3y'], 2) if r['ret3y'] is not None else None,
                '최대낙폭(%)': round(r['max_dd'], 2), '연환산변동성(%)': round(r['vol'], 2) if r['vol'] is not None else None,
                'PER': r['per'], 'PBR': r['pbr'], 'EPS': r['eps'], 'DPS': r['dps'],
                '예산': int(capital), '계산수량(주)': qty, '예상투입금액': round(qty * r['close']), '데이터경과일': r['age'],
            }])
            st.download_button("장기 분석 결과 CSV 다운로드", out.to_csv(index=False).encode('utf-8-sig'), file_name=f"장기투자_분석_{r['date']}.csv", mime="text/csv", use_container_width=True)
    except Exception as e:
        st.error("데이터 제공처에 연결하지 못했습니다. 잠시 후 다시 시도하거나 분석 종목 수를 줄여 주세요.")
        st.caption(f"오류 유형: {type(e).__name__}")
else:
    st.info("분석을 시작하려면 **알려줘** 버튼을 누르세요.")

st.divider()
st.caption("교육·연구용 도구이며 투자자문이 아닙니다. 과거 성과는 미래 수익을 보장하지 않습니다. 자동 주문 기능은 없습니다.")
