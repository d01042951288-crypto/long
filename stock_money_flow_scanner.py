"""Korean stock money-flow candidate scanner (research-only).

Data source: FinanceDataReader; estimated turnover = Close * Volume.
This is a candidate screener, not investment advice or an order-execution bot.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, datetime, timedelta
from zoneinfo import ZoneInfo
import os
import sys
import time
import warnings

try:
    import FinanceDataReader as fdr
    import pandas as pd
except ImportError as exc:
    raise SystemExit(
        "필요한 패키지가 없습니다. 먼저 아래 명령을 실행하세요:\n"
        "python -m pip install -U finance-datareader pandas\n"
    ) from exc

# ===== 사용자 설정: 필요에 따라 이 값만 바꾸세요 =====
MARKET = "KRX"               # "KRX" = 코스피+코스닥+코넥스, 또는 "KOSPI", "KOSDAQ"
UNIVERSE_LIMIT = 500         # 시가총액 상위 몇 종목을 조사할지. 0이면 전체(느릴 수 있음)
MAX_WORKERS = 4              # 데이터 요청 동시 실행 수. 너무 크게 올리지 마세요.
HISTORY_CALENDAR_DAYS = 120  # 거래일 약 80일 확보
MIN_PRICE_WON = 1000
MIN_EST_TURNOVER_WON = 1_000_000_000   # 10억 원: 종가×거래량 추정치
MIN_TURNOVER_RATIO = 1.5               # 최근 20거래일 평균 대비 추정 거래대금 배수
MIN_VOLUME_RATIO = 1.2                 # 최근 20거래일 평균 대비 거래량 배수
MIN_DAILY_CHANGE_PCT = 0.5
MAX_DAILY_CHANGE_PCT = 15.0             # 급등·과열 종목은 기본 후보에서 제외
TOP_DISPLAY = 30
# =====================================================


def _score(turnover_ratio: float, volume_ratio: float, daily_pct: float,
           return_5d_pct: float, close_vs_high20_pct: float) -> int:
    score = 0
    if turnover_ratio >= 3.0:
        score += 35
    elif turnover_ratio >= 2.0:
        score += 28
    elif turnover_ratio >= 1.5:
        score += 20
    elif turnover_ratio >= 1.2:
        score += 10

    if volume_ratio >= 3.0:
        score += 20
    elif volume_ratio >= 2.0:
        score += 16
    elif volume_ratio >= 1.5:
        score += 12
    elif volume_ratio >= 1.2:
        score += 7

    if 3.0 <= daily_pct <= 10.0:
        score += 20
    elif 0.5 <= daily_pct < 3.0:
        score += 15
    elif 0.0 < daily_pct < 0.5:
        score += 5
    elif daily_pct > 10.0:
        score += 8  # 높은 점수를 주기보다는 과열 가능성을 반영

    if return_5d_pct >= 10.0:
        score += 15
    elif return_5d_pct >= 3.0:
        score += 12
    elif return_5d_pct > 0.0:
        score += 8
    elif return_5d_pct >= -3.0:
        score += 4

    if close_vs_high20_pct >= -1.0:
        score += 10
    elif close_vs_high20_pct >= -5.0:
        score += 6

    return min(score, 100)


def _fetch_one(code: str, name: str, start_date: str, end_date: str):
    """Fetch one ticker and calculate transparent price/volume proxies."""
    try:
        hist = fdr.DataReader(code, start_date, end_date)
        if hist is None or hist.empty or not {"Close", "Volume"}.issubset(hist.columns):
            return None
        hist = hist.copy().sort_index()
        for col in ("Close", "Volume"):
            hist[col] = pd.to_numeric(hist[col], errors="coerce")
        if "High" in hist.columns:
            hist["High"] = pd.to_numeric(hist["High"], errors="coerce")
        hist = hist.dropna(subset=["Close", "Volume"])
        hist = hist[(hist["Close"] > 0) & (hist["Volume"] >= 0)]
        if len(hist) < 22:
            return None

        latest = hist.iloc[-1]
        prior = hist.iloc[:-1].tail(20)
        if len(prior) < 15:
            return None
        close = float(latest["Close"])
        volume = float(latest["Volume"])
        est_turnover = close * volume
        prior_turnover = (prior["Close"] * prior["Volume"]).mean()
        prior_volume = prior["Volume"].mean()
        if not prior_turnover or prior_turnover <= 0 or not prior_volume or prior_volume <= 0:
            return None

        turnover_ratio = est_turnover / float(prior_turnover)
        volume_ratio = volume / float(prior_volume)
        prev_close = float(hist["Close"].iloc[-2])
        daily_pct = (close / prev_close - 1.0) * 100.0 if prev_close else 0.0
        close_5_sessions_ago = float(hist["Close"].iloc[-6])
        return_5d_pct = (close / close_5_sessions_ago - 1.0) * 100.0 if close_5_sessions_ago else 0.0

        if "High" in hist.columns:
            high20 = float(hist["High"].tail(20).max())
        else:
            high20 = float(hist["Close"].tail(20).max())
        close_vs_high20_pct = (close / high20 - 1.0) * 100.0 if high20 else 0.0

        return {
            "기준일": pd.Timestamp(hist.index[-1]).date().isoformat(),
            "종목명": name,
            "종목코드": str(code).zfill(6),
            "종가(원)": round(close, 0),
            "당일등락률(%)": daily_pct,
            "추정거래대금(원)": est_turnover,
            "거래대금증가배수": turnover_ratio,
            "거래량증가배수": volume_ratio,
            "5일등락률(%)": return_5d_pct,
            "20일고점대비(%)": close_vs_high20_pct,
            "점수(100점)": _score(turnover_ratio, volume_ratio, daily_pct, return_5d_pct, close_vs_high20_pct),
        }
    except Exception:
        return None


def main() -> None:
    warnings.filterwarnings("ignore", category=FutureWarning)
    end = datetime.now(ZoneInfo("Asia/Seoul")).date()
    start = end - timedelta(days=HISTORY_CALENDAR_DAYS)
    start_s, end_s = start.isoformat(), end.isoformat()

    print(f"시장: {MARKET} | 조회 기간: {start_s} ~ {end_s}")
    print("상장 종목 목록을 불러오는 중...")
    listing = fdr.StockListing(MARKET)
    required = {"Symbol", "Name"}
    if listing is None or listing.empty or not required.issubset(listing.columns):
        raise RuntimeError("종목 목록 형식이 예상과 다릅니다. FinanceDataReader를 업데이트하거나 나중에 다시 시도하세요.")

    listing = listing.copy()
    listing["Symbol"] = listing["Symbol"].astype(str).str.strip()
    listing = listing[listing["Symbol"].str.fullmatch(r"\d{6}", na=False)]
    listing = listing.drop_duplicates(subset=["Symbol"])
    if "Marcap" in listing.columns:
        listing["Marcap"] = pd.to_numeric(listing["Marcap"], errors="coerce")
        listing = listing.sort_values("Marcap", ascending=False, na_position="last")
    if UNIVERSE_LIMIT and UNIVERSE_LIMIT > 0:
        listing = listing.head(UNIVERSE_LIMIT)
    items = [(row["Symbol"], str(row["Name"])) for _, row in listing.iterrows()]
    if not items:
        raise RuntimeError("분석할 종목을 찾지 못했습니다.")

    print(f"분석 대상: {len(items)}개 종목. 가격 데이터 조회를 시작합니다...")
    rows = []
    with ThreadPoolExecutor(max_workers=max(1, MAX_WORKERS)) as executor:
        futures = {
            executor.submit(_fetch_one, code, name, start_s, end_s): code
            for code, name in items
        }
        for count, future in enumerate(as_completed(futures), start=1):
            try:
                row = future.result()
                if row:
                    rows.append(row)
            except Exception:
                pass
            if count % 50 == 0 or count == len(futures):
                print(f"진행: {count}/{len(futures)} | 분석 가능한 종목 {len(rows)}개")

    if not rows:
        raise RuntimeError(
            "가격 데이터를 가져오지 못했습니다. 네트워크 연결, 데이터 제공처 상태, 패키지 버전을 확인하세요.\n"
            "설치 명령: python -m pip install -U finance-datareader pandas"
        )

    data = pd.DataFrame(rows)
    # Use the most recent date observed across the fetched universe; discard stale rows.
    latest_date = data["기준일"].max()
    data = data[data["기준일"] == latest_date].copy()

    candidates = data[
        (data["종가(원)"] >= MIN_PRICE_WON)
        & (data["추정거래대금(원)"] >= MIN_EST_TURNOVER_WON)
        & (data["거래대금증가배수"] >= MIN_TURNOVER_RATIO)
        & (data["거래량증가배수"] >= MIN_VOLUME_RATIO)
        & (data["당일등락률(%)"] >= MIN_DAILY_CHANGE_PCT)
        & (data["당일등락률(%)"] <= MAX_DAILY_CHANGE_PCT)
    ].copy()
    data = data.sort_values(["점수(100점)", "거래대금증가배수"], ascending=False)
    candidates = candidates.sort_values(["점수(100점)", "거래대금증가배수"], ascending=False)

    display_cols = [
        "기준일", "종목명", "종목코드", "종가(원)", "당일등락률(%)",
        "추정거래대금(원)", "거래대금증가배수", "거래량증가배수",
        "5일등락률(%)", "20일고점대비(%)", "점수(100점)",
    ]
    for frame in (data, candidates):
        for col in ["당일등락률(%)", "거래대금증가배수", "거래량증가배수", "5일등락률(%)", "20일고점대비(%)"]:
            frame[col] = frame[col].astype(float).round(2)
        frame["추정거래대금(억원)"] = (frame["추정거래대금(원)"] / 100_000_000).round(2)

    print("\n" + "=" * 68)
    print(f"기준일: {latest_date} | 조사한 데이터 종목: {len(data)}개")
    print(f"조건을 통과한 후보: {len(candidates)}개")
    print("\n[필터 조건 통과 후보 — 상위 30개]")
    if candidates.empty:
        print("조건을 통과한 종목이 없습니다. 설정에서 MIN_TURNOVER_RATIO, MIN_VOLUME_RATIO 등을 낮춰 다시 실행할 수 있습니다.")
    else:
        show_cols = ["종목명", "종목코드", "종가(원)", "당일등락률(%)", "추정거래대금(억원)", "거래대금증가배수", "거래량증가배수", "5일등락률(%)", "점수(100점)"]
        print(candidates[show_cols].head(TOP_DISPLAY).to_string(index=False))

    print("\n[전체 분석 대상 중 점수 상위 30개 — 필터 미통과 종목 포함]")
    show_cols = ["종목명", "종목코드", "종가(원)", "당일등락률(%)", "추정거래대금(억원)", "거래대금증가배수", "거래량증가배수", "5일등락률(%)", "점수(100점)"]
    print(data[show_cols].head(TOP_DISPLAY).to_string(index=False))

    tag = latest_date.replace("-", "")
    out_all = f"주식수급스캐너_전체점수_{tag}.csv"
    out_candidates = f"주식수급스캐너_필터통과_{tag}.csv"
    data[display_cols + ["추정거래대금(억원)"]].to_csv(out_all, index=False, encoding="utf-8-sig")
    candidates[display_cols + ["추정거래대금(억원)"]].to_csv(out_candidates, index=False, encoding="utf-8-sig")
    print(f"\n파일 저장 완료:\n- {out_all}\n- {out_candidates}")
    globals()["out_all"] = out_all
    globals()["out_candidates"] = out_candidates
    print("\n주의: 거래대금은 종가×거래량 추정치이며, 실제 투자자별 순매수 데이터가 아닙니다.")


if __name__ == "__main__":
    main()
