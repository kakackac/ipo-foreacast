"""
data/processors/financial_analyzer.py
───────────────────────────────────────
재무제표 기반 공모 가격 매력도 분석 모듈.

수행 작업:
  1. DART에서 수집한 재무 데이터를 정제·정규화
  2. PER, PBR, EV/EBITDA 등 밸류에이션 지표 산출
  3. 동일 섹터 상장사 대비 상대 밸류에이션 계산
  4. 성장성·수익성·안정성 점수 통합 → 재무 매력도 점수 (0~100)

재무 매력도 점수 구성:
  - 밸류에이션 (40%): PER, PBR 섹터 대비 할인 여부
  - 성장성 (30%):     매출·영업이익 CAGR
  - 수익성 (20%):     영업이익률, ROE
  - 안정성 (10%):     부채비율, 이자보상배율

적자 기업 처리:
  - PER 계산 불가 → 섹터 PER 중앙값 × 1.5 패널티 적용
  - 성장성 점수는 유지 (적자라도 고성장이면 가산)
"""

import logging
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ── 섹터별 상장사 평균 PER (2024 기준 추정, 주기적 업데이트 필요) ──
SECTOR_MEDIAN_PER: dict[str, float] = {
    "IT":         28.0,
    "바이오":      45.0,
    "제조":        12.0,
    "소비재":      18.0,
    "금융":        10.0,
    "화학":        11.0,
    "에너지":       9.0,
    "건설":         8.0,
    "통신":        14.0,
    "유통":        15.0,
    "기타":        16.0,
}

SECTOR_MEDIAN_PBR: dict[str, float] = {
    "IT":         3.5,
    "바이오":      5.0,
    "제조":        1.1,
    "소비재":      2.0,
    "금융":        0.7,
    "화학":        1.0,
    "에너지":       0.9,
    "건설":         0.8,
    "통신":        1.5,
    "유통":        1.2,
    "기타":        1.3,
}


class FinancialAnalyzer:
    """
    재무제표 → 밸류에이션 지표 + 매력도 점수 산출기.

    사용법:
        analyzer = FinancialAnalyzer()
        result = analyzer.analyze(corp_financials, offering_price, sector)
    """

    # ── 메인 분석 ─────────────────────────────────────────────

    def analyze(
        self,
        financials:     dict,       # DART 수집 재무 데이터
        offering_price: float,
        total_shares:   float,      # 상장 후 총 발행주식수
        sector:         str = "기타",
    ) -> dict:
        """
        단일 종목 재무 매력도 전체 분석.

        financials 키:
          revenue_t0, revenue_t1, revenue_t2 (최근 3년 매출)
          operating_income_t0, t1, t2
          net_income_t0
          total_assets, total_liabilities, equity
          eps (주당순이익)
          ebitda (없으면 영업이익 사용)

        반환:
          밸류에이션 지표 + 성장성 + 점수
        """
        result = {"sector": sector, "offering_price": offering_price}

        # ── 1. 기본 밸류에이션 ────────────────────────────────
        val = self._calc_valuation(financials, offering_price, total_shares, sector)
        result.update(val)

        # ── 2. 성장성 지표 ────────────────────────────────────
        growth = self._calc_growth(financials)
        result.update(growth)

        # ── 3. 수익성 지표 ────────────────────────────────────
        profitability = self._calc_profitability(financials)
        result.update(profitability)

        # ── 4. 안정성 지표 ────────────────────────────────────
        stability = self._calc_stability(financials)
        result.update(stability)

        # ── 5. 통합 매력도 점수 ───────────────────────────────
        result["valuation_score"]      = self._score_valuation(val)
        result["growth_score"]         = self._score_growth(growth)
        result["profitability_score"]  = self._score_profitability(profitability)
        result["stability_score"]      = self._score_stability(stability)

        result["financial_attractiveness"] = round(
            result["valuation_score"]     * 0.40 +
            result["growth_score"]        * 0.30 +
            result["profitability_score"] * 0.20 +
            result["stability_score"]     * 0.10,
            1,
        )
        result["attractiveness_grade"] = self._grade(result["financial_attractiveness"])
        return result

    # ── 밸류에이션 계산 ───────────────────────────────────────

    def _calc_valuation(
        self,
        fin:            dict,
        offering_price: float,
        total_shares:   float,
        sector:         str,
    ) -> dict:
        market_cap = offering_price * total_shares  # 공모 기준 시총

        # PER
        eps = fin.get("eps")
        if eps and eps > 0:
            per = offering_price / eps
            per_flag = "normal"
        elif fin.get("net_income_t0", 0) > 0 and total_shares > 0:
            per = market_cap / fin["net_income_t0"]
            per_flag = "calculated"
        else:
            per = None
            per_flag = "deficit"   # 적자

        # PBR
        equity = fin.get("equity")
        pbr = (market_cap / equity) if equity and equity > 0 else None

        # EV/EBITDA
        ebitda = fin.get("ebitda") or fin.get("operating_income_t0")
        liabilities = fin.get("total_liabilities", 0)
        ev = market_cap + (liabilities or 0)
        ev_ebitda = (ev / ebitda) if ebitda and ebitda > 0 else None

        # 섹터 대비 할인율
        sector_per = SECTOR_MEDIAN_PER.get(sector, SECTOR_MEDIAN_PER["기타"])
        sector_pbr = SECTOR_MEDIAN_PBR.get(sector, SECTOR_MEDIAN_PBR["기타"])

        per_vs_sector = (per / sector_per) if per and sector_per else None
        pbr_vs_sector = (pbr / sector_pbr) if pbr and sector_pbr else None

        return {
            "market_cap_billion":  round(market_cap / 1e8, 1) if market_cap else None,  # 억원
            "offering_per":        round(per, 1) if per else None,
            "per_flag":            per_flag,
            "offering_pbr":        round(pbr, 2) if pbr else None,
            "ev_ebitda":           round(ev_ebitda, 1) if ev_ebitda else None,
            "sector_median_per":   sector_per,
            "per_vs_sector":       round(per_vs_sector, 2) if per_vs_sector else None,
            "pbr_vs_sector":       round(pbr_vs_sector, 2) if pbr_vs_sector else None,
        }

    def _calc_growth(self, fin: dict) -> dict:
        rev = [fin.get(f"revenue_t{i}") for i in range(3)]
        opi = [fin.get(f"operating_income_t{i}") for i in range(3)]

        rev_cagr = self._cagr(rev[2], rev[0], 2)
        opi_cagr = self._cagr(opi[2], opi[0], 2) if all(v and v > 0 for v in [opi[0], opi[2]]) else None

        # 최근 1년 성장률 (YoY)
        rev_yoy = (rev[0] / rev[1] - 1) if rev[0] and rev[1] and rev[1] > 0 else None
        opi_yoy = (opi[0] / opi[1] - 1) if opi[0] and opi[1] and opi[1] > 0 else None

        return {
            "revenue_cagr_2y":      round(rev_cagr, 4) if rev_cagr is not None else None,
            "operating_income_cagr": round(opi_cagr, 4) if opi_cagr is not None else None,
            "revenue_yoy":          round(rev_yoy, 4) if rev_yoy is not None else None,
            "operating_income_yoy": round(opi_yoy, 4) if opi_yoy is not None else None,
        }

    def _calc_profitability(self, fin: dict) -> dict:
        rev = fin.get("revenue_t0")
        opi = fin.get("operating_income_t0")
        net = fin.get("net_income_t0")
        eq  = fin.get("equity")

        opm  = (opi / rev)     if opi is not None and rev and rev > 0 else None
        npm  = (net / rev)     if net is not None and rev and rev > 0 else None
        roe  = (net / eq)      if net is not None and eq  and eq  > 0 else None

        return {
            "operating_margin": round(opm, 4) if opm is not None else None,
            "net_margin":       round(npm, 4) if npm is not None else None,
            "roe":              round(roe, 4) if roe is not None else None,
            "is_profitable":    bool(net and net > 0),
        }

    def _calc_stability(self, fin: dict) -> dict:
        liab = fin.get("total_liabilities")
        eq   = fin.get("equity")
        opi  = fin.get("operating_income_t0")
        interest = fin.get("interest_expense_t0")

        debt_ratio  = (liab / eq)         if liab is not None and eq and eq > 0 else None
        interest_cov = (opi / interest)   if opi and interest and interest > 0 else None

        return {
            "debt_ratio":        round(debt_ratio, 2) if debt_ratio is not None else None,
            "interest_coverage": round(interest_cov, 1) if interest_cov is not None else None,
        }

    # ── 점수화 ────────────────────────────────────────────────

    def _score_valuation(self, val: dict) -> float:
        """밸류에이션 점수 (0~100). 섹터 대비 저평가일수록 높음."""
        score = 50.0  # 기본값 (섹터 평균)

        vs = val.get("per_vs_sector")
        if vs is not None:
            # vs < 1: 섹터 대비 저평가 → 가산
            # vs > 1: 고평가 → 감산
            score += (1 - vs) * 30
        else:
            # 적자 기업 패널티
            if val.get("per_flag") == "deficit":
                score -= 20

        vs_pbr = val.get("pbr_vs_sector")
        if vs_pbr is not None:
            score += (1 - vs_pbr) * 10

        return round(max(0, min(100, score)), 1)

    def _score_growth(self, growth: dict) -> float:
        score = 50.0
        cagr = growth.get("revenue_cagr_2y")
        if cagr is not None:
            # 연 20% 성장 = 만점 가까이, 역성장 = 감점
            score += cagr * 100   # 20% CAGR → +20점
        yoy = growth.get("revenue_yoy")
        if yoy is not None:
            score += yoy * 50
        return round(max(0, min(100, score)), 1)

    def _score_profitability(self, prof: dict) -> float:
        score = 50.0
        opm = prof.get("operating_margin")
        if opm is not None:
            score += opm * 100    # 10% 영업이익률 → +10점
        roe = prof.get("roe")
        if roe is not None:
            score += roe * 50
        if not prof.get("is_profitable", True):
            score -= 20
        return round(max(0, min(100, score)), 1)

    def _score_stability(self, stab: dict) -> float:
        score = 70.0
        dr = stab.get("debt_ratio")
        if dr is not None:
            score -= min(30, dr * 10)  # 부채비율 3배 → -30점
        ic = stab.get("interest_coverage")
        if ic is not None and ic > 0:
            score += min(20, ic * 2)
        return round(max(0, min(100, score)), 1)

    @staticmethod
    def _grade(score: float) -> str:
        if score >= 75: return "A"
        elif score >= 60: return "B"
        elif score >= 45: return "C"
        else: return "D"

    @staticmethod
    def _cagr(
        end_val:   Optional[float],
        start_val: Optional[float],
        years:     int,
    ) -> Optional[float]:
        if not end_val or not start_val or start_val <= 0 or years <= 0:
            return None
        try:
            return (end_val / start_val) ** (1 / years) - 1
        except Exception:
            return None

    # ── 배치 처리 ─────────────────────────────────────────────

    def analyze_batch(self, ipo_list: list[dict]) -> pd.DataFrame:
        """
        여러 종목을 한 번에 분석해 DataFrame으로 반환.
        ipo_list: 각 원소가 {'corp_name', 'financials', 'offering_price',
                            'total_shares', 'sector'} 형태의 dict
        """
        results = []
        for item in ipo_list:
            try:
                r = self.analyze(
                    financials     = item["financials"],
                    offering_price = item["offering_price"],
                    total_shares   = item["total_shares"],
                    sector         = item.get("sector", "기타"),
                )
                r["corp_name"] = item.get("corp_name", "")
                results.append(r)
            except Exception as e:
                logger.warning("분석 실패 %s: %s", item.get("corp_name"), e)

        return pd.DataFrame(results)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    analyzer = FinancialAnalyzer()

    # 예시: IT 고성장 기업 vs 제조업 저성장 기업
    cases = [
        {
            "corp_name": "하이퍼테크 (IT 고성장)",
            "financials": {
                "revenue_t0": 50_000_000_000,   "revenue_t1": 35_000_000_000,  "revenue_t2": 22_000_000_000,
                "operating_income_t0": 7_000_000_000, "operating_income_t1": 4_000_000_000, "operating_income_t2": 2_000_000_000,
                "net_income_t0": 5_500_000_000,
                "total_assets": 80_000_000_000,  "total_liabilities": 20_000_000_000, "equity": 60_000_000_000,
                "eps": 2750,
            },
            "offering_price": 45_000,
            "total_shares": 2_000_000,
            "sector": "IT",
        },
        {
            "corp_name": "삼일제조 (제조 저성장 적자)",
            "financials": {
                "revenue_t0": 30_000_000_000,   "revenue_t1": 29_000_000_000, "revenue_t2": 28_500_000_000,
                "operating_income_t0": -500_000_000, "operating_income_t1": 200_000_000, "operating_income_t2": 300_000_000,
                "net_income_t0": -800_000_000,
                "total_assets": 40_000_000_000,  "total_liabilities": 25_000_000_000, "equity": 15_000_000_000,
                "eps": None,
            },
            "offering_price": 8_000,
            "total_shares": 5_000_000,
            "sector": "제조",
        },
    ]

    for c in cases:
        r = analyzer.analyze(c["financials"], c["offering_price"], c["total_shares"], c["sector"])
        print(f"\n── {c['corp_name']} ──")
        print(f"  PER:                {r.get('offering_per')}배  ({r.get('per_flag')})")
        print(f"  섹터대비 PER:        {r.get('per_vs_sector')}x")
        print(f"  매출 CAGR(2Y):       {(r.get('revenue_cagr_2y') or 0)*100:.1f}%")
        print(f"  영업이익률:          {(r.get('operating_margin') or 0)*100:.1f}%")
        print(f"  재무 매력도 점수:    {r.get('financial_attractiveness')} ({r.get('attractiveness_grade')}등급)")
