# IPO 상장일 수익률 예측 시스템

공모주가 상장하기 전에 공모가를 기준으로 **상장일 시초가 수익률**과 **상장일 종가 수익률**을 각각 예측하는 ML 파이프라인입니다.

- 시초가 수익률: `(9시 시초가 / 확정 공모가 - 1) × 100`
- 종가 수익률: `(상장일 종가 / 확정 공모가 - 1) × 100`

---

## 현재 구현 상태

| 모듈 | 파일 | 상태 |
|------|------|------|
| 설정 | `config.py` | ✅ 완료 |
| 피처 정의 | `features/definitions.py` | ✅ 완료 |
| DART 수집기 | `data/collectors/dart_collector.py` | ✅ 완료 (API 키 필요) |
| KRX 수집기 | `data/collectors/krx_collector.py` | ✅ 완료 |
| 피처 엔지니어링 | `data/processors/feature_engineer.py` | ✅ 완료 |
| Phase 2 피처 연결 | `data/processors/feature_engineer.py` | ✅ 완료 (데모/학습 연동) |
| Phase 2 DART 파서 | `data/collectors/dart_collector.py` | ✅ 부분 완료 (정규식 기반) |
| 재무 분석기 | `data/processors/financial_analyzer.py` | ✅ 완료 |
| 예측 모델 | `models/baseline/gradient_boost_model.py` | ✅ 완료 (Open/Close 분리 모델) |
| 백테스터 | `models/evaluation/backtester.py` | ✅ 완료 |
| 아웃라이어 분석기 | `models/evaluation/outlier_analyzer.py` | ✅ 완료 |
| API 서버 | `api/server.py` | ✅ 완료 (uvicorn 설치 필요) |
| 파이프라인 진입점 | `pipeline.py` | ✅ 완료 |

---

## 피처 구성 (Phase 1 CORE)

| 피처 | 설명 | 출처 |
|------|------|------|
| `institutional_demand_ratio` | 기관 수요예측 경쟁률 | DART |
| `retail_subscription_ratio` | 개인 청약 경쟁률 | 증권사 |
| `lockup_6m_ratio` | 6개월 의무확약 비율 | DART |
| `lockup_3m_ratio` | 3개월 의무확약 비율 | DART |
| `lockup_1m_ratio` | 1개월 의무확약 비율 | DART |
| `lockup_15d_ratio` | 15일 의무확약 비율 | DART |
| `lockup_weighted_score` | 확약기간 가중 점수 (파생) | 계산 |
| `offering_price_band_position` | 공모가 밴드 위치 (0~1+) | DART |
| `band_exceeded` | 밴드 상단 초과 여부 | 파생 |
| `kospi_momentum_5d` | KOSPI 5일 수익률 | KRX |
| `kospi_momentum_20d` | KOSPI 20일 수익률 | KRX |
| `recent_ipo_avg_return_sector` | 섹터 최근 IPO 평균 수익률 | 내부 DB |
| `recent_ipo_avg_return_all` | 전체 최근 IPO 평균 수익률 | 내부 DB |

---

## 예측·실측 운영 흐름

1. IPO 상장 일정을 확인한다.
2. 상장 전 증권신고서·수요예측 결과에서 공모가 밴드, 확정 공모가, 신주·구주매출, 유통물량, 기관 수요예측 경쟁률, 의무보유확약, 최대주주 보호예수, 재무제표, 시장·섹터 지표를 수집한다.
3. 상장 전 시초가·종가 수익률을 별도 모델로 예측한다.
4. 상장일 21:00 KST에 시초가와 KRX 종가를 기록하고, 예측과 실제 결과를 비교한다. NXT 애프터마켓 체결 대상 여부도 함께 확인한다.
5. 신규 실측치 10건 축적 또는 월 1회 시점에만 재학습한다. 시계열 검증에서 개선된 모델만 배포한다.

## 데모 실행 (실제 API 없이)

```bash
cd ipo_predictor
python pipeline.py --mode demo

# Phase 2 피처까지 포함
python pipeline.py --mode demo --phase phase2
```

데모는 시뮬레이션 데이터 600건으로 시초가·종가 모델의 분리 학습, Walk-forward 검증, 모델 저장, 예측 리포트 저장을 검증합니다. 출력되는 MAE와 방향 정확도는 **실제 IPO 성능이 아닌 파이프라인 연동 검증 수치**입니다. 실제 성능은 과거 IPO 원본 데이터로 시계열 백테스트를 완료한 후에만 공개합니다.

---

## 실제 데이터 수집 순서

```bash
# 1. DART API 키 설정
export DART_API_KEY=your_key_here   # https://opendart.fss.or.kr

# 2. 히스토리 수집 (2015~2024)
python -c "
from data.collectors.dart_collector import DARTCollector
DARTCollector().collect_full_history(2015, 2024)
"

# 3. KRX 시장 데이터 수집
python -c "
from data.collectors.krx_collector import KRXCollector
KRXCollector().collect_market_data(2015, 2024)
"

# 4. 실제 학습 실행
python pipeline.py --mode train
```

`특징은 수집기가 포맷을 맞춘 과거 데이터 파일을 생성한 후에만 실제 학습을 시작합니다. 현재 DART 원문·수요예측 자료를 실제 상장 이력과 정확히 연결하는 수집 파이프라인은 제작 중입니다.

---

## API 서버 실행

```bash
pip install fastapi uvicorn
uvicorn api.server:app --host 0.0.0.0 --port 8000

# 예측 요청 예시
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "corp_name": "테스트AI",
    "institutional_demand_ratio": 1250,
    "lockup_6m_ratio": 0.42,
    "lockup_3m_ratio": 0.18,
    "lockup_1m_ratio": 0.10,
    "lockup_15d_ratio": 0.05,
    "offering_price_band_position": 1.1,
    "kospi_momentum_5d": 0.012,
    "kospi_momentum_20d": 0.035,
    "recent_ipo_avg_return_sector": 18.5,
    "recent_ipo_avg_return_all": 14.2
  }'
```

---

## 에러 주도 개발 루프

```
python pipeline.py --mode demo
    ↓
reports/outlier_report_*.txt 확인
    ↓
원인 A (시장 국면) 비율 > 30% ?
    → config.py에서 OPTIONAL 피처 활성화
    → kospi_momentum_60d, bear_market_flag 추가
    → pipeline.py --mode train 재실행
    ↓
성능 개선 확인 → 반복
```

---

## 다음 단계

- [x] `float_share_ratio` — 유통물량 비율 계산 연결
- [x] `secondary_offering_ratio` — 구주매출 비율 파싱/계산 연결
- [x] `major_shareholder_lockup_months` — 최대주주 의무보유기간 파싱
- [x] `same_day_ipo_count` — 동일일 상장 종목 수 자동 계산
- [x] `risk_factor_count` — 투자설명서 위험요소 항목 수 파싱
- [x] `offering_per` + `per_vs_sector_median` — 밸류에이션 피처
- [x] `revenue_growth_3y`, `operating_margin`, `debt_ratio` — 재무 피처 계산 연결
- [x] 시초가·종가 수익률 분리 예측/백테스트/API 지원
- [ ] 실제 DART 원문 샘플 기반 Phase 2 파서 보강
- [ ] IPO 일정·증권신고서·수요예측·상장일 시세 데이터의 실제 이력 수집·정합
- [ ] 상장일 21:00 KST 실습 결과 수집·NXT 체결대상 확인 스케줄러 연결
- [ ] 실제 원본 데이터 기반 시계열 성능 보고서 생성
- [ ] XGBoost / LightGBM 환경에서 재테스트 (현재는 sklearn GBM)
- [ ] React Native 앱 연동 (API 서버 → 모바일 앱)
