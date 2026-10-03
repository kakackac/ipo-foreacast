# IPO 상장일 수익률 예측 시스템

공모주가 상장하기 전에 공모가를 기준으로 **상장일 시초가 수익률**과 **상장일 종가 수익률**을 각각 예측하는 ML 파이프라인입니다.

서비스 범위, 데이터, 모델, 21:00 KST 실측 확정, 출시 로드맵은 [프로젝트 계획서](../docs/IPO_상장일_수익률_예측_서비스_계획서.docx)에 정리했습니다.

- 시초가 수익률: `(9시 시초가 / 확정 공모가 - 1) × 100`
- 종가 수익률: `(상장일 종가 / 확정 공모가 - 1) × 100`

---

## 현재 구현 상태

| 모듈 | 파일 | 상태 |
|------|------|------|
| 설정 | `config.py` | ✅ 완료 |
| 피처 정의 | `features/definitions.py` | ✅ 완료 |
| DART 수집기 | `data/collectors/dart_collector.py` | ✅ 원문 ZIP 다운로드·파싱 (API 키 필요) |
| KRX 수집기 | `data/collectors/krx_collector.py` | ✅ 공식 OpenAPI로 상장 일정·상장일 가격·일별 지수 수집 |
| 실제 이력 정합 | `data/pipelines/historical_ipo_pipeline.py` | ✅ DART·KRX 정합 → 학습 피처 생성 |
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

# 1-1. KRX OpenAPI 인증키 설정
# 개인 웹사이트 ID/비밀번호는 사용하지 않는다. 키 값은 터미널과 서버의 비밀 환경변수에만 둔다.
export KRX_API_KEY=your_krx_openapi_key

# 2. OpenDART 증권신고서 원문, 재무제표와 KRX 상장일 가격·지수를 수집하고 정합
# 2015년부터 실행일(오늘은 2026-08-25)까지 수집한다. 미래 날짜는 자동 제외된다.
python pipeline.py --mode collect --start-year 2015 --end-year 2026 --phase phase2

# 3. 생성된 실제 데이터로 시계열 백테스트와 두 모델 학습
python pipeline.py --mode train --phase phase2
```

수집 결과는 `data/raw/`에 원본별로, 학습용 결과는 `data/processed/features_all.parquet`에 저장됩니다. KRX 수집기는 승인된 KOSPI·KOSDAQ 일별 종목/지수 API를 `AUTH_KEY` 헤더로 호출하며, 웹사이트 로그인 자격증명은 사용하지 않습니다. `data_collection_summary.json`에는 일정·공시·가격·타깃의 행 수와 핵심 피처 충족 현황이 남습니다. API 키가 없거나 원문 파싱에 실패한 종목은 0으로 채우지 않고 결측으로 보존하므로, 이 파일로 데이터 품질을 먼저 확인한 뒤 학습합니다. 처음에는 `2015년부터 실행일`까지 전체를 한 번 수집하고, 이후에는 새 상장분만 해당 연도 범위로 갱신합니다.

### 확정 공모가 감사 절차

공모가를 단순 범위 규칙으로 삭제하지 않습니다. 수집 후 다음 파일을 확인합니다.

- `data/raw/dart_offering_price_audit.parquet`: 회사명, 신고서·수요예측 접수번호, 최신 정정 신고서 선택 여부, 원문 추출 금액·주변 문구, OpenDART 구조화 모집가액, 희망 공모가 밴드 및 대조 결과
- `data/raw/dart_offering_price_review_queue.parquet`: 화폐 단위가 없거나 값이 없거나 DART 원문 간 금액이 일치하지 않아 사람이 확인해야 하는 행

동일 날짜에 원본과 정정 신고서가 함께 있으면 최신 정정 신고서를 우선합니다. `verified_currency_unit`은 확정 공모가 문맥에서 `원` 또는 `KRW` 단위까지 확인된 값입니다. 원문 추출값이 부족할 때는 동일 접수번호의 OpenDART 지분증권 구조화 모집가액을 대조하고, `[발행조건확정]` 신고서일 때만 `verified_structured_api`로 학습에 포함합니다. 두 값이 다르면 `needs_review_structured_mismatch`로 격리합니다. `needs_review_*`와 `missing`은 삭제되지 않고 감사 로그에 남지만, 수동 확인 전에는 학습·백테스트에서 격리됩니다. 희망 밴드 밖의 공모가나 100원 미만 값은 **경고**일 뿐 자동 제외 사유가 아닙니다.

### 재실행 캐시

첫 전체 수집 뒤에는 `data/raw/`의 KRX 상장 캘린더·상장일 가격·지수 이력과 DART 접수번호별 원문 결과를 재사용합니다. 완료된 과거 연도와 이미 시가·종가가 있는 종목은 다시 호출하지 않으며, 마지막 연도는 새 상장을 반영하도록 갱신합니다. DART 목록은 정정 신고서를 찾기 위해 다시 확인하되, 같은 접수번호의 원문·재무제표와 원문이 존재하지 않는 `014` 접수번호는 재호출하지 않습니다. 검증 규칙이 추가된 버전으로 처음 실행할 때만 기존 DART 행을 한 번 보강합니다.

원문을 확인해 값을 승인할 때는 `data/manual/offering_price_overrides.example.csv`를 참고해 `data/manual/offering_price_overrides.csv`를 만들고, `decision`을 `verified`로 기록합니다. 다음 `collect` 실행에서 해당 값은 `manual_verified` 상태로 학습에 포함됩니다. 실제 검토 파일은 Git에 포함되지 않습니다.

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
- [ ] 실제 DART 원문 샘플 기반 Phase 2 파서 보강 및 예외 패턴 추가
- [x] IPO 일정·증권신고서·수요예측·상장일 시세 데이터의 실제 이력 수집·정합
- [ ] 상장일 21:00 KST 실습 결과 수집·NXT 체결대상 확인 스케줄러 연결
- [ ] 실제 원본 데이터 기반 시계열 성능 보고서 생성
- [ ] XGBoost / LightGBM 환경에서 재테스트 (현재는 sklearn GBM)
- [ ] React Native 앱 연동 (API 서버 → 모바일 앱)
