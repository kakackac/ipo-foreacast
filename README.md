# IPO 상장일 수익률 예측 시스템

공모주가 상장하기 전에 공모가를 기준으로 **상장일 시초가 수익률**과 **상장일 종가 수익률**을 각각 예측하는 ML 파이프라인입니다.

서비스 범위, 데이터, 모델, 21:00 KST 실측 확정, 출시 로드맵은 [프로젝트 계획서](docs/IPO_상장일_수익률_예측_서비스_계획서.docx)에 정리했습니다.

- 시초가 수익률: `(9시 시초가 / 확정 공모가 - 1) × 100`
- 종가 수익률: `(상장일 종가 / 확정 공모가 - 1) × 100`

---

## 현재 구현 상태

2026-11-06 기준: 데이터·모델·앱의 출시 완료 상태는 아닙니다. 최신 진행 및 미완료 조건은
[남은 작업](docs/출시까지_남은작업.md), 로컬 서비스 실행은 [Django 백엔드](backend/README.md)를 참고하세요.
개인청약 경쟁률/민간 사이트/코넥스는 현재 수집·모델 대상에서 제외합니다.
역사적 연구 데이터와 출시 승인 모델을 구분하며, 예측·실결제는 아직 공개하지 않습니다.
최신 연결 버전은 `f282c5f501138e21c65e`이며 연구 391행/개발 364행입니다.
재무 632건 중 승인 계정 보유 520건, 잔여 112건입니다. 기존 모델은 새 데이터로 재학습하지 않았습니다.
별도 이벤트 통합 버전 `67e6295fef1af532984d`는 11월 4일 스냅샷을 반영한 1,310건입니다. 해당 스냅샷 62건의 가격 처리는 완료(가격 확인 59/비대상 3)됐으며 가격 대기는 0건입니다.
로컬 서비스 DB는 412건이지만 연구 학습 데이터 391행과는 다른 범위입니다. 신규 이벤트의 예측은 공개하지 않습니다.
11월 6일 KIND 재조회 결과 같은 62건이며 비교한 코드·시장·날짜·공모 조건에 차이가 없었습니다.
신규 일반 IPO/스팩 21건의 DART 공시 버전은 `14ea95fe06be30ff4c7e`입니다. 기업 식별·최종 공모가·기관 경쟁률 21건, 통합 확약 13건, 희망 밴드 16건, 유통가능비율 13건을 연결했습니다.
모델 입력 연결은 `collect-event-context`로 별도 증분 실행합니다. 저장 지수가 9월 9일까지여서 일반 IPO 16건은 시장 지수 보완 대기이고 스팩 5건은 분리했습니다. 기존 연구/모델 승인은 변경하지 않습니다.

| 모듈 | 파일 | 상태 |
|------|------|------|
| 설정·피처 정의 | `ipo_predictor/config.py`, `ipo_predictor/features/definitions.py` | 구현됨. 수집·모델 승인과 별개 |
| DART 수집기 | `ipo_predictor/data/collectors/dart_collector.py` | 인증 API 및 공개 원문 경로. 일부 형식/공시 검증 미완료 |
| KRX 수집기 | `ipo_predictor/data/collectors/krx_collector.py` | KIND 공식 이벤트, OpenAPI 가격·지수. 신규 가격 갱신에는 키 필요 |
| 피처 엔지니어링 | `ipo_predictor/data/processors/feature_engineer.py` | 구현됨. 결측/출처/공개 시점 확인 필요 |
| 재무 복구 | `ipo_predictor/data/pipelines/disclosure_financials.py` | 실제 전체 표본 처리, 일부 미지원/검토 상태 유지 |
| 연구 데이터 버전 | `ipo_predictor/data/pipelines/data_delivery.py` | 해시·중복·시점 검사 후 연결. 학습/배포 승인 아님 |
| 예측 모델 | `ipo_predictor/models/training/listing_eve.py` | 시초가/종가 Ridge 후보 실제 학습·저장 완료, 시간순 개발 검증 및 출시 미승인 |
| 백테스터·오차 분석 | `ipo_predictor/models/evaluation/` | 구현됨. 과거 평가 소진 및 전향 검증 조건 확인 필요 |
| 예측 API | `ipo_predictor/api/server.py` | 승인 모델이 없으면 차단. 서비스 전체 백엔드와 별개 |
| Django 서비스 | `backend/` | 계정·관심종목·내부 알림·검토형 게시판·관리자 기본 API, 실결제/외부 푸시/운영 배포 미완료 |
| 파이프라인 진입점 | `ipo_predictor/pipeline.py` | 수집/연구 저장 연결, 명령 완료와 데이터 품질 완성은 구분 |

---

## 피처 구성 (Phase 1 CORE)

| 피처 | 설명 | 출처 |
|------|------|------|
| `institutional_demand_ratio` | 통합 기관 수요예측 경쟁률 | DART 최종 `[발행조건확정]` 원문 표 (1차), 대표주관사 공식 통합 결과 공지 (보조) |
| `lockup_commitment_ratio` | 기관 의무보유확약 통합 비율 | DART 최종 `[발행조건확정]` 원문 (1차), 대표주관사 공식 통합 결과 공지 (보조) |
| `offering_price_band_position` | 공모가 밴드 위치 (0~1+) | DART |
| `band_exceeded` | 밴드 상단 초과 여부 | 파생 |
| `kospi_momentum_5d` | KOSPI 5일 수익률 | KRX |
| `kospi_momentum_20d` | KOSPI 20일 수익률 | KRX |
| `recent_ipo_avg_return_sector` | 섹터 최근 IPO 평균 수익률 | 내부 DB |
| `recent_ipo_avg_return_all` | 전체 최근 IPO 평균 수익률 | 내부 DB |

---

## 예측·실측 운영 흐름

1. IPO 상장 일정을 확인한다.
2. 상장 전 DART 신고서에서 공모가 밴드, 확정 공모가, 신주·구주매출, 직접 공시된 유통물량, 최대주주 보호예수, 재무제표를 수집한다. 기관 수요예측 경쟁률과 의무보유확약 통합 비율은 같은 최종 `[발행조건확정]` 문서에서 승인한다. 통합 확약값이 직접 있으면 우선 사용하고, 없을 때만 같은 문서의 6/3/1개월·15일 표가 완전할 경우 합산한다. DART 원문에 해당 묶음이 없을 때만 대표주관사의 공식 통합 결과 공지를 보조 원천으로 검증한다.
   개인청약 경쟁률은 수집·모델·서비스 입력 범위에서 제외한다.
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

# 2. KRX OpenAPI 키 설정 (웹사이트 ID/비밀번호는 사용하지 않음)
export KRX_API_KEY=your_krx_openapi_key

# 3. OpenDART와 KRX OpenAPI 실제 이력 정합
cd ipo_predictor
python pipeline.py --mode collect --start-year 2020 --end-year 2025 --phase phase2

# 4. 실제 학습 실행
python pipeline.py --mode train --phase phase2
```

KRX 수집은 승인된 일별 API에 `AUTH_KEY` 헤더를 사용합니다. 개인 웹사이트 로그인 ID·비밀번호를 코드나 배포 환경에 넣지 않습니다. 실제 학습은 수집기가 포맷을 맞춘 과거 데이터 파일을 생성한 후에만 시작하며, DART 원문·수요예측 자료와 KRX 상장 이력을 정합한 결과의 품질 요약을 먼저 확인합니다.

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
    "lockup_commitment_ratio": 0.42,
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
- [x] 실제 DART 원문 표본 검증 (재무 9개 회사, 공모·기관 등 10개 문서)
- [ ] 잔여 형식·단위·회계기준 지원과 미수집 공식 원천 연결
- [x] KRX OpenAPI 기반 IPO 일정·상장일 시세·시장 지수 실제 이력 수집 연결
- [x] 기존 연구 코호트의 공시/가격/시점 연결 및 데이터 버전 저장
- [ ] 최신 이벤트의 인증 가격·공시 정합 및 pre_demand/post_demand 당시 스냅샷 준비
- [ ] 상장일 21:00 KST 결과 수집 스케줄러 운영 설치·실제 발화 검증
- [x] 기존 동결 연구 모델의 시간순 개발 보고서 (출시 성능 검증 아님)
- [ ] 신규 상장 전 예측 기록과 공식 결과를 이용한 전향 검증
- [ ] React Native 앱 연동 (API 서버 → 모바일 앱)
