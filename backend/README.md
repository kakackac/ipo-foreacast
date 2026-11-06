# Django Service Backend

로컬 통합 개발용 Django 서비스다. 출시 승인/운영 배포 완료 상태가 아니다.
공식 인증/배포 설정은 [Django 인증 문서](https://docs.djangoproject.com/en/5.2/topics/auth/default/)와
[배포 체크리스트](https://docs.djangoproject.com/en/5.2/howto/deployment/checklist/)를 기준으로 했다.

## 실행

저장소 루트에서 실행한다. 운영 비밀번호나 API 키를 명령에 넣지 않는다.

```bash
python -m venv --system-site-packages .venv
.venv/bin/python -m pip install -r backend/requirements.txt
.venv/bin/python backend/manage.py migrate
.venv/bin/python backend/manage.py import_research
.venv/bin/python backend/manage.py import_event_snapshot
.venv/bin/python backend/manage.py test service_core
.venv/bin/python backend/manage.py runserver 127.0.0.1:8010 --noreload
```

`import_research`는 `current_research_data.json`의 해시/기존 입력을 검증한 뒤 연구 코호트의
역사적 이벤트만 가져온다. 같은 버전의 재실행은 행/알림을 중복 생성하지 않는다.
예정 공모주 레지스트리나 예측값을 임의로 승인해 가져오지 않는다.
로컬 DB와 세션 데이터는 gitignore된 `backend/var/`에만 저장한다.

`import_event_snapshot`은 별도 `current_event_data.json`의 동결 이벤트 버전을 검증한 뒤
신규 일반 IPO/스팩을 목록에 반영한다. 먼저 저장소 루트의 `integrate-event-snapshot`으로
버전을 생성해야 한다. 가격 대기는 `awaiting_price_collection`, 가격만 확인된 건은
`awaiting_disclosure_verification`이며 예측을 공개하지 않는다. 중복 이벤트는 재생성하지
않고 기존 연구/검토 상태를 임의로 낮추지 않는다. 2026-11-06 확인한 로컬 DB 412건과
연구 데이터 391행은 서로 다른 범위다.

## API

| 경로(`/api/v1/` 아래) | 동작 |
|---|---|
| `health/` | 프로세스 상태 및 예측/결제 미출시 상태 |
| `auth/csrf/` | 세션 API용 CSRF 토큰 |
| `auth/register/`, `auth/login/`, `auth/logout/` | 계정 생성/로그인/로그아웃 |
| `auth/password/` | 현재 비밀번호 확인 후 변경 |
| `me/` | 본인 정보/알림 선호 설정 |
| `offerings/` | 종목 목록, 검색 `q`, 페이지 `page`(50개씩) |
| `watchlist/`, `watchlist/<id>/` | 본인 관심종목 조회/추가/삭제 |
| `notifications/`, `notifications/<id>/read/` | 본인 내부 알림 조회/읽음 처리 |
| `subscription/` | 서버 검증 구독 상태 조회. 구매 요청은 503 차단 |
| `posts/`, `posts/<id>/` | 게시글 조회/작성/본인 수정/삭제 |
| `/admin/` | 운영자 계정·게시글 검토 등 Django 관리 화면 |

세션 쿠키와 CSRF 검사를 사용한다. 먼저 CSRF 토큰을 받아 변경 요청의 `X-CSRFToken` 헤더로
보낸다. 로그인 후 토큰은 회전하므로 다시 받는다. 로그인/가입도 CSRF 검사를 생략하지 않는다.
비밀번호는 Django 해시로 저장하며 최소 12자 및 기본 검증기를 적용한다. 클라이언트가 보낸
`is_staff`, 사용자 ID, 구독 활성화 플래그, 게시글 승인 플래그를 신뢰하지 않는다.

새 게시글과 수정한 게시글은 검토 대기다. 운영자가 관리 화면에서 공개한 글만 타인에게 보인다.
관리자 생성은 `createsuperuser` 대화식 명령을 사용하며 기본 관리자 비밀번호는 제공하지 않는다.
구독은 서버 검증 시각이 현재 이전이고 만료 전인 active 기록만 유효하다. 관리자 화면에서도
임의 구매 생성/활성화는 허용하지 않는다. App Store 서버 검증/알림 API는 아직 연결하지 않았다.

## 출시 전 필수 작업

- 모바일 세션/키체인 정책, 이메일 검증·비밀번호 복구·계정 삭제 및 개인정보 보존 정책.
- App Store 제품 설정, 서버 영수증/거래 서명 검증, 환불/갱신 알림, 재전송 방지 및 원장.
- 외부 푸시(현재는 DB 내부 알림만 구현), 상장일 21시 작업 및 실패 복구 운영 검증.
- 운영용 PostgreSQL/백업, 다중 프로세스 공용 rate limiter/분산 캐시, 모니터링/부하/보안 검토.
- 현재 인증 제한은 개발용 프로세스별 캐시다. 운영에서는 프록시/공용 캐시 기반 제한이 필요하다.
- 별도 승인된 모델을 기존 FastAPI 예측 계층과 연결. 현재 Django에는 예측 호출 기능이 없다.
- 운영 배포 설정으로 `check --deploy`를 실행하고 검토할 것. 개발 서버를 외부에 공개하지 않을 것.

관리 명령은 기본 개발 설정(SQLite)을 사용할 수 있다. 운영 WSGI는 개발 환경으로 명시하지
않는 한 비밀키와 DB 환경변수 없이는 시작하지 않는다. 운영은 `DJANGO_ENV=production`,
`DJANGO_SECRET_KEY`, `DJANGO_ALLOWED_HOSTS`, `POSTGRES_DB`, `POSTGRES_USER`,
`POSTGRES_PASSWORD`, `POSTGRES_HOST`(선택 `POSTGRES_PORT`)를 비밀 설정으로 주입한다.
키/비밀번호 값은 문서·앱·GitHub·로그에 넣지 않는다. 개발용 임시 비밀키는 프로세스마다
생성되므로 재시작하면 이전 세션이 유효하지 않을 수 있다.
