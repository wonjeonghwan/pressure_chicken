# AI 제작진 시스템 — Antigravity 인수인계 문서

---

## 프로젝트 개요

**프로젝트명:** AI 제작진 (가칭)

**한 줄 정의:** 일반인이 카메라 앞에서 말하기 시작하면, AI 제작진이 실시간으로 붙어서 콘텐츠를 함께 만들어주는 시스템

**목적:** 유튜브를 시작하고 싶지만 뭘 찍어야 할지, 어떻게 말해야 할지 모르는 일반인에게 AI 제작진을 붙여준다. 제작진 없이도, 돈 없이도, 누구나 콘텐츠 크리에이터가 될 수 있게 한다.

**플랫폼:** 웹앱 (브라우저에서 링크 하나로 접속, 앱 설치 없음)

**기술 스택:**
- Frontend: React 또는 vanilla JS
- Backend: Python
- AI: Gemini 3 Flash / Gemini 3 Pro / Gemini Vision
- DB: Firebase Firestore
- 외부 API: pytrends, YouTube Data API v3
- 이미지 생성: Nano Banana Pro (Tier 3, 나중에)
- 에이전트 플랫폼: Google Antigravity

---

## 시스템 구조

```
[사용자: 브라우저에서 카메라/마이크 허용 후 시작]
                    ↓
            [Script Agent] ← 모든 것의 backbone
                    ↓
         Firebase Firestore (공유 상태 저장소)
                    ↓ (1초 폴링)
    ┌───────────────┼───────────────┐
[Director]     [Audience]       [Editor]
                    ↓
              [Publisher]
```

**에이전트 간 통신 방식:** 1초 폴링. Script Agent가 Firebase에 데이터 업데이트하면, 나머지 에이전트들이 1초마다 읽어서 반응. 단순하게 유지할 것. 복잡한 이벤트 기반 구조는 구현하지 않는다.

---

## Firebase 구조

```
firestore/
  sessions/
    {session_id}/
      metadata:
        channel_name: string
        created_at: timestamp
        episode_count: number
      episodes/
        {episode_id}/
          script: array of {timestamp, text, tag, emotion}
          director_logs: array of {timestamp, trigger, message}
          audience_analysis: object
          edit_timeline: array
          publisher_output: object
```

---

## Tier 1 — 최우선 구현 (데모 핵심)

**완성 기준:** Script Agent와 Director Agent가 실시간으로 돌아가고, 카메라 앞에서 말하다가 침묵하는 순간 Director가 화면에 개입 메시지를 띄운다.

---

### Script Agent

**역할**
- 실시간 음성 → 텍스트 변환
- 발화 내용 누적 기록
- 침묵, 말 더듬음, 반복 감지
- 편집점 태깅
- Firebase에 1초마다 업데이트

**사용 모델:** Gemini 3 Flash (속도 최우선)

**구현 순서**
1. 브라우저 카메라/마이크 권한 획득
2. Gemini Flash 음성인식 연결
3. 실시간 텍스트 변환 및 Firebase 저장
4. 침묵 감지 로직 (5초 이상 발화 없으면 silence_detected: true)
5. 말 더듬음/반복 감지 (같은 단어 3회 이상 반복)
6. Gemini Vision으로 표정 분석 추가

**Firebase 저장 형태**
```json
{
  "timestamp": "00:03:24",
  "text": "오늘은 파스타를 만들어볼게요",
  "tag": "highlight",
  "silence_detected": false,
  "repeated_word": false,
  "emotion": "긍정적"
}
```

**유념할 것**
- Flash 모델 써야 함. Pro 쓰면 딜레이 생겨서 실시간 느낌 안 남
- 침묵 감지는 5초 기준으로 잡을 것
- Firebase 업데이트 너무 자주 하면 비용 나올 수 있으니 1초 배치로 묶어서 저장

---

### Director Agent

**역할**
- Script Agent 데이터 1초마다 읽기
- Micro 개입: 침묵/반복/방향 상실 감지 후 한 줄 메시지 화면에 표시
- Macro 개입: 채널 회차 기반 전략 제안
- pytrends로 실시간 트렌드 연결해서 제안에 근거 추가

**사용 모델:** Gemini 3 Pro (채널 전체 컨텍스트 판단 필요)

**Micro 개입 트리거와 메시지**
```
silence_detected: true (5초)
→ "방금 말한 거 예시 하나 들어보세요"

repeated_word: true
→ "다른 각도로 얘기해볼까요"

emotion: 부정적 3회 연속
→ "카메라 보면서 편하게 말해보세요"

발화 주제 이탈 감지
→ "처음 얘기하던 [주제]로 돌아가볼까요"
```

**Macro 개입 로직**
```
episode_count == 1
→ "후킹이 필요해요. 첫 30초 안에
   '왜 이 채널을 봐야 하는지' 말해보세요"

episode_count == 2 or 3
→ 이전 에피소드 Script 분석 후
   "지난번에 [키워드] 얘기가 반응 좋을 것 같아요,
   이번엔 더 깊게 파보세요"

episode_count >= 4
→ "이제 고정 포맷을 만들 때예요.
   매 영상 시작을 [패턴]으로 해보세요"
```

**pytrends 연동**
```python
from pytrends.request import TrendReq
pytrends = TrendReq(hl='ko', tz=540)
# Script에서 추출한 키워드로 트렌드 조회
# 트렌드 급상승 키워드 있으면 Director 제안에 반영
# "지금 [키워드] 검색량 급상승 중이에요, 이 방향 어때요?"
```

**구현 순서**
1. Firebase에서 Script 데이터 1초 폴링
2. Micro 개입 트리거 로직 구현
3. Gemini Pro로 개입 메시지 생성
4. 화면에 조용히 한 줄 표시 (방해되지 않게)
5. pytrends 연동 추가
6. Macro 로직 추가 (episode_count 기반)

**유념할 것**
- 개입 메시지는 짧게. 한 줄, 최대 두 줄
- 너무 자주 개입하면 거슬림. 최소 30초 간격 유지
- pytrends는 한국어 키워드로 조회할 것 (hl='ko', tz=540)
- Pro 모델 쓰되, 채널 전체 히스토리를 컨텍스트로 넣을 것

---

**YouTube Data API 활용**
```python
# 카테고리별 인기 영상 조회
# 발화자 톤과 유사한 채널 패턴 분석
# 조회수, 좋아요 패턴으로 타겟 추론
```

**유념할 것**
- "25-34세 남성" 같은 단정적 표현 쓰지 말 것
- 항상 "~일 것 같아요", "~스타일이 잘 맞을 것 같아요" 로 표현
- 실제 YouTube 데이터를 근거로 붙일 것 (근거 없는 추측 금지)

---

### Editor Agent

**역할**
- Script 태깅 기반 편집 타임라인 자동 생성
- 롱폼(전체 맥락 유지본)과 숏폼(1분 이내 하이라이트/자극적 추출본) 타임라인 동시 기획
- 삭제 권장 구간 식별 (NG, 무음)
- 타임라인 기반 실제 영상 컷편집 (ffmpeg) 및 자막 병합

**사용 모델:** Gemini 2.5 Flash

**출력 형태**
```
편집 타임라인:
[롱폼 본편]
00:00-00:15 → 인트로 추천
00:15-01:23 → 본편 설명
...

[숏폼/쇼츠용]
00:45-00:55 → 핵심 꿀팁 하이라이트 (시청률 가장 높은 부분)
```

**유념할 것**
- Script의 tag(특히 Audience Agent의 분석 참고) 필드 기반으로 자동 생성
- 백엔드 ffmpeg와 직접 연동하여 롱폼/숏폼 2가지 인코딩 결과물 제공
- 말자막은 Script 텍스트 그대로 활용

---

## Tier 3 — 시간 남으면 구현

---

### Publisher Agent

**역할**
- 제목 후보 3개 생성
- 썸네일 이미지 3종 생성 (Nano Banana Pro)
- 본문/태그 자동 작성
- 최적 업로드 시간 제안

**사용 모델:** Gemini 3 Pro + Nano Banana Pro (Imagen)

**썸네일 생성 로직**
```
Script 핵심 키워드 추출
+ Audience 톤/스타일 분석
+ 유튜브 썸네일 가이드라인
  (큰 텍스트, 강한 표정, 단순한 배경, 대비 강한 색상)
→ Nano Banana Pro로 이미지 3종 생성
```

**YouTube Data API 활용**
```python
# 유사 영상 제목 패턴 분석
# 최적 업로드 시간 (카테고리별 조회수 높은 시간대)
```

**유념할 것**
- Nano Banana 크레딧 확인 필요
- Publisher는 녹화 종료 후 실행. 실시간 아님

---

## 데모 시나리오 (심사위원 앞에서)

1. 브라우저 열고 링크 접속
2. 카메라/마이크 허용
3. "안녕하세요, 오늘은 파스타 만드는 법을 알려드릴게요" 말하기 시작
4. 30초 후 의도적으로 침묵
5. Director가 "방금 파스타 얘기 나왔는데, 실패했던 경험 하나 추가해보세요" 표시
6. pytrends 기반 "지금 에어프라이어 검색량 급상승 중이에요" 추가 표시
7. 계속 말하면 Script 실시간 자막 쌓이는 거 보여주기
8. 녹화 종료 후 편집 타임라인 자동 생성 보여주기

---

## 주의사항

**반드시 지킬 것**
- Tier 1 완전히 완성 후 Tier 2로 넘어갈 것
- 에이전트 간 통신은 1초 폴링으로 단순하게 유지
- Director 개입 메시지는 30초 간격 이상 유지
- Gemini Flash vs Pro 구분해서 쓸 것 (Flash: 속도, Pro: 판단)
- pytrends 한국어 설정 필수 (hl='ko', tz=540)

**하지 말 것**
- 에이전트 5개 동시에 만들려고 하지 말 것
- 실제 영상 편집 기능 구현하지 말 것
- Audience Agent에서 근거 없는 단정적 표현 쓰지 말 것
- Nano Banana는 Tier 3. 지금 건드리지 말 것

---

## 최종 체크리스트

**Script Agent**
- [x] 브라우저에서 카메라/마이크 실행 (`getUserMedia video+audio`)
- [x] 실시간 음성 → 텍스트 변환 (Web Speech API `ko-KR`)
- [x] Firebase Firestore 저장 (`sessions/{uid}/script` 배열)
- [x] 로컬 영상 자동 저장 (`MediaRecorder` → `.webm` 다운로드)
- [x] `tag`, `silence_detected`, `repeated_word`, `emotion` 필드 저장
- [x] 말 더듬음/반복 단어 감지 (`Counter` 기반, 같은 단어 3회+)
- [x] Gemini Vision 표정/감정 분석 → `emotion` 필드 (긍정적/중립/긴장됨/부정적)

**Director Agent**
- [x] 침묵 10초 감지 후 개입 트리거 (`asyncio` background watcher)
- [x] 30초 이상 개입 간격 제한 (스팸 방지)
- [x] Gemini 1.5 Pro로 발화 내용 기반 맥락 피드백 생성
- [x] 카메라 화면 10초마다 캡처 → Gemini Vision 시각 분석 포함 피드백
- [x] **pytrends 연동** — 발화 키워드로 한국 실시간 검색 트렌드 조회 (hl='ko', tz=540)
- [x] 반복 단어 감지 트리거 처리
- [x] 화면 상단 팝업 UI (보라색 글로우, 10초 후 자동 소멸)
- [ ] Macro 개입 (에피소드 회차 기반 전략 제안) — Tier 2

**Tier 2 (Post-Recording Analytics & Basic Editing)**
- **Audience Agent**: 녹화 종료 후 `script`와 시각 데이터를 바탕으로 발화자의 톤과 성향 분석. YouTube Data API 및 pytrends 트렌드를 참고하여 타겟 시청자층 제안. (분석 결과 로컬 텍스트 저장)
- **Editor Agent**: Script Agent가 달아놓은 태그 기반으로 편집 타임라인을 두 가지 버전(본편 롱폼용 / 자극적인 숏폼용)으로 자동 텍스트 생성. 생성된 타임라인을 기반으로 원본 영상에서 롱폼과 숏폼 2가지 버전으로 영상을 각각 자르고(ffmpeg), 발화 내용을 화면 하단에 말자막으로 병합하여 다운로드 제공.`edited_long.webm` 과 `edited_short.webm` 두 버전 출력 (수정 및 테스트 예정)

**Tier 3 완료 기준 (Post-Production & Publisher)**
- [ ] **Post Production Agent**: 여러 에이전트 분석 기반 자막/효과 스타일링 적용
- [ ] 제목 후보 3개 생성
- [ ] 썸네일 3종 생성
- [ ] 업로드 최적 시간 제안

---

### 실제 구현에서 달라진 기술 결정 사항제안
```
