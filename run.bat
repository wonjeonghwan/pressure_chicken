@echo off
REM ─────────────────────────────────────────────────────────────────
REM  압력밥솥 타이머 — 매장 운영자용 실행 스크립트
REM  더블클릭으로 실행하거나 바탕화면에 바로가기를 만들어두세요.
REM
REM  실행 흐름:
REM    1) 이 파일이 있는 폴더로 이동
REM    2) uv 설치 확인 (없으면 안내 메시지 후 종료)
REM    3) 의존성 자동 동기화 (uv sync — 첫 실행 시 시간 걸림)
REM    4) main.py 실행
REM    5) 종료 시 콘솔 창 유지 (오류 확인용)
REM ─────────────────────────────────────────────────────────────────

cd /d "%~dp0"

where uv >nul 2>nul
if errorlevel 1 (
    echo.
    echo [!] uv 가 설치되어 있지 않습니다.
    echo     PowerShell에서 다음을 1회 실행 후 다시 시도하세요:
    echo.
    echo     winget install astral-sh.uv
    echo.
    pause
    exit /b 1
)

if not exist ".venv\" (
    echo [setup] 첫 실행 — 의존성 설치 중. 몇 분 걸릴 수 있습니다 ...
    uv sync
    if errorlevel 1 (
        echo [!] 의존성 설치 실패. 인터넷 연결을 확인하고 다시 시도하세요.
        pause
        exit /b 1
    )
)

REM 기본 config로 실행. 다른 config를 쓰고 싶으면 인자로 넘기세요.
REM   예: run.bat --config config/examples/store_4cam.json
if "%~1"=="" (
    uv run python main.py
) else (
    uv run python main.py %*
)

REM 종료 후 콘솔 유지 — 오류 확인 가능. 자동 닫기를 원하면 아래 한 줄을 지우세요.
pause
