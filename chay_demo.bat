@echo off
chcp 65001 >nul
cd /d "%~dp0"
if not exist venv\Scripts\activate.bat (
  echo Dang tao moi truong Python lan dau...
  py -3.11 -m venv venv || py -3.12 -m venv venv || python -m venv venv
)
call venv\Scripts\activate.bat
if not exist venv\.da_cai (
  echo Dang cai thu vien lan dau, co the mat vai phut...
  pip install -r requirements.txt && echo ok> venv\.da_cai
)
start "" cmd /c "timeout /t 5 >nul & start http://127.0.0.1:5000"
python app.py
pause
