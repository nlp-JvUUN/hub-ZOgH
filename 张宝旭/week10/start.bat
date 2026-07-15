@echo off
chcp 65001 >nul

rem 检查 python-docx
py -3 -c "import docx" 2>nul
if errorlevel 1 (
  echo 正在安装依赖 python-docx ...
  py -3 -m pip install python-docx
)

py -3 "%~dp0start.py"
pause
