@echo off
chcp 65001 >nul
echo === 打包给 Mac/同事使用 ===
echo  1) 含当前知识库 + 含 api_key（自用备份）
echo  2) 含当前知识库 + 不含 api_key（发给同事，他们直接能用）
echo  3) 不含知识库   + 不含 api_key（干净壳，让对方上传自己的 Word）
echo.
set /p choice=请选择 [1/2/3，默认 2]:
if "%choice%"=="" set choice=2

if "%choice%"=="1" (
  py -3 "%~dp0tools\package.py"
) else if "%choice%"=="2" (
  py -3 "%~dp0tools\package.py" --no-key
) else if "%choice%"=="3" (
  py -3 "%~dp0tools\package.py" --clean --no-key
) else (
  echo 输入错误
)
echo.
echo zip 在 dist\ 目录里。
pause
