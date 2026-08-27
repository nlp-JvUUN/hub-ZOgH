@echo off
chcp 65001 >nul
set MODEL=Qwen/Qwen2.5-0.5B-Instruct
python src/evaluate.py --model %MODEL% --output outputs/baseline_new.json
if errorlevel 1 exit /b 1
python src/train_grpo.py --model %MODEL% --output-dir outputs/grpo_model
if errorlevel 1 exit /b 1
python src/evaluate.py --model outputs/grpo_model --output outputs/grpo_new.json
if errorlevel 1 exit /b 1
python src/analyze_results.py
pause
