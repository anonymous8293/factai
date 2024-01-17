@echo off
del results.csv

py -m experiments.hmm.reset -e main
py -m experiments.hmm.main --explainers extremal_mask --device cuda --fold 1

for /f "tokens=1-6 delims=/:. " %%a in ('echo %date% %time%') do (
    set current_time=%%c.%%b.%%a-%%d.%%e.%%f
)

copy results.csv .\experiments\hmm\reproducibility_results\hmm_%current_time%_extremal_results.csv