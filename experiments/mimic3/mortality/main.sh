#!/bin/bash

processes=${processes:-5}
device=${device:-cpu}
seed=${seed:-42}
outputfile=${outputfile:-mimic_results_per_fold.csv}
preservation=${preservation:-true}
minfold=${minfold:-0}
maxfold=${maxfold:-4}
explainers=${explainers:-all}

while [ $# -gt 0 ]
do
  if [[ $1 == *"--"* ]]
  then
    param="${1/--/}"
    declare $param="$2"
  fi
  shift
done

trap ctrl_c INT

function ctrl_c() {
    echo " Stopping running processes..."
    kill -- -$$
}

for fold in $(seq $minfold $maxfold)
do
  if [[ $preservation = true ]]; then 
    if [[ $explainers = all ]]; then
      python -m experiments.mimic3.mortality.main --device "$device" --fold "$fold" --seed "$seed" --deterministic --output-file "$outputfile" &
    else
      python -m experiments.mimic3.mortality.main --device "$device" --fold "$fold" --seed "$seed" --deterministic --output-file "$outputfile" --explainers "$explainers" &
    fi
  else 
    python -m experiments.mimic3.mortality.main --device "$device" --fold "$fold" --seed "$seed" --deletion-mode --explainers extremal_mask --output-file "$outputfile"&
  fi

  # Support lower versions
  if ((BASH_VERSINFO[0] >= 4)) && ((BASH_VERSINFO[1] >= 3))
  then
    # allow to execute up to $processes jobs in parallel
    if [[ $(jobs -r -p | wc -l) -ge $processes ]]
    then
      # now there are $processes jobs already running, so wait here for any job
      # to be finished so there is a place to start next one.
      wait -n
    fi
  else
    # allow to execute up to $processes jobs in parallel
    while [[ $(jobs -r -p | wc -l) -ge $processes ]]
    do
      # now there are $processes jobs already running, so wait here for any job
      # to be finished so there is a place to start next one.
      sleep 1
    done
  fi

done

wait