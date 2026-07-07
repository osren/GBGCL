#!/usr/bin/env bash
LOG=/home/ubuntu/workplace/tc2/GBGCL-main/logs/nohup/phaseR_$(date +%Y%m%d_%H%M%S).out
cd /home/ubuntu/workplace/tc2/GBGCL-main
nohup bash scripts/phases/run_phase_nohup.sh R > "$LOG" 2>&1 &
echo "PID=$!"
echo "LOG=$LOG"