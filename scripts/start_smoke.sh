#!/usr/bin/env bash
LOG=/home/ubuntu/workplace/tc2/GBGCL-main/logs/nohup/btcm_smoke4.out
nohup bash /home/ubuntu/workplace/tc2/GBGCL-main/scripts/smoke_btcm.sh > "$LOG" 2>&1 &
echo "PID=$!"
echo "LOG=$LOG"