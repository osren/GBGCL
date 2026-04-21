#!/bin/bash

LOGFILE="sweepX_$(date +%Y%m%d_%H%M%S).log"
ERRFILE="sweepX_errors_$(date +%Y%m%d_%H%M%S).log"

echo "Log: $LOGFILE"
echo "Errors: $ERRFILE"
echo "Workers: $SWEEP_WORKERS"
echo "Stage: $SWEEP_STAGE"

nohup python tools/sweepX.py > "$LOGFILE" 2>&1 &
echo "PID: $!"
