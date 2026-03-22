#!/bin/bash
# Polls every 5 min until exp5_optb_l finishes
while true; do
    if grep -q "Training complete" logs_exp5_l.txt 2>/dev/null; then
        echo "$(date): exp5_optb_l DONE"
        python3 -c "
import os; os.chdir('/lambda/nfs/team-shared/vision-nmai')
with open('runs/detect/exp5_optb_l/results.csv') as f:
    lines = f.readlines()
header = [h.strip() for h in lines[0].split(',')]
best = max(lines[1:], key=lambda l: float(l.split(',')[header.index('metrics/mAP50(B)')].strip() or 0))
vals = dict(zip(header, [v.strip() for v in best.split(',')]))
print(f'Best epoch={vals[\"epoch\"]}  mAP50={vals[\"metrics/mAP50(B)\"]}  mAP50-95={vals[\"metrics/mAP50-95(B)\"]}')
" 2>/dev/null
        break
    fi
    EPOCH=$(grep -o "[0-9]*/150" logs_exp5_l.txt 2>/dev/null | tail -1)
    echo "$(date): still training... $EPOCH"
    sleep 300
done
