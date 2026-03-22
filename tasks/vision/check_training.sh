#!/bin/bash
echo "=== $(date) ==="
echo "GPUs:"
nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv,noheader
echo ""
echo "Model A (exp5_optb_x) - epoch:"
grep -o "[0-9]*/150" logs_exp5_x.txt 2>/dev/null | tail -1
echo "Model A best mAP50:"
grep "metrics/mAP50(B)" logs_exp5_x.txt 2>/dev/null | tail -1

echo ""
echo "Model B (exp5_optb_l) - epoch:"
grep -o "[0-9]*/150" logs_exp5_l.txt 2>/dev/null | tail -1
echo "Model B best mAP50:"
grep "metrics/mAP50(B)" logs_exp5_l.txt 2>/dev/null | tail -1
echo "=========================="
