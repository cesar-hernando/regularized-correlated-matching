#!/bin/bash
# d = 7 failing sets of Fig. 4 by plain uniform sampling: weight 3 at all 18 points, then weight 4; resumable.
#   bash effective_distance_loss/uniform_d7/run.sh
cd "$(dirname "$0")/../.."
python effective_distance_loss/enumerate_faults.py --tasks effective_distance_loss/uniform_d7/tasks_d7.txt \
    --distances 7 --sampler uniform --samples 6.4e8 3.2e8 --n-workers 15 --out-dir effective_distance_loss/uniform_d7
