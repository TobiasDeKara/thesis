#!/bin/bash
# expected time 20 min for p=5, time steps ~ 6k

jobID_1=$(sbatch batch_train_q.sh 12 | cut -f 4 -d' ')
jobID_2=$(sbatch --dependency=afterok:$jobID_1 batch_train_q.sh 13 | cut -f 4 -d' ')
sbatch  --dependency=afterok:$jobID_2 batch_train_q.sh 14

