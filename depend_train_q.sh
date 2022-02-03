#!/bin/bash
# expected time 2-20 min for p=5, time steps ~ 6k

jobID_1=$(sbatch batch_train_q.sh 26 | cut -f 4 -d' ')

#jobID_2=$(sbatch --dependency=afterok:$jobID_1 batch_train_q.sh 16 | cut -f 4 -d' ')
#jobID_17=$(sbatch --dependency=afterok:$jobID_2 batch_train_q.sh 17 | cut -f 4 -d' ')
#jobID_18=$(sbatch --dependency=afterok:$jobID_17 batch_train_q.sh 18 | cut -f 4 -d' ')
#jobID_19=$(sbatch --dependency=afterok:$jobID_18 batch_train_q.sh 19 | cut -f 4 -d' ')
#jobID_20=$(sbatch --dependency=afterok:$jobID_19 batch_train_q.sh 20 | cut -f 4 -d' ')
#jobID_21=$(sbatch --dependency=afterok:$jobID_20 batch_train_q.sh 21 | cut -f 4 -d' ')
#jobID_22=$(sbatch --dependency=afterok:$jobID_21 batch_train_q.sh 22 | cut -f 4 -d' ')
#jobID_23=$(sbatch --dependency=afterok:$jobID_22 batch_train_q.sh 23 | cut -f 4 -d' ')
#jobID_24=$(sbatch --dependency=afterok:$jobID_23 batch_train_q.sh 24 | cut -f 4 -d' ')
#jobID_25=$(sbatch --dependency=afterok:$jobID_24 batch_train_q.sh 25 | cut -f 4 -d' ')
#jobID_26=$(sbatch --dependency=afterok:$jobID_25 batch_train_q.sh 26 | cut -f 4 -d' ')

jobID_27=$(sbatch --dependency=afterok:$jobID_1 batch_train_q.sh 27 | cut -f 4 -d' ')
jobID_28=$(sbatch --dependency=afterok:$jobID_27 batch_train_q.sh 28 | cut -f 4 -d' ')
jobID_29=$(sbatch --dependency=afterok:$jobID_28 batch_train_q.sh 29 | cut -f 4 -d' ')
jobID_30=$(sbatch --dependency=afterok:$jobID_29 batch_train_q.sh 30 | cut -f 4 -d' ')
jobID_31=$(sbatch --dependency=afterok:$jobID_30 batch_train_q.sh 31 | cut -f 4 -d' ')
jobID_32=$(sbatch --dependency=afterok:$jobID_31 batch_train_q.sh 32 | cut -f 4 -d' ')
jobID_33=$(sbatch --dependency=afterok:$jobID_32 batch_train_q.sh 33 | cut -f 4 -d' ')
jobID_34=$(sbatch --dependency=afterok:$jobID_33 batch_train_q.sh 34 | cut -f 4 -d' ')
jobID_35=$(sbatch --dependency=afterok:$jobID_34 batch_train_q.sh 35 | cut -f 4 -d' ')
jobID_36=$(sbatch --dependency=afterok:$jobID_35 batch_train_q.sh 36 | cut -f 4 -d' ')
jobID_37=$(sbatch --dependency=afterok:$jobID_36 batch_train_q.sh 37 | cut -f 4 -d' ')
jobID_38=$(sbatch --dependency=afterok:$jobID_37 batch_train_q.sh 38 | cut -f 4 -d' ')
jobID_39=$(sbatch --dependency=afterok:$jobID_38 batch_train_q.sh 39 | cut -f 4 -d' ')

jobID_30=$(sbatch --dependency=afterok:$jobID_29 batch_train_q.sh 30 | cut -f 4 -d' ')
jobID_31=$(sbatch --dependency=afterok:$jobID_30 batch_train_q.sh 31 | cut -f 4 -d' ')
jobID_32=$(sbatch --dependency=afterok:$jobID_31 batch_train_q.sh 32 | cut -f 4 -d' ')
jobID_33=$(sbatch --dependency=afterok:$jobID_32 batch_train_q.sh 33 | cut -f 4 -d' ')
jobID_34=$(sbatch --dependency=afterok:$jobID_33 batch_train_q.sh 34 | cut -f 4 -d' ')
jobID_35=$(sbatch --dependency=afterok:$jobID_34 batch_train_q.sh 35 | cut -f 4 -d' ')
jobID_36=$(sbatch --dependency=afterok:$jobID_35 batch_train_q.sh 36 | cut -f 4 -d' ')
jobID_37=$(sbatch --dependency=afterok:$jobID_36 batch_train_q.sh 37 | cut -f 4 -d' ')
jobID_38=$(sbatch --dependency=afterok:$jobID_37 batch_train_q.sh 38 | cut -f 4 -d' ')
jobID_39=$(sbatch --dependency=afterok:$jobID_38 batch_train_q.sh 39 | cut -f 4 -d' ')

jobID_40=$(sbatch --dependency=afterok:$jobID_39 batch_train_q.sh 40 | cut -f 4 -d' ')
jobID_41=$(sbatch --dependency=afterok:$jobID_40 batch_train_q.sh 41 | cut -f 4 -d' ')
jobID_42=$(sbatch --dependency=afterok:$jobID_41 batch_train_q.sh 42 | cut -f 4 -d' ')
jobID_43=$(sbatch --dependency=afterok:$jobID_42 batch_train_q.sh 43 | cut -f 4 -d' ')
jobID_44=$(sbatch --dependency=afterok:$jobID_43 batch_train_q.sh 44 | cut -f 4 -d' ')
jobID_45=$(sbatch --dependency=afterok:$jobID_44 batch_train_q.sh 45 | cut -f 4 -d' ')
jobID_46=$(sbatch --dependency=afterok:$jobID_45 batch_train_q.sh 46 | cut -f 4 -d' ')
jobID_47=$(sbatch --dependency=afterok:$jobID_46 batch_train_q.sh 47 | cut -f 4 -d' ')
jobID_48=$(sbatch --dependency=afterok:$jobID_47 batch_train_q.sh 48 | cut -f 4 -d' ')
jobID_49=$(sbatch --dependency=afterok:$jobID_48 batch_train_q.sh 49 | cut -f 4 -d' ')

jobID_50=$(sbatch --dependency=afterok:$jobID_49 batch_train_q.sh 50 | cut -f 4 -d' ')
jobID_51=$(sbatch --dependency=afterok:$jobID_50 batch_train_q.sh 51 | cut -f 4 -d' ')
jobID_52=$(sbatch --dependency=afterok:$jobID_51 batch_train_q.sh 52 | cut -f 4 -d' ')
jobID_53=$(sbatch --dependency=afterok:$jobID_52 batch_train_q.sh 53 | cut -f 4 -d' ')
jobID_54=$(sbatch --dependency=afterok:$jobID_53 batch_train_q.sh 54 | cut -f 4 -d' ')
jobID_55=$(sbatch --dependency=afterok:$jobID_54 batch_train_q.sh 55 | cut -f 4 -d' ')
jobID_56=$(sbatch --dependency=afterok:$jobID_55 batch_train_q.sh 56 | cut -f 4 -d' ')
jobID_57=$(sbatch --dependency=afterok:$jobID_56 batch_train_q.sh 57 | cut -f 4 -d' ')
jobID_58=$(sbatch --dependency=afterok:$jobID_57 batch_train_q.sh 58 | cut -f 4 -d' ')
jobID_59=$(sbatch --dependency=afterok:$jobID_58 batch_train_q.sh 59 | cut -f 4 -d' ')

jobID_60=$(sbatch --dependency=afterok:$jobID_59 batch_train_q.sh 60 | cut -f 4 -d' ')
jobID_61=$(sbatch --dependency=afterok:$jobID_60 batch_train_q.sh 61 | cut -f 4 -d' ')
jobID_62=$(sbatch --dependency=afterok:$jobID_61 batch_train_q.sh 62 | cut -f 4 -d' ')
jobID_63=$(sbatch --dependency=afterok:$jobID_62 batch_train_q.sh 63 | cut -f 4 -d' ')
jobID_64=$(sbatch --dependency=afterok:$jobID_63 batch_train_q.sh 64 | cut -f 4 -d' ')
jobID_65=$(sbatch --dependency=afterok:$jobID_64 batch_train_q.sh 65 | cut -f 4 -d' ')
jobID_66=$(sbatch --dependency=afterok:$jobID_65 batch_train_q.sh 66 | cut -f 4 -d' ')
jobID_67=$(sbatch --dependency=afterok:$jobID_66 batch_train_q.sh 67 | cut -f 4 -d' ')
jobID_68=$(sbatch --dependency=afterok:$jobID_67 batch_train_q.sh 68 | cut -f 4 -d' ')
jobID_69=$(sbatch --dependency=afterok:$jobID_68 batch_train_q.sh 69 | cut -f 4 -d' ')
sbatch  --dependency=afterok:$jobID_69 batch_train_q.sh 70

