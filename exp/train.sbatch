#!/bin/bash
#SBATCH --job-name=train_ppo
# give it any name you want

#SBATCH --cpus-per-task=4
# max 24 per node

#SBATCH --partition=day
# choose out of day, week, month depending on job duration

#SBATCH --mem-per-cpu=3G
# max 251GB per node

#SBATCH --gres=gpu:1
# how many gpus to use
# each node has 4 gpus

#SBATCH --time=1-00:00:00
# job length: the job will run either until completion or until this timer runs out

#SBATCH --error=job.%J.err
# %J is the job ID, errors will be written to this file

#SBATCH --output=job.%J.out
# the output will be written in this file

#SBATCH --mail-type=ALL
# write a mail if a job begins, ends, fails, gets requeued or stages out
# options: NONE, BEGIN, END, FAIL, REQUEUE, ALL

#SBATCH --mail-user=kim-isabella.zierahn@student.uni-tuebingen.de
# your email

# here will be your commands for running the script

# Copy dataset to scratch folder (if needed)
cp -R KI_PPO_ppg2.py /scratch/$SLURM_JOB_ID/
cp -R PPG_batch.py /scratch/$SLURM_JOB_ID/

# Run script inside Singularity container
singularity exec --nv /common/singularityImages/TCML-CUDA12_4_TF2_17_PT_2_4.simg python3 ~/Comparison_batch.py

echo "DONE!"

