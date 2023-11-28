#!/bin/bash 
# 
#SBATCH --job-name=hecktor 
#SBATCH --output=/WAVE/users/unix/smalladi/varian_ml/MedicalImaging/logs/outputlogs-%j.out 
# 
#SBATCH --partition=gpu
#SBATCH --nodes=1 
#SBATCH --ntasks=1 
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=16G
#SBATCH --time=2-00:00:00
#SBATCH --gres=gpu:1
#SBATCH --mail-user=smalladi@scu.edu
#SBATCH --mail-type=FAIL,END
#
module load Anaconda3
conda activate venv
cd /WAVE/users/unix/smalladi/varian_ml/MedicalImaging
python train_folds.py
