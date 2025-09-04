#!/bin/bash
#SBATCH -J diabetes-xgb
#SBATCH -o ../logs/output_%j.out
#SBATCH -e ../logs/error_%j.err
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -p skx-dev
#SBATCH -t 01:00:00

set -e
mkdir -p ../logs

module load python/3.9
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r ../requirements.txt

python ../src/main.py
