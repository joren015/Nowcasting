#!/bin/bash -l
#SBATCH --time=10:00:00
#SBATCH --ntasks=16
#SBATCH --nodes=4
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=bikax003@umn.edu
#SBATCH -p a100-4
#SBATCH --gres=gpu:a100:1

# cd /panfs/jay/groups/6/csci8523/bikax003/Nowcasting/src/nowcasting/

module load python3
module load cudnn/8.2.0
module load cuda/11.2
module load cuda-sdk/11.2

pip install -r ../../requirements_dev.txt
pip install -r ../../requirements.txt
pip install .
pip install git+https://github.com/pySTEPS/pysteps
# pip install git+https://github.com/hydrogo/rainymotion


python3 pysteps_.py
