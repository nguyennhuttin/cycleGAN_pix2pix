#!/bin/bash
#SBATCH --account=zd26
#SBATCH --job-name=cycle_gan_3rdloss_wandb2
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:2
#SBATCH --partition=desktop
#SBATCH --qos=desktopq
#SBATCH --ntasks=8
#SBATCH --mem=50G
#SBATCH --mail-user=tinplay41@gmail.com
#SBATCH --mail-type=FAIL
#SBATCH --mail-type=END
#SBATCH --output=sbatch_out.out
#SBATCH --error=sbatch_err.err


# A script to monitor gpu and cpu usage for arbitrary commands (Python flavour?) 
pathToRepo=/home/nngu0068/zd26/collage_main/Tin/pytorch-CycleGAN-and-pix2pix
# Define BASH functions for monitoring 

echo 'train pix-to-pix'
#module load cuda/10.1
module load pytorch
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt

#module load anaconda/2020.07-Python3.8-gcc8
#conda create -n myconda python=3.8 anaconda
#source activate myconda
# module load anaconda/2019.03-Python3.7-gcc5
#source /scratch/vf38/jtos0003/miniconda/bin/activate 
#conda activate pytorch-CycleGAN-and-pix2pix
echo $(which python3)

cd $pathToRepo
# python3 -m cProfile -o output/replicate_umm.prof UMM_discovery_BBBC021_copy.py
#python train.py --dataroot ./datasets/facades --name facades_pix2pix --model pix2pix --direction BtoA --use_wandb
#python train.py --dataroot ./datasets/collage_testing --name collage_test_pix2pix --model pix2pix --direction AtoB --dataset_mode unaligned --use_wandb
#wandb login myAPI
#wandb offline
#python -m visdom.server
python train.py --dataroot ./datasets/collage_testing --name collage_test_cycle_gan_3rdloss_wandb2 --model cycle_gan_3rdloss --direction AtoB --dataset_mode unaligned --display_id 0 --use_wandb 
#python train.py --dataroot ./datasets/john_multiple_match --name abc --model pix2pix --direction BtoA --use_wandb

echo 'Run finished!'
