#!/bin/bash
#SBATCH --job-name=mon_concatCG3rdloss
#SBATCH --time=48:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:P100:2
#SBATCH --ntasks=8
#SBATCH --mem=50G
#SBATCH --mail-user=tinplay41@gmail.com
#SBATCH --mail-type=FAIL
#SBATCH --mail-type=END
#SBATCH --output=/home/nngu0068/zd26/nngu0068/Tin2/cycleGAN_pix2pix/output_folder/sbatch_out.out
#SBATCH --error=/home/nngu0068/zd26/nngu0068/Tin2/cycleGAN_pix2pix/output_folder/sbatch_err.err

##SBATCH --qos=desktopq
##SBATCH --account=zd26


# A script to monitor gpu and cpu usage for arbitrary commands (Python flavour?) 
pathToRepo=/home/nngu0068/zd26/nngu0068/Tin2/cycleGAN_pix2pix/
# Define BASH functions for monitoring 

echo 'train pix-to-pix'
#module load cuda/10.1
module load pytorch
python3 -m venv .mon_venv
source .mon_venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
echo $(which python3)

cd $pathToRepo
wandb login 4e12773a4724334345ecccf11dcf4d485280365e #myAPI
python train.py --dataroot ./datasets/collage_testing --name mon_concatCG3rdloss --model cycle_gan_3rdloss --direction AtoB --dataset_mode unaligned  --use_wandb  --lr_decay_iters 15 --concat --batch_size 5 --input_nc 15 --lambda_identity 0 --continue_train


echo 'Run finished!'
