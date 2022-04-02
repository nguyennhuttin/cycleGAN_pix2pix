#!/bin/bash
#SBATCH --account=zd26
#SBATCH --job-name=concatCG3rdloss
#SBATCH --time=48:00:00
#SBATCH --gres=gpu:2
#SBATCH --ntasks=4
#SBATCH --mem=50G
#SBATCH --mail-user=tinplay41@gmail.com
#SBATCH --mail-type=FAIL
#SBATCH --mail-type=END
#SBATCH --output=/fs02/zd26/collage_main/Tin2/cycleGAN_pix2pix/output_folder/sbatch_out.out
#SBATCH --error=/fs02/zd26/collage_main/Tin2/cycleGAN_pix2pix/output_folder/sbatch_err.err
#SBATCH --partition=desktop
#SBATCH --qos=desktopq

# A script to monitor gpu and cpu usage for arbitrary commands (Python flavour?) 
pathToRepo=/fs02/zd26/collage_main/Tin2/cycleGAN_pix2pix/
# Define BASH functions for monitoring 

echo 'train pix-to-pix'
#module load cuda/10.1
module load pytorch
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
echo $(which python3)

cd $pathToRepo
# wandb login #myAPI
python train.py --dataroot ./datasets/collage_testing --name concatCG3rdloss --model cycle_gan_3rdloss --direction AtoB --dataset_mode unaligned  --use_wandb  --lr_decay_iters 10 --concat --batch_size 5 --input_nc 15 --lambda_identity 0 --lr 0.00000002 --continue_train 


echo 'Run finished!'
