#!/bin/bash
# SBATCH --account=zd26
# SBATCH --job-name=concat_idt
# SBATCH --ntasks=1
# SBATCH --ntasks-per-node=1
# SBATCH --cpus-per-task=1
# SBATCH --gres=gpu:1
# SBATCH --mem-per-cpu=40G
# SBATCH --time=0-24:00:00
# SBATCH --mail-user=tinplay41@gmail.com
# SBATCH --mail-type=END
# SBATCH --mail-type=FAIL
# SBATCH --output=/fs02/zd26/collage_main/Tin2/cycleGAN_pix2pix/output_folder/sbatch_out2.out
# SBATCH --error=/fs02/zd26/collage_main/Tin2/cycleGAN_pix2pix/output_folder/sbatch_err2.err

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
# --lambda_identity 0
python train.py --dataroot ./datasets/collage_testing --name concat_idt --model cycle_gan_3rdloss_concat --direction AtoB --dataset_mode unaligned  --use_wandb  --lr_decay_iters 50 --concat --batch_size 5 --input_nc 15 --lr 0.00000002 --netG resnet_6blocks_concat --drop_last


echo 'Run finished!'
