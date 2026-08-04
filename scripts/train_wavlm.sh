#!/bin/bash
#SBATCH --job-name=train_wavlm_iemocap_mlp
#SBATCH --output=/scratch/data/bikash_rs/Vivek/PRC-Emo/logs/%x_%j.out
#SBATCH --error=/scratch/data/bikash_rs/Vivek/PRC-Emo/logs/%x_%j.err
#SBATCH --partition=dgx
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=2:00:00
#SBATCH --qos=fatqos
#SBATCH -D /scratch/data/bikash_rs/Vivek/PRC-Emo

# Create logs directory
mkdir -p logs

# Load CUDA module (adjust version based on your system)
# module load cuda/11.8

# Activate virtual environment
source prc-emo-env/bin/activate

for seed in 42;
do 
python src/wavlm_large_mlp_emotion.py \
  --mode train_eval \
  --dataset iemocap \
  --seed ${seed} \
  --epochs 50 \
  --embedding_analysis \
  # --prototype_learning \
  # --prototype_supcon_lambda 0.3 \
  # --supcon_lambda 0.3 \
  # --lora_contrastive_finetune \
  # --lora_target_modules "q_proj,k_proj,v_proj,out_proj" \
  # --use_memory_bank_supcon \
  # --proj_supcon \

done

wait

