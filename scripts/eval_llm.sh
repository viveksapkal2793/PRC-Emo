#!/bin/bash
#SBATCH --job-name=eval_prc_emo
#SBATCH --output=/scratch/data/bikash_rs/Vivek/PRC-Emo/logs/%x_%j.out
#SBATCH --error=/scratch/data/bikash_rs/Vivek/PRC-Emo/logs/%x_%j.err
#SBATCH --partition=dgx
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=8:00:00
#SBATCH --qos=fatqos
#SBATCH -D /scratch/data/bikash_rs/Vivek/PRC-Emo

# Create logs directory
mkdir -p logs

# Load CUDA module (adjust version based on your system)
# module load cuda/11.8

# Activate virtual environment
source prc-emo-env/bin/activate

# Model to evaluate
MODEL_PATH="/scratch/data/bikash_rs/Vivek/PRC-Emo/finetuned_llm/meld_scratch_data_bikash_rs_Vivek_PRC-Emo_models_qwen_3_8b_ep4_step-1_lrs-linear3e-4_0shot_r32_w5_ImplicitEmotion_V3_seed42_L2048_llmdescqwen_3_14b_ED100000_final_full_finetune"

python ./src/ft_llm_cl_copy.py \
  --do_eval_test \
  --do_eval_dev \
  --ft_model_path ${MODEL_PATH} \
  --base_model_id /scratch/data/bikash_rs/Vivek/PRC-Emo/models/qwen_3_8b \
  --data_name meld \
  --prompting_type ImplicitEmotion_V3 \
  --extract_prompting_llm_id qwen_3_14b \
  --window 5 \
  --kshot 0 \
  --seed 42 \
  --max_seq_len 2048 \
  --data_folder ./data/