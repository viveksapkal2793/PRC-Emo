#!/bin/bash
#SBATCH --job-name=preprocess_prc_emo
#SBATCH --output=/scratch/data/bikash_rs/Vivek/PRC-Emo/logs/%x_%j.out
#SBATCH --error=/scratch/data/bikash_rs/Vivek/PRC-Emo/logs/%x_%j.err
#SBATCH --partition=dgx
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=6:00:00
#SBATCH --qos=fatqos
#SBATCH -D /scratch/data/bikash_rs/Vivek/PRC-Emo

# Create logs directory
mkdir -p logs

# Load CUDA module (adjust version based on your system)
# module load cuda/11.8

# Activate virtual environment
source prc-emo-env/bin/activate

python src/llm_emotion_extract_v2.py

# python src/llm_bio_extract_v2.py

# python src/get_rag_final.py

# python test.py

# python src/reformat_data_ft_llm_combine.py \
#     --data_name meld \
#     --around_window 5 \
#     --prompting_type ImplicitEmotion_V3 \
#     --extract_prompting_llm_id qwen_3_14b \
#     --re_gen_data
