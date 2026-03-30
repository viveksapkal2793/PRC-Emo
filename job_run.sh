#!/bin/bash
#SBATCH --job-name=llm_bio_extract
#SBATCH --output=/scratch/data/bikash_rs/Vivek/PRC-Emo/logs/%x_%j.out
#SBATCH --error=/scratch/data/bikash_rs/Vivek/PRC-Emo/logs/%x_%j.err
#SBATCH --partition=dgx
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=2-00:00:00
#SBATCH --qos=fatqos
#SBATCH -D /scratch/data/bikash_rs/Vivek/PRC-Emo

# Create logs directory
mkdir -p logs

# Load CUDA module (adjust version based on your system)
# module avail
# nvidia-smi
# which nvcc
# module load cuda/11.8

# Activate virtual environment
unset LD_LIBRARY_PATH
unset BNB_CUDA_VERSION
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:/usr/local/cuda/targets/x86_64-linux/lib:$LD_LIBRARY_PATH
source prc-emo-env/bin/activate
# pip install --upgrade bitsandbytes scipy
# pip install bitsandbytes==0.43.2 --no-cache-dir
# python -m bitsandbytes
# MAX_JOBS=4 pip install flash-attn==2.3.6 --no-build-isolation --cache-dir /scratch/data/bikash_rs/Vivek/pip-cache

# python src/llm_emotion_extract_v2.py

python src/llm_bio_extract_v2.py

# python src/get_rag_final.py

# python test.py

# python src/reformat_data_ft_llm_combine.py \
#     --data_name meld \
#     --around_window 5 \
#     --prompting_type ImplicitEmotion_V3 \
#     --extract_prompting_llm_id qwen_3_14b \
#     --re_gen_data
