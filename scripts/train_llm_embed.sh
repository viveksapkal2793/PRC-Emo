#!/bin/bash
#SBATCH --job-name=train_qwen3_embed_8b_meld_aud_vis_lora_mlp_prototype
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
# module load cuda/11.8

# Activate virtual environment
source prc-emo-env/bin/activate

for seed in 42;
do 
python src/qwen3_embedding_mlp_emotion.py \
  --mode train_eval \
  --dataset meld \
  --load_in_4bit \
  --seed ${seed} \
  --epochs 10 \
  --embedding_analysis \
  --supcon_lambda 0.3 \
  --lora_contrastive_finetune \
  --lora_target_modules "q_proj,k_proj,v_proj,o_proj" \
  --use_memory_bank_supcon \
  --prototype_learning \
  --prototype_supcon_lambda 0.3 \
  --train_file "data/meld.train.0shot_w5_ImplicitEmotion_V3_qwen_3_14b_Aud_Vis.jsonl" \
  --valid_file "data/meld.valid.0shot_w5_ImplicitEmotion_V3_qwen_3_14b_Aud_Vis.jsonl" \
  --test_file "data/meld.test.0shot_w5_ImplicitEmotion_V3_qwen_3_14b_Aud_Vis.jsonl" \
  --include_conversation_context \
  --include_audio_description \
  --include_visual_description \
  # --include_llm_aud_vis_desc \
  # --proj_supcon \
  # --include_reference_similar_emotions \
  # --include_explicit_emotion \
  # --include_implicit_emotion \
  # --include_speaker_description \

done

wait

