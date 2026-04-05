import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0" 
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
import argparse
import json
from sklearn.metrics import classification_report
from datasets import load_dataset, Dataset
import torch
from dataclasses import dataclass
from typing import Any, Dict, List, Optional
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, TrainingArguments, DataCollatorForLanguageModeling
from trl import setup_chat_format, set_seed as trl_seed
from peft import LoraConfig, AutoPeftModelForCausalLM
from trl import SFTTrainer
from transformers import set_seed as transf_seed
from tqdm import tqdm
from sklearn.metrics import f1_score, accuracy_score
import numpy as np 
from torch.utils.data import DataLoader
from transformers.trainer_utils import EvalLoopOutput
import random, glob
import shutil
from lightning import seed_everything 
from transformers.trainer_utils import get_last_checkpoint
from reformat_data_ft_llm_combine import process
from transformers import TrainerCallback
from peft import PeftModel, PeftConfig

#4090用不了这两个，需要禁用
os.environ["NCCL_P2P_DISABLE"] = "1"
os.environ["NCCL_IB_DISABLE"] = "1"
# 添加 numpy.ndarray 到安全白名单
torch.serialization.add_safe_globals([np.ndarray])
# 覆盖默认加载行为（确保所有 torch.load 调用生效）
_original_torch_load = torch.load
def patched_torch_load(*args, **kwargs):
    kwargs.setdefault("weights_only", False)  # 强制关闭安全模式
    return _original_torch_load(*args, **kwargs)
torch.load = patched_torch_load

#随机种子设置
def set_random_seed(seed: int):
    """set seeds for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    seed_everything(seed=seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    trl_seed(seed)
    transf_seed(seed)
    
def formatting_prompts_func(samples):

    messages = samples["messages"]

    if isinstance(messages, list) and len(messages) > 0:

        if isinstance(messages[0], dict):

            prompt = tokenizer.apply_chat_template(
                messages[:-1],
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False
            )

            answer = messages[-1]["content"].strip()

            full_text = prompt

            if not hasattr(formatting_prompts_func, "_debug"):
                print("\nDEBUG TRAIN TEXT:\n")
                print(full_text[-500:])
                formatting_prompts_func._debug = True

            return full_text


        elif isinstance(messages[0], list):

            texts = []

            for conversation in messages:

                prompt = tokenizer.apply_chat_template(
                    conversation[:-1],
                    tokenize=False,
                    add_generation_prompt=True,
                    enable_thinking=False
                )

                answer = conversation[-1]["content"].strip()

                texts.append(prompt)

            return texts

    return ""

def split_label(sample):
    """提取并编码标签"""
    tokenized_lb = tokenizer.encode(sample['messages'][-1]['content'], padding='max_length',max_length=10 )
    sample['labels'] = tokenized_lb 
    return sample

class CurriculumDataset:
    """课程学习数据集管理器"""
    def __init__(self, dataset, bucket_number=8, curriculum=True):
        self.full_dataset = dataset
        self.bucket_number = bucket_number
        self.curriculum = curriculum
        self.buckets = None     #初始化桶列表为空，后续根据需要创建
        
        if self.curriculum:
            self.buckets = self._create_buckets()
            print(f"Created {len(self.buckets)} buckets for curriculum learning")
            for i, bucket in enumerate(self.buckets):
                print(f"Bucket {i}: {len(bucket)} samples, difficulty range: {min([s['difficulty'] for s in bucket]):.4f} - {max([s['difficulty'] for s in bucket]):.4f}")
    
    def _create_buckets(self):
        """根据difficulty字段创建桶"""
        # 按difficulty难度值升序排序
        sorted_data = sorted(self.full_dataset, key=lambda x: x.get('difficulty', 0))
        
        # 分桶
        bucket_size = len(sorted_data) // self.bucket_number
        buckets = []
        
        for i in range(self.bucket_number):
            start_idx = i * bucket_size
            if i == self.bucket_number - 1:  # 最后一个桶包含剩余所有数据
                end_idx = len(sorted_data)
            else:
                end_idx = (i + 1) * bucket_size
            
            bucket = sorted_data[start_idx:end_idx]
            buckets.append(bucket)
        
        return buckets
    
    def get_curriculum_dataset(self, step_index):
        """获取当前课程步骤的数据集"""
        if not self.curriculum or step_index >= len(self.buckets):
            # 如果不使用课程学习或已达到最后一步，返回全部数据
            data = list(self.full_dataset)
        else:
            # 返回从第0桶到第step_index桶的所有数据
            data = []
            for i in range(step_index + 1):
                data.extend(self.buckets[i])
        
        print(f"Curriculum step {step_index}: using {len(data)} samples")
        
        # 随机打乱数据
        random.shuffle(data)
        
        # 转换为HuggingFace Dataset格式
        return Dataset.from_list(data)

@dataclass
class DataCollatorForEvaluationWithLabels:
    """
    Custom data collator that:
    - Pads input_ids and attention_mask normally
    - Preserves text labels without padding
    """
    tokenizer: Any
    pad_to_multiple_of: Optional[int] = None
    return_tensors: str = "pt"
    
    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        # Extract labels before padding (they're text)
        labels = [feature.pop('labels') for feature in features if 'labels' in feature]
        
        # Pad the input tensors (input_ids, attention_mask)
        batch = self.tokenizer.pad(
            features,
            padding=True,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors=self.return_tensors,
        )
        
        # Add labels back as list (not tensor)
        if labels:
            batch['labels'] = labels
        
        return batch

# 移除了课程学习相关逻辑的简化版Trainer
class SimplifiedTrainer(SFTTrainer):
    
    def __init__(self, *args, **kwargs):
        # ✅ CRITICAL: Detect if we're in training or evaluation mode
        train_dataset = kwargs.get('train_dataset')
        eval_dataset = kwargs.get('eval_dataset')
        
        # Store references
        self.custom_train_dataset = train_dataset
        self.custom_eval_dataset = eval_dataset
        self.custom_max_seq_length = kwargs.get('max_seq_length', 2048)
        
        # Determine mode
        self.is_training_mode = train_dataset is not None
        
        # Ensure required parameters for old TRL version
        kwargs.setdefault('formatting_func', formatting_prompts_func)
        kwargs.setdefault('dataset_text_field', None)
        
        # Store data processing configuration
        self.data_process_args = argparse.Namespace(
            packing=kwargs.get('packing', False),
            dataset_text_field=None,
            max_seq_length=kwargs.get('max_seq_length', 2048),
            formatting_func=formatting_prompts_func,
            num_of_sequences=kwargs.get('num_of_sequences', 1024),
            chars_per_token=kwargs.get('chars_per_token', 3.6),
            remove_unused_columns=False,  # ✅ Keep all columns for debugging
            dataset_kwargs=kwargs.get('dataset_kwargs', {})
        )
        
        # ✅ Initialize parent SFTTrainer FIRST
        super().__init__(*args, **kwargs)

        # ✅ AFTER parent init, set custom data collator ONLY for evaluation mode
        if not self.is_training_mode:
            tokenizer = kwargs.get('tokenizer')
            if tokenizer is not None:
                self.data_collator = DataCollatorForEvaluationWithLabels(
                    tokenizer=tokenizer,
                    pad_to_multiple_of=8,
                    return_tensors="pt"
                )
        
        # ✅ Process evaluation dataset AFTER parent init (only if NOT training)
        if self.custom_eval_dataset is not None and not self.is_training_mode:
            print("Processing evaluation dataset...")
            self.eval_dataset = self._process_raw_data(self.custom_eval_dataset, is_training=False)

        if self.eval_dataset is not None:
            print("len(eval dataset) = ",  len(self.eval_dataset))
    
    def _process_raw_data(self, dataset, is_training=False):
        """
        Data preprocessing pipeline.
        
        Args:
            dataset: Input dataset
            is_training: If True, DON'T add labels (SFTTrainer handles them)
                        If False, extract text labels for evaluation metrics
        """
        # ✅ Only extract labels for EVALUATION
        original_labels = []
        if not is_training:
            print("📋 Extracting labels for evaluation...")
            for sample in dataset:
                if 'messages' in sample and len(sample['messages']) > 0:
                    emotion = sample['messages'][-1]['content'].strip().lower()
                    original_labels.append(emotion)
                else:
                    original_labels.append("unknown")
            print(f"✅ Extracted {len(original_labels)} labels")
        
        # ✅ Tokenize the dataset
        print("🔄 Tokenizing dataset...")
        processed_dataset = self._prepare_dataset(
            dataset=dataset,
            tokenizer=self.tokenizer,
            packing=False,  # False for cleaner processing
            dataset_text_field=None,
            max_seq_length=self.data_process_args.max_seq_length,
            formatting_func=self.data_process_args.formatting_func,
            num_of_sequences=self.data_process_args.num_of_sequences,
            chars_per_token=self.data_process_args.chars_per_token,
            remove_unused_columns=False,  # ✅ Keep all columns
            **self.data_process_args.dataset_kwargs, 
        )
        
        # ✅ Add text labels ONLY for evaluation
        if not is_training and original_labels:
            # Remove any existing labels column first
            if 'labels' in processed_dataset.column_names:
                print("⚠️ Removing existing labels column...")
                processed_dataset = processed_dataset.remove_columns(['labels'])
            
            # Add text labels for evaluation metrics
            processed_dataset = processed_dataset.add_column('labels', original_labels)
            print(f"✅ Added {len(original_labels)} text labels for evaluation")
        # elif is_training:
        #     # ✅ For training: ensure NO labels column exists
        #     if 'labels' in processed_dataset.column_names:
        #         print("⚠️ Training mode: Removing labels column (SFTTrainer will auto-generate)")
        #         processed_dataset = processed_dataset.remove_columns(['labels'])
        #     print("✅ Training dataset ready (no labels column)")
        # elif is_training:

        #     labels = []

        #     for sample in dataset:
        #         emotion = sample["messages"][-1]["content"]
        #         labels.append(tokenizer.encode(emotion))

        #     processed_dataset = processed_dataset.add_column("labels", labels)

        return processed_dataset
    
    def get_eval_dataloader(self, eval_dataset=None) -> DataLoader:
        """Get evaluation dataloader"""
        if eval_dataset is None:
            eval_dataset = self.eval_dataset
        
        # ✅ Process dataset if not already processed
        if eval_dataset is not None:
            if "input_ids" not in eval_dataset.column_names:
                print("Processing evaluation dataset in get_eval_dataloader...")
                eval_dataset = self._process_raw_data(eval_dataset, is_training=False)
        
        return super().get_eval_dataloader(eval_dataset)
    
    def _prepare_inputs(self, inputs):
        """Override to clean up malformed inputs with leading <|im_end|> tokens"""
        inputs = super()._prepare_inputs(inputs)
        
        # ✅ FIX: Remove leading <|im_end|> and pad tokens from input_ids
        if "input_ids" in inputs:
            batch_size = inputs["input_ids"].shape[0]
            cleaned_input_ids = []
            cleaned_attention_mask = []
            
            eos_token_id = self.tokenizer.eos_token_id
            pad_token_id = self.tokenizer.pad_token_id
            
            for i in range(batch_size):
                input_ids = inputs["input_ids"][i]
                attention_mask = inputs["attention_mask"][i]
                
                # Find first non-<|im_end|>, non-pad token
                start_idx = 0
                while start_idx < len(input_ids):
                    token_id = input_ids[start_idx].item()
                    if token_id not in [eos_token_id, pad_token_id]:
                        break
                    start_idx += 1
                
                # Keep cleaned sequence (or original if no cleanup needed)
                if start_idx > 0:
                    cleaned_input_ids.append(input_ids[start_idx:])
                    cleaned_attention_mask.append(attention_mask[start_idx:])
                else:
                    cleaned_input_ids.append(input_ids)
                    cleaned_attention_mask.append(attention_mask)
            
            # Pad all sequences to same length (left padding)
            max_len = max(len(seq) for seq in cleaned_input_ids)
            
            padded_input_ids = []
            padded_attention_mask = []
            
            for input_ids, attention_mask in zip(cleaned_input_ids, cleaned_attention_mask):
                padding_length = max_len - len(input_ids)
                
                # Left padding (since tokenizer.padding_side = 'left')
                if padding_length > 0:
                    padded_input_ids.append(
                        torch.cat([
                            torch.full((padding_length,), pad_token_id, dtype=input_ids.dtype, device=input_ids.device),
                            input_ids
                        ])
                    )
                    padded_attention_mask.append(
                        torch.cat([
                            torch.zeros(padding_length, dtype=attention_mask.dtype, device=attention_mask.device),
                            attention_mask
                        ])
                    )
                else:
                    padded_input_ids.append(input_ids)
                    padded_attention_mask.append(attention_mask)
            
            inputs["input_ids"] = torch.stack(padded_input_ids)
            inputs["attention_mask"] = torch.stack(padded_attention_mask)
        
        return inputs
    
    def evaluation_loop(
        self,
        dataloader: DataLoader,
        description: str,
        prediction_loss_only=None,
        ignore_keys=None,
        metric_key_prefix="eval",
    ) -> EvalLoopOutput:
        model = self.model
        model = model.to(dtype=torch.bfloat16)
        model.eval()
            
        all_preds = []
        all_labels = []
        all_raw_decoded = []
        
        debug_file = f"{self.args.output_dir}/evaluation_debug_{metric_key_prefix}.jsonl"
        debug_fp = open(debug_file, 'w', encoding='utf-8')
        
        def post_process(text):
            """
            Simple emotion extraction: just search for emotion keywords.
            Works because we already removed the input prompt.
            """
            try:
                # Available emotions
                emotions = ['neutral', 'surprise', 'fear', 'sadness', 'joy', 'disgust', 'anger']
                
                # Convert to lowercase for matching
                text_lower = text.lower()
                
                # ✅ SIMPLE: Just find which emotion appears in the text
                # Scan from end (emotion usually appears early in generation)
                for emotion in emotions:
                    if emotion in text_lower:
                        return emotion
                
                # ✅ Fallback: No emotion found
                print(f"⚠️ No emotion found in: '{text[:100]}'")
                return "neutral"
                
            except Exception as e:
                print(f"❌ ERROR in post_process: {e}")
                return "neutral"
        # def post_process(text):
        #     """Extract emotion label from model output"""
        #     try:
        #         # Step 1: Remove special tokens first
        #         text = text.replace("<|im_start|>", "").replace("<|im_end|>", "").strip()
                
        #         # Step 2: If there's "assistant" tag, take content after it
        #         if "assistant" in text:
        #             parts = text.split("assistant")
        #             # Get last non-empty part
        #             for part in reversed(parts):
        #                 part = part.strip()
        #                 if part:
        #                     text = part
        #                     break
                
        #         # Step 3: Handle empty output
        #         if not text:
        #             return "neutral"
                
        #         # Step 4: Take first line/word
        #         lines = text.split('\n')
        #         if lines:
        #             text = lines[0].strip()
                
        #         words = text.split()
        #         if not words:
        #             return "neutral"
                
        #         word = words[0].lower()
                
        #         # Step 5: Match or fix truncated emotions
        #         emotions = ["neutral", "surprise", "fear", "sadness", "joy", "disgust", "anger"]
                
        #         # Exact match
        #         if word in emotions:
        #             return word
                
        #         # Prefix match (for truncations like "sur" → "surprise")
        #         for emo in emotions:
        #             if word.startswith(emo[:3]) and len(word) >= 3:
        #                 return emo
                
        #         return word
                
        #     except Exception as e:
        #         print(f"❌ ERROR in post_process: {e}, text: '{text[:100]}'")
        #         return "neutral"
        
        with torch.no_grad():
            for step, inputs in enumerate(tqdm(dataloader, desc="Evaluating")):
                inputs = self._prepare_inputs(inputs)
                
                batch_size = inputs["input_ids"].shape[0]
                start_idx = step * batch_size
                end_idx = start_idx + batch_size
                
                # Extract true labels
                str_labels = []
                eval_dataset_with_labels = self.eval_dataset
                
                for idx in range(start_idx, min(end_idx, len(eval_dataset_with_labels))):
                    original_sample = eval_dataset_with_labels[idx]
                    if "messages" in original_sample and len(original_sample["messages"]) > 0:
                        emotion = original_sample["messages"][-1]["content"]
                    else:
                        emotion = "unknown"
                    emotion = emotion.strip().lower()
                    str_labels.append(emotion)
                
                # Generation parameters
                if step == 0:
                    print("\n" + "="*80)
                    print("🔍 DETAILED DEBUG - FIRST BATCH")
                    print("="*80)
                    print(f"\n⚙️ GENERATION PARAMETERS:")
                    gen_kwargs = {
                        'max_new_tokens': 10,
                        'do_sample': False,
                        'eos_token_id': self.tokenizer.eos_token_id,
                        'pad_token_id': self.tokenizer.pad_token_id,
                        'temperature': 1.0,
                    }
                    print(json.dumps(gen_kwargs, indent=2))
                else:
                    gen_kwargs = {
                        'max_new_tokens': 10,
                        'do_sample': False,
                        'eos_token_id': self.tokenizer.eos_token_id,
                        'pad_token_id': self.tokenizer.pad_token_id,
                    }
                
                # ✅ Generate predictions
                generated_tokens = model.generate(
                    inputs["input_ids"],
                    attention_mask=inputs["attention_mask"],
                    **gen_kwargs,
                )
                
                # ✅ CRITICAL FIX: Extract ONLY newly generated tokens
                input_length = inputs["input_ids"].shape[1]
                
                generated_only = []
                for i in range(generated_tokens.shape[0]):
                    new_tokens = generated_tokens[i][input_length:]
                    generated_only.append(new_tokens)
                
                # ✅ Decode ONLY generated tokens
                raw_decoded = [
                    self.tokenizer.decode(tokens, skip_special_tokens=False)
                    for tokens in generated_only
                ]
                
                # ✅ Apply post-processing
                str_decoded = [post_process(text) for text in raw_decoded]
                
                # Debug first batch
                if step == 0:
                    print(f"\n📤 GENERATED TOKENS (first sample):")
                    print(f"Input length: {input_length} tokens")
                    print(f"Generated length: {len(generated_only[0])} tokens")
                    print(f"Generated token IDs: {generated_only[0].tolist()}")
                    print(f"\n📝 RAW OUTPUT:")
                    print(f"'{raw_decoded[0]}'")
                    print(f"\n🔍 PROCESSED:")
                    print(f"'{str_decoded[0]}'")
                    print(f"\n📊 COMPARISON:")
                    print(f"True: '{str_labels[0]}'")
                    print(f"Pred: '{str_decoded[0]}'")
                    print(f"Match: {str_decoded[0] == str_labels[0]}")
                    print("="*80 + "\n")
                
                #Write debug info
                for i in range(len(str_decoded)):
                    debug_entry = {
                        "sample_idx": start_idx + i,
                        "true_label": str_labels[i] if i < len(str_labels) else "unknown",
                        "predicted_label": str_decoded[i],
                        "raw_output": raw_decoded[i],
                        "match": str_decoded[i] == (str_labels[i] if i < len(str_labels) else "unknown")
                    }
                    debug_fp.write(json.dumps(debug_entry, ensure_ascii=False) + '\n')
                
                all_preds += str_decoded
                all_labels += str_labels
                all_raw_decoded += raw_decoded
        
        debug_fp.close()
        print(f"\n Debug log saved to: {debug_file}")
        
        # ✅ Calculate metrics
        print(f"\n" + "="*80)
        print(" EVALUATION RESULTS")
        print("="*80)
        print(f"Total samples: {len(all_labels)}")
        print(f"Unique predicted labels: {set(all_preds)}")
        print(f"Unique true labels: {set(all_labels)}")
        
        # Count predictions
        from collections import Counter
        pred_counts = Counter(all_preds)
        label_counts = Counter(all_labels)
        
        print(f"\n Prediction distribution:")
        for label, count in pred_counts.most_common():
            print(f"  {label}: {count} ({100*count/len(all_preds):.1f}%)")
        
        print(f"\n True label distribution:")
        for label, count in label_counts.most_common():
            print(f"  {label}: {count} ({100*count/len(all_labels):.1f}%)")
        
        f1_weighted = f1_score(all_labels, all_preds, average="weighted", zero_division=0)
        f1_macro = f1_score(all_labels, all_preds, average="macro", zero_division=0)
        accuracy = accuracy_score(all_labels, all_preds)
        
        print(f"\n Classification Report:")
        print(classification_report(all_labels, all_preds, digits=4, zero_division=0))
        
        metrics = {
            f"{metric_key_prefix}_weighted-f1": f1_weighted,
            f"{metric_key_prefix}_macro-f1": f1_macro,
            f"{metric_key_prefix}_accuracy": accuracy
        }
        
        # ✅ Save results with better formatting
        result_data = {
            "metrics": metrics,
            "summary": {
                "total_samples": len(all_labels),
                "correct_predictions": sum(p == l for p, l in zip(all_preds, all_labels)),
                "predicted_label_counts": dict(pred_counts),
                "true_label_counts": dict(label_counts)
            },
            "detail_pred": [
                {
                    "predicted": pred,
                    "true_label": label,
                    "raw_output": raw[:500]  # Truncate for readability
                }
                for pred, label, raw in zip(all_preds[:100], all_labels[:100], all_raw_decoded[:100])
            ]
        }
        
        output_file = f"{self.args.output_dir}/result_{metric_key_prefix}_step-{self.state.global_step}.json"
        with open(output_file, "w") as f:
            json.dump(result_data, f, indent=2, ensure_ascii=False)
        
        print(f"\n Metrics:")
        print(f"  Weighted F1: {f1_weighted:.4f}")
        print(f"  Macro F1: {f1_macro:.4f}")
        print(f"  Accuracy: {accuracy:.4f}")
        print(f"\n Results saved to: {output_file}")
        print("="*80 + "\n")
        
        torch.cuda.empty_cache()
        return EvalLoopOutput(
            predictions=all_preds, 
            label_ids=all_labels, 
            metrics=metrics, 
            num_samples=len(all_labels)
        )
        
if __name__=='__main__':
    parser = argparse.ArgumentParser(description='Process ...')
    parser.add_argument('--do_train', action="store_true", help='fine tuning a LLM model with LoRA', default=False)
    parser.add_argument('--do_eval_test', action="store_true", help='eval on test set', default=False)
    parser.add_argument('--do_eval_dev', action="store_true", help='eval on dev set', default=False)
    parser.add_argument('--ft_model_path', type=str, default=None, help='fintuned model path') 
    parser.add_argument('--ft_model_id', type=str, default=None, help='fintuned model id for saving after train it')
    parser.add_argument('--prompting_type', type=str, default='spdescV2', help='prompting style in {cot, fewshot, zeroshot}')
    parser.add_argument('--base_model_id', type=str, default='/usr/3Tusr/lixinran/workspace/LLM/LLM_bases/LLaMA2/', help='base llm model id')
    parser.add_argument('--extract_prompting_llm_id', type=str, default='LLaMA2', help='base llm model id')
    parser.add_argument('--epoch', type=int, default=None, help='training epoch')
    parser.add_argument('--max_steps', type=int, default=None, help='training steps')
    parser.add_argument('--lr_scheduler', type=str, default='constant', help='learning rate scheduler')
    parser.add_argument('--lr', type=float, default=2e-4, help='learning rate value')
    parser.add_argument('--seed', type=int, default=42, help='random seed value')
    parser.add_argument('--kshot', type=int, default=0, help='k shot examples for llm')
    parser.add_argument('--lora_r', type=int, default=32, help='lora rank')
    parser.add_argument('--eval_delay', type=int, default=200, help='eval delay')
    parser.add_argument('--window', type=int, default=5, help='local context window size')
    parser.add_argument('--max_seq_len', type=int, default=None, help='max sequence length for chunking/packing')
    parser.add_argument('--re_gen_data', action="store_true", help='re generate data', default=False)
    parser.add_argument('--data_name', type=str,  help='data name in {iemocap, meld, emorynlp}', default='iemocap')
    parser.add_argument('--data_folder', type=str,  help='path folder save all data', default='./data/')
    parser.add_argument('--output_folder', type=str,  help='path folder save all data', default='./finetuned_llm/')
    # 课程学习相关参数
    parser.add_argument('--curriculum', action="store_true", help='enable curriculum learning', default=False)
    parser.add_argument('--bucket_number', type=int, default=8, help='number of buckets for curriculum learning')
    parser.add_argument('--curriculum_update_epochs', type=int, default=None, help='epochs between curriculum updates')

    args, unknown = parser.parse_known_args()
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    if args.prompting_type == 'zeroshot':
        args.kshot = 0
    print(args)
    
    set_random_seed(args.seed)
    
    # all_path_folder_preprocessed_data = [f"{args.data_folder}/{args.data_name}.{d_type}.{args.kshot}shot_w{args.window}_{args.prompting_type}_{args.extract_prompting_llm_id}.jsonl" \
    all_path_folder_preprocessed_data = [f"{args.data_folder}/{args.data_name}.{d_type}.{args.kshot}shot_w{args.window}_{args.prompting_type}.jsonl" \
        for d_type in [ 'train' , 'valid',  'test']]
    if args.re_gen_data:
        process(all_path_folder_preprocessed_data, args) 
                    
    # Load jsonl data from disk
    full_dataset = load_dataset("json", data_files=all_path_folder_preprocessed_data[0], split="train", cache_dir=f'{args.output_folder}/{args.ft_model_id}')
    valid_dataset = load_dataset("json", data_files=all_path_folder_preprocessed_data[1], split="train", cache_dir=f'{args.output_folder}/{args.ft_model_id}')
    test_dataset = load_dataset("json", data_files=all_path_folder_preprocessed_data[2], split="train", cache_dir=f'{args.output_folder}/{args.ft_model_id}')
    
    # 创建课程学习管理器
    curriculum_manager = None
    if args.curriculum and args.do_train:
        print("Initializing curriculum learning...")
        curriculum_manager = CurriculumDataset(
            dataset=list(full_dataset), 
            bucket_number=args.bucket_number,
            curriculum=True
        )
    
    # 定义量化配置（全局）
    tensor_data_type = torch.bfloat16
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True, 
        bnb_4bit_use_double_quant=True, 
        bnb_4bit_quant_type="nf4", 
        bnb_4bit_compute_dtype=tensor_data_type
    )
    
    # Load model and tokenizer
    model_id = args.base_model_id
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    
    # 非训练模式下单独处理
    if not args.do_train:
        tensor_data_type = torch.bfloat16
        ft_model_path = f"{args.output_folder}/{args.ft_model_id}" if args.ft_model_path is None else args.ft_model_path
        
        print(f"Loading model for evaluation from: {ft_model_path}")
        
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(ft_model_path)
        tokenizer.padding_side = 'left'
        
        # Load PEFT config
        peft_config = PeftConfig.from_pretrained(ft_model_path)
        
        # Load base model
        base_model = AutoModelForCausalLM.from_pretrained(
            peft_config.base_model_name_or_path,
            device_map="auto",
            torch_dtype=tensor_data_type,
            quantization_config=bnb_config
        )
        
        # Apply chat format BEFORE loading adapter
        base_model, tokenizer = setup_chat_format(base_model, tokenizer)
        
        # Load LoRA adapter
        model = PeftModel.from_pretrained(base_model, ft_model_path)
        model.eval()
        
        print("Model loaded successfully for evaluation")
        
        # CREATE TRAINING ARGS FOR EVALUATION
        dummy_train_args = TrainingArguments(
            output_dir=ft_model_path,
            per_device_eval_batch_size=1,
            bf16=True,
            dataloader_pin_memory=False,
            report_to="none",
        )
        
        # RUN EVALUATIONS
        results = {}
        
        # ✅ EVALUATE ON DEV SET
        if args.do_eval_dev:
            print("\n" + "="*50)
            print("Evaluating on VALIDATION set...")
            print("="*50)
            
            dev_trainer = SimplifiedTrainer(
                model=model,
                args=dummy_train_args,
                train_dataset=None,
                eval_dataset=valid_dataset,  # ✅ Pass actual dataset
                tokenizer=tokenizer,
                packing=False,
                max_seq_length=args.max_seq_len,
            )
            
            dev_results = dev_trainer.evaluate(metric_key_prefix='eval_dev')
            results['dev'] = dev_results
            
            print(f"\n Dev Results:")
            for key, value in dev_results.items():
                print(f"  {key}: {value:.4f}")
            
            del dev_trainer
            torch.cuda.empty_cache()
        
        # ✅ EVALUATE ON TEST SET
        if args.do_eval_test:
            print("\n" + "="*50)
            print("Evaluating on TEST set...")
            print("="*50)
            
            test_trainer = SimplifiedTrainer(
                model=model,
                args=dummy_train_args,
                train_dataset=None,
                eval_dataset=test_dataset,  # ✅ Pass actual dataset
                tokenizer=tokenizer,
                packing=False,
                max_seq_length=args.max_seq_len,
            )
            
            test_results = test_trainer.evaluate(metric_key_prefix='eval_test')
            results['test'] = test_results
            
            print(f"\n Test Results:")
            for key, value in test_results.items():
                print(f"  {key}: {value:.4f}")
            
            del test_trainer
            torch.cuda.empty_cache()
        
        # ✅ SAVE RESULTS
        if results:
            results_file = os.path.join(ft_model_path, "evaluation_results.json")
            with open(results_file, 'w') as f:
                json.dump(results, f, indent=2)
            print(f"\n Results saved to: {results_file}")
        else:
            print("\n No evaluation performed. Use --do_eval_test or --do_eval_dev")
        
        print("\n Evaluation completed!")
        exit(0)
        
    # 设置课程学习更新频率
    if args.curriculum and args.curriculum_update_epochs is None:
        # 确保有足够的epoch数
        if args.epoch:
            args.curriculum_update_epochs = max(1, args.epoch // args.bucket_number)
        else:
            # 如果没有指定epoch数，则使用默认值
            args.curriculum_update_epochs = 1
    
    # 分阶段训练实现
    if args.do_train and args.curriculum:
        print("="*50)
        print("Starting curriculum learning with phased training")
        print(f"Total buckets: {args.bucket_number}")
        print(f"Epochs per phase: {args.curriculum_update_epochs}")
        print("="*50)
        
        # 计算每个阶段需要的epoch数
        remaining_epochs = args.epoch or 0  # 确保不是None
        base_output_dir = f'{args.output_folder}/{args.ft_model_id}'
        
        # 创建卸载文件夹
        offload_folder = os.path.join(base_output_dir, "offload")
        os.makedirs(offload_folder, exist_ok=True)
        
        # 第一阶段模型初始化
        model = None
        trainer = None
        
        for phase in range(args.bucket_number):
            print("\n" + "="*50)
            print(f"Starting training phase {phase+1}/{args.bucket_number}")
            print("="*50)
            
            # 获取当前阶段的训练集
            current_dataset = curriculum_manager.get_curriculum_dataset(phase)
            
            # 设置当前阶段的输出目录
            phase_output_dir = f"{base_output_dir}_phase_{phase}"
            os.makedirs(phase_output_dir, exist_ok=True)
            
            # 计算当前阶段训练的epoch数
            if remaining_epochs > 0:
                current_epochs = min(args.curriculum_update_epochs, remaining_epochs)
                remaining_epochs -= current_epochs
            else:
                current_epochs = 0
            
            # 如果当前阶段没有epoch需要训练，跳过
            if current_epochs <= 0:
                print(f"Skipping phase {phase} as no epochs remaining")
                continue
            
            # 创建训练参数
            training_args = TrainingArguments(
                output_dir=phase_output_dir,
                num_train_epochs=current_epochs,
                max_steps=args.max_steps,
                per_device_train_batch_size=1,  # 减少批大小以节省显存
                per_device_eval_batch_size=1,
                gradient_accumulation_steps=4,
                gradient_checkpointing=True,
                save_total_limit=1,
                # save_only_model=True,
                optim="adamw_torch_fused", 
                eval_delay=args.eval_delay,
                logging_steps=50,
                eval_steps=50,
                save_steps=50,
                load_best_model_at_end=True,
                metric_for_best_model='weighted-f1',
                greater_is_better=True,
                eval_strategy='steps',
                logging_strategy='steps',
                save_strategy="steps",
                learning_rate=args.lr,
                bf16=True,
                tf32=False,
                max_grad_norm=0.3,
                warmup_ratio=0.03,
                lr_scheduler_type=args.lr_scheduler,
                push_to_hub=False,
                group_by_length=True,
                report_to="tensorboard",
            )

            # 如果是第一阶段，使用基础模型；否则加载上一阶段的模型
            if phase == 0:
                # 第一阶段：加载基础模型
                model = AutoModelForCausalLM.from_pretrained(
                    model_id,
                    device_map="auto",
                    # attn_implementation="flash_attention_2",
                    torch_dtype=tensor_data_type,
                    quantization_config=bnb_config
                )
                
                # 应用聊天格式
                tokenizer = AutoTokenizer.from_pretrained(model_id)
                tokenizer.padding_side = 'left'
                model, tokenizer = setup_chat_format(model, tokenizer)
                
                # 保存原始模型的分词器
                tokenizer.save_pretrained(phase_output_dir)
                print(f"Tokenizer vocab size after setup_chat_format: {len(tokenizer)}")
                print(f"Model embed_tokens size: {model.get_input_embeddings().weight.shape}")
                print(f"Special tokens: {tokenizer.special_tokens_map}")
                
                # 保存这些信息到文件
                debug_info = {
                    "vocab_size": len(tokenizer),
                    "embed_shape": list(model.get_input_embeddings().weight.shape),
                    "special_tokens": tokenizer.special_tokens_map
                }
                with open(f"{phase_output_dir}/debug_info.json", "w") as f:
                    json.dump(debug_info, f, indent=2)
                
                # 第一阶段：创建带有LoRA配置的训练器
                trainer = SimplifiedTrainer(
                    model=model,
                    args=training_args,
                    train_dataset=current_dataset,
                    eval_dataset=valid_dataset,
                    neftune_noise_alpha=5,
                    peft_config=LoraConfig(
                        lora_alpha=128,
                        lora_dropout=0.05,
                        r=args.lora_r,
                        bias="none",
                        target_modules="all-linear",
                        task_type="CAUSAL_LM", 
                    ),
                    max_seq_length=args.max_seq_len,
                    tokenizer=tokenizer,
                    packing=False,
                    dataset_kwargs={
                        "add_special_tokens": False,
                        "append_concat_token": False,
                    }
                )

                print("\n==============================")
                print("TRAIN DATASET SANITY CHECK")
                print("==============================\n")

                sample = formatting_prompts_func(full_dataset[0])

                print(sample[-500:])
                
            else:
                # 后续阶段：加载上一阶段保存的模型
                prev_phase_dir = f"{base_output_dir}_phase_{phase-1}"
                print(f"Loading model from previous phase: {prev_phase_dir}")
                
                # 释放内存
                if model is not None:
                    del model
                if trainer is not None:
                    del trainer
                torch.cuda.empty_cache()
                
                # **关键修改：先加载tokenizer，确保词汇表一致**
                tokenizer = AutoTokenizer.from_pretrained(prev_phase_dir)
                tokenizer.padding_side = 'left'
                
                # 首先加载配置
                peft_config = PeftConfig.from_pretrained(prev_phase_dir)
                
                # 加载基础模型
                base_model = AutoModelForCausalLM.from_pretrained(
                    peft_config.base_model_name_or_path,
                    device_map="auto",
                    torch_dtype=tensor_data_type,
                    quantization_config=bnb_config,
                    offload_folder=offload_folder,
                )
                # 设置 gradient checkpointing，确保输入也可参与反向传播
                base_model.gradient_checkpointing_enable()
                base_model.enable_input_require_grads()
                # 应用聊天格式
                base_model, tokenizer = setup_chat_format(base_model, tokenizer)
                for name, param in base_model.named_parameters():
                    param.requires_grad = False
                # 加载上阶段 LoRA adapter
                model = PeftModel.from_pretrained(
                    base_model,
                    prev_phase_dir,
                    offload_folder=offload_folder,
                )
                for name, param in model.named_parameters():
                    if not any(key in name for key in ['lora_A', 'lora_B', 'lora_embedding_A', 'lora_embedding_B']):
                        param.requires_grad = False
                    else:
                        param.requires_grad = True
                print("Embedding requires_grad:", model.get_input_embeddings().weight.requires_grad)
                print("Before parameter adjustment:")
                model.print_trainable_parameters()
                
                # **关键修改：确保LoRA参数可训练**
                # 方法1：使用内置方法
                model.train()  # 设置为训练模式
                
                # 方法2：手动设置LoRA参数为可训练
                for name, param in model.named_parameters():
                    if any(lora_key in name for lora_key in ['lora_A', 'lora_B', 'lora_embedding_A', 'lora_embedding_B']):
                        param.requires_grad = True
                        #print(f"Set {name} requires_grad = True")
                
                print("After parameter adjustment:")
                model.print_trainable_parameters()
                
                # **重要：后续阶段不要重新初始化LoRA配置**
                # 创建训练器时不传入peft_config，因为模型已经是PeftModel
                trainer = SimplifiedTrainer(
                    model=model,  # 直接使用已经加载LoRA的模型
                    args=training_args,
                    train_dataset=current_dataset,
                    eval_dataset=valid_dataset,
                    neftune_noise_alpha=5,
                    # **关键：不传入peft_config，因为模型已经是PeftModel**
                    peft_config=None,  
                    max_seq_length=args.max_seq_len,
                    tokenizer=tokenizer,
                    packing=False,
                    dataset_kwargs={
                        "add_special_tokens": False,
                        "append_concat_token": False,
                    }
                )

                print("\n==============================")
                print("TRAIN DATASET SANITY CHECK")
                print("==============================\n")

                sample = formatting_prompts_func(full_dataset[0])

                print(sample[-500:])
            
            # 开始训练当前阶段
            print(f"Training phase {phase} with {len(current_dataset)} samples for {current_epochs} epochs")
            
            # 训练前再次确认参数状态
            print("\n=== Final parameter check before training ===")
            trainable_params = 0
            all_params = 0
            for name, param in model.named_parameters():
                all_params += param.numel()
                if param.requires_grad:
                    trainable_params += param.numel()
                    #if 'lora' in name.lower():
                        #print(f"Trainable LoRA param: {name}, shape: {param.shape}")
            
            print(f"Trainable params: {trainable_params:,} || All params: {all_params:,} || Trainable%: {100 * trainable_params / all_params:.4f}")
            
            if trainable_params == 0:
                raise RuntimeError("No trainable parameters found! Training will fail.")
            
            trainer.train()
            
            # 保存当前阶段模型
            trainer.save_model(phase_output_dir)
            print(f"Saved model for phase {phase} to {phase_output_dir}")
            
            # 保存分词器
            tokenizer.save_pretrained(phase_output_dir)
            print(f"Evaluating on test dataset after phase {phase}...")
            # 如果所有epoch已完成，提前退出
            if remaining_epochs <= 0:
                print("All epochs completed. Exiting curriculum training.")
                break

        if remaining_epochs > 0:
            print("\n" + "="*50)
            print(f"Starting final training on full dataset for {remaining_epochs} additional epoch(s)")
            print("="*50)
            print(f"len(full_dataset):{len(full_dataset)}")
            # 设置输出路径
            full_phase_output_dir = f"{base_output_dir}_final_full_finetune"
            os.makedirs(full_phase_output_dir, exist_ok=True)

            # 清理内存，加载最后一阶段模型
            del model
            del trainer
            torch.cuda.empty_cache()

            last_phase_dir = f"{base_output_dir}_phase_{args.bucket_number - 1}"
            tokenizer = AutoTokenizer.from_pretrained(last_phase_dir)
            tokenizer.padding_side = 'left'
            peft_config = PeftConfig.from_pretrained(last_phase_dir)

            base_model = AutoModelForCausalLM.from_pretrained(
                peft_config.base_model_name_or_path,
                device_map="auto",
                torch_dtype=tensor_data_type,
                quantization_config=bnb_config,
                offload_folder=offload_folder,
            )
            base_model.gradient_checkpointing_enable()
            base_model.enable_input_require_grads()
            base_model, tokenizer = setup_chat_format(base_model, tokenizer)
            for name, param in base_model.named_parameters():
                param.requires_grad = False
            model = PeftModel.from_pretrained(
                base_model,
                last_phase_dir,
                offload_folder=offload_folder,
            )
            for name, param in model.named_parameters():
                if any(key in name for key in ['lora_A', 'lora_B', 'lora_embedding_A', 'lora_embedding_B']):
                    param.requires_grad = True
                else:
                    param.requires_grad = False
            model.train()

            # 新的训练参数（只修改 epoch）
            training_args = TrainingArguments(
                output_dir=full_phase_output_dir,
                num_train_epochs=remaining_epochs,
                max_steps=args.max_steps,
                per_device_train_batch_size=1,
                per_device_eval_batch_size=1,
                gradient_accumulation_steps=4,
                gradient_checkpointing=True,
                save_total_limit=1,
                # save_only_model=True,
                optim="adamw_torch_fused",
                eval_delay=args.eval_delay,
                logging_steps=50,
                eval_steps=50,
                save_steps=50,
                load_best_model_at_end=True,
                metric_for_best_model='weighted-f1',
                greater_is_better=True,
                eval_strategy='steps',
                logging_strategy='steps',
                save_strategy="steps",
                learning_rate=args.lr,
                bf16=True,
                tf32=False,
                fp16=False,
                dataloader_pin_memory=False,
                max_grad_norm=0.3,
                warmup_ratio=0.03,
                lr_scheduler_type=args.lr_scheduler,
                push_to_hub=False,
                group_by_length=True,
                report_to="tensorboard",
            )

            trainer = SimplifiedTrainer(
                model=model,
                args=training_args,
                train_dataset=full_dataset,
                eval_dataset=valid_dataset,
                peft_config=None,
                neftune_noise_alpha=5,
                max_seq_length=args.max_seq_len,
                tokenizer=tokenizer,
                packing=False,
                dataset_kwargs={
                    "add_special_tokens": False,
                    "append_concat_token": False,
                }
            )

            print("\n==============================")
            print("TRAIN DATASET SANITY CHECK")
            print("==============================\n")

            sample = formatting_prompts_func(full_dataset[0])

            print(sample[-500:])

            print(f"Training final phase with {len(full_dataset)} samples for {remaining_epochs} epochs")
            trainer.train()
            
            # 保存最终模型和测试评估结果
            trainer.save_model(full_phase_output_dir)
            tokenizer.save_pretrained(full_phase_output_dir)

            test_results = trainer.evaluate(test_dataset, metric_key_prefix=f'test_final_full')
            all_results_file = os.path.join(base_output_dir, "all_phase_test_results.json")
            if os.path.exists(all_results_file):
                with open(all_results_file, "r", encoding="utf-8") as f:
                    all_results = json.load(f)
            else:
                all_results = {}
            all_results.update({f"final_full": test_results})
            with open(all_results_file, "w", encoding="utf-8") as f:
                json.dump(all_results, f, indent=2, ensure_ascii=False)
            
            print(f"Saved final full-finetune model to: {full_phase_output_dir}")
            print(f"Final test results saved to: {all_results_file}")
            ft_model_path = full_phase_output_dir

        else:
            # 训练完成后，将最终模型复制到主目录
            final_phase_dir = f"{base_output_dir}_phase_{min(phase, args.bucket_number-1)}"
            print(f"\nCurriculum training completed. Final model saved to: {final_phase_dir}")
            ft_model_path = final_phase_dir

    # 非课程学习训练
    elif args.do_train:
        print("Starting standard training without curriculum learning")
        # 加载基础模型
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            device_map="auto",
            # attn_implementation="flash_attention_2",
            torch_dtype=tensor_data_type,
            quantization_config=bnb_config
        )
        
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        tokenizer.padding_side = 'left'
        model, tokenizer = setup_chat_format(model, tokenizer)
        
        # 训练参数
        training_args = TrainingArguments(
            output_dir=f'{args.output_folder}/{args.ft_model_id}',
            num_train_epochs=args.epoch,
            max_steps=args.max_steps,
            per_device_train_batch_size=2,  # 减少批大小以节省显存
            per_device_eval_batch_size=1,
            gradient_accumulation_steps=4,
            gradient_checkpointing=True,
            save_total_limit=1,
            # save_only_model=True,
            optim="adamw_torch", 
            eval_delay=args.eval_delay,
            logging_steps=50,
            eval_steps=50,
            save_steps=50,
            load_best_model_at_end=True,
            metric_for_best_model='weighted-f1',
            greater_is_better=True,
            eval_strategy='steps',
            logging_strategy='steps',
            save_strategy="steps",
            learning_rate=args.lr,
            bf16=True,
            tf32=False,
            max_grad_norm=0.3,
            warmup_ratio=0.03,
            lr_scheduler_type=args.lr_scheduler,
            push_to_hub=False,
            group_by_length=True,
            report_to="tensorboard",
        )

        trainer = SimplifiedTrainer(
            model=model,
            args=training_args,
            train_dataset=full_dataset,
            eval_dataset=valid_dataset,
            neftune_noise_alpha=5,
            peft_config=LoraConfig(
                lora_alpha=128,
                lora_dropout=0.05,
                r=args.lora_r,
                bias="none",
                target_modules="all-linear",
                task_type="CAUSAL_LM", 
            ),
            max_seq_length=args.max_seq_len,
            tokenizer=tokenizer,
            packing=False,
            dataset_kwargs={
                "add_special_tokens": False,
                "append_concat_token": False,
            }
        )

        print("\n==============================")
        print("TRAIN DATASET SANITY CHECK")
        print("==============================\n")

        sample = formatting_prompts_func(full_dataset[0])

        print(sample[-500:])
        
        trainer.train()
        trainer.save_model()
        tokenizer.save_pretrained(f'{args.output_folder}/{args.ft_model_id}')