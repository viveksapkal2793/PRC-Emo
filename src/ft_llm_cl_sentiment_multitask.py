import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["NCCL_P2P_DISABLE"] = "1"
os.environ["NCCL_IB_DISABLE"] = "1"

import argparse
import json
import random
import shutil
import sys
import warnings
from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
from datasets import Dataset, load_dataset
from lightning import seed_everything
from peft import LoraConfig, PeftConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training
from sklearn.metrics import accuracy_score, classification_report, f1_score
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, DataCollatorForSeq2Seq, TrainingArguments
from transformers import set_seed as transf_seed
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.trainer_utils import EvalLoopOutput
from trl import SFTTrainer, setup_chat_format
from trl import set_seed as trl_seed

from reformat_data_ft_llm_combine import process

warnings.filterwarnings("ignore", category=DeprecationWarning)

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(line_buffering=True, write_through=True)
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(line_buffering=True, write_through=True)


SENTIMENT_TO_ID = {
    "positive": 0,
    "neutral": 1,
    "negative": 2,
}
SENTIMENT_LOSS_WEIGHT = 0.3
SENTIMENT_HEAD_FILENAME = "aux_sentiment_head.bin"
SENTIMENT_CONFIG_FILENAME = "aux_sentiment_config.json"

torch.serialization.add_safe_globals([np.ndarray])
_original_torch_load = torch.load


def patched_torch_load(*args, **kwargs):
    kwargs.setdefault("weights_only", False)
    return _original_torch_load(*args, **kwargs)


torch.load = patched_torch_load


def set_random_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    seed_everything(seed=seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    trl_seed(seed)
    transf_seed(seed)


def extract_assistant_content(message):
    content = message.get("content", "")
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        text_parts = [item.get("text", "") for item in content if item.get("type") == "text"]
        return " ".join(text_parts).strip()
    return str(content).strip()


class CurriculumDataset:
    def __init__(self, dataset, bucket_number=8, curriculum=True):
        self.full_dataset = dataset
        self.bucket_number = bucket_number
        self.curriculum = curriculum
        self.buckets = None

        if self.curriculum:
            self.buckets = self._create_buckets()
            print(f"Created {len(self.buckets)} buckets for curriculum learning")
            for i, bucket in enumerate(self.buckets):
                print(
                    f"Bucket {i}: {len(bucket)} samples, difficulty range: "
                    f"{min([s['difficulty'] for s in bucket]):.4f} - "
                    f"{max([s['difficulty'] for s in bucket]):.4f}"
                )

    def _create_buckets(self):
        sorted_data = sorted(self.full_dataset, key=lambda x: x.get("difficulty", 0))
        bucket_size = len(sorted_data) // self.bucket_number
        buckets = []

        for i in range(self.bucket_number):
            start_idx = i * bucket_size
            end_idx = len(sorted_data) if i == self.bucket_number - 1 else (i + 1) * bucket_size
            buckets.append(sorted_data[start_idx:end_idx])
        return buckets

    def get_curriculum_dataset(self, step_index):
        if not self.curriculum or step_index >= len(self.buckets):
            data = list(self.full_dataset)
        else:
            data = []
            for i in range(step_index + 1):
                data.extend(self.buckets[i])

        print(f"Curriculum step {step_index}: using {len(data)} samples")
        random.shuffle(data)
        return Dataset.from_list(data)


@dataclass
class MultitaskCausalLMOutput(CausalLMOutputWithPast):
    sentiment_logits: Optional[torch.FloatTensor] = None
    sentiment_loss: Optional[torch.FloatTensor] = None
    lm_loss: Optional[torch.FloatTensor] = None


class AuxiliarySentimentWrapper(torch.nn.Module):
    def __init__(self, base_model, loss_weight=SENTIMENT_LOSS_WEIGHT):
        super().__init__()
        self.base_model = base_model
        self.loss_weight = loss_weight
        self.num_sentiment_labels = 3
        self.hidden_size = getattr(base_model.config, "hidden_size", None)
        if self.hidden_size is None and hasattr(base_model.config, "text_config"):
            self.hidden_size = getattr(base_model.config.text_config, "hidden_size", None)
        if self.hidden_size is None:
            raise ValueError("Could not determine hidden size for auxiliary sentiment head.")
        self.sentiment_head = torch.nn.Linear(self.hidden_size, self.num_sentiment_labels)
        self.config = base_model.config
        self.generation_config = getattr(base_model, "generation_config", None)

    def forward(self, sentiment_id=None, **kwargs):
        outputs = self.base_model(
            **kwargs,
            output_hidden_states=True,
            return_dict=True,
        )
        pooled_hidden = outputs.hidden_states[-1][:, -1, :]
        if self.sentiment_head.weight.device != pooled_hidden.device:
            self.sentiment_head = self.sentiment_head.to(pooled_hidden.device)
        sentiment_logits = self.sentiment_head(pooled_hidden)

        sentiment_loss = None
        if sentiment_id is not None:
            sentiment_loss = F.cross_entropy(sentiment_logits, sentiment_id)

        total_loss = outputs.loss
        if total_loss is not None and sentiment_loss is not None:
            total_loss = total_loss + self.loss_weight * sentiment_loss

        return MultitaskCausalLMOutput(
            loss=total_loss,
            logits=outputs.logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            sentiment_logits=sentiment_logits,
            sentiment_loss=sentiment_loss,
            lm_loss=outputs.loss,
        )

    def generate(self, *args, **kwargs):
        return self.base_model.generate(*args, **kwargs)

    def get_input_embeddings(self):
        return self.base_model.get_input_embeddings()

    def save_pretrained(self, save_directory, **kwargs):
        os.makedirs(save_directory, exist_ok=True)
        self.base_model.save_pretrained(save_directory, **kwargs)
        torch.save(self.sentiment_head.state_dict(), os.path.join(save_directory, SENTIMENT_HEAD_FILENAME))
        with open(os.path.join(save_directory, SENTIMENT_CONFIG_FILENAME), "w", encoding="utf-8") as f:
            json.dump(
                {
                    "hidden_size": self.hidden_size,
                    "num_sentiment_labels": self.num_sentiment_labels,
                    "loss_weight": self.loss_weight,
                },
                f,
                indent=2,
            )

    def load_auxiliary_head(self, model_dir):
        head_path = os.path.join(model_dir, SENTIMENT_HEAD_FILENAME)
        if os.path.exists(head_path):
            state_dict = torch.load(head_path, map_location="cpu")
            self.sentiment_head.load_state_dict(state_dict)


def print_trainable_parameters(model):
    trainable_params = 0
    all_params = 0
    for _, param in model.named_parameters():
        all_params += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    percent = 100 * trainable_params / max(1, all_params)
    print(
        f"trainable params: {trainable_params:,} || all params: {all_params:,} || "
        f"trainable%: {percent:.4f}"
    )


class MultitaskTrainer(SFTTrainer):
    def __init__(self, *args, max_seq_length=None, **kwargs):
        self.max_seq_length = max_seq_length
        self._debug_logged = False

        raw_train_dataset = kwargs.pop("train_dataset")
        raw_eval_dataset = kwargs.pop("eval_dataset")
        tokenizer_arg = kwargs["tokenizer"]
        self.tokenizer = tokenizer_arg

        train_dataset = self._process_train_dataset(raw_train_dataset)
        eval_dataset = self._process_eval_dataset(raw_eval_dataset)

        data_collator = DataCollatorForSeq2Seq(
            tokenizer=self.tokenizer,
            padding=True,
            return_tensors="pt",
        )

        super().__init__(
            *args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            data_collator=data_collator,
            formatting_func=None,
            dataset_text_field=None,
            packing=False,
            **kwargs,
        )

    def _process_train_dataset(self, dataset):
        return dataset.map(
            self._tokenize_train_sample,
            remove_columns=dataset.column_names,
            desc="Tokenizing train dataset",
        )

    def _process_eval_dataset(self, dataset):
        return dataset.map(
            self._tokenize_eval_sample,
            remove_columns=dataset.column_names,
            desc="Tokenizing eval dataset",
        )

    def _tokenize_train_sample(self, sample):
        full_text = self.tokenizer.apply_chat_template(
            sample["messages"],
            tokenize=False,
            add_generation_prompt=False,
        )
        prompt_text = self.tokenizer.apply_chat_template(
            sample["messages"][:-1],
            tokenize=False,
            add_generation_prompt=True,
        )

        full_tokens = self._tokenize_text(full_text)
        prompt_tokens = self._tokenize_text(prompt_text)

        input_ids = list(full_tokens["input_ids"])
        attention_mask = list(full_tokens["attention_mask"])
        labels = list(input_ids)
        prompt_length = min(len(prompt_tokens["input_ids"]), len(labels))
        labels[:prompt_length] = [-100] * prompt_length

        if all(label == -100 for label in labels):
            raise RuntimeError(
                f"Encountered a train sample with zero supervised label tokens. "
                f"conversation_id={sample.get('conversation_id')} utterance_id={sample.get('utterance_id')}"
            )

        sentiment_label = str(sample["sentiment_label"]).lower()
        if sentiment_label not in SENTIMENT_TO_ID:
            raise ValueError(f"Unexpected sentiment label: {sample['sentiment_label']}")

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "sentiment_id": SENTIMENT_TO_ID[sentiment_label],
        }

    def _tokenize_eval_sample(self, sample):
        prompt_text = self.tokenizer.apply_chat_template(
            sample["messages"][:-1],
            tokenize=False,
            add_generation_prompt=True,
        )
        prompt_tokens = self._tokenize_text(prompt_text)
        label_text = extract_assistant_content(sample["messages"][-1])
        label_tokens = self.tokenizer.encode(
            label_text,
            padding="max_length",
            max_length=10,
        )

        return {
            "input_ids": prompt_tokens["input_ids"],
            "attention_mask": prompt_tokens["attention_mask"],
            "labels": label_tokens,
        }

    def _tokenize_text(self, text):
        tokenize_kwargs = {
            "add_special_tokens": False,
        }
        if self.max_seq_length is not None:
            tokenize_kwargs["truncation"] = True
            tokenize_kwargs["max_length"] = self.max_seq_length
        return self.tokenizer(text, **tokenize_kwargs)

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        sentiment_id = inputs.pop("sentiment_id", None)
        outputs = model(**inputs, sentiment_id=sentiment_id)
        loss = outputs.loss

        if not self._debug_logged:
            self._debug_logged = True
            print("[multitask debug] sentiment labels:", sentiment_id.detach().cpu().tolist())
            print("[multitask debug] sentiment logits shape:", tuple(outputs.sentiment_logits.shape))
            print("[multitask debug] sentiment loss:", None if outputs.sentiment_loss is None else float(outputs.sentiment_loss.detach().cpu()))
            print("[multitask debug] total loss:", None if loss is None else float(loss.detach().cpu()))

        return (loss, outputs) if return_outputs else loss

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

        def post_process(str_out):
            try:
                gen_text = str_out.split("assistant\n")[-1].split("<|im_end|>")[0]
            except Exception:
                gen_text = "error"
            return gen_text

        with torch.no_grad():
            for inputs in tqdm(dataloader):
                inputs = self._prepare_inputs(inputs)
                generated_tokens = model.generate(
                    inputs["input_ids"],
                    attention_mask=inputs["attention_mask"],
                    max_new_tokens=10,
                    do_sample=False,
                    eos_token_id=self.tokenizer.eos_token_id,
                    pad_token_id=self.tokenizer.pad_token_id,
                    temperature=0.1,
                )
                labels = inputs.pop("labels")
                str_labels = self.tokenizer.batch_decode(labels, skip_special_tokens=True)
                raw_decoded = self.tokenizer.batch_decode(generated_tokens, skip_special_tokens=False)
                str_decoded = [post_process(e) for e in raw_decoded]
                all_preds += str_decoded
                all_labels += str_labels
                all_raw_decoded += raw_decoded

        num_samples = len(dataloader)
        f1_weighted = f1_score(all_labels, all_preds, average="weighted")
        f1_macro = f1_score(all_labels, all_preds, average="macro")
        accuracy = accuracy_score(all_labels, all_preds)

        print(set(all_preds))
        print(set(all_labels))
        print(classification_report(all_labels, all_preds, digits=4))

        metrics = {
            f"{metric_key_prefix}_weighted-f1": f1_weighted,
            f"{metric_key_prefix}_macro-f1": f1_macro,
            f"{metric_key_prefix}_accuracy": accuracy,
        }
        result_data = {
            "metrics": metrics,
            "detail_pred": list(zip(all_preds, all_labels, all_raw_decoded)),
        }
        with open(f"{self.args.output_dir}/result_{metric_key_prefix}_step-{self.state.global_step}.json", "w", encoding="utf-8") as f:
            json.dump(result_data, f, indent=1, ensure_ascii=False)

        return EvalLoopOutput(predictions=all_preds, label_ids=all_labels, metrics=metrics, num_samples=num_samples)


def create_training_args(output_dir, num_train_epochs, args):
    return TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=num_train_epochs,
        max_steps=args.max_steps,
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=4,
        gradient_checkpointing=True,
        save_total_limit=1,
        optim="adamw_torch_fused",
        eval_delay=args.eval_delay,
        logging_first_step=True,
        logging_steps=args.logging_steps,
        eval_steps=50,
        save_steps=50,
        load_best_model_at_end=True,
        metric_for_best_model="eval_weighted-f1",
        greater_is_better=True,
        eval_strategy="steps",
        logging_strategy="steps",
        save_strategy="steps",
        learning_rate=args.lr,
        bf16=True,
        tf32=False,
        max_grad_norm=0.3,
        warmup_ratio=0.03,
        lr_scheduler_type=args.lr_scheduler,
        push_to_hub=False,
        group_by_length=False,
        report_to="tensorboard",
        remove_unused_columns=False,
    )


def build_quant_config(tensor_dtype):
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=tensor_dtype,
    )


def load_tokenizer(model_path):
    tokenizer_local = AutoTokenizer.from_pretrained(model_path)
    tokenizer_local.padding_side = "left"
    return tokenizer_local


def load_base_model(model_path, tensor_dtype, bnb_config, offload_folder=None):
    load_kwargs = {
        "device_map": "auto",
        "torch_dtype": tensor_dtype,
        "quantization_config": bnb_config,
    }
    if offload_folder is not None:
        load_kwargs["offload_folder"] = offload_folder

    model = AutoModelForCausalLM.from_pretrained(model_path, **load_kwargs)
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()
    return model


def build_lora_config(args):
    return LoraConfig(
        lora_alpha=128,
        lora_dropout=0.05,
        r=args.lora_r,
        bias="none",
        target_modules="all-linear",
        task_type="CAUSAL_LM",
    )


def attach_lora_multitask_model(base_model, args):
    base_model = prepare_model_for_kbit_training(base_model, use_gradient_checkpointing=True)
    peft_model = get_peft_model(base_model, build_lora_config(args))
    peft_model.train()
    return AuxiliarySentimentWrapper(peft_model)


def load_multitask_model(model_dir, tensor_dtype, bnb_config, offload_folder=None):
    peft_config = PeftConfig.from_pretrained(model_dir)
    tokenizer_local = load_tokenizer(model_dir)
    base_model = load_base_model(peft_config.base_model_name_or_path, tensor_dtype, bnb_config, offload_folder=offload_folder)
    base_model, tokenizer_local = setup_chat_format(base_model, tokenizer_local)
    base_model = prepare_model_for_kbit_training(base_model, use_gradient_checkpointing=True)
    base_model.enable_input_require_grads()
    peft_model = PeftModel.from_pretrained(
        base_model,
        model_dir,
        offload_folder=offload_folder,
        is_trainable=True,
    )
    peft_model.train()
    model = AuxiliarySentimentWrapper(peft_model)
    model.load_auxiliary_head(model_dir)
    return model, tokenizer_local


def build_trainer(model, tokenizer_local, train_dataset, eval_dataset, training_args, args):
    return MultitaskTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer_local,
        max_seq_length=args.max_seq_len,
        peft_config=None,
        neftune_noise_alpha=5,
    )


def maybe_generate_data(data_paths, args):
    if args.re_gen_data:
        process(data_paths, args)


def save_results_json(result_path, key, metrics):
    result_dir = os.path.dirname(result_path)
    if result_dir:
        os.makedirs(result_dir, exist_ok=True)
    if os.path.exists(result_path):
        with open(result_path, "r", encoding="utf-8") as f:
            all_results = json.load(f)
    else:
        all_results = {}
    all_results[key] = metrics
    with open(result_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fine-tune PRC-Emo Qwen with auxiliary sentiment multitask learning.")
    parser.add_argument("--do_train", action="store_true", default=False)
    parser.add_argument("--do_eval_test", action="store_true", default=False)
    parser.add_argument("--do_eval_dev", action="store_true", default=False)
    parser.add_argument("--ft_model_path", type=str, default=None)
    parser.add_argument("--ft_model_id", type=str, default=None)
    parser.add_argument("--prompting_type", type=str, default="ImplicitEmotion_V3")
    parser.add_argument("--base_model_id", type=str, required=True)
    parser.add_argument("--extract_prompting_llm_id", type=str, default="qwen_3_14b")
    parser.add_argument("--epoch", type=int, default=None)
    parser.add_argument("--max_steps", type=int, default=None)
    parser.add_argument("--lr_scheduler", type=str, default="linear")
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--kshot", type=int, default=0)
    parser.add_argument("--lora_r", type=int, default=32)
    parser.add_argument("--eval_delay", type=int, default=200)
    parser.add_argument("--window", type=int, default=5)
    parser.add_argument("--max_seq_len", type=int, default=None)
    parser.add_argument("--re_gen_data", action="store_true", default=False)
    parser.add_argument("--data_name", type=str, default="meld")
    parser.add_argument("--data_folder", type=str, default="./data/")
    parser.add_argument("--output_folder", type=str, default="./finetuned_llm/")
    parser.add_argument("--curriculum", action="store_true", default=False)
    parser.add_argument("--bucket_number", type=int, default=8)
    parser.add_argument("--curriculum_update_epochs", type=int, default=None)
    parser.add_argument("--logging_steps", type=int, default=1)
    args, unknown = parser.parse_known_args()

    print(args)
    set_random_seed(args.seed)

    all_path_folder_preprocessed_data = [
        f"{args.data_folder}/{args.data_name}.{d_type}.{args.kshot}shot_w{args.window}_{args.prompting_type}_{args.extract_prompting_llm_id}_Vis.jsonl"
        for d_type in ["train", "valid", "test"]
    ]
    maybe_generate_data(all_path_folder_preprocessed_data, args)

    full_dataset = load_dataset("json", data_files=all_path_folder_preprocessed_data[0], split="train", cache_dir=f"{args.output_folder}/{args.ft_model_id}")
    valid_dataset = load_dataset("json", data_files=all_path_folder_preprocessed_data[1], split="train", cache_dir=f"{args.output_folder}/{args.ft_model_id}")
    test_dataset = load_dataset("json", data_files=all_path_folder_preprocessed_data[2], split="train", cache_dir=f"{args.output_folder}/{args.ft_model_id}")

    curriculum_manager = None
    if args.curriculum and args.do_train:
        curriculum_manager = CurriculumDataset(list(full_dataset), bucket_number=args.bucket_number, curriculum=True)

    tensor_dtype = torch.bfloat16
    bnb_config = build_quant_config(tensor_dtype)
    model_id = args.base_model_id
    base_output_dir = f"{args.output_folder}/{args.ft_model_id}"

    if args.curriculum and args.curriculum_update_epochs is None:
        args.curriculum_update_epochs = max(1, (args.epoch or 1) // args.bucket_number)

    if not args.do_train:
        ft_model_path = f"{args.output_folder}/{args.ft_model_id}" if args.ft_model_path is None else args.ft_model_path
        tokenizer = load_tokenizer(ft_model_path)
        model, tokenizer = load_multitask_model(ft_model_path, torch.float32, bnb_config)
        eval_args = create_training_args(ft_model_path, 1, args)
        trainer = build_trainer(model, tokenizer, full_dataset, valid_dataset, eval_args, args)
        if args.do_eval_dev:
            print(trainer.evaluate(metric_key_prefix="dev"))
        if args.do_eval_test:
            print(trainer.evaluate(test_dataset, metric_key_prefix="test"))
        raise SystemExit(0)

    if args.do_train and args.curriculum:
        print("=" * 50)
        print("Starting curriculum learning with phased training")
        print(f"Total buckets: {args.bucket_number}")
        print(f"Epochs per phase: {args.curriculum_update_epochs}")
        print("=" * 50)

        remaining_epochs = args.epoch or 0
        model = None
        trainer = None
        tokenizer = None

        offload_folder = os.path.join(base_output_dir, "offload")
        os.makedirs(offload_folder, exist_ok=True)

        for phase in range(args.bucket_number):
            current_dataset = curriculum_manager.get_curriculum_dataset(phase)
            phase_output_dir = f"{base_output_dir}_phase_{phase}"
            os.makedirs(phase_output_dir, exist_ok=True)

            current_epochs = min(args.curriculum_update_epochs, remaining_epochs) if remaining_epochs > 0 else 0
            remaining_epochs -= current_epochs
            if current_epochs <= 0:
                continue

            training_args = create_training_args(phase_output_dir, current_epochs, args)

            if phase == 0:
                tokenizer = load_tokenizer(model_id)
                base_model = load_base_model(model_id, tensor_dtype, bnb_config)
                base_model, tokenizer = setup_chat_format(base_model, tokenizer)
                model = attach_lora_multitask_model(base_model, args)
            else:
                prev_phase_dir = f"{base_output_dir}_phase_{phase - 1}"
                del model
                del trainer
                torch.cuda.empty_cache()
                model, tokenizer = load_multitask_model(prev_phase_dir, tensor_dtype, bnb_config, offload_folder=offload_folder)

            print_trainable_parameters(model)
            trainer = build_trainer(model, tokenizer, current_dataset, valid_dataset, training_args, args)
            trainer.train()
            trainer.save_model(phase_output_dir)
            tokenizer.save_pretrained(phase_output_dir)

            if remaining_epochs <= 0:
                break

        if remaining_epochs > 0:
            full_phase_output_dir = f"{base_output_dir}_final_full_finetune"
            os.makedirs(full_phase_output_dir, exist_ok=True)
            last_phase_dir = f"{base_output_dir}_phase_{args.bucket_number - 1}"

            del model
            del trainer
            torch.cuda.empty_cache()

            model, tokenizer = load_multitask_model(last_phase_dir, tensor_dtype, bnb_config, offload_folder=offload_folder)
            training_args = create_training_args(full_phase_output_dir, remaining_epochs, args)
            print_trainable_parameters(model)
            trainer = build_trainer(model, tokenizer, full_dataset, valid_dataset, training_args, args)
            trainer.train()
            trainer.save_model(full_phase_output_dir)
            tokenizer.save_pretrained(full_phase_output_dir)

            test_results = trainer.evaluate(test_dataset, metric_key_prefix="test_final_full")
            save_results_json(os.path.join(base_output_dir, "all_phase_test_results.json"), "final_full", test_results)
            ft_model_path = full_phase_output_dir
        else:
            ft_model_path = f"{base_output_dir}_phase_{min(phase, args.bucket_number - 1)}"

        phase_dirs = [d for d in os.listdir(args.output_folder) if d.startswith(f"{args.ft_model_id}_phase_")]
        for phase_dir in phase_dirs:
            shutil.rmtree(os.path.join(args.output_folder, phase_dir), ignore_errors=True)
        shutil.rmtree(offload_folder, ignore_errors=True)
        print(f"Training complete. Final checkpoint: {ft_model_path}")

    elif args.do_train:
        print("Starting standard training without curriculum learning")
        output_dir = f"{args.output_folder}/{args.ft_model_id}"
        os.makedirs(output_dir, exist_ok=True)

        tokenizer = load_tokenizer(model_id)
        base_model = load_base_model(model_id, tensor_dtype, bnb_config)
        base_model, tokenizer = setup_chat_format(base_model, tokenizer)
        model = attach_lora_multitask_model(base_model, args)
        print_trainable_parameters(model)

        training_args = create_training_args(output_dir, args.epoch, args)
        trainer = build_trainer(model, tokenizer, full_dataset, valid_dataset, training_args, args)
        trainer.train()
        trainer.save_model(output_dir)
        tokenizer.save_pretrained(output_dir)

        if args.do_eval_test:
            test_metrics = trainer.evaluate(test_dataset, metric_key_prefix="test")
            save_results_json(os.path.join(output_dir, "all_phase_test_results.json"), "test", test_metrics)
