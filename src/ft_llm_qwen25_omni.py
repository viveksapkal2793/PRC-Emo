import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["NCCL_P2P_DISABLE"] = "1"
os.environ["NCCL_IB_DISABLE"] = "1"
os.environ.setdefault("TORCH_CUDNN_V8_API_DISABLED", "1")

import argparse
import glob
import json
import random
import shutil
import warnings
from typing import Iterable, List, Optional, Tuple

import numpy as np
import torch
import av
from lightning import seed_everything
from peft import LoraConfig, PeftConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training
from sklearn.metrics import accuracy_score, classification_report, f1_score
from torch.utils.data import DataLoader, Dataset as TorchDataset
from tqdm import tqdm
from transformers import BitsAndBytesConfig, Trainer, TrainingArguments
from transformers import set_seed as transf_seed
from trl import set_seed as trl_seed

from reformat_data_ft_llm_combine import get_label_map, process

warnings.filterwarnings("ignore", category=DeprecationWarning)

try:
    from transformers import Qwen2_5OmniProcessor, Qwen2_5OmniThinkerForConditionalGeneration
except ImportError as exc:
    raise ImportError(
        "This script requires a transformers version that includes "
        "Qwen2_5OmniProcessor and Qwen2_5OmniThinkerForConditionalGeneration."
    ) from exc


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
    torch.backends.cudnn.enabled = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    trl_seed(seed)
    transf_seed(seed)


def extract_assistant_text(message):
    content = message.get("content", "")
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        text_parts = [item.get("text", "") for item in content if item.get("type") == "text"]
        return " ".join(text_parts).strip()
    return str(content).strip()


def normalize_prediction(prediction, label_space):
    pred = prediction.strip().lower()
    label_map = {label.lower(): label for label in label_space}
    if pred in label_map:
        return label_map[pred]

    for label in label_space:
        if label.lower() in pred:
            return label

    first_line = pred.splitlines()[0].strip() if pred else ""
    if first_line in label_map:
        return label_map[first_line]

    return prediction.strip()


def dataset_label_text(sample):
    return extract_assistant_text(sample["messages"][-1])


class JsonlMessageDataset(TorchDataset):
    def __init__(self, records):
        self.records = records

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        return self.records[idx]


def iter_media_paths_from_content(content) -> Iterable[Tuple[str, str]]:
    if isinstance(content, list):
        for item in content:
            if not isinstance(item, dict):
                continue
            if item.get("type") == "video" and item.get("video"):
                yield "video", item["video"]
            elif item.get("type") == "audio" and item.get("audio"):
                yield "audio", item["audio"]


def extract_media_paths(sample) -> List[Tuple[str, str]]:
    media_paths = []
    for message in sample.get("messages", []):
        media_paths.extend(iter_media_paths_from_content(message.get("content")))
    if sample.get("video_path"):
        media_paths.append(("video", sample["video_path"]))
    if sample.get("audio_path"):
        media_paths.append(("audio", sample["audio_path"]))

    deduped = []
    seen = set()
    for media_type, media_path in media_paths:
        key = (media_type, media_path)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(key)
    return deduped


def validate_video_file(video_path: str, media_cache: dict) -> Tuple[bool, Optional[str]]:
    cache_key = ("video", video_path)
    if cache_key in media_cache:
        return media_cache[cache_key]

    if not os.path.exists(video_path):
        result = (False, f"missing video file: {video_path}")
        media_cache[cache_key] = result
        return result

    try:
        container = av.open(video_path)
        container.close()
        result = (True, None)
    except Exception as exc:
        result = (False, f"{type(exc).__name__}: {exc}")

    media_cache[cache_key] = result
    return result


def validate_audio_file(audio_path: str, media_cache: dict) -> Tuple[bool, Optional[str]]:
    cache_key = ("audio", audio_path)
    if cache_key in media_cache:
        return media_cache[cache_key]

    if not os.path.exists(audio_path):
        result = (False, f"missing audio file: {audio_path}")
    else:
        result = (True, None)

    media_cache[cache_key] = result
    return result


def filter_invalid_media_records(records, split_name: str, validate_media: bool = True, media_cache: Optional[dict] = None):
    if not validate_media:
        return records

    media_cache = media_cache if media_cache is not None else {}
    filtered_records = []
    skipped_records = []

    for sample in records:
        sample_media = extract_media_paths(sample)
        is_valid = True
        failure_reason = None

        for media_type, media_path in sample_media:
            if media_type == "video":
                valid, reason = validate_video_file(media_path, media_cache)
            else:
                valid, reason = validate_audio_file(media_path, media_cache)

            if not valid:
                is_valid = False
                failure_reason = f"{media_type} {reason}"
                break

        if is_valid:
            filtered_records.append(sample)
        else:
            skipped_records.append(
                (
                    sample.get("conversation_id"),
                    sample.get("utterance_id"),
                    failure_reason,
                )
            )

    if skipped_records:
        print(f"[media validation] Skipped {len(skipped_records)} invalid samples from {split_name}.")
        for conversation_id, utterance_id, reason in skipped_records[:10]:
            print(
                "[media validation] "
                f"{split_name} conversation_id={conversation_id} utterance_id={utterance_id}: {reason}"
            )
        if len(skipped_records) > 10:
            print(f"[media validation] ... and {len(skipped_records) - 10} more skipped samples.")

    return filtered_records


def load_jsonl_dataset(path, split_name: Optional[str] = None, validate_media: bool = False, media_cache: Optional[dict] = None):
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))

    records = filter_invalid_media_records(
        records,
        split_name=split_name or os.path.basename(path),
        validate_media=validate_media,
        media_cache=media_cache,
    )
    return JsonlMessageDataset(records)


def safe_video_sample_indices(metadata, num_frames=None, fps=None, **kwargs):
    total_num_frames = metadata.total_num_frames
    video_fps = metadata.fps

    if total_num_frames is None or total_num_frames <= 0:
        raise ValueError(f"Invalid video metadata.total_num_frames={total_num_frames}")

    if num_frames is None and fps is not None:
        if video_fps is None or video_fps <= 0:
            num_frames = 1
        else:
            num_frames = max(1, int(total_num_frames / video_fps * fps))

    if num_frames is not None:
        num_frames = max(1, min(int(num_frames), int(total_num_frames)))
        indices = np.arange(0, total_num_frames, total_num_frames / num_frames, dtype=int)
    else:
        indices = np.arange(0, total_num_frames, dtype=int)
    return indices


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

        random.shuffle(data)
        print(f"Curriculum step {step_index}: using {len(data)} samples")
        return JsonlMessageDataset(data)


class OmniEmotionDataCollator:
    def __init__(self, processor, video_fps=1.0, video_num_frames=1):
        self.processor = processor
        self.video_fps = video_fps
        self.video_num_frames = video_num_frames
        self._logged_bad_media = set()

    def _apply_batches(self, conversations, prompts):
        full_batch = self.processor.apply_chat_template(
            conversations,
            add_generation_prompt=False,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
            padding=True,
            num_frames=self.video_num_frames,
            load_audio_from_video=False,
            use_audio_in_video=False,
        )
        prompt_batch = self.processor.apply_chat_template(
            prompts,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
            padding=True,
            num_frames=self.video_num_frames,
            load_audio_from_video=False,
            use_audio_in_video=False,
        )
        return full_batch, prompt_batch

    def _log_bad_sample(self, feature, exc):
        sample_key = (
            feature.get("conversation_id"),
            feature.get("utterance_id"),
            tuple(extract_media_paths(feature)),
        )
        if sample_key in self._logged_bad_media:
            return
        self._logged_bad_media.add(sample_key)
        print(
            "[collator] Skipping invalid sample "
            f"conversation_id={feature.get('conversation_id')} "
            f"utterance_id={feature.get('utterance_id')} "
            f"media={extract_media_paths(feature)} "
            f"error={type(exc).__name__}: {exc}"
        )

    def __call__(self, features):
        conversations = [feature["messages"] for feature in features]
        prompts = [conversation[:-1] for conversation in conversations]
        try:
            full_batch, prompt_batch = self._apply_batches(conversations, prompts)
        except Exception as batch_exc:
            valid_features = []
            for feature in features:
                try:
                    self._apply_batches([feature["messages"]], [feature["messages"][:-1]])
                    valid_features.append(feature)
                except Exception as sample_exc:
                    self._log_bad_sample(feature, sample_exc)

            if not valid_features:
                raise RuntimeError(
                    "All samples in the current batch failed media loading. "
                    "Consider enabling media validation on dataset load to skip corrupt files earlier."
                ) from batch_exc

            conversations = [feature["messages"] for feature in valid_features]
            prompts = [conversation[:-1] for conversation in conversations]
            full_batch, prompt_batch = self._apply_batches(conversations, prompts)

        labels = full_batch["input_ids"].clone()
        labels[full_batch["attention_mask"] == 0] = -100
        prompt_lengths = prompt_batch["attention_mask"].sum(dim=1).tolist()
        for row_idx, prompt_len in enumerate(prompt_lengths):
            labels[row_idx, :prompt_len] = -100

        full_batch["labels"] = labels
        return full_batch


class MultimodalTrainer(Trainer):
    def __init__(self, *args, processor=None, label_space=None, video_fps=1.0, video_num_frames=1, generation_max_new_tokens=10, **kwargs):
        super().__init__(*args, **kwargs)
        self.processor = processor
        self.label_space = label_space or []
        self.video_fps = video_fps
        self.video_num_frames = video_num_frames
        self.generation_max_new_tokens = generation_max_new_tokens
        self._logged_eval_bad_media = set()

    def get_eval_dataloader(self, eval_dataset=None) -> DataLoader:
        dataset = eval_dataset if eval_dataset is not None else self.eval_dataset
        return DataLoader(dataset, batch_size=1, shuffle=False, collate_fn=lambda batch: batch[0])

    def evaluate(self, eval_dataset=None, ignore_keys=None, metric_key_prefix="eval"):
        dataset = eval_dataset if eval_dataset is not None else self.eval_dataset
        dataloader = self.get_eval_dataloader(dataset)
        model = self.model
        model.eval()

        all_preds = []
        all_labels = []
        all_raw_decoded = []

        for sample in tqdm(dataloader, desc=f"{metric_key_prefix}"):
            prompt_conversation = sample["messages"][:-1]
            gold_label = dataset_label_text(sample)
            try:
                inputs = self.processor.apply_chat_template(
                    prompt_conversation,
                    add_generation_prompt=True,
                    tokenize=True,
                    return_dict=True,
                    return_tensors="pt",
                    padding=True,
                    num_frames=self.video_num_frames,
                    load_audio_from_video=False,
                    use_audio_in_video=False,
                )
            except Exception as exc:
                sample_key = (
                    sample.get("conversation_id"),
                    sample.get("utterance_id"),
                    tuple(extract_media_paths(sample)),
                )
                if sample_key not in self._logged_eval_bad_media:
                    self._logged_eval_bad_media.add(sample_key)
                    print(
                        "[eval] Skipping invalid sample "
                        f"conversation_id={sample.get('conversation_id')} "
                        f"utterance_id={sample.get('utterance_id')} "
                        f"media={extract_media_paths(sample)} "
                        f"error={type(exc).__name__}: {exc}"
                    )
                continue
            inputs = {k: v.to(model.device) if torch.is_tensor(v) else v for k, v in inputs.items()}
            prompt_length = inputs["input_ids"].shape[1]

            with torch.no_grad():
                generated = model.generate(
                    **inputs,
                    max_new_tokens=self.generation_max_new_tokens,
                    do_sample=False,
                    eos_token_id=self.processor.tokenizer.eos_token_id,
                    pad_token_id=self.processor.tokenizer.pad_token_id,
                )

            generated_text = self.processor.batch_decode(
                generated[:, prompt_length:],
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )[0].strip()
            normalized_pred = normalize_prediction(generated_text, self.label_space)

            all_preds.append(normalized_pred)
            all_labels.append(gold_label)
            all_raw_decoded.append(generated_text)

        if not all_labels:
            raise RuntimeError(f"No valid evaluation samples remained for {metric_key_prefix}.")

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
        with open(
            f"{self.args.output_dir}/result_{metric_key_prefix}_step-{self.state.global_step}.json",
            "w",
            encoding="utf-8",
        ) as f:
            json.dump(result_data, f, indent=1, ensure_ascii=False)

        self.log(metrics)
        return metrics


def create_training_args(output_dir, num_train_epochs, args):
    return TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=num_train_epochs,
        max_steps=args.max_steps,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        gradient_checkpointing=True,
        save_total_limit=1,
        optim=args.optim,
        eval_delay=args.eval_delay,
        logging_steps=50,
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
        remove_unused_columns=False,
        report_to="tensorboard",
        dataloader_num_workers=0,
    )


def build_quant_config(tensor_dtype):
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=tensor_dtype,
    )


def load_processor(processor_path):
    processor = Qwen2_5OmniProcessor.from_pretrained(processor_path)
    processor.tokenizer.padding_side = "right"
    if processor.tokenizer.pad_token is None:
        processor.tokenizer.pad_token = processor.tokenizer.eos_token
    return processor


def load_base_model(model_path, tensor_dtype, bnb_config):
    model = Qwen2_5OmniThinkerForConditionalGeneration.from_pretrained(
        model_path,
        device_map="auto",
        torch_dtype=tensor_dtype,
        quantization_config=bnb_config,
    )
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()
    return model


def attach_lora(model, args):
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
    peft_config = LoraConfig(
        lora_alpha=128,
        lora_dropout=0.05,
        r=args.lora_r,
        bias="none",
        target_modules="all-linear",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()
    return model


def load_peft_model(model_dir, tensor_dtype, bnb_config):
    peft_config = PeftConfig.from_pretrained(model_dir)
    base_model = load_base_model(peft_config.base_model_name_or_path, tensor_dtype, bnb_config)
    base_model = prepare_model_for_kbit_training(base_model, use_gradient_checkpointing=True)
    model = PeftModel.from_pretrained(base_model, model_dir)
    model.train()
    return model


def build_trainer(model, processor, train_dataset, eval_dataset, training_args, label_space, args):
    return MultimodalTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=OmniEmotionDataCollator(
            processor,
            video_fps=args.video_fps,
            video_num_frames=args.video_num_frames,
        ),
        processor=processor,
        label_space=label_space,
        video_fps=args.video_fps,
        video_num_frames=args.video_num_frames,
        generation_max_new_tokens=args.generation_max_new_tokens,
    )


def maybe_generate_data(data_paths, args):
    if not args.re_gen_data:
        return

    args.multimodal_chat_format = True
    process(data_paths, args)


def save_results_json(result_path, key, metrics):
    if os.path.exists(result_path):
        with open(result_path, "r", encoding="utf-8") as f:
            all_results = json.load(f)
    else:
        all_results = {}
    all_results[key] = metrics
    with open(result_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fine-tune Qwen2.5 Omni Thinker on MELD emotion classification.")
    parser.add_argument("--do_train", action="store_true", default=False)
    parser.add_argument("--do_eval_test", action="store_true", default=False)
    parser.add_argument("--do_eval_dev", action="store_true", default=False)
    parser.add_argument("--ft_model_path", type=str, default=None)
    parser.add_argument("--ft_model_id", type=str, default=None)
    parser.add_argument("--prompting_type", type=str, default="ImplicitEmotion_V3")
    parser.add_argument("--base_model_id", type=str, required=True)
    parser.add_argument("--extract_prompting_llm_id", type=str, default="qwen_3_14b")
    parser.add_argument("--epoch", type=int, default=None)
    parser.add_argument("--max_steps", type=int, default=-1)
    parser.add_argument("--lr_scheduler", type=str, default="linear")
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--kshot", type=int, default=0)
    parser.add_argument("--lora_r", type=int, default=32)
    parser.add_argument("--eval_delay", type=int, default=100000)
    parser.add_argument("--window", type=int, default=5)
    parser.add_argument("--max_seq_len", type=int, default=None)
    parser.add_argument("--re_gen_data", action="store_true", default=False)
    parser.add_argument("--data_name", type=str, default="meld")
    parser.add_argument("--data_folder", type=str, default="./data/")
    parser.add_argument("--output_folder", type=str, default="./finetuned_llm/")
    parser.add_argument("--curriculum", action="store_true", default=False)
    parser.add_argument("--bucket_number", type=int, default=8)
    parser.add_argument("--curriculum_update_epochs", type=int, default=None)
    parser.add_argument("--video_fps", type=float, default=1.0)
    parser.add_argument("--video_num_frames", type=int, default=1)
    parser.add_argument("--generation_max_new_tokens", type=int, default=10)
    parser.add_argument("--per_device_train_batch_size", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4)
    parser.add_argument("--optim", type=str, default="adamw_torch_fused")
    parser.add_argument("--multimodal_chat_format", action="store_true", default=True)
    parser.add_argument("--skip_invalid_media", action="store_true", default=False)
    parser.add_argument("--meld_train_video_dir", type=str, default="/scratch/data/bikash_rs/Vivek/dataset/MELD/MELD.Raw/train_splits")
    parser.add_argument("--meld_valid_video_dir", type=str, default="/scratch/data/bikash_rs/Vivek/dataset/MELD/MELD.Raw/dev_splits_complete")
    parser.add_argument("--meld_test_video_dir", type=str, default="/scratch/data/bikash_rs/Vivek/dataset/MELD/MELD.Raw/output_repeated_splits_test")
    parser.add_argument("--meld_train_audio_dir", type=str, default="/scratch/data/bikash_rs/Vivek/dataset/MELD_audio/train")
    parser.add_argument("--meld_valid_audio_dir", type=str, default="/scratch/data/bikash_rs/Vivek/dataset/MELD_audio/dev")
    parser.add_argument("--meld_test_audio_dir", type=str, default="/scratch/data/bikash_rs/Vivek/dataset/MELD_audio/test")

    args, unknown = parser.parse_known_args()
    print(args)

    if args.data_name != "meld":
        raise ValueError("This first Omni script is currently wired for MELD only.")

    set_random_seed(args.seed)

    all_path_folder_preprocessed_data = [
        f"{args.data_folder}/{args.data_name}.{d_type}.{args.kshot}shot_w{args.window}_{args.prompting_type}_{args.extract_prompting_llm_id}_Aud_Vis_Omni.jsonl"
        for d_type in ["train", "valid", "test"]
    ]
    maybe_generate_data(all_path_folder_preprocessed_data, args)

    media_cache = {}
    full_dataset = load_jsonl_dataset(
        all_path_folder_preprocessed_data[0],
        split_name="train",
        validate_media=args.skip_invalid_media,
        media_cache=media_cache,
    )
    valid_dataset = load_jsonl_dataset(
        all_path_folder_preprocessed_data[1],
        split_name="valid",
        validate_media=args.skip_invalid_media,
        media_cache=media_cache,
    )
    test_dataset = load_jsonl_dataset(
        all_path_folder_preprocessed_data[2],
        split_name="test",
        validate_media=args.skip_invalid_media,
        media_cache=media_cache,
    )

    curriculum_manager = None
    if args.curriculum and args.do_train:
        curriculum_manager = CurriculumDataset(list(full_dataset), bucket_number=args.bucket_number, curriculum=True)

    label_space = get_label_map(args.data_name)
    tensor_dtype = torch.bfloat16
    bnb_config = build_quant_config(tensor_dtype)

    if not args.do_train:
        ft_model_path = f"{args.output_folder}/{args.ft_model_id}" if args.ft_model_path is None else args.ft_model_path
        processor = load_processor(ft_model_path)
        model = load_peft_model(ft_model_path, torch.float32, bnb_config)
        eval_args = create_training_args(ft_model_path, 1, args)
        trainer = build_trainer(model, processor, full_dataset, valid_dataset, eval_args, label_space, args)

        if args.do_eval_dev:
            dev_metrics = trainer.evaluate(valid_dataset, metric_key_prefix="dev")
            print(dev_metrics)
        if args.do_eval_test:
            test_metrics = trainer.evaluate(test_dataset, metric_key_prefix="test")
            print(test_metrics)
        raise SystemExit(0)

    if args.curriculum and args.curriculum_update_epochs is None:
        args.curriculum_update_epochs = max(1, (args.epoch or 1) // args.bucket_number)

    model_id = args.base_model_id
    base_output_dir = f"{args.output_folder}/{args.ft_model_id}"

    if args.curriculum:
        print("=" * 50)
        print("Starting curriculum learning with phased multimodal training")
        print("=" * 50)

        remaining_epochs = args.epoch or 0
        model = None
        trainer = None

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
                processor = load_processor(model_id)
                model = load_base_model(model_id, tensor_dtype, bnb_config)
                model = attach_lora(model, args)
            else:
                prev_phase_dir = f"{base_output_dir}_phase_{phase - 1}"
                del model
                del trainer
                torch.cuda.empty_cache()
                processor = load_processor(prev_phase_dir)
                model = load_peft_model(prev_phase_dir, tensor_dtype, bnb_config)

            trainer = build_trainer(model, processor, current_dataset, valid_dataset, training_args, label_space, args)
            trainer.train()
            trainer.save_model(phase_output_dir)
            processor.save_pretrained(phase_output_dir)

            if remaining_epochs <= 0:
                break

        if remaining_epochs > 0:
            full_phase_output_dir = f"{base_output_dir}_final_full_finetune"
            os.makedirs(full_phase_output_dir, exist_ok=True)
            last_phase_dir = f"{base_output_dir}_phase_{args.bucket_number - 1}"

            del model
            del trainer
            torch.cuda.empty_cache()

            processor = load_processor(last_phase_dir)
            model = load_peft_model(last_phase_dir, tensor_dtype, bnb_config)
            training_args = create_training_args(full_phase_output_dir, remaining_epochs, args)
            trainer = build_trainer(model, processor, full_dataset, valid_dataset, training_args, label_space, args)
            trainer.train()
            trainer.save_model(full_phase_output_dir)
            processor.save_pretrained(full_phase_output_dir)

            final_test_metrics = trainer.evaluate(test_dataset, metric_key_prefix="test_final_full")
            save_results_json(os.path.join(base_output_dir, "all_phase_test_results.json"), "final_full", final_test_metrics)
            ft_model_path = full_phase_output_dir
        else:
            ft_model_path = f"{base_output_dir}_phase_{min(phase, args.bucket_number - 1)}"

        phase_dirs = glob.glob(f"{base_output_dir}_phase_*")
        for phase_dir in phase_dirs:
            try:
                shutil.rmtree(phase_dir)
            except Exception as exc:
                print(f"Failed to delete {phase_dir}: {exc}")

        print(f"Training complete. Final checkpoint: {ft_model_path}")

    else:
        print("Starting standard multimodal training without curriculum learning")
        output_dir = f"{args.output_folder}/{args.ft_model_id}"
        os.makedirs(output_dir, exist_ok=True)

        processor = load_processor(model_id)
        model = load_base_model(model_id, tensor_dtype, bnb_config)
        model = attach_lora(model, args)

        training_args = create_training_args(output_dir, args.epoch, args)
        trainer = build_trainer(model, processor, full_dataset, valid_dataset, training_args, label_space, args)
        trainer.train()
        trainer.save_model(output_dir)
        processor.save_pretrained(output_dir)

        if args.do_eval_test:
            test_metrics = trainer.evaluate(test_dataset, metric_key_prefix="test")
            save_results_json(os.path.join(output_dir, "all_phase_test_results.json"), "test", test_metrics)
