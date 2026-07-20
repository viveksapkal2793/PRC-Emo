import argparse
import json
import os
import random
import re
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from tqdm.auto import tqdm

from common.dataset import BalancedPairBatchSampler, TextEmotionDataset, labels_to_ids, load_samples
from common.lora_utils import (
    apply_lora_to_embedding_model,
    encode_batch,
    embed_from_loader,
    embed_texts,
    load_embedding_model,
    load_lora_adapter_from_checkpoint,
)
from common.losses import memory_bank_local_supcon_loss, supervised_contrastive_loss
from common.metrics import make_classification_report, make_confusion_matrix
from common.models import EmotionMLP, FusionClassifier, ProjectionSupConClassifier
from common.prompt import build_embedding_input, parse_prompt_sample
from common.prototype import PrototypeManager, SamplePrototypeSupConLoss
from common.trainer import predict_mlp, train_step_prediction, train_mlp, train_frozen_prototype_mlp, train_lora_contrastive, train_proj_supcon, load_classifier_state
from common.types import EmotionSample
from common.utils import normalize_text, section_between
from common.memory import ClassBalancedMemoryBank
from common.visualization import save_embedding_analysis

MELD_LABELS = ["neutral", "surprise", "fear", "sadness", "joy", "disgust", "anger"]
IEMOCAP_LABELS = ["happy", "sad", "neutral", "angry", "excited", "frustrated"]
UTTERANCE_TASK_INSTRUCTION = "Represent the emotional state of the utterance."
CONTEXT_TASK_INSTRUCTION = "Represent the emotional state and conversational behavior of the target utterance."


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def print_embedding_debug(
    samples: Sequence[EmotionSample],
    embeddings: torch.Tensor,
    limit: int,
    prefix: str,
    preds: Optional[Sequence[str]] = None,
) -> None:
    count = min(limit, len(samples))
    for idx in range(count):
        print("=" * 100)
        print(f"{prefix} sample {idx}")
        print(samples[idx].text)
        print(f"Embedding dimension: {tuple(embeddings[idx].shape)}")
        print(f"True label: {samples[idx].label}")
        if preds is not None:
            print(f"Predicted label: {preds[idx]}")


def print_prompt_debug(samples: Sequence[EmotionSample], limit: int, prefix: str) -> None:
    count = min(limit, len(samples))
    for idx in range(count):
        print("=" * 100)
        print(f"{prefix} prompt {idx}")
        print(samples[idx].text)
        print(f"True label: {samples[idx].label}")


def get_section_suffix(args: argparse.Namespace) -> str:
    section_bits = []
    if getattr(args, "proj_supcon", False):
        section_bits.append("proj_supcon")
    if args.lora_contrastive_finetune:
        section_bits.append("lora_supcon")
    if getattr(args, "prototype_learning", False):
        section_bits.append("prototype")
    for name, enabled in [
        ("context", args.include_conversation_context or args.include_target_speaker),
        ("explicit", args.include_explicit_emotion),
        ("implicit", args.include_implicit_emotion),
        ("speaker", args.include_speaker_description),
        ("visual", args.include_visual_description),
        ("audio", args.include_audio_description),
        ("llm_aud_vis", args.include_llm_aud_vis_desc),
        ("semantic", args.include_semantic_contrastive_cues),
        ("refs", args.include_reference_similar_emotions),
    ]:
        if enabled:
            section_bits.append(name)
    return "_".join(section_bits) if section_bits else "utt"


def get_model_tag(args: argparse.Namespace) -> str:
    path_text = str(args.model_path).lower()
    if "f2llm" in path_text:
        return "f2llm_4b"
    if "0_6b" in path_text or "0.6b" in path_text or "0-6b" in path_text:
        return "qwen3_embedding_0_6b"
    if "4b" in path_text:
        return "qwen3_embedding_4b"
    if "8b" in path_text:
        return "qwen3_embedding_8b"
    return Path(args.model_path).name.replace("-", "_").replace(".", "_")


def get_timestamp_suffix() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def get_run_stem(args: argparse.Namespace) -> str:
    return f"{get_model_tag(args)}_{args.dataset}_mlp_{args.num_labels}cls_{get_section_suffix(args)}"


def save_checkpoint(model: nn.Module, args: argparse.Namespace, labels: Sequence[str], embedding_model=None) -> Path:
    save_dir = Path(args.output_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    run_stem = f"{get_run_stem(args)}_{get_timestamp_suffix()}"
    filename = f"{run_stem}.pt"
    path = save_dir / filename
    lora_adapter_path = None
    if args.lora_contrastive_finetune and embedding_model is not None and hasattr(embedding_model, "save_pretrained"):
        lora_adapter_path = save_dir / f"{run_stem}_lora_adapter"
        embedding_model.save_pretrained(lora_adapter_path)
        print(f"Saved LoRA adapter to {lora_adapter_path}")
    model_type = "mlp"
    prototype_vectors = None
    prototype_labels = None
    alpha_param = None
    if isinstance(model, ProjectionSupConClassifier):
        model_type = "proj_supcon"
    elif isinstance(model, FusionClassifier):
        model_type = "prototype_fusion"
        prototype_vectors = model.prototype_classifier.prototype_vectors.detach().cpu()
        prototype_labels = model.prototype_classifier.prototype_labels.detach().cpu()
        alpha_param = model.alpha_param.detach().cpu()
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "model_type": model_type,
            "embedding_dim": args.embedding_dim,
            "num_labels": args.num_labels,
            "labels": list(labels),
            "args": vars(args),
            "lora_adapter_path": str(lora_adapter_path) if lora_adapter_path is not None else None,
            "prototype_vectors": prototype_vectors,
            "prototype_labels": prototype_labels,
            "alpha_param": alpha_param,
        },
        path,
    )
    print(f"Saved MLP checkpoint to {path}")
    return path


def load_checkpoint(path: Path, args: argparse.Namespace, labels: Sequence[str]) -> nn.Module:
    checkpoint = torch.load(path, map_location="cpu")
    embedding_dim = int(checkpoint.get("embedding_dim", args.embedding_dim))
    num_labels = int(checkpoint.get("num_labels", args.num_labels))
    model_type = checkpoint.get("model_type", "proj_supcon" if args.proj_supcon else "mlp")
    if model_type == "proj_supcon":
        model = ProjectionSupConClassifier(embedding_dim, num_labels)
    elif model_type == "prototype_fusion":
        fixed_alpha = checkpoint.get("args", {}).get("prototype_fixed_alpha")
        model = FusionClassifier(
            embedding_dim,
            num_labels,
            args.dropout,
            checkpoint.get("args", {}).get("prototype_temperature", args.prototype_temperature),
            fixed_alpha,
        )
    else:
        model = EmotionMLP(embedding_dim, num_labels, args.dropout)
    load_classifier_state(model, checkpoint["model_state_dict"])
    if isinstance(model, FusionClassifier):
        prototype_vectors = checkpoint.get("prototype_vectors")
        prototype_labels = checkpoint.get("prototype_labels")
        if prototype_vectors is not None and prototype_labels is not None:
            model.set_prototypes(prototype_vectors, prototype_labels)
        alpha_param = checkpoint.get("alpha_param")
        if alpha_param is not None:
            model.alpha_param.data.copy_(alpha_param)
    device = torch.device("cuda" if torch.cuda.is_available() and not args.force_cpu_mlp else "cpu")
    model.to(device)
    checkpoint_labels = checkpoint.get("labels")
    if checkpoint_labels and list(checkpoint_labels) != list(labels):
        print(f"Warning: checkpoint labels {checkpoint_labels} differ from current labels {list(labels)}")
    return model


def evaluate_split(
    model: nn.Module,
    samples: Sequence[EmotionSample],
    embeddings: torch.Tensor,
    y_true: torch.Tensor,
    labels: Sequence[str],
    args: argparse.Namespace,
    split_name: str,
) -> None:
    loss, acc, pred_ids = predict_mlp(model, embeddings, y_true, args)
    y_true_np = y_true.numpy()
    pred_labels = [labels[idx] for idx in pred_ids]
    print_embedding_debug(samples, embeddings, args.debug_samples, f"{split_name} evaluation", pred_labels)
    print(f"\n{split_name} loss: {loss:.4f} accuracy: {acc:.4f}")
    print(f"\n{split_name} confusion matrix:")
    print(make_confusion_matrix(y_true_np, pred_ids, len(labels)))
    print(f"\n{split_name} classification report:")
    print(make_classification_report(y_true_np, pred_ids, labels))


def get_labels(args: argparse.Namespace) -> List[str]:
    if args.label_names:
        labels = [label.strip().lower() for label in args.label_names.split(",") if label.strip()]
    elif args.dataset == "iemocap":
        labels = IEMOCAP_LABELS
    else:
        labels = MELD_LABELS
    if args.num_labels is None:
        args.num_labels = 6 if args.dataset == "iemocap" else 7
    if args.num_labels != len(labels):
        raise ValueError(f"--num_labels={args.num_labels} but label set has {len(labels)} labels: {labels}")
    return labels


def infer_embedding_dim_from_model_path(model_path: str, explicit_embedding_dim: Optional[int] = None) -> int:
    if explicit_embedding_dim is not None:
        return int(explicit_embedding_dim)

    path_text = str(model_path).lower()
    if "0_6b" in path_text or "0.6b" in path_text or "0-6b" in path_text:
        return 1024
    if "f2llm" in path_text or "4b" in path_text:
        return 2560
    if "8b" in path_text:
        return 4096
    return 4096


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Qwen3-Embedding-8B + MLP emotion classifier.")
    parser.add_argument("--mode", choices=["train_eval", "train", "eval"], default="train_eval")
    parser.add_argument("--dataset", choices=["meld", "iemocap"], default="meld")
    parser.add_argument("--model_path", default="/scratch/data/bikash_rs/Vivek/PRC-Emo/models/qwen_3_embedding_8b")
    parser.add_argument("--train_file", default="data/meld.train.0shot_w5_ImplicitEmotion_V3_qwen_3_14b.jsonl")
    parser.add_argument("--valid_file", default="data/meld.valid.0shot_w5_ImplicitEmotion_V3_qwen_3_14b.jsonl")
    parser.add_argument("--test_file", default="data/meld.test.0shot_w5_ImplicitEmotion_V3_qwen_3_14b.jsonl")
    parser.add_argument("--checkpoint", default="")
    parser.add_argument("--output_dir", default="/scratch/data/bikash_rs/Vivek/PRC-Emo/mlp")
    parser.add_argument("--label_names", default="", help="Comma-separated labels in class-id order.")
    parser.add_argument("--num_labels", type=int, default=None)
    parser.add_argument(
        "--embedding_dim",
        type=int,
        default=None,
        help="Embedding dimension override. Defaults to the width inferred from --model_path.",
    )
    parser.add_argument("--context_window", type=int, default=5, help="Used only for non-prompt dialogue JSON files.")

    parser.add_argument("--include_explicit_emotion", action="store_true")
    parser.add_argument("--include_implicit_emotion", action="store_true")
    parser.add_argument(
        "--include_conversation_context",
        action="store_true",
        help="Include Conversation Context and Target Speaker, and use the conversational-behavior instruction.",
    )
    parser.add_argument("--include_target_speaker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--include_speaker_description", action="store_true")
    parser.add_argument("--include_visual_description", action="store_true")
    parser.add_argument("--include_audio_description", action="store_true")
    parser.add_argument(
        "--include_llm_aud_vis_desc",
        dest="include_llm_aud_vis_desc",
        action="store_true",
        help="Include the LLM-generated audio and visual description block.",
    )
    parser.add_argument("--include_semantic_contrastive_cues", action="store_true")
    parser.add_argument("--include_reference_similar_emotions", action="store_true")

    parser.add_argument("--load_in_4bit", action="store_true")
    parser.add_argument("--load_in_8bit", action="store_true")
    parser.add_argument("--attn_implementation", default="", help="Example: flash_attention_2")
    parser.add_argument("--max_length", type=int, default=8192)
    parser.add_argument("--embed_batch_size", type=int, default=4)
    parser.add_argument("--mlp_batch_size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-2)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--debug_samples", type=int, default=3)
    parser.add_argument("--force_cpu_mlp", action="store_true")
    parser.add_argument("--l2_normalize", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--embedding_analysis",
        action="store_true",
        help="Save t-SNE, PCA, and UMAP embedding analysis plots for train/valid/test splits.",
    )
    parser.add_argument(
        "--lora_contrastive_finetune",
        action="store_true",
        help="Fine-tune the embedding model with LoRA using CE + supervised contrastive loss.",
    )
    parser.add_argument("--supcon_lambda", type=float, default=0.1)
    parser.add_argument("--supcon_temperature", type=float, default=0.07)
    parser.add_argument("--use_memory_bank_supcon", action="store_true")
    parser.add_argument("--memory_per_class", type=int, default=128)
    parser.add_argument("--top_k_pos", type=int, default=2)
    parser.add_argument("--top_m_neg", type=int, default=5)
    parser.add_argument("--prototype_learning", action="store_true")
    parser.add_argument("--prototype_supcon_lambda", type=float, default=0.1)
    parser.add_argument("--prototype_temperature", type=float, default=0.07)
    parser.add_argument("--prototype_min_cluster_size", type=int, default=5)
    parser.add_argument("--prototype_fixed_alpha", type=float, default=None)
    parser.add_argument("--prototype_warmup_epochs", type=int, default=1)
    parser.add_argument("--prototype_recompute_every_epoch", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--prototype_hdbscan_metric", default="euclidean")
    parser.add_argument("--lora_lr", type=float, default=2e-5)
    parser.add_argument("--lora_batch_size", type=int, default=14)
    parser.add_argument("--lora_grad_accum_steps", type=int, default=8)
    parser.add_argument("--lora_max_length", type=int, default=512)
    parser.add_argument("--lora_r", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    parser.add_argument("--lora_target_modules", default="q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj")
    parser.add_argument("--gradient_checkpointing", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument(
        "--proj_supcon",
        action="store_true",
        help="Train a frozen embedding model with a projection head plus supervised contrastive loss and MLP classifier.",
    )
    args = parser.parse_args()
    if args.dataset == "iemocap":
        args.train_file = args.train_file.replace("meld", "iemocap")
        args.valid_file = args.valid_file.replace("meld", "iemocap")
        args.test_file = args.test_file.replace("meld", "iemocap")
    if args.lora_batch_size < 2:
        raise ValueError("--lora_batch_size must be at least 2.")
    if args.lora_grad_accum_steps < 1:
        raise ValueError("--lora_grad_accum_steps must be at least 1.")
    if args.include_llm_aud_vis_desc and (args.include_visual_description or args.include_audio_description):
        raise ValueError(
            "--include_llm_aud_vis_desc cannot be combined with --include_visual_description or --include_audio_description."
        )
    if args.proj_supcon and args.lora_contrastive_finetune:
        raise ValueError("--proj_supcon cannot be combined with --lora_contrastive_finetune.")
    if args.use_memory_bank_supcon and not args.lora_contrastive_finetune:
        raise ValueError("--use_memory_bank_supcon requires --lora_contrastive_finetune.")
    if args.prototype_learning and args.proj_supcon:
        raise ValueError("--prototype_learning cannot be combined with --proj_supcon.")
    if args.memory_per_class < 1:
        raise ValueError("--memory_per_class must be at least 1.")
    if args.top_k_pos < 1:
        raise ValueError("--top_k_pos must be at least 1.")
    if args.top_m_neg < 1:
        raise ValueError("--top_m_neg must be at least 1.")
    if args.prototype_fixed_alpha is not None and not (0.0 <= args.prototype_fixed_alpha <= 1.0):
        raise ValueError("--prototype_fixed_alpha must be between 0 and 1.")
    if args.prototype_warmup_epochs < 0:
        raise ValueError("--prototype_warmup_epochs must be non-negative.")
    args.embedding_dim = infer_embedding_dim_from_model_path(args.model_path, args.embedding_dim)
    if args.proj_supcon:
        args.l2_normalize = True
    return args


def main() -> None:
    args = parse_args()
    if args.load_in_4bit and args.load_in_8bit:
        raise ValueError("Choose only one of --load_in_4bit or --load_in_8bit.")

    set_seed(args.seed)
    labels = get_labels(args)
    label_to_id = {label: idx for idx, label in enumerate(labels)}

    root = Path.cwd()
    train_path = root / args.train_file if not Path(args.train_file).is_absolute() else Path(args.train_file)
    valid_path = root / args.valid_file if args.valid_file and not Path(args.valid_file).is_absolute() else Path(args.valid_file) if args.valid_file else None
    test_path = root / args.test_file if args.test_file and not Path(args.test_file).is_absolute() else Path(args.test_file) if args.test_file else None

    tokenizer, embedding_model = load_embedding_model(args)
    if args.lora_contrastive_finetune and args.mode in {"train_eval", "train"}:
        embedding_model = apply_lora_to_embedding_model(embedding_model, args)

    classifier = None
    analysis_payloads: Dict[str, Tuple[Sequence[EmotionSample], torch.Tensor, torch.Tensor]] = {}
    split_paths = {"train": train_path, "valid": valid_path, "test": test_path}

    def load_embed_split(split_name: str) -> Tuple[Sequence[EmotionSample], torch.Tensor, torch.Tensor]:
        split_path = split_paths[split_name]
        if split_path is None or not split_path.exists():
            raise FileNotFoundError(f"{split_name} file not found: {split_path}")
        samples = load_samples(split_path, args, labels)
        print(f"Loaded {len(samples)} {split_name} samples.")
        embeddings = embed_texts([s.text for s in samples], tokenizer, embedding_model, args, split_name)
        y = labels_to_ids(samples, label_to_id)
        return samples, embeddings, y

    if args.mode in {"train_eval", "train"}:
        train_samples = load_samples(train_path, args, labels)
        valid_samples = load_samples(valid_path, args, labels) if valid_path and valid_path.exists() else []
        print(f"Loaded {len(train_samples)} train samples and {len(valid_samples)} valid samples.")

        print_prompt_debug(train_samples, args.debug_samples, "train")
        if valid_samples:
            print_prompt_debug(valid_samples, args.debug_samples, "valid")

        train_y = labels_to_ids(train_samples, label_to_id)
        valid_y = labels_to_ids(valid_samples, label_to_id) if valid_samples else None

        if args.proj_supcon:
            train_embeddings = embed_texts([s.text for s in train_samples], tokenizer, embedding_model, args, "train")
            analysis_payloads["train"] = (train_samples, train_embeddings, train_y)
            print_embedding_debug(train_samples, train_embeddings, args.debug_samples, "train")

            valid_embeddings = None
            if valid_samples:
                valid_embeddings = embed_texts([s.text for s in valid_samples], tokenizer, embedding_model, args, "valid")
                analysis_payloads["valid"] = (valid_samples, valid_embeddings, valid_y)
                print_embedding_debug(valid_samples, valid_embeddings, args.debug_samples, "valid")

            classifier = train_proj_supcon(train_embeddings, train_y, valid_embeddings, valid_y, args)
        elif args.lora_contrastive_finetune:
            classifier = train_lora_contrastive(
                embedding_model,
                tokenizer,
                train_samples,
                train_y,
                valid_samples,
                valid_y,
                args,
            )
            train_embeddings = embed_texts([s.text for s in train_samples], tokenizer, embedding_model, args, "train")
            analysis_payloads["train"] = (train_samples, train_embeddings, train_y)
            print_embedding_debug(train_samples, train_embeddings, args.debug_samples, "train")
            if valid_samples:
                valid_embeddings = embed_texts([s.text for s in valid_samples], tokenizer, embedding_model, args, "valid")
                analysis_payloads["valid"] = (valid_samples, valid_embeddings, valid_y)
                print_embedding_debug(valid_samples, valid_embeddings, args.debug_samples, "valid")
        else:
            train_embeddings = embed_texts([s.text for s in train_samples], tokenizer, embedding_model, args, "train")
            analysis_payloads["train"] = (train_samples, train_embeddings, train_y)
            print_embedding_debug(train_samples, train_embeddings, args.debug_samples, "train")

            valid_embeddings = None
            if valid_samples:
                valid_embeddings = embed_texts([s.text for s in valid_samples], tokenizer, embedding_model, args, "valid")
                analysis_payloads["valid"] = (valid_samples, valid_embeddings, valid_y)
                print_embedding_debug(valid_samples, valid_embeddings, args.debug_samples, "valid")

            if args.prototype_learning:
                classifier = train_frozen_prototype_mlp(train_embeddings, train_y, valid_embeddings, valid_y, args)
            else:
                classifier = train_mlp(train_embeddings, train_y, valid_embeddings, valid_y, args)

        checkpoint_path = save_checkpoint(classifier, args, labels, embedding_model)
        args.checkpoint = str(checkpoint_path)

    if args.mode == "eval":
        if not args.checkpoint:
            raise ValueError("--checkpoint is required for --mode eval.")
        if not args.proj_supcon:
            embedding_model = load_lora_adapter_from_checkpoint(Path(args.checkpoint), embedding_model)
        classifier = load_checkpoint(Path(args.checkpoint), args, labels)

    if args.mode in {"train_eval", "eval"}:
        if test_path is None or not test_path.exists():
            raise FileNotFoundError(f"Test file not found: {test_path}")
        test_samples = load_samples(test_path, args, labels)
        print(f"Loaded {len(test_samples)} test samples.")
        test_embeddings = embed_texts([s.text for s in test_samples], tokenizer, embedding_model, args, "test")
        test_y = labels_to_ids(test_samples, label_to_id)
        analysis_payloads["test"] = (test_samples, test_embeddings, test_y)
        evaluate_split(classifier, test_samples, test_embeddings, test_y, labels, args, "test")

    if args.embedding_analysis:
        for split_name in ["train", "valid", "test"]:
            if split_name not in analysis_payloads:
                analysis_payloads[split_name] = load_embed_split(split_name)
        save_embedding_analysis(analysis_payloads, labels, args)


if __name__ == "__main__":
    main()
