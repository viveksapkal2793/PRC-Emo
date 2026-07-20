import argparse
import csv
import random
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torchaudio

from common.dataset import AudioCollator, AudioEmotionDataset
from common.lora_utils import embed_from_loader, load_lora_adapter_from_checkpoint
from common.metrics import make_classification_report, make_confusion_matrix
from common.models import EmotionMLP, FusionClassifier, ProjectionSupConClassifier
from common.trainer import (
    load_classifier_state,
    predict_mlp,
    train_frozen_prototype_mlp,
    train_lora_contrastive,
    train_mlp,
    train_proj_supcon,
)
from common.types import EmotionSample
from common.wavlm_lora_utils import apply_lora_to_wavlm_embedding_model
from common.wavlm_wrapper import WavLMEmbeddingWrapper

MELD_LABELS = ["neutral", "surprise", "fear", "sadness", "joy", "disgust", "anger"]
IEMOCAP_LABELS = ["happy", "sad", "neutral", "angry", "excited", "frustrated"]
MELD_METADATA = {
    "train": "/scratch/data/bikash_rs/Vivek/dataset/MELD/MELD.Raw/train_sent_emo.csv",
    "valid": "/scratch/data/bikash_rs/Vivek/dataset/MELD/MELD.Raw/dev_sent_emo.csv",
    "test": "/scratch/data/bikash_rs/Vivek/dataset/MELD/MELD.Raw/test_sent_emo.csv",
}
MELD_AUDIO_DIRS = {
    "train": "/scratch/data/bikash_rs/Vivek/dataset/MELD_audio/train",
    "valid": "/scratch/data/bikash_rs/Vivek/dataset/MELD_audio/dev",
    "test": "/scratch/data/bikash_rs/Vivek/dataset/MELD_audio/test",
}
IEMOCAP_METADATA = {
    "train": "/scratch/data/bikash_rs/Vivek/PRC-Emo/iemocap_train_opensmile_utterance_wise.csv",
    "valid": "/scratch/data/bikash_rs/Vivek/PRC-Emo/iemocap_valid_opensmile_utterance_wise.csv",
    "test": "/scratch/data/bikash_rs/Vivek/PRC-Emo/iemocap_test_opensmile_utterance_wise.csv",
}


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_model_tag(args: argparse.Namespace) -> str:
    path_text = str(args.model_path).lower()
    if "wavlm" in path_text and "large" in path_text:
        return "wavlm_large"
    return Path(args.model_path).name.replace("-", "_").replace(".", "_")


def get_section_suffix(args: argparse.Namespace) -> str:
    section_bits = []
    if getattr(args, "proj_supcon", False):
        section_bits.append("proj_supcon")
    if args.lora_contrastive_finetune:
        section_bits.append("lora_supcon")
    if getattr(args, "prototype_learning", False):
        section_bits.append("prototype")
    section_bits.append("audio")
    return "_".join(section_bits)


def get_timestamp_suffix() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def get_run_stem(args: argparse.Namespace) -> str:
    return f"{get_model_tag(args)}_{args.dataset}_mlp_{args.num_labels}cls_{get_section_suffix(args)}"


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
        print(f"Audio path: {samples[idx].text}")
        print(f"Embedding dimension: {tuple(embeddings[idx].shape)}")
        print(f"True label: {samples[idx].label}")
        if preds is not None:
            print(f"Predicted label: {preds[idx]}")


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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="WavLM-Large + MLP emotion classifier.")
    parser.add_argument("--mode", choices=["train_eval", "train", "eval"], default="train_eval")
    parser.add_argument("--dataset", choices=["meld", "iemocap"], default="meld")
    parser.add_argument("--model_path", default="/scratch/data/bikash_rs/Vivek/PRC-Emo/models/wavlm_large")
    parser.add_argument("--train_file", default=MELD_METADATA["train"])
    parser.add_argument("--valid_file", default=MELD_METADATA["valid"])
    parser.add_argument("--test_file", default=MELD_METADATA["test"])
    parser.add_argument("--train_audio_dir", default=MELD_AUDIO_DIRS["train"])
    parser.add_argument("--valid_audio_dir", default=MELD_AUDIO_DIRS["valid"])
    parser.add_argument("--test_audio_dir", default=MELD_AUDIO_DIRS["test"])
    parser.add_argument("--checkpoint", default="")
    parser.add_argument("--output_dir", default="/scratch/data/bikash_rs/Vivek/PRC-Emo/wavlm")
    parser.add_argument("--label_names", default="", help="Comma-separated labels in class-id order.")
    parser.add_argument("--num_labels", type=int, default=None)
    parser.add_argument("--embedding_dim", type=int, default=1024)
    parser.add_argument("--pooling", choices=["mean"], default="mean")
    parser.add_argument("--max_audio_seconds", type=float, default=20.0)

    parser.add_argument("--load_in_4bit", action="store_true")
    parser.add_argument("--load_in_8bit", action="store_true")
    parser.add_argument("--attn_implementation", default="", help="Example: flash_attention_2")
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
    parser.add_argument("--embedding_analysis", action="store_true")
    parser.add_argument("--lora_contrastive_finetune", action="store_true")
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
    parser.add_argument("--lora_target_modules", default="q_proj,k_proj,v_proj,out_proj")
    parser.add_argument("--gradient_checkpointing", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--proj_supcon", action="store_true")
    args = parser.parse_args()

    if args.dataset == "iemocap":
        args.train_file = IEMOCAP_METADATA["train"]
        args.valid_file = IEMOCAP_METADATA["valid"]
        args.test_file = IEMOCAP_METADATA["test"]
        args.train_audio_dir = ""
        args.valid_audio_dir = ""
        args.test_audio_dir = ""

    if args.lora_batch_size < 2:
        raise ValueError("--lora_batch_size must be at least 2.")
    if args.lora_grad_accum_steps < 1:
        raise ValueError("--lora_grad_accum_steps must be at least 1.")
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
    if args.proj_supcon:
        args.l2_normalize = True
    args.model_tag = get_model_tag(args)
    return args


def resolve_split_path(path_text: str) -> Optional[Path]:
    if not path_text:
        return None
    path = Path(path_text)
    return path if path.is_absolute() else Path.cwd() / path


def build_meld_samples(metadata_path: Path, audio_dir: Path) -> List[EmotionSample]:
    samples: List[EmotionSample] = []
    with metadata_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            audio_path = audio_dir / f"dia{row['Dialogue_ID']}_utt{row['Utterance_ID']}.wav"
            samples.append(
                EmotionSample(
                    text=str(audio_path),
                    label=row["Emotion"].strip().lower(),
                    target_speaker=row.get("Speaker", "").strip(),
                    target_utterance=row.get("Utterance", "").strip(),
                )
            )
    return samples


def build_iemocap_samples(metadata_path: Path) -> List[EmotionSample]:
    samples: List[EmotionSample] = []
    with metadata_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            samples.append(
                EmotionSample(
                    text=row["wav_path"].strip(),
                    label=row["emotion"].strip().lower(),
                    target_speaker=row.get("session", "").strip(),
                    target_utterance=row.get("turn_name", "").strip(),
                )
            )
    return samples


def validate_audio_samples(samples: Sequence[EmotionSample], split_name: str) -> List[EmotionSample]:
    valid_samples: List[EmotionSample] = []
    skipped: List[Tuple[str, str]] = []
    for sample in samples:
        try:
            waveform, _ = torchaudio.load(sample.text)
            if waveform.numel() == 0:
                raise ValueError("empty waveform")
        except Exception as exc:
            skipped.append((sample.text, str(exc)))
            continue
        valid_samples.append(sample)

    print(f"{split_name}: kept {len(valid_samples)} audio files, skipped {len(skipped)} invalid files.")
    for audio_path, reason in skipped:
        print(f"Skipped {split_name} audio: {audio_path} | reason: {reason}")
    return valid_samples


def load_audio_samples(split_name: str, metadata_path: Path, audio_dir: Optional[Path], args: argparse.Namespace) -> List[EmotionSample]:
    if args.dataset == "meld":
        if audio_dir is None:
            raise ValueError(f"Audio directory is required for MELD {split_name}.")
        samples = build_meld_samples(metadata_path, audio_dir)
    else:
        samples = build_iemocap_samples(metadata_path)
    return validate_audio_samples(samples, split_name)


def embed_audio_samples(
    samples: Sequence[EmotionSample],
    label_ids: torch.Tensor,
    embedding_wrapper: WavLMEmbeddingWrapper,
    collator: AudioCollator,
    args: argparse.Namespace,
    split_name: str,
) -> Tuple[torch.Tensor, torch.Tensor]:
    if not samples:
        return torch.empty((0, args.embedding_dim), dtype=torch.float32), label_ids.clone()
    loader = torch.utils.data.DataLoader(
        AudioEmotionDataset(samples, label_ids),
        batch_size=args.embed_batch_size,
        shuffle=False,
        collate_fn=collator,
    )
    return embed_from_loader(loader, None, embedding_wrapper, args, split_name)


def save_checkpoint(
    model: nn.Module,
    args: argparse.Namespace,
    labels: Sequence[str],
    embedding_wrapper: Optional[WavLMEmbeddingWrapper] = None,
) -> Path:
    save_dir = Path(args.output_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    run_stem = f"{get_run_stem(args)}_{get_timestamp_suffix()}"
    checkpoint_path = save_dir / f"{run_stem}.pt"

    embedding_model_path = None
    feature_extractor_path = None
    lora_adapter_path = None
    if embedding_wrapper is not None:
        if args.lora_contrastive_finetune:
            lora_adapter_path = embedding_wrapper.save(save_dir / f"{run_stem}_lora_adapter")
            feature_extractor_path = lora_adapter_path
            print(f"Saved LoRA adapter to {lora_adapter_path}")
        else:
            embedding_model_path = embedding_wrapper.save(save_dir / f"{run_stem}_wavlm_model")
            feature_extractor_path = embedding_model_path
            print(f"Saved WavLM model to {embedding_model_path}")

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
            "embedding_model_path": str(embedding_model_path) if embedding_model_path is not None else None,
            "feature_extractor_path": str(feature_extractor_path) if feature_extractor_path is not None else None,
            "lora_adapter_path": str(lora_adapter_path) if lora_adapter_path is not None else None,
            "prototype_vectors": prototype_vectors,
            "prototype_labels": prototype_labels,
            "alpha_param": alpha_param,
        },
        checkpoint_path,
    )
    print(f"Saved MLP checkpoint to {checkpoint_path}")
    return checkpoint_path


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


def load_embedding_wrapper(args: argparse.Namespace, checkpoint_path: Optional[Path] = None) -> WavLMEmbeddingWrapper:
    model_path = args.model_path
    feature_extractor_path = args.model_path
    if checkpoint_path is not None:
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        saved_model_path = checkpoint.get("embedding_model_path")
        saved_feature_extractor_path = checkpoint.get("feature_extractor_path")
        if saved_model_path:
            model_path = saved_model_path
        if saved_feature_extractor_path:
            feature_extractor_path = saved_feature_extractor_path

    wrapper = WavLMEmbeddingWrapper.load(
        model_path,
        feature_extractor_path=feature_extractor_path,
        load_in_4bit=args.load_in_4bit,
        load_in_8bit=args.load_in_8bit,
        attn_implementation=args.attn_implementation,
    )
    if checkpoint_path is not None:
        wrapper.model = load_lora_adapter_from_checkpoint(checkpoint_path, wrapper.model)
    if args.lora_contrastive_finetune and args.mode in {"train_eval", "train"}:
        wrapper.model.train()
    else:
        wrapper.model.eval()
    return wrapper


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


def main() -> None:
    args = parse_args()
    if args.load_in_4bit and args.load_in_8bit:
        raise ValueError("Choose only one of --load_in_4bit or --load_in_8bit.")

    set_seed(args.seed)
    labels = get_labels(args)
    label_to_id = {label: idx for idx, label in enumerate(labels)}

    split_paths = {
        "train": resolve_split_path(args.train_file),
        "valid": resolve_split_path(args.valid_file),
        "test": resolve_split_path(args.test_file),
    }
    audio_dirs = {
        "train": resolve_split_path(args.train_audio_dir),
        "valid": resolve_split_path(args.valid_audio_dir),
        "test": resolve_split_path(args.test_audio_dir),
    }

    required_splits: List[str] = []
    if args.mode in {"train_eval", "train"}:
        required_splits.extend(["train"])
        if split_paths["valid"] is not None:
            required_splits.append("valid")
    if args.mode in {"train_eval", "eval"} and split_paths["test"] is not None:
        required_splits.append("test")
    required_splits = list(dict.fromkeys(required_splits))

    split_samples: Dict[str, List[EmotionSample]] = {}
    for split_name in required_splits:
        split_path = split_paths[split_name]
        if split_path is None or not split_path.exists():
            raise FileNotFoundError(f"{split_name} file not found: {split_path}")
        split_samples[split_name] = load_audio_samples(split_name, split_path, audio_dirs[split_name], args)

    embedding_wrapper = load_embedding_wrapper(args, Path(args.checkpoint) if args.mode == "eval" and args.checkpoint else None)
    if args.lora_contrastive_finetune and args.mode in {"train_eval", "train"}:
        embedding_wrapper.model = apply_lora_to_wavlm_embedding_model(embedding_wrapper.model, args)

    collator = AudioCollator(
        embedding_wrapper.feature_extractor,
        target_sample_rate=16000,
        max_audio_seconds=args.max_audio_seconds,
    )
    classifier = None
    analysis_payloads: Dict[str, Tuple[Sequence[EmotionSample], torch.Tensor, torch.Tensor]] = {}

    def get_label_ids(samples: Sequence[EmotionSample]) -> torch.Tensor:
        ids = []
        for sample in samples:
            if sample.label not in label_to_id:
                raise KeyError(f"Unknown label '{sample.label}' for audio sample: {sample.text}")
            ids.append(label_to_id[sample.label])
        return torch.tensor(ids, dtype=torch.long)

    if args.mode in {"train_eval", "train"}:
        train_samples = split_samples["train"]
        valid_samples = split_samples.get("valid", [])
        print(f"Loaded {len(train_samples)} train samples and {len(valid_samples)} valid samples.")

        train_y = get_label_ids(train_samples)
        valid_y = get_label_ids(valid_samples) if valid_samples else None

        if args.proj_supcon:
            train_embeddings, train_y = embed_audio_samples(
                train_samples, train_y, embedding_wrapper, collator, args, "train"
            )
            analysis_payloads["train"] = (train_samples, train_embeddings, train_y)
            print_embedding_debug(train_samples, train_embeddings, args.debug_samples, "train")

            valid_embeddings = None
            if valid_samples and valid_y is not None:
                valid_embeddings, valid_y = embed_audio_samples(
                    valid_samples, valid_y, embedding_wrapper, collator, args, "valid"
                )
                analysis_payloads["valid"] = (valid_samples, valid_embeddings, valid_y)
                print_embedding_debug(valid_samples, valid_embeddings, args.debug_samples, "valid")

            classifier = train_proj_supcon(train_embeddings, train_y, valid_embeddings, valid_y, args)
        elif args.lora_contrastive_finetune:
            classifier = train_lora_contrastive(
                embedding_wrapper,
                None,
                train_samples,
                train_y,
                valid_samples,
                valid_y,
                args,
                dataset_cls=AudioEmotionDataset,
                collate_fn=collator,
            )
            train_embeddings, train_y = embed_audio_samples(
                train_samples, train_y, embedding_wrapper, collator, args, "train"
            )
            analysis_payloads["train"] = (train_samples, train_embeddings, train_y)
            print_embedding_debug(train_samples, train_embeddings, args.debug_samples, "train")
            if valid_samples and valid_y is not None:
                valid_embeddings, valid_y = embed_audio_samples(
                    valid_samples, valid_y, embedding_wrapper, collator, args, "valid"
                )
                analysis_payloads["valid"] = (valid_samples, valid_embeddings, valid_y)
                print_embedding_debug(valid_samples, valid_embeddings, args.debug_samples, "valid")
        else:
            train_embeddings, train_y = embed_audio_samples(
                train_samples, train_y, embedding_wrapper, collator, args, "train"
            )
            analysis_payloads["train"] = (train_samples, train_embeddings, train_y)
            print_embedding_debug(train_samples, train_embeddings, args.debug_samples, "train")

            valid_embeddings = None
            if valid_samples and valid_y is not None:
                valid_embeddings, valid_y = embed_audio_samples(
                    valid_samples, valid_y, embedding_wrapper, collator, args, "valid"
                )
                analysis_payloads["valid"] = (valid_samples, valid_embeddings, valid_y)
                print_embedding_debug(valid_samples, valid_embeddings, args.debug_samples, "valid")

            if args.prototype_learning:
                classifier = train_frozen_prototype_mlp(train_embeddings, train_y, valid_embeddings, valid_y, args)
            else:
                classifier = train_mlp(train_embeddings, train_y, valid_embeddings, valid_y, args, EmotionMLP)

        checkpoint_path = save_checkpoint(classifier, args, labels, embedding_wrapper)
        args.checkpoint = str(checkpoint_path)

    if args.mode == "eval":
        if not args.checkpoint:
            raise ValueError("--checkpoint is required for --mode eval.")
        classifier = load_checkpoint(Path(args.checkpoint), args, labels)

    if args.mode in {"train_eval", "eval"}:
        test_samples = split_samples["test"]
        print(f"Loaded {len(test_samples)} test samples.")
        test_y = get_label_ids(test_samples)
        test_embeddings, test_y = embed_audio_samples(test_samples, test_y, embedding_wrapper, collator, args, "test")
        analysis_payloads["test"] = (test_samples, test_embeddings, test_y)
        evaluate_split(classifier, test_samples, test_embeddings, test_y, labels, args, "test")

    if args.embedding_analysis:
        from common.visualization import save_embedding_analysis

        for split_name in required_splits:
            if split_name in analysis_payloads:
                continue
            samples = split_samples[split_name]
            y = get_label_ids(samples)
            embeddings, y = embed_audio_samples(samples, y, embedding_wrapper, collator, args, split_name)
            analysis_payloads[split_name] = (samples, embeddings, y)
        save_embedding_analysis(analysis_payloads, labels, args)


if __name__ == "__main__":
    main()
