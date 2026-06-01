import argparse
import json
import os
import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from tqdm.auto import tqdm

try:
    from sklearn.metrics import classification_report, confusion_matrix
except Exception as exc:
    SKLEARN_IMPORT_ERROR = exc
    classification_report = None
    confusion_matrix = None
else:
    SKLEARN_IMPORT_ERROR = None

MELD_LABELS = ["neutral", "surprise", "fear", "sadness", "joy", "disgust", "anger"]
IEMOCAP_LABELS = ["happy", "sad", "neutral", "angry", "excited", "frustrated"]
UTTERANCE_TASK_INSTRUCTION = "Represent the emotional state of the utterance."
CONTEXT_TASK_INSTRUCTION = "Represent the emotional state and conversational behavior of the target utterance."


@dataclass
class EmotionSample:
    text: str
    label: str
    target_speaker: str = ""
    target_utterance: str = ""


class EmotionMLP(nn.Module):
    def __init__(self, input_dim: int = 4096, num_labels: int = 7, dropout: float = 0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 1024),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(1024, 256),
            nn.GELU(),
            nn.Linear(256, num_labels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def normalize_text(text: str) -> str:
    return text.replace("\u0092", "'").replace("\u2019", "'").strip()


def section_between(text: str, start_pattern: str, end_patterns: Sequence[str]) -> str:
    match = re.search(start_pattern, text, flags=re.IGNORECASE)
    if not match:
        return ""
    start = match.end()
    end = len(text)
    for pattern in end_patterns:
        end_match = re.search(pattern, text[start:], flags=re.IGNORECASE)
        if end_match:
            end = min(end, start + end_match.start())
    return normalize_text(text[start:end])


def parse_user_target(user_text: str) -> Tuple[str, str]:
    user_text = normalize_text(user_text)
    match = re.search(
        r"emotional label of\s+(.+?)\s+in the utterance\s+([\"'])(.+?)\2",
        user_text,
        flags=re.IGNORECASE | re.DOTALL,
    )
    if match:
        return normalize_text(match.group(1)), normalize_text(match.group(3))
    return "", ""


def parse_prompt_sample(record: Dict, args: argparse.Namespace) -> Optional[EmotionSample]:
    messages = record.get("messages", [])
    if len(messages) < 3:
        return None

    system_text = normalize_text(messages[0].get("content", ""))
    user_text = normalize_text(messages[1].get("content", ""))
    label = normalize_text(messages[-1].get("content", "")).lower()
    target_speaker, target_utterance = parse_user_target(user_text)

    headings = [
        r"\n### Visual Expressions",
        r"\n### Audio Characteristics",
        r"\n### Given the characteristic",
        r"\n### Given the speaker",
        r"\n### Semantic Contrastive Cues",
        r"\n### Reference Similar Emotional Expressions",
        r"\n### Available emotion labels",
    ]
    context = section_between(
        system_text,
        r"### Given the following conversation as a context\s*",
        headings,
    )
    speaker_desc = section_between(
        system_text,
        r"### Given the characteristic of this speaker:\s*",
        [r"\n### Given the speaker", r"\n### Visual Expressions", r"\n### Audio Characteristics", r"\n### Semantic Contrastive Cues", r"\n### Reference Similar Emotional Expressions", r"\n### Available emotion labels"],
    )
    emotion_block = section_between(
        system_text,
        r"### Given the speaker(?:'|’)?s Explicit Emotion Interpretation and Implicit Emotion Interpretation.*?:\s*",
        [r"\n### Semantic Contrastive Cues", r"\n### Reference Similar Emotional Expressions", r"\n### Available emotion labels"],
    )
    visual_desc = section_between(
        system_text,
        r"### Visual Expressions of the speaker present in this utterance:\s*",
        [r"\n### Audio Characteristics", r"\n### Given the characteristic", r"\n### Given the speaker", r"\n### Semantic Contrastive Cues", r"\n### Reference Similar Emotional Expressions", r"\n### Available emotion labels"],
    )
    audio_desc = section_between(
        system_text,
        r"### Audio Characteristics of the speaker present in this utterance:\s*",
        [r"\n### Visual Expressions", r"\n### Given the characteristic", r"\n### Given the speaker", r"\n### Semantic Contrastive Cues", r"\n### Reference Similar Emotional Expressions", r"\n### Available emotion labels"],
    )
    semantic_cues = section_between(
        system_text,
        r"### Semantic Contrastive Cues for the speaker in this utterance:\s*",
        [r"\n### Reference Similar Emotional Expressions", r"\n### Available emotion labels"],
    )
    reference_similar = section_between(
        system_text,
        r"### Reference Similar Emotional Expressions:\s*",
        [r"\n### Available emotion labels"],
    )

    explicit = ""
    implicit = ""
    explicit_match = re.search(
        r"Explicit Emotion Interpretation:\s*(.*?)(?:\n-\s*Implicit Emotion Interpretation:|\Z)",
        emotion_block,
        flags=re.IGNORECASE | re.DOTALL,
    )
    implicit_match = re.search(
        r"Implicit Emotion Interpretation:\s*(.*)",
        emotion_block,
        flags=re.IGNORECASE | re.DOTALL,
    )
    if explicit_match:
        explicit = normalize_text(explicit_match.group(1).lstrip("- "))
    if implicit_match:
        implicit = normalize_text(implicit_match.group(1).lstrip("- "))

    embedding_text = build_embedding_input(
        context=context,
        target_speaker=target_speaker,
        target_utterance=target_utterance,
        explicit=explicit,
        implicit=implicit,
        speaker_desc=speaker_desc,
        visual_desc=visual_desc,
        audio_desc=audio_desc,
        semantic_cues=semantic_cues,
        reference_similar=reference_similar,
        args=args,
    )
    return EmotionSample(embedding_text, label, target_speaker, target_utterance)


def build_embedding_input(
    context: str,
    target_speaker: str,
    target_utterance: str,
    explicit: str,
    implicit: str,
    speaker_desc: str,
    visual_desc: str,
    audio_desc: str,
    semantic_cues: str,
    reference_similar: str,
    args: argparse.Namespace,
) -> str:
    include_context = args.include_conversation_context or args.include_target_speaker
    instruction = CONTEXT_TASK_INSTRUCTION if include_context else UTTERANCE_TASK_INSTRUCTION
    sections = [
        f"Instruct: {instruction}",
        f"Target Utterance:\n\"{target_utterance}\"",
    ]
    if include_context:
        sections.insert(1, f"Conversation Context:\n{context}")
        sections.insert(2, f"Target Speaker:\n{target_speaker}")
    if args.include_explicit_emotion:
        sections.append(f"Explicit Emotion:\n{explicit or 'Not available.'}")
    if args.include_implicit_emotion:
        sections.append(f"Implicit Emotion:\n{implicit or 'Not available.'}")
    if args.include_speaker_description:
        sections.append(f"Speaker Description:\n{speaker_desc or 'Not available.'}")
    if args.include_visual_description:
        sections.append(f"Visual Description:\n{visual_desc or 'Not available.'}")
    if args.include_audio_description:
        sections.append(f"Audio Description:\n{audio_desc or 'Not available.'}")
    if args.include_semantic_contrastive_cues:
        sections.append(f"Semantic Contrastive Cues:\n{semantic_cues or 'Not available.'}")
    if args.include_reference_similar_emotions:
        sections.append(f"Reference Similar Emotions:\n{reference_similar or 'Not available.'}")
    return "\n\n".join(sections)


def parse_dialogue_json(path: Path, args: argparse.Namespace, labels: Sequence[str]) -> List[EmotionSample]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    samples = []
    window = args.context_window
    for dialog in data.values():
        sentences = dialog.get("sentences", [])
        speakers = dialog.get("speakers") or dialog.get("Speaker") or []
        raw_labels = dialog.get("labels", [])
        for idx, utterance in enumerate(sentences):
            start = max(0, idx - window)
            end = min(len(sentences), idx + window + 1)
            context_lines = []
            for j in range(start, end):
                speaker = speakers[j] if j < len(speakers) else f"Speaker_{j}"
                context_lines.append(f"{speaker}: {normalize_text(sentences[j])}")
            speaker = speakers[idx] if idx < len(speakers) else ""
            raw_label = raw_labels[idx] if idx < len(raw_labels) else None
            if raw_label is None:
                continue
            label = labels[int(raw_label)] if isinstance(raw_label, int) and int(raw_label) < len(labels) else str(raw_label).lower()
            text = build_embedding_input(
                context="\n".join(context_lines),
                target_speaker=str(speaker),
                target_utterance=normalize_text(utterance),
                explicit="",
                implicit="",
                speaker_desc="",
                visual_desc="",
                audio_desc="",
                semantic_cues="",
                reference_similar="",
                args=args,
            )
            samples.append(EmotionSample(text, label, str(speaker), normalize_text(utterance)))
    return samples


def load_samples(path: Path, args: argparse.Namespace, labels: Sequence[str]) -> List[EmotionSample]:
    if path.suffix == ".jsonl":
        samples = []
        with path.open("r", encoding="utf-8") as f:
            for line_no, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    sample = parse_prompt_sample(json.loads(line), args)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"Invalid JSON on {path}:{line_no}") from exc
                if sample is not None:
                    samples.append(sample)
        return samples
    return parse_dialogue_json(path, args, labels)


def last_token_pool(last_hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    left_padding = attention_mask[:, -1].sum() == attention_mask.shape[0]
    if left_padding:
        return last_hidden_states[:, -1]
    sequence_lengths = attention_mask.sum(dim=1) - 1
    batch_size = last_hidden_states.shape[0]
    return last_hidden_states[torch.arange(batch_size, device=last_hidden_states.device), sequence_lengths]


def load_embedding_model(args: argparse.Namespace):
    try:
        from transformers import AutoModel, AutoTokenizer
        from transformers import BitsAndBytesConfig
    except ImportError as exc:
        raise ImportError(
            "Loading Qwen3 embeddings requires transformers and huggingface_hub. "
            "Install the project requirements in the environment you use to run this script."
        ) from exc

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, padding_side="left", trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model_kwargs = {"trust_remote_code": True}
    if torch.cuda.is_available():
        model_kwargs["device_map"] = "auto"
        model_kwargs["torch_dtype"] = torch.float16
        if args.attn_implementation:
            model_kwargs["attn_implementation"] = args.attn_implementation
        if args.load_in_4bit or args.load_in_8bit:
            model_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=args.load_in_4bit,
                load_in_8bit=args.load_in_8bit,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
            )
    model = AutoModel.from_pretrained(args.model_path, **model_kwargs)
    if not torch.cuda.is_available():
        model.to("cpu")
    model.eval()
    return tokenizer, model


@torch.no_grad()
def embed_texts(
    texts: Sequence[str],
    tokenizer,
    model,
    args: argparse.Namespace,
    split_name: str,
) -> torch.Tensor:
    all_embeddings = []
    device = model.device
    for start in tqdm(range(0, len(texts), args.embed_batch_size), desc=f"Embedding {split_name}"):
        batch_texts = list(texts[start : start + args.embed_batch_size])
        batch = tokenizer(
            batch_texts,
            padding=True,
            truncation=True,
            max_length=args.max_length,
            return_tensors="pt",
        ).to(device)
        outputs = model(**batch)
        embeddings = last_token_pool(outputs.last_hidden_state, batch["attention_mask"])
        embeddings = F.normalize(embeddings, p=2, dim=1)
        all_embeddings.append(embeddings.detach().float().cpu())
    return torch.cat(all_embeddings, dim=0)


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


def labels_to_ids(samples: Sequence[EmotionSample], label_to_id: Dict[str, int]) -> torch.Tensor:
    ids = []
    skipped = []
    for idx, sample in enumerate(samples):
        label = sample.label.lower()
        if label not in label_to_id:
            skipped.append((idx, label))
            continue
        ids.append(label_to_id[label])
    if skipped:
        preview = ", ".join(f"{i}:{label}" for i, label in skipped[:10])
        raise ValueError(f"Found labels not in label set: {preview}")
    return torch.tensor(ids, dtype=torch.long)


def make_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, num_labels: int) -> np.ndarray:
    if confusion_matrix is not None:
        return confusion_matrix(y_true, y_pred, labels=list(range(num_labels)))
    matrix = np.zeros((num_labels, num_labels), dtype=int)
    for true_id, pred_id in zip(y_true, y_pred):
        if 0 <= int(true_id) < num_labels and 0 <= int(pred_id) < num_labels:
            matrix[int(true_id), int(pred_id)] += 1
    return matrix


def make_classification_report(y_true: np.ndarray, y_pred: np.ndarray, labels: Sequence[str]) -> str:
    if classification_report is not None:
        return classification_report(
            y_true,
            y_pred,
            labels=list(range(len(labels))),
            target_names=list(labels),
            digits=4,
            zero_division=0,
        )

    rows = []
    rows.append(f"scikit-learn unavailable ({SKLEARN_IMPORT_ERROR}); using fallback metrics.")
    rows.append(f"{'label':>14} {'precision':>10} {'recall':>10} {'f1-score':>10} {'support':>10}")
    f1_values = []
    supports = []
    for idx, label in enumerate(labels):
        tp = int(((y_true == idx) & (y_pred == idx)).sum())
        fp = int(((y_true != idx) & (y_pred == idx)).sum())
        fn = int(((y_true == idx) & (y_pred != idx)).sum())
        support = int((y_true == idx).sum())
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
        f1_values.append(f1)
        supports.append(support)
        rows.append(f"{label:>14} {precision:10.4f} {recall:10.4f} {f1:10.4f} {support:10d}")
    accuracy = float((y_true == y_pred).mean()) if len(y_true) else 0.0
    macro_f1 = float(np.mean(f1_values)) if f1_values else 0.0
    weighted_f1 = float(np.average(f1_values, weights=supports)) if sum(supports) else 0.0
    rows.append("")
    rows.append(f"{'accuracy':>14} {'':>10} {'':>10} {accuracy:10.4f} {len(y_true):10d}")
    rows.append(f"{'macro avg':>14} {'':>10} {'':>10} {macro_f1:10.4f} {len(y_true):10d}")
    rows.append(f"{'weighted avg':>14} {'':>10} {'':>10} {weighted_f1:10.4f} {len(y_true):10d}")
    return "\n".join(rows)


def train_mlp(
    train_embeddings: torch.Tensor,
    train_labels: torch.Tensor,
    valid_embeddings: Optional[torch.Tensor],
    valid_labels: Optional[torch.Tensor],
    args: argparse.Namespace,
) -> EmotionMLP:
    device = torch.device("cuda" if torch.cuda.is_available() and not args.force_cpu_mlp else "cpu")
    model = EmotionMLP(args.embedding_dim, args.num_labels, args.dropout).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    train_loader = DataLoader(
        TensorDataset(train_embeddings, train_labels),
        batch_size=args.mlp_batch_size,
        shuffle=True,
    )

    best_state = None
    best_valid_loss = float("inf")
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        total_correct = 0
        total_seen = 0
        for x, y in tqdm(train_loader, desc=f"MLP epoch {epoch}/{args.epochs}"):
            x = x.to(device)
            y = y.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(x)
            loss = F.cross_entropy(logits, y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * y.size(0)
            total_correct += (logits.argmax(dim=-1) == y).sum().item()
            total_seen += y.size(0)
        train_loss = total_loss / max(total_seen, 1)
        train_acc = total_correct / max(total_seen, 1)

        valid_loss = None
        valid_acc = None
        if valid_embeddings is not None and valid_labels is not None:
            valid_loss, valid_acc, _ = predict_mlp(model, valid_embeddings, valid_labels, args)
            if valid_loss < best_valid_loss:
                best_valid_loss = valid_loss
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

        if valid_loss is None:
            print(f"Epoch {epoch}: train_loss={train_loss:.4f} train_acc={train_acc:.4f}")
        else:
            print(
                f"Epoch {epoch}: train_loss={train_loss:.4f} train_acc={train_acc:.4f} "
                f"valid_loss={valid_loss:.4f} valid_acc={valid_acc:.4f}"
            )

    if best_state is not None:
        model.load_state_dict(best_state)
    return model


@torch.no_grad()
def predict_mlp(
    model: EmotionMLP,
    embeddings: torch.Tensor,
    labels: Optional[torch.Tensor],
    args: argparse.Namespace,
) -> Tuple[Optional[float], Optional[float], np.ndarray]:
    device = next(model.parameters()).device
    loader = DataLoader(
        TensorDataset(embeddings, labels if labels is not None else torch.zeros(len(embeddings), dtype=torch.long)),
        batch_size=args.mlp_batch_size,
        shuffle=False,
    )
    model.eval()
    losses = []
    correct = 0
    seen = 0
    preds = []
    for x, y in loader:
        x = x.to(device)
        y = y.to(device)
        logits = model(x)
        pred = logits.argmax(dim=-1)
        preds.extend(pred.detach().cpu().tolist())
        if labels is not None:
            loss = F.cross_entropy(logits, y)
            losses.append(loss.item() * y.size(0))
            correct += (pred == y).sum().item()
            seen += y.size(0)
    loss_value = sum(losses) / max(seen, 1) if labels is not None else None
    acc_value = correct / max(seen, 1) if labels is not None else None
    return loss_value, acc_value, np.array(preds)


def save_checkpoint(model: EmotionMLP, args: argparse.Namespace, labels: Sequence[str]) -> Path:
    save_dir = Path(args.output_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    section_bits = []
    for name, enabled in [
        ("context", args.include_conversation_context or args.include_target_speaker),
        ("explicit", args.include_explicit_emotion),
        ("implicit", args.include_implicit_emotion),
        ("speaker", args.include_speaker_description),
        ("visual", args.include_visual_description),
        ("audio", args.include_audio_description),
        ("semantic", args.include_semantic_contrastive_cues),
        ("refs", args.include_reference_similar_emotions),
    ]:
        if enabled:
            section_bits.append(name)
    section_suffix = "_".join(section_bits) if section_bits else "utt"
    filename = f"qwen3_embedding_8b_{args.dataset}_mlp_{args.num_labels}cls_{section_suffix}.pt"
    path = save_dir / filename
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "embedding_dim": args.embedding_dim,
            "num_labels": args.num_labels,
            "labels": list(labels),
            "args": vars(args),
        },
        path,
    )
    print(f"Saved MLP checkpoint to {path}")
    return path


def load_checkpoint(path: Path, args: argparse.Namespace, labels: Sequence[str]) -> EmotionMLP:
    checkpoint = torch.load(path, map_location="cpu")
    embedding_dim = int(checkpoint.get("embedding_dim", args.embedding_dim))
    num_labels = int(checkpoint.get("num_labels", args.num_labels))
    model = EmotionMLP(embedding_dim, num_labels, args.dropout)
    model.load_state_dict(checkpoint["model_state_dict"])
    device = torch.device("cuda" if torch.cuda.is_available() and not args.force_cpu_mlp else "cpu")
    model.to(device)
    checkpoint_labels = checkpoint.get("labels")
    if checkpoint_labels and list(checkpoint_labels) != list(labels):
        print(f"Warning: checkpoint labels {checkpoint_labels} differ from current labels {list(labels)}")
    return model


def evaluate_split(
    model: EmotionMLP,
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
    parser.add_argument("--embedding_dim", type=int, default=4096)
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
    return parser.parse_args()


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

    classifier = None
    if args.mode in {"train_eval", "train"}:
        train_samples = load_samples(train_path, args, labels)
        valid_samples = load_samples(valid_path, args, labels) if valid_path and valid_path.exists() else []
        print(f"Loaded {len(train_samples)} train samples and {len(valid_samples)} valid samples.")

        train_embeddings = embed_texts([s.text for s in train_samples], tokenizer, embedding_model, args, "train")
        train_y = labels_to_ids(train_samples, label_to_id)
        print_embedding_debug(train_samples, train_embeddings, args.debug_samples, "train")

        valid_embeddings = None
        valid_y = None
        if valid_samples:
            valid_embeddings = embed_texts([s.text for s in valid_samples], tokenizer, embedding_model, args, "valid")
            valid_y = labels_to_ids(valid_samples, label_to_id)
            print_embedding_debug(valid_samples, valid_embeddings, args.debug_samples, "valid")

        classifier = train_mlp(train_embeddings, train_y, valid_embeddings, valid_y, args)
        checkpoint_path = save_checkpoint(classifier, args, labels)
        args.checkpoint = str(checkpoint_path)

    if args.mode == "eval":
        if not args.checkpoint:
            raise ValueError("--checkpoint is required for --mode eval.")
        classifier = load_checkpoint(Path(args.checkpoint), args, labels)

    if args.mode in {"train_eval", "eval"}:
        if test_path is None or not test_path.exists():
            raise FileNotFoundError(f"Test file not found: {test_path}")
        test_samples = load_samples(test_path, args, labels)
        print(f"Loaded {len(test_samples)} test samples.")
        test_embeddings = embed_texts([s.text for s in test_samples], tokenizer, embedding_model, args, "test")
        test_y = labels_to_ids(test_samples, label_to_id)
        evaluate_split(classifier, test_samples, test_embeddings, test_y, labels, args, "test")


if __name__ == "__main__":
    main()
