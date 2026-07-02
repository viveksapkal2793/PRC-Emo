import argparse
from collections import deque
import json
import os
import random
import re
from datetime import datetime
from dataclasses import dataclass
from pathlib import Path
from typing import Deque, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import BatchSampler, DataLoader, Dataset, TensorDataset
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


class ProjectionHead(nn.Module):
    def __init__(self, input_dim: int = 4096):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 1024),
            nn.LayerNorm(1024),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(1024, 256),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class SupConMLPHead(nn.Module):
    def __init__(self, num_labels: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(256, 128),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(128, num_labels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ProjectionSupConClassifier(nn.Module):
    def __init__(self, input_dim: int = 4096, num_labels: int = 7):
        super().__init__()
        self.projection = ProjectionHead(input_dim)
        self.classifier = SupConMLPHead(num_labels)

    def project(self, x: torch.Tensor) -> torch.Tensor:
        return self.projection(x)

    def classify(self, projected: torch.Tensor) -> torch.Tensor:
        return self.classifier(projected)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classify(self.project(x))

    def forward_with_projection(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        projected = self.project(x)
        return projected, self.classify(projected)


def _alpha_to_logit(alpha: float) -> float:
    alpha = min(max(float(alpha), 1e-6), 1.0 - 1e-6)
    return float(np.log(alpha / (1.0 - alpha)))


class PrototypeManager:
    def __init__(self, args: argparse.Namespace):
        self.args = args
        self._clusterer_cls = None

    def _reduce_for_prototype_clustering(self, embeddings: torch.Tensor) -> torch.Tensor:
        if embeddings.ndim != 2 or embeddings.size(0) < 2 or embeddings.size(1) < 2:
            return embeddings.detach().float().cpu()

        n_samples, n_features = embeddings.shape
        n_components = min(50, n_features, n_samples - 1)
        if n_components < 1 or n_components >= n_features:
            return embeddings.detach().float().cpu()

        try:
            from sklearn.decomposition import IncrementalPCA, PCA
        except Exception as exc:
            raise ImportError(
                "Prototype learning requires scikit-learn PCA support for prototype clustering."
            ) from exc

        embeddings_np = embeddings.detach().float().cpu().numpy()
        if n_samples > 512:
            batch_size = min(n_samples, max(n_components, 256))
            reducer = IncrementalPCA(n_components=n_components, batch_size=batch_size)
            for start in range(0, n_samples, batch_size):
                reducer.partial_fit(embeddings_np[start : start + batch_size])
            reduced_chunks = []
            for start in range(0, n_samples, batch_size):
                reduced_chunks.append(reducer.transform(embeddings_np[start : start + batch_size]))
            reduced_np = np.concatenate(reduced_chunks, axis=0)
        else:
            reduced_np = PCA(n_components=n_components).fit_transform(embeddings_np)
        return torch.from_numpy(reduced_np).float()

    def _get_clusterer_cls(self):
        if self._clusterer_cls is not None:
            return self._clusterer_cls
        try:
            import hdbscan  # type: ignore

            self._clusterer_cls = hdbscan.HDBSCAN
            return self._clusterer_cls
        except Exception as first_exc:
            try:
                from sklearn.cluster import HDBSCAN as SklearnHDBSCAN  # type: ignore

                self._clusterer_cls = SklearnHDBSCAN
                return self._clusterer_cls
            except Exception as second_exc:
                raise ImportError(
                    "Prototype learning requires an HDBSCAN implementation. "
                    "Install the `hdbscan` package or a scikit-learn version that provides sklearn.cluster.HDBSCAN."
                ) from second_exc if second_exc is not None else first_exc

    def _distance_to_centroids(self, points: torch.Tensor, centroids: torch.Tensor) -> torch.Tensor:
        metric = str(self.args.prototype_hdbscan_metric).lower()
        if metric == "cosine":
            point_norm = F.normalize(points, p=2, dim=1)
            centroid_norm = F.normalize(centroids, p=2, dim=1)
            return 1.0 - torch.matmul(point_norm, centroid_norm.T)
        return torch.cdist(points, centroids, p=2)

    def _fallback_class_prototype(self, class_embeddings: torch.Tensor, class_id: int) -> Tuple[torch.Tensor, torch.Tensor]:
        centroid = F.normalize(class_embeddings.mean(dim=0, keepdim=True), p=2, dim=1)
        print(f"Prototype fallback: class {class_id} uses a single centroid prototype.")
        return centroid.cpu(), torch.tensor([class_id], dtype=torch.long)

    def recompute(
        self,
        embeddings: torch.Tensor,
        labels: torch.Tensor,
        num_labels: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        clusterer_cls = self._get_clusterer_cls()
        embeddings = F.normalize(embeddings.detach().float().cpu(), p=2, dim=1)
        labels = labels.detach().cpu().long()
        reduced_embeddings = self._reduce_for_prototype_clustering(embeddings)
        prototype_vectors = []
        prototype_labels = []

        for class_id in range(num_labels):
            class_mask = labels == class_id
            class_size = int(class_mask.sum().item())
            if class_size == 0:
                continue
            class_embeddings = embeddings[class_mask]
            min_cluster_size = max(int(self.args.prototype_min_cluster_size), int(0.03 * class_size), 1)
            if class_size < max(min_cluster_size, 2):
                fallback_vectors, fallback_labels = self._fallback_class_prototype(class_embeddings, class_id)
                prototype_vectors.append(fallback_vectors)
                prototype_labels.append(fallback_labels)
                continue

            clusterer = clusterer_cls(
                min_cluster_size=min_cluster_size,
                metric=self.args.prototype_hdbscan_metric,
                cluster_selection_method="leaf",
            )
            class_reduced_embeddings = reduced_embeddings[class_mask]
            cluster_assignments = np.asarray(clusterer.fit_predict(class_reduced_embeddings.numpy()), dtype=np.int64)
            valid_clusters = sorted(int(cluster_id) for cluster_id in np.unique(cluster_assignments) if cluster_id != -1)
            if not valid_clusters:
                print(f"Prototype warning: HDBSCAN found no valid clusters for class {class_id}. Falling back to one centroid.")
                fallback_vectors, fallback_labels = self._fallback_class_prototype(class_embeddings, class_id)
                prototype_vectors.append(fallback_vectors)
                prototype_labels.append(fallback_labels)
                continue

            centroid_ids = []
            centroid_vectors = []
            for cluster_id in valid_clusters:
                cluster_points = class_embeddings[cluster_assignments == cluster_id]
                centroid_ids.append(cluster_id)
                centroid_vectors.append(cluster_points.mean(dim=0))
            centroid_tensor = torch.stack(centroid_vectors, dim=0)

            outlier_mask = cluster_assignments == -1
            if outlier_mask.any():
                outliers = class_embeddings[outlier_mask]
                distances = self._distance_to_centroids(outliers, centroid_tensor)
                nearest_centroids = distances.argmin(dim=1).tolist()
                for outlier_idx, centroid_pos in zip(np.where(outlier_mask)[0].tolist(), nearest_centroids):
                    cluster_assignments[outlier_idx] = centroid_ids[int(centroid_pos)]

            final_cluster_ids = sorted(int(cluster_id) for cluster_id in np.unique(cluster_assignments))
            class_prototypes = []
            for cluster_id in final_cluster_ids:
                cluster_points = class_embeddings[cluster_assignments == cluster_id]
                centroid = F.normalize(cluster_points.mean(dim=0, keepdim=True), p=2, dim=1).squeeze(0)
                class_prototypes.append(centroid)

            prototype_vectors.append(torch.stack(class_prototypes, dim=0).cpu())
            prototype_labels.append(torch.full((len(class_prototypes),), class_id, dtype=torch.long))
            print(
                f"Prototype update: class {class_id} size={class_size} min_cluster_size={min_cluster_size} "
                f"clusters={len(class_prototypes)}"
            )

        if not prototype_vectors:
            return torch.empty((0, embeddings.size(1)), dtype=torch.float32), torch.empty((0,), dtype=torch.long)
        return torch.cat(prototype_vectors, dim=0), torch.cat(prototype_labels, dim=0)


class SamplePrototypeSupConLoss(nn.Module):
    def __init__(self, temperature: float):
        super().__init__()
        self.temperature = temperature

    def forward(
        self,
        embeddings: torch.Tensor,
        labels: torch.Tensor,
        prototype_vectors: torch.Tensor,
        prototype_labels: torch.Tensor,
    ) -> torch.Tensor:
        if embeddings.size(0) == 0 or prototype_vectors.numel() == 0:
            return embeddings.new_tensor(0.0)

        norm_embeddings = F.normalize(embeddings, p=2, dim=1)
        norm_prototypes = F.normalize(prototype_vectors, p=2, dim=1)
        similarity = torch.matmul(norm_embeddings, norm_prototypes.T) / self.temperature
        losses = []
        for idx in range(norm_embeddings.size(0)):
            same_class_mask = prototype_labels == labels[idx]
            negative_mask = prototype_labels != labels[idx]
            if not same_class_mask.any() or not negative_mask.any():
                continue
            positive_logit = similarity[idx, same_class_mask].max()
            logits = torch.cat([positive_logit.unsqueeze(0), similarity[idx, negative_mask]], dim=0)
            losses.append(-(positive_logit - torch.logsumexp(logits, dim=0)))
        if not losses:
            return embeddings.new_tensor(0.0)
        return torch.stack(losses).mean()


class PrototypeClassifier(nn.Module):
    def __init__(self, num_labels: int, temperature: float):
        super().__init__()
        self.num_labels = num_labels
        self.temperature = temperature
        self.register_buffer("prototype_vectors", torch.empty(0, 0))
        self.register_buffer("prototype_labels", torch.empty(0, dtype=torch.long))

    def set_prototypes(self, prototype_vectors: torch.Tensor, prototype_labels: torch.Tensor) -> None:
        if prototype_vectors.numel() == 0:
            feature_dim = self.prototype_vectors.size(1) if self.prototype_vectors.ndim == 2 else 0
            self.prototype_vectors = torch.empty((0, feature_dim), device=self.prototype_vectors.device)
            self.prototype_labels = torch.empty((0,), dtype=torch.long, device=self.prototype_labels.device)
            return
        normalized = F.normalize(prototype_vectors.detach().float(), p=2, dim=1)
        self.prototype_vectors = normalized.to(self.prototype_vectors.device)
        self.prototype_labels = prototype_labels.detach().long().to(self.prototype_labels.device)

    def has_prototypes(self) -> bool:
        return self.prototype_vectors.numel() > 0 and self.prototype_labels.numel() > 0

    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
        if not self.has_prototypes():
            return embeddings.new_zeros((embeddings.size(0), self.num_labels))
        norm_embeddings = F.normalize(embeddings, p=2, dim=1)
        norm_prototypes = F.normalize(self.prototype_vectors, p=2, dim=1)
        similarity = torch.matmul(norm_embeddings, norm_prototypes.T) / self.temperature
        class_scores = embeddings.new_full((embeddings.size(0), self.num_labels), -1e4)
        for class_id in range(self.num_labels):
            class_mask = self.prototype_labels == class_id
            if class_mask.any():
                class_scores[:, class_id] = torch.logsumexp(similarity[:, class_mask], dim=1)
        return class_scores


class FusionClassifier(nn.Module):
    def __init__(
        self,
        input_dim: int,
        num_labels: int,
        dropout: float,
        prototype_temperature: float,
        fixed_alpha: Optional[float] = None,
    ):
        super().__init__()
        self.mlp = EmotionMLP(input_dim, num_labels, dropout)
        self.prototype_classifier = PrototypeClassifier(num_labels, prototype_temperature)
        alpha_init = 0.0 if fixed_alpha is None else _alpha_to_logit(fixed_alpha)
        self.alpha_param = nn.Parameter(torch.tensor(alpha_init, dtype=torch.float32), requires_grad=fixed_alpha is None)

    def set_prototypes(self, prototype_vectors: torch.Tensor, prototype_labels: torch.Tensor) -> None:
        self.prototype_classifier.set_prototypes(prototype_vectors, prototype_labels)

    def has_prototypes(self) -> bool:
        return self.prototype_classifier.has_prototypes()

    def alpha(self) -> torch.Tensor:
        return torch.sigmoid(self.alpha_param)

    def forward_components(self, embeddings: torch.Tensor) -> Dict[str, torch.Tensor]:
        mlp_logits = self.mlp(embeddings)
        mlp_probs = F.softmax(mlp_logits, dim=-1)
        if not self.has_prototypes():
            final_probs = mlp_probs
            proto_class_scores = embeddings.new_zeros(mlp_logits.shape)
            proto_probs = mlp_probs.new_zeros(mlp_probs.shape)
        else:
            proto_class_scores = self.prototype_classifier(embeddings)
            proto_probs = F.softmax(proto_class_scores, dim=-1)
            alpha = self.alpha().to(embeddings.device)
            final_probs = alpha * mlp_probs + (1.0 - alpha) * proto_probs
        final_log_probs = torch.log(final_probs.clamp_min(1e-12))
        return {
            "mlp_logits": mlp_logits,
            "proto_class_scores": proto_class_scores,
            "proto_probs": proto_probs,
            "final_probs": final_probs,
            "final_log_probs": final_log_probs,
        }

    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
        return self.forward_components(embeddings)["final_log_probs"]

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


PROMPT_SECTION_BOUNDARY = (
    r"\n\s*### (?:"
    r"Given the following conversation as a context|"
    r"Visual Expressions of the speaker present in this utterance|"
    r"Audio Characteristics of the speaker present in this utterance|"
    r"Audio and Visual Descriptions of the speaker present in this utterance|"
    r"Given the characteristic of this speaker|"
    r"Given the speaker(?:'|’)?s Explicit Emotion Interpretation and Implicit Emotion Interpretation|"
    r"Semantic Contrastive Cues for the speaker in this utterance|"
    r"Reference Similar Emotional Expressions|"
    r"Available emotion labels"
    r")"
)


def extract_prompt_section(text: str, start_pattern: str) -> str:
    return section_between(text, start_pattern, [PROMPT_SECTION_BOUNDARY])


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

    context = extract_prompt_section(
        system_text,
        r"### Given the following conversation as a context\s*",
    )
    speaker_desc = extract_prompt_section(
        system_text,
        r"### Given the characteristic of this speaker:\s*",
    )
    emotion_block = extract_prompt_section(
        system_text,
        r"### Given the speaker(?:'|’)?s Explicit Emotion Interpretation and Implicit Emotion Interpretation.*?:\s*",
    )
    visual_desc = extract_prompt_section(
        system_text,
        r"### Visual Expressions of the speaker present in this utterance:\s*",
    )
    audio_desc = extract_prompt_section(
        system_text,
        r"### Audio Characteristics of the speaker present in this utterance:\s*",
    )
    llm_audio_visual_desc = extract_prompt_section(
        system_text,
        r"### Audio and Visual Descriptions of the speaker present in this utterance:\s*",
    )
    semantic_cues = extract_prompt_section(
        system_text,
        r"### Semantic Contrastive Cues for the speaker in this utterance:\s*",
    )
    reference_similar = extract_prompt_section(
        system_text,
        r"### Reference Similar Emotional Expressions:\s*",
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
        llm_audio_visual_desc=llm_audio_visual_desc,
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
    llm_audio_visual_desc: str,
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
    if args.include_visual_description:
        sections.append(f"Visual Description:\n{visual_desc or 'Not available.'}")
    if args.include_audio_description:
        sections.append(f"Audio Description:\n{audio_desc or 'Not available.'}")
    if args.include_llm_aud_vis_desc:
        sections.append(f"Audio and Visual Descriptions:\n{llm_audio_visual_desc or 'Not available.'}")
    if args.include_speaker_description:
        sections.append(f"Speaker Description:\n{speaker_desc or 'Not available.'}")
    if args.include_explicit_emotion:
        sections.append(f"Explicit Emotion:\n{explicit or 'Not available.'}")
    if args.include_implicit_emotion:
        sections.append(f"Implicit Emotion:\n{implicit or 'Not available.'}")
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
                llm_audio_visual_desc="",
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
    if args.lora_contrastive_finetune and args.mode in {"train_eval", "train"}:
        model.train()
    else:
        model.eval()
    return tokenizer, model


def apply_lora_to_embedding_model(model, args: argparse.Namespace):
    try:
        from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    except ImportError as exc:
        raise ImportError("LoRA contrastive fine-tuning requires peft.") from exc

    if hasattr(model, "config"):
        model.config.use_cache = False

    if args.load_in_4bit or args.load_in_8bit:
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=args.gradient_checkpointing)
    elif args.gradient_checkpointing and hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()

    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()

    target_modules = [module.strip() for module in args.lora_target_modules.split(",") if module.strip()]
    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        target_modules=target_modules,
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    model.train()
    return model


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
    was_training = model.training
    model.eval()
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
        if args.l2_normalize:
            embeddings = F.normalize(embeddings, p=2, dim=1)
        all_embeddings.append(embeddings.detach().float().cpu())
    if was_training:
        model.train()
    return torch.cat(all_embeddings, dim=0)


def encode_batch(batch: Dict[str, torch.Tensor], model, args: argparse.Namespace) -> torch.Tensor:
    outputs = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"])
    embeddings = last_token_pool(outputs.last_hidden_state, batch["attention_mask"])
    if args.l2_normalize:
        embeddings = F.normalize(embeddings, p=2, dim=1)
    return embeddings


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


def _normalize_embedding_batch(x: torch.Tensor) -> torch.Tensor:
    return F.normalize(x, p=2, dim=1)


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


class TextEmotionDataset(Dataset):
    def __init__(self, samples: Sequence[EmotionSample], label_ids: torch.Tensor):
        self.samples = list(samples)
        self.label_ids = label_ids

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[str, torch.Tensor]:
        return self.samples[idx].text, self.label_ids[idx]


def make_text_collator(tokenizer, args: argparse.Namespace):
    def collate(batch: Sequence[Tuple[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        texts = [item[0] for item in batch]
        labels = torch.stack([item[1] for item in batch]).long()
        max_length = args.lora_max_length if args.lora_contrastive_finetune and args.lora_max_length else args.max_length
        tokenized = tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        tokenized["labels"] = labels
        return tokenized

    return collate


class BalancedPairBatchSampler(BatchSampler):
    def __init__(self, label_ids: torch.Tensor, batch_size: int):
        if batch_size < 2:
            raise ValueError("--lora_batch_size must be at least 2 for supervised contrastive learning.")
        self.label_ids = label_ids.cpu()
        self.batch_size = batch_size
        self.class_to_indices = {
            int(label): torch.where(self.label_ids == label)[0].tolist()
            for label in torch.unique(self.label_ids)
        }
        self.classes = list(self.class_to_indices.keys())
        self.num_batches = int(np.ceil(len(self.label_ids) / batch_size))

    def __iter__(self):
        for _ in range(self.num_batches):
            batch = []
            pairs_needed = max(1, self.batch_size // 2)
            if len(self.classes) >= 2:
                replace_classes = pairs_needed > len(self.classes)
                selected_positions = np.random.choice(len(self.classes), size=pairs_needed, replace=replace_classes)
                selected_classes = [self.classes[int(pos)] for pos in selected_positions]
            else:
                selected_classes = [self.classes[0]] * pairs_needed

            for class_id in selected_classes:
                indices = self.class_to_indices[class_id]
                replace = len(indices) < 2
                sampled = np.random.choice(indices, size=2, replace=replace).tolist()
                batch.extend(int(idx) for idx in sampled)

            while len(batch) < self.batch_size:
                class_id = random.choice(self.classes)
                batch.append(int(random.choice(self.class_to_indices[class_id])))
            random.shuffle(batch)
            yield batch[: self.batch_size]

    def __len__(self) -> int:
        return self.num_batches


class ClassBalancedMemoryBank:
    def __init__(self, num_labels: int, memory_per_class: int):
        if memory_per_class < 1:
            raise ValueError("--memory_per_class must be at least 1.")
        self.num_labels = num_labels
        self.memory_per_class = memory_per_class
        self.embedding_queues: Dict[int, Deque[torch.Tensor]] = {
            class_id: deque(maxlen=memory_per_class) for class_id in range(num_labels)
        }
        self.label_queues: Dict[int, Deque[int]] = {
            class_id: deque(maxlen=memory_per_class) for class_id in range(num_labels)
        }

    def get_class_embeddings(self, class_id: int, device: torch.device) -> torch.Tensor:
        queue = self.embedding_queues[int(class_id)]
        if not queue:
            return torch.empty((0, 0), device=device)
        return torch.stack(list(queue), dim=0).to(device=device)

    def enqueue(self, embeddings: torch.Tensor, labels: torch.Tensor) -> None:
        norm_embeddings = F.normalize(embeddings.detach().float(), p=2, dim=1)
        norm_labels = labels.detach().long()
        for embedding, label in zip(norm_embeddings, norm_labels):
            class_id = int(label.item())
            self.embedding_queues[class_id].append(embedding)
            self.label_queues[class_id].append(class_id)


def supervised_contrastive_loss(
    embeddings: torch.Tensor,
    labels: torch.Tensor,
    temperature: float,
) -> torch.Tensor:
    if embeddings.size(0) < 2:
        return embeddings.new_tensor(0.0)
    if labels.unique().numel() < 2:
        return embeddings.new_tensor(0.0)

    embeddings = F.normalize(embeddings, p=2, dim=1)
    labels = labels.view(-1, 1)
    positive_mask = torch.eq(labels, labels.T).float().to(embeddings.device)
    logits_mask = torch.ones_like(positive_mask) - torch.eye(positive_mask.size(0), device=embeddings.device)
    positive_mask = positive_mask * logits_mask

    positives_per_anchor = positive_mask.sum(dim=1)
    valid_anchor_mask = positives_per_anchor > 0
    if not valid_anchor_mask.any():
        return embeddings.new_tensor(0.0)

    logits = torch.matmul(embeddings, embeddings.T) / temperature
    logits = logits - logits.max(dim=1, keepdim=True).values.detach()
    exp_logits = torch.exp(logits) * logits_mask
    log_prob = logits - torch.log(exp_logits.sum(dim=1, keepdim=True).clamp_min(1e-12))
    mean_log_prob_pos = (positive_mask * log_prob).sum(dim=1) / positives_per_anchor.clamp_min(1.0)
    return -mean_log_prob_pos[valid_anchor_mask].mean()


def memory_bank_local_supcon_loss(
    embeddings: torch.Tensor,
    labels: torch.Tensor,
    temperature: float,
    memory_bank: ClassBalancedMemoryBank,
    top_k_pos: int,
    top_m_neg: int,
) -> Optional[torch.Tensor]:
    if embeddings.size(0) == 0:
        return embeddings.new_tensor(0.0)
    if top_k_pos < 1:
        raise ValueError("--top_k_pos must be at least 1.")
    if top_m_neg < 1:
        raise ValueError("--top_m_neg must be at least 1.")

    norm_embeddings = F.normalize(embeddings, p=2, dim=1)
    class_embeddings = {
        class_id: memory_bank.get_class_embeddings(class_id, norm_embeddings.device)
        for class_id in range(memory_bank.num_labels)
    }

    losses = []
    for idx, anchor in enumerate(norm_embeddings):
        class_id = int(labels[idx].item())
        positive_bank = class_embeddings.get(class_id)
        if positive_bank is None or positive_bank.size(0) < top_k_pos:
            return None

        negative_candidates = [
            queued_embeddings
            for other_class_id, queued_embeddings in class_embeddings.items()
            if other_class_id != class_id and queued_embeddings.numel() > 0
        ]
        if not negative_candidates:
            return None
        negative_bank = torch.cat(negative_candidates, dim=0)
        if negative_bank.size(0) < top_m_neg:
            return None

        positive_similarity = torch.matmul(positive_bank, anchor)
        top_positive_values = torch.topk(
            positive_similarity,
            k=min(top_k_pos, positive_similarity.size(0)),
            largest=True,
        ).values / temperature

        negative_similarity = torch.matmul(negative_bank, anchor)
        top_negative_values = torch.topk(
            negative_similarity,
            k=min(top_m_neg, negative_similarity.size(0)),
            largest=True,
        ).values / temperature

        numerator = torch.logsumexp(top_positive_values, dim=0)
        denominator = torch.logsumexp(torch.cat([top_positive_values, top_negative_values], dim=0), dim=0)
        losses.append(-(numerator - denominator))

    if not losses:
        return embeddings.new_tensor(0.0)
    return torch.stack(losses).mean()


def _train_step_prediction(model: nn.Module, embeddings: torch.Tensor, labels: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    if isinstance(model, FusionClassifier):
        outputs = model.forward_components(embeddings)
        loss = F.nll_loss(outputs["final_log_probs"], labels)
        predictions = outputs["final_log_probs"].argmax(dim=-1)
        return loss, predictions
    logits = model(embeddings)
    loss = F.cross_entropy(logits, labels)
    predictions = logits.argmax(dim=-1)
    return loss, predictions


def load_classifier_state(model: nn.Module, state_dict: Dict[str, torch.Tensor]) -> None:
    if isinstance(model, FusionClassifier):
        prototype_vectors = state_dict.get("prototype_classifier.prototype_vectors")
        prototype_labels = state_dict.get("prototype_classifier.prototype_labels")
        filtered_state = {
            key: value
            for key, value in state_dict.items()
            if key not in {"prototype_classifier.prototype_vectors", "prototype_classifier.prototype_labels"}
        }
        if prototype_vectors is not None and prototype_labels is not None:
            if prototype_vectors.numel() == 0:
                model.prototype_classifier.prototype_vectors = torch.empty(
                    (0, 0), device=model.prototype_classifier.prototype_vectors.device
                )
                model.prototype_classifier.prototype_labels = torch.empty(
                    (0,), dtype=torch.long, device=model.prototype_classifier.prototype_labels.device
                )
            else:
                model.set_prototypes(prototype_vectors, prototype_labels)
        model.load_state_dict(filtered_state, strict=False)
        return
    model.load_state_dict(state_dict, strict=False)


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


def train_lora_contrastive(
    embedding_model,
    tokenizer,
    train_samples: Sequence[EmotionSample],
    train_labels: torch.Tensor,
    valid_samples: Sequence[EmotionSample],
    valid_labels: Optional[torch.Tensor],
    args: argparse.Namespace,
) -> nn.Module:
    device = embedding_model.device
    classifier: nn.Module
    prototype_manager = None
    prototype_supcon_loss = None
    prototype_refresh_loader = None
    memory_bank = None
    if args.prototype_learning:
        classifier = FusionClassifier(
            args.embedding_dim,
            args.num_labels,
            args.dropout,
            args.prototype_temperature,
            args.prototype_fixed_alpha,
        ).to(device)
        prototype_manager = PrototypeManager(args)
        prototype_supcon_loss = SamplePrototypeSupConLoss(args.prototype_temperature).to(device)
    else:
        classifier = EmotionMLP(args.embedding_dim, args.num_labels, args.dropout).to(device)
    if args.use_memory_bank_supcon:
        memory_bank = ClassBalancedMemoryBank(args.num_labels, args.memory_per_class)
    train_dataset = TextEmotionDataset(train_samples, train_labels)
    class_counts = torch.bincount(train_labels, minlength=args.num_labels)
    print(f"LoRA SupCon class counts: {class_counts.tolist()}")
    if args.lora_batch_size < 2 * int((class_counts > 0).sum().item()):
        print(
            "Warning: --lora_batch_size cannot include all classes with positive pairs in every batch. "
            "The balanced pair sampler still forces same-label positives and rotates classes across batches."
        )
    train_loader = DataLoader(
        train_dataset,
        batch_sampler=BalancedPairBatchSampler(train_labels, args.lora_batch_size),
        collate_fn=make_text_collator(tokenizer, args),
    )
    if args.prototype_learning:
        prototype_refresh_loader = DataLoader(
            train_dataset,
            batch_size=args.embed_batch_size,
            shuffle=False,
            collate_fn=make_text_collator(tokenizer, args),
        )

    valid_loader = None
    if valid_samples and valid_labels is not None:
        valid_loader = DataLoader(
            TextEmotionDataset(valid_samples, valid_labels),
            batch_size=args.embed_batch_size,
            shuffle=False,
            collate_fn=make_text_collator(tokenizer, args),
        )

    trainable_params = [param for param in embedding_model.parameters() if param.requires_grad]
    if not trainable_params:
        raise ValueError(
            "No trainable LoRA parameters were found. Check --lora_target_modules for this embedding model."
        )
    classifier_params = [param for param in classifier.parameters() if param.requires_grad]
    optimizer = torch.optim.AdamW(
        [{"params": trainable_params, "lr": args.lora_lr}, {"params": classifier_params, "lr": args.lr}],
        weight_decay=args.weight_decay,
    )

    best_state = None
    best_valid_loss = float("inf")
    for epoch in range(1, args.epochs + 1):
        embedding_model.train()
        classifier.train()
        if args.prototype_learning and prototype_manager is not None and prototype_refresh_loader is not None:
            should_recompute = epoch > args.prototype_warmup_epochs and (
                epoch == args.prototype_warmup_epochs + 1
                or args.prototype_recompute_every_epoch
                or not classifier.has_prototypes()
            )
            if should_recompute:
                train_embeddings_epoch, train_labels_epoch = embed_from_loader(
                    prototype_refresh_loader,
                    tokenizer,
                    embedding_model,
                    args,
                    f"train-prototypes-epoch{epoch}",
                )
                prototype_vectors, prototype_labels = prototype_manager.recompute(
                    train_embeddings_epoch,
                    train_labels_epoch,
                    args.num_labels,
                )
                classifier.set_prototypes(
                    prototype_vectors.to(device=device, dtype=torch.float32),
                    prototype_labels.to(device=device, dtype=torch.long),
                )
                print(f"Prototype update: epoch={epoch} total_prototypes={int(prototype_labels.numel())}")
        total_ce = 0.0
        total_supcon = 0.0
        total_proto_supcon = 0.0
        total_loss = 0.0
        memory_bank_batches = 0
        fallback_supcon_batches = 0
        total_correct = 0
        total_seen = 0

        optimizer.zero_grad(set_to_none=True)
        for batch in tqdm(train_loader, desc=f"LoRA SupCon epoch {epoch}/{args.epochs}"):
            batch = {key: value.to(device) for key, value in batch.items()}
            y = batch.pop("labels")
            embeddings = encode_batch(batch, embedding_model, args)
            ce_loss, predictions = _train_step_prediction(classifier, embeddings, y)
            if memory_bank is not None:
                memory_bank_supcon = memory_bank_local_supcon_loss(
                    embeddings,
                    y,
                    args.supcon_temperature,
                    memory_bank,
                    args.top_k_pos,
                    args.top_m_neg,
                )
                if memory_bank_supcon is None:
                    supcon_loss = supervised_contrastive_loss(embeddings, y, args.supcon_temperature)
                    fallback_supcon_batches += 1
                else:
                    supcon_loss = memory_bank_supcon
                    memory_bank_batches += 1
            else:
                supcon_loss = supervised_contrastive_loss(embeddings, y, args.supcon_temperature)
            if args.prototype_learning and prototype_supcon_loss is not None and isinstance(classifier, FusionClassifier):
                proto_supcon_value = prototype_supcon_loss(
                    embeddings,
                    y,
                    classifier.prototype_classifier.prototype_vectors,
                    classifier.prototype_classifier.prototype_labels,
                )
            else:
                proto_supcon_value = embeddings.new_tensor(0.0)
            loss = ce_loss + args.supcon_lambda * supcon_loss + args.prototype_supcon_lambda * proto_supcon_value
            (loss / args.lora_grad_accum_steps).backward()

            is_update_step = (total_seen // y.size(0) + 1) % args.lora_grad_accum_steps == 0
            is_last_step = (total_seen + y.size(0)) >= len(train_loader) * args.lora_batch_size
            if is_update_step or is_last_step:
                torch.nn.utils.clip_grad_norm_(trainable_params + classifier_params, args.max_grad_norm)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)

            total_ce += ce_loss.item() * y.size(0)
            total_supcon += supcon_loss.item() * y.size(0)
            total_proto_supcon += proto_supcon_value.item() * y.size(0)
            total_loss += loss.item() * y.size(0)
            total_correct += (predictions == y).sum().item()
            total_seen += y.size(0)
            if memory_bank is not None:
                memory_bank.enqueue(embeddings, y)

        train_loss = total_loss / max(total_seen, 1)
        train_ce = total_ce / max(total_seen, 1)
        train_supcon = total_supcon / max(total_seen, 1)
        train_proto_supcon = total_proto_supcon / max(total_seen, 1)
        train_acc = total_correct / max(total_seen, 1)

        valid_loss = None
        valid_acc = None
        if valid_loader is not None:
            valid_embeddings, valid_y = embed_from_loader(valid_loader, tokenizer, embedding_model, args, "valid-lora")
            valid_loss, valid_acc, _ = predict_mlp(classifier, valid_embeddings, valid_y, args)
            if valid_loss < best_valid_loss:
                best_valid_loss = valid_loss
                best_state = {
                    "classifier": {k: v.detach().cpu().clone() for k, v in classifier.state_dict().items()},
                    "lora": {k: v.detach().cpu().clone() for k, v in embedding_model.state_dict().items() if "lora_" in k},
                }

        if valid_loss is None:
            print(
                f"Epoch {epoch}: train_loss={train_loss:.4f} ce={train_ce:.4f} "
                f"supcon={train_supcon:.4f} proto_supcon={train_proto_supcon:.4f} train_acc={train_acc:.4f}"
                f"{f' memory_bank_batches={memory_bank_batches} fallback_batches={fallback_supcon_batches}' if memory_bank is not None else ''}"
            )
        else:
            print(
                f"Epoch {epoch}: train_loss={train_loss:.4f} ce={train_ce:.4f} "
                f"supcon={train_supcon:.4f} proto_supcon={train_proto_supcon:.4f} train_acc={train_acc:.4f} "
                f"valid_loss={valid_loss:.4f} valid_acc={valid_acc:.4f}"
                f"{f' memory_bank_batches={memory_bank_batches} fallback_batches={fallback_supcon_batches}' if memory_bank is not None else ''}"
            )

    if best_state is not None:
        load_classifier_state(classifier, best_state["classifier"])
        embedding_model.load_state_dict(best_state["lora"], strict=False)
    return classifier


def train_proj_supcon(
    train_embeddings: torch.Tensor,
    train_labels: torch.Tensor,
    valid_embeddings: Optional[torch.Tensor],
    valid_labels: Optional[torch.Tensor],
    args: argparse.Namespace,
) -> ProjectionSupConClassifier:
    device = torch.device("cuda" if torch.cuda.is_available() and not args.force_cpu_mlp else "cpu")
    model = ProjectionSupConClassifier(args.embedding_dim, args.num_labels).to(device)
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
        total_ce = 0.0
        total_supcon = 0.0
        total_correct = 0
        total_seen = 0
        for x, y in tqdm(train_loader, desc=f"ProjSupCon epoch {epoch}/{args.epochs}"):
            x = _normalize_embedding_batch(x.to(device))
            y = y.to(device)
            optimizer.zero_grad(set_to_none=True)
            projected, logits = model.forward_with_projection(x)
            ce_loss = F.cross_entropy(logits, y)
            supcon_loss = supervised_contrastive_loss(projected, y, args.supcon_temperature)
            loss = ce_loss + args.supcon_lambda * supcon_loss
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * y.size(0)
            total_ce += ce_loss.item() * y.size(0)
            total_supcon += supcon_loss.item() * y.size(0)
            total_correct += (logits.argmax(dim=-1) == y).sum().item()
            total_seen += y.size(0)

        train_loss = total_loss / max(total_seen, 1)
        train_ce = total_ce / max(total_seen, 1)
        train_supcon = total_supcon / max(total_seen, 1)
        train_acc = total_correct / max(total_seen, 1)

        valid_loss = None
        valid_acc = None
        if valid_embeddings is not None and valid_labels is not None:
            valid_loss, valid_acc, _ = predict_mlp(model, valid_embeddings, valid_labels, args)
            if valid_loss < best_valid_loss:
                best_valid_loss = valid_loss
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

        if valid_loss is None:
            print(
                f"Epoch {epoch}: train_loss={train_loss:.4f} ce={train_ce:.4f} "
                f"supcon={train_supcon:.4f} train_acc={train_acc:.4f}"
            )
        else:
            print(
                f"Epoch {epoch}: train_loss={train_loss:.4f} ce={train_ce:.4f} "
                f"supcon={train_supcon:.4f} train_acc={train_acc:.4f} "
                f"valid_loss={valid_loss:.4f} valid_acc={valid_acc:.4f}"
            )

    if best_state is not None:
        model.load_state_dict(best_state)
    return model


@torch.no_grad()
def embed_from_loader(
    loader: DataLoader,
    tokenizer,
    embedding_model,
    args: argparse.Namespace,
    split_name: str,
) -> Tuple[torch.Tensor, torch.Tensor]:
    del tokenizer
    was_training = embedding_model.training
    embedding_model.eval()
    device = embedding_model.device
    all_embeddings = []
    all_labels = []
    for batch in tqdm(loader, desc=f"Embedding {split_name}"):
        batch = {key: value.to(device) for key, value in batch.items()}
        y = batch.pop("labels")
        embeddings = encode_batch(batch, embedding_model, args)
        all_embeddings.append(embeddings.detach().float().cpu())
        all_labels.append(y.detach().cpu())
    if was_training:
        embedding_model.train()
    return torch.cat(all_embeddings, dim=0), torch.cat(all_labels, dim=0)


@torch.no_grad()
def predict_mlp(
    model: nn.Module,
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
        if isinstance(model, FusionClassifier):
            log_probs = model(x)
            pred = log_probs.argmax(dim=-1)
        else:
            logits = model(x)
            pred = logits.argmax(dim=-1)
        preds.extend(pred.detach().cpu().tolist())
        if labels is not None:
            if isinstance(model, FusionClassifier):
                loss = F.nll_loss(log_probs, y)
            else:
                loss = F.cross_entropy(logits, y)
            losses.append(loss.item() * y.size(0))
            correct += (pred == y).sum().item()
            seen += y.size(0)
    loss_value = sum(losses) / max(seen, 1) if labels is not None else None
    acc_value = correct / max(seen, 1) if labels is not None else None
    return loss_value, acc_value, np.array(preds)


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


def get_class_colors(num_labels: int) -> List[str]:
    palette = [
        "#1f77b4",
        "#d62728",
        "#2ca02c",
        "#ff7f0e",
        "#9467bd",
        "#17becf",
        "#e377c2",
        "#8c564b",
        "#bcbd22",
        "#7f7f7f",
    ]
    return palette[:num_labels]


def compute_projection(
    method: str,
    embeddings: np.ndarray,
    labels_np: np.ndarray,
    seed: int,
) -> Tuple[np.ndarray, float]:
    try:
        from sklearn.decomposition import PCA
        from sklearn.manifold import TSNE
        from sklearn.metrics import silhouette_score
    except Exception as exc:
        raise ImportError(
            "Embedding analysis requires scikit-learn for PCA, t-SNE, and silhouette score."
        ) from exc

    if len(embeddings) < 2:
        projection = np.zeros((len(embeddings), 2), dtype=np.float32)
    elif method == "pca":
        projection = PCA(n_components=2, random_state=seed).fit_transform(embeddings)
    elif method == "tsne":
        if len(embeddings) < 4:
            projection = PCA(n_components=2, random_state=seed).fit_transform(embeddings)
        else:
            perplexity = min(30, max(2, (len(embeddings) - 1) // 3))
            perplexity = min(perplexity, len(embeddings) - 1)
            projection = TSNE(
                n_components=2,
                perplexity=perplexity,
                init="pca",
                learning_rate="auto",
                random_state=seed,
            ).fit_transform(embeddings)
    elif method == "umap":
        if len(embeddings) < 4:
            projection = PCA(n_components=2, random_state=seed).fit_transform(embeddings)
        else:
            try:
                import umap
            except Exception as exc:
                raise ImportError("Embedding analysis requires umap-learn for UMAP visualization.") from exc
            n_neighbors = min(15, max(2, len(embeddings) - 1))
            projection = umap.UMAP(
                n_components=2,
                n_neighbors=n_neighbors,
                min_dist=0.1,
                metric="cosine",
                random_state=seed,
            ).fit_transform(embeddings)
    else:
        raise ValueError(f"Unknown projection method: {method}")

    unique_labels = np.unique(labels_np)
    if len(unique_labels) < 2 or len(unique_labels) >= len(labels_np):
        silhouette = float("nan")
    else:
        silhouette = float(silhouette_score(embeddings, labels_np, metric="cosine"))
    return projection, silhouette


def add_color_key(fig, labels: Sequence[str], colors: Sequence[str]) -> None:
    fig.text(0.06, 0.018, "Class colors:", ha="left", va="center", fontsize=14, fontweight="bold")
    x = 0.17
    step = 0.80 / max(len(labels), 1)
    for label, color in zip(labels, colors):
        fig.text(x, 0.018, label, ha="left", va="center", fontsize=14, color=color, fontweight="bold")
        x += step


def save_embedding_analysis(
    split_payloads: Dict[str, Tuple[Sequence[EmotionSample], torch.Tensor, torch.Tensor]],
    labels: Sequence[str],
    args: argparse.Namespace,
) -> Path:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:
        raise ImportError("Embedding analysis requires matplotlib.") from exc

    methods = [("tsne", "t-SNE"), ("pca", "PCA"), ("umap", "UMAP")]
    split_order = ["train", "valid", "test"]
    colors = get_class_colors(len(labels))
    fig, axes = plt.subplots(len(split_order), len(methods), figsize=(27, 24))
    row_silhouettes: Dict[str, float] = {}

    for row_idx, split_name in enumerate(split_order):
        samples, embeddings, y = split_payloads[split_name]
        x_np = embeddings.numpy()
        x_np = x_np / np.clip(np.linalg.norm(x_np, axis=1, keepdims=True), 1e-12, None)
        y_np = y.numpy()
        for col_idx, (method_key, method_title) in enumerate(methods):
            ax = axes[row_idx, col_idx]
            projection, silhouette = compute_projection(method_key, x_np, y_np, args.seed)
            row_silhouettes.setdefault(split_name, silhouette)
            for label_idx, label_name in enumerate(labels):
                mask = y_np == label_idx
                if mask.any():
                    ax.scatter(
                        projection[mask, 0],
                        projection[mask, 1],
                        s=13,
                        alpha=0.78,
                        color=colors[label_idx],
                        linewidths=0,
                    )
            ax.set_title(f"{split_name.upper()} - {method_title}", fontsize=18, fontweight="bold")
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_xlabel("")
            ax.set_ylabel("")
            ax.grid(False)

    fig.subplots_adjust(left=0.035, right=0.985, top=0.955, bottom=0.08, hspace=0.26, wspace=0.055)
    for row_idx, split_name in enumerate(split_order):
        row_axes = axes[row_idx, :]
        bottom = min(ax.get_position().y0 for ax in row_axes)
        silhouette = row_silhouettes.get(split_name, float("nan"))
        silhouette_text = "nan" if np.isnan(silhouette) else f"{silhouette:.4f}"
        fig.text(
            0.5,
            bottom - 0.018,
            f"{split_name.upper()} silhouette score: {silhouette_text}",
            ha="center",
            va="top",
            fontsize=16,
            fontweight="bold",
        )
    add_color_key(fig, labels, colors)

    output_dir = Path("/scratch/data/bikash_rs/Vivek/PRC-Emo/analysis/embed_analysis")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{get_run_stem(args)}_{get_timestamp_suffix()}_embedding_analysis.png"
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved embedding analysis visualization to {output_path}")
    return output_path


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


def load_lora_adapter_from_checkpoint(path: Path, embedding_model):
    checkpoint = torch.load(path, map_location="cpu")
    lora_adapter_path = checkpoint.get("lora_adapter_path")
    if not lora_adapter_path:
        return embedding_model
    try:
        from peft import PeftModel
    except ImportError as exc:
        raise ImportError("Loading a LoRA adapter checkpoint requires peft.") from exc
    print(f"Loading LoRA adapter from {lora_adapter_path}")
    model = PeftModel.from_pretrained(embedding_model, lora_adapter_path)
    model.eval()
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
        raise ValueError("--prototype_learning is supported only with the LoRA contrastive fine-tuning pipeline.")
    if args.prototype_learning and not args.lora_contrastive_finetune:
        raise ValueError("--prototype_learning requires --lora_contrastive_finetune.")
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
