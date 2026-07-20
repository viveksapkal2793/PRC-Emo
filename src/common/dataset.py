import argparse
import json
import random
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np
import torch
import torchaudio
from torch.utils.data import BatchSampler, DataLoader, Dataset

from .prompt import build_embedding_input, parse_prompt_sample
from .utils import normalize_text
from .types import EmotionSample


def parse_dialogue_json(path: Path, args: argparse.Namespace, labels: Sequence[str]) -> List[EmotionSample]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    samples: List[EmotionSample] = []
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
        samples: List[EmotionSample] = []
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


def labels_to_ids(samples: Sequence[EmotionSample], label_to_id: Dict[str, int]) -> torch.Tensor:
    ids: List[int] = []
    for sample in samples:
        if sample.label not in label_to_id:
            raise KeyError(f"Unknown label '{sample.label}' in sample text: {sample.text}")
        ids.append(label_to_id[sample.label])
    return torch.tensor(ids, dtype=torch.long)


class TextEmotionDataset(Dataset):
    def __init__(self, samples: Sequence[EmotionSample], label_ids: torch.Tensor):
        self.samples = list(samples)
        self.label_ids = label_ids

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[str, torch.Tensor]:
        return self.samples[idx].text, self.label_ids[idx]


class AudioEmotionDataset(Dataset):
    def __init__(self, samples: Sequence[EmotionSample], label_ids: torch.Tensor):
        self.samples = list(samples)
        self.label_ids = label_ids

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[str, torch.Tensor]:
        return self.samples[idx].text, self.label_ids[idx]


class VideoEmotionDataset(Dataset):
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


class AudioCollator:
    def __init__(self, feature_extractor, target_sample_rate: int = 16000, max_audio_seconds: Optional[float] = None):
        self.feature_extractor = feature_extractor
        self.target_sample_rate = target_sample_rate
        self.max_audio_seconds = max_audio_seconds

    def _load_audio(self, audio_path: str) -> torch.Tensor:
        waveform, sample_rate = torchaudio.load(audio_path)
        if waveform.numel() == 0:
            raise ValueError(f"Audio file is empty: {audio_path}")
        if waveform.dim() != 2:
            raise ValueError(f"Unexpected waveform shape {tuple(waveform.shape)} for {audio_path}")
        if waveform.size(0) > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        if sample_rate != self.target_sample_rate:
            waveform = torchaudio.functional.resample(waveform, sample_rate, self.target_sample_rate)
        if self.max_audio_seconds is not None:
            max_num_samples = int(self.target_sample_rate * self.max_audio_seconds)
            if waveform.size(-1) > max_num_samples:
                waveform = waveform[..., :max_num_samples]
        return waveform.squeeze(0)

    def __call__(self, batch: Sequence[Tuple[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        audio_arrays = [self._load_audio(item[0]).numpy() for item in batch]
        labels = torch.stack([item[1] for item in batch]).long()
        encoded = self.feature_extractor(
            audio_arrays,
            sampling_rate=self.target_sample_rate,
            padding=True,
            return_tensors="pt",
        )
        encoded["labels"] = labels
        return encoded


def load_video_frames_uniform(video_path: str, num_frames: int) -> List[np.ndarray]:
    capture = cv2.VideoCapture(video_path)
    if not capture.isOpened():
        capture.release()
        raise ValueError(f"Unable to open video file: {video_path}")

    frames: List[np.ndarray] = []
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            if frame is None or frame.size == 0:
                continue
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    finally:
        capture.release()

    if not frames:
        raise ValueError(f"No decodable frames found in video: {video_path}")

    if len(frames) >= num_frames:
        indices = np.linspace(0, len(frames) - 1, num=num_frames, dtype=int)
        return [frames[int(idx)] for idx in indices]

    padded_frames = list(frames)
    last_frame = frames[-1]
    while len(padded_frames) < num_frames:
        padded_frames.append(last_frame.copy())
    return padded_frames


class VideoCollator:
    def __init__(self, image_processor, num_frames: int = 16):
        self.image_processor = image_processor
        self.num_frames = num_frames

    def _load_video(self, video_path: str) -> List[np.ndarray]:
        return load_video_frames_uniform(video_path, self.num_frames)

    def __call__(self, batch: Sequence[Tuple[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        videos = [self._load_video(item[0]) for item in batch]
        labels = torch.stack([item[1] for item in batch]).long()
        encoded = self.image_processor(videos, return_tensors="pt")
        encoded["labels"] = labels
        return encoded


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
