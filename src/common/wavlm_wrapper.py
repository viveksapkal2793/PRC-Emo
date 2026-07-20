from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn


class WavLMEmbeddingWrapper(nn.Module):
    embedding_dim = 1024

    def __init__(self, model, feature_extractor, pooling: str = "mean"):
        super().__init__()
        self.model = model
        self.feature_extractor = feature_extractor
        self.pooling = pooling

    @property
    def device(self) -> torch.device:
        return next(self.model.parameters()).device

    def _mean_pool(self, last_hidden_state: torch.Tensor, attention_mask: Optional[torch.Tensor]) -> torch.Tensor:
        if attention_mask is None:
            return last_hidden_state.mean(dim=1)
        if hasattr(self.model, "_get_feature_vector_attention_mask"):
            attention_mask = self.model._get_feature_vector_attention_mask(last_hidden_state.shape[1], attention_mask)
        mask = attention_mask.unsqueeze(-1).to(last_hidden_state.dtype)
        summed = (last_hidden_state * mask).sum(dim=1)
        denom = mask.sum(dim=1).clamp_min(1.0)
        return summed / denom

    def encode(self, batch) -> torch.Tensor:
        model_dtype = next(self.model.parameters()).dtype
        outputs = self.model(
            input_values=batch["input_values"].to(dtype=model_dtype),
            attention_mask=batch.get("attention_mask"),
        )
        if self.pooling != "mean":
            raise ValueError(f"Unsupported pooling mode: {self.pooling}")
        return self._mean_pool(outputs.last_hidden_state, batch.get("attention_mask"))

    def save(self, save_dir: str | Path) -> Path:
        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)
        self.model.save_pretrained(save_path)
        self.feature_extractor.save_pretrained(save_path)
        return save_path

    @classmethod
    def load(
        cls,
        model_path: str | Path,
        *,
        feature_extractor_path: Optional[str | Path] = None,
        load_in_4bit: bool = False,
        load_in_8bit: bool = False,
        attn_implementation: str = "",
        trust_remote_code: bool = True,
    ) -> "WavLMEmbeddingWrapper":
        try:
            from transformers import AutoFeatureExtractor, AutoModel
            from transformers import BitsAndBytesConfig
        except ImportError as exc:
            raise ImportError(
                "Loading WavLM embeddings requires transformers and huggingface_hub."
            ) from exc

        model_kwargs = {"trust_remote_code": trust_remote_code}
        if torch.cuda.is_available():
            model_kwargs["device_map"] = "auto"
            model_kwargs["torch_dtype"] = torch.float16
            if attn_implementation:
                model_kwargs["attn_implementation"] = attn_implementation
            if load_in_4bit or load_in_8bit:
                model_kwargs["quantization_config"] = BitsAndBytesConfig(
                    load_in_4bit=load_in_4bit,
                    load_in_8bit=load_in_8bit,
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_use_double_quant=True,
                )

        feature_extractor = AutoFeatureExtractor.from_pretrained(
            feature_extractor_path or model_path,
            trust_remote_code=trust_remote_code,
        )
        model = AutoModel.from_pretrained(model_path, **model_kwargs)
        if not torch.cuda.is_available():
            model.to("cpu")
        return cls(model=model, feature_extractor=feature_extractor)
