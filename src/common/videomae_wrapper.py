from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn


class VideoMAEEmbeddingWrapper(nn.Module):
    embedding_dim = 768

    def __init__(self, model, image_processor, pooling: str = "cls"):
        super().__init__()
        self.model = model
        self.image_processor = image_processor
        self.pooling = pooling

    @property
    def device(self) -> torch.device:
        return next(self.model.parameters()).device

    def _prepare_pixel_values(self, pixel_values: torch.Tensor) -> torch.Tensor:
        if pixel_values.ndim != 5:
            raise ValueError(f"Expected 5D video tensor, got shape {tuple(pixel_values.shape)}")

        # HF VideoMAE processors commonly emit (B, T, C, H, W), while some custom
        # VideoMAE variants expect (B, C, T, H, W) before Conv3d patch embedding.
        if pixel_values.shape[1] != 3 and pixel_values.shape[2] == 3:
            pixel_values = pixel_values.permute(0, 2, 1, 3, 4).contiguous()
        elif pixel_values.shape[1] == 3:
            pixel_values = pixel_values.contiguous()
        else:
            raise ValueError(
                "Unable to infer video tensor layout. Expected channels=3 in either dim 1 or dim 2, "
                f"got shape {tuple(pixel_values.shape)}"
            )
        return pixel_values

    def encode(self, batch) -> torch.Tensor:
        model_dtype = next(self.model.parameters()).dtype
        pixel_values = self._prepare_pixel_values(batch["pixel_values"]).to(dtype=model_dtype)
        outputs = self.model(pixel_values=pixel_values)

        # Custom VideoMAEv2 backbones may return the pooled embedding tensor directly.
        if isinstance(outputs, torch.Tensor):
            if outputs.ndim != 2:
                raise ValueError(f"Expected 2D embedding tensor from VideoMAE model, got shape {tuple(outputs.shape)}")
            return outputs

        if isinstance(outputs, tuple):
            if not outputs:
                raise ValueError("VideoMAE model returned an empty tuple.")
            first = outputs[0]
            if not isinstance(first, torch.Tensor):
                raise ValueError(f"Unexpected first tuple output type from VideoMAE model: {type(first)!r}")
            if first.ndim == 2:
                return first
            outputs = type("VideoMAEOutputs", (), {"last_hidden_state": first, "pooler_output": None})()

        if self.pooling != "cls":
            raise ValueError(f"Unsupported pooling mode: {self.pooling}")
        if hasattr(outputs, "pooler_output") and outputs.pooler_output is not None:
            return outputs.pooler_output
        if not hasattr(outputs, "last_hidden_state"):
            raise AttributeError(
                f"VideoMAE model output of type {type(outputs)!r} does not expose `last_hidden_state` or pooled tensor output."
            )
        return outputs.last_hidden_state[:, 0]

    def save(self, save_dir: str | Path) -> Path:
        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)
        self.model.save_pretrained(save_path)
        self.image_processor.save_pretrained(save_path)
        return save_path

    @classmethod
    def load(
        cls,
        model_path: str | Path,
        *,
        image_processor_path: Optional[str | Path] = None,
        load_in_4bit: bool = False,
        load_in_8bit: bool = False,
        attn_implementation: str = "",
        trust_remote_code: bool = True,
    ) -> "VideoMAEEmbeddingWrapper":
        try:
            from transformers import AutoImageProcessor, AutoModel
            from transformers import BitsAndBytesConfig
        except ImportError as exc:
            raise ImportError(
                "Loading VideoMAE embeddings requires transformers and huggingface_hub."
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

        image_processor = AutoImageProcessor.from_pretrained(
            image_processor_path or model_path,
            trust_remote_code=trust_remote_code,
        )
        model = AutoModel.from_pretrained(model_path, **model_kwargs)
        if not torch.cuda.is_available():
            model.to("cpu")
        return cls(model=model, image_processor=image_processor)
