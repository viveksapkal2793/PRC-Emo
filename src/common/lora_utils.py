import argparse
from pathlib import Path
from typing import Dict, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm.auto import tqdm


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


def _safe_prepare_model_for_kbit_training(model, use_gradient_checkpointing: bool):
    from peft import prepare_model_for_kbit_training

    try:
        return prepare_model_for_kbit_training(model, use_gradient_checkpointing=use_gradient_checkpointing)
    except (AttributeError, NotImplementedError, ValueError) as exc:
        print(
            "Warning: prepare_model_for_kbit_training could not enable gradient-checkpoint-related hooks "
            f"for {model.__class__.__name__}: {exc}. Retrying without gradient checkpointing."
        )
        return prepare_model_for_kbit_training(model, use_gradient_checkpointing=False)


def _safe_enable_gradient_checkpointing(model) -> None:
    if not hasattr(model, "gradient_checkpointing_enable"):
        return
    try:
        model.gradient_checkpointing_enable()
    except (AttributeError, NotImplementedError, ValueError) as exc:
        print(f"Warning: gradient checkpointing is unavailable for {model.__class__.__name__}: {exc}")


def _safe_enable_input_require_grads(model) -> None:
    if not hasattr(model, "enable_input_require_grads"):
        return
    try:
        model.enable_input_require_grads()
    except (AttributeError, NotImplementedError, ValueError) as exc:
        print(f"Warning: input require grads hook is unavailable for {model.__class__.__name__}: {exc}")


def _build_lora_model(model, args: argparse.Namespace):
    from peft import LoraConfig, get_peft_model

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


def apply_lora_to_wavlm_embedding_model(model, args: argparse.Namespace):
    try:
        import peft  # noqa: F401
    except ImportError as exc:
        raise ImportError("LoRA contrastive fine-tuning requires peft.") from exc

    if hasattr(model, "config"):
        model.config.use_cache = False

    if args.load_in_4bit or args.load_in_8bit:
        model = _safe_prepare_model_for_kbit_training(model, use_gradient_checkpointing=False)

    if args.gradient_checkpointing:
        _safe_enable_gradient_checkpointing(model)

    _safe_enable_input_require_grads(model)
    return _build_lora_model(model, args)


def apply_lora_to_videomae_embedding_model(model, args: argparse.Namespace):
    try:
        import peft  # noqa: F401
    except ImportError as exc:
        raise ImportError("LoRA contrastive fine-tuning requires peft.") from exc

    if hasattr(model, "config"):
        model.config.use_cache = False

    if args.load_in_4bit or args.load_in_8bit:
        model = _safe_prepare_model_for_kbit_training(model, use_gradient_checkpointing=False)

    if args.gradient_checkpointing:
        _safe_enable_gradient_checkpointing(model)

    # VideoMAE-style backbones do not expose text input embeddings, so this hook is best-effort only.
    _safe_enable_input_require_grads(model)
    return _build_lora_model(model, args)


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
    if hasattr(model, "encode"):
        embeddings = model.encode(batch)
    else:
        outputs = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"])
        embeddings = last_token_pool(outputs.last_hidden_state, batch["attention_mask"])
    embeddings = embeddings.float()
    if args.l2_normalize:
        embeddings = F.normalize(embeddings, p=2, dim=1)
    return embeddings


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
        try:
            embeddings = encode_batch(batch, embedding_model, args)
        except RuntimeError as exc:
            if "out of memory" not in str(exc).lower() or y.size(0) <= 1:
                raise
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            print(
                f"Warning: CUDA OOM while embedding {split_name} batch of size {y.size(0)}. "
                "Retrying one sample at a time."
            )
            per_sample_embeddings = []
            for idx in range(y.size(0)):
                single_batch = {key: value[idx : idx + 1] for key, value in batch.items()}
                try:
                    per_sample_embeddings.append(encode_batch(single_batch, embedding_model, args))
                except RuntimeError as single_exc:
                    if "out of memory" not in str(single_exc).lower():
                        raise
                    raise RuntimeError(
                        f"CUDA OOM while embedding a single sample in {split_name}. "
                        "Consider lowering the input duration limit for this modality."
                    ) from single_exc
            embeddings = torch.cat(per_sample_embeddings, dim=0)
        all_embeddings.append(embeddings.detach().float().cpu())
        all_labels.append(y.detach().cpu())
    if was_training:
        embedding_model.train()
    return torch.cat(all_embeddings, dim=0), torch.cat(all_labels, dim=0)


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
