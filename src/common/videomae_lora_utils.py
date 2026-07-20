import argparse
from types import MethodType


def _videomae_make_embedding_outputs_require_grad(module, inputs, outputs):
    del module, inputs
    outputs.requires_grad_(True)
    return outputs


def _patch_videomae_enable_input_require_grads(model) -> None:
    if getattr(model, "_videomae_input_require_grads_patched", False):
        return

    def enable_input_require_grads(self):
        old_hook = getattr(self, "_videomae_require_grads_hook", None)
        if old_hook is not None:
            old_hook.remove()

        embeddings = getattr(self, "embeddings", None)
        if embeddings is None:
            raise AttributeError("VideoMAE model does not expose `embeddings` for gradient-requiring hook setup.")
        self._videomae_require_grads_hook = embeddings.register_forward_hook(
            _videomae_make_embedding_outputs_require_grad
        )

    model.enable_input_require_grads = MethodType(enable_input_require_grads, model)
    model._videomae_input_require_grads_patched = True


def _supports_gradient_checkpointing(model) -> bool:
    return bool(getattr(model, "supports_gradient_checkpointing", False)) and hasattr(
        model, "gradient_checkpointing_enable"
    )


def apply_lora_to_videomae_embedding_model(model, args: argparse.Namespace):
    try:
        from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    except ImportError as exc:
        raise ImportError("LoRA contrastive fine-tuning for VideoMAE requires peft.") from exc

    if hasattr(model, "config"):
        model.config.use_cache = False

    _patch_videomae_enable_input_require_grads(model)

    use_gradient_checkpointing = bool(args.gradient_checkpointing and _supports_gradient_checkpointing(model))

    if args.load_in_4bit or args.load_in_8bit:
        model = prepare_model_for_kbit_training(
            model,
            use_gradient_checkpointing=use_gradient_checkpointing,
        )
    elif use_gradient_checkpointing:
        model.gradient_checkpointing_enable()
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
