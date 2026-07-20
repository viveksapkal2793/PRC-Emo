import argparse
from types import MethodType


def _wavlm_make_feature_projection_outputs_require_grad(module, inputs, outputs):
    del module, inputs
    if isinstance(outputs, tuple):
        hidden_states = outputs[0]
        hidden_states.requires_grad_(True)
        return (hidden_states, *outputs[1:])
    outputs.requires_grad_(True)
    return outputs


def _patch_wavlm_enable_input_require_grads(model) -> None:
    if getattr(model, "_wavlm_input_require_grads_patched", False):
        return

    def enable_input_require_grads(self):
        feature_encoder = getattr(self, "feature_extractor", None)
        if feature_encoder is not None and hasattr(feature_encoder, "_requires_grad"):
            feature_encoder._requires_grad = True

        old_hook = getattr(self, "_wavlm_require_grads_hook", None)
        if old_hook is not None:
            old_hook.remove()

        feature_projection = getattr(self, "feature_projection", None)
        if feature_projection is None:
            raise AttributeError("WavLM model does not expose `feature_projection` for gradient-requiring hook setup.")
        self._wavlm_require_grads_hook = feature_projection.register_forward_hook(
            _wavlm_make_feature_projection_outputs_require_grad
        )

    model.enable_input_require_grads = MethodType(enable_input_require_grads, model)
    model._wavlm_input_require_grads_patched = True


def _safe_enable_gradient_checkpointing(model) -> None:
    if not hasattr(model, "gradient_checkpointing_enable"):
        return
    try:
        model.gradient_checkpointing_enable()
    except (AttributeError, NotImplementedError, ValueError) as exc:
        print(f"Warning: gradient checkpointing is unavailable for {model.__class__.__name__}: {exc}")


def apply_lora_to_wavlm_embedding_model(model, args: argparse.Namespace):
    try:
        from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    except ImportError as exc:
        raise ImportError("LoRA contrastive fine-tuning for WavLM requires peft.") from exc

    if hasattr(model, "config"):
        model.config.use_cache = False

    _patch_wavlm_enable_input_require_grads(model)

    if args.load_in_4bit or args.load_in_8bit:
        try:
            model = prepare_model_for_kbit_training(
                model,
                use_gradient_checkpointing=args.gradient_checkpointing,
            )
        except (AttributeError, NotImplementedError, ValueError) as exc:
            print(
                "Warning: WavLM k-bit preparation could not enable gradient-checkpoint-related hooks "
                f"for {model.__class__.__name__}: {exc}. Retrying without gradient checkpointing."
            )
            model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=False)
    else:
        if args.gradient_checkpointing:
            _safe_enable_gradient_checkpointing(model)
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
