from typing import Callable, Optional, Sequence, Tuple
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from tqdm.auto import tqdm

from .dataset import BalancedPairBatchSampler, TextEmotionDataset, make_text_collator
from .lora_utils import encode_batch, embed_from_loader
from .losses import memory_bank_local_supcon_loss, supervised_contrastive_loss
from .memory import ClassBalancedMemoryBank
from .models import EmotionMLP, FusionClassifier, ProjectionSupConClassifier
from .prototype import PrototypeManager, SamplePrototypeSupConLoss

def _normalize_embedding_batch(x: torch.Tensor) -> torch.Tensor:
    return F.normalize(x, p=2, dim=1)

def train_step_prediction(model: nn.Module, embeddings: torch.Tensor, labels: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    if hasattr(model, "forward_components"):
        outputs = model.forward_components(embeddings)
        loss = F.nll_loss(outputs["final_log_probs"], labels)
        predictions = outputs["final_log_probs"].argmax(dim=-1)
        return loss, predictions
    logits = model(embeddings)
    loss = F.cross_entropy(logits, labels)
    predictions = logits.argmax(dim=-1)
    return loss, predictions

def predict_mlp(
    model: nn.Module,
    embeddings: torch.Tensor,
    labels: torch.Tensor,
    args,
) -> tuple[float, float, list[int]]:
    device = next(model.parameters()).device
    loader = DataLoader(
        TensorDataset(embeddings, labels),
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
        if hasattr(model, "forward_components"):
            log_probs = model(x)
            pred = log_probs.argmax(dim=-1)
            loss = F.nll_loss(log_probs, y)
        else:
            logits = model(x)
            pred = logits.argmax(dim=-1)
            loss = F.cross_entropy(logits, y)
        preds.extend(pred.detach().cpu().tolist())
        losses.append(loss.item() * y.size(0))
        correct += (pred == y).sum().item()
        seen += y.size(0)
    loss_value = sum(losses) / max(seen, 1)
    acc_value = correct / max(seen, 1)
    return loss_value, acc_value, preds

def load_classifier_state(model: nn.Module, state_dict: dict[str, torch.Tensor]) -> None:
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
    model_class: type[nn.Module] = EmotionMLP,
) -> nn.Module:
    device = torch.device("cuda" if torch.cuda.is_available() and not args.force_cpu_mlp else "cpu")
    model = model_class(args.embedding_dim, args.num_labels, args.dropout).to(device)
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

def train_frozen_prototype_mlp(
    train_embeddings: torch.Tensor,
    train_labels: torch.Tensor,
    valid_embeddings: Optional[torch.Tensor],
    valid_labels: Optional[torch.Tensor],
    args: argparse.Namespace,
) -> FusionClassifier:
    device = torch.device("cuda" if torch.cuda.is_available() and not args.force_cpu_mlp else "cpu")
    model = FusionClassifier(
        args.embedding_dim,
        args.num_labels,
        args.dropout,
        args.prototype_temperature,
        args.prototype_fixed_alpha,
    ).to(device)
    prototype_manager = PrototypeManager(args)
    prototype_vectors, prototype_labels = prototype_manager.recompute(
        train_embeddings,
        train_labels,
        args.num_labels,
    )
    model.set_prototypes(
        prototype_vectors.to(device=device, dtype=torch.float32),
        prototype_labels.to(device=device, dtype=torch.long),
    )
    print(f"Prototype update: total_prototypes={int(prototype_labels.numel())}")

    optimizer = torch.optim.AdamW(
        [param for param in model.parameters() if param.requires_grad],
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
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
        for x, y in tqdm(train_loader, desc=f"FrozenProto epoch {epoch}/{args.epochs}"):
            x = x.to(device)
            y = y.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss, predictions = train_step_prediction(model, x, y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * y.size(0)
            total_correct += (predictions == y).sum().item()
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
        load_classifier_state(model, best_state)
    return model

def train_lora_contrastive(
    embedding_model,
    tokenizer,
    train_samples: Sequence[torch.nn.Module],
    train_labels: torch.Tensor,
    valid_samples: Sequence[torch.nn.Module],
    valid_labels: Optional[torch.Tensor],
    args: argparse.Namespace,
    dataset_cls: type = TextEmotionDataset,
    collate_fn: Optional[Callable] = None,
) -> nn.Module:
    device = embedding_model.device
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
        prototype_manager = None
        prototype_supcon_loss = None
    memory_bank = None
    if args.use_memory_bank_supcon:
        memory_bank = ClassBalancedMemoryBank(args.num_labels, args.memory_per_class)
    train_dataset = dataset_cls(train_samples, train_labels)
    class_counts = torch.bincount(train_labels, minlength=args.num_labels)
    print(f"LoRA SupCon class counts: {class_counts.tolist()}")
    if args.lora_batch_size < 2 * int((class_counts > 0).sum().item()):
        print(
            "Warning: --lora_batch_size cannot include all classes with positive pairs in every batch. "
            "The balanced pair sampler still forces same-label positives and rotates classes across batches."
        )
    effective_collate_fn = collate_fn or make_text_collator(tokenizer, args)
    train_loader = DataLoader(
        train_dataset,
        batch_sampler=BalancedPairBatchSampler(train_labels, args.lora_batch_size),
        collate_fn=effective_collate_fn,
    )
    prototype_refresh_loader = None
    if args.prototype_learning:
        prototype_refresh_loader = DataLoader(
            train_dataset,
            batch_size=args.embed_batch_size,
            shuffle=False,
            collate_fn=effective_collate_fn,
        )

    valid_loader = None
    if valid_samples and valid_labels is not None:
        valid_loader = DataLoader(
            dataset_cls(valid_samples, valid_labels),
            batch_size=args.embed_batch_size,
            shuffle=False,
            collate_fn=effective_collate_fn,
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
            ce_loss, predictions = train_step_prediction(classifier, embeddings, y)
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
    model: Optional[ProjectionSupConClassifier] = None,
) -> ProjectionSupConClassifier:
    device = torch.device("cuda" if torch.cuda.is_available() and not args.force_cpu_mlp else "cpu")
    if model is None:
        model = ProjectionSupConClassifier(args.embedding_dim, args.num_labels)
    model = model.to(device)
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
