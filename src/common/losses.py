from typing import Optional

import torch
import torch.nn.functional as F


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
    memory_bank,
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
