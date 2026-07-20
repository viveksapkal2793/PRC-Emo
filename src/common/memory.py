from collections import deque
from typing import Dict

import torch
import torch.nn.functional as F


class ClassBalancedMemoryBank:
    def __init__(self, num_labels: int, memory_per_class: int):
        if memory_per_class < 1:
            raise ValueError("--memory_per_class must be at least 1.")
        self.num_labels = num_labels
        self.memory_per_class = memory_per_class
        self.embedding_queues: Dict[int, deque[torch.Tensor]] = {
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
