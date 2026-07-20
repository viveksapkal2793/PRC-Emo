import torch
import torch.nn as nn
import torch.nn.functional as F

from .prototype import PrototypeClassifier


def _alpha_to_logit(alpha: float) -> float:
    alpha = min(max(float(alpha), 1e-6), 1.0 - 1e-6)
    return float(torch.log(torch.tensor(alpha / (1.0 - alpha))).item())


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

    def forward_with_projection(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        projected = self.project(x)
        return projected, self.classify(projected)


class FusionClassifier(nn.Module):
    def __init__(
        self,
        input_dim: int,
        num_labels: int,
        dropout: float,
        prototype_temperature: float,
        fixed_alpha: float | None = None,
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

    def forward_components(self, embeddings: torch.Tensor) -> dict[str, torch.Tensor]:
        mlp_logits = self.mlp(embeddings)
        mlp_probs = F.softmax(mlp_logits, dim=-1)
        if not self.has_prototypes():
            proto_class_scores = embeddings.new_zeros(mlp_logits.shape)
            proto_probs = mlp_probs.new_zeros(mlp_probs.shape)
        else:
            proto_class_scores = self.prototype_classifier(embeddings)
            proto_probs = F.softmax(proto_class_scores, dim=-1)
            alpha = self.alpha().to(embeddings.device)
            mlp_probs = mlp_probs
            proto_probs = proto_probs
            final_probs = alpha * mlp_probs + (1.0 - alpha) * proto_probs
            final_log_probs = torch.log(final_probs.clamp_min(1e-12))
            return {
                "mlp_logits": mlp_logits,
                "proto_class_scores": proto_class_scores,
                "proto_probs": proto_probs,
                "final_probs": final_probs,
                "final_log_probs": final_log_probs,
            }
        final_log_probs = torch.log(proto_probs.clamp_min(1e-12))
        return {
            "mlp_logits": mlp_logits,
            "proto_class_scores": proto_class_scores,
            "proto_probs": proto_probs,
            "final_probs": proto_probs,
            "final_log_probs": final_log_probs,
        }

    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
        return self.forward_components(embeddings)["final_log_probs"]
