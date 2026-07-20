import argparse
from typing import Dict, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class PrototypeManager:
    def __init__(self, args: argparse.Namespace):
        self.args = args
        self._clusterer_cls = None

    def _reduce_for_prototype_clustering(self, embeddings: torch.Tensor) -> torch.Tensor:
        if embeddings.ndim != 2 or embeddings.size(0) < 2 or embeddings.size(1) < 2:
            return embeddings.detach().float().cpu()

        n_samples, n_features = embeddings.shape
        n_components = min(50, n_features, n_samples - 1)
        if n_components < 1 or n_components >= n_features:
            return embeddings.detach().float().cpu()

        try:
            from sklearn.decomposition import IncrementalPCA, PCA
        except Exception as exc:
            raise ImportError(
                "Prototype learning requires scikit-learn PCA support for prototype clustering."
            ) from exc

        embeddings_np = embeddings.detach().float().cpu().numpy()
        if n_samples > 512:
            batch_size = min(n_samples, max(n_components, 256))
            reducer = IncrementalPCA(n_components=n_components, batch_size=batch_size)
            for start in range(0, n_samples, batch_size):
                reducer.partial_fit(embeddings_np[start : start + batch_size])
            reduced_chunks = []
            for start in range(0, n_samples, batch_size):
                reduced_chunks.append(reducer.transform(embeddings_np[start : start + batch_size]))
            reduced_np = np.concatenate(reduced_chunks, axis=0)
        else:
            reduced_np = PCA(n_components=n_components).fit_transform(embeddings_np)
        return torch.from_numpy(reduced_np).float()

    def _get_clusterer_cls(self):
        if self._clusterer_cls is not None:
            return self._clusterer_cls
        try:
            import hdbscan  # type: ignore

            self._clusterer_cls = hdbscan.HDBSCAN
            return self._clusterer_cls
        except Exception as first_exc:
            try:
                from sklearn.cluster import HDBSCAN as SklearnHDBSCAN  # type: ignore

                self._clusterer_cls = SklearnHDBSCAN
                return self._clusterer_cls
            except Exception as second_exc:
                raise ImportError(
                    "Prototype learning requires an HDBSCAN implementation. "
                    "Install the `hdbscan` package or a scikit-learn version that provides sklearn.cluster.HDBSCAN."
                ) from second_exc if second_exc is not None else first_exc

    def _distance_to_centroids(self, points: torch.Tensor, centroids: torch.Tensor) -> torch.Tensor:
        metric = str(self.args.prototype_hdbscan_metric).lower()
        if metric == "cosine":
            point_norm = F.normalize(points, p=2, dim=1)
            centroid_norm = F.normalize(centroids, p=2, dim=1)
            return 1.0 - torch.matmul(point_norm, centroid_norm.T)
        return torch.cdist(points, centroids, p=2)

    def _fallback_class_prototype(self, class_embeddings: torch.Tensor, class_id: int) -> Tuple[torch.Tensor, torch.Tensor]:
        centroid = F.normalize(class_embeddings.mean(dim=0, keepdim=True), p=2, dim=1)
        print(f"Prototype fallback: class {class_id} uses a single centroid prototype.")
        return centroid.cpu(), torch.tensor([class_id], dtype=torch.long)

    def recompute(
        self,
        embeddings: torch.Tensor,
        labels: torch.Tensor,
        num_labels: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        clusterer_cls = self._get_clusterer_cls()
        embeddings = F.normalize(embeddings.detach().float().cpu(), p=2, dim=1)
        labels = labels.detach().cpu().long()
        reduced_embeddings = self._reduce_for_prototype_clustering(embeddings)
        prototype_vectors = []
        prototype_labels = []

        for class_id in range(num_labels):
            class_mask = labels == class_id
            class_size = int(class_mask.sum().item())
            if class_size == 0:
                continue
            class_embeddings = embeddings[class_mask]
            min_cluster_size = max(int(self.args.prototype_min_cluster_size), int(0.03 * class_size), 1)
            if class_size < max(min_cluster_size, 2):
                fallback_vectors, fallback_labels = self._fallback_class_prototype(class_embeddings, class_id)
                prototype_vectors.append(fallback_vectors)
                prototype_labels.append(fallback_labels)
                continue

            clusterer = clusterer_cls(
                min_cluster_size=min_cluster_size,
                metric=self.args.prototype_hdbscan_metric,
                cluster_selection_method="leaf",
            )
            class_reduced_embeddings = reduced_embeddings[class_mask]
            cluster_assignments = np.asarray(clusterer.fit_predict(class_reduced_embeddings.numpy()), dtype=np.int64)
            valid_clusters = sorted(int(cluster_id) for cluster_id in np.unique(cluster_assignments) if cluster_id != -1)
            if not valid_clusters:
                print(f"Prototype warning: HDBSCAN found no valid clusters for class {class_id}. Falling back to one centroid.")
                fallback_vectors, fallback_labels = self._fallback_class_prototype(class_embeddings, class_id)
                prototype_vectors.append(fallback_vectors)
                prototype_labels.append(fallback_labels)
                continue

            centroid_ids = []
            centroid_vectors = []
            for cluster_id in valid_clusters:
                cluster_points = class_embeddings[cluster_assignments == cluster_id]
                centroid_ids.append(cluster_id)
                centroid_vectors.append(cluster_points.mean(dim=0))
            centroid_tensor = torch.stack(centroid_vectors, dim=0)

            outlier_mask = cluster_assignments == -1
            if outlier_mask.any():
                outliers = class_embeddings[outlier_mask]
                distances = self._distance_to_centroids(outliers, centroid_tensor)
                nearest_centroids = distances.argmin(dim=1).tolist()
                for outlier_idx, centroid_pos in zip(np.where(outlier_mask)[0].tolist(), nearest_centroids):
                    cluster_assignments[outlier_idx] = centroid_ids[int(centroid_pos)]

            final_cluster_ids = sorted(int(cluster_id) for cluster_id in np.unique(cluster_assignments))
            class_prototypes = []
            for cluster_id in final_cluster_ids:
                cluster_points = class_embeddings[cluster_assignments == cluster_id]
                centroid = F.normalize(cluster_points.mean(dim=0, keepdim=True), p=2, dim=1).squeeze(0)
                class_prototypes.append(centroid)

            prototype_vectors.append(torch.stack(class_prototypes, dim=0).cpu())
            prototype_labels.append(torch.full((len(class_prototypes),), class_id, dtype=torch.long))
            print(
                f"Prototype update: class {class_id} size={class_size} min_cluster_size={min_cluster_size} "
                f"clusters={len(class_prototypes)}"
            )

        if not prototype_vectors:
            return torch.empty((0, embeddings.size(1)), dtype=torch.float32), torch.empty((0,), dtype=torch.long)
        return torch.cat(prototype_vectors, dim=0), torch.cat(prototype_labels, dim=0)


class SamplePrototypeSupConLoss(nn.Module):
    def __init__(self, temperature: float):
        super().__init__()
        self.temperature = temperature

    def forward(
        self,
        embeddings: torch.Tensor,
        labels: torch.Tensor,
        prototype_vectors: torch.Tensor,
        prototype_labels: torch.Tensor,
    ) -> torch.Tensor:
        if embeddings.size(0) == 0 or prototype_vectors.numel() == 0:
            return embeddings.new_tensor(0.0)

        norm_embeddings = F.normalize(embeddings, p=2, dim=1)
        norm_prototypes = F.normalize(prototype_vectors, p=2, dim=1)
        similarity = torch.matmul(norm_embeddings, norm_prototypes.T) / self.temperature
        losses = []
        for idx in range(norm_embeddings.size(0)):
            same_class_mask = prototype_labels == labels[idx]
            negative_mask = prototype_labels != labels[idx]
            if not same_class_mask.any() or not negative_mask.any():
                continue
            positive_logit = similarity[idx, same_class_mask].max()
            logits = torch.cat([positive_logit.unsqueeze(0), similarity[idx, negative_mask]], dim=0)
            losses.append(-(positive_logit - torch.logsumexp(logits, dim=0)))
        if not losses:
            return embeddings.new_tensor(0.0)
        return torch.stack(losses).mean()


class PrototypeClassifier(nn.Module):
    def __init__(self, num_labels: int, temperature: float):
        super().__init__()
        self.num_labels = num_labels
        self.temperature = temperature
        self.register_buffer("prototype_vectors", torch.empty(0, 0))
        self.register_buffer("prototype_labels", torch.empty(0, dtype=torch.long))

    def set_prototypes(self, prototype_vectors: torch.Tensor, prototype_labels: torch.Tensor) -> None:
        if prototype_vectors.numel() == 0:
            feature_dim = self.prototype_vectors.size(1) if self.prototype_vectors.ndim == 2 else 0
            self.prototype_vectors = torch.empty((0, feature_dim), device=self.prototype_vectors.device)
            self.prototype_labels = torch.empty((0,), dtype=torch.long, device=self.prototype_labels.device)
            return
        normalized = F.normalize(prototype_vectors.detach().float(), p=2, dim=1)
        self.prototype_vectors = normalized.to(self.prototype_vectors.device)
        self.prototype_labels = prototype_labels.detach().long().to(self.prototype_labels.device)

    def has_prototypes(self) -> bool:
        return self.prototype_vectors.numel() > 0 and self.prototype_labels.numel() > 0

    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
        if not self.has_prototypes():
            return embeddings.new_full((embeddings.size(0), self.num_labels), -1e4)
        norm_embeddings = F.normalize(embeddings, p=2, dim=1)
        norm_prototypes = F.normalize(self.prototype_vectors, p=2, dim=1)
        similarity = torch.matmul(norm_embeddings, norm_prototypes.T) / self.temperature
        class_scores = embeddings.new_full((embeddings.size(0), self.num_labels), -1e4)
        for class_id in range(self.num_labels):
            class_mask = self.prototype_labels == class_id
            if class_mask.any():
                class_scores[:, class_id] = torch.logsumexp(similarity[:, class_mask], dim=1)
        return class_scores
