from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import torch

from .datastore import load_lomap_datastore
from .local_manifold import LocalManifold


@dataclass
class LoMAPProjector:
    datastore: object
    device: torch.device
    topk: int = 8
    prefilter_k: int = 256
    pca_tau: float = 0.95
    blend: float = 1.0
    active_start_ratio: float = 0.5
    active_end_ratio: float = 1.0
    stats: dict = field(default_factory=lambda: {"calls": 0})

    def __post_init__(self):
        self.local_manifold = LocalManifold(device=self.device)
        self._flat = self.datastore.trajectories.reshape(self.datastore.size, -1)

    def enabled_for_step(self, step_index: Optional[int], total_steps: Optional[int]) -> bool:
        if step_index is None or total_steps is None or total_steps <= 0:
            return True
        ratio = float(step_index) / float(total_steps)
        return self.active_start_ratio <= ratio <= self.active_end_ratio

    @torch.no_grad()
    def project(self, xt: torch.Tensor, step_index: Optional[int] = None, total_steps: Optional[int] = None):
        if not self.enabled_for_step(step_index, total_steps):
            return xt

        self.stats["calls"] = int(self.stats.get("calls", 0)) + 1
        neighbors = self._query_neighbors(xt)
        self.local_manifold.compute_pca(neighbors, tau=self.pca_tau)
        projected = self.local_manifold.project_points(xt)
        if self.blend < 1.0:
            projected = xt + self.blend * (projected - xt)
        return projected

    @torch.no_grad()
    def _query_neighbors(self, xt: torch.Tensor) -> torch.Tensor:
        batch = xt.shape[0]
        xt_flat = xt.reshape(batch, -1)
        query_end = xt[:, -1, :2]

        prefilter_k = min(int(self.prefilter_k), self.datastore.size)
        topk = min(int(self.topk), prefilter_k)

        end_dist = torch.cdist(query_end, self.datastore.final_xy)
        candidate_idx = torch.topk(end_dist, k=prefilter_k, largest=False, dim=1).indices

        flat_candidates = self._flat[candidate_idx]
        full_dist = torch.linalg.norm(flat_candidates - xt_flat.unsqueeze(1), dim=-1)
        local_idx = torch.topk(full_dist, k=topk, largest=False, dim=1).indices

        gather_idx = candidate_idx.gather(1, local_idx)
        return self.datastore.trajectories[gather_idx]


def build_lomap_projector(config, device):
    store_path = getattr(config, "lomap_store_path", "")
    if not store_path:
        candidates = [
            Path("/home/linux/tianyu/final/lomap/results/lomapdiffuser_d4rl_maze2d") / config.env_name,
            Path("/home/linux/tianyu/final/lomap/results/lomapdiffuser") / config.env_name,
        ]
        for candidate in candidates:
            expanded = candidate.expanduser()
            if expanded.exists():
                store_path = str(expanded)
                break
        if not store_path:
            store_path = str(candidates[0])

    datastore = load_lomap_datastore(store_path, device=device)
    return LoMAPProjector(
        datastore=datastore,
        device=device,
        topk=int(getattr(config, "lomap_topk", 8)),
        prefilter_k=int(getattr(config, "lomap_prefilter_k", 256)),
        pca_tau=float(getattr(config, "lomap_pca_tau", 0.95)),
        blend=float(getattr(config, "lomap_blend", 1.0)),
        active_start_ratio=float(getattr(config, "lomap_active_start_ratio", 0.5)),
        active_end_ratio=float(getattr(config, "lomap_active_end_ratio", 1.0)),
    )
