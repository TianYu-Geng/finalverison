from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from tqdm import tqdm

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
        self._runtime_traj = None
        self._flat = None
        self._branch_cache = {}

    def enabled_for_step(self, step_index: Optional[int], total_steps: Optional[int]) -> bool:
        if step_index is None or total_steps is None or total_steps <= 0:
            return True
        ratio = float(step_index) / float(total_steps)
        return self.active_start_ratio <= ratio <= self.active_end_ratio

    @torch.no_grad()
    def project(self, xt: torch.Tensor, step_index: Optional[int] = None, total_steps: Optional[int] = None, context=None):
        if not self.enabled_for_step(step_index, total_steps):
            return xt

        self.stats["calls"] = int(self.stats.get("calls", 0)) + 1
        self._ensure_runtime_view(xt)
        neighbors = self._query_neighbors(xt, context=context)
        self.local_manifold.compute_pca(neighbors, tau=self.pca_tau)
        projected = self.local_manifold.project_points(xt)
        if self.blend < 1.0:
            projected = xt + self.blend * (projected - xt)
        return projected

    def _ensure_runtime_view(self, xt: torch.Tensor):
        traj = self.datastore.trajectories
        target_horizon = int(xt.shape[1])
        target_dim = int(xt.shape[2])

        if self._runtime_traj is not None:
            if self._runtime_traj.shape[1] == target_horizon and self._runtime_traj.shape[2] == target_dim:
                return

        if traj.shape[1] < target_horizon:
            raise ValueError(
                f"LoMAP datastore horizon {traj.shape[1]} is smaller than planner horizon {target_horizon}"
            )
        if traj.shape[2] < target_dim:
            raise ValueError(
                f"LoMAP datastore dim {traj.shape[2]} is smaller than planner dim {target_dim}"
            )

        runtime_traj = traj[:, :target_horizon, :target_dim].contiguous()
        self._runtime_traj = runtime_traj
        self._flat = runtime_traj.reshape(runtime_traj.shape[0], -1)
        self.stats["matched_horizon"] = target_horizon
        self.stats["matched_dim"] = target_dim

    @torch.no_grad()
    def _bilinear_sample_grid_torch(self, arr, x, y, scale):
        hh, ww = arr.shape
        c = (x + 0.5) * scale - 0.5
        r = (y + 0.5) * scale - 0.5

        c0 = torch.floor(c).long().clamp(0, ww - 1)
        r0 = torch.floor(r).long().clamp(0, hh - 1)
        c1 = (c0 + 1).clamp(0, ww - 1)
        r1 = (r0 + 1).clamp(0, hh - 1)

        dc = c - c0.to(c.dtype)
        dr = r - r0.to(r.dtype)

        v00 = arr[r0, c0]
        v01 = arr[r0, c1]
        v10 = arr[r1, c0]
        v11 = arr[r1, c1]
        return (
            (1.0 - dr) * (1.0 - dc) * v00
            + (1.0 - dr) * dc * v01
            + dr * (1.0 - dc) * v10
            + dr * dc * v11
        )

    @torch.no_grad()
    def _get_branch_labels(self, context):
        if context is None:
            return None
        map_prior = context["map_prior"]
        show_progress = bool(context.get("show_progress", False))
        cache_key = (
            id(map_prior),
            int(self._runtime_traj.shape[1]),
            int(self._runtime_traj.shape[2]),
        )
        cached = self._branch_cache.get(cache_key)
        if cached is not None:
            return cached

        obs_mean_xy = torch.as_tensor(context["obs_mean_xy"], device=self.device, dtype=self._runtime_traj.dtype)
        obs_std_xy = torch.as_tensor(context["obs_std_xy"], device=self.device, dtype=self._runtime_traj.dtype).clamp_min(1e-6)

        scores = []
        chunk = 2048
        branch_iterator = map_prior.branch_fields
        if show_progress:
            branch_iterator = tqdm(
                map_prior.branch_fields,
                total=len(map_prior.branch_fields),
                desc="LoMAP branch cache",
                leave=False,
                dynamic_ncols=True,
            )

        for branch_field in branch_iterator:
            support = torch.as_tensor(branch_field.support_soft_hr, device=self.device, dtype=self._runtime_traj.dtype)
            branch_scores = []
            chunk_iter = range(0, self._runtime_traj.shape[0], chunk)
            if show_progress:
                chunk_iter = tqdm(
                    chunk_iter,
                    total=(self._runtime_traj.shape[0] + chunk - 1) // chunk,
                    desc=f"branch {branch_field.branch_id}",
                    leave=False,
                    dynamic_ncols=True,
                )
            for start in chunk_iter:
                end = min(start + chunk, self._runtime_traj.shape[0])
                traj = self._runtime_traj[start:end, :, :2]
                xy = traj * obs_std_xy.view(1, 1, 2) + obs_mean_xy.view(1, 1, 2)
                y = xy[:, :, 0]
                x = xy[:, :, 1]
                vals = self._bilinear_sample_grid_torch(support, x, y, map_prior.scale)
                branch_scores.append(torch.mean(vals, dim=1))
            scores.append(torch.cat(branch_scores, dim=0))

        branch_scores = torch.stack(scores, dim=1)
        branch_labels = torch.argmax(branch_scores, dim=1)
        self._branch_cache[cache_key] = branch_labels
        self.stats["branch_cache_size"] = len(self._branch_cache)
        return branch_labels

    @torch.no_grad()
    def _query_neighbors(self, xt: torch.Tensor, context=None) -> torch.Tensor:
        batch = xt.shape[0]
        xt_flat = xt.reshape(batch, -1)
        query_start = xt[:, 0, :2]
        query_end = xt[:, -1, :2]

        branch_labels = self._get_branch_labels(context)
        selected_branch_id = None if context is None else int(context["selected_branch_id"])

        per_batch_candidates = []
        for b in range(batch):
            if branch_labels is not None:
                valid_idx = torch.nonzero(branch_labels == selected_branch_id, as_tuple=False).squeeze(1)
                if valid_idx.numel() == 0:
                    valid_idx = torch.arange(self.datastore.size, device=self.device)
            else:
                valid_idx = torch.arange(self.datastore.size, device=self.device)

            branch_start = self.datastore.start_xy[valid_idx]
            branch_end = self.datastore.final_xy[valid_idx]
            sg_dist = torch.cdist(query_start[b:b + 1], branch_start) + torch.cdist(query_end[b:b + 1], branch_end)

            prefilter_k = min(int(self.prefilter_k), int(valid_idx.numel()))
            topk = min(int(self.topk), prefilter_k)
            candidate_local = torch.topk(sg_dist.squeeze(0), k=prefilter_k, largest=False, dim=0).indices
            candidate_idx = valid_idx[candidate_local]

            flat_candidates = self._flat[candidate_idx]
            full_dist = torch.linalg.norm(flat_candidates - xt_flat[b:b + 1], dim=-1)
            local_idx = torch.topk(full_dist, k=topk, largest=False, dim=0).indices
            per_batch_candidates.append(candidate_idx[local_idx])

        gather_idx = torch.stack(per_batch_candidates, dim=0)
        self.stats["selected_branch_id"] = selected_branch_id
        return self._runtime_traj[gather_idx]


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
