from dataclasses import dataclass, field
from typing import Any, Dict, Optional

import numpy as np
import torch


@dataclass
class MapStructPrior:
    occupancy: np.ndarray
    start_xy: np.ndarray
    goal_xy: np.ndarray
    start_cell: tuple
    goal_cell: tuple
    multi_paths: list
    support_hr: np.ndarray
    support_soft_hr: np.ndarray
    distance_to_center_hr: np.ndarray
    distance_to_boundary_hr: np.ndarray
    centerline_potential_hr: np.ndarray
    grad_center_row_hr: np.ndarray
    grad_center_col_hr: np.ndarray
    scale: int
    support_threshold: float
    debug_info: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PGProposal:
    candidates_future_obs: torch.Tensor
    score_pg: torch.Tensor
    debug_info: Dict[str, Any] = field(default_factory=dict)


@dataclass
class StructuredPrior:
    support: MapStructPrior
    pg_density: PGProposal
    selected_future_obs: torch.Tensor
    score_support: torch.Tensor
    score_total: torch.Tensor
    selected_index: int
    selected_indices: list
    guidance: Dict[str, Any]
    debug_info: Dict[str, Any] = field(default_factory=dict)
    selected_future_obs_world: Optional[np.ndarray] = None
