from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np
import torch


@dataclass
class LoMAPDatastore:
    trajectories: torch.Tensor
    final_xy: torch.Tensor
    metadata: dict

    @property
    def size(self) -> int:
        return int(self.trajectories.shape[0])

    @property
    def horizon(self) -> int:
        return int(self.trajectories.shape[1])

    @property
    def dim(self) -> int:
        return int(self.trajectories.shape[2])


def load_lomap_datastore(path, device):
    store_path = Path(path).expanduser()
    if store_path.is_dir():
        h5_candidate = store_path / "dataset.h5"
        if h5_candidate.exists():
            store_path = h5_candidate

    if not store_path.exists():
        raise FileNotFoundError(f"LoMAP datastore not found: {store_path}")

    if store_path.suffix in {".h5", ".hdf5"}:
        with h5py.File(store_path, "r") as f:
            if "traj_dataset" not in f:
                raise KeyError(f"{store_path} does not contain 'traj_dataset'")
            traj_np = f["traj_dataset"][:].astype(np.float32)

            if "sg_dataset" in f:
                sg_np = f["sg_dataset"][:].astype(np.float32)
                final_xy_np = sg_np[:, -1, :2]
            else:
                final_xy_np = traj_np[:, -1, :2]

        metadata = {
            "source": str(store_path),
            "format": "h5",
            "num_trajectories": int(traj_np.shape[0]),
            "horizon": int(traj_np.shape[1]),
            "dim": int(traj_np.shape[2]),
        }
        trajectories = torch.from_numpy(traj_np).float().to(device)
        final_xy = torch.from_numpy(final_xy_np).float().to(device)
        return LoMAPDatastore(trajectories=trajectories, final_xy=final_xy, metadata=metadata)

    raise ValueError(f"Unsupported LoMAP datastore format: {store_path}")
