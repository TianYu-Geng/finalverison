import torch


class LocalManifold:
    def __init__(self, device: torch.device):
        self.device = device
        self.pc = None
        self.mean_vec = None

    @torch.no_grad()
    def compute_pca(self, manifold_samples: torch.Tensor, tau: float = 0.95):
        batch_size, k = manifold_samples.shape[:2]
        manifold_samples_flat = manifold_samples.reshape(batch_size, k, -1)

        mean_vec = manifold_samples_flat.mean(dim=1, keepdim=True)
        x_centered = manifold_samples_flat - mean_vec

        gram = torch.bmm(x_centered, x_centered.transpose(2, 1))
        eigvals, eigvecs = torch.linalg.eigh(gram)

        order = torch.argsort(eigvals, dim=1, descending=True)
        eigvals = torch.gather(eigvals, 1, order)
        eigvecs = eigvecs[torch.arange(batch_size, device=manifold_samples.device).unsqueeze(-1), :, order]

        total_var = eigvals.sum(dim=1, keepdim=True).clamp_min(1e-8)
        cum_ratio = torch.cumsum(eigvals / total_var, dim=1)
        keep = torch.sum(cum_ratio < tau, dim=1) + 1
        max_keep = int(keep.max().item())

        eigvals = eigvals[:, :max_keep].clamp_min(1e-8)
        eigvecs = eigvecs[:, :max_keep, :].transpose(2, 1)

        pc = torch.bmm(x_centered.transpose(1, 2), eigvecs)
        pc = pc * (1.0 / eigvals.sqrt()).unsqueeze(1)

        self.mean_vec = mean_vec
        self.pc = pc

    @torch.no_grad()
    def project_points(self, points: torch.Tensor):
        batch_size, dims = points.shape[0], points.shape[1:]
        points_flat = points.reshape(batch_size, -1)
        centered = points_flat - self.mean_vec.squeeze(1)
        coeff = torch.bmm(centered.unsqueeze(1), self.pc)
        projected = self.mean_vec.squeeze(1) + torch.bmm(coeff, self.pc.transpose(1, 2)).squeeze(1)
        return projected.reshape(batch_size, *dims)
