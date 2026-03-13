import torch

from .types import PGProposal


def _cfg(config, name, default):
    return getattr(config, name, default)


def generate_pg_proposals(prior_model, observation_t, config, eval_deterministic=True):
    num_candidates = int(_cfg(config, "prior_struct_num_candidates", _cfg(config, "num_vis_samples", 16)))
    obs_repeat = observation_t.unsqueeze(0).repeat(num_candidates, 1)

    dist = prior_model.eval_forward(obs_repeat)
    mean = dist.mean

    if eval_deterministic:
        samples = dist.sample() if num_candidates > 1 else mean
        candidates = samples
        candidates[0] = mean[0]
    else:
        candidates = dist.sample()

    raw_candidates = candidates.clone()
    score_pg = dist.log_prob(raw_candidates)
    if score_pg.ndim > 1:
        score_pg = score_pg.mean(dim=tuple(range(1, score_pg.ndim)))

    if _cfg(config, "use_tanh_squash", True):
        candidates = torch.tanh(candidates) * float(_cfg(config, "prior_squash_mean", 2.0))

    return PGProposal(
        candidates_future_obs=candidates,
        score_pg=score_pg,
        debug_info={"mean_future_obs": mean.detach().clone()},
    )
