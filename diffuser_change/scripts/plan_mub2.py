from os.path import join
import numpy as np

import diffuser.datasets as datasets
import diffuser.utils as utils

from maze2d_prior.astar_utils import (
    extract_occupancy,
    nearest_free_cell,
    astar_multi_paths,
    filter_diverse_paths,
    build_sequence,
)
from maze2d_prior.field_utils import (
    build_gated_line_centered_prior_and_potential,
    query_guidance_field,
)
from maze2d_prior.viz_utils import (
    render_walls_and_paths,
    render_highres_line_centered_prior,
    render_potential_field,
)


class Parser(utils.Parser):
    dataset: str = 'maze2d-large-dense-v1'
    config: str = 'config.maze2d'

    astar_allow_diagonal: bool = False
    astar_steps_per_cell: int = 8
    astar_num_paths: int = 5
    astar_max_ratio: float = 2.0
    astar_max_overlap: float = 0.8
    astar_candidate_multiplier: int = 4

    prior_beta: float = 1.1
    prior_sigma_min: float = 0.8
    prior_sigma_max: float = 2.8
    prior_scale: int = 12
    prior_line_sigma: float = 0.36
    prior_tau: float = 0.35
    prior_eps: float = 1e-6

    def read_config(self, args, experiment):
        args = super().read_config(args, experiment)
        if experiment == 'plan':
            args.prefix = 'plans/astar_multi'
            self._dict['prefix'] = args.prefix
            if not hasattr(args, 'renderer'):
                args.renderer = 'utils.Maze2dRenderer'
                self._dict['renderer'] = args.renderer
        return args

def main():
    args = Parser().parse_args('plan')

    env = datasets.load_environment(args.dataset)
    render_config = utils.Config(args.renderer, env=args.dataset)
    _ = render_config()

    observation = env.reset()

    if args.conditional:
        env.set_target()

    target = np.asarray(env._target, dtype=np.float32)
    occupancy = extract_occupancy(env)

    start_cell = nearest_free_cell(observation[:2], occupancy)
    goal_cell = nearest_free_cell(target, occupancy)

    if start_cell == goal_cell:
        multi_paths = [([start_cell], 0.0)]
        shortest_cost = 0.0
        candidate_paths = multi_paths
    else:
        num_candidates = max(
            args.astar_num_paths * args.astar_candidate_multiplier,
            args.astar_num_paths,
        )

        candidate_paths, shortest_cost = astar_multi_paths(
            occupancy=occupancy,
            start=start_cell,
            goal=goal_cell,
            allow_diagonal=args.astar_allow_diagonal,
            num_paths=num_candidates,
            max_ratio=args.astar_max_ratio,
        )

        if len(candidate_paths) == 0:
            raise RuntimeError(f'A* failed to find a path from {start_cell} to {goal_cell}')

        multi_paths = filter_diverse_paths(
            candidate_paths,
            max_overlap=args.astar_max_overlap,
            max_paths=args.astar_num_paths,
        )

        if len(multi_paths) == 0:
            raise RuntimeError('A* found candidate paths, but all were removed by diversity filtering.')

    field_dict = build_gated_line_centered_prior_and_potential(
        occupancy=occupancy,
        multi_paths=multi_paths,
        scale=args.prior_scale,
        line_sigma=args.prior_line_sigma,
        beta=args.prior_beta,
        sigma_min=args.prior_sigma_min,
        sigma_max=args.prior_sigma_max,
        tau=args.prior_tau,
        eps=args.prior_eps,
    )

    # 这里把后面要用的变量从 field_dict 里取出来
    fused_prior_hr = field_dict['fused_prior_hr']
    potential_hr = field_dict['potential_hr']
    grad_row_hr = field_dict['grad_row_hr']
    grad_col_hr = field_dict['grad_col_hr']

    query_info = query_guidance_field(
        field_dict,
        x=float(observation[1]),
        y=float(observation[0]),
    )
    print(query_info)

    observation_dim = int(np.prod(env.observation_space.shape))
    all_sequences = []
    for cell_path, _ in multi_paths:
        seq = build_sequence(
            observation,
            target,
            cell_path,
            observation_dim,
            steps_per_cell=args.astar_steps_per_cell,
            horizon=args.horizon,
        )
        all_sequences.append(seq)

    save_path_paths = join(args.savepath, 'astar_multi_plan_manual.png')
    render_walls_and_paths(
        occupancy=occupancy,
        multi_paths=multi_paths,
        observation=observation,
        target=target,
        save_path=save_path_paths,
        draw_cells=True,
        draw_polyline=True,
    )

    save_path_prior = join(args.savepath, 'astar_line_centered_gated_prior.png')
    render_highres_line_centered_prior(
        occupancy=occupancy,
        fused_prior_hr=fused_prior_hr,
        scale=args.prior_scale,
        multi_paths=multi_paths,
        observation=observation,
        target=target,
        save_path=save_path_prior,
    )

    save_path_potential = join(args.savepath, 'astar_guidance_potential_field.png')
    render_potential_field(
        occupancy=occupancy,
        potential_hr=potential_hr,
        grad_row_hr=grad_row_hr,
        grad_col_hr=grad_col_hr,
        scale=args.prior_scale,
        multi_paths=multi_paths,
        observation=observation,
        target=target,
        save_path=save_path_potential,
        quiver_stride=10,
        vmax_percentile=95,
        use_log_display=False,
    )

if __name__ == '__main__':
    main()