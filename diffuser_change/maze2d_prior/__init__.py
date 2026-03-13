# maze2d_prior/__init__.py

from .astar_utils import (
    extract_occupancy,
    nearest_free_cell,
    astar_multi_paths,
    filter_diverse_paths,
    build_sequence,
)

from .field_utils import (
    build_gated_line_centered_prior_and_potential,
    query_guidance_field,
)

from .geometry_utils import (
    point_to_segment_distance,
    point_to_polyline_distance,
)

from .viz_utils import (
    render_walls_and_paths,
    render_highres_line_centered_prior,
    render_potential_field,
)

__all__ = [
    # astar
    "extract_occupancy",
    "nearest_free_cell",
    "astar_multi_paths",
    "filter_diverse_paths",
    "build_sequence",

    # field
    "build_gated_line_centered_prior_and_potential",
    "query_guidance_field",

    # geometry
    "point_to_segment_distance",
    "point_to_polyline_distance",

    # viz
    "render_walls_and_paths",
    "render_highres_line_centered_prior",
    "render_potential_field",
]