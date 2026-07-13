from __future__ import annotations

import numpy as np

from autonomous_sar_ops.envs.grid_utils import (
    GridPosition,
    get_available_cells,
)

def generate_obstacle_grid(
    *,
    grid_height: int,
    grid_width: int,
    obstacle_ratio: float,
    base_position: GridPosition,
    rng: np.random.Generator,
    free_value: int = 0,
    obstacle_value: int = 1,
) -> np.ndarray:
    """
    Generate obstacle cells.

    Returns np.array grid map.
    """
    grid = np.full(
        (grid_height, grid_width),
        free_value,
        dtype=np.int8,
    )

    num_cells = grid_height * grid_width
    num_obstacles = int(num_cells * obstacle_ratio)

    available_cells = get_available_cells(
        grid_height=grid_height,
        grid_width=grid_width,
        excluded_cells={base_position},
    )

    if num_obstacles > len(available_cells):
        raise ValueError(
            "Requested more obstacles than available non-base cells."
        )

    if num_obstacles > 0:
        selected_indices = rng.choice(
            len(available_cells),
            size=num_obstacles,
            replace=False,
        )

        for index in selected_indices:
            row, col = available_cells[int(index)]
            grid[row, col] = obstacle_value

    base_row, base_col = base_position
    grid[base_row, base_col] = free_value

    return grid