from __future__ import annotations

import numpy as np


GridPosition = tuple[int, int]

def get_available_cells(
    grid_height: int,
    grid_width: int,
    excluded_cells: set[GridPosition] | None = None,
) -> list[GridPosition]:
    """
    Get free cells from map.

    Returns list[positions]
    """
    excluded_cells = excluded_cells or set()

    return [
        (row, col)
        for row in range(grid_height)
        for col in range(grid_width)
        if (row, col) not in excluded_cells
    ]

def get_cardinal_neighbours(
    position: GridPosition,
    grid_height: int,
    grid_width: int,
) -> list[GridPosition]:
    """
    Get the neighbours within reach for agent.

    Returns list[cells_near_by=up,down,left,right]
    """
    row, col = position

    candidates = [
        (row - 1, col),
        (row + 1, col),
        (row, col - 1),
        (row, col + 1),
    ]

    return [
        cell
        for cell in candidates
        if (
            0 <= cell[0] < grid_height
            and 0 <= cell[1] < grid_width
        )
    ]


def get_reachable_cells(
    grid: np.ndarray,
    start: GridPosition,
    blocked_value: int = 1,
) -> set[GridPosition]:
    """
    Get cells that are reachable from start pos.

    Returns set{reachable_cells}
    """
    grid_height, grid_width = grid.shape
    start_row, start_col = start

    if grid[start_row, start_col] == blocked_value:
        return set()

    frontier = [start]
    reachable = {start}

    while frontier:
        current = frontier.pop()

        for neighbour in get_cardinal_neighbours(
            current,
            grid_height,
            grid_width,
        ):
            if neighbour in reachable:
                continue

            row, col = neighbour

            if grid[row, col] == blocked_value:
                continue

            reachable.add(neighbour)
            frontier.append(neighbour)

    return reachable

def get_chebyshev_distance(agent_position, victim_position):
    agent_row, agent_col = agent_position
    victim_row, victim_col = victim_position
    return max(
                abs(agent_row - victim_row),
                abs(agent_col - victim_col),
            )