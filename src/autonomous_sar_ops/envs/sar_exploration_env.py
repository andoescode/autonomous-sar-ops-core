# Run module from src:
# uv run python -m autonomous_sar_ops.envs.sar_exploration_env

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, StrEnum, IntEnum, auto
from typing import Any, ClassVar

import gymnasium as gym
import numpy as np
from gymnasium import spaces


class Action(IntEnum):
    """Discrete movement actions available to each agent."""

    STAY = 0
    UP = 1
    DOWN = 2
    LEFT = 3
    RIGHT = 4

class AgentType(StrEnum):
    """
        An agent can be either ugv (ground robot) or uav (drone).
    
        ugv = ground robot
            - ignores no-fly zones
            - blocked by ground obstacles
            - uses Manhattan/grid-style movement cost
            - movement cost = 1.0 per step
    
        uav = aerial drone
            - ignores ground obstacles
            - blocked by no-fly zones
            - uses Euclidean-style travel cost for planning    
            - movement cost = 1.0 (if cardinal) | sqrt(2) (if diagonal)
    """
    UGV = auto()
    UAV = auto()

class AgentMode(StrEnum):
    """
    Decide what type(s) of agents will be included in the env.
    """
    UGV = auto() # all ugv
    UAV = auto() # all uav
    BOTH = auto() # mix ugv + uav

class SpawnMode(StrEnum):
    """
    Decide what spawning type(s) will be in the env.
    """
    SINGLE_BASE = auto() # one base spawn all
    RANDOM_DEPLOYED = auto() # multiple bases

class RewardName(StrEnum):
    """
    Reward given to the agent per step taken.
    """
    STEP = auto()
    NEW_CELL = auto()
    NEW_MAP_CELL = auto()
    VICTIM_FOUND = auto()
    ALL_VICTIMS_FOUND = auto()
    REVISIT = auto()
    COLLISION = auto()
    IDLE = auto()
    TIMEOUT = auto()

DEFAULT_REWARDS_: dict[RewardName, float] = {
    RewardName.STEP: -0.01, # Take action/step
    RewardName.NEW_CELL: 0.20, # Discover new local cell
    RewardName.NEW_MAP_CELL: 0.05, # Discover new global cell
    RewardName.VICTIM_FOUND: 10.00, # Succesfully found a victim
    RewardName.ALL_VICTIMS_FOUND: 25.00, # Succesfully found all victims
    RewardName.REVISIT: -0.05, # Revisit old cell
    RewardName.COLLISION: -0.50, # Collided with obstacles or other agent(s)
    RewardName.IDLE: -0.03, # Idle/ not progressing
    RewardName.TIMEOUT: -5.00, # Ran out of given time
}

@dataclass
class AgentState:
    """Minimal state required for exploration-only agents."""

    id: int
    position: tuple[int, int]
    agent_type: AgentType = AgentType.UGV 
    active: bool = True

class SARExplorationEnv(gym.Env):
    """
    Multi-agent search-and-rescue exploration environment.
    ---

    ### Descriptions

    The environment contains:
        - a hidden true obstacle map;
        - a partially observed map visible to the policy;
        - hidden victim locations;
        - one or more agents controlled through direct movement actions.

    The task is exploration only:
        - discover new map cells;
        - locate hidden victims;
        - avoid collisions and unnecessary revisits.

    This environment intentionally excludes:
        - MILP scheduling;
        - route assignment;
        - A* execution;
        - victim transport;
        - target priorities;
        - heterogeneous UAV/UGV movement;
        - battery constraints.

    ### Observation

    A float32 tensor with shape:

        (6, grid_height, grid_width)

    Channels:
        0: known free cells
        1: known obstacle cells
        2: unknown cells
        3: agent positions
        4: visited cells
        5: discovered victim cells

    ### Action

    For one agent:
        spaces.Discrete(5)

    For multiple agents:
        spaces.MultiDiscrete([5, ..., 5])

    Each agent selects one action from:
        0 = stay
        1 = up
        2 = down
        3 = left
        4 = right

    ### Episode termination

    terminated = True:
        all victims have been found.

    truncated = True:
        max_steps has been reached before all victims are found.
    """

    metadata = {
        "render_modes": ["ansi"],
        "render_fps": 4,
    }

    UNKNOWN: ClassVar[np.int8] = np.int8(-1)
    FREE: ClassVar[np.int8] = np.int8(0)
    OBSTACLE: ClassVar[np.int8] = np.int8(1)

    ACTION_TO_DELTA: ClassVar[dict[Action, tuple[int, int]]] = {
        Action.STAY: (0, 0),
        Action.UP: (-1, 0),
        Action.DOWN: (1, 0),
        Action.LEFT: (0, -1),
        Action.RIGHT: (0, 1),
    }

    DEFAULT_REWARDS: ClassVar[dict[str, float]] = {
        "step": DEFAULT_REWARDS_[RewardName.STEP],
        "new_cell": DEFAULT_REWARDS_[RewardName.NEW_CELL],
        "new_map_cell": DEFAULT_REWARDS_[RewardName.NEW_MAP_CELL],
        "victim_found": DEFAULT_REWARDS_[RewardName.VICTIM_FOUND],
        "all_victims_found": DEFAULT_REWARDS_[RewardName.ALL_VICTIMS_FOUND],
        "revisit": DEFAULT_REWARDS_[RewardName.REVISIT],
        "collision": DEFAULT_REWARDS_[RewardName.COLLISION],
        "idle": DEFAULT_REWARDS_[RewardName.IDLE],
        "timeout": DEFAULT_REWARDS_[RewardName.TIMEOUT],
    }

    def __init__(
        self,
        grid_size: tuple[int, int] = (8, 8),
        num_agents: int = 1,
        num_victims: int = 1,
        obstacle_ratio: float = 0.10,
        sensor_range: int = 1,
        max_steps: int = 100,
        base_position: tuple[int, int] = (0, 0),
        allow_agent_overlap: bool = True,
        reward_weights: dict[str, float] | None = None,
        max_reset_tries: int = 200,
        render_mode: str | None = None,
    ) -> None:
        super().__init__()

        self._validate_init_args(
            grid_size=grid_size,
            num_agents=num_agents,
            num_victims=num_victims,
            obstacle_ratio=obstacle_ratio,
            sensor_range=sensor_range,
            max_steps=max_steps,
            max_reset_tries=max_reset_tries,
        )

        self.grid_height, self.grid_width = grid_size
        self.num_agents = int(num_agents)
        self.num_victims = int(num_victims)
        self.obstacle_ratio = float(obstacle_ratio)
        self.sensor_range = int(sensor_range)
        self.max_steps = int(max_steps)
        self.base_position = tuple(base_position)
        self.allow_agent_overlap = bool(allow_agent_overlap)
        self.max_reset_tries = int(max_reset_tries)
        self.render_mode = render_mode

        if not self._in_bounds(self.base_position):
            raise ValueError("base_position must be inside the grid.")

        self.reward_weights = self.DEFAULT_REWARDS.copy()

        if reward_weights is not None:
            unknown_keys = set(reward_weights) - set(self.DEFAULT_REWARDS)

            if unknown_keys:
                raise ValueError(
                    "Unknown reward keys: "
                    f"{sorted(unknown_keys)}. "
                    f"Valid keys are: {sorted(self.DEFAULT_REWARDS)}"
                )

            self.reward_weights.update(
                {
                    key: float(value)
                    for key, value in reward_weights.items()
                }
            )

        if self.num_agents == 1:
            self.action_space = spaces.Discrete(len(Action))
        else:
            self.action_space = spaces.MultiDiscrete(
                np.full(
                    self.num_agents,
                    len(Action),
                    dtype=np.int64,
                )
            )

        self.observation_space = spaces.Box(
            low=0.0,
            high=1.0,
            shape=(
                6,
                self.grid_height,
                self.grid_width,
            ),
            dtype=np.float32,
        )

        self.true_obstacle_grid: np.ndarray
        self.known_obstacle_grid: np.ndarray
        self.discovered_victim_grid: np.ndarray
        self.visited: np.ndarray

        self.agents: list[AgentState]
        self.victim_positions: list[tuple[int, int]]
        self.victim_found: np.ndarray

        self.step_count: int = 0

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        super().reset(seed=seed)

        if options:
            self._apply_reset_options(options)

        self.step_count = 0

        for _ in range(self.max_reset_tries):
            self.true_obstacle_grid = self._generate_true_obstacle_grid()

            free_reachable_cells = sorted(
                self._get_reachable_free_cells_from_base()
            )

            candidate_victim_cells = [
                cell
                for cell in free_reachable_cells
                if cell != self.base_position
            ]

            if len(candidate_victim_cells) < self.num_victims:
                continue

            selected_indices = self.np_random.choice(
                len(candidate_victim_cells),
                size=self.num_victims,
                replace=False,
            )

            self.victim_positions = [
                candidate_victim_cells[int(index)]
                for index in selected_indices
            ]

            break
        else:
            raise RuntimeError(
                "Could not generate a valid reachable map with enough "
                "victim cells. Reduce obstacle_ratio or num_victims."
            )

        self.known_obstacle_grid = np.full(
            (self.grid_height, self.grid_width),
            self.UNKNOWN,
            dtype=np.int8,
        )

        self.discovered_victim_grid = np.zeros(
            (self.grid_height, self.grid_width),
            dtype=np.int8,
        )

        self.visited = np.zeros(
            (self.grid_height, self.grid_width),
            dtype=np.int32,
        )

        self.victim_found = np.zeros(
            self.num_victims,
            dtype=np.int8,
        )

        self.agents = [
            AgentState(
                id=agent_id,
                position=self.base_position,
            )
            for agent_id in range(self.num_agents)
        ]

        for agent in self.agents:
            row, col = agent.position
            self.visited[row, col] += 1

            self._reveal_around(agent.position)
            self._detect_victims(agent.position)

        obs = self._get_obs()
        info = self._get_info()

        return obs, info

    def step(
        self,
        action: int | np.ndarray | list[int] | tuple[int, int],
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        actions = self._normalise_actions(action)

        self.step_count += 1

        reward_parts = {
            key: 0.0
            for key in self.reward_weights
        }

        reward_parts["step"] += self.reward_weights["step"]

        newly_visited = 0
        newly_revealed = 0
        newly_found = 0
        collisions = 0
        revisits = 0
        idle_actions = 0

        events: list[dict[str, Any]] = []

        current_positions = [
            agent.position
            for agent in self.agents
        ]

        proposed_positions: list[tuple[int, int]] = []

        for agent, action_id in zip(
            self.agents,
            actions,
            strict=True,
        ):
            selected_action = Action(int(action_id))
            delta_row, delta_col = self.ACTION_TO_DELTA[selected_action]

            proposed_positions.append(
                (
                    agent.position[0] + delta_row,
                    agent.position[1] + delta_col,
                )
            )

        conflicting_destinations = self._get_conflicting_destinations(
            proposed_positions=proposed_positions,
            current_positions=current_positions,
        )

        for agent_index, (
            agent,
            action_id,
            proposed_position,
        ) in enumerate(
            zip(
                self.agents,
                actions,
                proposed_positions,
                strict=True,
            )
        ):
            selected_action = Action(int(action_id))
            previous_position = agent.position

            if selected_action == Action.STAY:
                idle_actions += 1
                reward_parts["idle"] += self.reward_weights["idle"]

            collision_reason: str | None = None

            if selected_action != Action.STAY:
                if not self._in_bounds(proposed_position):
                    collision_reason = "out_of_bounds"

                else:
                    row, col = proposed_position

                    if self.true_obstacle_grid[row, col] == self.OBSTACLE:
                        self.known_obstacle_grid[row, col] = self.OBSTACLE
                        collision_reason = "obstacle"

                    elif (
                        not self.allow_agent_overlap
                        and agent_index in conflicting_destinations
                    ):
                        collision_reason = "agent_conflict"

            if collision_reason is not None:
                collisions += 1
                reward_parts["collision"] += self.reward_weights["collision"]

                events.append(
                    {
                        "type": "collision",
                        "agent_id": agent.id,
                        "reason": collision_reason,
                        "attempted_position": proposed_position,
                    }
                )

                final_position = previous_position

            else:
                final_position = proposed_position
                agent.position = final_position

            row, col = final_position

            if self.visited[row, col] == 0:
                newly_visited += 1
                reward_parts["new_cell"] += self.reward_weights["new_cell"]
            else:
                revisits += 1
                reward_parts["revisit"] += self.reward_weights["revisit"]

            self.visited[row, col] += 1

            revealed_count = self._reveal_around(final_position)
            newly_revealed += revealed_count
            reward_parts["new_map_cell"] += (
                revealed_count
                * self.reward_weights["new_map_cell"]
            )

            found_victim_ids = self._detect_victims(final_position)
            found_count = len(found_victim_ids)
            newly_found += found_count

            reward_parts["victim_found"] += (
                found_count
                * self.reward_weights["victim_found"]
            )

            for victim_id in found_victim_ids:
                events.append(
                    {
                        "type": "victim_found",
                        "agent_id": agent.id,
                        "victim_id": victim_id,
                        "position": self.victim_positions[victim_id],
                    }
                )

        all_victims_found = bool(np.all(self.victim_found))
        terminated = all_victims_found
        truncated = (
            self.step_count >= self.max_steps
            and not terminated
        )

        if terminated:
            reward_parts["all_victims_found"] += self.reward_weights[
                "all_victims_found"
            ]

        if truncated:
            reward_parts["timeout"] += self.reward_weights["timeout"]

        reward = float(sum(reward_parts.values()))

        info = self._get_info()
        info.update(
            {
                "reward_parts": reward_parts,
                "events": events,
                "metrics": {
                    "newly_visited": newly_visited,
                    "newly_revealed": newly_revealed,
                    "newly_found": newly_found,
                    "collisions": collisions,
                    "revisits": revisits,
                    "idle_actions": idle_actions,
                    "coverage_percentage": self._get_coverage_percentage(),
                    "explored_percentage": self._get_explored_percentage(),
                },
            }
        )

        return (
            self._get_obs(),
            reward,
            terminated,
            truncated,
            info,
        )

    def render(self) -> str | None:
        if self.render_mode != "ansi":
            return None

        canvas = np.full(
            (self.grid_height, self.grid_width),
            " ? ",
            dtype=object,
        )

        for row in range(self.grid_height):
            for col in range(self.grid_width):
                cell_state = self.known_obstacle_grid[row, col]

                if cell_state == self.FREE:
                    canvas[row, col] = " . "
                elif cell_state == self.OBSTACLE:
                    canvas[row, col] = " # "

        base_row, base_col = self.base_position
        canvas[base_row, base_col] = " B "

        for victim_id, victim_position in enumerate(self.victim_positions):
            if not self.victim_found[victim_id]:
                continue

            row, col = victim_position
            canvas[row, col] = " V "

        for agent in self.agents:
            row, col = agent.position

            if self.num_agents == 1:
                canvas[row, col] = " A "
            else:
                canvas[row, col] = f"A{agent.id:1d}"

        output = "\n".join(
            " ".join(str(cell) for cell in row)
            for row in canvas
        )

        print(output)
        print(
            f"\nStep: {self.step_count}/{self.max_steps} | "
            f"Victims: {int(self.victim_found.sum())}/{self.num_victims} | "
            f"Coverage: {self._get_coverage_percentage():.1%} | "
            f"Explored: {self._get_explored_percentage():.1%}"
        )
        print()

        return output

    def close(self) -> None:
        """No external rendering resources are used."""
        return None

    def get_true_state(self) -> dict[str, Any]:
        """
        Return privileged simulator state for debugging and evaluation.

        Do not include this output in the policy observation.
        """

        return {
            "true_obstacle_grid": self.true_obstacle_grid.copy(),
            "victim_positions": list(self.victim_positions),
            "victim_found": self.victim_found.copy(),
            "agent_positions": [
                agent.position
                for agent in self.agents
            ],
        }

    def _normalise_actions(
        self,
        action: int | np.ndarray | list[int] | tuple[int, int],
    ) -> np.ndarray:
        if self.num_agents == 1:
            if isinstance(action, np.ndarray):
                if action.shape == ():
                    scalar_action = int(action.item())
                elif action.shape == (1,):
                    scalar_action = int(action[0])
                else:
                    raise ValueError(
                        "Single-agent environment expects a scalar action "
                        "or an array with shape (1,)."
                    )

            elif isinstance(action, (list, tuple)):
                if len(action) != 1:
                    raise ValueError(
                        "Single-agent environment expects exactly one action."
                    )
                scalar_action = int(action[0])

            else:
                scalar_action = int(action)

            actions = np.array([scalar_action], dtype=np.int64)

        else:
            actions = np.asarray(action, dtype=np.int64)

            if actions.shape != (self.num_agents,):
                raise ValueError(
                    f"Expected action shape ({self.num_agents},), "
                    f"received {actions.shape}."
                )

        if np.any(actions < 0) or np.any(actions >= len(Action)):
            raise ValueError(
                f"Actions must be integers in [0, {len(Action) - 1}]."
            )

        return actions

    def _get_conflicting_destinations(
        self,
        *,
        proposed_positions: list[tuple[int, int]],
        current_positions: list[tuple[int, int]],
    ) -> set[int]:
        """
        Identify moves that conflict when agent overlap is disabled.

        Conflicts include:
            - multiple agents moving into the same destination;
            - two agents swapping positions in the same step.
        """

        if self.allow_agent_overlap:
            return set()

        conflicting_indices: set[int] = set()

        destination_to_indices: dict[tuple[int, int], list[int]] = {}

        for index, destination in enumerate(proposed_positions):
            destination_to_indices.setdefault(destination, []).append(index)

        for indices in destination_to_indices.values():
            if len(indices) > 1:
                conflicting_indices.update(indices)

        for first_index in range(self.num_agents):
            for second_index in range(first_index + 1, self.num_agents):
                first_swaps_with_second = (
                    proposed_positions[first_index]
                    == current_positions[second_index]
                    and proposed_positions[second_index]
                    == current_positions[first_index]
                )

                if first_swaps_with_second:
                    conflicting_indices.add(first_index)
                    conflicting_indices.add(second_index)

        return conflicting_indices

    def _generate_true_obstacle_grid(self) -> np.ndarray:
        grid = np.zeros(
            (self.grid_height, self.grid_width),
            dtype=np.int8,
        )

        num_cells = self.grid_height * self.grid_width
        num_obstacles = int(num_cells * self.obstacle_ratio)

        available_cells = [
            (row, col)
            for row in range(self.grid_height)
            for col in range(self.grid_width)
            if (row, col) != self.base_position
        ]

        if num_obstacles > len(available_cells):
            raise RuntimeError(
                "Requested more obstacles than available non-base cells."
            )

        if num_obstacles > 0:
            selected_indices = self.np_random.choice(
                len(available_cells),
                size=num_obstacles,
                replace=False,
            )

            for index in selected_indices:
                row, col = available_cells[int(index)]
                grid[row, col] = self.OBSTACLE

        base_row, base_col = self.base_position
        grid[base_row, base_col] = self.FREE

        return grid

    def _get_reachable_free_cells_from_base(
        self,
    ) -> set[tuple[int, int]]:
        if (
            self.true_obstacle_grid[
                self.base_position[0],
                self.base_position[1],
            ]
            == self.OBSTACLE
        ):
            return set()

        frontier = [self.base_position]
        reachable = {self.base_position}

        while frontier:
            current = frontier.pop()

            for neighbour in self._get_cardinal_neighbours(current):
                if neighbour in reachable:
                    continue

                row, col = neighbour

                if self.true_obstacle_grid[row, col] == self.OBSTACLE:
                    continue

                reachable.add(neighbour)
                frontier.append(neighbour)

        return reachable

    def _get_cardinal_neighbours(
        self,
        position: tuple[int, int],
    ) -> list[tuple[int, int]]:
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
            if self._in_bounds(cell)
        ]

    def _reveal_around(
        self,
        position: tuple[int, int],
    ) -> int:
        newly_revealed = 0
        centre_row, centre_col = position

        for row_offset in range(
            -self.sensor_range,
            self.sensor_range + 1,
        ):
            for col_offset in range(
                -self.sensor_range,
                self.sensor_range + 1,
            ):
                cell = (
                    centre_row + row_offset,
                    centre_col + col_offset,
                )

                if not self._in_bounds(cell):
                    continue

                row, col = cell

                if self.known_obstacle_grid[row, col] == self.UNKNOWN:
                    newly_revealed += 1

                self.known_obstacle_grid[row, col] = (
                    self.OBSTACLE
                    if self.true_obstacle_grid[row, col] == self.OBSTACLE
                    else self.FREE
                )

        return newly_revealed

    def _detect_victims(
        self,
        position: tuple[int, int],
    ) -> list[int]:
        newly_found_ids: list[int] = []
        agent_row, agent_col = position

        for victim_id, victim_position in enumerate(self.victim_positions):
            if self.victim_found[victim_id]:
                continue

            victim_row, victim_col = victim_position

            chebyshev_distance = max(
                abs(agent_row - victim_row),
                abs(agent_col - victim_col),
            )

            if chebyshev_distance <= self.sensor_range:
                self.victim_found[victim_id] = 1
                self.discovered_victim_grid[
                    victim_row,
                    victim_col,
                ] = 1

                newly_found_ids.append(victim_id)

        return newly_found_ids

    def _get_obs(self) -> np.ndarray:
        known_free = (
            self.known_obstacle_grid == self.FREE
        ).astype(np.float32)

        known_obstacles = (
            self.known_obstacle_grid == self.OBSTACLE
        ).astype(np.float32)

        unknown = (
            self.known_obstacle_grid == self.UNKNOWN
        ).astype(np.float32)

        agent_grid = np.zeros(
            (self.grid_height, self.grid_width),
            dtype=np.float32,
        )

        for agent in self.agents:
            if not agent.active:
                continue

            row, col = agent.position
            agent_grid[row, col] = 1.0

        visited_grid = (
            self.visited > 0
        ).astype(np.float32)

        victim_grid = self.discovered_victim_grid.astype(np.float32)

        observation = np.stack(
            [
                known_free,
                known_obstacles,
                unknown,
                agent_grid,
                visited_grid,
                victim_grid,
            ],
            axis=0,
        ).astype(np.float32)

        return observation

    def _get_info(self) -> dict[str, Any]:
        return {
            "step_count": self.step_count,
            "num_agents": self.num_agents,
            "num_victims": self.num_victims,
            "victims_found": int(self.victim_found.sum()),
            "all_victims_found": bool(np.all(self.victim_found)),
            "coverage_percentage": self._get_coverage_percentage(),
            "explored_percentage": self._get_explored_percentage(),
            "agent_positions": [
                agent.position
                for agent in self.agents
            ],
            "discovered_victim_positions": [
                self.victim_positions[victim_id]
                for victim_id in range(self.num_victims)
                if self.victim_found[victim_id]
            ],
        }

    def _get_coverage_percentage(self) -> float:
        reachable_cells = self._get_reachable_free_cells_from_base()

        if not reachable_cells:
            return 0.0

        visited_reachable_cells = sum(
            1
            for row, col in reachable_cells
            if self.visited[row, col] > 0
        )

        return float(
            visited_reachable_cells
            / len(reachable_cells)
        )

    def _get_explored_percentage(self) -> float:
        known_cells = np.count_nonzero(
            self.known_obstacle_grid != self.UNKNOWN
        )

        total_cells = self.grid_height * self.grid_width

        if total_cells == 0:
            return 0.0

        return float(known_cells / total_cells)

    def _apply_reset_options(
        self,
        options: dict[str, Any],
    ) -> None:
        if "obstacle_ratio" in options:
            obstacle_ratio = float(options["obstacle_ratio"])

            if not 0.0 <= obstacle_ratio < 1.0:
                raise ValueError(
                    "obstacle_ratio must be in [0.0, 1.0)."
                )

            self.obstacle_ratio = obstacle_ratio

        if "sensor_range" in options:
            sensor_range = int(options["sensor_range"])

            if sensor_range < 0:
                raise ValueError(
                    "sensor_range must be greater than or equal to 0."
                )

            self.sensor_range = sensor_range

        if "max_steps" in options:
            max_steps = int(options["max_steps"])

            if max_steps <= 0:
                raise ValueError(
                    "max_steps must be greater than 0."
                )

            self.max_steps = max_steps

    def _in_bounds(
        self,
        position: tuple[int, int],
    ) -> bool:
        row, col = position

        return (
            0 <= row < self.grid_height
            and 0 <= col < self.grid_width
        )

    @staticmethod
    def _validate_init_args(
        *,
        grid_size: tuple[int, int],
        num_agents: int,
        num_victims: int,
        obstacle_ratio: float,
        sensor_range: int,
        max_steps: int,
        max_reset_tries: int,
    ) -> None:
        if (
            len(grid_size) != 2
            or grid_size[0] <= 1
            or grid_size[1] <= 1
        ):
            raise ValueError(
                "grid_size must be a two-dimensional size "
                "of at least (2, 2)."
            )

        if num_agents <= 0:
            raise ValueError(
                "num_agents must be greater than 0."
            )

        if num_victims <= 0:
            raise ValueError(
                "num_victims must be greater than 0."
            )

        if not 0.0 <= obstacle_ratio < 1.0:
            raise ValueError(
                "obstacle_ratio must be in [0.0, 1.0)."
            )

        if sensor_range < 0:
            raise ValueError(
                "sensor_range must be greater than or equal to 0."
            )

        if max_steps <= 0:
            raise ValueError(
                "max_steps must be greater than 0."
            )

        if max_reset_tries <= 0:
            raise ValueError(
                "max_reset_tries must be greater than 0."
            )


def run_random_policy_demo() -> None:
    """Run a short random-policy demonstration."""

    env = SARExplorationEnv(
        grid_size=(8, 8),
        num_agents=1,
        num_victims=2,
        obstacle_ratio=0.10,
        sensor_range=1,
        max_steps=100,
        render_mode="ansi",
    )

    observation, info = env.reset(seed=42)

    print("Initial observation shape:", observation.shape)
    print("Initial info:", info)
    env.render()

    terminated = False
    truncated = False
    total_reward = 0.0

    while not (terminated or truncated):
        action = env.action_space.sample()

        observation, reward, terminated, truncated, info = env.step(action)

        total_reward += reward

        print(
            f"Action={Action(int(action)).name:<5} | "
            f"Reward={reward:7.3f} | "
            f"Victims={info['victims_found']}/{info['num_victims']} | "
            f"Coverage={info['coverage_percentage']:.1%}"
        )

        env.render()

    print("Episode finished")
    print("Total reward:", total_reward)
    print("Final info:", info)

    env.close()


if __name__ == "__main__":
    run_random_policy_demo()
