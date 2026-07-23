# Run module from src:
# cd <project_dir>/src
# uv run python -m autonomous_sar_ops.envs.sar_exploration_env

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, StrEnum, IntEnum, auto
from typing import Any, ClassVar

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from autonomous_sar_ops.envs.grid_utils import get_reachable_cells, get_chebyshev_distance
from autonomous_sar_ops.envs.map_generation import generate_obstacle_grid


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
        - a partially observed map visible to the policy
        - hidden map structure
        - hidden victim locations
        - one/more agents controlled through direct movement actions

    The task is exploration only:
        - discover new map cells
        - locate hidden victims
        - avoid collisions and unnecessary revisits

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

    # Actions dict
    ACTION_TO_DELTA: ClassVar[dict[Action, tuple[int, int]]] = {
        Action.STAY: (0, 0), # Idle (stand still to let other pass)
        Action.UP: (-1, 0),
        Action.DOWN: (1, 0),
        Action.LEFT: (0, -1),
        Action.RIGHT: (0, 1),
    }

    # Rewards dict
    DEFAULT_REWARDS: ClassVar[dict[str, float]] = {
        reward_name.value: reward_value
        for reward_name, reward_value in DEFAULT_REWARDS_.items()
    }

    def __init__(
        self,
        grid_size: tuple[int, int] = (8, 8), # env size
        num_agents: int = 1, # total number of agents in sim
        num_victims: int = 1, # total number of victims in sim
        obstacle_ratio: float = 0.10, # ratios of obstacles in map
        sensor_range: int = 1, # range of scanning sensor of each agent
        max_steps: int = 100, # max steps agent can take in this run
        base_position: tuple[int, int] = (0, 0), # spawn position of the agent(s)
        allow_agent_overlap: bool = True,  
        reward_weights: dict[str, float] | None = None, # reward weights based on given reward set
        max_reset_tries: int = 200, # max times the map can reset to reconstruct the map
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

        # Set up action space for single agent mode and multi-agent mode
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

        # Map details from agent(s) view
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

        # Capture map structure details
        self.obstacle_grid: np.ndarray
        self.known_obstacle_grid: np.ndarray
        self.discovered_victim_grid: np.ndarray
        self.visited: np.ndarray

        self.agents: list[AgentState]
        self.victim_positions: list[tuple[int, int]]
        self.victim_found: np.ndarray

        # Keep track of step count
        self.step_count: int = 0

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        """
        Set up env based on given options/seed.
        """
        super().reset(seed=seed)

        if options:
            self._apply_reset_options(options)

        self.step_count = 0

        self._generate_map()
        
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

    def _generate_map(self):
        """
        Generate map for env.
        """
        for _ in range(self.max_reset_tries):
            # Choose cells to have obstacles (not base)
            self.obstacle_grid = generate_obstacle_grid(
                grid_height=self.grid_height,
                grid_width=self.grid_width,
                obstacle_ratio=self.obstacle_ratio,
                base_position=self.base_position,
                rng=self.np_random,
                free_value=int(self.FREE),
                obstacle_value=int(self.OBSTACLE),
            )

            free_reachable_cells = sorted(
                get_reachable_cells(
                    grid=self.obstacle_grid,
                    start=self.base_position,
                    blocked_value=int(self.OBSTACLE),
                )
            )

            # Candidate cells that could be victims
            # There should always be a feasible way for the agent to reach the victim(s) - default
            candidate_victim_cells = [
                cell
                for cell in free_reachable_cells
                if cell != self.base_position
            ]

            # Skip if there is not enough feasible candidate cells
            if len(candidate_victim_cells) < self.num_victims:
                continue

            # Randomly choose cells from candidates cells that could be victims
            selected_indices = self.np_random.choice(
                len(candidate_victim_cells),
                size=self.num_victims,
                replace=False,
            )

            # Record the positions of the victims (in env)
            self.victim_positions = [
                candidate_victim_cells[int(index)]
                for index in selected_indices
            ]

            return
        else:
            raise RuntimeError(
                "Could not generate a valid reachable map with enough "
                "victim cells. Reduce obstacle_ratio or num_victims."
            ) 
        
    def _add_reward(
        self,
        *,
        reward_parts: dict[str, float],
        reward_reasons: list[dict[str, Any]],
        reward_name: RewardName,
        agent_id: int | None,
        reason: str,
        multiplier: float = 1.0,
        metadata: dict[str, Any] | None = None,
    ) -> float:
        """
        Add one reward contribution and record reasonings (why it was applied).
        """
        base_weight = float(self.reward_weights[reward_name.value])
        contribution = base_weight * float(multiplier)

        reward_parts[reward_name.value] += contribution
        reward_reasons.append(
            {
                "reward_name": reward_name.value,
                "value": float(contribution),
                "base_weight": base_weight,
                "multiplier": float(multiplier),
                "agent_id": agent_id,
                "reason": reason,
                "metadata": metadata or {},
            }
        )
        return float(contribution)

    def step(
        self,
        action: int | np.ndarray | list[int] | tuple[int, ...],
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        """
        Agent takes action.
        """

        # Take validated action(s)
        actions = self._normalise_actions(action)
        self.step_count += 1

        reward_parts = {key: 0.0 for key in self.reward_weights}
        reward_reasons: list[dict[str, Any]] = []

        # Add reward from step taken
        self._add_reward(
            reward_parts=reward_parts,
            reward_reasons=reward_reasons,
            reward_name=RewardName.STEP,
            agent_id=None,
            reason="Environment advanced by one time step.",
            metadata={"step_count": self.step_count},
        )

        newly_visited = 0
        newly_revealed = 0
        newly_found = 0
        collisions = 0
        revisits = 0
        failed_map_movements = 0
        deliberate_stays = 0
        agent_conflicts = 0
        events: list[dict[str, Any]] = []

        current_positions = [agent.position for agent in self.agents]
        proposed_positions: list[tuple[int, int]] = []

        for agent, action_id in zip(self.agents, actions, strict=True):
            selected_action = Action(int(action_id))
            delta_row, delta_col = self.ACTION_TO_DELTA[selected_action]
            proposed_positions.append(
                (agent.position[0] + delta_row, agent.position[1] + delta_col)
            )

        conflicting_destinations = self._get_conflicting_destinations(
            proposed_positions=proposed_positions,
            current_positions=current_positions,
        )

        for agent_index, (agent, action_id, proposed_position) in enumerate(
            zip(self.agents, actions, proposed_positions, strict=True)
        ):
            selected_action = Action(int(action_id))
            previous_position = agent.position
            movement_attempted = selected_action != Action.STAY
            collision_reason: str | None = None

            if selected_action == Action.STAY:
                # Deliberate waiting/yielding is not an IDLE failure.
                deliberate_stays += 1
                final_position = previous_position
                events.append(
                    {
                        "type": "deliberate_stay",
                        "agent_id": agent.id,
                        "position": previous_position,
                        "action": selected_action.name,
                    }
                )
            else:
                if not self._in_bounds(proposed_position):
                    collision_reason = "out_of_bounds"
                else:
                    row, col = proposed_position
                    if self.obstacle_grid[row, col] == self.OBSTACLE:
                        self.known_obstacle_grid[row, col] = self.OBSTACLE
                        collision_reason = "obstacle"
                    elif (
                        not self.allow_agent_overlap
                        and agent_index in conflicting_destinations
                    ):
                        collision_reason = "agent_conflict"

                if collision_reason is not None:
                    collisions += 1
                    final_position = previous_position

                    self._add_reward(
                        reward_parts=reward_parts,
                        reward_reasons=reward_reasons,
                        reward_name=RewardName.COLLISION,
                        agent_id=agent.id,
                        reason=f"Movement failed because of {collision_reason}.",
                        metadata={
                            "previous_position": previous_position,
                            "attempted_position": proposed_position,
                            "collision_reason": collision_reason,
                            "action": selected_action.name,
                        },
                    )

                    # IDLE applies only to failed movement against the map.
                    # It is not applied to agent_conflict or deliberate STAY.
                    if collision_reason in {"obstacle", "out_of_bounds"}:
                        failed_map_movements += 1
                        self._add_reward(
                            reward_parts=reward_parts,
                            reward_reasons=reward_reasons,
                            reward_name=RewardName.IDLE,
                            agent_id=agent.id,
                            reason=(
                                "Agent remained in the same cell because "
                                "movement failed against the map."
                            ),
                            metadata={
                                "position": previous_position,
                                "attempted_position": proposed_position,
                                "collision_reason": collision_reason,
                                "action": selected_action.name,
                            },
                        )
                    else:
                        agent_conflicts += 1

                    events.append(
                        {
                            "type": "collision",
                            "agent_id": agent.id,
                            "reason": collision_reason,
                            "attempted_position": proposed_position,
                        }
                    )
                else:
                    final_position = proposed_position
                    agent.position = final_position

            row, col = final_position
            movement_succeeded = (
                movement_attempted
                and collision_reason is None
                and final_position != previous_position
            )

            # A failed move or deliberate STAY is not counted as REVISIT.
            if movement_succeeded:
                if self.visited[row, col] == 0:
                    newly_visited += 1
                    self._add_reward(
                        reward_parts=reward_parts,
                        reward_reasons=reward_reasons,
                        reward_name=RewardName.NEW_CELL,
                        agent_id=agent.id,
                        reason="Agent entered a previously unvisited cell.",
                        metadata={
                            "previous_position": previous_position,
                            "new_position": final_position,
                            "action": selected_action.name,
                        },
                    )
                else:
                    revisits += 1
                    self._add_reward(
                        reward_parts=reward_parts,
                        reward_reasons=reward_reasons,
                        reward_name=RewardName.REVISIT,
                        agent_id=agent.id,
                        reason="Agent entered a previously visited cell.",
                        metadata={
                            "previous_position": previous_position,
                            "new_position": final_position,
                            "previous_visit_count": int(self.visited[row, col]),
                            "action": selected_action.name,
                        },
                    )
                self.visited[row, col] += 1

            revealed_count = self._reveal_around(final_position)
            newly_revealed += revealed_count
            if revealed_count > 0:
                self._add_reward(
                    reward_parts=reward_parts,
                    reward_reasons=reward_reasons,
                    reward_name=RewardName.NEW_MAP_CELL,
                    agent_id=agent.id,
                    reason=(
                        f"Agent revealed {revealed_count} previously unknown "
                        "map cell(s)."
                    ),
                    multiplier=float(revealed_count),
                    metadata={
                        "position": final_position,
                        "revealed_count": revealed_count,
                    },
                )

            found_victim_ids = self._detect_victims(final_position)
            found_count = len(found_victim_ids)
            newly_found += found_count
            if found_count > 0:
                self._add_reward(
                    reward_parts=reward_parts,
                    reward_reasons=reward_reasons,
                    reward_name=RewardName.VICTIM_FOUND,
                    agent_id=agent.id,
                    reason=f"Agent discovered {found_count} victim(s).",
                    multiplier=float(found_count),
                    metadata={
                        "victim_ids": found_victim_ids,
                        "agent_position": final_position,
                    },
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
        truncated = self.step_count >= self.max_steps and not terminated

        if terminated:
            self._add_reward(
                reward_parts=reward_parts,
                reward_reasons=reward_reasons,
                reward_name=RewardName.ALL_VICTIMS_FOUND,
                agent_id=None,
                reason="All victims were discovered.",
                metadata={"step_count": self.step_count},
            )

        if truncated:
            self._add_reward(
                reward_parts=reward_parts,
                reward_reasons=reward_reasons,
                reward_name=RewardName.TIMEOUT,
                agent_id=None,
                reason=(
                    "Episode reached max_steps before all victims were discovered."
                ),
                metadata={
                    "step_count": self.step_count,
                    "victims_found": int(self.victim_found.sum()),
                    "num_victims": self.num_victims,
                },
            )

        reward = float(sum(reward_parts.values()))
        info = self._get_info()
        info.update(
            {
                "reward_parts": reward_parts,
                "reward_reasons": reward_reasons,
                "events": events,
                "metrics": {
                    "newly_visited": newly_visited,
                    "newly_revealed": newly_revealed,
                    "newly_found": newly_found,
                    "collisions": collisions,
                    "revisits": revisits,
                    "failed_map_movements": failed_map_movements,
                    "deliberate_stays": deliberate_stays,
                    "agent_conflicts": agent_conflicts,
                    "coverage_percentage": self._get_coverage_percentage(),
                    "explored_percentage": self._get_explored_percentage(),
                },
            }
        )

        return self._get_obs(), reward, terminated, truncated, info

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
            "obstacle_grid": self.obstacle_grid.copy(),
            "victim_positions": list(self.victim_positions),
            "victim_found": self.victim_found.copy(),
            "agent_positions": [
                agent.position
                for agent in self.agents
            ],
        }

    def _normalise_actions(
        self,
        action: int | np.ndarray | list[int] | tuple[int, ...],
    ) -> np.ndarray:
        """
        Validate action input(s).
        """
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

    def _reveal_around(
        self,
        position: tuple[int, int],
    ) -> int:
        """
        Reveal the cells around the position (agent pos).

        Returns number of cells revealed (that is not victims).
        """
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

                # Update counter if the grid is newly discovered
                if self.known_obstacle_grid[row, col] == self.UNKNOWN:
                    newly_revealed += 1

                self.known_obstacle_grid[row, col] = (
                    self.OBSTACLE
                    if self.obstacle_grid[row, col] == self.OBSTACLE
                    else self.FREE
                )

        return newly_revealed

    def _detect_victims(
        self,
        position: tuple[int, int],
    ) -> list[int]:
        """
        Reveal victims if cell got victims.

        Returns list of newly found victims ids.
        """
        newly_found_ids: list[int] = []

        for victim_id, victim_position in enumerate(self.victim_positions):
            if self.victim_found[victim_id]:
                continue

            victim_row, victim_col = victim_position

            chebyshev_distance = get_chebyshev_distance(position, victim_position)

            if chebyshev_distance <= self.sensor_range:
                self.victim_found[victim_id] = 1
                self.discovered_victim_grid[
                    victim_row,
                    victim_col,
                ] = 1

                newly_found_ids.append(victim_id)

        return newly_found_ids

    def _get_reachable_cells(self, start_pos: tuple[int, int]):
        """
        Get reachable cells from starting position.

        Returns set{reachable_cells}
        """
        return get_reachable_cells(
                            grid=self.obstacle_grid,
                            start=start_pos,
                            blocked_value=int(self.OBSTACLE),
                            )

    def _get_reachable_cells_from_base(
        self,
    ) -> set[tuple[int, int]]:
        """
        Get reachable cells from base position.

        Returns set{reachable_cells}
        """       
        return self._get_reachable_cells(start_pos=self.base_position)

    def _get_reachable_cells_from_agent(
        self, agent: AgentState
    ) -> set[tuple[int, int]]:
        """
        Get reachable cells from agent's position.

        Returns set{reachable_cells}
        """
        return self._get_reachable_cells(start_pos=agent.position)

    def _get_reachable_cells_for_team(
        self,
    ) -> set[tuple[int, int]]:
        """
        Get team local reachable cells (for multiagent mode).

        Returns the union of cells reachable by active agents.
        """
        reachable_cells: set[tuple[int, int]] = set()

        for agent in self.agents:
            if agent.active:
                reachable_cells.update(
                    self._get_reachable_cells_from_agent(agent)
                )
                # print("Agent: ", agent)

        return reachable_cells
    
    def _get_obs(self) -> np.ndarray:
        """
        Observations of the env.

        Returns np.array observation={
                        known_free, # discovered free cells (pos)
                        known_obstacles, # discovered obstacles (pos)
                        unknown, # unknown cells (pos)
                        agent_grid, # agent grid mask
                        visited_grid, # visited mask
                        victim_grid, # discovered victim mask
                        } 
        """
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
        """
        Get info of the current state.

        info=[step count, # of agents, # of victims, victims found (current), all victim found (T/F),
            coverage % (cells that agent been to), explored % (cells that agent knew/aware of), positions of discovered victims]

        Return dict{info}.
        """
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

    ##### INFO UTILS #####
    def _get_coverage_percentage(self) -> float:
        """
        Get the ratio of cells agent(s) covered.

        Returns ratio.
        """
        # Reachable cells
        reachable_cells = self._get_reachable_cells_for_team()
        # print("In get coverage: reachable_cells = ", reachable_cells)

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
        """
        Set up env based on given hyperparameters in options.
        """
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
        """
        Check if position is in map.
        """
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
        """
        Validate env inputs.
        """
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
        grid_size=(9, 9),
        num_agents=1,
        num_victims=2,
        obstacle_ratio=0.30,
        sensor_range=1,
        max_steps=1000,
        render_mode="ansi",
        base_position=(0, 4),
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

        for reward_reason in info["reward_reasons"]:
            print(
                f"  - {reward_reason['reward_name']}: "
                f"{reward_reason['value']:+.3f} | "
                f"{reward_reason['reason']}"
            )

        env.render()

    print("Episode finished")
    print("Total reward:", total_reward)
    print("Final info:", info)

    env.close()


if __name__ == "__main__":
    run_random_policy_demo()
