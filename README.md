# Autonomous SAR Ops Core

Autonomous SAR Ops Core is a Python-based multi-agent search-and-rescue autonomy project organised around two distinct but connected subprojects:

1. **Search** — reinforcement learning agents explore an initially unknown environment, build a partial map, and detect victims.
2. **Rescue** — optimisation-based agents use discovered victim locations and the accumulated map to allocate suitable resources, plan routes, and execute rescue tasks.

The long-term objective is to integrate both stages into a hierarchical RL–MILP autonomy system in which learning handles uncertainty and exploration, while mathematical optimisation supports safe, feasible, and interpretable mission decisions.

The Unity visualisation layer is maintained separately so that Python autonomy logic and Unity assets can be versioned independently.

---

## Project Goal

The project aims to build a modular autonomy system for multi-agent search-and-rescue missions.

During the **search phase**, agents must:

* explore an unknown environment;
* scan surrounding cells;
* discover obstacles and traversable regions;
* identify victim locations;
* coordinate coverage while avoiding unnecessary revisits.

During the **rescue phase**, the system must:

* use the discovered map and victim locations;
* identify the required rescue resources;
* allocate suitable agents to victims;
* generate feasible rescue routes;
* adapt assignments when conditions change.

The complete mission flow is:

```
Unknown environment
        |
        v
RL-based search and exploration
        |
        v
Discovered map and victim locations
        |
        v
MILP-based resource and task allocation
        |
        v
Path planning and rescue execution
        |
        v
Mission evaluation and replanning
```

---

## Core Research Questions

### Search

> Can reinforcement learning agents efficiently explore a partially observable environment and locate hidden victims?

The search project focuses on:

* exploration under uncertainty;
* partial observability;
* map discovery;
* victim detection;
* multi-agent coordination;
* reward design;
* generalisation to unseen maps.

### Rescue

> Given discovered victims, agent capabilities, travel costs, and mission constraints, how should rescue tasks be allocated and scheduled?

The rescue project focuses on:

* heterogeneous resource allocation;
* capability matching;
* travel and service costs;
* battery or endurance limits;
* priority and deadline handling;
* route feasibility;
* dynamic replanning.

### Future Integration

> Can MILP-based planning improve or constrain RL policy decisions without replacing the exploration policy?

Potential future integrations include:

* MILP-generated action masks;
* MILP-guided target or frontier assignment;
* optimisation-based safety constraints;
* MILP-generated demonstrations for imitation learning;
* RL selection of when to invoke the optimiser;
* RL prediction of uncertain MILP parameters;
* MILP-based correction of infeasible policy outputs.

---

## System Architecture

```
Python Core Repository
│
├── Search Subproject
│   ├── Partially observable SAR environment
│   ├── RL exploration policy
│   ├── Classical exploration baselines
│   ├── Map memory
│   ├── Victim detection
│   └── Search evaluation
│
├── Rescue Subproject
│   ├── Mission-state conversion
│   ├── Greedy task allocator
│   ├── MILP resource allocator
│   ├── A* path planner
│   ├── Rescue execution environment
│   └── Replanning and evaluation
│
├── Integration Layer
│   ├── Search-to-rescue handover
│   ├── Shared map representation
│   ├── Agent and victim schemas
│   ├── Mission controller
│   └── Replay and experiment logging
│
└── JSON mission replay/state export
        │
        v
Unity Visualisation Repository
└── Visualises exploration, discoveries, assignments, routes, and rescue events
```

Python owns the decision-making logic. Unity acts as an external visualisation and replay client.

---

# Combined Mission Flow

The integrated system should follow this sequence:

```
1. Generate unknown mission environment

2. Deploy search agents

3. RL policy explores and scans map

4. Agents update shared map memory

5. Victims are discovered

6. Search state is converted into rescue state

7. MILP assigns suitable rescue resources

8. A* generates paths using discovered map

9. Agents execute rescue tasks

10. Mission controller detects disruptions

11. MILP replans when necessary

12. Results are logged and exported to Unity
```

Search and rescue may initially run sequentially.

A later version may allow them to overlap:

```
search continues
      +
new victims are added dynamically
      +
MILP periodically updates rescue assignments
```

---

# Repository Structure

```
autonomous-sar-ops-core/
│
├── README.md
├── pyproject.toml
├── requirements.txt
├── .gitignore
│
├── configs/
│   ├── search_env.yaml
│   ├── search_training.yaml
│   ├── rescue_planner.yaml
│   └── scenario.yaml
│
├── src/
│   └── autonomous_sar_ops/
│       │
│       ├── search/
│       │   ├── envs/
│       │   │   ├── sar_exploration_env.py
│       │   │   ├── observations.py
│       │   │   ├── rewards.py
│       │   │   └── map_generation.py
│       │   │
│       │   ├── agents/
│       │   │   ├── random_agent.py
│       │   │   ├── frontier_agent.py
│       │   │   └── rl_agent.py
│       │   │
│       │   ├── training/
│       │   │   ├── train_dqn.py
│       │   │   ├── train_ppo.py
│       │   │   └── callbacks.py
│       │   │
│       │   └── evaluation/
│       │       ├── search_metrics.py
│       │       └── evaluate_search.py
│       │
│       ├── rescue/
│       │   ├── allocation/
│       │   │   ├── greedy_allocator.py
│       │   │   ├── milp_allocator.py
│       │   │   └── constraints.py
│       │   │
│       │   ├── path_planning/
│       │   │   └── astar.py
│       │   │
│       │   ├── replanning/
│       │   │   └── replanner.py
│       │   │
│       │   └── evaluation/
│       │       ├── rescue_metrics.py
│       │       └── evaluate_rescue.py
│       │
│       ├── mission/
│       │   ├── agent.py
│       │   ├── victim.py
│       │   ├── map_state.py
│       │   ├── mission_state.py
│       │   ├── handover.py
│       │   └── mission_controller.py
│       │
│       ├── integration/
│       │   ├── search_rescue_pipeline.py
│       │   ├── state_converter.py
│       │   └── replay_exporter.py
│       │
│       ├── simulation/
│       │   ├── unity_bridge.py
│       │   ├── message_schema.py
│       │   └── simulation_runner.py
│       │
│       └── utils/
│           ├── config.py
│           ├── grid_utils.py
│           ├── geometry.py
│           └── logging.py
│
├── scenarios/
│   ├── search_small.yaml
│   ├── search_obstacles.yaml
│   ├── rescue_multi_victim.yaml
│   └── integrated_mission.yaml
│
├── experiments/
│   ├── search/
│   │   ├── run_random.py
│   │   ├── run_frontier.py
│   │   ├── train_ppo.py
│   │   └── results/
│   │
│   ├── rescue/
│   │   ├── run_greedy.py
│   │   ├── run_milp.py
│   │   └── results/
│   │
│   └── integrated/
│       ├── run_pipeline.py
│       └── results/
│
├── tests/
│   ├── search/
│   │   ├── test_exploration_env.py
│   │   ├── test_observations.py
│   │   ├── test_rewards.py
│   │   └── test_frontier_agent.py
│   │
│   ├── rescue/
│   │   ├── test_astar.py
│   │   ├── test_greedy_allocator.py
│   │   ├── test_milp_allocator.py
│   │   └── test_replanner.py
│   │
│   └── integration/
│       └── test_search_rescue_handover.py
│
└── docs/
    ├── architecture.md
    ├── search_design.md
    ├── rescue_formulation.md
    └── architecture.png
```

---

# Development Plan

## Phase 1 — Search Environment

Build a stable single-agent exploration environment with:

* unknown obstacle map;
* hidden victim locations;
* partial observations;
* direct movement actions;
* sensor-based map discovery;
* coverage and explored-percentage metrics;
* reward-component logging;
* deterministic test scenarios.

Start with:

```
1 agent
5 × 5 map
1 victim
no obstacles
```

Then gradually add obstacles and larger maps.

---

## Phase 2 — Search Baselines

Implement:

* random movement;
* least-visited neighbour;
* frontier exploration.

Evaluate all baselines over fixed test seeds.

---

## Phase 3 — RL Search Policy

Train:

* DQN for the single-agent environment;
* PPO as the main baseline;
* optional recurrent PPO for map memory and partial observability.

Compare RL against frontier exploration.

---

## Phase 4 — Multi-Agent Search

Extend the environment to:

* multiple homogeneous agents;
* shared discovered map;
* joint or per-agent actions;
* agent collision handling;
* duplicate-exploration penalties;
* team coverage metrics.

Do not introduce heterogeneous UAV and UGV behaviour until this version is stable.

---

## Phase 5 — Rescue Allocation

Create a separate rescue scenario using known victim locations.

Implement:

* A* route costs;
* nearest-agent greedy allocation;
* capability-aware greedy allocation;
* MILP task allocation;
* battery and priority constraints.

---

## Phase 6 — Search-to-Rescue Handover

Create a conversion layer that transforms search output into rescue input.

Validate that:

* only discovered map cells are available;
* only discovered victims are allocated;
* agent locations and states are transferred correctly;
* unreachable victims are identified;
* rescue routes do not use unknown or blocked cells.

---

## Phase 7 — Integrated Sequential Mission

Run the full pipeline:

```
RL search
→ victim discovery
→ map export
→ MILP allocation
→ A* rescue execution
→ mission metrics
```

This should become the first complete end-to-end portfolio demonstration.

---

## Phase 8 — Dynamic Integration

Allow search and rescue to operate concurrently.

Examples:

* MILP re-solves whenever a new victim is found;
* search agents continue exploring while rescue agents execute;
* changing map knowledge invalidates previous rescue routes;
* newly discovered obstacles trigger replanning.

---

## Phase 9 — Learning-Augmented Optimisation

Investigate tighter RL–MILP integration.

Possible experiments:

* MILP selects exploration regions while RL handles local movement;
* MILP assigns frontiers to search agents;
* RL predicts travel or service times for MILP;
* MILP generates expert trajectories;
* RL learns when MILP should be invoked;
* MILP repairs infeasible RL decisions;
* MILP action masks restrict unsafe RL actions.

---

# Initial Methods Compared

## Search

* Random policy
* Least-visited heuristic
* Frontier exploration
* DQN
* PPO

## Rescue

* Nearest-agent greedy allocation
* Capability-aware greedy allocation
* MILP allocation
* MILP allocation with replanning

## Integrated

* Frontier search + greedy rescue
* Frontier search + MILP rescue
* RL search + greedy rescue
* RL search + MILP rescue

This creates a clear experimental matrix and isolates the contribution of each component.

---

# How to Run

Install dependencies:

```bash
pip install -r requirements.txt
```

Run tests:

```bash
pytest
```

Run a random search baseline:

```bash
python experiments/search/run_random.py
```

Run frontier exploration:

```bash
python experiments/search/run_frontier.py
```

Train PPO:

```bash
python experiments/search/train_ppo.py
```

Run greedy rescue allocation:

```bash
python experiments/rescue/run_greedy.py
```

Run MILP rescue allocation:

```bash
python experiments/rescue/run_milp.py
```

Run the integrated mission:

```bash
python experiments/integrated/run_pipeline.py
```

Generated results should be saved under the relevant experiment directory.

---

# Unity Integration

Unity visualisation is maintained separately in:

```
autonomous-sar-ops-unity
```

The Python core exports JSON replay files containing:

* discovered map state over time;
* agent search movements;
* newly revealed cells;
* victim detections;
* search completion;
* rescue assignments;
* planned routes;
* rescue execution;
* replanning events;
* mission metrics.

Unity visualises the mission but does not make autonomy decisions.

---

# Current Scope

## Search v0.1

* single-agent RL exploration environment;
* partial map observations;
* hidden victims;
* random and frontier baselines;
* PPO or DQN training;
* exploration metrics;
* reward-reason logging.

## Rescue v0.1

* known victim locations;
* known discovered map;
* homogeneous rescue agents;
* A* route costs;
* greedy allocation;
* basic MILP allocation;
* rescue metrics.

## Integration v0.1

* sequential search-to-rescue handover;
* shared mission-state schema;
* RL search followed by MILP rescue;
* JSON replay export;
* Unity visualisation support.

---

# Future Roadmap

## v0.2 — Multi-Agent Search

* multiple search agents;
* shared map memory;
* multi-agent PPO;
* coordination and collision handling;
* frontier assignment baselines.

## v0.3 — Heterogeneous Resources

* UAV and UGV agents;
* different movement constraints;
* different sensing ranges;
* no-fly zones;
* capability-aware rescue allocation.

## v0.4 — Dynamic Search and Rescue

* concurrent search and rescue;
* victim discovery during execution;
* dynamic MILP replanning;
* agent failures;
* changing obstacles.

## v0.5 — Learning-Augmented MILP

* MILP-guided RL actions;
* optimisation-based action masks;
* RL-predicted MILP parameters;
* expert demonstrations from optimisation;
* RL-controlled optimiser invocation.

## v0.6 — ROS2 and Gazebo

* ROS2 mission nodes;
* Gazebo robot simulation;
* map and state exchange;
* robotics middleware integration.

## v0.7 — Perception

* synthetic Unity data generation;
* victim and hazard detection;
* uncertainty-aware detections;
* detection-triggered replanning.

## v0.8 — API and Dashboard

* FastAPI mission service;
* Streamlit dashboard;
* live mission state;
* experiment comparison;
* replay and metric visualisation.

---

# Portfolio Positioning

This project should be presented as:

## Hierarchical Search-and-Rescue Autonomy with Reinforcement Learning and Mathematical Optimisation

It demonstrates:

* Gymnasium environment design;
* reinforcement learning;
* partial observability;
* multi-agent systems;
* exploration and map memory;
* mathematical optimisation;
* task allocation;
* path planning;
* replanning;
* simulation;
* experiment design;
* software engineering;
* interpretable decision-making.

The key design principle is:

```
RL handles uncertain exploration.

MILP handles constrained rescue allocation.

The mission controller connects both stages.
```
