# Autonomous SAR Ops Core — Plan

## Goal

Build a small autonomous search-and-rescue simulation project that demonstrates:

* Gym-style environment design
* multi-agent mission logic
* A* path planning
* greedy vs MILP task allocation
* disruption-aware replanning
* Unity-based mission visualisation

---

## System Design

The system includes 2 separated repos, responsible for different sections of system: 

* **LOGIC**: `autonomous-sar-ops-core` = main resume repo, Python environment/planning/optimisation.

* **VISUALISATION**: `autonomous-sar-ops-unity` = Unity visualisation client.

The Python core exports mission state and replay data as JSON. Unity reads the replay file and visualises the mission.

## Development Plan

### Phase 1 — Search Environment

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

### Phase 2 — Search Baselines

Implement:

* random movement;
* least-visited neighbour;
* frontier exploration.

Evaluate all baselines over fixed test seeds.

---

### Phase 3 — RL Search Policy

Train:

* DQN for the single-agent environment;
* PPO as the main baseline;
* optional recurrent PPO for map memory and partial observability.

Compare RL against frontier exploration.

---

### Phase 4 — Multi-Agent Search

Extend the environment to:

* multiple homogeneous agents;
* shared discovered map;
* joint or per-agent actions;
* agent collision handling;
* duplicate-exploration penalties;
* team coverage metrics.

Do not introduce heterogeneous UAV and UGV behaviour until this version is stable.

---

### Phase 5 — Rescue Allocation

Create a separate rescue scenario using known victim locations.

Implement:

* A* route costs;
* nearest-agent greedy allocation;
* capability-aware greedy allocation;
* MILP task allocation;
* battery and priority constraints.

---

### Phase 6 — Search-to-Rescue Handover

Create a conversion layer that transforms search output into rescue input.

Validate that:

* only discovered map cells are available;
* only discovered victims are allocated;
* agent locations and states are transferred correctly;
* unreachable victims are identified;
* rescue routes do not use unknown or blocked cells.

---

### Phase 7 — Integrated Sequential Mission

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

### Phase 8 — Dynamic Integration

Allow search and rescue to operate concurrently.

Examples:

* MILP re-solves whenever a new victim is found;
* search agents continue exploring while rescue agents execute;
* changing map knowledge invalidates previous rescue routes;
* newly discovered obstacles trigger replanning.

---

### Phase 9 — Learning-Augmented Optimisation

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
