# Subproject 1 — RL Search and Exploration

## Objective

Train one or more agents to search an initially unknown grid environment and discover hidden victims.

The RL policy must decide where to move using only currently available information rather than the complete ground-truth map.

---

## Search Environment

The Gymnasium-style search environment contains:

* hidden obstacle map;
* partially observed occupancy map;
* unknown, free, and blocked cells;
* agent positions;
* visited-cell history;
* hidden victim locations;
* discovered victim locations;
* configurable sensor range;
* episode time limit.

The simulator knows the complete map, but the policy observes only the discovered portion.

```
Simulator knows:
- complete obstacle map
- hidden victim locations
- valid reachable regions

Policy observes:
- discovered free cells
- discovered obstacles
- unknown cells
- agent positions
- visited cells
- discovered victims
```

---

## Search Actions

Initial discrete actions:

```
STAY - UP - DOWN - LEFT - RIGHT
```

For multiple agents, the environment receives one action per agent.

Future versions may add:

* UAV diagonal movement;
* variable movement costs;
* scanning actions;
* communication actions;
* local rather than global observations.

---

## Search Rewards

The search reward encourages useful exploration while discouraging inefficient behaviour.

Possible reward components:

* positive reward for entering a new cell;
* positive reward for revealing new map cells;
* large positive reward for discovering a victim;
* completion reward for locating all victims;
* small penalty per environment step;
* penalty for revisiting known cells;
* penalty for invalid movements;
* penalty for unnecessary waiting;
* timeout penalty.

Each reward contribution is logged with:

* reward type;
* value;
* agent ID;
* reason;
* relevant event metadata.

---

## Search Baselines

The RL policy should be compared against classical and simple heuristic baselines:

* random movement;
* least-visited-neighbour policy;
* frontier-based exploration;
* greedy information-gain exploration;
* optional A*-to-frontier baseline.

These baselines make it possible to determine whether the RL policy genuinely improves exploration efficiency.

---

## Search Methods

Initial methods:

* Random policy
* Frontier exploration
* DQN for single-agent exploration
* PPO for single-agent exploration
* PPO with multi-agent joint actions

Future methods:

* parameter-sharing PPO;
* MAPPO;
* recurrent PPO for partial observability;
* communication-aware policies;
* curiosity or intrinsic-motivation rewards;
* graph neural network policies.

---

## Search Metrics

The search project should evaluate:

* victim discovery success rate;
* percentage of victims found;
* time to first victim;
* time to find all victims;
* coverage percentage;
* explored percentage;
* revisit ratio;
* collision count;
* unnecessary-stay count;
* average episode reward;
* generalisation to unseen maps;
* generalisation to different obstacle densities;
* policy inference time.

### Coverage Percentage

Measures how much of the reachable free environment agents physically visited.

```
visited reachable free cells
----------------------------
total reachable free cells
```

### Explored Percentage

Measures how much of the complete environment has been observed by agent sensors.

```
known map cells
---------------
total map cells
```

Coverage represents physical traversal. Explored percentage represents accumulated knowledge.

---
