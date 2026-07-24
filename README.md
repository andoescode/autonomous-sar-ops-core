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
