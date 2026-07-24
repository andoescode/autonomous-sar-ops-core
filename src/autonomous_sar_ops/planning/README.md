# Subproject 2 — MILP Rescue Allocation

## Objective

Use the discovered map and known victim locations to assign appropriate rescue resources and produce feasible mission plans.

The rescue optimiser does not search for victims. It receives the output of the search phase.

---

## Search-to-Rescue Handover

The search stage produces a mission-state object containing:

```
discovered map
known obstacles
agent locations
agent capabilities
remaining battery or endurance
victim locations
victim requirements
victim priorities
search completion status
```

This state becomes the input to the rescue allocator.

Example:

```python
mission_state = {
    "known_map": ...,
    "agents": [
        {
            "id": "uav_01",
            "type": "uav",
            "position": [4, 7],
            "battery": 72.5,
            "capabilities": ["scan", "medical_supply"],
        },
        {
            "id": "ugv_01",
            "type": "ugv",
            "position": [1, 2],
            "battery": 88.0,
            "capabilities": ["transport", "heavy_payload"],
        },
    ],
    "victims": [
        {
            "id": "victim_01",
            "position": [8, 5],
            "priority": 3,
            "required_capabilities": ["medical_supply"],
        }
    ],
}
```

---

## Rescue Problem

The rescue system must decide:

* which agent should respond to each victim;
* whether multiple agents are required;
* the order in which victims should be served;
* whether assignments are feasible;
* which routes agents should follow;
* when replanning is required.

---

## Rescue Constraints

Possible MILP constraints include:

* each victim must be assigned at most once;
* only compatible agents may serve a victim;
* required capabilities must be available;
* agents cannot exceed battery or endurance limits;
* routes must use currently known traversable cells;
* one agent cannot execute overlapping tasks;
* high-priority victims should be served earlier;
* some victims may require multiple resources;
* agents may need to return to base;
* unreachable victims must remain explicitly unassigned.

---

## Rescue Objective

A possible objective is:

[
\min
\left(
\alpha \cdot \text{response time}
+
\beta \cdot \text{travel cost}
+
\gamma \cdot \text{priority-weighted lateness}
+
\delta \cdot \text{battery usage}
+
\epsilon \cdot \text{unassigned victims}
\right)
]

The objective weights should be configurable for experiments.

---

## Rescue Methods Compared

Initial methods:

* nearest-compatible-agent heuristic;
* priority-first greedy allocation;
* greedy allocation with A* routes;
* MILP allocation with A* travel costs;
* MILP allocation with battery and capability constraints;
* MILP allocation with disruption-aware replanning.

Future methods:

* rolling-horizon optimisation;
* stochastic MILP;
* robust optimisation;
* multi-stage rescue scheduling;
* coalition formation for multi-agent tasks;
* learning-augmented MILP.

---

## Rescue Replanning

The rescue plan should be updated when:

* a route becomes blocked;
* a new victim is discovered;
* victim priority changes;
* an agent becomes unavailable;
* battery consumption differs from expectation;
* a victim requires additional resources;
* the discovered map changes;
* an assignment becomes infeasible.

The replanning loop is:

```
Observe mission state
        |
        v
Update travel and feasibility data
        |
        v
Re-solve allocation model
        |
        v
Generate new routes
        |
        v
Continue execution
```

---

## Rescue Metrics

The rescue subproject should evaluate:

* rescue completion rate;
* average victim response time;
* time to rescue all reachable victims;
* priority-weighted lateness;
* total travel distance;
* battery consumption;
* unassigned-victim count;
* infeasible-assignment count;
* number of replans;
* solver runtime;
* optimality gap;
* resource utilisation.

---
