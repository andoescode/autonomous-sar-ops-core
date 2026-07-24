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
