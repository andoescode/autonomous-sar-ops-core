from __future__ import annotations

import math
from datetime import datetime
import highspy
import polars as pl

from .utils import *
# from .simple_data import *

class DispatchModel:
    """
    Input:
        tasks_df:
            task_id, task_type, task_des, x, y,
            priority_lvl, priority_weight,
            created_time, due_time,
            estimated_service_duration, status

        resources_df:
            resource_id, resource_type, resource_des, x, y,
            velocity, time_shift, start_shift, end_shift, status

    """
    UNSCHEDULED_TASK_PENALTY = 1000.0
    PRIORITY_ORDER_PENALTY = 100.0
    BIG_M_BUFFER = 10

    def __init__(
        self,
        tasks_df: pl.DataFrame,
        resources_df: pl.DataFrame,
        *,
        time_spent_ratio: float = 6.0, # ratios of min time spent
        late_and_priority_penalty_ratio: float = 4.0, # ratios of penalty
        output_flag: bool = False, 
    ):
        self.tasks_df = tasks_df
        self.resources_df = resources_df
        self.time_spent_ratio = time_spent_ratio
        self.late_and_priority_penalty_ratio = late_and_priority_penalty_ratio

        self.solver = highspy.Highs()
        self.solver.setOptionValue("output_flag", output_flag)
        self.solver.setOptionValue("log_to_console", output_flag)

        # Allowed task - resources mapping 
        self.allowed_map: dict[str, set[str]] = {
            "light duty task": {"R2", "R3"},
            "heavy duty task": {"R1"},
            "T3": {"R3"},
            "T4": {"R1", "R3"},
            "T5": {"R2"},
        }

        # Sets of Tasks and Resources (represent by int)
        self.T: list[int] = [] # tasks
        self.R: list[int] = [] # resources

        # Raw rows (store other info)
        self.task_rows: dict[int, dict] = {}
        self.resource_rows: dict[int, dict] = {}

        # Parameters
        self.base_time: datetime | None = None # time start

        # Store info from raw rows into vars dict 
        # {int represent task : data}
        self.task_type: dict[int, str] = {} 
        self.priority_weight: dict[int, float] = {}
        self.release_min: dict[int, int] = {} # time that task is created
        self.due_min: dict[int, int] = {}
        self.service_min: dict[int, int] = {}
        self.task_loc: dict[int, tuple[int, int]] = {}

        # {int represent resource : data}
        self.resource_type: dict[int, str] = {}
        self.velocity: dict[int, int] = {}
        self.shift_start_min: dict[int, int] = {}
        self.shift_end_min: dict[int, int] = {}
        self.work_limit_min: dict[int, int] = {}
        self.resource_loc: dict[int, tuple[int, int]] = {}

        # Feasible assignment pairs (t, r)
        self.A: list[tuple[int, int]] = [] # feasible (t, r) assignment based on allowed map
        self.travel0: dict[tuple[int, int], int] = {} # time trsvel from r_i to t 

        # Feasible sequencing arcs (i, t, r)
        self.U: list[tuple[int, int, int]] = [] # upcoming/next task t for resource from task i
        self.travel: dict[tuple[int, int, int], int] = {} # time travel between task i and task t

        self.tasks_for_resource: dict[int, list[int]] = {} # list of tasks that resources can do 

        self.M: int = 0

        # Decision variables
        self.y = {}       # y[t] = 1 if task t scheduled
        self.a = {}       # a[t,r] = 1 if task t assigned to resource r
        self.first = {}   # first[t,r] = 1 if t is first task on resource r
        self.u = {}       # u[i,t,r] = 1 if t immediately follows i on resource r

        self.s = {}       # service start time of task t (s[t] = service start time of task t)
        self.c = {}       # completion time of task t (c[t] = completion time of task t)
        self.late = {}    # lateness of task t (late[t] = latesness time of task t)
        self.v = {}       # v[k,l] = 1 if lower-priority task k is scheduled before higher-priority task l

    # BUILD
    def build_model(self) -> None:
        self._prepare_sets_and_parameters()
        self._build_feasible_assignment_pairs()
        self._build_feasible_route_arcs()
        self._build_big_M_method()
        self._define_variables()
        self._construct_constraints()
        self._construct_objective()

    def _prepare_sets_and_parameters(self) -> None:
        """
        Set values for sets and paramters. 
        Prepare unscheduled tasks and available resources to assign.
        """
        # Keep pending tasks and available resources
        tasks_df = self.tasks_df.filter(pl.col("status") == "pending")
        resources_df = self.resources_df.filter(pl.col("status") == "available")

        self.T = tasks_df["task_id"].to_list() # tasks id
        self.R = resources_df["resource_id"].to_list() # resources id

        self.task_rows = {row["task_id"]: row for row in tasks_df.iter_rows(named=True)}
        self.resource_rows = {row["resource_id"]: row for row in resources_df.iter_rows(named=True)}

        # choose a common time origin
        min_task_time = min(tasks_df["created_time"].to_list()) if len(tasks_df) > 0 else None # earliest created time of the tasks
        min_resource_time = min(resources_df["start_shift"].to_list()) if len(resources_df) > 0 else None # earliest start shift of the resources

        if min_task_time is None and min_resource_time is None:
            raise ValueError("No tasks or resources available.")
        elif min_task_time is None:
            self.base_time = min_resource_time
        elif min_resource_time is None:
            self.base_time = min_task_time
        else:
            self.base_time = min(min_task_time, min_resource_time)

        # task parameters
        for t in self.T:
            row = self.task_rows[t]
            self.task_type[t] = row["task_type"]
            self.priority_weight[t] = float(row["priority_weight"])
            self.release_min[t] = max(0, minutes_between(self.base_time, row["created_time"]))
            self.due_min[t] = max(0, minutes_between(self.base_time, row["due_time"]))
            self.service_min[t] = int(row["estimated_service_duration"])
            self.task_loc[t] = (int(row["x"]), int(row["y"]))

        # resource parameters
        for r in self.R:
            row = self.resource_rows[r]
            self.resource_type[r] = row["resource_type"]
            self.velocity[r] = int(row["velocity"])
            self.shift_start_min[r] = max(0, minutes_between(self.base_time, row["start_shift"]))
            self.shift_end_min[r] = max(0, minutes_between(self.base_time, row["end_shift"]))
            self.work_limit_min[r] = int(row["time_shift"])
            self.resource_loc[r] = (int(row["x"]), int(row["y"]))

    def _assignment_feasible(self, t: int, r: int) -> bool:
        """
        Check if the assignment (t, r) is possible or not, for all t, r satisfy:
        + (t,r) in allowed map
        + whether the time take to finish the task is within the working time shift of resource r
        """
        # (t, r) has to be in allowed map
        if self.resource_type[r] not in self.allowed_map[self.task_type[t]]:
            return False

        # time travel from task t to resource r
        travel0 = travel_minute(manhattan(self.task_loc[t], self.resource_loc[r]), self.velocity[r])

        earliest_start = max(self.release_min[t], self.shift_start_min[r] + travel0)
        earliest_finish = earliest_start + self.service_min[t]

        return earliest_finish <= self.shift_end_min[r]

    def _build_feasible_assignment_pairs(self) -> None:
        """
        Build possible (t, r) pairs, for all t, r satisfy:
        + _assignment_feasible(t,r) = True
        + travel time for each legal (t, r)
        """
        self.A.clear()
        self.travel0.clear()
        self.tasks_for_resource = {r: [] for r in self.R} # feasible tasks that resources can do

        for t in self.T:
            for r in self.R:
                # (t, r) has to be feasible
                if not self._assignment_feasible(t,r):
                    continue

                tr0 = travel_minute(manhattan(self.task_loc[t], self.resource_loc[r]), self.velocity[r])

                # Log <(t, r) : travel time from t to r> pairs of possible assignments 
                self.A.append((t,r))
                self.travel0[(t,r)] = tr0
                self.tasks_for_resource[r].append(t)

    def _build_feasible_route_arcs(self) -> None:
        """
        Build possible prescheduled route with an expected travel time of resource r from finishing current task to the next feasible task.
        """
        self.U.clear()
        self.travel.clear()

        for r in self.R:
            tasks = self.tasks_for_resource[r] # tasks can be done by resource r
            for i in tasks:
                for t in tasks:
                    if i == t:
                        continue

                    tr = travel_minute(manhattan(self.task_loc[i], self.task_loc[t]), self.velocity[r])

                    # Travel time of resource r from finishing task i -> moving to next task t
                    self.U.append((i,t,r))
                    self.travel[(i,t,r)] = tr

    def _build_big_M_method(self) -> None:
        max_shift_end = max(self.shift_end_min.values(), default=0) # latest shift end of resources (minutes)
        max_due = max(self.due_min.values(), default=0) # latest deadline of tasks (minutes)
        max_service = max(self.service_min.values(), default=0) # max service duration (minutes)
        max_travel0 = max(self.travel0.values(), default=0) # max travel time from resource to task (minutes)
        max_travel = max(self.travel.values(), default=0) # max travel time of resource from task_i to task_j (minutes)

        self.M = max(max_shift_end, max_due) + max_service + max(max_travel0, max_travel) + DispatchModel.BIG_M_BUFFER # max time
    
    # VARIABLES
    def _define_variables(self) -> None:
        self.y = {t: self.solver.addBinary() for t in self.T} # 1: scheduled, 0: not scheduled
        self.s = {t: self.solver.addIntegral(lb=0) for t in self.T} # service start time of task t
        self.c = {t: self.solver.addIntegral(lb=0) for t in self.T} # completion time of task t
        self.late = {t: self.solver.addIntegral(lb=0) for t in self.T} # lateness time of task t

        self.a = {(t,r): self.solver.addBinary() for (t,r) in self.A} # 1: assigned, 0: not assigned
        self.first = {(t,r): self.solver.addBinary() for (t,r) in self.A} # first task of resource r 1: chosen, 0: not chosen
        self.u = {(i,t,r): self.solver.addBinary() for (i,t,r) in self.U} # schedule upcoming tasks of resource r 1: chosen, 0: not chosen
        self.v = {
            (k, l): self.solver.addBinary()
            for k in self.T
            for l in self.T
            if k != l and self.priority_weight[k] < self.priority_weight[l]
        } # 1: priority incorrect order; 0: priority correct order

    # CONSTRAINTS
    def _construct_constraints(self) -> None:
        self._task_assignment_constraints()
        self._resource_flow_constraints()
        self._start_completion_constraints()
        self._task_release_constraints()
        self._first_task_timing_constraints()
        self._precedence_constraints()
        self._shift_end_constraints()
        self._lateness_constraints()
        self._zero_if_unscheduled_constraints()
        self._resource_capacity_constraints()
        self._priority_order_constraints()

    def _task_assignment_constraints(self) -> None:
        """
        Each scheduled task assigned to exactly one resource.
        """
        for t in self.T:
            assign_vars = [self.a[(t,r)] for r in self.R if (t,r) in self.a] # assignment status of (t,r)
            if assign_vars:
                self.solver.addConstr(sum(assign_vars) == self.y[t]) # number of assigned (t,r) == state of scheduled tasks = 1
            else:
                self.solver.addConstr(self.y[t] == 0) 

    def _resource_flow_constraints(self) -> None:
        """
        Line up sequence of resources: each resource starts with one random feasible task, has only one previous task and one upcoming task lining up.
        """
        for r in self.R:
            tasks = self.tasks_for_resource[r]

            # at most one first task
            first_vars = [self.first[(t,r)] for t in tasks]
            if first_vars:
                self.solver.addConstr(sum(first_vars) <= 1)

            for t in tasks:
                incoming = [self.u[(i,t,r)] for i in tasks if i != t and (i,t,r) in self.u] # t is the upcoming task
                outgoing = [self.u[(t,k,r)] for k in tasks if k != t and (t,k,r) in self.u] # t is the task is about to be finished

                # assigned task must either be first or have exactly one previous task
                self.solver.addConstr(self.first[(t,r)] + sum(incoming) == self.a[(t,r)])

                # assigned task has at most one next task
                self.solver.addConstr(sum(outgoing) <= self.a[(t,r)])

                # first implies assigned
                self.solver.addConstr(self.first[(t,r)] <= self.a[(t,r)])

    def _start_completion_constraints(self) -> None:
        """
        Completion time point is calculated from after resource travel to task --> resource finished the task service.
        """
        for t in self.T:
            self.solver.addConstr(self.c[t] == self.s[t] + self.service_min[t]) # completion time point = time task service start point (after resource travel to the task) + service time point

    def _task_release_constraints(self) -> None:
        """
        Service of the task needs to be started before the task has been introduced.
        """
        for t in self.T:
            # only activate when task is assigned, if task is assigned -> time service start point > 0/created time of task
            self.solver.addConstr(self.M * (1 - self.y[t]) + self.s[t] >=  self.release_min[t]) #  release_min[t] <= max time * (1 - task assigned state[0 or 1] + time task service start point)

    def _first_task_timing_constraints(self) -> None:
        """
        Resource need to have enough time to travel to task for first task when start shift.
        """
        for (t,r), var in self.first.items():
            # only active when first[t,r] = 1, if task is first -> task service start time point has to be after start shift time + travel time of the resource. 
            self.solver.addConstr(
                self.s[t] + self.M * (1 - var) >= self.shift_start_min[r] + self.travel0[(t,r)] # time service start point + max time * (1 - is_first(t,r)[0 or 1]) >= time from start shift + travel time 
            )

    def _precedence_constraints(self) -> None:
        """
        Resource need to have enough time to travel from prev task to next scheduled task before starting service for next task.
        """
        for (i,t,r), var in self.u.items():
            # only active when sequence(prev, current) = 1, if there is prev task -> task service start time point has to start after the time resource travel from prev task to new task.
            self.solver.addConstr(
                self.s[t] + self.M * (1 - var)>= self.c[i] + self.travel[(i,t,r)] # time service start point + max time * (1 - sequence(previous task, curren task)[0 or 1]) >= time traveling from prev task to current task
            )

    def _shift_end_constraints(self) -> None:
        """
        Task has to be done within the working time frame of the assigned resource.
        """
        for (t,r), var in self.a.items():
            # only activate when (task, resource) is assigned, if assigned -> task has to be ended before the shift of resource end.
            self.solver.addConstr(
                self.c[t] <= self.shift_end_min[r] + self.M * (1 - var) # time task completed <= shift end of the assigned resource + max time * (1 - assigned(r,t)[0 or 1])
            )

    def _lateness_constraints(self) -> None:
        """
        Time late is how much time spent to complete task over due time.
        """
        for t in self.T:
            self.solver.addConstr(self.late[t] >= self.c[t] - self.due_min[t]) # time late >= time complete task - due time

    def _zero_if_unscheduled_constraints(self) -> None:
        """
        Service time point, complete time point and late time <= 0 if task is not scheduled.
        """
        for t in self.T:
            self.solver.addConstr(self.s[t] <= self.M * self.y[t]) # service time point <= 0 if task is not scheduled
            self.solver.addConstr(self.c[t] <= self.M * self.y[t]) # complete time point <= 0 if task is not scheduled
            self.solver.addConstr(self.late[t] <= self.M * self.y[t]) # late time <= 0 if task is not scheduled

    def _resource_capacity_constraints(self) -> None:
        for r in self.R:
            assigned_service = [
                self.service_min[t] * self.a[(t, r)]
                for t in self.tasks_for_resource[r]
                if (t, r) in self.a
            ]

            first_travel = [
                self.travel0[(t, r)] * self.first[(t, r)]
                for t in self.tasks_for_resource[r]
                if (t, r) in self.first
            ]

            inter_task_travel = [
                self.travel[(i, j, r)] * self.u[(i, j, r)]
                for (i, j, rr) in self.U
                if rr == r
            ]

            self.solver.addConstr(
                sum(assigned_service) + sum(first_travel) + sum(inter_task_travel)
                <= self.work_limit_min[r]
            )

    def _priority_order_constraints(self) -> None:
        """
        Penalise lower-priority tasks that start before higher-priority tasks.
        """
        epsilon = 1  # one-minute separation to avoid equal-start ambiguity

        for (k, l), var in self.v.items():
            # If both tasks are scheduled and v[k,l] = 0 aka order correct -> s[k] >= s[l] + epsilon 
            # => start time of task k is after/bigger than start time of task l (aka task l is scheduled before task k)
            self.solver.addConstr(
                self.s[k]
                + self.M * var
                + self.M * (2 - self.y[k] - self.y[l])
                >= self.s[l] + epsilon
            )

            # If either task is not scheduled, violation variable v[k,l] must be 0.
            self.solver.addConstr(var <= self.y[k])
            self.solver.addConstr(var <= self.y[l])

    # OBJECTIVE
    def _construct_objective(self) -> None:
        """
        Minimize:
            time_spent_ratio * (total completion time/time after finishing service + total travel time) # scheduling + late_and_priority_penalty_ratio  * (priority-weighted lateness + unscheduled high-priority task penalty) # penalty
        """
        obj = 0

        # completion times/ time after finishing task service
        for t in self.T:
            obj += self.time_spent_ratio * self.c[t]

        # travel from origin to first task
        for (t,r), var in self.first.items():
            obj += self.time_spent_ratio * self.travel0[(t,r)] * var # 0 if (t,r) is not first

        # inter-task travel (task i to task t)
        for (i,t,r), var in self.u.items():
            obj += self.time_spent_ratio * self.travel[(i,t,r)] * var # 0 if there is no task sequence

        # lateness weighted by priority
        for t in self.T:
            obj += self.late_and_priority_penalty_ratio * self.priority_weight[t] * self.late[t]

        # unscheduled penalty weighted by priority
        for t in self.T:
            obj += self.late_and_priority_penalty_ratio * DispatchModel.UNSCHEDULED_TASK_PENALTY * self.priority_weight[t] * (1 - self.y[t]) # 0 if task is scheduled

        # priority-order violation penalty
        for (k, l), var in self.v.items():
            obj += self.late_and_priority_penalty_ratio * DispatchModel.PRIORITY_ORDER_PENALTY * var

        self.solver.minimize(obj)

    # SOLVE / OUTPUT
    def solve_model(self, relative_gap: float = 0.0) -> None:
        """
        Solve model with closet optimised gap = relative_gap.
        """
        self.solver.setOptionValue("mip_rel_gap", relative_gap)
        self.solver.solve()

    def get_task_plan(self) -> pl.DataFrame:
        """
        Get solved task plan dataframe.
        """
        rows = []

        # Precompute chosen resource per task
        assigned_resource_of_task = {
            t: r
            for (t, r), var in self.a.items()
            if round(self.solver.variableValue(var)) == 1
        }

        # Precompute time values once
        start_min_of_task = {
            t: int(round(self.solver.variableValue(self.s[t])))
            for t in self.T
        }
        completion_min_of_task = {
            t: int(round(self.solver.variableValue(self.c[t])))
            for t in self.T
        }
        lateness_min_of_task = {
            t: int(round(self.solver.variableValue(self.late[t])))
            for t in self.T
        }
        scheduled_flag_of_task = {
            t: round(self.solver.variableValue(self.y[t])) == 1
            for t in self.T
        }

        for t in self.T:
            smin = start_min_of_task[t]
            cmin = completion_min_of_task[t]
            lmin = lateness_min_of_task[t]

            rows.append(
                {
                    "task_id": t,
                    "scheduled": scheduled_flag_of_task[t],
                    "resource_id": assigned_resource_of_task.get(t),
                    "start_min": smin,
                    "completion_min": cmin,
                    "lateness_min": lmin,
                    "start_time": self.base_time + timedelta(minutes=smin),
                    "completion_time": self.base_time + timedelta(minutes=cmin),
                }
            )

        return pl.DataFrame(rows).sort("task_id")

    def get_resource_routes(self) -> dict[int, list[int]]:
        """
        Get list of tasks that resources done.
        """
        routes: dict[int, list[int]] = {}

        # Precompute first task of each resource
        first_task_of_resource = {
            r: next(
                (
                    t
                    for t in self.tasks_for_resource[r]
                    if (t, r) in self.first
                    and round(self.solver.variableValue(self.first[(t, r)])) == 1
                ),
                None,
            )
            for r in self.R
        }

        # Precompute chosen next task map: (current_task, resource) -> next_task
        next_task_of = {
            (i, r): j
            for (i, j, r), var in self.u.items()
            if round(self.solver.variableValue(var)) == 1
        }

        for r in self.R:
            route = []
            current = first_task_of_resource[r]

            while current is not None:
                route.append(current)
                current = next_task_of.get((current, r))

            routes[r] = route

        return routes

    def get_objective_value(self) -> float:
        return self.solver.getInfo().objective_function_value
    
if __name__ == "__main__":
    model = DispatchModel(tasks_df, resources_df, time_spent_ratio=4.0, late_and_priority_penalty_ratio=6.0, output_flag=True)
    model.build_model()
    model.solve_model(relative_gap=0.07)

    print("Objective:", model.get_objective_value())
    print(model.get_task_plan())

    print("\nRoutes:")
    for rid, route in model.get_resource_routes().items():
        print(rid, "->", route)