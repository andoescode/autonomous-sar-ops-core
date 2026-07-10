import math 
import polars as pl

from .dataclasses import *

def manhattan(a: tuple[int, int], b: tuple[int, int]) -> int:
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def travel_minute(a: int, b: int) -> int:
    return math.ceil(a / max(1, b))

def minutes_between(t1, t2) -> int:
    return int((t2 - t1).total_seconds() // 60)

def tier_rank(sla: SLATier) -> int:
    """
    Higher is more important.
    TIER_1 -> 4, TIER_2 -> 3, TIER_3 -> 2, TIER_4 -> 1
    """
    return 5 - sla.tier_id

# Lookup maps
TASK_TYPE_MAP = {t.task_type_label: t for t in TaskType}
SLA_TIER_MAP = {s.label: s for s in SLATier}
TASK_STATUS_MAP = {s.value: s for s in TaskStatus}
TASK_OUTCOME_MAP = {o.value: o for o in TaskCompletionOutcome}

def tasks_df_to_objects(
    tasks_df: pl.DataFrame,
    resources_by_id: Optional[dict[int, Resource]] = None,
) -> list[Task]:
    """
    Convert a Polars DataFrame into list[Task].
    """
    if resources_by_id is None:
        resources_by_id = {}

    tasks: list[Task] = []

    for row in tasks_df.iter_rows(named=True):
        # support both (x, y) and (task_x, task_y)
        if "location" in row and row["location"] is not None:
            location = tuple(row["location"])
        else:
            x = row["x"] if "x" in row else row["task_x"]
            y = row["y"] if "y" in row else row["task_y"]
            location = (int(x), int(y))

        assigned_resource = None
        assigned_resource_id = row.get("assigned_resource_id")
        if assigned_resource_id is not None:
            assigned_resource = resources_by_id.get(int(assigned_resource_id))

        allowed_resources_distance = row.get("allowed_resources_distance")
        if allowed_resources_distance is None:
            allowed_resources_distance = {}

        status_str = row.get("status", TaskStatus.PENDING.value)
        outcome_str = row.get("completion_outcome", TaskCompletionOutcome.UNKNOWN.value)

        task = Task(
            task_id=int(row["task_id"]),
            task_type=TASK_TYPE_MAP[row["task_type"]],
            task_des=row["task_des"],
            location=location,
            priority_lvl=SLA_TIER_MAP[row["priority_lvl"]],
            created_time=row["created_time"],
            due_time=row["due_time"],
            status=TASK_STATUS_MAP[status_str],
            allowed_resources_distance=allowed_resources_distance,
            assigned_resource=assigned_resource,
            start_time=row.get("start_time"),
            end_time=row.get("end_time"),
            remaining_time=row.get("remaining_time"),
            estimated_service_duration=int(row["estimated_service_duration"]),
            completion_outcome=TASK_OUTCOME_MAP[outcome_str],
        )
        tasks.append(task)

    return tasks

RESOURCE_TYPE_MAP = {t.resource_type_label: t for t in ResourceType}
RESOURCE_STATUS_MAP = {s.value: s for s in ResourceStatus}

def resources_df_to_objects(
    resources_df: pl.DataFrame,
    tasks_by_id: Optional[dict[int, Task]] = None,
) -> list[Resource]:
    """
    Convert a Polars DataFrame into list[Resource].
    """
    if tasks_by_id is None:
        tasks_by_id = {}

    resources: list[Resource] = []

    for row in resources_df.iter_rows(named=True):
        # support both (x, y) and (task_x, task_y)
        if "location" in row and row["location"] is not None:
            location = tuple(row["location"])
        else:
            x = row["x"] if "x" in row else row["resource_x"]
            y = row["y"] if "y" in row else row["resource_y"]
            location = (int(x), int(y))

        current_assigned_task = None
        current_assigned_task_id = row.get("current_assigned_task_id")
        if current_assigned_task_id is not None:
            current_assigned_task = tasks_by_id.get(int(current_assigned_task_id))

        status_str = row.get("status", ResourceStatus.AVAILABLE.value)

        resource = Resource(
            resource_id=int(row["resource_id"]),
            resource_type=RESOURCE_TYPE_MAP[row["resource_type"]],
            resource_des=row["resource_des"],
            location=location,
            velocity=row["velocity"],
            time_shift=row["time_shift"],
            start_shift=row["start_shift"],
            end_shift=row["end_shift"],
            status=RESOURCE_STATUS_MAP[status_str],
            current_assigned_task=current_assigned_task,
            remaining_distance=row.get("remaining_distance"),
            remaining_working_time=row.get("remaining_working_time"),  # minutes till available
            estimated_service_duration=row.get("estimated_service_duration"),
        )
        resources.append(resource)

    return resources

##### OLD IMPLEMENTATION #####
def all_tasks_completed(state: GlobalState) -> bool:
    return all(task.status == TaskStatus.COMPLETED for task in state.all_tasks)

def has_active_tasks(state: GlobalState) -> bool:
    return any(
        task.status in {TaskStatus.SCHEDULED, TaskStatus.IN_PROGRESS}
        for task in state.all_tasks
    )

def resource_effective_location(resource: Resource) -> tuple[int, int]:
    """
    If resource is busy, after finishing it will end at its current task location.
    """
    if resource.current_assigned_task is not None:
        return resource.current_assigned_task.location
    return resource.location

def resource_next_ready_time(resource: Resource, now: datetime) -> Optional[datetime]:
    """
    Earliest time the resource could begin travelling to a new task.
    """
    ready_time = now

    if resource.busy_until is not None and resource.busy_until > ready_time:
        ready_time = resource.busy_until

    if resource.start_shift > ready_time:
        ready_time = resource.start_shift

    if ready_time >= resource.end_shift:
        return None

    return ready_time

def task_has_future_feasible_resource(task: Task, state: GlobalState) -> bool:
    """
    True if there exists at least one compatible resource that can still
    complete this task within its shift horizon.
    """
    if task.status == TaskStatus.COMPLETED:
        return False

    now = state.global_time
    task_release = max(now, task.created_time)

    for resource in state.all_resources:
        if resource.resource_type not in task.task_type.allowed_resources:
            continue

        ready_time = resource_next_ready_time(resource, task_release)
        if ready_time is None:
            continue

        origin = resource_effective_location(resource)
        distance = manhattan(origin, task.location)
        travel_min = travel_minute(distance, resource.velocity)

        finish_time = ready_time + timedelta(
            minutes=travel_min + task.estimated_service_duration
        )

        if finish_time <= resource.end_shift:
            return True

    return False

def should_truncate(state: GlobalState) -> bool:
    """
    Truncate only when:
      - not all tasks are completed
      - no task is currently active
      - every unfinished task has no feasible resource shift left
    """
    if all_tasks_completed(state):
        return False

    if has_active_tasks(state):
        return False

    unfinished_tasks = [
        task for task in state.all_tasks
        if task.status != TaskStatus.COMPLETED
    ]

    return all(
        not task_has_future_feasible_resource(task, state)
        for task in unfinished_tasks
    )

def next_event_time(state: GlobalState) -> Optional[datetime]:
    """
    Next future event that can change state or feasibility:
      - future task release
      - task start
      - task completion
      - resource busy_until
      - resource future shift start
    """
    now = state.global_time
    candidates: list[datetime] = []

    for task in state.all_tasks:
        if task.status == TaskStatus.PENDING and task.created_time > now:
            candidates.append(task.created_time)

        if task.status == TaskStatus.SCHEDULED and task.start_time is not None and task.start_time > now:
            candidates.append(task.start_time)

        if task.status in {TaskStatus.SCHEDULED, TaskStatus.IN_PROGRESS} and task.end_time is not None and task.end_time > now:
            candidates.append(task.end_time)

    for resource in state.all_resources:
        if resource.busy_until is not None and resource.busy_until > now:
            candidates.append(resource.busy_until)

        if resource.status == ResourceStatus.OFF_SHIFT and resource.start_shift > now:
            candidates.append(resource.start_shift)

    return min(candidates) if candidates else None