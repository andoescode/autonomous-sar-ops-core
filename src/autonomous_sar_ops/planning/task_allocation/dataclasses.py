from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Optional

class TaskStatus(str, Enum):
    PENDING = "pending"
    SCHEDULED = "scheduled"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    CANCELLED = "cancelled"


class ResourceStatus(str, Enum):
    AVAILABLE = "available"
    BUSY = "busy"
    OFF_SHIFT = "off_shift"


class TaskCompletionOutcome(str, Enum):
    COMPLETED_ON_TIME = "completed_on_time"
    COMPLETED_LATE = "completed_late"
    CANCELLED = "cancelled"
    UNKNOWN = "unknown"


class ResourceType(Enum):
    R1 = (1, "R1")
    R2 = (2, "R2")
    R3 = (3, "R3")

    def __init__(self, resource_type_id: int, resource_type_label: str):
        self.resource_type_id = resource_type_id
        self.resource_type_label = resource_type_label


class TaskType(Enum):
    """
    5 task types, each with allowed resource types.
    Use frozenset so the Enum value stays hashable.
    """
    T1 = (1, "T1", frozenset({ResourceType.R2, ResourceType.R3}))
    T2 = (2, "T2", frozenset({ResourceType.R1}))
    T3 = (3, "T3", frozenset({ResourceType.R3}))
    T4 = (4, "T4", frozenset({ResourceType.R1, ResourceType.R3}))
    T5 = (5, "T5", frozenset({ResourceType.R2}))

    def __init__(
        self,
        task_type_id: int,
        task_type_label: str,
        allowed_resources: frozenset[ResourceType],
    ):
        self.task_type_id = task_type_id
        self.task_type_label = task_type_label
        self.allowed_resources = allowed_resources


class SLATier(Enum):
    """
    weight is the business priority weight
    response_time_range is in minutes
    """
    TIER_1 = (1, "tier1", 3.0, (120, 300))          # 2-5 hours
    TIER_2 = (2, "tier2", 2.0, (300, 720))          # 6-12 hours
    TIER_3 = (3, "tier3", 1.0, (720, 1440))         # 12-24 hours
    TIER_4 = (4, "tier4", 0.5, (1440, float("inf")))  # 24+ hours

    def __init__(
        self,
        tier_id: int,
        label: str,
        weight: float,
        response_time_range: tuple[float, float],
    ):
        self.tier_id = tier_id
        self.label = label
        self.weight = weight
        self.response_time_range = response_time_range

@dataclass
class Task:
    task_id: int
    task_type: TaskType
    task_des: str
    location: tuple[int, int]
    priority_lvl: SLATier
    created_time: datetime
    due_time: datetime

    status: TaskStatus = TaskStatus.PENDING
    allowed_resources_distance: dict[int, int] = field(default_factory=dict)

    assigned_resource: Optional["Resource"] = None

    dispatch_time: Optional[datetime] = None # when resource travel
    start_time: Optional[datetime] = None # when service start
    end_time: Optional[datetime] = None # when task is done
    remaining_time: Optional[int] = None  # minutes
    estimated_service_duration: int = 0   # minutes, time needed to finish task (after arrived at task location)

    completion_outcome: TaskCompletionOutcome = TaskCompletionOutcome.UNKNOWN


@dataclass
class Resource:
    resource_id: int
    resource_type: ResourceType
    resource_des: str
    location: tuple[int, int]
    velocity: int                      # grid cells per minute
    time_shift: int                    # total shift length in minutes
    start_shift: datetime
    end_shift: datetime

    status: ResourceStatus = ResourceStatus.AVAILABLE

    current_assigned_task: Optional[Task] = None
    all_assigned_tasks: list[Task] = field(default_factory=list)
    remaining_distance: Optional[int] = None
    remaining_working_time: Optional[int] = None   # minutes till available
    estimated_service_duration: Optional[int] = None # minutes, time needed to finish current task (after arrived at task location)

    busy_until: Optional[datetime] = None

@dataclass
class GlobalState:
    global_time: datetime
    all_tasks: list[Task]
    all_resources: list[Resource]
    tasks_waiting: int = 0
    resources_available: int = 0

    def update_task_and_resource_states(self) -> None:
        """
        Update status transitions based on current global_time.

        Task:
            SCHEDULED -> IN_PROGRESS when global_time >= start_time and a resource is assigned to task
            IN_PROGRESS/SCHEDULED -> COMPLETED when global_time >= end_time

        Resource:
            BUSY -> AVAILABLE when current task has finished
        """
        for resource in self.all_resources:
            # Keep OFF_SHIFT state consistent
            if self.global_time < resource.start_shift or self.global_time >= resource.end_shift:
                if resource.status != ResourceStatus.BUSY:
                    resource.status = ResourceStatus.OFF_SHIFT
            else:
                if resource.status == ResourceStatus.OFF_SHIFT and resource.current_assigned_task is None:
                    resource.status = ResourceStatus.AVAILABLE

            if resource.current_assigned_task is None:
                if resource.status == ResourceStatus.BUSY:
                    resource.status = ResourceStatus.AVAILABLE
                    resource.busy_until = None
                    resource.remaining_distance = None
                    resource.remaining_working_time = None
                    resource.estimated_service_duration = None
                continue

            task = resource.current_assigned_task

            # Start task
            if (
                task.status == TaskStatus.SCHEDULED
                and task.start_time is not None
                and self.global_time >= task.start_time
                and (task.end_time is None or self.global_time < task.end_time)
                and task.assigned_resource is not None
            ):
                task.status = TaskStatus.IN_PROGRESS

            # Update remaining times while task is active
            if task.end_time is not None and self.global_time < task.end_time:
                task.remaining_time = max(
                    0, int((task.end_time - self.global_time).total_seconds() // 60)
                )
                resource.remaining_working_time = task.remaining_time

            # Finish task + release resource
            if task.end_time is not None and self.global_time >= task.end_time:
                task.status = TaskStatus.COMPLETED
                task.remaining_time = 0
                task.assigned_resource = resource

                if task.end_time <= task.due_time:
                    task.completion_outcome = TaskCompletionOutcome.COMPLETED_ON_TIME
                else:
                    task.completion_outcome = TaskCompletionOutcome.COMPLETED_LATE

                resource.location = task.location
                resource.current_assigned_task = None
                resource.busy_until = None
                resource.remaining_distance = None
                resource.remaining_working_time = None
                resource.estimated_service_duration = None

                if resource.start_shift <= self.global_time < resource.end_shift:
                    resource.status = ResourceStatus.AVAILABLE
                else:
                    resource.status = ResourceStatus.OFF_SHIFT

    def refresh_counts(self) -> None:
        """
            Refresh the global status of the problem.

            Update current tasks and resources states.

            Count number of resources available and task waiting to be assigned.
        """
        self.update_task_and_resource_states()

        self.tasks_waiting = sum(
            1 for task in self.all_tasks if task.status == TaskStatus.PENDING
        )
        self.resources_available = sum(
            1
            for resource in self.all_resources
            if resource.status == ResourceStatus.AVAILABLE
            and resource.current_assigned_task is None
            and resource.start_shift <= self.global_time < resource.end_shift
        )

    def advance_to(self, new_time: datetime) -> None:
        """
        Skip time forward to trigger state transitions and update counts.
        """
        if new_time < self.global_time:
            raise ValueError("new_time must be >= current global_time")
        self.global_time = new_time
        self.refresh_counts()

    # Advance time to trigger state transitions and update counts
    def advance_time(self, minutes: int) -> None:
        """
        Advance time by the specified number of minutes and update counts.
        """
        self.global_time += timedelta(minutes=minutes)
        self.refresh_counts()
