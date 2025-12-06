from enum import Enum, auto

class SchedulerType(Enum):
    COSINE = auto()
    STEP = auto()
    MULTISTEP = auto()
    EXPONENTIAL = auto()
    PLATEAU = auto() 

class SchedulerBuilder:
    def __init__(self, scheduler_type: SchedulerType, **kwargs):
        self.scheduler_type = scheduler_type
        self.kwargs = kwargs

    def build(self, optimizer):
        factory = FACTORIES.get(self.scheduler_type)
        try:
            factory = FACTORIES[self.scheduler_type]
        except KeyError:
            raise ValueError(f"Unsupported scheduler type: {self.scheduler_type}")

        
        return factory(optimizer=optimizer, **self.kwargs)


from typing import Any, Dict, Callable
from src.training.schedulers.schedulers import cos_lr, step_lr, multi_step_lr, exp_lr, plateau_lr

FACTORIES: Dict[SchedulerType, Callable[..., Any]] = {
    SchedulerType.COSINE: cos_lr,
    SchedulerType.STEP: step_lr,
    SchedulerType.MULTISTEP: multi_step_lr,
    SchedulerType.EXPONENTIAL: exp_lr,
    SchedulerType.PLATEAU: plateau_lr,
}

