from objects import NormalModel, TransitionDensity
from typing import List
from dataclasses import dataclass, field


@dataclass
class DLModelMemory:

    # Filtered models
    f1: "List[NormalModel]" = field(default_factory=list)
    f2: "List[NormalModel]" = field(default_factory=list)

    # Evolved models
    e1: "List[NormalModel]" = field(default_factory=list)
    e2: "List[NormalModel]" = field(default_factory=list)

    # Observed models
    o2: "List[NormalModel]" = field(default_factory=list)

    # Smoothed models
    s1: "List[NormalModel]" = field(default_factory=list)
    s2: "List[NormalModel]" = field(default_factory=list)

    # Predicted models
    p1: "List[NormalModel]" = field(default_factory=list)
    p2: "List[NormalModel]" = field(default_factory=list)

    # Transitions
    smoothers: "List[TransitionDensity]" = field(default_factory=list)
    evolvers: "List[TransitionDensity]" = field(default_factory=list)
    predictors: "List[TransitionDensity]" = field(default_factory=list)
    filters: "List[TransitionDensity]" = field(default_factory=list)


@dataclass
class DLModelPrimeMemory:

    # Dimensional parameters
    n: "int"
    m: "int"
    p: "int"

    # Model's periods
    period: "int"
    beyond_period: "int"

    n0: "NormalModel"
    te: "TransitionDensity"
    tp: "TransitionDensity"
