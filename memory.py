from objects import NormalModel, TransitionModel
from typing import List
from dataclasses import dataclass, field
from numpy import ndarray


@dataclass
class MemoryDLM:

    # -- Space Models
    filtered_states: "List[NormalModel]" = field(default_factory=list)
    evolved_states: "List[NormalModel]" = field(default_factory=list)
    smoothed_states: "List[NormalModel]" = field(default_factory=list)
    predicted_states: "List[NormalModel]" = field(default_factory=list)

    # -- Space Models
    filtered_spaces: "List[NormalModel]" = field(default_factory=list)
    evolved_spaces: "List[NormalModel]" = field(default_factory=list)
    smoothed_spaces: "List[NormalModel]" = field(default_factory=list)
    predicted_spaces: "List[NormalModel]" = field(default_factory=list)

    # -- Transitions
    smoothers: "List[TransitionModel]" = field(default_factory=list)


@dataclass
class PrimeMemoryDLM:

    # -- Periods
    forward_period: "int"
    beyond_period: "int"

    # -- Models
    primordial_model: "NormalModel"

    # -- Transitions
    evolver: "TransitionModel"
    observer: "TransitionModel"

    # -- Data
    observations: "ndarray"
