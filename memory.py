from objects import NormalModel, TransitionDensity
from typing import List
from dataclasses import dataclass, field
import numpy


@dataclass
class MemoryDLM:

    # -- Space Models

    # Create a list for storing filtered states
    filtered_states: "List[NormalModel]" = field(default_factory=list)

    # Create a list for storing evolved states
    evolved_states: "List[NormalModel]" = field(default_factory=list)

    # Create list for storing smoothed states
    smoothed_states: "List[NormalModel]" = field(default_factory=list)

    # Create list for storing predicted states
    predicted_states: "List[NormalModel]" = field(default_factory=list)

    # -- Space Models

    # Create a list for storing evolved spaces
    filtered_spaces: "List[NormalModel]" = field(default_factory=list)

    # Create a list for storing evolved spaces
    evolved_spaces: "List[NormalModel]" = field(default_factory=list)

    # Create list for storing smoothed spaces
    smoothed_spaces: "List[NormalModel]" = field(default_factory=list)

    # Create list for storing predicted spaces
    predicted_spaces: "List[NormalModel]" = field(default_factory=list)

    # -- Transitions

    # Create a list for storing smoothers
    smoothers: "List[TransitionDensity]" = field(default_factory=list)


@dataclass
class PrimeMemoryDLM:

    # -- Periods

    forward_period: "int"
    beyond_period: "int"

    # -- Models

    primordial_model: "NormalModel"

    # -- Transitions

    evolver: "TransitionDensity"
    observer: "TransitionDensity"

    # -- Data

    observations: "List[numpy.ndarray]"
