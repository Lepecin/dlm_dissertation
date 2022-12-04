from operators import transmutator, observator
from utils import relister
from memory import DLModelMemory, DLModelPrimeMemory
from generate import DLModelGenerator
from objects import NormalModel, TransitionDensity
from typing import List
import numpy

from dataclasses import dataclass, field


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


def new_forward(prime_memory: "PrimeMemoryDLM") -> "MemoryDLM":

    # Create memory
    memory = MemoryDLM()

    # Append the primordial state to filtered states
    memory.filtered_states.append(prime_memory.primordial_model)

    # For all times over the forward period
    for time in range(prime_memory.forward_period):

        # -- Evolution

        # Pick the filtered state model of the given time
        filtered_state = memory.filtered_states[time]

        # Create evolved state and smoother
        evolved_state, smoother = transmutator(
            model=filtered_state, transition=prime_memory.evolver
        )

        # Append the smoother to list of smoothers
        memory.smoothers.append(smoother)

        # Append the evolved states to list of evolved states
        memory.evolved_states.append(evolved_state)

        # -- Evolved Observation

        # Create evolved space and filterer
        evolved_space, filterer = transmutator(
            model=evolved_state, transition=prime_memory.observer
        )

        # Append evolved space to list of evolved spaces
        memory.evolved_spaces.append(evolved_space)

        # -- Filration

        # Extract current observation
        observation = prime_memory.observations[time]

        # Create filtered state
        filtered_state = observator(observation=observation, transition=filterer)

        # Append filtered state to list of filtered states
        memory.filtered_states.append(filtered_state)

        # -- Filtered Observation

        # Create filtered space
        filtered_space = transmutator(
            model=filtered_state, transition=prime_memory.observer, model_only=True
        )

        # Append filtered space to list of filtered spaces
        memory.filtered_spaces.append(filtered_space)

    return memory


def new_backward(
    prime_memory: "PrimeMemoryDLM",
    memory: "MemoryDLM",
) -> "MemoryDLM":

    # Extract final filtered state
    filtered_state = memory.filtered_states[prime_memory.forward_period]

    # Append last filtered state to list of smoothed states
    memory.smoothed_states.append(filtered_state)

    # For all times over the forward period
    for time in range(prime_memory.forward_period):

        # -- Smoothed Observation

        # Extract current smoothed state
        smoothed_state = memory.smoothed_states[time]

        # Create smoothed space
        smoothed_space = transmutator(
            model=smoothed_state, transition=prime_memory.observer, model_only=True
        )

        # Append smoothed space to list of smoothed spaces
        memory.smoothed_spaces.append(smoothed_space)

        # -- Smoothing

        # Extract current smoother
        smoother = memory.smoothers[-1 - time]

        # Create smoothed state
        smoothed_state = transmutator(
            model=smoothed_state, transition=smoother, model_only=True
        )

        # Append smoothed state to list of smoothed states
        memory.smoothed_states.append(smoothed_state)

    memory.smoothed_states = relister(memory.smoothed_states)

    memory.smoothed_spaces = relister(memory.smoothed_spaces)

    return memory


def new_beyond(
    prime_memory: "PrimeMemoryDLM",
    memory: "MemoryDLM",
):

    # Extract final filtered state
    filtered_state = memory.filtered_states[prime_memory.forward_period]

    memory.predicted_states.append(filtered_state)

    # Extract final filtered space
    filtered_space = memory.filtered_spaces[prime_memory.forward_period - 1]

    memory.predicted_spaces.append(filtered_space)

    for time in range(prime_memory.beyond_period):

        # -- Prediction

        predicted_state = memory.predicted_states[time]

        predicted_state = transmutator(
            model=predicted_state, transition=prime_memory.evolver, model_only=True
        )

        memory.predicted_states.append(predicted_state)

        # -- Predicted Observation

        predicted_space = transmutator(
            model=predicted_state, transition=prime_memory.observer, model_only=True
        )

        memory.predicted_spaces.append(predicted_space)

    return memory


def forward(
    prime: "DLModelPrimeMemory", memory: "DLModelMemory", gen: "DLModelGenerator"
) -> "DLModelMemory":

    # Primordial Initiation
    n0 = gen.gen_primordial(prime, memory)
    memory.f1.append(n0)

    # Forward pass
    for i in range(prime.period):

        # Evolution
        nf1 = memory.f1[i]
        te = gen.gen_evolver(prime, memory, i)
        memory.evolvers.append(te)
        ne1, ts = transmutator(nf1, te)
        memory.e1.append(ne1)
        memory.smoothers.append(ts)

        # Evolved Prediction
        tp = gen.gen_predictor(prime, memory, i)
        memory.predictors.append(tp)
        ne2, tf = transmutator(ne1, tp)
        memory.e2.append(ne2)

        # Filtration
        no2 = gen.gen_observation(prime, memory, i)
        memory.o2.append(no2)
        nf1 = transmutator(no2, tf, True)
        memory.f1.append(nf1)

        # Filtered Prediction
        nf2 = transmutator(nf1, tp, True)
        memory.f2.append(nf2)

    return memory


def backward(prime: "DLModelPrimeMemory", memory: "DLModelMemory") -> "DLModelMemory":

    # Smoothing Initiation
    ns0 = memory.f1[prime.period]
    memory.s1.append(ns0)

    # Backward pass
    for i in range(prime.period):

        # Smoothing
        ns1 = memory.s1[i]
        ts = memory.smoothers[-1 - i]
        tp = memory.predictors[-1 - i]
        ns2 = transmutator(ns1, tp, True)
        memory.s2.append(ns2)

        # Smoothed Prediction
        ns1 = transmutator(ns1, ts, True)
        memory.s1.append(ns1)

    # Reverse list orders
    memory.s1 = relister(memory.s1)
    memory.s2 = relister(memory.s2)

    return memory


def beyond(
    prime: "DLModelPrimeMemory", memory: "DLModelMemory", gen: "DLModelGenerator"
) -> "DLModelMemory":

    # State Prediction Intiation
    nf0 = memory.f1[-1]
    memory.p1.append(nf0)

    # Space Prediction Initiation
    nf00 = memory.f2[-1]
    memory.p2.append(nf00)

    for i in range(prime.beyond_period):

        # Predictive Evolution
        np1 = memory.p1[i]
        te = gen.gen_evolver(prime, memory, i, True)
        np1 = transmutator(np1, te, True)
        memory.p1.append(np1)

        # Predictive Prediction
        tp = gen.gen_predictor(prime, memory, i, True)
        np2 = transmutator(np1, tp, True)
        memory.p2.append(np2)

    return memory
