from numpy import ndarray
from typing import List, Any, Tuple
from objects import NormalModel, TransitionModel, JointModel, InvWishartModel


class ModelContainer:
    def __init__(self, start: "int", end: "int"):

        self.container: "List[Any]" = max(1, end - start + 1) * [None]
        self.start = start

    def get_from_time(self, time: "int") -> "Any":
        return self.container[time - self.start]

    def set_at_time(self, time: "int", object: "Any"):
        self.container[time - self.start] = object


class PrimeMemoryDLM:
    def __init__(
        self,
        observed_period: "int",
        predicted_period: "int",
        observations: "ndarray",
        primordial_model: "NormalModel",
        primordial_error: "InvWishartModel",
        evolvers: "ModelContainer",
        observers: "ModelContainer",
    ):

        self.S = observed_period
        self.P = predicted_period

        # -- Data
        self.observations = observations

        # -- Models
        self.primordial_model = primordial_model
        self.primordial_error = primordial_error

        # -- Transitions
        self.evolvers = evolvers
        self.observers = observers


class MemoryDLM:
    def __init__(self, observed_period: "int", predicted_period: "int"):

        S = observed_period
        P = predicted_period

        self.S = S
        self.P = P

        # -- Space Models
        self.filtered_states = ModelContainer(0, S)
        self.evolved_states = ModelContainer(1, S)
        self.smoothed_states = ModelContainer(0, S)
        self.predicted_states = ModelContainer(S, S + P)

        # -- Space Models
        self.filtered_spaces = ModelContainer(1, S)
        self.evolved_spaces = ModelContainer(1, S)
        self.smoothed_spaces = ModelContainer(1, S)
        self.predicted_spaces = ModelContainer(S, S + P)

        # -- Transitions
        self.smoothers = ModelContainer(1, S)
        self.filterers = ModelContainer(1, S)

        # -- Error Matrix
        self.wisharts = ModelContainer(0, S)
