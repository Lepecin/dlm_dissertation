from numpy import ndarray

from .objects import NormalModel, InvWishartModel
from .containers import (
    NormalContainer,
    InvWishartContainer,
    TransitionContainer,
    ModelContainer,
)


class PrimeMemoryDLM:
    def __init__(
        self,
        observed_period: "int",
        predicted_period: "int",
        observations: "ndarray",
        primordial_model: "NormalModel",
        primordial_error: "InvWishartModel",
        evolvers: "TransitionContainer",
        observers: "TransitionContainer",
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
        self.filtered_states: "NormalContainer" = ModelContainer(0, S)
        self.evolved_states: "NormalContainer" = ModelContainer(1, S)
        self.smoothed_states: "NormalContainer" = ModelContainer(0, S)
        self.predicted_states: "NormalContainer" = ModelContainer(S, S + P)

        # -- Space Models
        self.filtered_spaces: "NormalContainer" = ModelContainer(1, S)
        self.evolved_spaces: "NormalContainer" = ModelContainer(1, S)
        self.smoothed_spaces: "NormalContainer" = ModelContainer(1, S)
        self.predicted_spaces: "NormalContainer" = ModelContainer(S, S + P)

        # -- Transitions
        self.smoothers: "TransitionContainer" = ModelContainer(1, S)
        self.filterers: "TransitionContainer" = ModelContainer(1, S)

        # -- Error Matrix
        self.wisharts: "InvWishartContainer" = ModelContainer(0, S)
