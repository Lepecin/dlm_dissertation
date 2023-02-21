from numpy import ndarray
from typing import Dict, Generator, TypeVar, Generic, List

from .objects import NormalModel, InvWishartModel, TransitionModel


T = TypeVar("T")


class ModelContainer(Generic[T]):
    def __init__(self, start: "int", end: "int"):

        self.container: "Dict[int, T]" = dict()
        self.start = start
        self.end = end

    def __len__(self) -> "int":
        return self.container.__len__()

    def get_from_time(self, time: "int") -> "T":
        return self.container[time - self.start]

    def set_at_time(self, time: "int", object: "T"):
        self.container[time - self.start] = object

    def generate_container(self) -> "Generator[T]":
        for index in range(self.end - self.start + 1):
            if not index in self.container:
                raise BaseException("Incomplete container")
            yield self.container[index]

    def list_container(self) -> "List[T]":
        return list(self.generate_container())


class PrimeMemoryDLM:
    def __init__(
        self,
        observed_period: "int",
        predicted_period: "int",
        observations: "ndarray",
        primordial_model: "NormalModel",
        primordial_error: "InvWishartModel",
        evolvers: "ModelContainer[TransitionModel]",
        observers: "ModelContainer[TransitionModel]",
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
        self.filtered_states: "ModelContainer[NormalModel]" = ModelContainer(0, S)
        self.evolved_states: "ModelContainer[NormalModel]" = ModelContainer(1, S)
        self.smoothed_states: "ModelContainer[NormalModel]" = ModelContainer(0, S)
        self.predicted_states: "ModelContainer[NormalModel]" = ModelContainer(S, S + P)

        # -- Space Models
        self.filtered_spaces: "ModelContainer[NormalModel]" = ModelContainer(1, S)
        self.evolved_spaces: "ModelContainer[NormalModel]" = ModelContainer(1, S)
        self.smoothed_spaces: "ModelContainer[NormalModel]" = ModelContainer(1, S)
        self.predicted_spaces: "ModelContainer[NormalModel]" = ModelContainer(S, S + P)

        # -- Transitions
        self.smoothers: "ModelContainer[TransitionModel]" = ModelContainer(1, S)
        self.filterers: "ModelContainer[TransitionModel]" = ModelContainer(1, S)

        # -- Error Matrix
        self.wisharts: "ModelContainer[InvWishartModel]" = ModelContainer(0, S)
