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
        return self.container[time]

    def set_at_time(self, time: "int", object: "T"):
        self.container[time] = object

    def generate_container(self) -> "Generator[T]":
        for index in range(self.start, self.end + 1):
            if not index in self.container:
                raise BaseException("Incomplete container")
            yield self.container[index]

    def list_container(self) -> "List[T]":
        return list(self.generate_container())


class NormalContainer(ModelContainer[NormalModel]):
    def __init__(self, start: "int", end: "int"):
        super().__init__(start, end)

    def mean(self, feature: "int", subject: "int") -> "Generator[float]":
        for model in self.generate_container():
            value = model.mean[feature, subject]
            yield value

    def covariance(self, feature_x: "int", feature_y: "int") -> "Generator[float]":
        for model in self.generate_container():
            value = model.covariance[feature_x, feature_y]
            yield value


class InvWishartContainer(ModelContainer[InvWishartModel]):
    def __init__(self, start: "int", end: "int"):
        super().__init__(start, end)

    def scale(self, subject_x: "int", subject_y: "int") -> "Generator[float]":
        for model in self.generate_container():
            value = model.scale[subject_x, subject_y]
            yield value

    def shape(self) -> "Generator[int]":
        for model in self.generate_container():
            value = model.shape
            yield value


class TransitionContainer(ModelContainer[TransitionModel]):
    def __init__(self, start: "int", end: "int"):
        super().__init__(start, end)
