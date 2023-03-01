from typing import Dict, Generator, TypeVar, Generic, List
from numpy.typing import NDArray
from scipy.stats import t as gen_t
from numpy import array

from .objects import NormalModel, InvWishartModel, TransitionModel

T = TypeVar("T")


class ModelContainer(Generic[T]):
    def __init__(self, start: "int", end: "int"):
        self.start = start
        self.end = end
        self.container: "Dict[int, T]" = dict()

    def __len__(self) -> "int":
        return self.container.__len__()

    def get_from_time(self, time: "int") -> "T":
        if not self.start <= time < self.end:
            raise BaseException(f"Set outside interval [{self.start},{self.end})")

        return self.container[time]

    def set_at_time(self, time: "int", object: "T"):
        self.container[time] = object

    def generate_container(
        self, start: "int", end: "int"
    ) -> "Generator[T, None, None]":
        for index in range(start, end):
            if not index in self.container:
                raise BaseException("Incomplete container")
            yield self.container[index]


class NormalContainer(ModelContainer[NormalModel]):
    def __init__(self, start: "int", end: "int"):
        super().__init__(start, end)

    def mean(
        self, start: "int", end: "int", feature: "int", subject: "int"
    ) -> "NDArray":
        data_list: "List[float]" = []
        for model in self.generate_container(start, end):
            value = model.mean[feature, subject]
            data_list.append(value)
        return array(data_list)

    def covariance(
        self, start: "int", end: "int", feature_x: "int", feature_y: "int"
    ) -> "NDArray":
        data_list: "List[float]" = []
        for model in self.generate_container(start, end):
            value = model.covariance[feature_x, feature_y]
            data_list.append(value)
        return array(data_list)


class InvWishartContainer(ModelContainer[InvWishartModel]):
    def __init__(self, start: "int", end: "int"):
        super().__init__(start, end)

    def scale(
        self, start: "int", end: "int", subject_x: "int", subject_y: "int"
    ) -> "NDArray":
        data_list: "List[float]" = []
        for model in self.generate_container(start, end):
            value = model.scale[subject_x, subject_y]
            data_list.append(value)
        return array(data_list)

    def shape(self, start: "int", end: "int") -> "NDArray":
        data_list: "List[float]" = []
        for model in self.generate_container(start, end):
            value = model.shape
            data_list.append(value)
        return array(data_list)

    def t_shape(
        self, start: "int", end: "int", significance_level: "float"
    ) -> "NDArray":
        data_list: "List[float]" = []
        for shape in self.shape(start, end):
            value: "float" = gen_t.ppf(1 - significance_level / 2, shape).item()
            data_list.append(value)
        return array(data_list)


class TransitionContainer(ModelContainer[TransitionModel]):
    def __init__(self, start: "int", end: "int"):
        super().__init__(start, end)


class ArrayContainer(ModelContainer[NDArray]):
    def __init__(self, start: "int", end: "int"):
        super().__init__(start, end)
