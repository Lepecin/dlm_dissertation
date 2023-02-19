import math
import numpy

from scipy.linalg import block_diag
from typing import List, Dict, Tuple


def array_slicer(
    array: "numpy.ndarray", start: "int", amount: "int"
) -> "numpy.ndarray":

    output = numpy.zeros((amount,))

    length = len(array)
    end = start + amount
    left_bound = max(min(start, length), 0)
    right_bound = min(max(end, 0), length)
    left_slice = min(end, 0) - min(start, 0)
    right_slice = left_slice + right_bound - left_bound

    output[left_slice:right_slice] = array[left_bound:right_bound]

    return output


def clean_int_list(int_list: "List[int]", start: "int", end: "int") -> "List[int]":

    int_list = list(set(int_list))
    int_list.sort(key=(lambda x: x))
    int_list = [index for index in int_list if start <= index < end]

    return int_list


class ObservationFactory:
    def __init__(self):
        pass

    def basic(self, dimension: "int") -> "numpy.ndarray":

        observation = numpy.zeros((dimension,))
        if dimension > 0:
            observation[0] = 1

        return observation

    def harmonics(self, amount: "int") -> "numpy.ndarray":

        vector = self.basic(2)
        vectors = amount * [vector]
        observation = numpy.concatenate(vectors)

        return observation


class TransitionFactory:
    def __init__(self):
        pass

    def basic(self, dimension: "int", factor: "float" = 1) -> "numpy.ndarray":

        transition = factor * numpy.eye(dimension, dimension)

        return transition

    def polynomial(self, dimension: "int", factor: "float" = 1) -> "numpy.ndarray":

        transition = self.basic(dimension, factor)

        shifted = self.form_free(dimension)
        if dimension > 0:
            shifted[dimension - 1, 0] = 0

        transition += shifted

        return transition

    def harmonic(self, period: "int", factor: "float" = 1) -> "numpy.ndarray":

        frequency = 2 * math.pi / period
        cosine = math.cos(frequency)
        sine = math.sin(frequency)

        transition = factor * numpy.array(
            [
                [cosine, sine],
                [-sine, cosine],
            ]
        )

        return transition

    def harmonics(
        self, start: "int", amount: "int", factor: "float" = 1
    ) -> "numpy.ndarray":

        end = start + amount

        harmonics = [self.harmonic(period, factor) for period in range(start, end)]

        transition = block_diag(*harmonics)

        return transition

    def autoregression(
        self, dimension: "int", data: "numpy.ndarray"
    ) -> "numpy.ndarray":

        transition = self.form_free(dimension)

        transition[0] = data

        return transition

    def form_free(self, dimension: "int", factor: "float" = 1) -> "numpy.ndarray":

        transition = numpy.roll(self.basic(dimension, factor), shift=-1, axis=1)

        return transition


class ModelComponent:
    def __init__(
        self,
        dimension: "int",
        transition: "numpy.ndarray",
        observation: "numpy.ndarray",
    ):
        self.dimension = dimension
        self.transition = transition
        self.observation = observation

    def covariate(self, dimension: "int", indices: "List[int]" = []) -> "numpy.ndarray":

        template = numpy.zeros((dimension, self.dimension))

        if len(indices) > 0:
            indices = clean_int_list(indices, 0, dimension)
            template[
                indices,
            ] = self.observation

        return template


class ComponentFactory:
    def __init__(self):

        self.transition_factory = TransitionFactory()
        self.observation_factory = ObservationFactory()

    def form_free(self, dimension: "int", factor: "float" = 1) -> "ModelComponent":
        return ModelComponent(
            dimension,
            self.transition_factory.form_free(dimension, factor),
            self.observation_factory.basic(dimension),
        )

    def polynomial(self, dimension: "int", factor: "float" = 1) -> "ModelComponent":
        return ModelComponent(
            dimension,
            self.transition_factory.polynomial(dimension, factor),
            self.observation_factory.basic(dimension),
        )

    def harmonics(
        self, start: "int", amount: "int", factor: "float" = 1
    ) -> "ModelComponent":
        return ModelComponent(
            2 * amount,
            self.transition_factory.harmonics(start, amount, factor),
            self.observation_factory.harmonics(amount),
        )

    def regression(
        self, start: "int", amount: "int", data: "numpy.ndarray"
    ) -> "ModelComponent":
        return ModelComponent(
            amount,
            self.transition_factory.basic(amount),
            array_slicer(data, start, amount),
        )

    def autoregression(
        self, dimension: "int", data: "numpy.ndarray"
    ) -> "ModelComponent":
        return ModelComponent(
            dimension,
            self.transition_factory.autoregression(dimension, data),
            self.observation_factory.basic(dimension),
        )


class ModelCompiler:
    def __init__(
        self,
        root_dimension: "int",
    ):
        self.M = root_dimension
        self.components: "List[ModelComponent]" = list()
        self.vertices: "Dict[Tuple[int, int], List[int]]" = dict()

    def add_component_sequence(self, *args: "ModelComponent"):
        self.components = list(args)

    def set_vertex(self, x: "int", y: "int", indices: "List[int]"):
        if x == y:
            raise BaseException(f"x={x} and y={y} have to differ")
        self.vertices[(x, y)] = indices

    def transition_block(self, x: "int", y: "int") -> "numpy.ndarray":
        if x == y:
            return self.components[y].transition
        else:
            if not (x, y) in self.vertices:
                indices = []
            else:
                indices = self.vertices[(x, y)]
            return self.components[y].covariate(self.components[x].dimension, indices)

    def observation_block(self, y: "int") -> "numpy.ndarray":

        if not (-1, y) in self.vertices:
            indices = []
        else:
            indices = self.vertices[(-1, y)]

        return self.components[y].covariate(self.M, indices)

    def compile_transition(self) -> "numpy.ndarray":

        amount = len(self.components)

        block_matrix: "List[List[numpy.ndarray]]" = list()
        for x in range(amount):
            horizontal_block = [self.transition_block(x, y) for y in range(amount)]
            block_matrix.append(horizontal_block)

        return numpy.block(block_matrix)

    def compile_observation(self) -> "numpy.ndarray":

        amount = len(self.components)

        block_matrix: "List[numpy.ndarray]"
        block_matrix = [self.observation_block(y) for y in range(amount)]

        return numpy.block(block_matrix)
