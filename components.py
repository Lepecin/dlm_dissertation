import math
import numpy

from scipy.linalg import block_diag
from abc import ABC
from typing import Tuple, List


def basic_observation_matrix(dimension: "int") -> "numpy.ndarray":

    shape: "Tuple[int]" = (1, dimension)

    observation_matrix: "numpy.ndarray" = numpy.zeros(shape)

    if dimension:
        observation_matrix[0, 0] = 1

    return observation_matrix


def compound_observation_matrix(dimension: "int", amount: "int") -> "numpy.ndarray":

    matrix_template: "numpy.ndarray" = basic_observation_matrix(dimension=dimension)

    matricies: "List[numpy.ndarray]" = amount * [matrix_template]

    observation_matrix: "numpy.ndarray" = numpy.column_stack(tup=matricies)

    return observation_matrix


def basic_transition_matrix(dimension: "int") -> "numpy.ndarray":

    shape: "Tuple[int]" = (dimension, dimension)

    transition_matrix: "numpy.ndarray" = numpy.zeros(shape)

    return transition_matrix


def form_free_transition_matrix(dimension: "int", factor: "float") -> "numpy.ndarray":

    transition_matrix: "numpy.ndarray" = factor * numpy.eye(dimension)

    top_matrix: "numpy.ndarray" = transition_matrix[1:dimension]

    bottom_matrix: "numpy.ndarray" = transition_matrix[0:1]

    matricies: "List[numpy.ndarray]" = [top_matrix, bottom_matrix]

    transition_matrix = numpy.row_stack(matricies)

    return transition_matrix


def polynomial_transition_matrix(dimension: "int", factor: "float") -> "numpy.ndarray":

    transition_matrix: "numpy.ndarray" = factor * numpy.eye(dimension)

    top_matrix: "numpy.ndarray" = numpy.eye(dimension)[1:dimension]

    shape: "Tuple[int]" = (1, dimension)

    bottom_matrix: "numpy.ndarray" = numpy.zeros(shape)

    matricies: "List[numpy.ndarray]" = [top_matrix, bottom_matrix]

    transition_matrix = transition_matrix + numpy.row_stack(matricies)

    return transition_matrix


def harmonic_transition_matrix(period: "int", factor: "float") -> "numpy.ndarray":

    frequency: "float" = 2 * math.pi / period

    cosine: "float" = math.cos(frequency)

    sine: "float" = math.sin(frequency)

    transition_matrix: "numpy.ndarray" = factor * numpy.array(
        [
            [cosine, sine],
            [-sine, cosine],
        ]
    )

    return transition_matrix


def harmonics_transition_matrix(
    start: "int", amount: "int", factor: "float"
) -> "numpy.ndarray":

    end: "int" = start + amount

    matricies: "List[numpy.ndarray]" = [
        harmonic_transition_matrix(_, factor) for _ in range(start, end)
    ]

    transition_matrix: "numpy.ndarray" = block_diag(*matricies)

    return transition_matrix


class ModelComponent(ABC):

    dimension: "int"
    factor: "float"
    start: "int"
    amount: "int"
    data: "numpy.ndarray"

    def __init__(self: "ModelComponent") -> "None":
        pass

    def generate_transition(self: "ModelComponent") -> "numpy.ndarray":
        pass

    def generate_observation(self: "ModelComponent") -> "numpy.ndarray":
        pass


class BasicComponent(ModelComponent):
    def generate_transition(self: "ModelComponent") -> "numpy.ndarray":
        return basic_transition_matrix(dimension=0)

    def generate_observation(self: "ModelComponent") -> "numpy.ndarray":
        return basic_observation_matrix(dimension=0)


class FormFreeComponent(ModelComponent):
    def __init__(self: "ModelComponent", dimension: "int", factor: "float") -> "None":
        self.dimension = dimension
        self.factor = factor

    def generate_transition(self: "ModelComponent") -> "numpy.ndarray":
        return form_free_transition_matrix(dimension=self.dimension, factor=self.factor)

    def generate_observation(self: "ModelComponent") -> "numpy.ndarray":
        return basic_observation_matrix(dimension=self.dimension)


class PolynomialComponent(ModelComponent):
    def __init__(self: "ModelComponent", dimension: "int", factor: "float") -> "None":
        self.dimension = dimension
        self.factor = factor

    def generate_transition(self: "ModelComponent") -> "numpy.ndarray":
        return polynomial_transition_matrix(
            dimension=self.dimension, factor=self.factor
        )

    def generate_observation(self: "ModelComponent") -> "numpy.ndarray":
        return basic_observation_matrix(dimension=self.dimension)


class HarmonicsComponent(ModelComponent):
    def __init__(
        self: "ModelComponent", start: "int", amount: "int", factor: "float"
    ) -> "None":
        self.start = start
        self.amount = amount
        self.factor = factor

    def generate_transition(self: "ModelComponent") -> "numpy.ndarray":
        return harmonics_transition_matrix(
            start=self.start, amount=self.amount, factor=self.factor
        )

    def generate_observation(self: "ModelComponent") -> "numpy.ndarray":
        return compound_observation_matrix(dimension=2, amount=self.amount)


class RegressionComponent(ModelComponent):
    def __init__(
        self: "ModelComponent", dimension: "int", data: "numpy.ndarray"
    ) -> "None":
        self.data = data
        self.dimension = dimension

    pass


class AutoRegressionComponent(ModelComponent):
    def __init__(self: "ModelComponent") -> "None":
        pass
