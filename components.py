import math
import numpy

from scipy.linalg import block_diag
from typing import Tuple, List
from dataclasses import dataclass

from utils import array_slicer


def basic_observation_matrix(dimension: "int") -> "numpy.ndarray":

    observation_matrix: "numpy.ndarray" = numpy.zeros((dimension,))

    if dimension:
        observation_matrix[0] = 1

    return observation_matrix


def compound_observation_matrix(dimension: "int", amount: "int") -> "numpy.ndarray":

    matrix_template: "numpy.ndarray" = basic_observation_matrix(dimension=dimension)

    matricies: "List[numpy.ndarray]" = amount * [matrix_template]

    observation_matrix: "numpy.ndarray" = numpy.concatenate(matricies)

    return observation_matrix


def basic_transition_matrix(dimension: "int") -> "numpy.ndarray":

    shape: "Tuple[int]" = (dimension, dimension)

    transition_matrix: "numpy.ndarray" = numpy.eye(*shape)

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


def autoregression_transition_matrix(
    dimension: "int", data: "numpy.ndarray"
) -> "numpy.ndarray":

    data: "numpy.ndarray" = numpy.expand_dims(data, axis=0)

    identity_slice: "numpy.ndarray" = numpy.eye(dimension - 1, dimension)

    transition_matrix: "numpy.ndarray" = numpy.row_stack([data, identity_slice])

    return transition_matrix


def form_free_transition_matrix(dimension: "int", factor: "float") -> "numpy.ndarray":

    data: "numpy.ndarray" = numpy.zeros((dimension,))

    if dimension:
        data[-1] = 1

    transition_matrix: "numpy.ndarray" = factor * autoregression_transition_matrix(
        dimension=dimension, data=data
    )

    return transition_matrix


@dataclass
class ModelComponent:

    dimension: "int"
    transition_matrix: "numpy.ndarray"
    observation_matrix: "numpy.ndarray"


class ComponentFactory:
    @staticmethod
    def create_root(dimension: "int") -> "ModelComponent":

        transition_matrix: "numpy.ndarray" = basic_transition_matrix(dimension=0)

        observation_matrix: "numpy.ndarray" = basic_observation_matrix(dimension=0)

        return ModelComponent(dimension, transition_matrix, observation_matrix)

    @staticmethod
    def create_form_free(dimension: "int", factor: "float" = 1) -> "ModelComponent":

        transition_matrix: "numpy.ndarray" = form_free_transition_matrix(
            dimension=dimension, factor=factor
        )

        observation_matrix: "numpy.ndarray" = basic_observation_matrix(
            dimension=dimension
        )

        return ModelComponent(dimension, transition_matrix, observation_matrix)

    @staticmethod
    def create_polynomial(dimension: "int", factor: "float" = 1) -> "ModelComponent":

        transition_matrix: "numpy.ndarray" = polynomial_transition_matrix(
            dimension=dimension, factor=factor
        )

        observation_matrix: "numpy.ndarray" = basic_observation_matrix(
            dimension=dimension
        )

        return ModelComponent(dimension, transition_matrix, observation_matrix)

    @staticmethod
    def create_harmonics(
        start: "int", amount: "int", factor: "float" = 1
    ) -> "ModelComponent":

        transition_matrix: "numpy.ndarray" = harmonics_transition_matrix(
            start=start, amount=amount, factor=factor
        )

        observation_matrix: "numpy.ndarray" = compound_observation_matrix(
            dimension=2, amount=amount
        )

        return ModelComponent(2 * amount, transition_matrix, observation_matrix)

    @staticmethod
    def create_regression(
        dimension: "int", data: "numpy.ndarray", start: "int"
    ) -> "ModelComponent":

        transition_matrix: "numpy.ndarray" = basic_transition_matrix(
            dimension=dimension
        )

        observation_matrix: "numpy.ndarray" = array_slicer(
            array=data, start=start, amount=dimension
        )

        return ModelComponent(dimension, transition_matrix, observation_matrix)

    @staticmethod
    def create_autoregression(
        dimension: "int", data: "numpy.ndarray"
    ) -> "ModelComponent":

        transition_matrix: "numpy.ndarray" = autoregression_transition_matrix(
            dimension=dimension, data=data
        )

        observation_matrix: "numpy.ndarray" = basic_observation_matrix(
            dimension=dimension
        )

        return ModelComponent(dimension, transition_matrix, observation_matrix)
