import math
import numpy
from scipy.linalg import block_diag
from dataclasses import dataclass
from typing import List


def array_slicer(
    array: "numpy.ndarray", start: "int", amount: "int"
) -> "numpy.ndarray":
    """Slice an array given a starting index and
    amount of objects to return after the index."""

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

        vector = self.basic_observation(2)
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
            shifted[dimension, 0] = 0

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


@dataclass
class ModelComponent:

    dimension: "int"
    transition: "numpy.ndarray"
    observation: "numpy.ndarray"

    def covariate(self, dimension, indices=[]):
        # Create template matrix onto which obs vector is grafted
        template = numpy.zeros((dimension, self.dimension))
        if len(indices) > 0:
            # Clean list of indices
            indices = clean_int_list(indices, 0, dimension)
            # Graft obs vector to template
            template[
                indices,
            ] = self.observation

        return template


class ComponentFactory:
    def root(self, dimension):
        transition = basic_transition(0)
        observation = basic_observation(0)

        return ModelComponent(dimension, transition, observation)

    def form_free(self, dimension, factor=1):
        transition = form_free_transition(dimension, factor)
        observation = basic_observation(dimension)

        return ModelComponent(dimension, transition, observation)

    def polynomial(self, dimension, factor=1):
        transition = polynomial_transition(dimension, factor)
        observation = basic_observation(dimension)

        return ModelComponent(dimension, transition, observation)

    def harmonics(self, start, amount, factor=1):
        transition = harmonics_transition(start, amount, factor)
        observation = harmonics_observation(amount)

        return ModelComponent(2 * amount, transition, observation)

    def regression(self, dimension, data):
        transition = basic_transition(dimension)
        observation = array_slicer(data, 0, dimension)

        return ModelComponent(dimension, transition, observation)

    def autoregression(self, dimension, data):
        transition = autoregression_transition(dimension, data)
        observation = basic_observation(dimension)

        return ModelComponent(dimension, transition, observation)


if __name__ == "__main__":
    print(numpy.roll(numpy.eye(5), 1, 1))
