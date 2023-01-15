import math
import numpy
from scipy.linalg import block_diag
from dataclasses import dataclass


def array_slicer(array, start, amount):
    """Slice an array given a starting index and
    amount of objects to return after the index."""

    # Derive necessary constants
    length = len(array)
    end = start + amount
    left_bound = max(min(start, length), 0)
    right_bound = min(max(end, 0), length)
    left_length = min(end, 0) - min(start, 0)
    right_length = max(end, length) - max(start, length)

    # Create array
    array = array[left_bound:right_bound]
    if start < 0:
        array = numpy.concatenate(numpy.zeros((left_length,)), array)
    if length < end:
        array = numpy.concatenate(array, numpy.zeros((left_length,)))

    return array


def clean_int_list(int_list, start, end):
    # Remove repeats in integer list
    int_list = list(set(int_list))
    # Sort integer by size
    int_list.sort(key=(lambda x: x))
    # Filter out integers that don't lie in range(start, end)
    int_list = [_ for _ in int_list if _ in range(start, end)]

    return int_list


def basic_observation(dimension):
    # Make zero vector
    observation = numpy.zeros((dimension,))
    # If dimension non-zero, set first element to one
    if dimension > 0:
        observation[0] = 1

    return observation


def harmonics_observation(amount):
    # Create the observation matrix to replicate
    vector = basic_observation(2)
    # Create list of matrix template
    vectors = amount * [vector]
    # Concatenate vectors together
    observation = numpy.concatenate(vectors)

    return observation


def basic_transition(dimension):

    return numpy.eye(dimension, dimension)


def polynomial_transition(dimension, factor=1):
    # Create scaled identity
    transition = factor * numpy.eye(dimension)
    # Create shifted diagonal matrix
    top = numpy.eye(dimension)[1:dimension]
    bottom = numpy.zeros((1, dimension))
    shifted = numpy.row_stack([top, bottom])
    # Add matricies together
    transition = transition + shifted

    return transition


def harmonic_transition(period, factor=1):
    # Create sine and cosine
    frequency = 2 * math.pi / period
    cosine = math.cos(frequency)
    sine = math.sin(frequency)
    # Create 2d rotation matrix (harmonic matrix)
    transition = factor * numpy.array(
        [
            [cosine, sine],
            [-sine, cosine],
        ]
    )

    return transition


def harmonics_transition(start, amount, factor=1):
    # Create index + 1 of last harmonic
    end = start + amount
    # Create list of harmonics
    harmonics = [harmonic_transition(period, factor) for period in range(start, end)]
    # Join harmonics into one harmonic matrix
    transition = block_diag(*harmonics)

    return transition


def autoregression_transition(dimension, data):
    # Create form free matrix
    transition = form_free_transition(dimension)
    # Inject data into form free
    transition[0] = data

    return transition


def form_free_transition(dimension, factor=1):

    return factor * numpy.roll(numpy.eye(dimension), shift=-1, axis=1)


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
    print(ComponentFactory().regression(5, numpy.array([1, 2, 3, 4, 2, 5])).observation)
