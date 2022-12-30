from objects import NormalModel
import numpy
from numpy import ndarray as Array
from typing import Tuple, List, Generator


def rand_obs(shape: "Tuple[int, int]") -> "NormalModel":

    m: "Array" = numpy.random.random(shape)
    s: "Array" = numpy.zeros(2 * (shape[0],))

    return NormalModel(m, s)


def rand_nnd(dim: "int", shape: "int") -> "Array":

    a: "Array" = numpy.random.random((dim, dim + shape + 1))

    return a.dot(a.T)


def symmetrise(array: "Array") -> "Array":

    return (array + array.T) / 2


def relister(l: "list") -> "list":

    return [l[-1 - i] for i in range(len(l))]


def model_extract(model_list: "List[NormalModel]") -> "Generator[float]":
    for model in model_list:
        yield model.mean.item()


def array_slicer(
    array: "numpy.ndarray", start: "int", amount: "int"
) -> "numpy.ndarray":

    length: "int" = len(array)

    end: "int" = start + amount

    left_bound: "int" = max(min(start, length), 0)

    right_bound: "int" = min(max(end, 0), length)

    left_length: "int" = min(end, 0) - min(start, 0)

    right_length: "int" = max(end, length) - max(start, length)

    array = array[left_bound:right_bound]

    if start < 0:

        array = numpy.concatenate([numpy.zeros(left_length), array])

    if length < end:

        array = numpy.concatenate([array, numpy.zeros(right_length)])

    return array


def random_nan(array: "numpy.ndarray", nans: "int") -> "numpy.ndarray":

    random_indices: "numpy.ndarray" = numpy.random.choice(
        array.size, nans, replace=False
    )

    numpy.put(array, random_indices, numpy.nan)

    return array
