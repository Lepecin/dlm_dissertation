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


def random_nan(array: "numpy.ndarray", nans: "int") -> "numpy.ndarray":

    random_indices: "numpy.ndarray" = numpy.random.choice(
        array.size, nans, replace=False
    )

    numpy.put(array, random_indices, numpy.nan)

    return array
