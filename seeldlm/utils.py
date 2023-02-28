import numpy
from seeldlm import NormalModel
from numpy.typing import NDArray
from typing import Tuple


def rand_obs(shape: "Tuple[int, int]") -> "NormalModel":

    m: "NDArray" = numpy.random.random(shape)
    s: "NDArray" = numpy.zeros(2 * (shape[0],))

    return NormalModel(m, s)


def rand_nnd(dim: "int", shape: "int") -> "NDArray":

    a: "NDArray" = numpy.random.random((dim, dim + shape + 1))

    return a.dot(a.T)


def random_nan(NDArray: "NDArray", nans: "int") -> "NDArray":

    random_indices: "NDArray" = numpy.random.choice(NDArray.size, nans, replace=False)

    numpy.put(NDArray, random_indices, numpy.nan)

    return NDArray


def nan_detect(NDArray: "NDArray") -> "NDArray":

    nan_indices: "NDArray" = numpy.argwhere(~numpy.isnan(NDArray))[:, 0]

    return nan_indices
