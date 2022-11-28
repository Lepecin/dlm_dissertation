import math
from scipy.linalg import block_diag
import numpy
from numpy import ndarray as Array
from dataclasses import dataclass


@dataclass
class DLModelComponent:

    obs_matrix: "Array" = numpy.zeros((1, 0))
    trans_matrix: "Array" = numpy.zeros((0, 0))

    def __add__(
        self: "DLModelComponent", other: "DLModelComponent"
    ) -> "DLModelComponent":

        trans = block_diag(self.trans_matrix, other.trans_matrix)

        obs = numpy.column_stack([self.obs_matrix, other.obs_matrix])

        return DLModelComponent(obs, trans)


def gen_fullform(k: "int", l: "float") -> "DLModelComponent":

    trans = numpy.eye(k)
    trans = l * numpy.row_stack((trans[1:k], trans[0:1]))

    obs = numpy.zeros((1, k))
    obs[0, 0] = 1

    return DLModelComponent(obs, trans)


def gen_jordan(k: "int", l: "float") -> "DLModelComponent":

    trans = l * numpy.eye(k)
    trans = trans + numpy.row_stack((numpy.eye(k)[1:k], numpy.zeros((1, k))))

    obs = numpy.zeros((1, k))
    obs[0, 0] = 1

    return DLModelComponent(obs, trans)


def gen_harmonic(t: "float", l: "float") -> "DLModelComponent":

    trans = l * numpy.array(
        [
            [math.cos(2 * math.pi / t), math.sin(2 * math.pi / t)],
            [-math.sin(2 * math.pi / t), math.cos(2 * math.pi / t)],
        ]
    )

    obs = numpy.zeros((1, 2))
    obs[0, 0] = 1

    return DLModelComponent(obs, trans)


def gen_harmonics(start: "int", end: "int", l: "float") -> "DLModelComponent":

    model = DLModelComponent()

    for t in range(start, end):
        model += gen_harmonic(t, l)

    return model
