from objects import NormalModel, TransitionDensity
import numpy
from numpy import ndarray as Array
from typing import Tuple, Union


def transmutator(
    model: "NormalModel", transition: "TransitionDensity", modelonly: "bool" = False
) -> "Union[Tuple[NormalModel, TransitionDensity], NormalModel]":

    m1 = model.mean
    s1 = model.covariance

    a1 = transition.weights
    b1 = transition.bias
    v1 = transition.covariance

    m2: "Array" = b1 + a1.dot(m1)
    s2: "Array" = v1 + a1.dot(s1).dot(a1.T)

    if not modelonly:

        a2: "Array" = s1.dot(a1.T).dot(numpy.linalg.inv((s2 + s2.T) / 2))
        b2: "Array" = m1 - a2.dot(m2)
        v2: "Array" = s1 - a2.dot(s2).dot(a2.T)

        return (NormalModel(m2, s2), TransitionDensity(b2, a2, v2))

    return NormalModel(m2, s2)
