from objects import NormalModel, TransitionDensity, JointModel, InvWishart
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


def carpenter(model: "NormalModel", transition: "TransitionDensity") -> "JointModel":

    s1 = model.covariance
    a1 = transition.weights
    derived = transmutator(model, transition, True)

    return JointModel(model, derived, a1.dot(s1))


# def optimiser(joint: "JointModel", invwishart: "InvWishart") -> "Array":

#     m_y = joint.derived.mean
#     m_x = joint.basis.mean


def rand_obs(shape: "Tuple[int, int]") -> "NormalModel":

    m: "Array" = numpy.random.random(shape)
    s: "Array" = numpy.zeros(2 * (shape[0],))

    return NormalModel(m, s)


def rand_nnd(dim: "int", shape: "int") -> "Array":

    a: "Array" = numpy.random.random((dim, dim + shape + 1))

    return a.dot(a.T)


def symmetrise(array: "Array") -> "Array":

    return (array + array.T) / 2
