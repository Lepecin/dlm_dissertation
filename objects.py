from numpy import ndarray as Array
from dataclasses import dataclass


@dataclass
class NormalModel:

    mean: "Array"
    covariance: "Array"


@dataclass
class TransitionDensity:

    bias: "Array"
    weights: "Array"
    covariance: "Array"


@dataclass
class JointModel:

    basis: "NormalModel"
    derived: "NormalModel"
    covariance: "Array"


@dataclass
class InvWishart:

    scale: "Array"
    shape: "int"
