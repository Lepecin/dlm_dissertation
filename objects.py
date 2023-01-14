from numpy import ndarray
from dataclasses import dataclass


@dataclass
class NormalModel:

    mean: "ndarray"
    covariance: "ndarray"


@dataclass
class TransitionModel:

    bias: "ndarray"
    weights: "ndarray"
    covariance: "ndarray"


@dataclass
class JointModel:

    normal: "NormalModel"
    transition: "TransitionModel"


@dataclass
class InvWishart:

    scale: "ndarray"
    shape: "int"
