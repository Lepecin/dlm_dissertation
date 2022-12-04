from objects import NormalModel, TransitionDensity, JointModel
from typing import Union, Tuple
from typing import Tuple, Union
import numpy
from utils import symmetrise


def transmutator(
    model: "NormalModel", transition: "TransitionDensity", model_only: "bool" = False
) -> "Union[Tuple[NormalModel, TransitionDensity], NormalModel]":

    # Extract parameters of model
    mean_1 = model.mean

    covariance_1 = model.covariance

    # Extract parameters of transition
    weights_1 = transition.weights

    bias_1 = transition.bias

    variation_1 = transition.covariance

    # Create parameters of new model
    mean_2: "numpy.ndarray" = bias_1 + weights_1.dot(mean_1)

    covariance_2: "numpy.ndarray" = variation_1 + weights_1.dot(covariance_1).dot(
        weights_1.T
    )

    # Create new model and prepare it for output
    output = NormalModel(mean_2, covariance_2)

    # If output is not only model
    if not model_only:

        # Create parameters of new transition
        weights_2: "numpy.ndarray" = covariance_1.dot(weights_1.T).dot(
            numpy.linalg.inv(symmetrise(covariance_2))
        )

        bias_2: "numpy.ndarray" = mean_1 - weights_2.dot(mean_2)

        variation_2: "numpy.ndarray" = covariance_1 - weights_2.dot(covariance_2).dot(
            weights_2.T
        )

        # Create new transition and prepare it for output with new model
        output = (output, TransitionDensity(bias_2, weights_2, variation_2))

    return output


def observator(
    observation: "numpy.ndarray", transition: "TransitionDensity"
) -> "NormalModel":

    # Extract models of transition
    bias = transition.bias

    weights = transition.weights

    # Create new mean from observation
    mean = bias + weights.dot(observation)

    return NormalModel(mean=mean, covariance=transition.covariance)


def carpenter(model: "NormalModel", transition: "TransitionDensity") -> "JointModel":

    s1 = model.covariance
    a1 = transition.weights
    derived = transmutator(model, transition, True)

    return JointModel(model, derived, a1.dot(s1))


# def optimiser(joint: "JointModel", invwishart: "InvWishart") -> "Array":

#     m_y = joint.derived.mean
#     m_x = joint.basis.mean
