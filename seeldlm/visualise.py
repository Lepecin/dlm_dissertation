# import numpy
# import matplotlib.pyplot as plt
import math

from typing import Generator, List
from scipy.stats import t_gen

from .objects import NormalModel, InvWishartModel


class Extractor:
    def __init__(
        self,
        normal_container: "List[NormalModel]",
        wishart_container: "List[InvWishartModel]",
    ):
        self.normal_container = normal_container
        self.wishart_container = wishart_container

    def extract_values(self, feature: "int", subject: "int") -> "Generator[float]":

        for model in self.normal_container:
            value = model.mean[feature, subject]
            yield value

    def extract_errors(
        self, feature: "int", subject: "int", significance_level: "int"
    ) -> "Generator[float]":

        for model, wishart in zip(self.normal_container, self.wishart_container):

            row_weight = model.covariance[feature, feature]
            column_weight = wishart.shape[subject, subject]
            shape_value = wishart.shape
            t_value = t_gen.ppf(1 - (significance_level / 2), shape_value)

            deviation = t_value * math.sqrt((row_weight * column_weight) / shape_value)

            yield deviation
