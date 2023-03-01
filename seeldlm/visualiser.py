import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
import numpy
from numpy.typing import NDArray
from abc import ABC, abstractmethod


from .memory import PrimeMemoryDLM, MemoryDLM
from .containers import NormalContainer, InvWishartContainer


class VisualStategy(ABC):
    def __init__(
        self,
        observed_period: "int",
        predicted_period: "int",
    ):
        super().__init__()
        self.S = observed_period
        self.P = predicted_period

    @abstractmethod
    def create_domain(self) -> "NDArray":
        pass

    @abstractmethod
    def create_values(self) -> "NDArray":
        pass


class ObservableVisual(VisualStategy):
    def __init__(
        self,
        observed_period: "int",
        predicted_period: "int",
        normal_container: "NormalContainer",
        wishart_container: "InvWishartContainer",
    ):
        super().__init__(observed_period, predicted_period)
        self.normal_container = normal_container
        self.wishart_container = wishart_container

    def create_domain(self) -> "NDArray":
        domain = numpy.arange(1, self.S + 1, 1)
        return domain

    def create_values(self, feature: "int", subject: "int") -> "NDArray":
        values = self.normal_container.mean(1, self.S + 1, feature, subject)
        return values

    def create_error(
        self, feature: "int", subject: "int", significance_level: "float"
    ) -> "NDArray":

        row_covariances = self.normal_container.covariance(
            1, self.S + 1, feature, feature
        )
        column_covariances = self.wishart_container.scale(
            1, self.S + 1, subject, subject
        )
        shapes = self.wishart_container.shape(1, self.S + 1)
        t_shapes = self.wishart_container.t_shape(1, self.S + 1, significance_level)
        errors = t_shapes * numpy.sqrt(row_covariances * column_covariances / shapes)

        return errors


class PredictableVisual(VisualStategy):
    def __init__(
        self,
        observed_period: "int",
        predicted_period: "int",
        normal_container: "NormalContainer",
        wishart_container: "InvWishartContainer",
    ):
        super().__init__(observed_period, predicted_period)
        self.normal_container = normal_container
        self.wishart_container = wishart_container

    def create_domain(self) -> "NDArray":
        domain = numpy.arange(self.S, self.S + self.P + 1, 1)
        return domain

    def create_values(self, feature: "int", subject: "int") -> "NDArray":
        values = self.normal_container.mean(
            self.S, self.S + self.P + 1, feature, subject
        )
        return values

    def create_error(
        self, feature: "int", subject: "int", significance_level: "float"
    ) -> "NDArray":

        row_covariances = self.normal_container.covariance(
            self.S, self.S + self.P + 1, feature, feature
        )
        column_covariances = self.wishart_container.scale(
            self.S, self.S + 1, subject, subject
        )
        shapes = self.wishart_container.shape(self.S, self.S + 1)
        t_shapes = self.wishart_container.t_shape(
            self.S, self.S + 1, significance_level
        )
        errors = t_shapes * numpy.sqrt(row_covariances * column_covariances / shapes)

        return errors


class DataVisual(VisualStategy):
    def __init__(
        self, observed_period: "int", predicted_period: "int", observations: "NDArray"
    ):
        super().__init__(observed_period, predicted_period)
        self.observations = observations

    def create_domain(self) -> "NDArray":
        domain = numpy.arange(1, self.S + self.P + 1, 1)

        return domain

    def create_values(self, feature: "int", subject: "int") -> "NDArray":
        values = self.observations[feature, subject, 0 : self.S + self.P]

        return values


class VisualDLM:
    def __init__(self, memory: "MemoryDLM", prime_memory: "PrimeMemoryDLM"):
        self.memory = memory
        self.prime_memory = prime_memory

        figure, axes = plt.subplots()
        figure.set_figwidth(20)
        self.figure: "Figure" = figure
        self.axes: "Axes" = axes

    def add_filtered_predictions(
        self, feature: "int", subject: "int", significance_level: "float"
    ):
        COLOUR = "green"
        TYPE = "Filtered Predictions"

        generator = ObservableVisual(
            self.prime_memory.S,
            self.prime_memory.P,
            self.memory.filtered_spaces,
            self.memory.wisharts,
        )

        domain = generator.create_domain()
        values = generator.create_values(feature, subject)
        errors = generator.create_error(feature, subject, significance_level)

        self.axes.fill_between(
            domain, values - errors, values + errors, alpha=0.35, color=COLOUR
        )
        self.axes.plot(
            domain,
            values,
            label=f"{TYPE} ({feature}, {subject})",
            color=COLOUR,
        )

    def add_observations(
        self,
        feature: "int",
        subject: "int",
    ):
        COLOUR = "blue"
        TYPE = "Observations"

        generator = DataVisual(
            self.prime_memory.S, self.prime_memory.P, self.prime_memory.observations
        )

        domain = generator.create_domain()
        values = generator.create_values(feature, subject)

        self.axes.plot(
            domain,
            values,
            label=f"{TYPE} ({feature}, {subject})",
            color=COLOUR,
        )

    def add_smoothed_predictions(
        self, feature: "int", subject: "int", significance_level: "float"
    ):

        COLOUR = "red"
        TYPE = "Smoothed Predictions"

        generator = ObservableVisual(
            self.prime_memory.S,
            self.prime_memory.P,
            self.memory.smoothed_spaces,
            self.memory.wisharts,
        )

        domain = generator.create_domain()
        values = generator.create_values(feature, subject)
        errors = generator.create_error(feature, subject, significance_level)

        self.axes.fill_between(
            domain, values - errors, values + errors, alpha=0.35, color=COLOUR
        )
        self.axes.plot(
            domain,
            values,
            label=f"{TYPE} ({feature}, {subject})",
            color=COLOUR,
        )

    def add_predicted_predictions(
        self, feature: "int", subject: "int", significance_level: "float"
    ):

        COLOUR = "violet"
        TYPE = "Further Predictions"

        generator = PredictableVisual(
            self.prime_memory.S,
            self.prime_memory.P,
            self.memory.predicted_spaces,
            self.memory.wisharts,
        )

        domain = generator.create_domain()
        values = generator.create_values(feature, subject)
        errors = generator.create_error(feature, subject, significance_level)

        self.axes.fill_between(
            domain, values - errors, values + errors, alpha=0.35, color=COLOUR
        )
        self.axes.plot(
            domain,
            values,
            label=f"{TYPE} ({feature}, {subject})",
            color=COLOUR,
        )

    def add_evolved_predictions(
        self, feature: "int", subject: "int", significance_level: "float"
    ):

        COLOUR = "orange"
        TYPE = "Evolved Predictions"

        generator = ObservableVisual(
            self.prime_memory.S,
            self.prime_memory.P,
            self.memory.evolved_spaces,
            self.memory.wisharts,
        )

        domain = generator.create_domain()
        values = generator.create_values(feature, subject)
        errors = generator.create_error(feature, subject, significance_level)

        self.axes.fill_between(
            domain, values - errors, values + errors, alpha=0.35, color=COLOUR
        )
        self.axes.plot(
            domain,
            values,
            label=f"{TYPE} ({feature}, {subject})",
            color=COLOUR,
        )

    def show_image(self, save_image: "bool" = False):
        self.axes.legend()

        if save_image:
            self.figure.savefig("saved-dlm-image")

        self.figure.show(warn=False)
