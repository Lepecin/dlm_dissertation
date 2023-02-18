from numpy import ndarray
from typing import List, Any, Tuple
from objects import NormalModel, TransitionModel, JointModel, InvWishartModel


class ModelContainer:
    def __init__(self, start: "int", end: "int"):

        self.container: "List[Any]" = max(1, end - start + 1) * [None]
        self.start = start

    def get_from_time(self, time: "int") -> "Any":
        return self.container[time - self.start]

    def set_at_time(self, time: "int", object: "Any"):
        self.container[time - self.start] = object


class PrimeMemoryDLM:
    def __init__(
        self,
        observed_period: "int",
        predicted_period: "int",
        observations: "ndarray",
        primordial_model: "NormalModel",
        primordial_error: "InvWishartModel",
        evolvers: "ModelContainer",
        observers: "ModelContainer",
    ):

        self.S = observed_period
        self.P = predicted_period

        # -- Data
        self.observations = observations

        # -- Models
        self.primordial_model = primordial_model
        self.primordial_error = primordial_error

        # -- Transitions
        self.evolvers = evolvers
        self.observers = observers


class MemoryDLM:
    def __init__(self, observed_period: "int", predicted_period: "int"):

        S = observed_period
        P = predicted_period

        self.S = S
        self.P = P

        # -- Space Models
        self.filtered_states = ModelContainer(0, S)
        self.evolved_states = ModelContainer(1, S)
        self.smoothed_states = ModelContainer(0, S)
        self.predicted_states = ModelContainer(S, S + P)

        # -- Space Models
        self.filtered_spaces = ModelContainer(1, S)
        self.evolved_spaces = ModelContainer(1, S)
        self.smoothed_spaces = ModelContainer(1, S)
        self.predicted_spaces = ModelContainer(S, S + P)

        # -- Transitions
        self.smoothers = ModelContainer(1, S)
        self.filterers = ModelContainer(1, S)

        # -- Error Matrix
        self.wisharts = ModelContainer(0, S)


def transition_transumer(
    model: "NormalModel", transition: "TransitionModel"
) -> "Tuple[NormalModel, TransitionModel]":

    joint_model = JointModel(model, transition)
    joint_model = joint_model.mutate_joint_model()
    model = joint_model.give_normal()
    transition = joint_model.give_transition()

    return model, transition


def model_transmuter(
    model: "NormalModel", transition: "TransitionModel"
) -> "NormalModel":

    joint_model = JointModel(model, transition)
    model = joint_model.mutate_normal()

    return model


class UpdaterDLM(MemoryDLM):
    def __init__(self, observed_period: "int", predicted_period: "int"):
        super().__init__(observed_period, predicted_period)

    def evolve(self, time: "int", evolver: "TransitionModel"):

        filtered_state = self.filtered_states.get_from_time(time)

        evolved_state, smoother = transition_transumer(filtered_state, evolver)

        self.smoothers.set_at_time(time + 1, smoother)
        self.evolved_states.set_at_time(time + 1, evolved_state)

    def observe_evolved(self, time: "int", observer: "TransitionModel"):

        evolved_state = self.evolved_states.get_from_time(time + 1)

        evolved_space, filterer = transition_transumer(evolved_state, observer)

        self.filterers.set_at_time(time + 1, filterer)
        self.evolved_spaces.set_at_time(time + 1, evolved_space)

    def filter(self, time: "int", observation: "ndarray"):

        evolved_space: "NormalModel" = self.evolved_spaces.get_from_time(time + 1)
        filterer: "TransitionModel" = self.filterers.get_from_time(time + 1)
        error: "InvWishartModel" = self.wisharts.get_from_time(time)

        filtered_state = filterer.observe(observation)
        error = evolved_space.update_wishart(error, observation)

        self.filtered_states.set_at_time(time + 1, filtered_state)
        self.wisharts.set_at_time(time + 1, error)

    def observe_filtered(self, time: "int", observer: "TransitionModel"):

        filtered_state = self.filtered_states.get_from_time(time + 1)

        filtered_space = model_transmuter(filtered_state, observer)

        self.filtered_spaces.set_at_time(time + 1, filtered_space)

    def smoothen(self, time: "int"):

        smoother = self.smoothers.get_from_time(self.S - time)
        smoothed_state = self.smoothed_states.get_from_time(self.S - time)

        smoothed_state = model_transmuter(smoothed_state, smoother)

        self.smoothed_states.set_at_time(self.S - time - 1, smoothed_state)

    def observe_smoothed(self, time: "int", observer: "TransitionModel"):

        smoothed_state = self.smoothed_states.get_from_time(self.S - time)

        smoothed_space = model_transmuter(smoothed_state, observer)

        self.smoothed_spaces.set_at_time(self.S - time, smoothed_space)

    def predict(self, time: "int", evolver: "TransitionModel"):

        predicted_state = self.predicted_states.get_from_time(self.S + time)

        predicted_state = model_transmuter(predicted_state, evolver)

        self.predicted_states.set_at_time(self.S + time + 1, predicted_state)

    def observe_predicted(self, time: "int", observer: "TransitionModel"):

        predicted_state = self.predicted_states.get_from_time(self.S + time + 1)

        predicted_space = model_transmuter(predicted_state, observer)

        self.predicted_spaces.set_at_time(self.S + time + 1, predicted_space)
