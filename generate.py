from memory import DLModelMemory, DLModelPrimeMemory
from objects import TransitionDensity, NormalModel
from utils import rand_obs
from typing import Protocol


class DLModelGenerator(Protocol):
    @staticmethod
    def gen_evolver(
        prime: "DLModelPrimeMemory", memory: "DLModelMemory", index: "int"
    ) -> "TransitionDensity":

        ...

    @staticmethod
    def gen_predictor(
        prime: "DLModelPrimeMemory", memory: "DLModelMemory"
    ) -> "TransitionDensity":

        ...

    @staticmethod
    def gen_primordial(
        prime: "DLModelPrimeMemory", memory: "DLModelMemory", index: "int"
    ) -> "NormalModel":

        ...

    @staticmethod
    def gen_observation(
        prime: "DLModelPrimeMemory", memory: "DLModelMemory", index: "int"
    ) -> "NormalModel":

        ...


class DLModelTestGen:
    @staticmethod
    def gen_evolver(
        prime: "DLModelPrimeMemory",
        memory: "DLModelMemory",
        index: "int",
        ispred: "bool" = False,
    ) -> "TransitionDensity":

        return prime.te

    @staticmethod
    def gen_predictor(
        prime: "DLModelPrimeMemory",
        memory: "DLModelMemory",
        index: "int",
        ispred: "bool" = False,
    ) -> "TransitionDensity":

        return prime.tp

    @staticmethod
    def gen_primordial(
        prime: "DLModelPrimeMemory", memory: "DLModelMemory"
    ) -> "NormalModel":

        return prime.n0

    @staticmethod
    def gen_observation(
        prime: "DLModelPrimeMemory", memory: "DLModelMemory", index: "int"
    ) -> "NormalModel":

        return rand_obs((prime.m, prime.n))
