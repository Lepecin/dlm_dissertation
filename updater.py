from memory import DLModelMemory
from utils import transmutator
from copy import deepcopy


class DLModelUpdater:
    @staticmethod
    def evolve(memory: "DLModelMemory", index: "int") -> "DLModelMemory":

        memory.check("Evolution-Prepared")

        # Prepare initial filtered state
        nfst = memory.fst[index]

        # Prepare evolver
        te = memory.evolvers[index]

        # Create the evolved state and a smoother
        nest, ts = transmutator(nfst, te)

        # Store evolved state
        memory.est.append(nest)

        # Store smoother
        memory.smoothers.append(ts)

        return deepcopy(memory)

    @staticmethod
    def predict(memory: "DLModelMemory", index: "int") -> "DLModelMemory":

        memory.check("Prediction-Prepared")

        # Prepare initial filtered state
        nest = memory.est[index]

        # Prepare predictor
        tp = memory.predictors[index]

        # Create the evolved space and a filter
        nesp, tf = transmutator(nest, tp)

        # Store evolved space
        memory.esp.append(nesp)

        # Store filter
        memory.filters.append(tf)

        return deepcopy(memory)

    @staticmethod
    def filtrate(memory: "DLModelMemory", index: "int") -> "DLModelMemory":

        memory.check("Filtering-Prepared")

        # Prepare the observed space
        nosp = memory.osp[index]

        # Prepare evolver
        tf = memory.filters[index]

        # Prepare predictor
        tp = memory.predictors[index]

        # Create the filtered state
        nfst = transmutator(nosp, tf, True)

        # Create the filtered space
        nfsp = transmutator(nfst, tp, True)

        # Store filtered state
        memory.fst.append(nfst)

        # Store filtered space
        memory.fsp.append(nfsp)

        return deepcopy(memory)

    @staticmethod
    def smoothen(memory: "DLModelMemory", index: "int") -> "DLModelMemory":

        memory.check("Smoothing-Prepared")

        # Prepare smoothed state
        nsst = memory.sst[index]

        # Prepare smoother
        ts = memory.smoothers[-1 - index]

        # Prepare predictor
        tp = memory.predictors[-1 - index]

        # Create smoothed space
        nssp = transmutator(nsst, tp, True)

        # Create smoothed state
        nsst = transmutator(nsst, ts, True)

        # Store smoothed state
        memory.sst.append(nsst)

        # Store smoothed space
        memory.ssp.append(nssp)

        return deepcopy(memory)
