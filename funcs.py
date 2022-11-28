from operators import transmutator
from utils import relister
from memory import DLModelMemory, DLModelPrimeMemory
from generate import DLModelGenerator


def forward(
    prime: "DLModelPrimeMemory", memory: "DLModelMemory", gen: "DLModelGenerator"
) -> "DLModelMemory":

    # Primordial Initiation
    n0 = gen.gen_primordial(prime, memory)
    memory.f1.append(n0)

    # Forward pass
    for i in range(prime.period):

        # Evolution
        nf1 = memory.f1[i]
        te = gen.gen_evolver(prime, memory, i)
        memory.evolvers.append(te)
        ne1, ts = transmutator(nf1, te)
        memory.e1.append(ne1)
        memory.smoothers.append(ts)

        # Prediction
        tp = gen.gen_predictor(prime, memory, i)
        memory.predictors.append(tp)
        ne2, tf = transmutator(ne1, tp)
        memory.e2.append(ne2)

        # Filtration
        no2 = gen.gen_observation(prime, memory, i)
        memory.o2.append(no2)
        nf1 = transmutator(no2, tf, True)
        memory.f1.append(nf1)

        # Filtered Prediction
        nf2 = transmutator(nf1, tp, True)
        memory.f2.append(nf2)

    return memory


def backward(prime: "DLModelPrimeMemory", memory: "DLModelMemory") -> "DLModelMemory":

    # Smoothing Initiation
    ns0 = memory.f1[prime.period + 1]
    memory.s1.append(ns0)

    # Backward pass
    for i in range(prime.period):

        # Smoothing
        ns1 = memory.s1[i]
        ts = memory.smoothers[-1 - i]
        tp = memory.predictors[-1 - i]
        ns2 = transmutator(ns1, tp, True)
        memory.s2.append(ns2)

        # Smoothed Prediction
        ns1 = transmutator(ns1, ts, True)
        memory.s1.append(ns1)

    # Reverse list orders
    memory.s1 = relister(memory.s1)
    memory.s2 = relister(memory.s2)

    return memory


def beyond(prime: "DLModelPrimeMemory", memory: "DLModelMemory") -> "DLModelMemory":

    memory.p1.append(memory.f1[-1])

    memory.p2.append(memory.f2[-1])

    for i in range(prime.beyond_period):

        np1 = memory.p1[i]

        # XXX implement generator for evolver

        np1 = transmutator(np1, prime.te, True)

        memory.p1.append(np1)

        # XXX implement generator for predictor

        np2 = transmutator(np1, prime.tp, True)

        memory.p2.append(np2)

    return memory
