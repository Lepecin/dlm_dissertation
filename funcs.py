from utils import rand_obs, transmutator, relister
from objects import NormalModel, TransitionDensity
from memory import DLModelMemory


def forward(
    period: "int",
    m: "int",
    n: "int",
    n0: "NormalModel",
    te: "TransitionDensity",
    tp: "TransitionDensity",
    memory: "DLModelMemory",
):

    memory.f1.append(n0)

    # Forward pass
    for i in range(period):

        memory.o2.append(rand_obs((m, n)))

        nf1 = memory.f1[-1]
        ne1, ts = transmutator(nf1, te)
        ne2, tf = transmutator(ne1, tp)
        no2 = memory.o2[i]
        nf1 = transmutator(no2, tf, True)
        nf2 = transmutator(nf1, tp, True)

        memory.smoothers.append(ts)
        memory.e1.append(ne1)
        memory.e2.append(ne2)
        memory.f1.append(nf1)
        memory.f2.append(nf2)

    return memory


def backward(period: "int", tp: "TransitionDensity", memory: "DLModelMemory"):

    memory.s1.append(memory.f1[-1])

    # Backward pass
    for i in range(period):

        ns1 = memory.s1[-1]
        ts = memory.smoothers[-1 - i]
        ns2 = transmutator(ns1, tp, True)
        ns1 = transmutator(ns1, ts, True)

        memory.s1.append(ns1)
        memory.s2.append(ns2)

    memory.s1 = relister(memory.s1)
    memory.s2 = relister(memory.s2)

    return memory
