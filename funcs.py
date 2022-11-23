from utils import rand_obs, transmutator, relister
from memory import DLModelMemory, DLModelPrimeMemory


def forward(
    prime: "DLModelPrimeMemory",
    memory: "DLModelMemory",
) -> "DLModelMemory":

    memory.f1.append(prime.n0)

    # Forward pass
    for i in range(prime.period):

        memory.o2.append(rand_obs((prime.m, prime.n)))

        nf1 = memory.f1[-1]
        ne1, ts = transmutator(nf1, prime.te)
        ne2, tf = transmutator(ne1, prime.tp)
        no2 = memory.o2[i]
        nf1 = transmutator(no2, tf, True)
        nf2 = transmutator(nf1, prime.tp, True)

        memory.smoothers.append(ts)
        memory.e1.append(ne1)
        memory.e2.append(ne2)
        memory.f1.append(nf1)
        memory.f2.append(nf2)

    return memory


def backward(prime: "DLModelPrimeMemory", memory: "DLModelMemory") -> "DLModelMemory":

    memory.s1.append(memory.f1[-1])

    # Backward pass
    for i in range(prime.period):

        ns1 = memory.s1[-1]
        ts = memory.smoothers[-1 - i]
        ns2 = transmutator(ns1, prime.tp, True)
        ns1 = transmutator(ns1, ts, True)

        memory.s1.append(ns1)
        memory.s2.append(ns2)

    memory.s1 = relister(memory.s1)
    memory.s2 = relister(memory.s2)

    return memory


def beyond(prime: "DLModelPrimeMemory", memory: "DLModelMemory") -> "DLModelMemory":

    memory.p1.append(memory.f1[-1])
    memory.p2.append(memory.f2[-1])

    for _ in range(prime.beyond_period):

        np1 = memory.p1[-1]
        np1 = transmutator(np1, prime.te, True)
        np2 = transmutator(np1, prime.tp, True)

        memory.p1.append(np1)
        memory.p2.append(np2)

    return memory
