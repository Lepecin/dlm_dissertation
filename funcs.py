from utils import rand_obs, transmutator, relister
from memory import DLModelMemory, DLModelPrimeMemory


def forward(
    prime: "DLModelPrimeMemory",
    memory: "DLModelMemory",
) -> "DLModelMemory":

    # Add primordial state to filtered states
    # XXX implement generator for primordial state
    memory.f1.append(prime.n0)

    # Forward pass
    # For each time from zero to period - 1
    for i in range(prime.period):

        # Prepare filtered state
        nf1 = memory.f1[i]

        # XXX implement generator for evolver

        # Create evolved state and smoother
        ne1, ts = transmutator(nf1, prime.te)

        memory.e1.append(ne1)

        memory.smoothers.append(ts)

        # XXX implement generator for predictor

        # Create evolved space and filter
        ne2, tf = transmutator(ne1, prime.tp)

        memory.e2.append(ne2)

        # XXX implement generator for observation

        # Generate an observation
        memory.o2.append(rand_obs((prime.m, prime.n)))

        # Prepare observed space
        no2 = memory.o2[i]

        # Create filtered state
        nf1 = transmutator(no2, tf, True)

        memory.f1.append(nf1)

        # Create filtered space
        nf2 = transmutator(nf1, prime.tp, True)

        memory.f2.append(nf2)

    return memory


def backward(prime: "DLModelPrimeMemory", memory: "DLModelMemory") -> "DLModelMemory":

    memory.s1.append(memory.f1[-1])

    # Backward pass
    for i in range(prime.period):

        ns1 = memory.s1[i]

        ts = memory.smoothers[-1 - i]

        # XXX implement generator for predictor

        ns2 = transmutator(ns1, prime.tp, True)

        memory.s2.append(ns2)

        ns1 = transmutator(ns1, ts, True)

        memory.s1.append(ns1)

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
