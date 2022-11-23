from utils import rand_obs, transmutator, relister
from typing import List
from objects import NormalModel, TransitionDensity


def forward(
    period: "int",
    m: "int",
    n: "int",
    n0: "NormalModel",
    te: "TransitionDensity",
    tp: "TransitionDensity",
    f1: "List[NormalModel]",
    f2: "List[NormalModel]",
    e1: "List[NormalModel]",
    e2: "List[NormalModel]",
    o2: "List[NormalModel]",
    smoothers: "List[TransitionDensity]",
):

    f1.append(n0)

    # Forward pass
    for i in range(period):

        o2.append(rand_obs((m, n)))

        nf1 = f1[-1]
        ne1, ts = transmutator(nf1, te)
        ne2, tf = transmutator(ne1, tp)
        no2 = o2[i]
        nf1 = transmutator(no2, tf, True)
        nf2 = transmutator(nf1, tp, True)

        smoothers.append(ts)
        e1.append(ne1)
        e2.append(ne2)
        f1.append(nf1)
        f2.append(nf2)

    return (f1, f2, e1, e2, o2, smoothers)


def backward(
    period: "int",
    tp: "TransitionDensity",
    s1: "List[NormalModel]",
    s2: "List[NormalModel]",
    f1: "List[NormalModel]",
    smoothers: "List[TransitionDensity]",
):

    s1.append(f1[-1])

    # Backward pass
    for i in range(period):

        ns1 = s1[-1]
        ts = smoothers[-1 - i]
        ns2 = transmutator(ns1, tp, True)
        ns1 = transmutator(ns1, ts, True)

        s1.append(ns1)
        s2.append(ns2)

    s1 = relister(s1)
    s2 = relister(s2)

    return (s1, s2)
