from numpy.typing import NDArray


def symmetrise(array: "NDArray") -> "NDArray":

    return (array + array.T) / 2
