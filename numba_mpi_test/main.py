import numpy as np
from numba_mpi import rank, size, send, recv
from numba import njit


@njit
def do_work(a):
    r = rank()
    s = size()
    a[0][0] = -r
    a[0][1] = -s

    if r != 0:
        send(a, dest=0, tag=1)
        return
    for i in range(1, s):
        recv(a[i], source=i, tag=1)


def main():
    r = rank()
    if r == 0:
        a = np.zeros((size(), 2), np.int32)
    else:
        a = np.zeros((1, 2), np.int32)
    do_work(a)

    if r == 0:
        print(a)


if __name__ == '__main__':
    main()
