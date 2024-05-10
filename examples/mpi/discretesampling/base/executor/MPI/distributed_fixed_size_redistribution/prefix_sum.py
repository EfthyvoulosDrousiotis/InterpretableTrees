import numpy as np
from scipy.special import logsumexp
from mpi4py import MPI


def LSE(xmem, ymem, dt):
    x = np.frombuffer(xmem, dtype=np.float64)
    y = np.frombuffer(ymem, dtype=np.float64)
    y[:] = logsumexp(np.hstack((x, y)))


def inclusive_prefix_sum(array):
    comm = MPI.COMM_WORLD

    csum = np.cumsum(array).astype(array.dtype)
    offset = np.zeros(1, dtype=array.dtype)
    MPI_dtype = MPI._typedict[array.dtype.char]
    comm.Exscan(sendbuf=[csum[-1], MPI_dtype], recvbuf=[offset, MPI_dtype], op=MPI.SUM)

    return csum + offset


def inclusive_prefix_logsumexp(array):
    comm = MPI.COMM_WORLD

    op = MPI.Op.Create(LSE, commute=True)

    logcsum = np.logaddexp.accumulate(array).astype(array.dtype)
    leaf_node = np.array([-np.inf]).astype(array.dtype) if len(array) == 0 else logcsum[-1]
    offset = -np.inf * np.ones(1, dtype=array.dtype)
    MPI_dtype = MPI._typedict[array.dtype.char]

    comm.Exscan(sendbuf=[leaf_node, MPI_dtype], recvbuf=[offset, MPI_dtype], op=op)
    op.Free()

    return np.logaddexp(logcsum, offset)
