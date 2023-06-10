import cv2
import numpy as np
cimport cython
cimport numpy as np

ctypedef np.float32_t DTYPE_t


def points_to_rect(
        np.ndarray[DTYPE_t, ndim=3] boxes):
    """
    Parameters
    ----------
    boxes: (N, 6, 2) ndarray of float
    Returns
    -------
    rects: (N, 5) ndarray of float
    """
    cdef unsigned int N = boxes.shape[0]
    cdef DTYPE_t x, y, w, h, t
    cdef unsigned int n
    cdef np.ndarray[DTYPE_t, ndim=2] rects = np.zeros((N, 5), dtype=np.float32)

    for n in range(N):
        (x, y), (w, h) ,t = cv2.minAreaRect(boxes[n].astype(np.int32))
        rects[n] = (x, y, w, h, t)


    return rects

