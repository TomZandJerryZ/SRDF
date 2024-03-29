import cv2
import numpy as np
cimport cython
cimport numpy as np

ctypedef np.float32_t DTYPE_t


def locat_overlaps(
        np.ndarray[DTYPE_t, ndim=2] boxes,
        np.ndarray[DTYPE_t, ndim=2] query_boxes,
        np.ndarray[DTYPE_t, ndim=2] indicator
        ):
    """
    Parameters
    ----------
    boxes: (N, 5) ndarray of float
    query_boxes: (K, 5) ndarray of float
    Returns
    -------
    overlaps: (N, K) ndarray of overlap between boxes and query_boxes
    """
    cdef unsigned int N = boxes.shape[0]
    cdef unsigned int K = query_boxes.shape[0]
    cdef DTYPE_t box_area
    cdef DTYPE_t ua, ia
    cdef unsigned int k, n, rtn
    cdef np.ndarray[DTYPE_t, ndim=3] contours
    cdef np.ndarray[DTYPE_t, ndim=1] overlaps = np.zeros((N), dtype=np.float32)

    cdef np.ndarray[DTYPE_t, ndim=1] a_tt = boxes[:, 4]
    cdef np.ndarray[DTYPE_t, ndim=1] a_ws = boxes[:, 2] - boxes[:, 0]
    cdef np.ndarray[DTYPE_t, ndim=1] a_hs = boxes[:, 3] - boxes[:, 1]
    cdef np.ndarray[DTYPE_t, ndim=1] a_xx = boxes[:, 0] + a_ws * 0.5
    cdef np.ndarray[DTYPE_t, ndim=1] a_yy = boxes[:, 1] + a_hs * 0.5

    cdef np.ndarray[DTYPE_t, ndim=1] b_tt = query_boxes[:, 4]
    cdef np.ndarray[DTYPE_t, ndim=1] b_ws = query_boxes[:, 2] - query_boxes[:, 0]
    cdef np.ndarray[DTYPE_t, ndim=1] b_hs = query_boxes[:, 3] - query_boxes[:, 1]
    cdef np.ndarray[DTYPE_t, ndim=1] b_xx = query_boxes[:, 0] + b_ws * 0.5
    cdef np.ndarray[DTYPE_t, ndim=1] b_yy = query_boxes[:, 1] + b_hs * 0.5


    for n in range(N):
        if indicator[n][0] == 0:
            overlaps[n] = 0
        else:
            box_area = b_ws[n] * b_hs[n]
            ua = a_ws[n] * a_hs[n] + box_area
            rtn, contours = cv2.rotatedRectangleIntersection(
                    ((a_xx[n], a_yy[n]), (a_ws[n], a_hs[n]), a_tt[n]),
                    ((b_xx[k], b_yy[n]), (b_ws[n], b_hs[n]), b_tt[n])
                )
            if rtn == 1:
                ia = np.round(np.abs(cv2.contourArea(contours)))
                overlaps[n] = ia / (ua - ia)
            elif rtn == 2:
                ia = np.minimum(ua - box_area, box_area)
                overlaps[n] = ia / (ua - ia)

    return overlaps

