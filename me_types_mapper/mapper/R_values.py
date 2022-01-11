"""R-values from https://link.springer.com/article/10.1007/s10044-016-0583-6#rightslink"""
import numpy as np
from me_types_mapper.mapper.coclustering_functions import unique_elements, count_elements

def thresholding(x):
    """Thresholding function."""

    if x > 0.:
        y = 1
    else:
        y = 0.
    return y


def kNN_P_S(X, y, p, k, s):
    """Counts the number of elements of label s in the k nearest neighbors"""

    from sklearn.neighbors import KNeighborsClassifier

    kNN = KNeighborsClassifier(n_neighbors=k, weights='uniform', algorithm='auto',
                               leaf_size=30, p=2, metric='minkowski', metric_params=None,
                               n_jobs=None)
    kNN.fit(X, y)
    # return the distance and the index of the k nearest neighbors
    Neigh_ = kNN.kneighbors()
    # return the labels og these neighbors
    neigh_lbls = np.asarray([y[idx] for idx in Neigh_[1][p]])
    # return the number of elements of label s
    cts = count_elements(neigh_lbls).reindex(unique_elements(y)).fillna(0.)
    return cts.T[s].values[0]


def R_value_C(X, y, k, s, theta):
    """Check if the counts of elements with a label s is above a threshold value theta."""
    # 0<theta< k/2

    msk_ = [lbl == s for lbl in y]

    R_ = 0.
    for p in np.arange(len(X))[msk_]:

        for s_ in unique_elements(y):
            if s_ != s:
                R_ += thresholding(kNN_P_S(X, y, p, k, s_) - theta)

    return R_ / len(X[msk_])


def R_value_U(X, y, k=None, theta=None):
    """Compute the R-value. Return high values for overlapping classes, low values for well separated classes"""

    if k == None:
        k = int(0.1 * len(X))

    R_U_ = 0.
    for d_lbl in unique_elements(y):
        msk_ = np.asarray([lbl == d_lbl for lbl in y])
        theta = int((k / 2) * len(X[~msk_]) / len(X))
        R_U_ += R_value_C(X, y, k, d_lbl, theta) * len(X[msk_])

    return R_U_ / len(X)