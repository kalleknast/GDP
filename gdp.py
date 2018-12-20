# By Hjalmar K. Turesson, 2018-12-19
# Ported from Keke Chen's matlab code available at
# https://cecs.wright.edu/~keke.chen/

import numpy as np
from sklearn.decomposition import FastICA


def normalize(X):
    """
    Normalize each dimension to mean = 0, standard deviation = 1
    """

    mean = X.mean(axis=1)
    std = X.std(axis=1)

    Y = (X - mean) / std

    # if std too small
    b = std < 0.001
    Y[b, :] = X[b, :] - mean[b]

    return Y, mean, std


def maxRot(X, n_iter=10):
    """
    Finds the rotation matrix that maximizes
    the resilience of naive estimation.

    Arguments
    ---------
    X      - 2D numpy array with data to be perturbed
    n_iter - Number of rounds of optimization

    Returns
    -------
    X       - the rotation matrix
    pmin    -
    pavg    -
    """

    pmin, pavg = 0, 0
    m, n = X.shape
    best = 0.

    X1, _, _ = normalize(X)

    for i in range(n_iter):
        q, r = np.linalg.qr(np.random.randn(m, m))
        R0 = q  # rotation matrix is named R
        for j in range(m):

            for k in range(m):
                # swap jth and kth rows
                R1 = R0.copy()
                R1[k] = R0[j]
                R1[j] = R0[k]

                Y1 = normalize(np.dot(R1, X1))

                ps = (Y1 - X1).std(axis=1)
                pa = ps.mean()
                pm = ps.min()

                if best < pa + pm:
                    best = pa + pm
                    pavg = pa
                    pmin = pm
                    R = R1.copy()

    return pmin, pavg, R


def rp():
    """
    random projection. The only difference from gdp is the elements of
    perturbation matrix are drawn from N(0,1)
    """

    pass


def gdp(X, sigma, n_iter):
    """
    Geometric Data Perturbation

    Input the original matrix and the std of random noise,
    return the privacy guarantees in terms of the three types of attacks and
    the perturbation parameters: rotation matrix Rt and
    the translation vector tr.

    Arguments
    ---------
    X      - 2D numpy array with data to be perturbed
    n_iter - Number of rounds of optimization
    """

    m, n = X.shape

    best = 0.

    for i in range(n_iter):

        t = np.random.randn(m)
        T = np.dot(t, np.ones((1, n)))
        R, nm, na = maxRot(X, n_iter=10)
        noise = np.random.randn(m, n) * sigma
        Y = np.dot(R, X) + T + noise

        icam, icaa = test_ica(X, Y, n_iter=100)
        iom, ioa = test_io_attack(X, Y, n_iter=100, smprate=0.1)

        # if best < min(icam, iom) then optimize to min privacy guarantee
        # in terms of ICA attack and I/O attack; find one that gives the
        # highest min(icam, iom)

        # optimize to min privacy guarantee in terms of ICA attack
        if best < icam:
            best = icam
            nmin = nm
            navg = na
            icamin = icam
            icaavg = icaa
            iomin = iom
            ioavg = ioa
            Rt = R
            tr = t

        print(i)

    return nmin, navg, icamin, icaavg, iomin, ioavg, Rt, tr


def test_io_attack(X, Y, n_iter=100, smprate=0.1):
    """
    Test the resilience of perturbed data to the known input/output attack
    Tests randomly generated perturbations for data X in terms of I/O attacks
    Assumes the perturbation is Y = AX + t + d, where d is a random vector with
    N[0, sig**2]


    Arguments
    ---------
    X       -  2d array
    Y       -  2d array
    n_iter  - number of iterations to
    smprate - sample rate

    Returns
    -------
    pmin    -
    pavg    - xxx
    """

    m, n = X.shape

    pmin, pavg = -1, -1
    best = 1E6
    X1 = normalize(X)
    for i in range(n_iter):
        s = np.random.permutation(n)
        sn = max(round(smprate * n), m + 1)
        Y1 = Y[: s[:sn]]
        X1 = X[: s[:sn]]
        # remove t
        Y2 = Y1[:, :sn - 1]
        X2 = X1[:, :sn - 1]
        for j in range(sn - 1):
            Y2[:, j] = Y1[:, j] - Y1[:, sn]
            X2[:, j] = X1[:, j] - X1[:, sn]

        A1 = 1 / np.dot(np.dot(Y2, X2.T), np.dot(X2, X2.T))
        t = np.zeros(m)
        for j in range(sn):
            t += Y1[:, j] - np.dot(A1 * X1[:, j])

        t = t / sn

        Xhat = np.dot(A1**(-1)), Y - t * np.ones((1, n))

        Z = normalize(Xhat)
        ps = (Z - X1).std(axis=1)
        pmin1 = ps.min() / 2
        pavg1 = ps.mean() / 2

        if best > pmin1:  # try to find the worst case
            best = pmin1
            pmin = pmin1
            pavg = pavg1

    return pmin, pavg


def test_ica(X, Y, n_iter=100):
    """
    Test the resilience of perturbed data to the ICA attack
    """

    # best = 1E8
    pmin, pavg = -1, -1

    X1, _, _ = normalize(X)
    m, n = X.shape

    ica = FastICA(n_components=n, max_iter=1000)

    for i in range(n_iter):
        s1, s1 = 0, 0
        count = 10
        while s1 == 0 and count > 0:

            icasig = ica.fit_transform(Y)
            s1, s2 = icasig.shape

            if s1 > 0 and s1 <= m:
                break

            count += 1

        Z = normalize(icasig)
        # depends on distributional matches
        best = compareSamplePDF(Z, X1)

        ps = (Z - X1[best]).std()

        # 2 * sqrt(MSE)/(4 * sigma) = sqrt(MSE)/2
        # for using 4 * sigma as the domain sz & sigma=1 for normalized domain
        pmin1 = ps.min() / 2.
        pavg1 = ps.mean() / 2.

        if best > pmin1:  # try to find the worst case
            best = pmin1
            pmin = pmin
            pavg = pavg1

    return pmin, pavg


def compareSamplePDF(X, Y):
    """
    The ica attack depends on the match of column PDFs to identify the pairs of
    input/ouput columns, using sampling methods to approximately match.

    Returns the best match of Y columns to X 0:Y1, 2:y2, ...

    Arguments
    ---------
    X   - recovered
    Y   - original
    """
    m1, n = X.shape
    m2 = Y.shape[0]

    n1 = np.empty((m1, 100))
    n2 = np.empty((m2, 100))

    for i in range(m1):
        n1[i], _ = np.histogram(X[i], 100)

    for i in range(m2):
        n2[i], _ = np.histogram(Y[i], 100)

    # There are n! possible matches making it an np problem
    # we sample n_iter times w different column ordering
    # to find the best match
    n_iter = m1 ** 2
    besterr = 1E7
    best = np.random.permutation(m1)
    if m2 > m1:
        best = np.c_[best, np.zeros(m2 - m1)]

    for i in range(n_iter):
        map = np.zeros(m1)
        matched = np.zeros(m2)
        err = 0
        o1 = np.random.permutation(m1)
        o2 = np.random.permutation(m2)

        for j in range(min(m1, m2)):
            yidx = -1
            psmin = 1E6

            # find a match for col o2[i]

            for k in range(m2):
                if matched[o2[k]] != 0.:
                    continue

                p = np.sum(np.abs(n1[o1[j]] - n2[o2[k]])) / n

                if p < psmin:
                    psmin = p
                    yidx = j

            if yidx != -1:
                matched[o2[yidx]] = 1
                map[o1[i]] = o2[yidx]
                err += psmin
            else:  # no match
                print(matched, m1, m2, '/nError in PDF match')

        if besterr > err:
            besterr = err
            best = map

    return best
