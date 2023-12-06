import numpy as np
import time
import scipy
import matplotlib.pyplot as plt


def driver():
    print(testOne(4, Hes=True))

    return
    num1, dens1 = 5, 100
    num2, dens2 = 0, 0
    sizes = np.zeros(num1 + num2)
    time_qr, it_qr = np.zeros(num1 + num2), np.zeros(num1 + num2)
    time_real = np.zeros(num1 + num2)
    err_qr = []

    for i in range(num1):
        # size of matrix
        print("{:.0%}".format(i / (num1 + num2)))
        n = i * dens1 + 3
        sizes[i] = n

        # output run-time
        (it_qr[i], time_qr[i], time_real[i], accurate) = testOne(n, sym=True)

        if not accurate:
            err_qr.append(i)

    for i in range(num2):
        # size of matrix
        if i % (num2 / 5) == 0:
            print("{:.0%}".format((i + num1) / (num1 + num2)))
        n = num1 * dens1 + 3 + i * dens2
        sizes[i + num1] = n

        # output run-time
        (it_qr[i + num1], time_qr[i + num1], time_real[i + num1], accurate) = testOne(
            n, sym=True
        )

        if not accurate:
            err_qr.append(i + num1)

    top = num1 * dens1 + 3 + num2 * dens2
    x = np.linspace(0, top, 1000)
    f = lambda x: 50000

    plt.figure()
    plt.semilogy(x, list(map(f, x)), "k--", label="max_iterations")
    plt.semilogy(sizes, it_qr, "b.", label="QR Iteration")
    plt.semilogy(sizes[err_qr], it_qr[err_qr], "bx")
    plt.title("real random symmetric matrix", style="italic")
    plt.suptitle("Iterations to compute Eigenvalues")
    plt.xlabel("Size, $n$", fontsize=14)
    plt.ylabel("Num. Iterations", fontsize=14)
    plt.legend()
    plt.show

    plt.figure()
    plt.semilogy(sizes, time_qr, "b.", label="QR Iteration")
    plt.semilogy(sizes[err_qr], time_qr[err_qr], "bx")
    plt.semilogy(sizes, time_real, "k.", label="np.eig()")
    plt.title("real random symmetric matrix", style="italic")
    plt.suptitle("Time to compute Eigenvalues")
    plt.xlabel("Size, $n$", fontsize=14)
    plt.ylabel("Time, seconds", fontsize=14)
    plt.legend()
    plt.show()

    return


def testOne(n, sym=False, diag=False, Hes=False):
    if sym:
        A = np.random.rand(n, n)
        At = np.transpose(A)
        A = np.matmul(At, A)
    elif Hes:
        A = np.random.rand(n, n)
        A = scipy.linalg.hessenberg(A)
        print(A)
    elif diag:
        k = 1
        k1 = np.random.random((2 * k + 1, n))
        k2 = np.concatenate([np.arange(0, k + 1, 1), np.arange(-k, 0, 1)])
        sp = scipy.sparse.diags(k1, k2)
        A = sp.toarray()
        A = np.matmul(A, np.transpose(A))
        # print(A)
    else:
        A = np.random.rand(n, n)

    t0 = time.time()
    _, R, it_qr = simultaneousQR(A)
    t1 = time.time()
    time_qr = t1 - t0
    eps = 1e-10
    R[np.abs(R) < eps] = 0
    R = np.sort(np.diag(R))

    t0 = time.time()
    if sym or diag:
        eig_vals, _ = np.linalg.eigh(A)
    else:
        eig_vals, _ = np.linalg.eig(A)
    t1 = time.time()
    eig_vals = np.sort(eig_vals.real)

    time_real = t1 - t0

    accurate = np.allclose(eig_vals, R)

    return it_qr, time_qr, time_real, accurate


def pureQR(A, shift=0, tol=1e-8, Nmax=100000):
    # run power method iteration
    for _ in range(Nmax):
        Q, R = np.linalg.qr(A)
        A = np.matmul(R, Q)

    return A


def simultaneousQR(A, tol=1e-8):
    # create Q initial
    n = A.shape[1]
    Q1 = np.eye(n)
    Q0d = np.ones(n)
    Q1d = np.zeros(n)

    count = 1

    # run power method iteration
    while not np.allclose(Q1d, Q0d):
        Q0d = Q1d
        X = np.matmul(A, Q1)
        Q1, R = np.linalg.qr(X)
        Q1d = np.diag(Q1)
        count += 1
        if count % 10000 == 0:
            print(count)
            if count >= 1000000:
                break

    # print("converged after", count, "iterations.")
    return Q1, R, count


def generatePDmatrix(
    n=3, sparse=False, density=0.1, SPD=False, banded=False, k=0, verbose=False
):
    """
    Method for generating a positive definite matrix. \n
    Parameters:
        int n: size of returned matrices,\n
        bool sparse: if True, generates a sparse PD matrix, \n
        float density (0,1): density of sparse matrix, defaults to 0.1. 1 is fully populated, \n
        bool SPD = if True, returns a semi-positive matrix instead of a positive definite,\n
        bool banded = if True, returns a banded positive matrix instead of a positive definite,\n
        int k: used in conjuction with 'banded' parameter as width of banded matrix. 0 is just diagonal,\n
        bool verbose = if True, will print output including matrix, eigenvalues, and rate of convergence\n
    Returns:
        np.matrix A: random matrix,\n
    References:
        For generating random PD matrices:
            @Daryl on stack exchange - translated code from matlab
            link - https://math.stackexchange.com/questions/357980/how-to-generate-random-symmetric-positive-definite-matrices-using-matlab
    """

    # generate a positive definite matrix
    if not sparse:
        # normal random
        A = np.random.random((n, n))
        A = np.multiply(0.5, np.matmul(A, np.transpose(A)))
        if not SPD:
            # make matrix positive definite
            A = np.add(A, np.multiply(n, np.identity(n)))
    elif banded:
        # random banded positive definite matrix creation
        k = k + 1
        k1 = np.random.random((k, n))
        k2 = np.arange(0, k, 1)
        sp = scipy.sparse.diags(k1, k2)
        A = sp.toarray()
        A = A + np.transpose(A) - np.diag(np.diag(A))
        A = np.add(A, np.multiply(len(A), np.identity(len(A))))
    else:
        # random sparse positive definite matrix creation
        A = scipy.sparse.random(n, n, density)
        A = np.triu(A.A)
        A = A + np.transpose(A) - np.diag(np.diag(A))
        A = np.add(A, np.multiply(n, np.identity(n)))

    return A


if __name__ == "__main__":
    print("\n")
    driver()
    print("\n")
