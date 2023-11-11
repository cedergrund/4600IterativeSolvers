import numpy as np
import time
import scipy
import matplotlib.pyplot as plt


def driver():
    num1, dens1 = 100, 10
    num2, dens2 = 50, 20
    sizes = np.zeros(num1 + num2)
    time_p, it_p = np.zeros(num1 + num2), np.zeros(num1 + num2)
    time_inv, it_inv = np.zeros(num1 + num2), np.zeros(num1 + num2)
    time_ray, it_ray = np.zeros(num1 + num2), np.zeros(num1 + num2)
    time_real = np.zeros(num1 + num2)
    err_p, err_inv, err_ray = [], [], []

    for i in range(num1):
        # size of matrix
        if i % (num1 / 5) == 0:
            print("{:.0%}".format(i / (num1 + num2)))
        n = i * dens1 + 3
        sizes[i] = n

        # output run-time
        (
            time_p[i],
            it_p[i],
            time_inv[i],
            it_inv[i],
            time_ray[i],
            it_ray[i],
            time_real[i],
            accurate,
        ) = testOne(n, sparse=True)

        if not accurate[0]:
            err_p.append(i)
        if not accurate[1]:
            err_inv.append(i)
        if not accurate[2]:
            err_ray.append(i)

    for i in range(num2):
        # size of matrix
        if i % (num2 / 5) == 0:
            print("{:.0%}".format((i + num1) / (num1 + num2)))
        n = num1 * dens1 + 3 + i * dens2
        sizes[i + num1] = n

        # output run-time
        (
            time_p[i + num1],
            it_p[i + num1],
            time_inv[i + num1],
            it_inv[i + num1],
            time_ray[i + num1],
            it_ray[i + num1],
            time_real[i + num1],
            accurate,
        ) = testOne(n, sparse=True)

        if not accurate[0]:
            err_p.append(i + num1)
        if not accurate[1]:
            err_inv.append(i + num1)
        if not accurate[2]:
            err_ray.append(i + num1)

    top = num1 * dens1 + 3 + num2 * dens2
    x = np.linspace(0, top, 1000)
    f = lambda x: 10000

    plt.figure()
    plt.semilogy(x, list(map(f, x)), "k--", label="max_iterations")
    plt.semilogy(sizes, it_p, "b.", label="Power Method")
    plt.semilogy(sizes[err_p], it_p[err_p], "bx")
    plt.semilogy(sizes, it_inv, "r.", label="Inverse Power Method")
    plt.semilogy(sizes[err_inv], it_inv[err_inv], "rx")
    plt.semilogy(sizes, it_ray, "g.", label="Rayleigh Ritz Method")
    plt.semilogy(sizes[err_ray], it_ray[err_ray], "gx")
    plt.title("real tridiagonal matrix", style="italic")
    plt.suptitle("Iterations to compute Eigenvalues")
    plt.xlabel("Size, $n$", fontsize=14)
    plt.ylabel("Num. Iterations", fontsize=14)
    plt.legend()
    plt.show

    plt.figure()
    plt.semilogy(sizes, time_p, "b.", label="Power Method")
    plt.semilogy(sizes[err_p], time_p[err_p], "bx")
    plt.semilogy(sizes, time_inv, "r.", label="Inverse Power Method")
    plt.semilogy(sizes, time_ray, "g.", label="Rayleigh Ritz Method")
    plt.semilogy(sizes[err_inv], time_inv[err_inv], "rx")
    plt.semilogy(sizes[err_ray], time_ray[err_ray], "gx")
    plt.semilogy(sizes, time_real, "k.", label="np.eig()")
    plt.title("real tridiagonal matrix", style="italic")
    plt.suptitle("Time to compute Eigenvalues")
    plt.xlabel("Size, $n$", fontsize=14)
    plt.ylabel("Time, seconds", fontsize=14)
    plt.legend()
    plt.show()

    return


def testOne(n, sym=False, diag=False):
    if sym:
        A = np.random.rand(n, n)
        At = np.transpose(A)
        A = np.matmul(At, A)
    elif diag:
        k = 1
        k1 = np.random.random((2 * k + 1, n))
        k2 = np.concatenate([np.arange(0, k + 1, 1), np.arange(-k, 0, 1)])
        sp = scipy.sparse.diags(k1, k2)
        A = sp.toarray()
    else:
        A = np.random.rand(n, n)

    t0 = time.time()
    e1, _, it_p = powerMethod(A)
    t1 = time.time()
    time_p = t1 - t0

    t0 = time.time()
    e2, _, it_inv = inversePowerMethod(A)
    t1 = time.time()
    time_inv = t1 - t0

    t0 = time.time()
    e3, _, it_ray = rayleighQuotient(A)
    t1 = time.time()
    time_ray = t1 - t0

    t0 = time.time()
    if sym:
        eig_vals, _ = np.linalg.eigh(A)
    else:
        eig_vals, _ = np.linalg.eig(A)
    t1 = time.time()
    time_real = t1 - t0
    ind1 = np.argmax(abs(eig_vals))
    ind2 = np.argmin(abs(eig_vals))
    eig_val1 = eig_vals[ind1].real
    eig_val2 = eig_vals[ind2].real

    accurate = [(e1 - eig_val1) < 1e-5, (e2 - eig_val2) < 1e-5, (e3 - eig_val1) < 1e-5]

    return time_p, it_p, time_inv, it_inv, time_ray, it_ray, time_real, accurate


def powerMethod(A, shift=0, tol=1e-8, Nmax=10000):
    n = A.shape[1]
    count = Nmax

    # create a random vector x to check against
    v0 = np.random.rand(n, 1)
    A = A - np.multiply(shift, np.eye(n))

    # run power method iteration
    for i in range(Nmax):
        v1 = np.matmul(A, v0)
        v1 = np.multiply(1 / np.linalg.norm(v1), v1)

        if np.linalg.norm(abs(v1) - abs(v0)) < tol:
            count = i + 1
            break
        v0 = v1

    # predict eigenvalue and return
    vt1 = np.transpose(v1)
    ev = np.dot(np.matmul(vt1, A), v1)[0][0]
    ev += shift
    return ev, v1, count


def inversePowerMethod(A, shift=0, tol=1e-8, Nmax=10000):
    n = A.shape[1]
    count = Nmax

    # create a random vector x to check against
    v0 = np.random.rand(n, 1)
    A = A - np.multiply(shift, np.eye(n))
    A = np.linalg.inv(A)

    # run power method iteration
    for i in range(Nmax):
        v1 = np.matmul(A, v0)
        v1 = np.multiply(1 / np.linalg.norm(v1), v1)

        if np.linalg.norm(abs(v1) - abs(v0)) < tol:
            count = i + 1
            break
        v0 = v1

    # predict eigenvalue and return
    vt1 = np.transpose(v1)
    ev = np.dot(np.matmul(vt1, A), v1)[0][0]
    ev = 1 / ev + shift
    return ev, v1, count


def rayleighQuotient(A, tol=1e-10, Nmax=10000):
    n = A.shape[1]
    count = Nmax

    # create a random vector x to check against
    v0 = np.random.rand(n, 1)
    v0t = np.transpose(v0)
    e0 = (np.dot(np.matmul(v0t, A), v0) / np.dot(v0t, v0))[0][0]

    # run power method iteration
    for i in range(Nmax):
        v1 = np.linalg.solve(A - np.multiply(e0, np.eye(n)), v0)
        v1 = np.multiply(1 / np.linalg.norm(v1), v1)
        v1t = np.transpose(v1)
        e1 = np.dot(np.matmul(v1t, A), v1)[0][0]

        if abs(e1) - abs(e0) < tol:
            count = i + 1
            break
        v0 = v1
        e0 = e1

    return e1, v1, count


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
