import numpy as np
import time
import scipy

"""
todo:

arnoldi + lanczos write code

walk

go over hw solutions

mashed potatoes

apply arnoldi to richardsons to see if it is faster than alternate method

text kevin to check in after midterm


"""

"""
notes:
power method converges well and better than ritz

very good for all matrices and symmetric, not great for sparse

inverse power method not good and eigh method runs much better



"""


def driver():
    n = 2000
    t0 = time.time()
    sym = False
    A = generatePDmatrix(n, sparse=True)
    # A = np.random.rand(n, n)
    # At = np.transpose(A)
    # A = np.matmul(At, A)
    t1 = time.time()
    At = np.transpose(A)
    print("Generated matrix in {} seconds".format(t1 - t0))
    if np.array_equal(At, A):
        sym = True
        print("symmetric matrix")
    print()
    # A = np.matrix([[2, -12], [1, -5]])

    # print("A:\n", A, "\n")
    # x0 = np.matrix([[3], [1]])
    # x0 = x0 / np.linalg.norm(x0)

    t0 = time.time()
    v = powerMethod(A)
    vt = np.transpose(v)
    ev = np.dot(np.matmul(vt, A), v)
    ev /= np.dot(vt, v)
    t1 = time.time()
    print("Power method + Quotient took {} seconds".format(t1 - t0))
    print(ev[0][0])
    print()

    t0 = time.time()
    v, e = rayleighRitz(A)
    t1 = time.time()
    print("Rayleigh Ritz Iteration took {} seconds".format(t1 - t0))
    print(e)
    print()

    t0 = time.time()
    Ai = np.linalg.inv(A)
    v = powerMethod(Ai)
    vt = np.transpose(v)
    ev = np.dot(np.matmul(vt, A), v)
    ev /= np.dot(vt, v)
    t1 = time.time()
    print("Inverse Power method + Quotient took {} seconds".format(t1 - t0))
    print(ev[0][0])
    print()

    t0 = time.time()
    if sym:
        eig_vals, eig_vect = np.linalg.eigh(A)
    else:
        eig_vals, eig_vect = np.linalg.eig(A)
    t1 = time.time()
    ind1 = np.argmax(abs(eig_vals))
    ind2 = np.argmin(abs(eig_vals))
    print("Eigh method took {} seconds".format(t1 - t0))
    eig_val1 = float(eig_vals[ind1])
    eig_val2 = float(eig_vals[ind2])
    print(eig_val2, eig_val1)
    print()

    # print("error:")
    # print(
    #     "-> vector:", np.linalg.norm(abs(v) - abs(np.reshape(eig_vect[:, ind], (n, 1))))
    # )
    # print("-> eig_val:", np.linalg.norm(ev - eig_val))

    return


def powerMethod(A, tol=1e-8, Nmax=5000):
    # create a random vector x to check against
    x0 = np.random.rand(A.shape[1], 1)

    # run power method iteration
    for i in range(Nmax):
        x1 = np.matmul(A, x0)
        x1 = np.multiply(1 / np.linalg.norm(x1), x1)
        if np.linalg.norm(abs(x1) - abs(x0)) < tol:
            print("converged after", i, "iterations")
            return x1
        x0 = x1

    return x1


def rayleighRitz(A, tol=1e-8, Nmax=5000):
    # create a random vector v to check against + eigenvalue of v
    v0 = np.random.rand(A.shape[1], 1)
    v0t = np.transpose(v0)
    e0 = np.dot(np.matmul(v0t, A), v0) / np.dot(v0t, v0)

    # run power method iteration
    for i in range(Nmax):
        v1 = np.matmul(A, v0)
        v1 = np.multiply(1 / np.linalg.norm(v1), v1)
        v1t = np.transpose(v1)
        e1 = np.dot(np.matmul(v1t, A), v1) / np.dot(v1t, v1)
        if abs(e1 - e0) < tol:
            print("converged after", i, "iterations")
            return v1, e1[0][0]
        v0 = v1
        e0 = e1

    return v1, e1[0][0]


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
