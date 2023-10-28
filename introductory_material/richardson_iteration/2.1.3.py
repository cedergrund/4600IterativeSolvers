import numpy as np
import scipy
import matplotlib.pyplot as plt
import time

"""
current notes with testing:

large matrices: richardson is good. performs better
sparse matrices: richardson is very good. performs much better
semi-positive definite matrices: richardson fails with high n as convergence is near 1

Questions for TA/Professor: why is norm on numpy different from calculated norm via PD & Cauchy-Schwarz properties?

"""


def driver():
    # size of matrix
    n = 5
    sparse = False
    print("Testing on matrix of size", n, "| Sparse =", sparse, "\n")

    # generate matrices
    A, IalphA, alpha = generatePDmatrix(n=n, sparse=sparse, SPD=True, verbose=True)
    b = np.random.randint(-5, 5, (n, 1))
    x0 = np.zeros((n, 1))

    # Richardson's Iteration
    print("Running Richardson's Iteration:")
    rt0 = time.time()
    rich_sol = richardsonIteration(A, IalphA, np.multiply(alpha, b), x0, n, quick=True)
    rt1 = time.time()
    print("Error:", np.linalg.norm(np.matmul(A, rich_sol) - b))
    print("Time Taken:", rt1 - rt0, "\n")

    # Actual Solution
    print("Running Actual Solution:")
    rt0 = time.time()
    ex_sol = np.matmul(np.linalg.inv(A), b)
    rt1 = time.time()
    print("Error:", np.linalg.norm(np.matmul(A, ex_sol) - b))
    print("Time Taken:", rt1 - rt0)

    return


def richardsonIteration(A, IalphA, alphb, x0, n, tol=1e-05, Nmax=5000, quick=False):
    # quick iteration without extras
    if quick:
        for i in range(1, Nmax):
            x1 = np.add(np.matmul(IalphA, x0), alphb)
            if np.linalg.norm(x1 - x0) < tol:
                print("converged after", i, "iterations")
                break
            x0 = x1
        return x1

    # initial variable definition
    n = len(IalphA)
    it_err = 1
    msg = "No solution found"
    xk, xk1 = x0, x0
    iterations = np.zeros((2, Nmax, n))
    iterations[0][0] = np.transpose(x0)
    iterations[1][0] = np.matmul(A, xk1)

    # iteration
    j = 0
    for i in range(1, Nmax):
        xk1 = np.add(np.matmul(IalphA, xk), alphb)
        iterations[0][i] = np.transpose(xk1)
        iterations[1][i] = np.matmul(A, xk1)
        if np.linalg.norm(xk1 - xk) < tol:
            j = i
            it_err = 0
            msg = "solution found after " + str(i) + " iterations."
            break
        xk = xk1

    # output
    return xk1, it_err, msg, iterations[: j + 1]


def generatePDmatrix(n=3, sparse=False, density=0.1, SPD=False, verbose=False):
    """
    Method for generating a positive definite matrix. \n
    Parameters:
        int n: size of returned matrices,\n
        bool sparse: if True, generates a sparse PD matrix, \n
        float density (0,1): density of sparse matrix, defaults to 0.1. 1 is fully populated, \n
        bool SPD = if True, returns a semi-positive matrix instead of a positive definite,\n
        bool verbose = if True, will print output including matrix, eigenvalues, and rate of convergence\n
    Source: @Daryl on stack exchange - translated code from matlab
        link - https://math.stackexchange.com/questions/357980/how-to-generate-random-symmetric-positive-definite-matrices-using-matlab
    """

    # generate a positive definite matrix
    if not sparse:
        A = np.random.random((n, n))
        A = np.multiply(0.5, np.matmul(A, np.transpose(A)))
        if not SPD:
            A = np.add(A, np.multiply(n, np.identity(n)))
    else:
        # sparse matrix
        A = scipy.sparse.random(n, n, density * 2)
        A = np.triu(A.A)
        A = A + np.transpose(A) - np.diag(np.diag(A))
        A = np.add(A, np.multiply(n, np.identity(n)))

    # calculate alpha and ||I-alpha*A|| norm
    eig_vals, eig_vects = np.linalg.eigh(A)
    alpha = 2 / (eig_vals[0] + eig_vals[-1])
    convergence_mat = np.identity(n) - np.multiply(alpha, A)
    convergence = np.linalg.norm(convergence_mat)

    # print if desired before returning
    if verbose == True:
        # print("matrix:\n", A, "\nEigenvalues:\n", eig_vals, "\n")
        ceig, ceigvec = np.linalg.eigh(convergence_mat)
        print("Convergence (norm of I-alpha*A):", convergence)
        print("->by formula:", 1 - 2 * eig_vals[0] / (eig_vals[0] + eig_vals[-1]), "\n")

    return A, convergence_mat, alpha


print("\n")
driver()
print("\n")
