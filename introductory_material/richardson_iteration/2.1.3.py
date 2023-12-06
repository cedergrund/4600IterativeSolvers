import numpy as np
import scipy
import matplotlib.pyplot as plt
import time

"""
current notes with testing:

Rate of convergence of richardson's iteration is the same as spectral norm of I-alphaA, that is, 2*eig1/(eig1+eign)
where eig1 is smallest eigenvalue of A and eign is largest eigenvalue of A

running matrix testing to ~1300x1300 sized matrices. above that get some garbage data.
maybe change plots to include error as well.

notes (visuals in output folder):
large matrices: richardson is good. performs better
sparse matrices: richardson is very good. performs much better. The more sparse the better
banded matrices: richardson is pretty good. performs better but not as good as sparse
semi-positive definite matrices: richardson fails with high n as convergence is near 1

Questions for TA/Professor: why is norm on numpy different from calculated norm via spectral norm?

"""


def driver():
    # testing on various parameters. outputs plots. Included all of them in outputs folder
    num1, dens1 = 100, 10
    num2, dens2 = 50, 20
    sizes = np.zeros(num1 + num2)
    rtd = np.zeros(num1 + num2)
    itd = np.zeros(num1 + num2)
    rich_it = np.zeros(num1 + num2)
    err = []

    for i in range(num1):
        # size of matrix
        if i % (num1 / 5) == 0:
            print("{:.0%}".format(i / (num1 + num2)))
        n = i * dens1 + 5
        sizes[i] = n

        # generate matrices
        # A, IalphA, alpha = random_cov(n)
        A, IalphA, alpha = generatePDmatrix(n=n, SPD=True)
        b = np.random.randint(-5, 5, (n, 1))
        x0 = np.zeros((n, 1))

        # output run-time
        rtd[i], itd[i], rich_it[i], accurate = quickTesting(n, A, IalphA, alpha, b, x0)
        if not accurate:
            err.append(i)

    for i in range(num2):
        # size of matrix
        if i % (num2 / 5) == 0:
            print("{:.0%}".format((i + num1) / (num1 + num2)))
        n = num1 * dens1 + 3 + i * dens2
        sizes[i + num1] = n

        # generate matrices
        # A, IalphA, alpha = random_cov(n)
        A, IalphA, alpha = generatePDmatrix(n=n, SPD=True)
        b = np.random.randint(-5, 5, (n, 1))
        x0 = np.zeros((n, 1))

        (
            rtd[i + num1],
            itd[i + num1],
            rich_it[i + num1],
            accurate,
        ) = quickTesting(n, A, IalphA, alpha, b, x0)
        if not accurate:
            err.append(i + num1)

    plt.figure()
    plt.semilogy(sizes, rtd, "b.", label="Richardson's Iteration")
    plt.plot(sizes[err], rtd[err], "bx")
    plt.semilogy(sizes, itd, "r.", label="Direct Inversion")
    plt.title("positive semi-definite matrix", style="italic")
    plt.suptitle("Computational time for Richardson's Iteration vs. Direct Inversion")
    plt.xlabel("Size, $n$", fontsize=14)
    plt.ylabel("Time to compute, $seconds$", fontsize=14)
    plt.legend()

    plt.figure()
    plt.semilogy(sizes, rich_it, "b.", label="Richardson's Iteration")
    plt.semilogy(sizes[err], rich_it[err], "bx")
    plt.title("positive semi-definite matrix", style="italic")
    plt.suptitle("Iterations for Richardson's Iteration to converge")
    plt.xlabel("Size, $n$", fontsize=14)
    plt.ylabel("Iterations", fontsize=14)
    plt.legend()
    plt.show()
    return


def quickTesting(n, A, IalphA, alpha, b, x0, verbose=False):
    """
    Method for Testing differnce in performance\n
    Parameters:
        int n: size of matrices, \n
        np.matrix A: matrix of size n*n,\n
        np.matrix IalphA: I-alpha*A, \n
        np.matrix Ialphb: alpha*b, \n
        float alpha: alpha value for Richardson's iteration,
        np.matrix b: b vector, \n
        np.matrix x0: initial guess, \n
        bool verbose = if True, will print output including error\n
    Returns:
        float rtd: Richardson's Iteration time taken,\n
        float itd: Direct Inverse time taken, \n
    """

    # Richardson's Iteration
    if verbose:
        print("Running Richardson's Iteration:")
    rt0 = time.time()
    rich_sol, rich_it = richardsonIteration(
        A,
        IalphA,
        np.multiply(alpha, b),
        x0,
        n,
        verbose=verbose,
    )
    rt1 = time.time()
    rtd = rt1 - rt0
    if verbose:
        print("Error:", np.linalg.norm(np.matmul(A, rich_sol) - b))
        print("Time Taken:", rtd, "\n")

        # Actual Solution
        print("Running Actual Solution:")
    it0 = time.time()
    ex_sol = np.matmul(np.linalg.inv(A), b)
    it1 = time.time()
    itd = it1 - it0
    err = np.linalg.norm(abs(rich_sol) - abs(ex_sol))
    accurate = err < 1e-3
    print(rich_it, accurate)
    if verbose:
        print("Error:", np.linalg.norm(np.matmul(A, ex_sol) - b))
        print("Time Taken:", itd)
    return rtd, itd, rich_it, accurate


def rateOfConvergence(iterations, b, plot=False):
    """
    Method for testing rate of convergence. Same as spectral norm of A\n
    Parameters:
        np.matrix iterations: as returned by richardson's iteration,\n
        np.matrix b: b vector, \n
        bool plot = if True, will plot output on pyplot graph,\n
    """

    # finding norms of ||x_k-x*||
    norms = np.zeros(len(iterations[1]))
    for i, x in enumerate(iterations[1]):
        diff = np.subtract(np.reshape(x, (3, 1)), b)
        norms[i] = np.linalg.norm(diff)

    # order of convergence testing
    print("Running Order of Convergence:")
    print("->formula: ||x_k+1 - x*|| / ||x_k - x*||")
    for i in range(len(norms) - 1):
        print("k={} ->".format(i), norms[i + 1] / norms[i])

    if plot:
        # plot if desired
        plt.semilogy(norms, label="richardson's iteration")
        plt.xlabel("Number of iteration, $k$", fontsize=20)
        plt.ylabel("Residual norm, $\|Ax_k - b\|_2$", fontsize=20)
        plt.show()
    return


def richardsonIteration(
    A, IalphA, alphb, x0, n, tol=1e-05, Nmax=5000, quick=True, verbose=False
):
    """
    Method for running Richardson's Iteration. \n
    Parameters:
        np.matrix A: matrix of size n*n,\n
        np.matrix IalphA: I-alpha*A, \n
        np.matrix Ialphb: alpha*b, \n
        int n: size of matrices, \n
        float tol: stopping tolerance, \n
        bool quick = optimizes performance, will only return solution,\n
        bool verbose = if True, will print output including convergence\n
    Returns:
        np.matrix xstar: solution,\n
        bool ier: iteration error, \n
        string msg: iteration message, \n
        np.matrix iteration: 2dimensional -> iteration[0][k] = x_k, iteration[1][k] = A*x_k,\n
    """
    # quick iteration without extras
    if quick:
        for i in range(1, Nmax):
            x1 = np.add(np.matmul(IalphA, x0), alphb)
            if np.linalg.norm(x1 - x0) < tol:
                return x1, i
            x0 = x1
        return x1, Nmax

    # initial variable definition
    n = len(IalphA)
    it_err = 1
    msg = "No solution found"
    xk, xk1 = x0, x0
    iterations = np.zeros((2, Nmax, n))
    iterations[0][0] = np.transpose(x0)
    iterations[1][0] = np.transpose(np.matmul(A, xk1))

    # iteration
    j = 0
    for i in range(1, Nmax):
        xk1 = np.add(np.matmul(IalphA, xk), alphb)
        iterations[0][i] = np.transpose(xk1)
        iterations[1][i] = np.transpose(np.matmul(A, xk1))
        if np.linalg.norm(xk1 - xk) < tol:
            j = i
            it_err = 0
            msg = "solution found after " + str(i) + " iterations."
            break
        xk = xk1

    # formating iteration matrix before returning
    it_final = np.zeros((2, j + 1, n))
    it_final[0] = iterations[0][: j + 1]
    it_final[1] = iterations[1][: j + 1]
    return xk1, it_err, msg, it_final


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
        np.matrix IalphA: I-alpha*A matrix, \n
        float alpha: alpha value for richardson's, \n
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

    # calculate alpha and ||I-alpha*A|| norm
    eig_vals, _ = np.linalg.eigh(A)
    ind1 = np.argmax(abs(eig_vals))
    ind2 = np.argmin(abs(eig_vals))
    eig_val1 = eig_vals[ind1].real
    eig_val2 = eig_vals[ind2].real
    # print(eig_val1, eig_val2)
    alpha = 2 / (eig_val1 + eig_val2)
    convergence_mat = np.identity(n) - np.multiply(alpha, A)

    # print if desired before returning
    if verbose == True:
        # print("matrix:\n", A, "\nEigenvalues:\n", eig_vals, "\n")
        # print("Convergence (norm of I-alpha*A):", convergence)
        # print("->by formula:", 1 - 2 * eig_vals[0] / (eig_vals[0] + eig_vals[-1]))
        print(
            "Testing on Symmetric matrix of size",
            n,
            "| Sparse =",
            sparse,
            "| Semi-positive definite:",
            SPD,
        )
        print(
            "Convergence (norm of I-alpha*A):",
            1 - 2 * eig_vals[0] / (eig_vals[0] + eig_vals[-1]),
            "\n",
        )

    return A, convergence_mat, alpha


def random_cov(n):
    Q = np.random.random((n, n))

    eigen_mean = n
    Qt = np.transpose(Q)
    A = np.abs(eigen_mean + np.random.random((n, 1)))
    A = np.diag(A.flatten())
    A = np.matmul(Qt, A)
    A = np.matmul(A, Q)

    eig_vals, _ = np.linalg.eigh(A)
    ind1 = np.argmax(abs(eig_vals))
    ind2 = np.argmin(abs(eig_vals))
    eig_val1 = eig_vals[ind1].real
    eig_val2 = eig_vals[ind2].real
    # print(eig_val1, eig_val2)
    alpha = 2 / (eig_val1 + eig_val2)
    convergence_mat = np.eye(n) - np.multiply(alpha, A)

    return A, convergence_mat, alpha


if __name__ == "__main__":
    print("\n")
    driver()
    print("\n")
