import numpy as np
import scipy
import matplotlib.pyplot as plt
import time


def driver():
    # testing on various parameters. outputs plots. Included all of them in outputs folder
    to = time.time()
    num_avg = 10
    num1, dens1 = 100, 10
    num2, dens2 = 50, 20
    tot_sizes = np.zeros((num_avg, num1 + num2))
    tot_rtd = np.zeros((num_avg, num1 + num2))
    tot_itd = np.zeros((num_avg, num1 + num2))
    tot_rich_it = np.zeros((num_avg, num1 + num2))

    for j in range(num_avg):
        print(j + 1, time.time() - to)
        sizes = np.zeros(num1 + num2)
        rtd = np.zeros(num1 + num2)
        itd = np.zeros(num1 + num2)
        rich_it = np.zeros(num1 + num2)

        for i in range(num1):
            # size of matrix
            if i % (num1 / 5) == 0:
                print("{:.0%}".format(i / (num1 + num2)))
            n = i * dens1 + 5
            sizes[i] = n

            # generate matrices
            # A, IalphA, alpha = random_cov(n)
            A, IalphA, alpha = generatePDmatrix(n=n, cond_100=True)
            b = np.random.randint(-5, 5, (n, 1))
            x0 = np.zeros((n, 1))

            # output run-time
            rtd[i], itd[i], rich_it[i], _ = quickTesting(n, A, IalphA, alpha, b, x0)

        for i in range(num2):
            # size of matrix
            if i % (num2 / 5) == 0:
                print("{:.0%}".format((i + num1) / (num1 + num2)))
            n = num1 * dens1 + 3 + i * dens2
            sizes[i + num1] = n

            # generate matrices
            # A, IalphA, alpha = random_cov(n)
            A, IalphA, alpha = generatePDmatrix(n=n, cond_100=True)
            b = np.random.randint(-5, 5, (n, 1))
            x0 = np.zeros((n, 1))

            (
                rtd[i + num1],
                itd[i + num1],
                rich_it[i + num1],
                _,
            ) = quickTesting(n, A, IalphA, alpha, b, x0)

        tot_sizes[j] = sizes
        tot_rtd[j] = rtd
        tot_itd[j] = itd
        tot_rich_it[j] = rich_it
    avg_size = np.mean(tot_sizes, axis=0)
    avg_rtd = np.mean(tot_rtd, axis=0)
    avg_itd = np.mean(tot_itd, axis=0)
    avg_rich_it = np.mean(tot_rich_it, axis=0)

    plt.figure()
    plt.semilogy(avg_size, avg_rtd, "b.", label="Richardson's Iteration")
    # plt.plot(sizes[err], rtd[err], "bx")
    plt.semilogy(avg_size, avg_itd, "r.", label="Direct Inversion")
    plt.title("positive definite, $\kappa(A)=100$", style="italic")
    plt.suptitle("Computational time for Richardson's Iteration vs. Direct Inversion")
    plt.xlabel("Size, $n$", fontsize=14)
    plt.ylabel("Time to compute, $seconds$", fontsize=14)
    plt.legend()

    plt.figure()
    plt.plot(avg_size, avg_rich_it, "b.", label="Richardson's Iteration")
    # plt.plot(sizes[err], rich_it[err], "bx")
    plt.title("positive definite, $\kappa(A)=100$", style="italic")
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
    if verbose:
        print("Error:", np.linalg.norm(np.matmul(A, ex_sol) - b))
        print("Time Taken:", itd)
    return rtd, itd, rich_it, accurate


def richardsonIteration(
    A, IalphA, alphb, x0, n, tol=5e-06, Nmax=5000, quick=True, verbose=False
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


def generatePDmatrix(n=3, sparse=False, cond_one=False, cond_100=False, tri=False):
    """
    References:
        For generating random PD matrices:
            @Daryl on stack exchange - translated code from matlab
            link - https://math.stackexchange.com/questions/357980/how-to-generate-random-symmetric-positive-definite-matrices-using-matlab
    """

    # generate a positive definite matrix
    if tri:
        # condition number of 10
        h = int(0.81 * n)
        eig = np.linspace(n - h, n + h, n)
        np.random.shuffle(eig)
        D = np.diag(eig)
        k = 2
        k1 = np.random.random((k, n))
        k2 = np.arange(0, k, 1)
        sp = scipy.sparse.diags(k1, k2)
        A = sp.toarray()
        A = np.triu(A + D)
        A = A + np.transpose(A) - np.diag(np.diag(A))
    elif sparse:
        # condition number of 10
        h = int(0.81 * n)
        eig = np.linspace(n - h, n + h, n)
        np.random.shuffle(eig)
        D = np.diag(eig)
        A = scipy.sparse.random(n, n, 0.25)
        A = A.A
        A = np.triu(A + D)
        A = A + np.transpose(A) - np.diag(np.diag(A))
        # h = c_num * (n - 1) / (1 + c_num)
    elif cond_one:
        eig = np.array([n] * (n))
        A = generate_matrix_with_eigenvalues(n, eig)
    elif cond_100:
        h = int(0.98 * n)
        eig = np.linspace(n - h, n + h, n)
        A = generate_matrix_with_eigenvalues(n, eig)
    else:
        # condition number 10
        h = int(0.81 * n)
        eig = np.linspace(n - h, n + h, n)
        A = generate_matrix_with_eigenvalues(n, eig)

    # calculate alpha and ||I-alpha*A|| norm
    eig_vals, _ = np.linalg.eigh(A)
    eig_vals = np.sort(eig_vals)
    eig_val1 = eig_vals[0].real
    eig_val2 = eig_vals[-1].real
    alpha = 2 / (eig_val1 + eig_val2)
    print(n, "condition:", eig_val2 / eig_val1)
    convergence_mat = np.identity(n) - np.multiply(alpha, A)

    return A, convergence_mat, alpha


def generate_matrix_with_eigenvalues(n, eigenvalues):
    Q, _ = np.linalg.qr(np.random.rand(n, n))
    D = np.diag(eigenvalues)
    A = np.matmul(np.matmul(Q, D), np.linalg.inv(Q))
    return A


if __name__ == "__main__":
    print("\n")
    driver()
    print("\n")
