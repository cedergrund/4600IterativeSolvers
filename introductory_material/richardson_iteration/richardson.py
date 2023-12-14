import numpy as np
import scipy
import matplotlib.pyplot as plt
import time


def driver():
    # testing on various parameters. outputs plots. Included all of them in outputs folder
    num_avg = 5
    n_eval = 25
    n = 2000

    x_dense_part = np.logspace(0, 4, num=12, base=10)
    x_dense_part = x_dense_part[x_dense_part != 1e4]
    n_eval -= 1
    x_sparse_part = np.logspace(4, 20, num=n_eval - len(x_dense_part), base=10)
    x_values = np.concatenate([x_dense_part, x_sparse_part])

    tot_real_sizes = np.zeros((num_avg, n_eval))
    tot_err_rich = np.zeros((num_avg, n_eval))
    tot_err_inv = np.zeros((num_avg, n_eval))
    tot_rtd = np.zeros((num_avg, n_eval))
    tot_itd = np.zeros((num_avg, n_eval))
    tot_rich_it = np.zeros((num_avg, n_eval))
    errors = np.ones((2, n_eval))

    # generatePDmatrix(n, cond=True, c_num=1e15)

    to = time.time()
    for i in range(num_avg):
        print("{}%".format(i / num_avg), time.time() - to)
        real_sizes = np.zeros(n_eval)
        err_rich = np.zeros(n_eval)
        err_inv = np.zeros(n_eval)
        rtd = np.zeros(n_eval)
        itd = np.zeros(n_eval)
        rich_it = np.zeros(n_eval)
        rich_it = np.zeros(n_eval)
        for j, c_num in enumerate(x_values):
            print(
                "round: {}. {}/{}. condition num ->".format(i + 1, j + 1, n_eval), c_num
            )
            A, alpha, b, real_sizes[j] = generatePDmatrix(n, cond=True, c_num=c_num)
            IalphA = np.eye(n) - np.multiply(alpha, A)
            alphb = np.multiply(alpha, b)
            rtd[j], itd[j], rich_it[j], e, accurate = quickTesting(
                n, A, b, IalphA, alphb
            )
            err_rich[j], err_inv[j] = e
            if accurate[0]:
                errors[0][j] = 0
            if accurate[1]:
                errors[1][j] = 0

        tot_real_sizes[i] = real_sizes
        tot_err_rich[i] = err_rich
        tot_err_inv[i] = err_inv
        tot_rtd[i] = rtd
        tot_itd[i] = itd
        tot_rich_it[i] = rich_it
        print()

    avg_size = np.mean(tot_real_sizes, axis=0)
    indices_sorted = np.argsort(avg_size)
    avg_size = avg_size[indices_sorted]
    avg_err_rich = np.mean(tot_err_rich, axis=0)
    avg_err_inv = np.mean(tot_err_inv, axis=0)
    avg_rtd = np.mean(tot_rtd, axis=0)
    avg_itd = np.mean(tot_itd, axis=0)
    avg_rich_it = np.mean(tot_rich_it, axis=0)

    errs_rich = np.where(errors[0] == 1)
    errs_inv = np.where(errors[1] == 1)

    plt.figure()
    plt.loglog(x_values, avg_rtd, "b.-", label="Richardson's Iteration")
    plt.loglog(x_values[errs_rich], avg_rtd[errs_rich], "bx")
    plt.loglog(x_values, avg_itd, "r.-", label="Direct Inversion")
    plt.loglog(x_values[errs_inv], avg_itd[errs_inv], "rx")
    plt.title("positive definite matrices of size 2000", style="italic")
    plt.suptitle("Computational time for Richardson's Iteration vs. Direct Inversion")
    plt.xlabel("Condition Number, $\kappa(A)$", fontsize=14)
    plt.ylabel("Time to compute, $seconds$", fontsize=14)
    plt.legend()

    plt.figure()
    x = np.linspace(x_values[0], x_values[-1], 1000)
    f = lambda x: 30000
    plt.loglog(x, list(map(f, x)), "k--", label="max_iterations")
    plt.loglog(x_values, avg_rich_it, "b.-", label="Richardson's Iteration")
    plt.loglog(x_values[errs_rich], avg_rich_it[errs_rich], "bx")
    plt.title("positive definite matrices of size 2000", style="italic")
    plt.suptitle("Iterations for Convergence of Richardson's Iteration")
    plt.xlabel("Condition Number, $\kappa(A)$", fontsize=14)
    plt.ylabel("Iterations", fontsize=14)
    plt.legend()

    plt.figure()
    plt.loglog(x_values, avg_err_rich, "b.-", label="Richardson's Iteration")
    plt.loglog(x_values, avg_err_inv, "r.-", label="Direct Inversion")
    plt.title("positive definite matrices of size 2000", style="italic")
    plt.suptitle("Error of Richardson's Iteration and Direct Inversion")
    plt.xlabel("Size, $n$", fontsize=14)
    plt.ylabel("Error, $||Ax-b||$", fontsize=14)
    plt.legend()
    plt.show()

    # # double plot
    # fig, ax1 = plt.subplots()
    # plt.title("positive definite matrices of size 2000", style="italic")
    # plt.suptitle("Iterations and Error for Richardson's Iteration")
    # ax1.set_xlabel("Condition Number, $\kappa(\mathbf{A})$", fontsize=14)
    # ax1.set_ylabel("Iterations", fontsize=14, color="red")
    # ax1.loglog(x_values, avg_rich_it, "r.--")
    # ax1.loglog(x_values[errs_rich], avg_rich_it[errs_rich], "rx")
    # ax1.tick_params(axis="y", labelcolor="red")

    # ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

    # ax2.set_ylabel("Error", fontsize=14, color="blue")
    # # , $\norm{Ax-b}$
    # ax2.semilogy(x_values, avg_err_rich, "b.--")
    # ax2.tick_params(axis="y", labelcolor="blue")

    # fig.tight_layout()  # otherwise the right y-label is slightly clipped
    # plt.show()

    return


def allSizes():
    # testing on various parameters. outputs plots. Included all of them in outputs folder
    to = time.time()
    num_avg = 5

    # start = 205
    # num1, dens1 = 67, 15
    # num2, dens2 = 43, 30
    start = 50
    num1, dens1 = 100, 10
    num2, dens2 = 50, 20
    # num1, dens1 = 50, 10
    # num2, dens2 = 20, 20
    tot_sizes = np.zeros((num_avg, num1 + num2))
    tot_rtd = np.zeros((num_avg, num1 + num2))
    tot_itd = np.zeros((num_avg, num1 + num2))
    tot_rich_it = np.zeros((num_avg, num1 + num2))
    tot_rich_err = np.zeros((num_avg, num1 + num2))

    for j in range(num_avg):
        print("{}%".format(j / num_avg), time.time() - to)
        print(j + 1, time.time() - to)
        sizes = np.zeros(num1 + num2)
        rtd = np.zeros(num1 + num2)
        itd = np.zeros(num1 + num2)
        rich_it = np.zeros(num1 + num2)
        rich_err = np.zeros(num1 + num2)

        for i in range(num1):
            # size of matrix
            if i % (num1 / 5) == 0:
                print("{:.0%}".format(i / (num1 + num2)))
            n = start + i * dens1
            sizes[i] = n

            # generate matrices
            A, alpha, b, _ = generatePDmatrix(n)
            IalphA = np.eye(n) - np.multiply(alpha, A)
            alphb = np.multiply(alpha, b)

            # output run-time
            rtd[i], itd[i], rich_it[i], e, _ = quickTesting(n, A, b, IalphA, alphb)
            rich_err[i] = e[0]

        for i in range(num2):
            # size of matrix
            if i % (num2 / 5) == 0:
                print("{:.0%}".format((i + num1) / (num1 + num2)))
            n = start + num1 * dens1 + i * dens2
            sizes[i + num1] = n

            # generate matrices
            A, alpha, b, _ = generatePDmatrix(n)
            IalphA = np.eye(n) - np.multiply(alpha, A)
            alphb = np.multiply(alpha, b)

            # output run-time
            rtd[i + num1], itd[i + num1], rich_it[i + num1], e, _ = quickTesting(
                n, A, b, IalphA, alphb
            )
            rich_err[i + num1] = e[0]

        tot_sizes[j] = sizes
        tot_rtd[j] = rtd
        tot_itd[j] = itd
        tot_rich_it[j] = rich_it
        tot_rich_err[j] = rich_err
        print()
    avg_size = np.mean(tot_sizes, axis=0)
    avg_rtd = np.mean(tot_rtd, axis=0)
    avg_itd = np.mean(tot_itd, axis=0)
    avg_rich_it = np.mean(tot_rich_it, axis=0)
    avg_rich_err = np.mean(tot_rich_err, axis=0)

    plt.figure()
    plt.semilogy(avg_size, avg_rtd, "b.", label="Richardson's Iteration")
    # plt.plot(sizes[err], rtd[err], "bx")
    plt.semilogy(avg_size, avg_itd, "r.", label="Direct Inversion")
    plt.title("random positive definite matrix, $\kappa(A)=10$", style="italic")
    plt.suptitle("Computational time for Richardson's Iteration vs. Direct Inversion")
    plt.xlabel("Size, $n$", fontsize=14)
    plt.ylabel("Time to compute, $seconds$", fontsize=14)
    plt.legend()

    plt.figure()
    plt.plot(avg_size, avg_rich_it, "b.", label="Richardson's Iteration")
    # plt.plot(sizes[err], rich_it[err], "bx")
    plt.title("random positive definite matrix, $\kappa(A)=10$", style="italic")
    plt.suptitle("Iterations for Richardson's Iteration to converge")
    plt.xlabel("Size, $n$", fontsize=14)
    plt.ylabel("Iterations", fontsize=14)
    plt.legend()

    # double plot
    fig, ax1 = plt.subplots()
    plt.title("positive definite matrices of size 2000", style="italic")
    plt.suptitle("Iterations and Error for Richardson's Iteration")
    ax1.set_xlabel("Size, $n$", fontsize=14)
    ax1.set_ylabel("Error", fontsize=14, color="blue")
    # , $\norm{Ax-b}$
    ax1.semilogy(avg_size, avg_rich_err, "b.--")
    ax1.tick_params(axis="y", labelcolor="blue")

    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
    ax2.set_ylabel("Iterations", fontsize=14, color="red")
    ax2.plot(avg_size, avg_rich_it, "r.")
    ax2.tick_params(axis="y", labelcolor="red")

    fig.tight_layout()  # otherwise the right y-label is slightly clipped
    plt.show()
    return


def quickTesting(n, A, b, IalphA, alphb):
    # Richardson's Iteration
    rt0 = time.time()
    rich_sol, rich_it = richardsonIteration(A, n, IalphA, alphb)
    rt1 = time.time()
    rtd = rt1 - rt0

    # direct solving
    it0 = time.time()
    ex_sol = np.matmul(np.linalg.inv(A), b)
    it1 = time.time()
    itd = it1 - it0

    err_rich = np.linalg.norm(np.matmul(A, rich_sol) - b)
    err_inv = np.linalg.norm(np.matmul(A, ex_sol) - b)
    err = [err_rich, err_inv]
    accurate = [err_rich < 1e-2, err_inv < 1e-2]
    print(rich_it, err_rich)
    if not accurate[0]:
        print("richardsons fail!", err[0], "in {} iterations".format(rich_it))
    if not accurate[1]:
        print("inverse fail!", err[1])

    return rtd, itd, rich_it, err, accurate


def richardsonIteration(_, n, IalphA, alphb, tol=5e-08, Nmax=30000):
    x0 = np.zeros((n, 1))
    for i in range(1, Nmax):
        x1 = np.add(np.matmul(IalphA, x0), alphb)
        if np.linalg.norm(x1 - x0) < tol:
            return x1, i
        x0 = x1
    return x1, Nmax


def generatePDmatrix(n, sparse=False, cond=False, c_num=1, tri=False):
    """
    References:
        For generating random PD matrices:
            @Daryl on stack exchange - translated code from matlab
            link - https://math.stackexchange.com/questions/357980/how-to-generate-random-symmetric-positive-definite-matrices-using-matlab
    """
    if sparse:
        h = 9 / 11 * n  # condition number 10
        eig = np.linspace(n - h, n + h, n)
        np.random.shuffle(eig)
        D = np.diag(eig)
        if tri:  # create random tridiagonal matrix
            k = 2
            k1, k2 = np.random.random((k, n)), np.arange(0, k, 1)
            sp = scipy.sparse.diags(k1, k2)
            A = sp.toarray()
        else:  # sparse matrix, density 0.25
            A = scipy.sparse.random(n, n, 0.25)
            A = A.A
        A = np.triu(A + D)
        A = A + np.transpose(A) - np.diag(np.diag(A))
    elif cond:  # explicit condition number
        # define h st condition number will be c_num after linearly spacing eigenvalues bw n-h and n+h
        b = n * (1 - ((c_num - 1) / (1 + c_num)))
        t = n * (1 + (c_num - 1) / (1 + c_num))
        eig = np.linspace(b, t, n)
        np.random.shuffle(eig)
        A = generate_matrix_with_eigenvalues(n, np.abs(eig))
    else:  # random positive definite matrix with condition number 10
        eig = np.linspace(n / 10, n, n)  # condition number 10
        # eig = np.linspace(10, 100, n)
        np.random.shuffle(eig)
        A = generate_matrix_with_eigenvalues(n, eig)

    # calculate alpha and b vector
    eig_vals = np.sort(np.linalg.eigh(A)[0])
    eig_val1, eig_val2 = eig_vals[0].real, eig_vals[-1].real
    alpha = 2 / (eig_val1 + eig_val2)
    cond = eig_val2 / eig_val1
    b = np.random.rand(n, 1)
    return A, alpha, b, cond


def generate_matrix_with_eigenvalues(n, eigenvalues):
    Q, _ = np.linalg.qr(np.random.rand(n, n))
    D = np.diag(eigenvalues)
    A = np.matmul(np.matmul(Q, D), np.linalg.inv(Q))
    return A


def richardsonIterationOld(
    A, IalphA, alphb, x0, n, tol=1e-10, Nmax=5000, quick=True, verbose=False
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


if __name__ == "__main__":
    allSizes()
    # print("\n")
    # driver()
    # print("\n")
