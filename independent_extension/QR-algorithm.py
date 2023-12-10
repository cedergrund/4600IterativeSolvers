import numpy as np
import time
import scipy
import matplotlib.pyplot as plt


def driver():
    num_avg = 20
    t0 = time.time()
    n = 50
    iterations_sym = np.zeros((num_avg, 3))
    times_sym = np.zeros((num_avg, 4))
    error_sym = np.zeros((num_avg, 3))
    converged_sym = np.zeros(3)

    for i in range(num_avg):
        print("{}%".format(i / num_avg), time.time() - t0)
        A = createMatrix(n, spectral=True, nums=[1e15, 0])
        iterations_sym[i], times_sym[i], error_sym[i], c = testOne(
            A, n, [True, True, True]
        )
        converged_sym += c
        print()

    avg_it = np.mean(iterations_sym, axis=0)
    avg_t = np.mean(times_sym, axis=0)
    avg_err = np.mean(error_sym, axis=0)

    print("iterations", avg_it)
    print("times", avg_t)
    print("avg_error", avg_err)
    print("converged %", converged_sym / num_avg * 100)
    print()


def testOne(A, n, run):
    iterations = np.zeros(3)
    times = np.zeros(4)
    error = np.zeros(3)
    converged = np.zeros(3)

    if run[0]:
        t0 = time.time()
        e_pure, iterations[0] = simultaneousQR(A, n, max_iter=20000)
        t1 = time.time()
        times[0] = t1 - t0
        e_pure = np.sort(e_pure)
        print("done1", times[0], iterations[0])

    if run[1]:
        t0 = time.time()
        e_sim, iterations[1] = lazyShiftQR(A, n, max_iter=20000)
        t1 = time.time()
        times[1] = t1 - t0
        e_sim = np.sort(e_sim)
        print("done2", times[1], iterations[1])

    if run[2]:
        t0 = time.time()
        e_comp, iterations[2] = complex_shiftedQR(A, n, max_iter=20000)
        t1 = time.time()
        times[2] = t1 - t0
        e_comp = np.sort(e_comp)
        print("done3", times[2], iterations[2])

    t0 = time.time()
    e_real = np.sort(np.linalg.eig(A)[0])
    t1 = time.time()
    times[3] = t1 - t0

    if run[0]:
        error[0] = np.linalg.norm(e_pure - e_real)
        if error[0] < 1e-3:
            converged[0] = 1
        else:
            print("0 wrong", error[0])
    if run[1]:
        error[1] = np.linalg.norm(e_sim - e_real)
        if error[1] < 1e-3:
            converged[1] = 1
        else:
            print("1 wrong", error[1])
    if run[2]:
        if np.shape(e_comp) != np.shape(e_real):
            print("oops")
            print(len(e_comp) / len(e_real))
        else:
            error[2] = np.linalg.norm(e_comp - e_real)
            if error[2] < 1e-3:
                converged[2] = 1
            else:
                print("2 wrong", error[2])

    return iterations, times, error, converged


def createMatrix(
    n,
    neg=False,
    sym=False,
    eq_spaced=False,
    neq_spaced=False,
    spectral=False,
    ill_cond=False,
    nums=[1, 1, 1],
    distinct=False,
):
    # positive or positive and negative eigenvalues
    sign = np.ones(n)
    if neg:
        sign = np.random.choice([-1, 1], size=n)
    if sym:  # symmetric
        A = np.random.rand(n, n)
        A = np.multiply(1 / 2, np.add(A, np.transpose(A)))
    elif eq_spaced:  # equally spaced from a to b
        eig = np.linspace(nums[0], nums[1], n)
        eig = np.multiply(eig, sign)
        A = generate_matrix_with_eigenvalues(n, eig)
    elif neq_spaced:  # not equally spaced. factor between successive eigs
        f = nums[0]
        if f >= 1:
            eig = 1e-5 * f ** np.arange(n)
        else:
            eig = 1e5 * f ** np.arange(n)
        eig = np.multiply(eig, sign)
        A = generate_matrix_with_eigenvalues(n, eig)
    elif distinct:  # 3 or 1 distinct eigenvalues
        if nums[1] == 0:
            eig = np.multiply(nums[0], np.ones(n))
        else:
            eig = np.random.choice(nums, size=n)
        eig = np.multiply(eig, sign)
        A = generate_matrix_with_eigenvalues(n, eig)
    elif spectral:  # spectral radius r at x
        r = nums[0]
        x = nums[1]
        eig = np.random.uniform(x - r, x + r, size=n - 1)
        res = np.random.choice([1, -1]) * r
        eig = np.append(eig, x + res)
        A = generate_matrix_with_eigenvalues(n, eig)
    elif ill_cond:
        eig = np.linspace(1e-10, 1e10, n)
        eig = np.multiply(eig, sign)
        A = generate_matrix_with_eigenvalues(n, eig)
    else:
        A = np.random.rand(n, n)
    return A


def generate_matrix_with_eigenvalues(n, eigenvalues):
    Q, _ = np.linalg.qr(np.random.rand(n, n))
    D = np.diag(eigenvalues)
    A = np.matmul(np.matmul(Q, D), np.linalg.inv(Q))
    return A


def pureQR(A, n, max_iter=1000, tol=1e-8):
    Xk = A.copy()
    count = max_iter
    e1 = np.zeros(n)

    for i in range(max_iter):
        e0 = e1.copy()
        Q, R = np.linalg.qr(Xk)
        Xk = np.matmul(R, Q)
        e1 = np.diag(Xk)
        if np.all(np.abs(e1 - e0) < tol):
            count = i + 1
            break

    eigenvalues = np.diag(Xk)

    return eigenvalues, count


def simultaneousQR(A, n, max_iter=1000, tol=1e-7):
    X0 = np.eye(n)
    e1 = np.zeros(n)
    count = max_iter
    for i in range(max_iter):
        e0 = e1
        X1 = np.matmul(A, X0)
        Qk, Rk = np.linalg.qr(X1)
        X0 = Qk
        e1 = np.diag(X0)
        if np.all(np.abs(e1 - e0) < tol):
            count = i + 1
            break

    eigen = np.diag(np.matmul(np.matmul(np.transpose(X0), A), X0))
    return eigen, count


def lazyShiftQR(A, n, max_iter=1000, tol=1e-5):
    X0 = A.copy()
    count = 0
    m = n - 1

    # bottom right
    shift = X0[-1, -1]
    shifted_mat = np.multiply(shift, np.eye(n))
    X0 -= shifted_mat
    while abs(X0[m, m - 1]) > tol:
        count += 1
        Q, R = np.linalg.qr(X0)
        X1 = np.matmul(R, Q)
        if count >= max_iter:
            return np.diag(X1), max_iter
        X0 = X1
    X0 += shifted_mat

    # rest
    tol = 1e-7
    e1 = np.zeros(n)
    for i in range(count, max_iter):
        count = i + 1
        e0 = e1.copy()
        Q, R = np.linalg.qr(X0)
        X1 = np.matmul(R, Q)
        e1 = np.diag(X1)
        if np.all(np.abs(e1 - e0) < tol):
            break
        X0 = X1
    return e1, count


def complex_shiftedQR(A, n, max_iter=1000, tol=1e-5):
    X0 = A.copy()
    count = 1
    m = n - 1
    e = np.zeros(n)

    for m in range(n - 1, 0, -1):
        I = np.eye(m + 1)
        while abs(X0[m, m - 1]) > tol:
            count += 1
            shift = X0[m, m]
            shift_mat = np.multiply(shift, I)
            X0 -= shift_mat
            Q, R = np.linalg.qr(X0)
            X1 = np.matmul(R, Q) + shift_mat
            if count > max_iter:
                return np.diag(X1), max_iter
            X0 = X1
        e[m] = X0[m, m]
        X0 = X0[:m, :m]

    e[0] = X0[0, 0]
    return e, count


# A = createMatrix(4, sym=True)
# print(A)
# print()
# A = scipy.linalg.hessenberg(A)
# print(A)
# print()
# pureQR(A, 4, 20)
# e, i = simple_shiftedQR(A, 10, max_iter=20000)
# # e = np.sort(e)
# r, _ = np.linalg.eig(A)
# r = np.sort(r)
# min_absolute_value = np.min(np.abs(r))
# max_absolute_value = np.max(np.abs(r))
# print(np.max(np.abs(r)) / np.min(np.abs(r)))
# A = np.random.rand(100, 100)
# r, _ = np.linalg.eig(A)
# r = np.sort(r)
# min_absolute_value = np.min(np.abs(r))
# max_absolute_value = np.max(np.abs(r))
# print(np.max(np.abs(r)) / np.min(np.abs(r)))
# print(e)
# print(r)
# print(np.linalg.norm(e - r), i)

# A = np.random.rand(n, n)
# A = A + A.T - np.diag(A)
# # A = A + n * np.eye(n)
# r, _ = np.linalg.eig(A)
# r = np.sort(r)
# print(r)
# r2 = np.sort(abs(r))
# tot = 0
# for i in range(len(r2) - 1):
#     tot += r2[i + 1] / r2[i]
# e, i = simultaneousQR(A, n, max_iter=50000)
# e = np.sort(e)
# print(np.linalg.norm(e - r), i)
# e, i = simple_shiftedQR(A, n, max_iter=50000)
# e = np.sort(e)
# print(np.linalg.norm(e - r2), i)
# print(tot / (len(r) - 1), r2[-1] / r2[0])
# # print(r)
# print()

# n = 25
# # print()
# total_m1 = 0
# total_m2 = 0
# for i in range(100):
#     print(i + 1)
#     print("--------")
#     A = createMatrix(n, sym=True)
#     # print(A)
#     # print()

#     r, _ = np.linalg.eig(A)
#     r = np.sort(r)
#     #     # print(r)
#     #     # print()
#     #     # r2 = np.sort(abs(r))
#     #     # tot = 0
#     #     # for i in range(len(r2) - 1):
#     #     #     tot += r2[i + 1] / r2[i]

#     t0 = time.time()
#     e, i = lazyShiftQR(A, n, max_iter=50000)
#     t1 = time.time()
#     e = np.sort(e)
#     print("lazy res:", np.linalg.norm(e - r), i, t1 - t0)
#     total_m1 += i
#     t0 = time.time()
#     e, i = lazyShiftQR2(A, n, max_iter=50000)
#     t1 = time.time()
#     e = np.sort(e)
#     print("lazy converge:", np.linalg.norm(e - r), i, t1 - t0)

#     #     t0 = time.time()
#     #     e, i = lazyShiftQR(A, n, max_iter=10000)
#     #     t1 = time.time()
#     total_m2 += i
#     #     e = np.sort(e)
#     #     print("shift qr mix2:", np.linalg.norm(e - r), i, t1 - t0)
#     #     # # print(tot / (len(r2) - 1), r2[-1] / r2[0])
#     #     # # print(r)
#     #     # print()
#     #     print()
#     t0 = time.time()
#     e, i = simultaneousQR(A, n, max_iter=50000)
#     t1 = time.time()
#     e = np.sort(e)
#     print("simul qr:", np.linalg.norm(e - r), i, t1 - t0)

# #     t0 = time.time()
# #     e, i = pureQR(A, n, max_iter=50000)
# #     t1 = time.time()
# #     e = np.sort(e)
# #     print("pure qr:", np.linalg.norm(e - r), i, t1 - t0)
# #     print()
# #     print()
# print("mix1 avg:", total_m1 / 100)
# print("mix2 avg:", total_m2 / 100)
# # # t0 = time.time()
# # # e, i = complex_shiftedQR(A, n, max_iter=50000)
# # # t1 = time.time()
# # # e = np.sort(e)
# # # print(np.linalg.norm(e - r), i, t1 - t0)
# # # # print()
# # # # print(r)
# # # # e, i = simultaneousQR(A, n, max_iter=50000)
# # # # e = np.sort(e)
# # # # print(np.linalg.norm(e - r), i)
# # # # e, i = simple_shiftedQR(A, n, max_iter=50000)
# # # # e = np.sort(e)
# # # # # print(e)
# # # # print(np.linalg.norm(e - r), i)


def plot1():
    avg_sizes = [3, 10, 100, 10000, 1e16]

    # iterations
    vectors = np.array(
        [
            [913.15, 1059.55, 119.35],
            [664.5, 641.9, 127.5],
            [578.75, 644.65, 135.95],
            [601.55, 882.65, 141.05],
            [732.65, 872.15, 96.85],
        ]
    )

    # Split into three different vectors grouped by element index
    pos_si_iter = vectors[:, 0]
    pos_ls_iter = vectors[:, 1]
    pos_cs_iter = vectors[:, 2]

    vectors_set2 = np.array(
        [
            [1187.7, 1520.85, 71.5],
            [879.65, 666.9, 72.65],
            [757.85, 700.25, 87.5],
            [596.95, 976.6, 142.25],
            [758.6, 715.6, 94.35],
        ]
    )

    # Split into three different vectors grouped by element index
    neg_si_iter = vectors_set2[:, 0]
    neg_ls_iter = vectors_set2[:, 1]
    neg_cs_iter = vectors_set2[:, 2]

    vectors_set3 = np.array(
        [
            [0.09396802, 0.10505583, 0.00637033, 0.00068958],
            [0.07536761, 0.06852865, 0.00706369, 0.00065017],
            [0.06650832, 0.077739, 0.00737373, 0.00070755],
            [0.07123842, 0.09742026, 0.00874407, 0.00099396],
            [0.07438585, 0.08198733, 0.00525943, 0.00062884],
        ]
    )

    # Split into three different vectors grouped by element index
    pos_si_time = vectors_set3[:, 0]
    pos_ls_time = vectors_set3[:, 1]
    pos_cs_time = vectors_set3[:, 2]
    pos_eig_time = vectors_set3[:, 3]

    vectors_set4 = np.array(
        [
            [0.13672819, 0.15322791, 0.00373062, 0.00056467],
            [0.10225344, 0.07384119, 0.0063252, 0.00069571],
            [0.08485484, 0.07603853, 0.00553371, 0.00065397],
            [0.06887645, 0.09767246, 0.00763993, 0.00070417],
            [0.09150424, 0.08132341, 0.00558608, 0.000711],
        ]
    )

    # Split into four different vectors grouped by element index
    neg_si_time = vectors_set4[:, 0]
    neg_ls_time = vectors_set4[:, 1]
    neg_cs_time = vectors_set4[:, 2]
    neg_eig_time = vectors_set4[:, 3]

    plt.figure()
    plt.loglog(avg_sizes, pos_si_iter, "b.-", label="$+\lambda$ Sim. Iteration")
    plt.loglog(avg_sizes, neg_si_iter, "b.--", label="$\pm\lambda$ Sim. Iteration")
    plt.loglog(avg_sizes, pos_ls_iter, "r.-", label="$+\lambda$ Lazy Shift")
    plt.loglog(avg_sizes, neg_ls_iter, "r.--", label="$\pm\lambda$ Lazy Shift")
    plt.loglog(avg_sizes, pos_cs_iter, "g.-", label="$+\lambda$ Complex Shift")
    plt.loglog(avg_sizes, neg_cs_iter, "g.--", label="$\pm\lambda$ Complex Shift")
    plt.title("symmetric matrix, $n=50$", style="italic")
    plt.suptitle("Iterations to compute Eigenvalues")
    plt.xlabel("Condition number, $\kappa(A)$", fontsize=14)
    plt.ylabel("Num. Iterations", fontsize=14)
    plt.legend()

    plt.figure()
    plt.loglog(avg_sizes, pos_si_time, "b.-", label="$+\lambda$ Sim. Iteration")
    plt.loglog(avg_sizes, neg_si_time, "b.--", label="$\pm\lambda$ Sim. Iteration")
    plt.loglog(avg_sizes, pos_ls_time, "r.-", label="$+\lambda$ Lazy Shift")
    plt.loglog(avg_sizes, neg_ls_time, "r.--", label="$\pm\lambda$ Lazy Shift")
    plt.loglog(avg_sizes, pos_cs_time, "g.-", label="$+\lambda$ Complex Shift")
    plt.loglog(avg_sizes, neg_cs_time, "g.--", label="$\pm\lambda$ Complex Shift")
    plt.loglog(avg_sizes, pos_eig_time, "k.-", label="$+\lambda$ np.eig()")
    plt.loglog(avg_sizes, neg_eig_time, "k.-", label="$\pm\lambda$ np.eig()")
    plt.title("symmetric matrix, $n=50$", style="italic")
    plt.suptitle("Time to compute Eigenvalues")
    plt.xlabel("Condition number, $\kappa(A)$", fontsize=14)
    plt.ylabel("Time, seconds", fontsize=14)
    plt.legend()
    plt.show()
    return


def plot2():
    avg_sizes = [1e-3, 1, 10, 1e2, 1e3, 1e5, 1e10, 1e15]
    # iterations
    vectors = np.array(
        [
            [14872.75, 1482.1, 37.35],
            [15416.35, 5690.15, 71.6],
            [16358.8, 7661, 77.6],
            [13422.7, 7719.25, 81.5],
            [17682.6, 11900.15, 89.4],
            [18724.2, 13768.85, 99.3],
            [20000, 15952.2, 114.45],
            [20000, 20000, 20000],
        ]
    )

    # Split into three different vectors grouped by element index
    pos_si_iter = vectors[:, 0]
    pos_ls_iter = vectors[:, 1]
    pos_cs_iter = vectors[:, 2]

    vectors_set3 = np.array(
        [
            [1.31866004e00, 1.12409818e-01, 1.88260078e-03, 6.73365593e-04],
            [1.36990484e00, 4.28178716e-01, 3.29217911e-03, 5.81121445e-04],
            [1.41665795e00, 5.36195564e-01, 3.58897448e-03, 5.84304333e-04],
            [1.18583127e00, 5.43043637e-01, 3.75463963e-03, 6.87313080e-04],
            [1.54700093e00, 8.32814109e-01, 4.01214361e-03, 5.83994389e-04],
            [1.72730248e00, 9.73768795e-01, 4.40195799e-03, 5.68020344e-04],
            [1.75016188621521, 1.09930414e00, 4.97425795e-03, 5.81037998e-04],
            [1.75016188621521, 1.3713326454162598, 1.3713326454162598, 5.82027435e-04],
        ]
    )

    # Split into three different vectors grouped by element index
    pos_si_time = vectors_set3[:, 0]
    pos_ls_time = vectors_set3[:, 1]
    pos_cs_time = vectors_set3[:, 2]
    pos_eig_time = vectors_set3[:, 3]

    plt.figure()
    x = np.linspace(0, 1e15, 10000)
    f = lambda x: 20000
    plt.loglog(x, list(map(f, x)), "k--", label="max_iterations")
    plt.loglog(avg_sizes, pos_si_iter, "b.-", label="Sim. Iteration")
    plt.loglog(avg_sizes, pos_ls_iter, "r.-", label="Lazy Shift")
    plt.loglog(avg_sizes, pos_cs_iter, "g.-", label="Complex Shift")
    plt.title("symmetric matrix, $n=50$", style="italic")
    plt.suptitle("Iterations to compute Eigenvalues")
    plt.xlabel("Spectral Radius, $\\rho(A)$", fontsize=14)
    plt.ylabel("Num. Iterations", fontsize=14)
    plt.legend()
    plt.show

    plt.figure()
    plt.loglog(avg_sizes, pos_si_time, "b.-", label="Sim. Iteration")
    plt.loglog(avg_sizes, pos_ls_time, "r.-", label="Lazy Shift")
    plt.loglog(avg_sizes, pos_cs_time, "g.-", label="Complex Shift")
    plt.loglog(avg_sizes, pos_eig_time, "k.--", label="np.eig()")
    plt.title("symmetric matrix, $n=50$", style="italic")
    plt.suptitle("Time to compute Eigenvalues")
    plt.xlabel("Spectral Radius, $\\rho(A)$", fontsize=14)
    plt.ylabel("Time, seconds", fontsize=14)
    plt.legend()
    plt.show

    #
    plt.show()
    return


plot1()
plot2()
# if __name__ == "__main__":
#     print("\n")
#     driver()
#     print("\n")
