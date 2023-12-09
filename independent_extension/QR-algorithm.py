import numpy as np
import time
import scipy
import matplotlib.pyplot as plt


def driver():
    num_avg = 10
    t0 = time.time()
    n = 50
    iterations = np.zeros((num_avg, 3))
    times = np.zeros((num_avg, 4))
    error = np.zeros((num_avg, 3))
    converged = np.zeros(3)

    for i in range(num_avg):
        print("{}%".format(i / num_avg), time.time() - t0)
        A = createMatrix(n, cond_inf=True)
        iterations[i], times[i], error[i], c = testOne(A, n, [True, True, True])
        converged += c
        print()

    avg_it = np.mean(iterations, axis=0)
    avg_t = np.mean(times, axis=0)
    avg_err = np.mean(error, axis=0)

    print("iterations", avg_it)
    print("times", avg_t)
    print("avg_error", avg_err)
    print("converged %", converged / num_avg * 100)


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
        e_sim, iterations[1] = simple_shiftedQR(A, n, max_iter=20000)
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
    if run[1]:
        error[1] = np.linalg.norm(e_sim - e_real)
        if error[1] < 1e-3:
            converged[1] = 1
    if run[2]:
        if np.shape(e_comp) != np.shape(e_real):
            print("oops")
            print(len(e_comp) / len(e_real))
        else:
            error[2] = np.linalg.norm(e_comp - e_real)
            if error[2] < 1e-3:
                converged[2] = 1

    return iterations, times, error, converged


def createMatrix(n, Hes=False, same=False, cond_inf=False, factor=False, sym=False):
    # condition 10
    if Hes:
        h = int(0.82 * n)
        eig = np.linspace(n - h, n + h, n)
        sign = np.random.choice([-1, 1], size=n)
        eig = np.multiply(eig, sign)
        A = generate_matrix_with_eigenvalues(n, eig)
        A = scipy.linalg.hessenberg(A)
    elif sym:
        h = 0.96 * n
        eig = np.linspace(n - h, n + h, n)
        sign = np.random.choice([-1, 1], size=n)
        eig = np.multiply(eig, sign)
        A = generate_matrix_with_eigenvalues(n, eig)
    elif same:
        eig = np.array([n] * (n))
        sign = np.random.choice([-1, 1], size=n)
        eig = np.multiply(eig, sign)
        A = generate_matrix_with_eigenvalues(n, eig)
    elif cond_inf:
        eig = np.array([1e-5] + [n] * (n - 1))
        # eig = np.linspace(0, n, n)
        # eig = 5 * np.random.rand(n)
        sign = np.random.choice([-1, 1], size=n)
        eig = np.multiply(eig, sign)
        print(eig)
        A = generate_matrix_with_eigenvalues(n, eig)
    elif factor:
        f = 1 / 5
        eig = 10 * f ** np.arange(n)
        print(eig[0] / eig[-1])
        A = generate_matrix_with_eigenvalues(n, eig)
    else:
        A = np.random.rand(n, n)
    return A


def simultaneousQR(A, n, max_iter=1000, tol=1e-5):
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


def simple_shiftedQR(A, n, max_iter=1000, tol=1e-8):
    if n == 1:
        return A[0, 0], 0
    X0 = A.copy()
    count = 0
    e1 = np.zeros(n)
    und1 = 10000
    I = np.eye(n)
    shift1 = 0

    for i in range(max_iter):
        und0 = und1
        e0 = e1.copy()
        shift1 = X0[n - 1, n - 1]
        und1 = np.linalg.norm(np.diag(X0, k=-1))

        # check for false convergence
        smol = abs(abs(und1) - abs(und0)) < 1e-2
        if smol and und1 > 1:
            e1 = np.diag(X1)
            # e = X0[n - 1, n - 1]
            print("could not converge")
            # e1, c1 = simple_shiftedQR(
            #     X0[: n - 1, : n - 1], n - 1, max_iter=max_iter - i
            # )
            # e1 = np.append(e1, e)
            return e1, i + 1

        shift_mat = np.multiply(shift1, I)
        Q, R = np.linalg.qr(X0 - shift_mat)
        X1 = np.matmul(R, Q) + shift_mat
        e1 = np.diag(X1)
        if np.all(np.abs(e1 - e0) < tol):
            if np.linalg.norm(np.diag(X1, k=-1)) < 1e-2:
                print()
                count = i + 1
                break
            e, c = pureQR(X0, n, max_iter=max_iter - i)
            return e, c + i + 1
        X0 = X1.copy()
    return e1, count


# for m in range(n - 1, 0, -1):
#     I = np.eye(n)
#     while abs(X0[m, m - 1]) > tol:
#         count += 1
#         shift = X0[m, m]
#         shift_mat = np.multiply(shift, I)
#         X0 -= shift_mat
#         Q, R = np.linalg.qr(X0)
#         X1 = np.matmul(R, Q) + shift_mat
#         if count > max_iter:
#             return np.diag(X1), max_iter
#         X0 = X1
#     # X0 = X0[:m, :m]
#     # print(X0)
#     # print()
# return np.diag(X0), count
# # while abs(X0[m, m - 1]) > tol:
# #     count += 1
# #     shift = X0[m, m]
# #     shift_mat = np.multiply(shift, I)
# #     print(X0)
# #     print()
# #     X0 -= shift_mat
# #     Q, R = np.linalg.qr(X0)
# #     X1 = np.matmul(R, Q) + shift_mat
# #     if count > max_iter:
# #         return np.diag(X1), max_iter
# #     X0 = X1
# # print("running sim. QR after", count)
# # e, c = simultaneousQR(X0[: n - 1, : n - 1], n - 1, max_iter=max_iter - count)
# # e = np.append(e, X1[-1, -1])
# # return e, c + count


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


def complex_shiftedQR(A, n, max_iter=1000, tol=1e-5):
    X0 = A.copy()
    count = 0
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


def generate_matrix_with_eigenvalues(n, eigenvalues):
    Q, _ = np.linalg.qr(np.random.rand(n, n))
    D = np.diag(eigenvalues)
    A = np.matmul(np.matmul(Q, D), np.linalg.inv(Q))
    return A


# A = createMatrix(100, sym=True)
# # e, i = simple_shiftedQR(A, 10, max_iter=20000)
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

# n = 10
# A = createMatrix(n, sym=True)
# # print(A)

# r, _ = np.linalg.eig(A)
# r = np.sort(r)
# # print(r)
# r2 = np.sort(abs(r))
# tot = 0
# for i in range(len(r2) - 1):
#     tot += r2[i + 1] / r2[i]
# t0 = time.time()
# e, i = simple_shiftedQR(A, n, max_iter=50000)
# t1 = time.time()
# e = np.sort(e)
# print(np.linalg.norm(e - r), i, t1 - t0)
# # print(tot / (len(r2) - 1), r2[-1] / r2[0])
# # print(r)
# print()
# t0 = time.time()
# e, i = simultaneousQR(A, n, max_iter=50000)
# t1 = time.time()
# e = np.sort(e)
# print(np.linalg.norm(e - r), i, t1 - t0)
# print()
# t0 = time.time()
# e, i = complex_shiftedQR(A, n, max_iter=50000)
# t1 = time.time()
# e = np.sort(e)
# print(np.linalg.norm(e - r), i, t1 - t0)
# # print()
# # print(r)
# # e, i = simultaneousQR(A, n, max_iter=50000)
# # e = np.sort(e)
# # print(np.linalg.norm(e - r), i)
# # e, i = simple_shiftedQR(A, n, max_iter=50000)
# # e = np.sort(e)
# # # print(e)
# # print(np.linalg.norm(e - r), i)
if __name__ == "__main__":
    print("\n")
    driver()
    print("\n")
