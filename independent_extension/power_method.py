import numpy as np
import time
import scipy
import matplotlib.pyplot as plt


def driver():
    to = time.time()
    num_avg = 1
    num1, dens1 = 100, 10
    num2, dens2 = 50, 20
    tot_sizes = np.zeros((num_avg, num1 + num2))
    tot_time_p, tot_it_p = np.zeros((num_avg, num1 + num2)), np.zeros(
        (num_avg, num1 + num2)
    )
    tot_time_inv, tot_it_inv = np.zeros((num_avg, num1 + num2)), np.zeros(
        (num_avg, num1 + num2)
    )
    tot_time_ray, tot_it_ray = np.zeros((num_avg, num1 + num2)), np.zeros(
        (num_avg, num1 + num2)
    )
    err_p, err_inv, err_ray = [], [], []

    for j in range(num_avg):
        print(j, time.time() - to)
        sizes = np.zeros(num1 + num2)
        time_p, it_p = np.zeros(num1 + num2), np.zeros(num1 + num2)
        time_inv, it_inv = np.zeros(num1 + num2), np.zeros(num1 + num2)
        time_ray, it_ray = np.zeros(num1 + num2), np.zeros(num1 + num2)
        err_p, err_inv, err_ray = [], [], []

        for i in range(num1):
            # size of matrix
            if i % (num1 / 5) == 0:
                print("{:.0%}".format(i / (num1 + num2)))
            n = i * dens1 + 10
            sizes[i] = n
            A = createMatrix(n, separated=True)

            # output run-time
            (
                time_p[i],
                it_p[i],
                time_inv[i],
                it_inv[i],
                time_ray[i],
                it_ray[i],
                _,
                _,
            ) = testOne(A, n)

        for i in range(num2):
            # size of matrix
            if i % (num2 / 5) == 0:
                print("{:.0%}".format((i + num1) / (num1 + num2)))
            n = num1 * dens1 + 10 + i * dens2
            sizes[i + num1] = n

            A = createMatrix(n, separated=True)
            # output run-time
            (
                time_p[i + num1],
                it_p[i + num1],
                time_inv[i + num1],
                it_inv[i + num1],
                time_ray[i + num1],
                it_ray[i + num1],
                _,
                _,
            ) = testOne(A, n)

        tot_sizes[j] = sizes
        tot_time_p[j], tot_it_p[j] = time_p, it_p
        tot_time_inv[j], tot_it_inv[j] = time_inv, it_inv
        tot_time_ray[j], tot_it_ray[j] = time_ray, it_ray
        print("\n")

    avg_sizes = np.mean(tot_sizes, axis=0)
    avg_tp = np.mean(tot_time_p, axis=0)
    avg_ti = np.mean(tot_time_inv, axis=0)
    avg_tr = np.mean(tot_time_ray, axis=0)
    avg_ip = np.mean(tot_it_p, axis=0)
    avg_ii = np.mean(tot_it_inv, axis=0)
    avg_ir = np.mean(tot_it_ray, axis=0)

    err_p.append(np.where(avg_ip == 5000)[0])
    err_inv.append(np.where(avg_ii == 5000)[0])
    err_ray.append(np.where(avg_ir == 5000)[0])
    plt.figure()

    if err_p != [] or err_inv != [] or err_ray != []:
        top = num1 * dens1 + 10 + num2 * dens2
        x = np.linspace(0, top, 1000)
        f = lambda x: 5000
        plt.semilogy(x, list(map(f, x)), "k--", label="max_iterations")

    plt.semilogy(avg_sizes, avg_ip, "b.", label="Power Method")
    plt.semilogy(avg_sizes[err_p], avg_ip[err_p], "bx")
    plt.semilogy(avg_sizes, avg_ii, "r.", label="Inverse Power Method")
    plt.semilogy(avg_sizes[err_inv], avg_ii[err_inv], "rx")
    plt.semilogy(avg_sizes, avg_ir, "g.", label="Rayleigh Ritz Method")
    plt.semilogy(avg_sizes[err_ray], avg_ir[err_ray], "gx")
    plt.title("separated eigenvalues", style="italic")
    plt.suptitle("Iterations to compute Eigenvalues")
    plt.xlabel("Size, $n$", fontsize=14)
    plt.ylabel("Num. Iterations", fontsize=14)
    plt.legend()
    plt.show

    plt.figure()
    plt.semilogy(avg_sizes, avg_tp, "b.", label="Power Method")
    plt.semilogy(avg_sizes[err_p], avg_tp[err_p], "bx")
    plt.semilogy(avg_sizes, avg_ti, "r.", label="Inverse Power Method")
    plt.semilogy(avg_sizes, avg_tr, "g.", label="Rayleigh Ritz Method")
    plt.semilogy(avg_sizes[err_inv], avg_ti[err_inv], "rx")
    plt.semilogy(avg_sizes[err_ray], avg_tr[err_ray], "gx")
    plt.title("separated eigenvalues", style="italic")
    plt.suptitle("Time to compute Eigenvalues")
    plt.xlabel("Size, $n$", fontsize=14)
    plt.ylabel("Time, seconds", fontsize=14)
    plt.legend()
    plt.show()

    return


def testOne(A, n):
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
    eig_vals, _ = np.linalg.eig(A)
    t1 = time.time()
    time_real = t1 - t0
    ind1 = np.argmax(abs(eig_vals))
    ind2 = np.argmin(abs(eig_vals))
    eig_val1 = eig_vals[ind1].real
    eig_val2 = eig_vals[ind2].real

    accurate = [
        np.abs(e1 - eig_val1) < 1e-5,
        np.abs(e2 - eig_val2) < 1e-3,
        np.any(np.isclose(eig_vals, e3, rtol=1e-5)),
    ]
    # print(eig_vals)

    return time_p, it_p, time_inv, it_inv, time_ray, it_ray, time_real, accurate


def powerMethod(A, shift=0, tol=1e-8, Nmax=5000):
    n = A.shape[1]
    count = Nmax

    # create a random vector x to check against
    v0 = np.random.rand(n, 1)
    # A = A - np.multiply(shift, np.eye(n))

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
    # ev += shift
    return ev, v1, count


def inversePowerMethod(A, shift=0, tol=1e-8, Nmax=5000):
    n = A.shape[1]
    count = Nmax

    # create a random vector x to check against
    v0 = np.random.rand(n, 1)
    # A = A - np.multiply(shift, np.eye(n))
    A = np.linalg.inv(A)

    # run power method iteration
    for i in range(Nmax):
        v1 = np.matmul(A, v0)
        v1 = np.multiply(1 / np.linalg.norm(v1), v1)
        v1t = np.transpose(v1)

        if np.linalg.norm(abs(v1) - abs(v0)) < tol:
            count = i + 1
            break
        v0 = v1

    # predict eigenvalue and return
    vt1 = np.transpose(v1)
    ev = np.dot(np.matmul(vt1, A), v1)[0][0]
    ev = 1 / ev  # + shift
    return ev, v1, count


def rayleighQuotient(A, tol=1e-8, Nmax=5000):
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

        if abs(abs(e1) - abs(e0)) < tol:
            count = i + 1
            break
        v0 = v1
        e0 = e1

    return e1, v1, count


def random_cov(n):
    Q = np.random.random((n, n))

    eigen_mean = n
    Qt = np.transpose(Q)
    A = np.abs(eigen_mean + np.random.random((n, 1)))
    A = np.diag(A.flatten())
    A = np.matmul(Qt, A)
    A = np.matmul(A, Q)

    return A


def createMatrix(n, same=False, sym=False, clustered=False, separated=False):
    if clustered:
        eig = np.array([99] + [100] * (n - 2) + [101])
        sign = np.random.choice([-1, 1], size=n)
        eig = np.multiply(eig, sign)
        A = generate_matrix_with_eigenvalues(n, eig)
    elif separated:
        eig = np.array([1] + [50] * (n - 2) + [2500])
        sign = np.random.choice([-1, 1], size=n)
        eig = np.multiply(eig, sign)
        A = generate_matrix_with_eigenvalues(n, eig)
    elif same:
        eig = np.array([n] * (n))
        sign = np.random.choice([-1, 1], size=n)
        eig = np.multiply(eig, sign)
        A = generate_matrix_with_eigenvalues(n, eig)
    elif sym:
        A = np.random.random((n, n))
        A = np.multiply(0.5, A + np.transpose(A))
    else:
        A = np.random.rand(n, n)
    return A


def generate_matrix_with_eigenvalues(n, eigenvalues):
    Q, _ = np.linalg.qr(np.random.rand(n, n))
    D = np.diag(eigenvalues)
    A = np.matmul(np.matmul(Q, D), np.linalg.inv(Q))
    return A


# n = 2000
# A = createMatrix(n, same=True)
# # print(np.linalg.eig(A)[0])
# inversePowerMethod(A)

if __name__ == "__main__":
    print("\n")
    driver()
    print("\n")
