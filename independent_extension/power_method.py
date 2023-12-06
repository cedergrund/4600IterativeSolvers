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
        n = i * dens1 + 10
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
        n = num1 * dens1 + 10 + i * dens2
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

    top = num1 * dens1 + 10 + num2 * dens2

    plt.figure()
    if err_p != [] or err_inv != [] or err_ray != []:
        x = np.linspace(0, top, 1000)
        f = lambda x: 5000
        plt.semilogy(x, list(map(f, x)), "k--", label="max_iterations")

    plt.semilogy(sizes, it_p, "b.", label="Power Method")
    plt.semilogy(sizes[err_p], it_p[err_p], "bx")
    plt.semilogy(sizes, it_inv, "r.", label="Inverse Power Method")
    plt.semilogy(sizes[err_inv], it_inv[err_inv], "rx")
    plt.semilogy(sizes, it_ray, "g.", label="Rayleigh Ritz Method")
    plt.semilogy(sizes[err_ray], it_ray[err_ray], "gx")
    plt.title("sparse matrix, density 0.25", style="italic")
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
    # plt.semilogy(sizes, time_real, "k.", label="np.eig()")
    plt.title("sparse matrix, density 0.25", style="italic")
    plt.suptitle("Time to compute Eigenvalues")
    plt.xlabel("Size, $n$", fontsize=14)
    plt.ylabel("Time, seconds", fontsize=14)
    plt.legend()
    plt.show()

    return


def testOne(n, sym=False, sparse=False, PD=False, SPD=False, diag=False):
    if sym:
        A = np.random.rand(n, n)
        At = np.transpose(A)
        A = np.matmul(At, A)
    elif sparse:
        A = scipy.sparse.random(n, n, 0.25)
        A = A.A
    elif PD:
        # A = random_cov(n)
        A = np.random.random((n, n))
        A = np.multiply(0.5, np.matmul(A, np.transpose(A)))
        A = np.add(A, np.multiply(n, np.identity(n)))
    elif SPD:
        A = np.random.random((n, n))
        A = np.matmul(A, np.transpose(A))
    elif diag:
        k = 1
        k1 = np.random.random((2 * k + 1, n))
        # k1 = np.where(k1 == 0, 0.01, k1)
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
    if sym or PD or SPD:
        eig_vals, _ = np.linalg.eigh(A)
    else:
        eig_vals, _ = np.linalg.eig(A)
    t1 = time.time()
    time_real = t1 - t0
    ind1 = np.argmax(abs(eig_vals))
    ind2 = np.argmin(abs(eig_vals))
    eig_val1 = eig_vals[ind1].real
    eig_val2 = eig_vals[ind2].real
    print(eig_val1, eig_val2, np.average(eig_vals))

    accurate = [
        np.abs(e1 - eig_val1) < 1e-5,
        np.abs(e2 - eig_val2) < 1e-3,
        np.any(np.isclose(eig_vals, e3, rtol=1e-5)),
    ]
    print(accurate[1], it_inv, e2, eig_val2, np.abs(e2 - eig_val2))
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


if __name__ == "__main__":
    print("\n")
    driver()
    print("\n")
