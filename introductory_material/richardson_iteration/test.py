import numpy as np


def qr_algorithm(matrix):
    # Implement or use a QR algorithm function
    # This function should return eigenvalues and eigenvectors
    # For simplicity, we'll use NumPy's eig function as a placeholder

    # eigenvalues, _ = np.linalg.eig(matrix)
    # eigenvalues, _ = complex_shiftedQR(matrix, len(matrix), max_iter=20000)
    # eigenvalues, _, _ = rayleighQuotient(matrix)

    b = np.random.rand(len(matrix), 1)
    eigs = np.sort(np.linalg.eig(matrix)[0])
    alpha = 2 / (eigs[-1] + eigs[0])
    eigenvalues, _ = richardsonIteration(
        matrix, len(matrix), np.eye(len(matrix)) - alpha * matrix, alpha * b
    )
    # eigenvalues = np.linalg.inv(matrix)

    return eigenvalues


def check_numerical_stability(matrix_size=100, epsilon=1e-6, num_tests=5):
    for _ in range(num_tests):
        # Generate a random matrix
        matrix = np.random.rand(matrix_size, matrix_size)
        matrix = np.triu(matrix)
        matrix = matrix + np.transpose(matrix) - np.diag(np.diag(matrix))
        matrix += matrix_size * np.eye(matrix_size)

        # Introduce perturbations
        perturbation = epsilon * np.random.rand(matrix_size, matrix_size)
        print()
        perturbed_matrix = matrix + perturbation

        # Apply QR algorithm to the original matrix
        eigenvalues_original = qr_algorithm(matrix)

        # Apply QR algorithm to the perturbed matrix
        eigenvalues_perturbed = qr_algorithm(perturbed_matrix)

        # Calculate relative error in eigenvalues
        relative_error_eigenvalues = np.linalg.norm(
            eigenvalues_original - eigenvalues_perturbed
        ) / np.linalg.norm(eigenvalues_original)

        print(
            f"Test {_:2}: Relative Error in Eigenvalues: {relative_error_eigenvalues:.6f}"
        )


def richardsonIteration(_, n, IalphA, alphb, tol=5e-08, Nmax=30000):
    x0 = np.zeros((n, 1))
    for i in range(1, Nmax):
        x1 = np.add(np.matmul(IalphA, x0), alphb)
        if np.linalg.norm(x1 - x0) < tol:
            return x1, i
        x0 = x1
    return x1, Nmax


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


# Run the stability check with default parameters
check_numerical_stability()

# import numpy as np
# import time

# n = 5
# A = np.random.rand(n, n)

# v1 = np.random.rand(n, 1)
# print(A)
# print()
# print(v1)
# print()
# t0 = time.time()
# sol = np.dot(np.matmul(np.transpose(v1), A), v1)[0][0]
# t_1 = time.time() - t0
# print(t_1)
# print(sol)
# print()
# t0 = time.time()
# sol = np.matmul(np.matmul(np.transpose(v1), A), v1)[0][0]
# t_1 = time.time() - t0
# print(t_1)
# print(sol)
# print()
# t0 = time.time()
# sol = ((np.transpose(v1) @ A) * v1)[0][0]
# t_1 = time.time() - t0
# print(t_1)
# print(sol)

# # ((v1t @ A) * v1)[0][0]
