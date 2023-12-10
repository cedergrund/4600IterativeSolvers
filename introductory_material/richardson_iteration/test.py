import numpy as np
import scipy
import time

n = 2000
# sparse_matrix_a = csr_matrix(np.random.rand(4, 4) > 0.75)
# sparse_matrix_b = csr_matrix(np.random.rand(4, 4) > 0.75)

# sparse_matrix = np.random.rand(n, n) * (
#     np.random.rand(n, n) > 0.9
# )

# sparse_matrix_a.A

# print(sparse_matrix)

# print(sparse_matrix)

# condition number of 10
h = 0.81 * n
eig = np.linspace(n - h, n + h, n)
np.random.shuffle(eig)
D = np.diag(eig)
A = scipy.sparse.random(n, n, 0.25)
# print(A)
A = A.A
A = np.triu(A + D)
A = A + np.transpose(A) - np.diag(np.diag(A))
eigs = np.sort(np.linalg.eig(A)[0])
print("kappa =", eigs[-1] / eigs[0])
sparse_matrix = scipy.sparse.csr_matrix(A)
# print(sparse_matrix)

b = np.random.rand(n)
# print(b)
print()


t0 = time.time()
x = np.dot(A, A)
t1 = time.time()
print("np.dot", t1 - t0)
print()

t0 = time.time()
x = np.matmul(A, A)
t1 = time.time()
print("np.matmul", t1 - t0)
print()


t0 = time.time()
# sparse_matrix = scipy.sparse.csr_matrix(A)
x = sparse_matrix.dot(A)
# x = time_scipy_sparse_dot(A, b)
t1 = time.time()
print("scipy sparse", t1 - t0)
print()

# result = sparse_matrix_a.dot(sparse_matrix_b)
# print(result.A)


# import numpy as np
# from scipy.sparse import csr_matrix
# import timeit


# def time_np_dot(matrix, vector, repetitions=100):
#     setup = f"import numpy as np; matrix = np.array({matrix.tolist()}); vector = np.array({vector.tolist()})"
#     statement = "np.dot(matrix, vector)"
#     execution_time = timeit.timeit(statement, setup=setup, number=repetitions)
#     return execution_time / repetitions


# def time_scipy_sparse_dot(matrix, vector, repetitions=100):
#     sparse_matrix = csr_matrix(matrix)
#     setup = f"from scipy.sparse import csr_matrix; import numpy as np; matrix = csr_matrix({matrix.tolist()}); vector = np.array({vector.tolist()})"
#     statement = "matrix.dot(vector)"
#     execution_time = timeit.timeit(statement, setup=setup, number=repetitions)
#     return execution_time / repetitions


# # Example usage:
# print("hello")
# matrix_size = 1000
# sparse_matrix = np.random.rand(matrix_size, matrix_size) > 0.9
# vector = np.random.rand(matrix_size)

# np_dot_time = time_np_dot(sparse_matrix, vector)
# print("hello")
# scipy_sparse_dot_time = time_scipy_sparse_dot(sparse_matrix, vector)
# # print("hello")


# print(f"Execution time for np.dot: {np_dot_time:.6f} seconds")
# print(f"Execution time for scipy.sparse dot: {scipy_sparse_dot_time:.6f} seconds")
