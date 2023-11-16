import numpy as np

# finding minimizer (y) not implemented
# only works if k = dim(A)[0]
def gmres(A, b, x0, k, tol=1e-8):

    """
    GMRES algorithm for solving the linear system Ax = b.

    Parameters:
    - A: The coefficient matrix (n x n).
    - b: The right-hand side vector (n x 1).
    - x0: Initial guess for the solution (n x 1).
    - tol: Tolerance for convergence.
    - max_iter: Maximum number of iterations.

    Returns:
    - x: The solution vector.
    - converged: Boolean indicating whether the method converged.
    - num_iter: Number of iterations performed.
    """

    # shape compatibility 
    x0 = np.reshape(x0, [len(x0),])
    b = np.reshape(b, [len(b),])

    # initialization (HELP)
    H = np.zeros(np.shape(A))
    V = np.zeros([len(b), k])

    # 1.1
    r_init = b - A @ x0

    # 1.2
    v0 = r_init / np.linalg.norm(r_init)
    V[:,0] = v0

    # 2.1
    for j in range(k-1):

        # 2.2
        for i in range(j+1):
            H[i,j] = A @ V[:,j] @ V[:,j]

        # 2.3
        vhat = A @ V[:,j]
        for i in range(j+1):
            vhat -= H[i,j] * V[:,i]

        # 2.4
        H[j+1,j] = np.linalg.norm(vhat)

        # 2.5 
        V[:, j+1] = vhat / H[j+1, j]

        # TODO 
        # Calculate minimizing vector y for ||Be_1 - H_ky||
        y = x0

    # 3.1
    x = x0 + V @ np.transpose(y)

    print('H')
    print(H)
    print()
    print('V')
    print(V)
        
    # currently just returns x0
    return x, False, k 

# test case 
size  = 3
A = np.random.rand(size, size)
b = np.random.rand(size, 1)
x0 =  np.random.rand(size, 1)
solution, converged, num_iter = gmres(A, b, x0, size)
print('\n', "function returned")
print(solution, converged, num_iter)