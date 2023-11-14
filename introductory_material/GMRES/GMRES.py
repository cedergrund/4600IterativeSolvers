import numpy as np

# not working
# finding minimizer (y) not implemented
# only works if n = dim(A)
def gmres(A, b, x0, k=3, tol=1e-8):
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

    # initializations 
    x0 = np.reshape(x0, [len(x0),])
    b = np.reshape(b, [len(b),])
    H = np.zeros([k, k])
    V = np.zeros([len(b), k])

    # 1.1
    r_init = b - np.dot(A,b)

    # 1.2
    v0 = r_init / np.linalg.norm(r_init)
    V[:,0] = v0

    # 2.1
    for j in range(k-1):
        
        # 2.2
        for i in range(j):
            H[i,j] = np.dot(np.dot(A,V[:,j]), V[:,j])

        # 2.3
        vhat = np.dot(A,V[:,j])
        for i in range(j):
            vhat -= np.dot(H[i,j],V[:,i])

        # 2.4
        H[j+1,j] = np.linalg.norm(vhat)

        # 2.5 
        V[:, j+1] = vhat / H[j+1, j]
        
        # TODO 
        # Calculate minimizing vector y for ||Be_1 - H_ky||
        y = x0
        
    # 3.1
    x = x0 + np.dot(V,np.transpose(y))

    print('H')
    print(H)
    print()
    print('V')
    print(V)
        
    # currently just returns x0
    return x, False, k 

# test case 
A = np.random.rand(3, 3)
b = np.random.rand(3, 1)
x0 =  np.random.rand(3, 1)
solution, converged, num_iter = gmres(A, b, x0)
print(solution, converged, num_iter)