import numpy as np

# finding minimizer (y) not implemented
# only works if k = dim(A)[0]
def gmres(A, b, x0, k, tol=1e-10):

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

    # number of Arnoldi iterations 
    k = n

    # H is the Hessenberg matrix we will generate 
    H = np.zeros([k+1, k])

    # Qn is the first n columns of Q which provides the similarity transformation to H
    V = np.zeros([n,k+1])

    # e1 the first cannonical vector 
    e1 = np.zeros(k+1)
    e1[0] = 1

    # 1.1
    r_init = b - A @ x0
    Beta = np.linalg.norm(r_init)
    V[:,0] = r_init / Beta

    # 2.1
    for j in range(1, k+1):

        # 2.2
        for i in range(1, j+1):
            H[i-1,j-1] = (A @ V[:,j-1]) @ V[:,i-1]

        # 2.3
        vhat = A @ V[:,j-1]
        for i in range(1, j+1):
            vhat -= H[i-1,j-1] * V[:,i-1]

        # 2.4
        H[j,j-1] = np.linalg.norm(vhat)

        
        # 2.5 
        V[:, j] = vhat / H[j, j-1]


    # 3.1
    y, _, _, _ = np.linalg.lstsq(H, Beta*e1, rcond=None)
    x = x0 + V[:,:k] @ y

    # test 
    print('inside the function')
    print(np.linalg.norm(A@x - b))
    print(x)
        
    # currently just returns x0
    return x, False, k, A

# test case 
n  = 20
A = np.random.rand(n, n)
b = np.random.rand(n, 1)
x0 =  np.random.rand(n, 1)

x, converged, num_iter, A0 = gmres(A, b, x0, n)
print('\n', "function returned")
print(x, converged, num_iter)
print('norm(Ax - b):', np.linalg.norm(A@x - b.transpose()))