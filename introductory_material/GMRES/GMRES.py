import numpy as np

# REMOVE LATEDR
import make_matrix

# implements the GMRES algorithm
def gmres(A, b, x0, k, tol=1E-10):

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
    n = np.shape(A)[0]
    x0 = np.reshape(x0, [len(x0),])
    b = np.reshape(b, [len(b),])

    # H is the Hessenberg matrix we will generate 
    H = np.zeros([k+1, k])

    # Vn is the first n columns of V giving similarity transformation to H
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

        # 2.2 (this does not match pseudocode)
        for i in range(1, j+1):
            H[i-1,j-1] = (A @ V[:,j-1]) @ V[:,i-1] # uses i instead of j here 

        # 2.3
        vhat = A @ V[:,j-1]
        for i in range(1, j+1):
            vhat -= H[i-1,j-1] * V[:,i-1]

        # 2.4
        H[j,j-1] = np.linalg.norm(vhat)

        # 2.5 
        V[:, j] = vhat / H[j, j-1]

        # 3.1
        y, _, _, _ = np.linalg.lstsq(H[:,:j], Beta*e1, rcond=None)
        x = x0 + V[:,:j] @ y

        # we converged on an approximate solution within tol 
        if (np.linalg.norm(A@x - b.transpose()) <= tol*np.linalg.norm(b)):
            return x, True, j
        
    # did not converge of an approximate solution
    return x, False, k

if __name__ == "__main__":
    # test case 
    n  = 200
    A = np.random.rand(n, n)
    b = np.random.rand(n, 1)
    #x0 =  np.random.rand(n, 1)
    x0 = np.zeros([n,1])

    A, b = make_matrix.build_mat(n, 0)

    # calling GMRES 
    x, converged, num_iter = gmres(A, b, x0, n)

    # if it doesn't converge on the first try, keep trying with updated initial guess
    while converged == False and num_iter < 5*n:
        x, converged, _ = gmres(A, b, x, n)
        num_iter += _

    # function outputs 
    print('converged:', converged)
    print('number of iterations:', num_iter)
    print('norm(Ax - b):', np.linalg.norm(A @ x - b.transpose()))
