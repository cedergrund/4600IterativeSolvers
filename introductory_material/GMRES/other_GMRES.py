# citation for code https://johnwlambert.github.io/least-squares/ 
import numpy as np
from typing import Tuple

def arnoldi_single_iter(A: np.ndarray, Q: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray]:
    """Compute a single iteration of Arnoldi.

    Args:
        A:
        Q:
        k:

    Returns:
        h:
        q:
    """
    q = A.dot(Q[:,k])
    h = np.zeros(k+2)
    for i in range(k+1):
        h[i] = q.T.dot(Q[:,i])
        q -= h[i]*Q[:,i]
    h[k+1] = np.linalg.norm(q)
    q /= h[k+1]
    return h,q

def gmres(A: np.ndarray, b: np.ndarray, x: np.ndarray, max_iters: int, EPSILON = 1e-10) -> np.ndarray:
    """Solve linear system via the Generalized Minimal Residual Algorithm.

    Args:
        A: Square matrix of shape (n,n) (must be nonsingular).
        b:
        x: Vector of shape (n,1) representing an initial guess for x.

    Returns:
        x_k: Vector of shape (n,1) representing converged solution for x.
    """
    EPSILON = 1e-10
    n,_ = A.shape
    if A.shape[0] != A.shape[1]:
        raise ValueError("Input argument `A` must be square.")

    r = b - A.dot(x)
    q = r / np.linalg.norm(r)
    Q = np.zeros((n,max_iters))
    Q[:,0] = q.squeeze()
    beta = np.linalg.norm(r)
    xi = np.zeros((n,1))
    xi[0] = 1 # e_1 standard basis vector, xi will be updated
    H = np.zeros((n+1,n))

    F = np.zeros((max_iters,n,n))
    for i in range(max_iters):
        F[i] = np.eye(n)

    for k in range(max_iters-1):
        H[:k+2,k], Q[:,k+1] = arnoldi_single_iter(A,Q,k)

        # Don't need to do this for 0,...,k since completed previously!
        c,s = givens_coeffs(H[k,k], H[k+1,k])
        # kth rotation matrix
        F[k, k,k] = c
        F[k, k,k+1] = s
        F[k, k+1,k] = -s
        F[k, k+1,k+1] = c

        # apply the rotation to both of these
        H[:k+2,k] = F[k,:k+2,:k+2].dot(H[:k+2,k])
        xi = F[k].dot(xi)

        if beta * np.linalg.norm(xi[k+1]) < EPSILON:
            # TODO: add comment why.
            break

    # When terminated, solve the least squares problem.
    # `y` must be (k,1).
    y, _, _, _ = np.linalg.lstsq(H[:k+1,:k+1],xi[:k+1], rcond=None)
    # `Q_k` will have dimensions (n,k).
    x_k = x + Q[:,:k+1].dot(y)
    return x_k

def givens_coeffs(a,b):
    """ """
    c = a / np.sqrt(a**2 + b**2)
    s = b / np.sqrt(a**2 + b**2)
    return c, s

def arnoldi(A: np.ndarray, b: np.ndarray, k: int):
    """ 
    Computes a basis of the (k+1)-Krylov subspace of A: the space
    spanned by {b, Ab, ..., A^k b}.

    Args:
    A: Numpy array of shape (n,n)
    b: Vector of shape (n,1).
    k: Dimension of Krylov subspace.

    Returns:
    Q: Orthonormal basis for Krylov subspace.
    H: Upper Hessenberg matrix.
    """
    n = A.shape[0]

    H = np.zeros((k,k))
    Q = np.zeros((n,k))

    # Normalize the input vector
    # Use it as the first Krylov vector
    Q[:,0] = b / np.linalg.norm(b)

    for j in range(k-1):
        Q[:,j+1] = A.dot(Q[:,j])
        for i in range(j):
            H[i,j] = Q[:,j+1].dot(Q[:,i])
            Q[:,j+1] = Q[:,j+1] - H[i,j] * Q[:,i]

        H[j+1,j] = np.linalg.norm(Q[:,j+1])
        Q[:,j+1] /= H[j+1,j]
    return Q,H

if __name__ == "__main__":
    # test case 
    n  = 20
    A = np.random.rand(n, n)
    b = np.random.rand(n, 1)
    #x0 =  np.random.rand(n, 1)
    x0 = np.zeros([n,1])

    # calling GMRES 
    x = gmres(A, b, x0, n)
    print('norm(Ax - b):', np.linalg.norm(A @ x - b.transpose()))
