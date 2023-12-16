import numpy as np
import scipy
from scipy.linalg import eigvals, norm, qr

# builds matrix for testing
def build_mat(n, opt, density = 0.1):

    '''
    input: 
    n = number of entries 
    opt: encodes characteristic of desired matrix 
        opt == 0: well separated 
        opt == 1: full rank with 3 distinct e-vals
        opt == 2: full rank and all e-vals in ball of radius == 1e-5 centered at 1 
        opt == 3: ill-conditions ie K > 1E20
        opt -- 4: one zero eigenvector and b in range(A)
    Output:
    A: Matrix with desired specifications
    b: a random matrix 
        - unless opt == 4 then b in range(A)
    '''

    # Create a random matrix R with random entries
    R = scipy.sparse.random(n, n, density)
    R = R.toarray()

    # create a random b 
    b = np.random.rand(n, 1)

    # Perform QR factorization on R to obtain an orthogonal matrix Q
    Q, _ = qr(R, mode='economic')
    
    # well separated 
    if opt == 0: 
        diagonal_entries = np.linspace(n, 1e-10, n)

    # full rank with 3 distinct e-vals
    if opt == 1:
        diagonal_entries = []
        for i in range(n):
            diagonal_entries.append(np.random.randint(1, 4))
            
    # full rank and all e-vals in ball of radius == 1e-5 centered at 1 
    if opt == 2: 
        diagonal_entries = np.linspace(1-1E-5, 1 + 1E-5, n) 

    # ill-conditions ie K > 1E20
    if opt == 3: 
        diagonal_entries = []
        for i in range(n):
            diagonal_entries.append(np.random.randint(3))
    
    # one zero eigenvalue and b in range(A)
    if opt == 4:
        R = np.random.rand(n, n)
        Q, _ = qr(R, mode='economic')
        diagonal_entries = np.linspace(0, 1E-5, n) 
        D = np.diag(diagonal_entries)
        A = np.dot(Q.T, np.dot(D, Q))
        b = A[:,1]
        return A, b
    
    # full rank with one distinct eigenvalue
    if opt == 5: 
        diagonal_entries = []
        choice = np.random.randint(1, 4)
        for i in range(n):
            diagonal_entries.append(choice)
        
    # Create a diagonal matrix D with the chosen diagonal entries
    D = np.diag(diagonal_entries)
    
    # Construct the matrix A = Q^T * D * Q
    A = np.dot(Q.T, np.dot(D, Q))
    
    return A, b

if __name__ == "__main__":
    # Set the size of the matrix
    n = 20

    # test opt 1-3
    for opt in range(6):
        print('opt', opt)

        # Create matrix with desired specifications
        A, b = build_mat(n, opt)

        # Confirm properties of the matrix
        if opt != 4:
            eigenvalues = eigvals(A)
            condition_number = norm(A) * norm(np.linalg.inv(A))
            print("Number of eigenvalues:", len(eigenvalues))
            print("Well conditioned:", condition_number < 1E20)

        # extra tests for 2.5.e
        if opt == 4:
                x = np.zeros(n)
                x[1] = 1
                print('b in range(A):', np.linalg.norm(A@x.transpose() - b) == 0.0)


