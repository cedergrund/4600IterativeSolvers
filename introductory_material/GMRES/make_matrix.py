import numpy as np
from scipy.linalg import eigvals, norm, qr

# builds matrix for testing
def build_mat(n, opt):

    '''
    input: 
    n = number of entries 
    opt: encodes characteristic of desired matrix 
        opt == 0: well separated 
        opt == 1: full rank with 3 distinct e-vals
        opt == 2: full rank and all e-vals in ball of radius == 1e-5 centered at 1 
        opt == 3: ill-conditions ie K > 1E20
    Output:
    A: Matrix with desired specifications
    '''

    # Create a random matrix R with random entries
    R = np.random.rand(n, n)
    
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
        diagonal_entries = np.linspace(-1E-5, 1E-5, n) 

    # ill-conditions ie K > 1E20
    if opt == 3: 
        diagonal_entries = []
        for i in range(n):
            diagonal_entries.append(np.random.randint(3))
        
    # Create a diagonal matrix D with the chosen diagonal entries
    D = np.diag(diagonal_entries)
    
    # Construct the matrix A = Q^T * D * Q
    A = np.dot(Q.T, np.dot(D, Q))
    
    return A

# special function to build the matrix specified by 2.5.e
def build_mat_opt_4(n): 
    '''
    input: 
    n = number of entries 

    Output:
    A: Matrix with only one 0 eigenvector 
    b: in the range of A 
    '''
    R = np.random.rand(n, n)
    Q, _ = qr(R, mode='economic')
    diagonal_entries = np.linspace(0, 1E-5, n) 
    D = np.diag(diagonal_entries)
    A = np.dot(Q.T, np.dot(D, Q))
    b = A[:,1]
    return A, b

# Set the size of the matrix
n = 2000
opt = 3

# Create matrix with desired specifications
#A = build_mat(n, opt)
A, b = build_mat_opt_4(n)

# Confirm properties of the matrix
eigenvalues = eigvals(A)
condition_number = norm(A) * norm(np.linalg.inv(A))
print("Number of eigenvalues:", len(eigenvalues))
print("Well conditioned:", condition_number < 1E20)
