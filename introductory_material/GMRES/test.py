import numpy as np

def hessenberg_matrix(n):
    # Create a random upper triangular matrix
    upper_triangular = np.triu(np.random.rand(n, n))
    
    # Create a random vector for the subdiagonal
    subdiagonal = np.random.rand(n-1)
    
    # Construct the Hessenberg matrix
    hessenberg = upper_triangular + np.diag(subdiagonal, k=-1)
    
    return hessenberg

# computes output to be miniimized 
def f(beta, e1, H, y):
    return np.linalg.norm(beta*e1 - H@y)

size = 5
H = hessenberg_matrix(size)

x0 =  np.random.rand(size, 1)
x0 = np.reshape(x0, [len(x0),])
b = np.random.rand(size, 1)
b = np.reshape(b, [len(b),])
y = np.random.rand(size, 1)
y = np.reshape(b, [len(b),])
e1 = np.zeros([size,])
e1[0] = 1.0

r_init = b - H @ x0
beta = np.linalg.norm(r_init)

print('f')
print(f(beta, e1, H, y))