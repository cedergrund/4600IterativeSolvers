
# libraries
import numpy as np

# performs richardson iteration to solve a linear system 
def richardson_iteration(A, b, alpha, tol, i_max = 1E5):

    """
    Inputs: 
    A : (n, n) matrix representing the system of equations
    b : (n, 1) vector representing the solution to the system of equations
    alpha : (scalar) learning rate
    tol : the tolerance for the iteration approximation
    i_max (default = 1E5) : the max number of iterations allowed 

    Outputs: 
    x_star : approximation to the solution of the system of equations
    i : number of iterations to convergence 
    """

    # boiler-plate 
    A = np.array(A)
    b = np.array(b)
    n = A.shape[0]
    x_star = np.zeros((n, 1))
    assert np.shape(x_star) == np.shape(b), 'check shapes of input'

    # tracks the number of iterations to convergence 
    i = 0

    # Perform the iteration
    x_prev = np.ones((n, 1))
    while np.any(np.abs(x_star - x_prev) >= tol):

        # break if past i_max without solution
        if i >= i_max:
            print("Did not converge in", i_max, "iterations.")
            return x_star, i_max

        # Save the previous value of x_star
        x_prev = x_star.copy()

        # Update the value of x_star
        x_star = (np.identity(n) - alpha * A) @ x_prev + alpha * b

        # update counter 
        i+=1

    return x_star, i


if __name__ == '__main__':
    A = np.array([[1,2],
                  [-3,2]])
    b = np.array([[2],[5]])
    tol = 1E-11
    alpha = 0.25 #  how are we supposed to find this? 
    print(richardson_iteration(A, b, alpha, tol))

    # Verify the solution
    verification_result = np.allclose(A.dot(richardson_iteration(A, b, alpha, tol)[0]), b, atol=tol)
    print("Solution verification:", verification_result)

    A = np.array([[1,1],
                  [2,2]])
    b = np.array([[2],[5]])
    tol = 1E-11
    alpha = 0.25 #  how are we supposed to find this? 
    print(richardson_iteration(A, b, alpha, tol))

    # Verify the solution
    verification_result = np.allclose(A.dot(richardson_iteration(A, b, alpha, tol)[0]), b, atol=tol)
    print("Solution verification:", verification_result)

    # made this one using ChatGPT
    A = np.array([[1, 2, 0, 0, 0],
                [-3, 2, 0, 0, 0],
                [0, 0, 1, 2, 0],
                [0, 0, -3, 2, 0],
                [0, 0, 0, 0, 1]])

    b = np.array([[2],
                [5],
                [3],
                [4],
                [1]])
    tol = 1E-11
    alpha = 0.25 #  how are we supposed to find this? 
    print(richardson_iteration(A, b, alpha, tol))

    # Verify the solution
    verification_result = np.allclose(A.dot(richardson_iteration(A, b, alpha, tol)[0]), b, atol=tol)
    print("Solution verification:", verification_result)