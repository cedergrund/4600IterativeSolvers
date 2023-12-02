# libraries 
import os
import logging
import time 
import numpy as np

# import files 
import GMRES
import make_matrix

if __name__ == "__main__":
    # out_dir
    out_dir = os.path.join(os.getcwd(),'outputs')

    # logger
    logging.basicConfig(level=logging.DEBUG)
    logger = logging.getLogger(__name__)
    file_handler = logging.FileHandler(os.path.join(out_dir,'results.txt'))
    file_handler.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.debug(str(out_dir) + '\n')

    # parameters 
    n = 2000
    run_tests = [1, 1, 1, 1, 1]
    x0_zeros = True
    max_epochs = 3
    max_iter = max_epochs*n
    tol = 1E-10

    # running tests 
    if x0_zeros:
        x0 = np.zeros([n,1])
    else:
        x0 = np.random.rand(n, 1)

    data = []

    for i in range(len(run_tests)):
        if run_tests[i]:

            # build matrix 
            A, b = make_matrix.build_mat(n, i)

            # timing
            tic = time.time()

            # call GMRES 
            logger.debug(f'GMRES for test: {i}')
            x, converged, num_iter = GMRES.gmres(A, b, x0, n, tol)

            # if it doesn't converge on the first try, keep trying with updated initial guess
            while converged == False and num_iter < max_iter:
                logger.debug('Did not Converge on first epoch, performing additional epochs')
                logger.debug(f'number of iterations: {num_iter}')
                x, converged, _ = GMRES.gmres(A, b, x, n)
                num_iter += _

            # timing 
            toc = time.time()
            time_elapsed = toc - tic

            # Logger info 
            metadata = {
                'seconds elapsed' : time_elapsed,
                'size of n' : n,
                'number of iterations' : num_iter,
                'converged' : converged,
                'x0_zeros' : x0_zeros,
                'max_epochs' : max_epochs,
                '||Ax - b||' : np.linalg.norm(A@x - b.transpose()),
                'tol' : tol
            }
            logger.debug('metadata')
            for key, value in metadata.items():
                logger.debug(f'{key}: {value}')
            logger.debug('\n')