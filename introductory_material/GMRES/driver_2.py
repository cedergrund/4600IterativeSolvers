# libraries 
import os
import logging
import time 
import numpy as np

# import files 
from scipy.sparse.linalg import gmres
import make_matrix

if __name__ == "__main__":
    # out_dir
    out_dir = os.path.join(os.getcwd(),'outputs')

    # logger
    logging.basicConfig(level=logging.DEBUG)
    logger = logging.getLogger(__name__)
    file_handler = logging.FileHandler(os.path.join(out_dir,'scipy_GMRES.txt'), mode='w')
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
    max_iter = n
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
            x, info = gmres(A, b, x0, restart=2000, tol = tol / np.linalg.norm(b))

            # timing 
            toc = time.time()
            time_elapsed = toc - tic

            # Logger info 
            metadata = {
                'seconds elapsed' : time_elapsed,
                'size of n' : n,
                'converged' : info == 0,
                'x0_zeros' : x0_zeros,
                'max_epochs' : max_epochs, 
                '||Ax - b||' : np.linalg.norm(A@x - b),
                'tol' : tol,
                'A' : np.shape(A),
                'x' : np.shape(x),
                'b' : np.shape(b)
            }
            logger.debug('metadata')
            for key, value in metadata.items():
                logger.debug(f'{key}: {value}')
            logger.debug('\n')