import ctypes
import numpy as np
cuda_lib = ctypes.CDLL('./unit.so') #check that path exists
import time

def test1():
    # Define the wrapper function
    run_compute_mean = cuda_lib.run_compute_mean
    run_compute_mean.argtypes = [np.ctypeslib.ndpointer(dtype=np.float64),
                                np.ctypeslib.ndpointer(dtype=np.float64),
                                ctypes.c_int,
                                ctypes.c_int]


    size = 1000000
    blockSize = 256

    # Generate dummy data
    data = np.random.rand(size).astype(np.float64)
    mean = np.zeros(1, dtype=np.float64)

    # Call the wrapper function
    run_compute_mean(data, mean, size, blockSize)

    # Calculate expected mean using NumPy
    expected_mean = np.mean(data)

    # Check the result
    # print(mean[0], expected_mean);
    assert np.isclose(mean[0], expected_mean), f"Result {mean[0]} does not match expected {expected_mean}"

    print("Test passed")


def test2():
    run_compute_covariance_variance = cuda_lib.run_compute_covariance_variance
    run_compute_covariance_variance.argtypes = [np.ctypeslib.ndpointer(dtype=np.float64),
                                            np.ctypeslib.ndpointer(dtype=np.float64),
                                            np.ctypeslib.ndpointer(dtype=np.float64),
                                            np.ctypeslib.ndpointer(dtype=np.float64),
                                            np.ctypeslib.ndpointer(dtype=np.float64),
                                            np.ctypeslib.ndpointer(dtype=np.float64),
                                            ctypes.c_int,
                                            ctypes.c_int]

    size = 4177
    blockSize = 256

    xi = np.random.rand(size).astype(np.float64)
    xj = np.random.rand(size).astype(np.float64)
    mean_xi = np.array([np.mean(xi)], dtype=np.float64)
    mean_xj = np.array([np.mean(xj)], dtype=np.float64)
    covariance = np.zeros(1, dtype=np.float64)
    variance = np.zeros(1, dtype=np.float64)

    # Call the wrapper function
    run_compute_covariance_variance(xi, xj, mean_xi, mean_xj, covariance, variance, size, blockSize)

    # Calculate expected covariance and variance using NumPy
    expected_covariance = np.sum((xi - mean_xi) * (xj - mean_xj))
    expected_variance = np.sum((xj - mean_xj)**2)

    # print(covariance[0], expected_covariance)
    # print(variance[0], expected_variance)
    assert np.isclose(covariance[0], expected_covariance), "Covariance test failed {}, {}".format(covariance[0], expected_covariance)
    assert np.isclose(variance[0], expected_variance), "Variance test failed"

    print("Test passed")

def test3():
    run_element_wise_division = cuda_lib.run_element_wise_division
    run_element_wise_division.argtypes = [np.ctypeslib.ndpointer(dtype=np.float64),
                                        np.ctypeslib.ndpointer(dtype=np.float64),
                                        np.ctypeslib.ndpointer(dtype=np.float64),
                                        ctypes.c_int,
                                        ctypes.c_int]

    n = 4177
    blockSize = 256

    r = np.random.rand(n).astype(np.float64)
    constant_std = np.array([np.random.rand()], dtype=np.float64)
    result = np.zeros(n, dtype=np.float64)

    # Call the wrapper function
    run_element_wise_division(r, constant_std, result, n, blockSize)

    # Calculate expected result using NumPy
    expected_result = np.divide(r, constant_std) if constant_std != 0 else np.zeros(n, dtype=np.float64)

    assert np.allclose(result, expected_result), "element_wise_division Test failed"

    print("Test passed")

def test4():
    # Define the wrapper function
    run_compute_residual = cuda_lib.run_compute_residual
    run_compute_residual.argtypes = [np.ctypeslib.ndpointer(dtype=np.float64),
                                    np.ctypeslib.ndpointer(dtype=np.float64),
                                    np.ctypeslib.ndpointer(dtype=np.float64),
                                    np.ctypeslib.ndpointer(dtype=np.float64),
                                    ctypes.c_int,
                                    ctypes.c_int]

    size = 4177
    blockSize = 256

    xi = np.random.rand(size).astype(np.float64)
    xj = np.random.rand(size).astype(np.float64)
    scaling_factor = np.array([np.random.rand()], dtype=np.float64)
    residual = np.zeros(size, dtype=np.float64)

    # Call the wrapper function
    run_compute_residual(xi, xj, scaling_factor, residual, size, blockSize)

    # Calculate expected result using NumPy
    expected_residual = xi - scaling_factor * xj

    assert np.allclose(residual, expected_residual), "compute_residual Test failed"

    print("Test passed")

def test5():
    run_compute_std = cuda_lib.run_compute_std
    run_compute_std.argtypes = [np.ctypeslib.ndpointer(dtype=np.float64),
                                np.ctypeslib.ndpointer(dtype=np.float64),
                                np.ctypeslib.ndpointer(dtype=np.float64),
                                ctypes.c_int,
                                ctypes.c_int]

    size = 4177
    blockSize = 256

    A = np.random.rand(size).astype(np.float64)
    mean = np.array([np.mean(A)], dtype=np.float64)
    std = np.array([0.0], dtype=np.float64)

    # Call the wrapper function
    run_compute_std(A, mean, std, size, blockSize)

    # Calculate expected standard deviation using NumPy
    expected_std = np.std(A)
    # print(std[0], expected_std)
    assert np.isclose(std[0], expected_std), "compute_std Test failed"

    print("Test passed")


def test6():
    run_calculate_statistics = cuda_lib.run_calculate_statistics
    run_calculate_statistics.argtypes = [np.ctypeslib.ndpointer(dtype=np.float64),
                                        ctypes.c_int,
                                        ctypes.c_int,
                                        np.ctypeslib.ndpointer(dtype=np.float64),
                                        np.ctypeslib.ndpointer(dtype=np.float64),
                                        ctypes.c_int]

    m, n = 4177, 11
    blockSize = 256

    A = np.random.rand(m, n).astype(np.float64)
    means = np.zeros(n, dtype=np.float64)
    stds = np.zeros(n, dtype=np.float64)

    # Call the wrapper function
    run_calculate_statistics(A, m, n, means, stds, blockSize)

    # Calculate expected results using NumPy
    expected_means = np.mean(A, axis=0)
    expected_stds = np.std(A, axis=0)

    assert np.allclose(means, expected_means), "Means test failed"
    assert np.allclose(stds, expected_stds), "Standard deviations test failed"

    print("Test passed")

def test7():
    run_standardize_column = cuda_lib.run_standardize_column
    run_standardize_column.argtypes = [np.ctypeslib.ndpointer(dtype=np.float64),
                                    ctypes.c_int,
                                    ctypes.c_int,
                                    np.ctypeslib.ndpointer(dtype=np.float64),
                                    np.ctypeslib.ndpointer(dtype=np.float64),
                                    ctypes.c_int]

    m, n = 417700, 10
    unitthreads = 16

    A = np.random.rand(m, n).astype(np.float64)
    means = np.mean(A, axis=0).astype(np.float64)
    stds = np.std(A, axis=0).astype(np.float64)

    expected_A = np.zeros((m, n))
    import time
    start_time = time.time()
    for k in range(n):
        expected_A[:, k] = (A[:, k] - np.mean(A[:, k])) / np.std(A[:, k]);
    time_for_loop = time.time() - start_time
    start_time = time.time()
    run_standardize_column(A, m, n, means, stds, unitthreads)
    time_run_standardize_column = time.time() - start_time

    assert np.allclose(A, expected_A), "Test failed"

    # print("Test passed", time_for_loop, time_run_standardize_column)
    print("Test passed")


def test8():
    run_compute_log_cosh = cuda_lib.run_compute_log_cosh
    run_compute_log_cosh.argtypes = [np.ctypeslib.ndpointer(dtype=np.float64),
                                    np.ctypeslib.ndpointer(dtype=np.float64),
                                    ctypes.c_int,
                                    ctypes.c_int]
    size = 4177
    blockSize = 256

    u = np.random.rand(size).astype(np.float64) * 10
    log_cosh_sum = np.array([0.0], dtype=np.float64)
    run_compute_log_cosh(u, log_cosh_sum, size, blockSize)
    expected_log_cosh_sum = np.sum(np.log(np.cosh(u)))

    # print(log_cosh_sum[0], expected_log_cosh_sum);
    assert np.isclose(log_cosh_sum[0], expected_log_cosh_sum), "Test failed"

    print("Test passed")

def test9():
    run_compute_u_exp = cuda_lib.run_compute_u_exp
    run_compute_u_exp.argtypes = [np.ctypeslib.ndpointer(dtype=np.float64),
                                np.ctypeslib.ndpointer(dtype=np.float64),
                                ctypes.c_int,
                                ctypes.c_int]

    size = 417700
    blockSize = 256

    u = np.random.rand(size).astype(np.float64) * 10  # Random values scaled
    u_exp_sum = np.array([0.0], dtype=np.float64)
    run_compute_u_exp(u, u_exp_sum, size, blockSize)
    expected_u_exp_sum = np.sum(u * np.exp(-0.5 * u * u))

    # print(u_exp_sum[0], expected_u_exp_sum);
    assert np.isclose(u_exp_sum[0], expected_u_exp_sum), "Test failed"

    print("Test passed")


def test10():
    run_process_column = cuda_lib.run_process_column
    run_process_column.argtypes = [ctypes.POINTER(ctypes.c_double),
                                ctypes.POINTER(ctypes.c_double),
                                ctypes.c_int,
                                ctypes.c_int,
                                ctypes.c_int,
                                ctypes.c_int]

    m, n = 4177, 11
    col_idx = 2
    threads = 256

    d_X = np.random.rand(m, n).astype(np.float64)
    column =  np.zeros(m, dtype=np.float64)

    expected_column = d_X[:, col_idx]

    c_array = d_X.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
    d_column = column.ctypes.data_as(ctypes.POINTER(ctypes.c_double))

    run_process_column(c_array, d_column, m, n, col_idx, threads)
    column = np.frombuffer(ctypes.cast(d_column, ctypes.POINTER(ctypes.c_double * m)).contents, dtype=np.float64)
    assert np.allclose(column, expected_column), "Test failed"

    print("Test passed")


def test11():
    end_end_residual = cuda_lib.end_end_residual
    end_end_residual.argtypes = [np.ctypeslib.ndpointer(ctypes.c_double),
                                ctypes.c_int,
                                ctypes.c_int,
                                ctypes.c_int,
                                np.ctypeslib.ndpointer(ctypes.c_int), ctypes.c_int]

    def _residual(xi, xj):
            return xi - (np.cov(xi, xj, ddof=1)[0, 1] / np.var(xj, ddof=1)) * xj

    m, n = 1000, 110
    col_idx = 2
    threads = 256

    d_X = np.random.rand(m, n).astype(np.float64)
    U = np.array(range(n)).astype(np.int32)

    expected_arr = d_X.copy()

    for i in U:
        if i != col_idx:
            expected_arr[:, i] = _residual(expected_arr[:, i], expected_arr[:, col_idx])

    end_end_residual(d_X, m, n, col_idx, U, n)
    d_X = d_X.reshape(m, n)

    # print(d_X, expected_arr);
    assert np.allclose(d_X, expected_arr,), "Test failed"
    print("Test passed")


def test12():
    cuda_causal_order = cuda_lib.causal_order
    cuda_causal_order.argtypes = [ctypes.POINTER(ctypes.c_double), ctypes.c_int, ctypes.c_int]

    X = np.random.rand(1000, 10);
    U = np.array(range(10));

    cols = len(U)
    rows = len(X)

    arr = np.ascontiguousarray(X[:, np.array(U)])
    arr = arr.ravel(order='C')

    U = U.astype(np.float64)
    from lingam_cuda import causal_order

    res = causal_order(X, rows, cols)

    c_array = arr.ctypes.data_as(ctypes.POINTER(ctypes.c_double))

    output_buffer = (ctypes.c_double * cols)()
    cuda_causal_order(c_array, rows, cols, output_buffer)
    M_list = np.frombuffer(output_buffer, dtype=np.float64)
    print(res)
    assert np.allclose(res, M_list), "Test failed"
    print("Test passed")

if __name__ == "__main__":
    test1()
    test2()
    test3()
    test4()
    test5()
    test6()
    test7()
    test8()
    test9()
    test10()
    test11()
    test12()
