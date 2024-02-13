import numpy as np
from culingam.directlingam import DirectLiNGAM

# [[ 0.          0.          0.          2.99982982  0.          0.        ]
#  [ 2.99997222  0.          2.00008518  0.          0.          0.        ]
#  [ 0.          0.          0.          5.99981965  0.          0.        ]
#  [ 0.          0.          0.          0.          0.          0.        ]
#  [ 7.99857006  0.         -0.99911522  0.          0.          0.        ]
#  [ 3.99974733  0.          0.          0.          0.          0.        ]]
# [3, 0, 2, 5, 4, 1]

def main():
    np.random.seed(42)
    size = 100000
    x3 = np.random.uniform(size=size)
    x0 = 3.0*x3 + np.random.uniform(size=size)
    x2 = 6.0*x3 + np.random.uniform(size=size)
    x1 = 3.0*x0 + 2.0*x2 + np.random.uniform(size=size)
    x5 = 4.0*x0 + np.random.uniform(size=size)
    x4 = 8.0*x0 - 1.0*x2 + np.random.uniform(size=size)

    X = np.array([x0, x1, x2, x3, x4, x5]).T

    dlm = DirectLiNGAM(12)
    dlm.fit(X, disable_tqdm=False)

    np.set_printoptions(precision=3, suppress=True)

    print(dlm._adjacency_matrix)
    print(dlm.causal_order_)

if __name__ == "__main__":
    main()
