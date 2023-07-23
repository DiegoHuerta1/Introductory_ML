import numpy as np

# para soft svm with kernels
def circ(A):
    if len(A) != A.size:
        p = np.ones(len(A))
        a1 = A[:, 0] * 2**(1/2)
        a2 = A[:, 1] * 2**(1/2)
        a3 = A[:, 0] * A[:, 1] * 2**(1/2)
        a4 = A[:, 0] * A[:, 0]
        a5 = A[:, 1] * A[:, 1]
        A = np.column_stack([p, a1, a2, a3, a4, a5])
        return A
    elif len(A)== A.size:
        a1 = A[0] * 2**(1/2)
        a2 = A[1] * 2**(1/2)
        a3 = A[0] * A[1] * 2**(1/2)
        a4 = A[0] * A[0]
        a5 = A[1] * A[1]
        A = np.array([1, a1, a2, a3, a4, a5])
        return A

def kp2(a, b):
    return (1 + np.dot(a, b))**2
