import numpy as np

def gepp(A, b):
    """
    Gaussian Elimination with Partial Pivoting (GEPP)
    Solves the linear system A x = b.

    Parameters
    ----------
    A : np.ndarray
        Coefficient matrix (n x n)
    b : np.ndarray
        Right-hand side vector (n,)

    Returns
    -------
    x : np.ndarray
        Solution vector (n,)
    """

    A = A.astype(float).copy()
    b = b.astype(float).copy()
    n = len(b)

    # Forward elimination
    for k in range(n - 1):
        # Find pivot
        max_index = np.argmax(np.abs(A[k:, k])) + k
        max_val = abs(A[max_index, k])
        if max_val == 0:
            raise ValueError("A is singular.")

        # Swap rows in A and b
        if max_index != k:
            A[[k, max_index], k:] = A[[max_index, k], k:]
            b[[k, max_index]] = b[[max_index, k]]

        # Eliminate entries below pivot
        i = np.arange(k + 1, n)
        mult = A[i, k] / A[k, k]
        A[i, k:] -= np.outer(mult, A[k, k:])
        b[i] -= mult * b[k]

    # Back substitution
    x = np.zeros(n)
    for k in range(n - 1, -1, -1):
        x[k] = (b[k] - np.dot(A[k, k + 1:], x[k + 1:])) / A[k, k]

    return x

# Example usage
A = np.array([[3, 2, -4],
              [2, 3, 3],
              [5, -3, 1]], dtype=float)
b = np.array([3, 15, 14], dtype=float)

x = gepp(A, b)
print("Solution x:", x)
