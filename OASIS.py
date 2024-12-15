import numpy as np

from utils import calculate_scaling_factor


class OASIS:
    def __init__(self, tau=0.07, enforce_symmetric=True, enforce_psd=True, verbose=False):
        """
        Initializes the OasisMethod class.

        Args:
            tau (float): Maximum value for tau (step size).
            enforce_symmetric (bool): Whether to enforce the transformation matrix to be symmetric.
            enforce_psd (bool): Whether to enforce the transformation matrix to be positive semi-definite (PSD).
        """
        self.tau = tau
        self.enforce_symmetric = enforce_symmetric
        self.enforce_psd = enforce_psd
        self.verbose = verbose
        self.M = None  # Transformation matrix to be initialized later

    def _ensure_symmetric(self, M):
        """
        Ensures the matrix is symmetric.

        Args:
            M (ndarray): Matrix to be checked and made symmetric if necessary.

        Returns:
            ndarray: Symmetric matrix.
        """
        if np.allclose(M, M.T):
            if self.verbose:
                print("Matrix is symmetric.")
            return M
        return (M + M.T) / 2

    def _ensure_psd(self, M):
        """
        Ensures the matrix is positive semi-definite (PSD).

        Args:
            M (ndarray): Matrix to be checked and made PSD if necessary.

        Returns:
            ndarray: PSD matrix.
        """
        eigvals = np.linalg.eigvals(M)
        if np.all(eigvals >= 0):  # Check if all eigenvalues are non-negative
            if self.verbose:
                print("Matrix is PSD.")
            return M
        eigvals, eigvecs = np.linalg.eigh(M)
        eigvals = np.maximum(eigvals, 0)  # Clamp eigenvalues to be non-negative
        return eigvecs @ np.diag(eigvals) @ eigvecs.T

    def partial_fit(self, M, triplets):
        """
        Updates the transformation matrix using the OASIS algorithm on given triplets.

        Args:
            M (ndarray): Current transformation matrix.
            triplets (list of tuples): The triplets for training, where each triplet contains
                (anchor, positive, negative).

        Returns:
            ndarray: Updated transformation matrix.
        """
        self.M = M if M is not None else np.eye(len(triplets[0][0]))  # Initialize M if not provided
        n_iter = len(triplets)
        i = 0

        while i < n_iter:
            triplet = triplets[i]
            delta = np.array(triplet[1]) - np.array(triplet[2])
            loss = 1 - np.dot(np.dot(triplet[0], self.M), delta)

            if loss > 0:
                grad_w = np.outer(triplet[0], delta)
                fs = np.linalg.norm(grad_w, ord='fro') ** 2
                tau_val = loss / fs
                tau_i = np.minimum(self.tau, tau_val)
                self.M = np.add(self.M, tau_i * grad_w)

            i += 1

            if self.enforce_symmetric:
                self.M = self._ensure_symmetric(self.M)
            if self.enforce_psd:
                self.M = self._ensure_psd(self.M)

            if self.verbose:
                sf = calculate_scaling_factor(self.M)
                print(f"Scaling Factor of M: {sf}")

        return self.M

    def get_matrix(self):
        """
        Returns the learned matrix M.

        Returns:
            ndarray: The matrix M.
        """
        return self.M