"""
This is modified implementation of: https://github.com/Darel13712/ease_rec
Original paper: https://arxiv.org/abs/1905.03375
"""

from scipy.sparse import csr_matrix
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm


class EASE:
    def fit(self, ratings_matrix, lambda_: float = 0.5):
        X = ratings_matrix.copy()
        # matrix multiplications need matrix with NaNs
        X = np.nan_to_num(X, 0)
        self.max = np.abs(np.nanmax(X))
        X = X / self.max  # normalize by max value
        G = X.T.dot(X)
        diagIndices = np.diag_indices(G.shape[0])
        G[diagIndices] += lambda_
        P = np.linalg.inv(G)
        B = P / (-np.diag(P))
        B[diagIndices] = 0

        self.B = B
        self.pred = X.dot(B)

    def predict(self, user_row, user_id):
        user_mean = np.nanmean(user_row)
        return self.pred[user_id, :] * self.max + user_mean
