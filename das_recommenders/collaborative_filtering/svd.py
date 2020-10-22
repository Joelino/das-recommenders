import numpy as np
import pandas as pd
from scipy.sparse.linalg import svds


class SVDRecommender:
    def __init__(self, k):
        self.k = k

    def fit(self, ratings_matrix):
        self.ratings_matrix = ratings_matrix
        matrix = ratings_matrix.copy()
        matrix = matrix - np.nanmean(matrix, axis=1).reshape(-1, 1)
        self.norm_ratings_matrix = pd.DataFrame(matrix)

        print("performing SVD..")

        def fill_with_row_means(array):
            row_means = np.nanmean(array, axis=1)
            means_matrix = np.tile(row_means.reshape(-1, 1), reps=(1, array.shape[1]))
            return np.where(np.isnan(array), means_matrix, array)

        U_k, sigma_k, Vt_k = svds(fill_with_row_means(matrix.copy()), k=self.k)

        last_err = 3
        this_err = 2
        while last_err - this_err > 0.01:
            last_err = this_err
            R_k = U_k @ (np.eye(self.k) * sigma_k) @ Vt_k
            matrix2 = np.where(np.isnan(matrix), R_k, matrix)
            U_k, sigma_k, Vt_k = svds(matrix2, k=self.k)
            this_err = np.nanmean(np.abs(matrix - R_k))
            print(f"{last_err} - {this_err}")
        self.users_factors = pd.DataFrame(U_k)
        self.sigma = sigma_k * np.eye(sigma_k.shape[0])
        self.items_factors = Vt_k
        self.R_k = R_k

    def predict(self, user_row, user_id):
        user_mean = np.nanmean(user_row)

        # this code block could be used to "fold in" user unseen in training data
        # user_factors = np.asarray(
        #     np.nan_to_num(user_row.copy() - user_mean, 0)
        #     @ self.items_factors.T
        #     @ np.linalg.inv(self.sigma)
        # ).flatten()
        # pred = user_factors @ self.sigma @ self.items_factors

        return self.R_k[user_id, :] + user_mean
        return pred + user_mean
