import numpy as np


class ConstantRecommender:
    def fit(self, constant):
        self.constant = constant

    def predict(self, user_row, _):
        return np.full(shape=user_row.shape, fill_value=self.constant)


class ItemMeanRecommender:
    def fit(self, ratings_matrix):
        self.predictions = np.nanmean(ratings_matrix, axis=0)

    def predict(self, user_row, _):
        return self.predictions


class UserMeanRecommender:
    def fit(self):
        raise

    def predict(self, user_row, _):
        return np.full(shape=user_row.shape, fill_value=np.nanmean(user_row))
