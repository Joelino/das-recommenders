import numpy as np
import pandas as pd
import warnings

from sklearn.metrics.pairwise import cosine_similarity

K_NEAREST = 25
np.seterr(divide="ignore", invalid="ignore")
warnings.filterwarnings(action="ignore", message="Mean of empty slice")


class BasicKNNRecommender:
    def __init__(self, similarity_type, k=K_NEAREST):
        self.similarity_type = similarity_type
        self.k = k

    def fit(self, ratings_matrix):
        self.ratings_matrix = pd.DataFrame(ratings_matrix.copy())

    def _find_neighbours(self, similarities):
        return np.argsort(np.nan_to_num(similarities, 0),)[::-1][: self.k]

    def predict(self, user_row, _):
        if self.similarity_type == "correlation":
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=RuntimeWarning)
                with np.errstate(all="ignore"):
                    similarities = self.ratings_matrix.corrwith(
                        pd.Series(user_row), axis=1
                    ).values
        else:
            raise (f"Unknown similarity type: {self.similarity_type}!")

        neighbours = self._find_neighbours(similarities)
        predictions = np.nanmean(self.ratings_matrix.values[neighbours, :], axis=0)
        return predictions


class NormalizedKNNRecommender(BasicKNNRecommender):
    def fit(self, ratings_matrix):
        matrix = ratings_matrix.copy()
        matrix = matrix - np.nanmean(matrix, axis=1).reshape(-1, 1)
        self.ratings_matrix = ratings_matrix
        self.norm_ratings_matrix = pd.DataFrame(matrix)

    def _compute_similarities(self, matrix, row):
        if self.similarity_type == "correlation":
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=RuntimeWarning)
                with np.errstate(all="ignore"):
                    return matrix.corrwith(pd.Series(row), axis=1).values
        elif self.similarity_type == "cosine":
            return cosine_similarity(
                np.nan_to_num(matrix.copy(), 0),
                np.nan_to_num(row.copy(), 0).reshape(1, -1),
            )[:, 0]
        else:
            raise (f"Unknown similarity type: {self.similarity_type}!")

    def predict(self, user_row, _):
        user_mean = np.nanmean(user_row)
        row = user_row - user_mean

        similarities = self._compute_similarities(self.norm_ratings_matrix, row)
        neighbours = self._find_neighbours(similarities)
        predictions = (
            np.nanmean(self.norm_ratings_matrix.values[neighbours, :], axis=0)
            + user_mean
        )
        return predictions


class WeightedNormKNNRecommender(NormalizedKNNRecommender):
    def _predict_as_weight_mean_of_neighbours(self, similarities, neighbours):
        neighbour_similarities = similarities[neighbours].reshape(-1, 1)
        neighbour_ratings = self.norm_ratings_matrix.values[neighbours, :]
        weighted_sums = np.nansum(neighbour_similarities * neighbour_ratings, axis=0)

        # normalize by sum of weights that will not be resolved to nan
        normalization_factors = np.nansum(
            np.abs(neighbour_similarities) * ~np.isnan(neighbour_ratings), axis=0
        )
        predicted_deviations = weighted_sums / normalization_factors
        return predicted_deviations

    def predict(self, user_row, _):
        user_mean = np.nanmean(user_row)
        row = user_row - user_mean

        similarities = self._compute_similarities(self.norm_ratings_matrix, row)
        neighbours = self._find_neighbours(similarities)
        predicted_deviations = self._predict_as_weight_mean_of_neighbours(
            similarities, neighbours
        )

        return predicted_deviations + user_mean
