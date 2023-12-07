import gensim
import numpy as np
import pandas as pd
from das_recommenders.collaborative_filtering.knn import WeightedNormKNNRecommender


class W2VRecommender(WeightedNormKNNRecommender):

    def fit(self, ratings, ratings_matrix):
        sentences = generate_sentences(ratings)
        self.w2v_model = gensim.models.Word2Vec(
            sentences=sentences,
            vector_size=20,
            window=10,
            min_count=1,
            sg=1,
            workers=10
        )

        matrix = ratings_matrix.copy()
        matrix = matrix - np.nanmean(matrix, axis=1).reshape(-1, 1)
        self.ratings_matrix = ratings_matrix
        self.norm_ratings_matrix = pd.DataFrame(matrix)

    def _compute_similarities(self, matrix, user_id):
        sims = []
        for i in range(matrix.shape[0]):
             sims.append(self.w2v_model.wv.similarity(user_id, i))
        return np.array(sims)


    def predict(self, user_row, user_id):
        user_mean = np.nanmean(user_row)
        row = user_row - user_mean

        similarities = self._compute_similarities(self.norm_ratings_matrix, user_id)
        neighbours = self._find_neighbours(similarities)
        predicted_deviations = self._predict_as_weight_mean_of_neighbours(
            similarities, neighbours
        )

        return predicted_deviations + user_mean


def generate_sentences(ratings):
    sentences = []
    for item in ratings.item_id.unique():
        item_ratings = ratings[
            (ratings['item_id'] == item) &
            (ratings['rating'] >= 4)
        ]
        if not item_ratings.empty:
            sentence = []
            for _, rating in item_ratings.iterrows():
                sentence.append(rating['user_id'])

            sentences.append(sentence)
    return sentences