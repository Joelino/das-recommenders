from tqdm import tqdm
import numpy as np

def compute_recs(y_pred, user_row):
    sorted = np.argsort(y_pred)
    # remove nans as they sorted as highest values
    sorted = sorted[~np.isnan(y_pred[sorted])]
    # remove already seen items in train set
    already_seen = np.nonzero(~np.isnan(user_row))
    return sorted[~np.isin(sorted, already_seen)]


def get_recall_at_k(recs, user_test_ratings, k):
    hits = user_test_ratings.item_id.isin(recs[-k:]).sum()
    recall = hits / len(user_test_ratings)
    return recall


def get_precision_at_k(recs, user_test_ratings, k):
    hits = user_test_ratings.item_id.isin(recs[-k:]).sum()
    recall = hits / k
    return recall


def evaluate(recommender, train_ratings_matrix, test_ratings, fraction=1.0):
    deviations = []
    recalls = []
    precisions = []
    n_users = round(train_ratings_matrix.shape[0] * fraction)
    for user_id, user_row in tqdm(
        enumerate(train_ratings_matrix[:n_users, :]), total=n_users
    ):
        y_pred = recommender.predict(user_row, user_id)
        user_test_ratings = test_ratings[test_ratings.user_id == user_id]
        recs = compute_recs(y_pred, user_row)
        recall = get_recall_at_k(recs, user_test_ratings, 50)
        precision = get_recall_at_k(recs, user_test_ratings, 5)
        recalls.append(recall)
        precisions.append(precision)
        for _, test_rating in user_test_ratings.iterrows():
            deviation = y_pred[int(test_rating.item_id)] - test_rating.rating
            deviations.append(deviation)
    deviations = np.array(deviations)
    rmse = np.sqrt(np.nanmean(deviations ** 2))
    mae = np.nanmean(np.abs(deviations))
    recall = np.array(recalls).mean()
    precision = np.array(precisions).mean()
    print(f"RMSE: {rmse}, MAE: {mae}, Recall at 50: {recall}, Precision at 5: {precision}")
