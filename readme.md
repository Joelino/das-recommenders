# 01DAS Recommender systems referemce implementations

This repo contains few easy-to-implement recommender systems and their benchmark
on small movielens dataset.

# List of recommenders

## Basic baselines
1. Constant
2. Dataset mean
3. Item mean
4. User mean

## KNN recommenders
1. Basic KNN recommender
2. Normelized KNN recommender
3. Normalized & Weighted KNN recommender

# Others
1. iterative SVD recommender
2. EASE recommedner

# Evaluation metrics
1. RMSE
2. MAE
3. Recall @ 50

# How to run it yourself

1. Create virtualenv from `requirements.txt`.
2. Download [Movielens 100k dataset](http://files.grouplens.org/datasets/movielens/ml-latest-small.zip)
   and unpack it to folder `ml-latest-small`.
3. (Optional): Download [Movielens full dataset](http://files.grouplens.org/datasets/movielens/ml-latest.zip)
   and unpack it to folder `ml-latest-full`.
4. Run `Benchmarks.ipynb` notebook.
