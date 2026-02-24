'''k_fold.py
Duilio Lucio, Vivian Hu
CS343: Neural Networks
Project 1: Single Layer Networks
K-Fold cross validation (STRATIFIED)
'''
import numpy as np


def kfold_cv(X, y, model_class, k=5, n_epochs=1000, lr=0.01, seed=0, **model_kwargs):
    """
    Runs STRATIFIED K-fold cross validation for any ADALINE-style model.

    Ensures each fold has similar class proportions.
    """

    rng = np.random.default_rng(seed)
    y = np.asarray(y)
    X = np.asarray(X)

    classes = np.unique(y)

    # ---- Build stratified folds ----
    folds = [[] for _ in range(k)]

    for c in classes:
        c_idx = np.where(y == c)[0]      # indices of class c
        c_idx = rng.permutation(c_idx)   # shuffle within class

        splits = np.array_split(c_idx, k)

        for i in range(k):
            folds[i].append(splits[i])

    # combine class pieces into full folds
    folds = [np.concatenate(parts) for parts in folds]

    accuracies = []

    for i in range(k):

        test_idx = folds[i]
        train_idx = np.concatenate([folds[j] for j in range(k) if j != i])

        X_train, y_train = X[train_idx], y[train_idx]
        X_test, y_test = X[test_idx], y[test_idx]

        # ---- Standardize using TRAIN statistics only ----
        mean = X_train.mean(axis=0)
        std = X_train.std(axis=0) + 1e-8

        X_train = (X_train - mean) / std
        X_test = (X_test - mean) / std

        # ---- Train model ----
        model = model_class(**model_kwargs)
        model.fit(X_train, y_train, n_epochs=n_epochs, lr=lr, r_seed=seed)

        # ---- Evaluate ----
        y_pred = model.predict(X_test)
        acc = model.accuracy(y_test, y_pred)

        accuracies.append(acc)

    return np.array(accuracies)