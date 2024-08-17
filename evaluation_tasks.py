import json
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn import linear_model
from sklearn.model_selection import KFold
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, r2_score, mean_squared_error, mean_absolute_error

def compute_metrics(y_pred, y_test):
    y_pred[y_pred < 0] = 0
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    return mae, np.sqrt(mse), r2

def regression(X_train, y_train, X_test, alpha):
    reg = linear_model.Ridge(alpha=alpha)
    X_train = np.array(X_train, dtype=float)
    y_train = np.array(y_train, dtype=float)
    reg.fit(X_train, y_train)
    return reg.predict(X_test)

def k_fold_prediction(X, Y):
    kf = KFold(n_splits=5)
    y_preds = []
    y_truths = []
    for train_index, test_index in kf.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = Y[train_index], Y[test_index]
        y_pred = regression(X_train, y_train, X_test, 1)
        y_preds.append(y_pred)
        y_truths.append(y_test)
    return np.concatenate(y_preds), np.concatenate(y_truths)

def evaluate_predictions(embs, labels, display=False):
    y_pred, y_test = k_fold_prediction(embs, labels)
    mae, rmse, r2 = compute_metrics(y_pred, y_test)
    if display:
        print(f"MAE: {mae:.2f}")
        print(f"RMSE: {rmse:.2f}")
        print(f"R2: {r2:.4f}")
    return mae, rmse, r2

def classify_land_usage(emb, display=False):
    lu_label_filename = "./Data/landusage.json"
    cd = json.load(open(lu_label_filename))
    cd_labels = np.array([cd[str(i)] for i in range(69)])
    kmeans = KMeans(n_clusters=14, random_state=3,n_init=10)
    emb_labels = kmeans.fit_predict(emb)
    nmi = normalized_mutual_info_score(cd_labels, emb_labels)
    ars = adjusted_rand_score(cd_labels, emb_labels)
    if display:
        print(f"Emb NMI: {nmi:.3f}")
        print(f"Emb ARS: {ars:.3f}")
    return nmi, ars

def perform_evaluation(embs, display=True):
    if display:
        print("### Popularity Prediction ###")
    population_label = np.load("./Data/population.npy", allow_pickle=True)
    pop_mae, pop_rmse, pop_r2 = evaluate_predictions(embs, population_label, display=display)

    if display:
        print("### Check-in Prediction ###")
    check_in_label = np.load("./Data/check_in.npy")
    check_mae, check_rmse, check_r2 = evaluate_predictions(embs, check_in_label, display=display)

    if display:
        print("### Land Usage Prediction ###")
    nmi, ars = classify_land_usage(embs, display=display)

    if display:
        print("### Summary ###\n")
        print(f"Popularity Prediction - MAE: {pop_mae:.2f}, RMSE: {pop_rmse:.2f}, R2: {pop_r2:.4f}")
        print(f"Check-in Prediction - MAE: {check_mae:.2f}, RMSE: {check_rmse:.2f}, R2: {check_r2:.4f}")
        print(f"Land Usage Prediction - NMI: {nmi:.4f}, ARS: {ars:.4f}")

    return pop_mae, pop_rmse, pop_r2, check_mae, check_rmse, check_r2, nmi, ars
