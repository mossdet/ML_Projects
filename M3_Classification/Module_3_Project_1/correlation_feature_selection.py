import numpy as np
import matplotlib.pyplot as plt

def get_relevant_non_redundant_features(X, y, feat_names, relevance_th=0.5, redundancy_th=0.5, plot=False):

    # Get relevant features
    relevance_sort_idxs, feature_relevance = sort_feature_by_relevance(feat_matrix=X, target_array=y, th=relevance_th)
    X = X[:,relevance_sort_idxs]
    y = y[relevance_sort_idxs]
    feat_names = feat_names[relevance_sort_idxs]


    # Get non-redundant features
    non_redundant_feats_idx = select_sorted_non_redundant_features(feat_matrix=X, th=redundancy_th)
    X = X[:,non_redundant_feats_idx]
    y = y[non_redundant_feats_idx]
    feat_names = feat_names[non_redundant_feats_idx]
    feature_relevance = feature_relevance[non_redundant_feats_idx]

    if plot:
        plt.figure(figsize=(10, 10))
        plt.bar(x=np.arange(len(feature_relevance)) + 1, height=feature_relevance, tick_label=feat_names)
        plt.xticks(rotation=60)
        plt.title("Relevant Non-Redundant Features")
        plt.show(block=False)

    return X, feat_names, feature_relevance


def sort_feature_by_relevance(feat_matrix, target_array, th=0):
    nr_feats = feat_matrix.shape[1]
    feature_relevance = np.array([])
    for fi in range(nr_feats):
        rho = np.abs(np.corrcoef(feat_matrix[:,fi], target_array))[0, 1]
        feature_relevance = np.append(feature_relevance, rho)

    sort_idxs = np.flip(np.argsort(feature_relevance))
    feature_relevance = feature_relevance[sort_idxs]

    sel_array = feature_relevance > th
    sort_idxs = sort_idxs[sel_array]
    feature_relevance = feature_relevance[sel_array]

    return sort_idxs, feature_relevance


def select_sorted_non_redundant_features(feat_matrix, th=0.5):
    sel_feats_idx = []
    nr_feats = feat_matrix.shape[1]
    for i in range(nr_feats):
        if len(sel_feats_idx) == 0:
            sel_feats_idx.append(i)
            continue

        non_redundant = True
        for srfi in sel_feats_idx:
            if i == srfi:
                continue
            feat_to_test = feat_matrix[:, i]
            sel_relevant_feature = feat_matrix[:, srfi]
            rho = np.abs(np.corrcoef(feat_to_test, sel_relevant_feature))[0, 1]
            if rho > th:
                non_redundant = False
                break

        if non_redundant:
            sel_feats_idx.append(i)

    return sel_feats_idx