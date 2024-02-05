import numpy as np
import matplotlib.pyplot as plt

def get_relevant_non_redundant_features(X:np.ndarray, y:np.ndarray, feat_names:np.array, relevance_th:float=0.5, redundancy_th:float=0.5, plot:bool=False):
    """
    This function sorts the features based on their relevance to the target variable and then selects the most relevant and non-redundant features.

    Parameters
    ----------
    X : np.ndarray
        The feature matrix.
    y : np.ndarray
        The target vector.
    feat_names : np.ndarray
        The list of feature names.
    relevance_th : float, optional
        The threshold for selecting the most relevant features, by default 0.5
    redundancy_th : float, optional
        The threshold for selecting the non-redundant features, by default 0.5
    plot : bool, optional
        Whether to plot the feature relevance bar graph, by default False

    Returns
    -------
    Tuple[np.ndarray, List[str], np.ndarray]
        The feature matrix with the selected relevant and non-redundant features, the list of feature names, and the feature relevance vector.
    """
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
        pass

    return X, feat_names, feature_relevance


def sort_feature_by_relevance(feat_matrix, target_array, th=0):
    """
    This function sorts the features based on their relevance to the target variable.

    Parameters
    ----------
    feat_matrix : np.ndarray
        The feature matrix.
    target_array : np.ndarray
        The target vector.
    th : float, optional
        The threshold for selecting the most relevant features, by default 0.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        The index array of the sorted features and the feature relevance vector.

    """
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
    """
    This function selects the most relevant and non-redundant features from a feature matrix.

    Parameters
    ----------
    feat_matrix : np.ndarray
        The feature matrix.
    th : float, optional
        The threshold for selecting the non-redundant features, by default 0.5

    Returns
    -------
    List[int]
        The index list of the selected relevant and non-redundant features.

    """
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