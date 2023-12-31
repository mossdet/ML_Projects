import numpy as np
import optuna
from scipy import stats
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score


def kmeans_cluster(img):
    # Flatten Image
    nr_color_cols = 3
    if len(img.shape) < 3:
        nr_color_cols = 1

    flat_image = img.reshape((-1, nr_color_cols))

    # Normalize Features
    scaler = StandardScaler()
    flat_image = scaler.fit_transform(flat_image)

    # K-Means
    inertias_log = {'k':[], 'inertia':[]}
    for k in range(2, 51):
        model = KMeans(n_clusters=k, init='k-means++', n_init='auto', max_iter=1000).fit(flat_image)
        inertias_log['k'].append(k)
        inertias_log['inertia'].append(model.inertia_)

    #  Find best K automatically basd on inertia decrease at each K
    last_inertia = 1e16
    best_k = 0
    for i, current_inertia in enumerate(inertias_log['inertia']):

        k = inertias_log['k'][i]                  
        inertia_desc = (last_inertia-current_inertia) > 0
        inertia_decrease_th = ((last_inertia-current_inertia)/last_inertia) > 0.05

        #print("K = ", k)
        #print("current_inertia = ", current_inertia)
        #print("last_inertia = ", last_inertia)
        #print("Inertia decrease = ", ((last_inertia-current_inertia)/last_inertia))

        if inertia_desc and inertia_decrease_th:
            best_k = k
        else:
            #print("Inertia not decreasing anymore at k = ", k)
            break
        
        last_inertia = current_inertia

    best_model = KMeans(n_clusters=best_k, init='k-means++', n_init='auto', max_iter=1000).fit(flat_image)
    #best_model = find_best_k_means_model(flat_image)
    clustered_labels = best_model.labels_

    # get number of segments
    labels = np.unique(clustered_labels)
    segments = labels
    print('Number of segments: ', segments.shape[0])

    # Assign the aggregated color to each segment

    flat_image = scaler.inverse_transform(flat_image)
    flat_segm_image = np.zeros(flat_image.shape)
    for i, label in enumerate(labels):
        row_sel = clustered_labels == label
        cluster_color = np.mean(flat_image[row_sel, :], axis=0)
        flat_segm_image[row_sel, :] = cluster_color

    flat_segm_image = np.uint8(flat_segm_image)
    segmented_image = flat_segm_image.reshape((img.shape))

    return segmented_image, flat_segm_image, clustered_labels


# Objective function for hyperparameter tuning of DecisionTree
def objective(trial, X, n_clusters, random_state):
    params = {        
    "init": trial.suggest_categorical("init", ["k-means++", "random"]),
    "tol": trial.suggest_float("tol", 1e-9, 1e9, log=True),
    "algorithm": trial.suggest_categorical("algorithm", ["lloyd", "elkan"]),
    # Constants
    "n_clusters": trial.suggest_categorical("n_clusters", [n_clusters]),
    "n_init": trial.suggest_categorical("n_init", [10]),
    "max_iter": trial.suggest_categorical("max_iter", [1000]),
    "random_state": trial.suggest_categorical("random_state", [random_state]),
    }
    
    cluster_model = KMeans(**params).fit(X)
    silhouette_avg = silhouette_score(X, cluster_model.labels_)

    trial.set_user_attr("cluster_model", cluster_model)
    
    return silhouette_avg
    
def find_best_k_means_model(img_data):
    random_state = 42
    optuna.logging.set_verbosity(optuna.logging.INFO)
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    best_score = -1
    best_k = 2
    best_cluster_model  = []

    for k in range(2,3):
        study = optuna.create_study(direction = "maximize")
        func = lambda trial: objective(trial, img_data, k, random_state)
        study.optimize(func, n_trials = 2, timeout=600)
        silhouette_score_val = study.best_trial.value

        cluster_model = study.best_trial.user_attrs['cluster_model']
        sse_val = cluster_model.inertia_
        silhouette_avg = silhouette_score(X, cluster_model.labels_)
        
        print("K = ", k)
        print("Silhouette Score = ", silhouette_score_val)
        
        if silhouette_score_val > best_score:
            best_score = silhouette_score_val
            best_k = k
            best_cluster_model = cluster_model
            print("New Best K = ", k)
            print("New Best Silhouette Score = ", silhouette_score_val)

    
    return best_cluster_model