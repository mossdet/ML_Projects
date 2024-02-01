import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import optuna

from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from correlation_feature_selection import get_relevant_non_redundant_features

def load_preprocessed_bcd():
    """
    This function loads, cleans, and preprocesses the breast cancer data.

    Returns:
        df_hypo (pandas.DataFrame): The preprocessed data with hypothesized features removed.
        df_pca (pandas.DataFrame): The preprocessed data with PCA features.
        df_correl (pandas.DataFrame): The preprocessed data with correlation-based features.
    """
    # 1. Data import and cleaning
    df = pd.read_csv('breast_cancer_data.csv')
    print(df.head(5))
    print(df.info())

    # Drop the last column because it only has missing values and the id column because it does not provided valuable information for this project
    df.drop(columns=['id', 'Unnamed: 32'], inplace=True)

    # Show clean version of data
    print(df.info())
    print(df.isna().sum())
    print(df.head(5))


    # 2. Data Standardization
    # Get column names from features
    feature_names = df.columns.to_list()
    feature_names.remove('diagnosis')
    print(feature_names)

    df_scaled = df.copy()
    print(df_scaled.head(10))
    df_scaled[feature_names] = StandardScaler().fit_transform(df_scaled[feature_names].to_numpy())
    print(df_scaled.head(10))


    # 3. EDA, plot the distribution from each feature and for each diagnosis group (malign and benign)
    hypotest_discarded_feats = []
    hypotest_selected_feats = []
    feats_plot_range = [0,8]
    f, axes = plt.subplots(1, 8, figsize=(30,5))
    for idx, feature_name in enumerate(feature_names[feats_plot_range[0]:feats_plot_range[1]]):
        g = sns.histplot(df_scaled, x=feature_name, hue="diagnosis", ax=axes[idx])
        sel_mal = df_scaled['diagnosis'] == 'M'
        sel_ben = df_scaled['diagnosis'] != 'M'
        res_mwu = stats.mannwhitneyu(df_scaled[feature_name][sel_mal], df_scaled[feature_name][sel_ben])
        res_kw = stats.kruskal(df_scaled[feature_name][sel_mal], df_scaled[feature_name][sel_ben])
        g.set(title= f"{feats_plot_range[0]+idx}_{feature_name}\n(MWU-p={res_mwu.pvalue:.6f})\n(KW-p={res_kw.pvalue:.6f})")
        if res_mwu.pvalue > 0.001 or res_kw.pvalue > 0.001:
            hypotest_discarded_feats.append(feature_name)
        else:
            hypotest_selected_feats.append(feature_name)
    plt.show()

    feats_plot_range = [8,16]
    f, axes = plt.subplots(1, 8, figsize=(30,5))
    for idx, feature_name in enumerate(feature_names[feats_plot_range[0]:feats_plot_range[1]]):
        g = sns.histplot(df_scaled, x=feature_name, hue="diagnosis", ax=axes[idx])
        sel_mal = df_scaled['diagnosis'] == 'M'
        sel_ben = df_scaled['diagnosis'] != 'M'
        res_mwu = stats.mannwhitneyu(df_scaled[feature_name][sel_mal], df_scaled[feature_name][sel_ben])
        res_kw = stats.kruskal(df_scaled[feature_name][sel_mal], df_scaled[feature_name][sel_ben])
        g.set(title= f"{feats_plot_range[0]+idx}_{feature_name}\n(MWU-p={res_mwu.pvalue:.6f})\n(KW-p={res_kw.pvalue:.6f})")
        if res_mwu.pvalue > 0.001 or res_kw.pvalue > 0.001:
            hypotest_discarded_feats.append(feature_name)
        else:
            hypotest_selected_feats.append(feature_name)
    plt.show()

    feats_plot_range = [16,24]
    f, axes = plt.subplots(1, 8, figsize=(30,5))
    for idx, feature_name in enumerate(feature_names[feats_plot_range[0]:feats_plot_range[1]]):
        g = sns.histplot(df_scaled, x=feature_name, hue="diagnosis", ax=axes[idx])
        sel_mal = df_scaled['diagnosis'] == 'M'
        sel_ben = df_scaled['diagnosis'] != 'M'
        res_mwu = stats.mannwhitneyu(df_scaled[feature_name][sel_mal], df_scaled[feature_name][sel_ben])
        res_kw = stats.kruskal(df_scaled[feature_name][sel_mal], df_scaled[feature_name][sel_ben])
        g.set(title= f"{feats_plot_range[0]+idx}_{feature_name}\n(MWU-p={res_mwu.pvalue:.6f})\n(KW-p={res_kw.pvalue:.6f})")
        if res_mwu.pvalue > 0.001 or res_kw.pvalue > 0.001:
            hypotest_discarded_feats.append(feature_name)
        else:
            hypotest_selected_feats.append(feature_name)
    plt.show()

    feats_plot_range = [24,32]
    f, axes = plt.subplots(1, 8, figsize=(30,5))
    for idx, feature_name in enumerate(feature_names[feats_plot_range[0]:feats_plot_range[1]]):
        g = sns.histplot(df_scaled, x=feature_name, hue="diagnosis", ax=axes[idx])
        sel_mal = df_scaled['diagnosis'] == 'M'
        sel_ben = df_scaled['diagnosis'] != 'M'
        res_mwu = stats.mannwhitneyu(df_scaled[feature_name][sel_mal], df_scaled[feature_name][sel_ben])
        res_kw = stats.kruskal(df_scaled[feature_name][sel_mal], df_scaled[feature_name][sel_ben])
        g.set(title= f"{feats_plot_range[0]+idx}_{feature_name}\n(MWU-p={res_mwu.pvalue:.6f})\n(KW-p={res_kw.pvalue:.6f})")
        if res_mwu.pvalue > 0.001 or res_kw.pvalue > 0.001:
            hypotest_discarded_feats.append(feature_name)
        else:
            hypotest_selected_feats.append(feature_name)
    plt.show()

    # 4. Selection of features
    ##  4.1 Based on hypothesis testing
    print("hypotest_discarded_feats:\n", hypotest_discarded_feats)
    print("\n\nhypotest_selected_feats:\n", hypotest_selected_feats)
    df_hypo = df_scaled.drop(columns=hypotest_discarded_feats, inplace=False)
    df_hypo.head(5)

    ## 4.2 Based on PCA
    pca = PCA(n_components=0.95, svd_solver='full')
    pca_data = pca.fit_transform(df_scaled[feature_names].to_numpy())
    pca_feat_names = pca.get_feature_names_out()
    print("Principal Components Explained Variance: ", np.sum(pca.explained_variance_ratio_))
    print("Nr. of PCA Components: ", pca.n_components_)
    print("PCA Feature NAmes:", pca_feat_names)
    pca_data.shape
    df_pca = df[['diagnosis']].copy()
    df_pca[pca_feat_names] = pca_data
    df_pca.head(5)
    #sns.pairplot(df_pca,hue="diagnosis",palette="rainbow")
    #plt.suptitle('PCA Components Pairplot')

    ## 4.3 Based on Correlation
    relevance_th=0.2
    redundancy_th=0.7
    X = df_scaled[feature_names].to_numpy()
    y = np.round(df_scaled['diagnosis']=='M').to_numpy()
    X, feat_names, feature_relevance = get_relevant_non_redundant_features(X, y, np.array(feature_names), relevance_th=relevance_th, redundancy_th=redundancy_th, plot=False)
    df_correl = pd.DataFrame(data=X, columns=feat_names)
    df_correl = pd.concat([df_scaled['diagnosis'], df_correl], axis=1)

    return df_hypo, df_pca, df_correl




