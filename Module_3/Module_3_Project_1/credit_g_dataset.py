import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import fetch_openml
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
from correlation_feature_selection import get_relevant_non_redundant_features
from correlation_feature_selection import sort_feature_by_relevance


def get_preprocessed_credit_g_dataset():
    # Load version 3 of the dataset credit-g
    X, y = fetch_openml(name='credit-g', version=1, parser='auto',return_X_y=True)
    print(type(X))
    print(type(y))
    print(X)
    print(y)

    # manually_sel_feats = ['credit_history', 'saving_status', 'credit_amount', 'employment', 'installment_comitment', 'age', 'duration']

    # Join data and target so that we can drop the na values without affecting the data<->target pairing
    df = pd.concat([X, y], axis=1)
    df.rename(columns={df.columns[-1]: "Target"},inplace=True)
    nr_nulls = df.isnull().sum().sum()
    print("Nr. Nulls: ", nr_nulls)
    # Get Data and Target again
    X = df.iloc[:,0:-1]
    y = df.iloc[:,-1]


    # Traverse the columns of the data and encode the categorical ones, save all data to a new dataframe
    num_feature_flags = []
    data_dict = {}
    for i, c in enumerate(X.columns):

        x_column = X[[c]].values
        encoded_x_col = x_column
        categorical_feat = type(X[c][0]) == str
        if categorical_feat:
            nr_cases = len(np.unique(x_column))
            if nr_cases > 2:
                encoded_x_col = OneHotEncoder().fit_transform(x_column).toarray()
            else:
                encoded_x_col = LabelEncoder().fit_transform(x_column.ravel())

        if len(encoded_x_col.shape) == 1:
            data_dict[c] = encoded_x_col
            num_feature_flags.append(not categorical_feat)
        else:
            for decd_i in range(encoded_x_col.shape[1]):
                col_name = c + "_" + str(decd_i)
                data_dict[col_name] = encoded_x_col[:,decd_i]
                num_feature_flags.append(not categorical_feat)

    X_encoded_df = pd.DataFrame(data_dict)
    y_encoded_df = LabelEncoder().fit_transform(y)
    nr_x_nulls = X_encoded_df.isnull().sum().sum()
    nr_y_nulls = sum(np.isnan(y_encoded_df))
    # Delete unused dataframe
    del df, X, y

    print("Encoded Y Shape: ", y_encoded_df.shape)
    print("Encoded X Shape: ", X_encoded_df.shape)
    print("Nr. X Nulls: ", nr_x_nulls)
    print("Nr. Y Nulls: ", nr_y_nulls)



    # Z-score the numerical features
    X_encoded_scaled_df = X_encoded_df.copy()
    feat_names = list(X_encoded_df.columns)
    for fidx, fname in enumerate(feat_names):
        if num_feature_flags[fidx]:
            X_encoded_scaled_df[fname] = (X_encoded_scaled_df[fname] - X_encoded_scaled_df[fname].mean()) / X_encoded_scaled_df[fname].std()
            #X_encoded_scaled_df[fname] = MinMaxScaler().fit_transform(X_encoded_scaled_df[[fname]].values) # StandardScaler, MinMaxScaler



    # Show relevance from all features
    sort_idxs, feature_relevance = sort_feature_by_relevance(X_encoded_df.values, y_encoded_df, th=0)
    feat_names = np.array(X_encoded_df.columns)
    plt.figure(figsize=(10, 10))
    plt.bar(x=np.arange(len(feature_relevance)) + 1, height=feature_relevance, tick_label=feat_names[sort_idxs])
    plt.xticks(rotation=60)
    plt.title("Relevant Non-Redundant Features")
    plt.show(block=False)



    # Partition the data in Training and Test sets
    X_train, X_test, y_train, y_test = train_test_split(X_encoded_scaled_df, y_encoded_df, test_size=0.1, random_state=0)
    print("X_train type: ", type(X_train))
    print("X_train Shape: ", X_train.shape)
    print("y_train type: ", type(y_train))
    print("y_train Shape: ", y_train.shape)



    # Select relevant and non-redundant features
    ## Select only numerical features
    # relevance_th=0.05, redundancy_th=0.5
    # relevance_th=0.2, redundancy_th=1
    relevance_th = 0.00
    redundancy_th = 1.00
    X_continuous = X_train.values[:,num_feature_flags]
    continuous_feat_names = np.array(feat_names)[num_feature_flags]
    X_continuous_slct, continuous_feat_names_slct, continuous_feature_relevance = get_relevant_non_redundant_features(X_continuous, y_train, continuous_feat_names, relevance_th=relevance_th, redundancy_th=redundancy_th, plot=True)
    print("Total Nr. of numeric Features: ", len(continuous_feat_names))
    print("Nr. of selected numeric Features: ", len(continuous_feat_names_slct))
    print("Relevant and non-redundant numerical features:", continuous_feat_names_slct)


    ## Select only categorical features
    # relevance_th=0.15, redundancy_th=0.17
    # relevance_th=0.2, redundancy_th=1
    relevance_th = 0.0
    redundancy_th = 1.00
    X_categorical = X_train.values[:,np.logical_not(num_feature_flags)]
    categorical_feat_names = np.array(feat_names)[np.logical_not(num_feature_flags)]
    X_categorical_slct, categorical_feat_names_slct, categorical_feature_relevance = get_relevant_non_redundant_features(X_categorical, y_train, categorical_feat_names, relevance_th=relevance_th, redundancy_th=redundancy_th, plot=True)
    print("Total Nr. of categorical features: ", len(categorical_feat_names))
    print("Nr. of selected categorical Features: ", len(categorical_feat_names_slct))
    print("Relevant and non-redundant categorical features:", continuous_feat_names_slct)


    ## Construct the training and test data based on the selected features
    all_selected_feature_names = np.append(continuous_feat_names_slct, categorical_feat_names_slct).tolist()
    print("All selected features:", all_selected_feature_names)
    X_train[all_selected_feature_names]
    X_train_lowdim = X_train[all_selected_feature_names]
    X_test_lowdim = X_test[all_selected_feature_names]


    ## Split the training data in training and validation sets
    X_train, X_validation, y_train, y_validation = train_test_split(X_train_lowdim, y_train, test_size=0.3, random_state=0)

    print("\nTraining Set\n")
    print("X_train type: ", type(X_train))
    print("X_train Shape: ", X_train.shape)
    print("Y_train type: ", type(y_train))
    print("Y_train Shape: ", y_train.shape)

    print("\n\nValidation Set\n")
    print("X_validation type: ", type(X_validation))
    print("X_validation Shape: ", X_validation.shape)
    print("Y_validation type: ", type(y_validation))
    print("Y_validation Shape: ", y_validation.shape)

    print("\n\nTest Set\n")
    print("X_test_lowdim type: ", type(X_test_lowdim))
    print("X_test_lowdim Shape: ", X_test_lowdim.shape)
    print("Y_test type: ", type(y_test))
    print("Y_test Shape: ", y_test.shape)


    X_train = X_train.values
    X_validation = X_validation.values
    X_test = X_test_lowdim.values

    return X_train, X_validation, X_test, y_train, y_validation, y_test

