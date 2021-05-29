import os
import json
import pandas as pd
import numpy as np
from sklearn.inspection import permutation_importance
from sklearn.ensemble import RandomForestClassifier
import config

import warnings
warnings.filterwarnings('ignore')


def read_train_v3_csv_file():
    train_file_gzip = os.path.join(config.DATA_DIR, config.TRAIN_V3_FILE)
    train_v3 = pd.read_csv(train_file_gzip, compression='gzip')
    return train_v3


def read_test_v3_csv_file():
    test_file_gzip = os.path.join(config.DATA_DIR, config.TEST_V3_FILE)
    test_v3 = pd.read_csv(test_file_gzip, compression = 'gzip')
    return test_v3


def feature_importance_ranking(train_data, n_features, feature_importance_scores):
    """
    train_data: training pandas dataframe data
    n_features: an integer; number of features in X_train
    feature_importance_scores: numpy ndarray from sklearn.ensemble.RandomForestClassifier.feature_importances_
    """
    feature_importances = {}
    for idx in range(n_features):
        key = train_data.columns[idx]
        val = feature_importance_scores[idx]
        feature_importances[key] = val
    important_features = pd.DataFrame.from_dict(feature_importances, orient = 'index', columns = ['importance'])
    ranked_features = important_features.sort_values('importance', ascending = False)
    return ranked_features


def permutation_importance_ranking(train_data, importances, sorted_idx):
    feature_names = train_data.columns[sorted_idx]
    feature_perm_imp_vals = importances.importances_mean[sorted_idx]
    feature_perm_dict = {}
    for idx in range(len(feature_names)):
        k = feature_names[idx]
        v = feature_perm_imp_vals[idx]
        feature_perm_dict[k] = v
    perm_imp = pd.DataFrame.from_dict(feature_perm_dict, orient = 'index', columns = ['importance'])
    return perm_imp


# feature selection using Pearson corr, RF feature importance, and permutation importance
def feature_selection_per_offer_type(X):
    X = X.copy()
    offer_types = X[X.offer_type != 'informational'].offer_type_v2.unique().tolist()

    # empty dictionary to append important features for everey offer_type
    features_per_offer_type = {}

    # loop through each offer_type and determine important features
    for offer_type in offer_types:
        train = X[X.offer_type_v2 == offer_type]
        train.drop(columns = config.DROP_FEAT_SELECT_COLUMNS, inplace = True)
        train.drop_duplicates(inplace = True)

        # create X and y sets
        X_train = train.iloc[:, :-1]
        y_train = train.iloc[:, -1:]
        # identify correlated features and drop
        corr_matrix = X_train.corr().abs()
        upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k = 1).astype(np.bool_))
        DROP_CORRELATED_COLUMNS = [col for col in upper_tri.columns if any(upper_tri[col] > 0.95)]
        X_train.drop(columns = DROP_CORRELATED_COLUMNS, inplace = True)
        print("Correlated features dropped.")

        # RF feature importances
        rf_estimator = RandomForestClassifier(random_state = 0)
        rf_estimator.fit(X_train, y_train)
        rf_imp = feature_importance_ranking(X_train, len(X_train.columns), rf_estimator.feature_importances_)
        # set arbitrary importance threshold
        rf_imp = rf_imp[rf_imp.importance >= 0.01]
        print("RF feature importances identified.")

        # permutation importances
        perm_importances = permutation_importance(rf_estimator, X_train, y_train, n_repeats = 10, random_state = 0)
        perm_sorted_idx = perm_importances.importances_mean.argsort()[::-1]
        perm_imp = permutation_importance_ranking(X_train, perm_importances, perm_sorted_idx)
        # set arbitrary importance threshold
        perm_imp = perm_imp[perm_imp.importance >= 0.001]
        print("Permutation feature importances identified.")

        # identify features intersecting between RF feat imp and perm imp
        feature_set = list(rf_imp.index.intersection(perm_imp.index))
        print("Intersecting features identified between RF Feature Importances and Permutation Importances.")
        # add to dictionary
        features_per_offer_type[offer_type] = feature_set
        print("{} feature set created.".format(offer_type))
        print("")
    return features_per_offer_type


# save feature set per offer_type_v2 as json file
def serialize_feature_sets(dict_feature_sets):
    # serialize feature_sets dictionary to be used later
    with open(os.path.join(config.TRAINED_MODELS_DIR, config.TRAIN_FEATURE_SETS_FILE), "w") as outfile:
        json.dump(dict_feature_sets, outfile)
        print("Serialization complete: Model feature sets saved.")
        print("  File_path_name: trained_models/training_feature_sets.json")


if __name__ == '__main__':
    X_train_v3 = read_train_v3_csv_file()
    # generate model feature sets using feature_selection_per_offer_type function
    feature_sets = feature_selection_per_offer_type(X_train_v3)
    # execute to serialize feature sets as json
    serialize_feature_sets(feature_sets)
