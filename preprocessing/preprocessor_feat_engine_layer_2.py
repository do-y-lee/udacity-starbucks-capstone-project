import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder, LabelEncoder

from base_transforms import BaseTransformDF
import config

import warnings
warnings.filterwarnings('ignore')


# read train and test csv files
def read_train_test_csv_files():
    train_file_path = os.path.join(config.DATA_DIR, config.TRAIN_FILE)
    test_file_path = os.path.join(config.DATA_DIR, config.TEST_FILE)
    train_data = pd.read_csv(train_file_path, compression = 'gzip')
    test_data = pd.read_csv(test_file_path, compression = 'gzip')
    return train_data, test_data


# drop constant features
def drop_constant_features(X):
    X = X.copy()
    DROP_CONSTANT_COLUMNS = []
    for column in X.columns:
        if X[column].nunique() == 1:
            DROP_CONSTANT_COLUMNS.append(column)
    X.drop(columns=DROP_CONSTANT_COLUMNS, inplace = True)
    return X


# drop generic features
def drop_generic_features(X):
    X = X.copy()
    X.drop(columns=config.DROP_GENERIC_COLUMNS, inplace = True)
    return X


# one hot encoding of gender feature into gender_F, gender_M, and gender_Missing
def gender_one_hot_encoder(train_data, test_data):
    """
    :param train_data: train dataframe
    :param test_data: test dataframe
    :return: outputs train, test dataframes
    """
    # make copies
    X_train = train_data.copy()
    X_test = test_data.copy()

    # Re-label 'O' to 'Missing' for gender
    X_train['gender'] = X_train['gender'].apply(lambda x: 'Missing' if x == 'O' else x)
    X_test['gender'] = X_test['gender'].apply(lambda x: 'Missing' if x == 'O' else x)

    # integer encode
    label_encoder = LabelEncoder()
    train_integer_encoded = label_encoder.fit_transform(X_train.gender)
    test_integer_encoded = label_encoder.transform(X_test.gender)

    # binary encode
    n_train = len(train_integer_encoded)
    n_test = len(test_integer_encoded)

    onehot_encoder = OneHotEncoder(sparse = False)
    train_integer_encoded_reshaped = train_integer_encoded.reshape(n_train, 1)
    test_integer_encoded_reshaped = test_integer_encoded.reshape(n_test, 1)
    train_onehot_encoded = onehot_encoder.fit_transform(train_integer_encoded_reshaped)
    test_onehot_encoded = onehot_encoder.transform(test_integer_encoded_reshaped)

    # create column names and append one-hot-encoded gender columns to X_train and X_test
    train_label_mapping = [item for item in zip(train_integer_encoded, np.array(X_train.gender))]
    train_label_mapping = sorted(set(train_label_mapping))
    train_column_names = ['gender_' + i[1] for i in train_label_mapping]
    train_ohe_gender = pd.DataFrame(train_onehot_encoded, columns = train_column_names)
    X_train = pd.concat([X_train, train_ohe_gender], axis = 1)

    test_label_mapping = [item for item in zip(test_integer_encoded, np.array(X_test.gender))]
    test_label_mapping = sorted(set(test_label_mapping))
    test_column_names = ['gender_' + i[1] for i in test_label_mapping]
    test_ohe_gender = pd.DataFrame(test_onehot_encoded, columns = test_column_names)
    X_test = pd.concat([X_test, test_ohe_gender], axis = 1)
    return X_train, X_test


def ordinal_feature_encoder(train_data, test_data):
    # make copies
    X_train = train_data.copy()
    X_test = test_data.copy()

    # integer encode
    label_encoder = LabelEncoder()
    ordinal_columns = config.ORDINAL_COLUMNS

    for column in ordinal_columns:
        train_labels = X_train[column]
        test_labels = X_test[column]
        train_integer_encoded = label_encoder.fit_transform(train_labels)
        test_integer_encoded = label_encoder.transform(test_labels)

        adj_column = column + '_enc'
        train_column = pd.DataFrame(train_integer_encoded, columns = [adj_column])
        test_column = pd.DataFrame(test_integer_encoded, columns = [adj_column])
        X_train = pd.concat([X_train, train_column], axis = 1)
        X_test = pd.concat([X_test, test_column], axis = 1)
    return X_train, X_test


# drop redundant features
def drop_redundant_features(X):
    X.drop(columns=config.DROP_REDUNDANT_COLUMNS, inplace = True)
    return X


def create_train_test_v1():
    # step 1: read csv files
    train_data, test_data = read_train_test_csv_files()
    # step 2: drop constant features
    X_train = drop_constant_features(train_data)
    X_test = drop_constant_features(test_data)
    # step 3: drop generic features
    X_train = drop_generic_features(X_train)
    X_test = drop_generic_features(X_test)
    # step 4: feature transformers: one hot and ordinal encoding
    X_train, X_test = gender_one_hot_encoder(X_train, X_test)
    X_train, X_test = ordinal_feature_encoder(X_train, X_test)
    # step 5: drop redundant features
    X_train = drop_redundant_features(X_train)
    X_test = drop_redundant_features(X_test)
    return X_train, X_test


# create offer_type_v2 and with apply function
def _create_offer_type_v2(data):
    """Used in lambda function across columns in dataframe"""
    rw = str(data.reward)
    dif = str(data.difficulty)
    dur = str(data.duration)
    otype = data.offer_type
    offer_type_v2 = otype + '-' + rw + '-' + dif + '-' + dur
    return offer_type_v2


def create_portfolio_expanded():
    """
    lambda function: _create_offer_type_v2
    :return: portfolio expanded dataframe
    """
    portfolio = pd.read_json(os.path.join(config.DATA_DIR, config.PORTFOLIO_FILE), orient = 'records', lines = True)
    portfolio_exp = BaseTransformDF.portfolio_expanded(portfolio)
    portfolio_exp['offer_type_v2'] = portfolio_exp.apply(_create_offer_type_v2, axis = 1)
    portfolio_exp.drop(columns = ['channels'], inplace = True)
    return portfolio_exp


def create_received_expanded():
    # create base transcript df
    transcript = pd.read_json(os.path.join(config.DATA_DIR, config.TRANSCRIPT_FILE), orient = 'records', lines = True)
    base_transcript = BaseTransformDF.create_transcript_copy(transcript)
    base_received = BaseTransformDF.create_df_base_received(base_transcript)
    # create portfolio expanded df using create_portfolio_expanded() function
    portfolio_exp = create_portfolio_expanded()
    # merge dataframes to create expanded offers received
    received_exp = base_received.merge(portfolio_exp, how = 'left', on = 'offer_id')
    received_exp.drop(columns = ['offer_received'], inplace = True)
    return received_exp


def create_train_test_v2(X):
    received_exp = create_received_expanded()
    X_customer_ids = X.customer_id.reset_index().drop(columns = ['index'])
    X_received = received_exp.merge(X_customer_ids, how = 'inner', on = 'customer_id')
    X_v2 = X_received.merge(X, how = 'left', on = 'customer_id')
    return X_v2


def create_train_test_v3(X_v2):
    # read transaction engagement csv file
    file_gzip = os.path.join(config.DATA_DIR, config.TRANSACTION_ENGAGEMENT_FILE)
    trans_engage = pd.read_csv(file_gzip, compression = 'gzip')

    # using transaction engagement data mart isolate all transaction_id's with offer_completed = 1
    selected_columns = ['customer_id', 'offer_id', 'offer_completed', 'offer_completed_time', 'offer_received_time']
    trans_offer_completed = trans_engage[selected_columns][trans_engage.offer_completed == 1]
    trans_offer_completed.drop_duplicates(inplace = True)

    # join by customer_id, offer_id, and offer_received_time
    X_v3 = X_v2.merge(trans_offer_completed, how = 'left', on = ['customer_id', 'offer_id', 'offer_received_time'])
    X_v3.drop_duplicates(inplace = True)
    X_v3['offer_completed'].fillna(0, inplace = True)
    X_v3['offer_completed_time'].fillna(-1, inplace = True)
    return X_v3


if __name__ == '__main__':
    # write df to gzipped csv file
    train_v3_file_path = os.path.join(config.DATA_DIR, config.TRAIN_V3_FILE)
    test_v3_file_path = os.path.join(config.DATA_DIR, config.TEST_V3_FILE)

    train, test = create_train_test_v1()
    received_expanded = create_received_expanded()

    X_train_v2 = create_train_test_v2(train)
    X_test_v2 = create_train_test_v2(test)

    X_train_v3 = create_train_test_v3(X_train_v2)
    X_test_v3 = create_train_test_v3(X_test_v2)
    X_train_v3.fillna(0, inplace = True)
    X_test_v3.fillna(0, inplace = True)

    try:
        X_train_v3.to_csv(train_v3_file_path, index = False, compression = 'gzip')
        print("Success: {} created".format(config.TRAIN_V3_FILE))
        print("  File_path_name: {}".format(train_v3_file_path))
        X_test_v3.to_csv(test_v3_file_path, index = False, compression = 'gzip')
        print("Success: {} created".format(config.TEST_V3_FILE))
        print("  File_path_name: {}".format(test_v3_file_path))
    except:
        print("Error: File creations failed.")
