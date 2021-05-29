import os
import json
import pickle
import config

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

import warnings
warnings.filterwarnings('ignore')


def read_train_test_v3_csv_files():
    train_file_gzip = os.path.join(config.DATA_DIR, config.TRAIN_V3_FILE)
    test_file_gzip = os.path.join(config.DATA_DIR, config.TEST_V3_FILE)
    train_v3 = pd.read_csv(train_file_gzip, compression = 'gzip')
    test_v3 = pd.read_csv(test_file_gzip, compression = 'gzip')
    return train_v3, test_v3


def train_predict_serialize_estimators():
    """
    Task 1: Serialize trained model instances and save to disk -> trained_models directory
    Task 2: Generate test predictions with columns, cid, y_test, y_predict, y_pred_proba, and save results
    :return: None
    """
    # load train and test v3 files
    X_train_v3, X_test_v3 = read_train_test_v3_csv_files()
    # load feature sets for different offer types
    feature_sets = json.load(open(os.path.join(config.TRAINED_MODELS_DIR, config.TRAIN_FEATURE_SETS_FILE)))

    accuracy_scores = {}
    for offer_type, feature_set in feature_sets.items():
        # ---- Task 1: train models and serialize ---- #
        train = X_train_v3[X_train_v3.offer_type_v2 == offer_type]
        X_train = train[feature_set]
        y_train = train.offer_completed
        model = RandomForestClassifier(n_estimators = 100, random_state = 0)
        model.fit(X_train, y_train)

        file_name = '{}/{}_model.pickle'.format(config.TRAINED_MODELS_DIR, offer_type)
        with open(file_name, 'wb') as pickle_file:
            pickle.dump(model, pickle_file)
        print(" Success: Trained {} model pickled.".format(offer_type))

        # ---- Task 2: predict on test data and save results ---- #
        test = X_test_v3[X_test_v3.offer_type_v2 == offer_type]
        test_cid = test.customer_id.tolist()
        X_test = test[feature_set]
        y_test = test.offer_completed

        # model predict and pred_proba
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)
        y_max_pred_proba = [max(vals) for vals in y_pred_proba]

        df_test_pred = pd.DataFrame({'customer_id': test_cid,
                                     'y_test': list(y_test),
                                     'y_predict': list(y_pred),
                                     'y_max_pred_proba': y_max_pred_proba,
                                     'y_pred_proba': list(y_pred_proba)})

        # write df to csv file
        df_test_pred.to_csv('{}/{}_test_predictions.csv.gz'.format(config.PREDICTIONS_DIR, offer_type),
                            index = False,
                            compression = 'gzip')
        print("Success: {} prediction file created".format(offer_type))
        print("  File_path_name: {}/{}_test_predictions.csv.gz".format(config.PREDICTIONS_DIR, offer_type))
        print("")

        # ---- test accuracy scores ---- #
        accuracy_value = accuracy_score(y_test, y_pred)
        accuracy_scores[offer_type + '_accuracy_score'] = accuracy_value

    # display accuracy scores per offer_type
    print(accuracy_scores)
    print("")
    # serialize accuracy_test_scores dictionary to json file
    with open("{}/accuracy_test_scores.json".format(config.PREDICTIONS_DIR), "w") as outfile:
        json.dump(accuracy_scores, outfile)
        print("Serialization complete: accuracy_test_scores saved.")
        print("  File_path_name: {}/{}".format(config.PREDICTIONS_DIR, config.ACCURACY_TEST_FILE))


if __name__ == '__main__':
    # execute trained estimator serialization and output test csv prediction files
    train_predict_serialize_estimators()
