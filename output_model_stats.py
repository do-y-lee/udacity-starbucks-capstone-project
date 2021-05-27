import os
import json
import pickle
import warnings
import config

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import confusion_matrix, precision_score, recall_score
from sklearn.metrics import roc_auc_score, roc_curve, precision_recall_curve
plt.style.use('seaborn')
warnings.filterwarnings('ignore')


def output_train_model_stats(X_v3, feature_sets):
    """
    :param X_v3: train dataframe (X_train_v3)
    :param feature_sets: dictionary with offer_type as key and feature_set as value
    :return: diagnostic metrics dataframe
    """
    # make copy of X; input X is X_train_v3
    X_v3 = X_v3.copy()

    model_stats = {}
    column_names = ['accuracy', 'accuracy_cv_score', 'accuracy_cv_stddev', 'precision_score',
                    'recall_score', 'f1_score', 'roc_auc_score (cross_val_score)']

    for offer_type, feature_set in feature_sets.items():
        # generate empty list for model diagnostic metrics
        model_scores = []

        # isolate offer_type data from X_v3, and create train and test data
        X_offer_v3 = X_v3[X_v3.offer_type_v2 == offer_type]
        X_train = X_offer_v3[feature_set]
        y_train = X_offer_v3.offer_completed

        # load trained model from disk
        pickled_model_file = '{}/{}_model.pickle'.format(config.TRAINED_MODELS_DIR, offer_type)
        model = pickle.load(open(pickled_model_file, 'rb'))

        # calculate accuracy score and append to model_scores list
        model_scores.append(model.score(X_train, y_train))

        # implement cross validation model accuracy score
        scores = cross_val_score(model, X_train, y_train, cv = 6, scoring = 'accuracy')
        model_scores.append(scores.mean())
        model_scores.append(scores.std())

        # implement cross validation predictions on train data
        y_cv_pred = cross_val_predict(model, X_train, y_train, cv = 6)

        # calculate precision and recall
        p = precision_score(y_train, y_cv_pred)
        r = recall_score(y_train, y_cv_pred)
        model_scores.append(p)
        model_scores.append(r)

        # calculate F1 score
        f1 = (2 * p * r) / (p + r)
        model_scores.append(f1)

        # calculate ROC AUC score using cross_val_score
        roc_auc_cvs = cross_val_score(model, X_train, y_train, cv = 6, scoring = 'roc_auc').mean()
        model_scores.append(roc_auc_cvs)

        # create dictionary key/pair value
        model_stats[offer_type] = model_scores

    # put model stats into a dataframe
    df_model_stats = pd.DataFrame.from_dict(
        model_stats, orient = 'index', columns = column_names).reset_index().rename(columns = {'index': 'offer_type'})
    df_model_stats_ranked = df_model_stats.sort_values(by = 'accuracy', ascending = False)
    df_model_stats_ranked.to_csv('{}/train_model_stats.csv.gz'.format(config.DIAGNOSTIC_METRICS_DIR),
                                 index = False, compression = 'gzip')


def output_test_model_stats(X_v3, feature_sets):
    # make copy of X_v3
    X_v3 = X_v3.copy()
    # create empty dictionary
    model_stats = {}
    column_names = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc', 'tn', 'fp', 'fn', 'tp']

    for offer_type, feature_set in feature_sets.items():
        # empty list
        model_scores = []
        # load pickled model
        pickled_model = '{}/{}_model.pickle'.format(config.TRAINED_MODELS_DIR, offer_type)
        model = pickle.load(open(pickled_model, 'rb'))

        # define X_test and y_test
        X_offer_v3 = X_v3[X_v3.offer_type_v2 == offer_type]
        X_test = X_offer_v3[feature_set]
        y_test = X_offer_v3.offer_completed
        # model predict and pred_proba
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)

        # accuracy
        accuracy = model.score(X_test, y_test)
        model_scores.append(accuracy)
        # precision
        precision = precision_score(y_test, y_pred)
        model_scores.append(precision)
        # recall
        recall = recall_score(y_test, y_pred)
        model_scores.append(recall)
        # F1 score
        f1 = (2 * precision * recall) / (precision + recall)
        model_scores.append(f1)
        # ROC AUC
        auc_score = roc_auc_score(y_test, y_pred_proba[:, 1])
        model_scores.append(auc_score)
        # confusion matrix
        tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
        model_scores.append(tn)
        model_scores.append(fp)
        model_scores.append(fn)
        model_scores.append(tp)
        # append offer type metrics to dictionary
        model_stats[offer_type] = model_scores
    # create output dataframe
    df_model_stats = pd.DataFrame.from_dict(
        model_stats, orient = 'index', columns = column_names).reset_index().rename(columns={'index': 'offer_type'})
    df_model_stats.to_csv('{}/test_model_stats.csv.gz'.format(config.DIAGNOSTIC_METRICS_DIR),
                          index = False, compression = 'gzip')


# function used within create_and_save_roc_curve() function
def _calc_positive_y_pred_proba(df):
    if df.y_predict == 0:
        pred_proba = df.y_max_pred_proba - 1.0
        return pred_proba
    else:
        return df.y_max_pred_proba


# create ROC curve plot for all eight models
def create_and_save_roc_curve(test_v3, feature_sets):
    fig, ax = plt.subplots(figsize = (8, 8))
    for offer in feature_sets.keys():
        df_pred = pd.read_csv('{}/{}_test_predictions.csv.gz'.format(config.PREDICTIONS_DIR, offer), compression = 'gzip')
        df_pred['y_positive_pred_proba'] = df_pred.apply(_calc_positive_y_pred_proba, axis = 1)

        y_test_actual = df_pred.y_test
        y_pred_probabilities = df_pred.y_positive_pred_proba
        fpr, tpr, _ = roc_curve(y_test_actual, y_pred_probabilities, pos_label = 1)
        ax.plot(fpr, tpr, linestyle = '--', label = offer)

    # roc curve for tpr = fpr (no-skill) line
    n_rows = test_v3.offer_type_v2.value_counts().max()
    y_test_actual_stg = test_v3.offer_type_v2.value_counts().reset_index()
    y_test_offer_type = y_test_actual_stg[y_test_actual_stg.offer_type_v2 == n_rows]['index'].values[0]
    y_test_actual = test_v3[test_v3.offer_type_v2 == y_test_offer_type].offer_completed

    random_probs = [0 for _ in range(n_rows)]
    p_fpr, p_tpr, _ = roc_curve(y_test_actual, random_probs, pos_label = 1)
    ax.plot(p_fpr, p_tpr, linestyle = '--')

    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.legend(loc = 'lower right', fontsize = 12)
    plt.savefig('{}/test_roc_curves.png'.format(config.DIAGNOSTIC_METRICS_DIR), dpi = 300, bbox_inches = 'tight')


# create precision-recall curve plot for all eight models
def create_and_save_precision_recall_curve(feature_sets):
    """
    :param feature_sets: dictionary with list of features per offer type
    :return: None; saves precision-recall curve plots as .png and optimal model threshold using F-measure as CSV
    """
    fig, ax = plt.subplots(figsize = (8, 8))
    avg_no_skills = []
    optimal_thresholds = {}
    for offer in feature_sets.keys():
        df_pred = pd.read_csv('{}/{}_test_predictions.csv.gz'.format(config.PREDICTIONS_DIR, offer), compression = 'gzip')
        df_pred['y_positive_pred_proba'] = df_pred.apply(_calc_positive_y_pred_proba, axis = 1)

        # plot precision and recall for each offer type
        y_test_actual = df_pred.y_test
        y_pred_probabilities = df_pred.y_positive_pred_proba
        precision, recall, thresholds = precision_recall_curve(y_test_actual, y_pred_probabilities)
        ax.plot(recall, precision, marker = '.', label = offer)
        # ax.scatter(recall[idx], precision[idx], marker = 'o', color = 'gray', label = 'Optimal')

        # calculate no skill precision value per offer type
        no_skill = 1.0 * sum(y_test_actual) / len(y_test_actual)
        avg_no_skills.append(no_skill)

        # collect optimal threshold metrics per model
        optimal_threshold_metrics = []
        f1 = (2 * precision * recall) / (precision + recall)
        idx = np.argmax(f1)
        optimal_threshold_metrics.append(thresholds[idx])
        optimal_threshold_metrics.append(f1[idx])
        optimal_threshold_metrics.append(precision[idx])
        optimal_threshold_metrics.append(recall[idx])
        optimal_thresholds[offer] = optimal_threshold_metrics

    # take average of avg_no_skills and plot
    avg_no_skill = np.mean(avg_no_skills)
    ax.plot([0, 1], [avg_no_skill, avg_no_skill], linestyle = '--', label = 'Average No Skill')
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.legend(loc = 'center left', fontsize = 12)
    plt.savefig('{}/test_pr_curves.png'.format(config.DIAGNOSTIC_METRICS_DIR), dpi = 300, bbox_inches = 'tight')

    # save optimal threshold metrics per model as csv file
    optimal_column_names = ['optimal_decision_threshold', 'optimal_f1_score', 'optimal_precision', 'optimal_recall']
    df_optimal_thresholds = pd.DataFrame.from_dict(
        optimal_thresholds, orient = 'index', columns = optimal_column_names).reset_index().rename(columns={'index': 'offer_type'})
    df_optimal_thresholds.to_csv(os.path.join(config.DIAGNOSTIC_METRICS_DIR, 'test_optimal_f1_thresholds.csv.gz'),
                                 index = False,
                                 compression = 'gzip')


def main():
    # read train and test csv files
    X_train_v3 = pd.read_csv(os.path.join(config.DATA_DIR, config.TRAIN_V3_FILE), compression = 'gzip')
    X_test_v3 = pd.read_csv(os.path.join(config.DATA_DIR, config.TEST_V3_FILE), compression = 'gzip')

    # load training_feature_sets.json as dictionary
    offer_features = json.load(open(os.path.join(config.TRAINED_MODELS_DIR, config.TRAIN_FEATURE_SETS_FILE)))

    # generate train and test model stats
    output_train_model_stats(X_train_v3, offer_features)
    output_test_model_stats(X_test_v3, offer_features)

    # create and save ROC curve plots
    create_and_save_roc_curve(X_test_v3, offer_features)
    # create and save PR curve plots
    create_and_save_precision_recall_curve(offer_features)


if __name__ == '__main__':
    main()
