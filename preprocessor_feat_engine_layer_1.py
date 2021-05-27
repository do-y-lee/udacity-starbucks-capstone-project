import os
import pandas as pd
import numpy as np
from datetime import datetime
from pathlib import Path

from sklearn.model_selection import train_test_split
from base_transforms import BaseTransformDF
import config

import warnings
warnings.filterwarnings('ignore')


# inputs: profile.json, starbucks_transaction_engagement.csv.gz
def base_train_test_split(df_profile, df_trans_engage):
    # read in profile data and column names
    df_profile = df_profile.rename(columns = {'id': 'customer_id', 'became_member_on': 'date_registered'})
    df_profile['date_registered'] = df_profile['date_registered'].apply(lambda x: datetime.strptime(str(x), "%Y%m%d"))
    # calculate total transactions and total transaction amount per customer_id
    rename_columns = {'transaction_id': 'transaction_cnt'}
    transaction_cnt = df_trans_engage.groupby('customer_id')['transaction_id'].nunique().reset_index().rename(columns=rename_columns)
    transaction_amount = df_trans_engage.groupby('customer_id')['transaction_amount'].sum().reset_index()
    transactions = transaction_amount.merge(transaction_cnt, how='left', on='customer_id')
    transactions['transaction_aos'] = round(1.0 * transactions['transaction_amount'] / transactions['transaction_cnt'], 2)
    # merge profile and transactions, and create members dataframe
    members = df_profile.merge(transactions, how = 'left', on = 'customer_id')
    members['transaction_amount'] = members['transaction_amount'].fillna(0)
    members['transaction_cnt'] = members['transaction_cnt'].fillna(0)
    members['transaction_aos'] = members['transaction_aos'].fillna(0)
    # split into train and test sets
    dfx_train, dfx_test, dfy_train, dfy_test = train_test_split(members, members.transaction_aos, test_size = 0.33, random_state = 0)
    return dfx_train, dfx_test


# helper function: e.g., unittest_var_quantile_intervals(age_intervals, age_quantile_range)
def unittest_var_quantile_intervals(pd_generated_var_intervals, var_quantile_range):
    """
    Check if pandas generated variable intervals are not rounded up versus the actual intervals
    """
    # create function to manually generate the variable interval values
    def generate_quantile_intervals(quantile_range):
        n_indices = len(quantile_range.unique()) - 1
        quantile_intervals = []
        cnt = 0
        for i in sorted(quantile_range.unique()):
            if cnt == n_indices:
                val0 = float(str(i).replace("(", "").split(",")[0])
                val1 = float(str(i).replace("(", "").replace("]", "").split(",")[1])
                # arbitrary number ten added to compensate for edge cases where some var instances
                # the max value of the bin range does not cover all var instance values
                val1 = val1 + 10
                quantile_intervals.append(val0)
                quantile_intervals.append(val1)
            else:
                val = float(str(i).replace("(", "").split(",")[0])
                quantile_intervals.append(val)
            cnt += 1
        var_quantile_intervals = sorted(quantile_intervals)
        return var_quantile_intervals

    # execute function and generate manual list of intervals
    func_generated_var_intervals = generate_quantile_intervals(var_quantile_range)
    # compare to pandas generated intervals b/c pandas round up and causes issue
    pd_generated_var_intervals = list(pd_generated_var_intervals)

    if pd_generated_var_intervals == func_generated_var_intervals:
        return pd_generated_var_intervals
    else:
        return func_generated_var_intervals


# grab X_train and df_test from base_train_test_split()
def missing_outlier_imputer(df_train, df_test):
    # create flag to show which rows have been imputed
    df_train['gender_NA'] = np.where(df_train['gender'].isnull(), 1, 0)
    df_train['income_NA'] = np.where(df_train['income'].isnull(), 1, 0)
    df_train['age_NA'] = np.where(df_train['age'] == 118, 1, 0)
    df_train['age'] = df_train['age'].apply(lambda x: np.nan if x == 118 else x)
    df_test['gender_NA'] = np.where(df_test['gender'].isnull(), 1, 0)
    df_test['income_NA'] = np.where(df_test['income'].isnull(), 1, 0)
    df_test['age_NA'] = np.where(df_test['age'] == 118, 1, 0)
    df_test['age'] = df_test['age'].apply(lambda x: np.nan if x == 118 else x)
    # impute income and age with median value in each column
    # only use values calculated in train set to impute test set
    age_median = df_train.age.median()
    income_median = df_train.income.median()
    df_train['age'].fillna(age_median, inplace = True)
    df_test['age'].fillna(age_median, inplace = True)
    df_train['income'].fillna(income_median, inplace = True)
    df_test['income'].fillna(income_median, inplace = True)
    # impute gender with 'Missing' category
    df_train['gender'].fillna('Missing', inplace = True)
    df_test['gender'].fillna('Missing', inplace = True)
    return df_train, df_test


def age_quantile_transformer(df_train, df_test):
    labels = ['0-20Q', '20-40Q', '40-60Q', '60-80Q', '80-100Q']
    # calculate quantile ranges
    df_train_age_quantile_range, df_train_age_intervals = pd.qcut(df_train['age'], 5,
                                                                  retbins = True,
                                                                  precision = 3,
                                                                  duplicates = 'raise')
    # check for rounding issue with pd.qcut(retbin=True); unittest df_train_age_intervals and modify
    df_train_age_intervals = unittest_var_quantile_intervals(df_train_age_intervals, df_train_age_quantile_range)
    # calculate quantile labels
    df_train_age_quantile_labels = pd.qcut(df_train['age'], 5, labels = labels, duplicates = 'raise')
    # append columns based on index location
    df_train.insert(loc = 2, column = 'age_quantile_range', value = df_train_age_quantile_range)
    df_train.insert(loc = 3, column = 'age_quantile_label', value = df_train_age_quantile_labels)
    # transform test set: pass the quantile edges calculated in the training set
    df_test_age_quantile_range = pd.cut(x = df_test['age'], bins = df_train_age_intervals)
    df_test_age_quantile_labels = pd.cut(x = df_test['age'], bins = df_train_age_intervals, labels = labels)
    # append columns based on index location
    df_test.insert(loc = 2, column = 'age_quantile_range', value = df_test_age_quantile_range)
    df_test.insert(loc = 3, column = 'age_quantile_label', value = df_test_age_quantile_labels)
    return df_train, df_test


def income_quantile_transformer(df_train, df_test):
    labels = ['0-20Q', '20-40Q', '40-60Q', '60-80Q', '80-100Q']
    # create train quantile range, intervals, and labels
    df_train_income_quantile_range, df_train_income_intervals = pd.qcut(
        df_train['income'], 5, retbins = True, precision = 3, duplicates = 'raise')
    # check for rounding issue with pd.qcut(retbin=True); unittest df_train_income_intervals and modify
    df_train_income_intervals = unittest_var_quantile_intervals(df_train_income_intervals, df_train_income_quantile_range)
    # generate quantile labels
    df_train_income_quantile_labels = pd.qcut(df_train['income'], 5, labels = labels, duplicates = 'raise')
    # append to df_train
    df_train.insert(loc = 7, column = 'income_quantile_range', value = df_train_income_quantile_range)
    df_train.insert(loc = 8, column = 'income_quantile_label', value = df_train_income_quantile_labels)
    # apply df_train learned parameters to df_test
    df_test_income_quantile_range = pd.cut(x = df_test['income'], bins = df_train_income_intervals)
    df_test_income_quantile_labels = pd.cut(x = df_test['income'], bins = df_train_income_intervals, labels = labels)
    # append columns based on index location
    df_test.insert(loc = 7, column = 'income_quantile_range', value = df_test_income_quantile_range)
    df_test.insert(loc = 8, column = 'income_quantile_label', value = df_test_income_quantile_labels)
    return df_train, df_test


def date_registered_transformer(df_train, df_test):
    """
    1. convert string date into datetime date
    2. calculate days registered using the available max date registered in train dataset
    3. create quantile range and labels

    :param df_train:
    :param df_test:
    :return: df_train, df_test
    """
    # calculate max date and calculate days_registered for df_train and df_test
    max_date_registered = df_train.date_registered.max()
    df_train_days_registered = df_train['date_registered'].apply(lambda x: (max_date_registered - x).days)
    df_train.insert(loc = 6, column = 'days_registered', value = df_train_days_registered)
    df_test_days_registered = df_test['date_registered'].apply(lambda x: (max_date_registered - x).days)
    df_test.insert(loc=6, column='days_registered', value=df_test_days_registered)

    # retbins = True captures the limits of each interval (can use them to cut the test set)
    days_reg_labels = ['0-10Q', '10-20Q', '20-30Q', '30-40Q', '40-50Q', '50-60Q', '60-70Q', '70-80Q', '80-90Q',
                       '90-100Q']
    # calculate quantile ranges
    days_reg_quantile_range, days_reg_intervals = pd.qcut(df_train['days_registered'], 10, retbins = True)
    # check for rounding issue with pd.qcut(retbin=True); unittest df_train_days_reg_intervals and modify
    days_reg_intervals = unittest_var_quantile_intervals(days_reg_intervals, days_reg_quantile_range)
    # calculate quantile labels
    days_reg_quantile_labels = pd.qcut(df_train['days_registered'], 10, labels = days_reg_labels)
    # insert into df_train
    df_train.insert(loc = 7, column = 'days_reg_quantile_range', value = days_reg_quantile_range)
    df_train.insert(loc = 8, column = 'days_reg_quantile_label', value = days_reg_quantile_labels)
    # use pandas cut method (instead of qcut) and pass the quantile edges calculated in the training set
    df_test_days_reg_quantile_range = pd.cut(x = df_test['days_registered'], bins = days_reg_intervals)
    df_test_days_reg_quantile_labels = pd.cut(
        x = df_test['days_registered'], bins = days_reg_intervals, labels = days_reg_labels)
    # append columns based on index location
    df_test.insert(loc = 7, column = 'days_reg_quantile_range', value = df_test_days_reg_quantile_range)
    df_test.insert(loc = 8, column = 'days_reg_quantile_label', value = df_test_days_reg_quantile_labels)
    return df_train, df_test


# num_offer_received, num_bogo_offer_received, num_info_offer_received, num_discount_offer_received
def offer_received_transformer(df_train, df_test, df_received, df_portfolio):
    # number of total offers received per customer_id
    df_num_offer_received = df_received.groupby('customer_id')['offer_received'].sum()
    df_num_offer_received = df_num_offer_received.reset_index().rename(
        columns = {'offer_received': 'num_offer_received'})

    # append to df_train and df_test
    df_train = df_train.merge(df_num_offer_received, how = 'left', on = 'customer_id')
    df_train['num_offer_received'].fillna(0, inplace = True)

    df_test = df_test.merge(df_num_offer_received, how = 'left', on = 'customer_id')
    df_test['num_offer_received'].fillna(0, inplace = True)

    # num_bogo_offer_received, num_info_offer_received, num_discount_offer_received
    # merge df_received with df_portfolio
    offer_received_agg = df_received.merge(df_portfolio, how = 'left', on = 'offer_id')

    # bogo
    num_bogo_offer_received = offer_received_agg[offer_received_agg['type_bogo'] == 1].groupby('customer_id')[
        'offer_received'].sum()
    num_bogo_offer_received = num_bogo_offer_received.reset_index().rename(
        columns = {'offer_received': 'num_bogo_offer_received'})

    # informational
    num_info_offer_received = offer_received_agg[offer_received_agg['type_informational'] == 1].groupby('customer_id')[
        'offer_received'].sum()
    num_info_offer_received = num_info_offer_received.reset_index().rename(
        columns = {'offer_received': 'num_info_offer_received'})

    # discount
    num_discount_offer_received = offer_received_agg[offer_received_agg['type_discount'] == 1].groupby('customer_id')[
        'offer_received'].sum()
    num_discount_offer_received = num_discount_offer_received.reset_index().rename(
        columns = {'offer_received': 'num_discount_offer_received'})

    # append features to df_train and df_test
    feature_list = [num_bogo_offer_received, num_info_offer_received, num_discount_offer_received]

    for df_feature in feature_list:
        feature_name = df_feature.columns[1]
        df_train = df_train.merge(df_feature, how = 'left', on = 'customer_id')
        df_train[feature_name] = df_train[feature_name].fillna(0)

    for df_feature in feature_list:
        feature_name = df_feature.columns[1]
        df_test = df_test.merge(df_feature, how = 'left', on = 'customer_id')
        df_test[feature_name] = df_test[feature_name].fillna(0)
    return df_train, df_test


# num_offer_viewed, num_bogo_offer_viewed, num_info_offer_viewed, num_discount_offer_viewed
def offer_viewed_transformer(df_train, df_test, df_viewed, df_portfolio):
    offer_viewed_agg = df_viewed.groupby('customer_id')['offer_viewed'].sum().reset_index()
    offer_viewed_agg.rename(columns = {'offer_viewed': 'num_offer_viewed'}, inplace = True)

    df_train = df_train.merge(offer_viewed_agg, how = 'left', on = 'customer_id')
    df_train['num_offer_viewed'].fillna(0, inplace = True)

    df_test = df_test.merge(offer_viewed_agg, how = 'left', on = 'customer_id')
    df_test['num_offer_viewed'].fillna(0, inplace = True)

    # Features: num_bogo_offer_viewed, num_info_offer_viewed, num_discount_offer_viewed
    # merge df_viewed with df_portfolio
    offer_viewed_agg = df_viewed.merge(df_portfolio, how = 'left', on = 'offer_id')

    # bogo
    num_bogo_offer_viewed = offer_viewed_agg[offer_viewed_agg['type_bogo'] == 1].groupby('customer_id')[
        'offer_viewed'].sum()
    num_bogo_offer_viewed = num_bogo_offer_viewed.reset_index().rename(
        columns = {'offer_viewed': 'num_bogo_offer_viewed'})

    # informational
    num_info_offer_viewed = offer_viewed_agg[offer_viewed_agg['type_informational'] == 1].groupby('customer_id')[
        'offer_viewed'].sum()
    num_info_offer_viewed = num_info_offer_viewed.reset_index().rename(
        columns = {'offer_viewed': 'num_info_offer_viewed'})

    # discount
    num_discount_offer_viewed = offer_viewed_agg[offer_viewed_agg['type_discount'] == 1].groupby('customer_id')[
        'offer_viewed'].sum()
    num_discount_offer_viewed = num_discount_offer_viewed.reset_index().rename(
        columns = {'offer_viewed': 'num_discount_offer_viewed'})

    # append features to df_train and df_test
    feature_list = [num_bogo_offer_viewed, num_info_offer_viewed, num_discount_offer_viewed]

    for df_feature in feature_list:
        feature_name = df_feature.columns[1]
        df_train = df_train.merge(df_feature, how = 'left', on = 'customer_id')
        df_train[feature_name] = df_train[feature_name].fillna(0)

    for df_feature in feature_list:
        feature_name = df_feature.columns[1]
        df_test = df_test.merge(df_feature, how = 'left', on = 'customer_id')
        df_test[feature_name] = df_test[feature_name].fillna(0)
    return df_train, df_test


# num_offer_completed, num_offer_completed_viewed, num_offer_completed_not_viewed
def offer_completed_transformer(df_train, df_test, df_trans_engage):
    offer_completed_agg = df_trans_engage.groupby('customer_id')['offer_completed'].sum().reset_index()
    offer_completed_agg.rename(columns = {'offer_completed': 'num_offer_completed'}, inplace = True)

    offer_completed_viewed_agg = df_trans_engage[df_trans_engage['offer_viewed'] == 1].groupby('customer_id')[
        'offer_completed'].sum().reset_index()
    offer_completed_viewed_agg.rename(columns = {'offer_completed': 'num_offer_completed_viewed'}, inplace = True)

    offer_completed_not_viewed_agg = df_trans_engage[df_trans_engage['offer_viewed'] == 0].groupby('customer_id')[
        'offer_completed'].sum().reset_index()
    offer_completed_not_viewed_agg.rename(columns = {'offer_completed': 'num_offer_completed_not_viewed'},
                                          inplace = True)

    feature_list = [offer_completed_agg, offer_completed_viewed_agg, offer_completed_not_viewed_agg]

    for df_feature in feature_list:
        feature_name = df_feature.columns[1]
        df_train = df_train.merge(df_feature, how = 'left', on = 'customer_id')
        df_train[feature_name] = df_train[feature_name].fillna(0)

    for df_feature in feature_list:
        feature_name = df_feature.columns[1]
        df_test = df_test.merge(df_feature, how = 'left', on = 'customer_id')
        df_test[feature_name] = df_test[feature_name].fillna(0)
    return df_train, df_test


# num_bogo_offer_completed, num_info_offer_completed, num_discount_offer_completed
def offer_completed_by_offer_type(df_train, df_test, df_completed, df_portfolio):
    # merge df_completed with df_portfolio
    offer_completed_agg = df_completed.merge(df_portfolio, how = 'left', on = 'offer_id')
    # bogo
    num_bogo_offer_completed = offer_completed_agg[offer_completed_agg['type_bogo'] == 1].groupby('customer_id')[
        'offer_completed'].sum()
    num_bogo_offer_completed = num_bogo_offer_completed.reset_index().rename(
        columns = {'offer_completed': 'num_bogo_offer_completed'})
    # informational
    num_info_offer_completed = offer_completed_agg[offer_completed_agg['type_informational'] == 1].groupby(
        'customer_id')['offer_completed'].sum()
    num_info_offer_completed = num_info_offer_completed.reset_index().rename(
        columns = {'offer_completed': 'num_info_offer_completed'})
    # discount
    num_discount_offer_completed = offer_completed_agg[offer_completed_agg['type_discount'] == 1].groupby(
        'customer_id')['offer_completed'].sum()
    num_discount_offer_completed = num_discount_offer_completed.reset_index().rename(
        columns = {'offer_completed': 'num_discount_offer_completed'})

    # append features to df_train and df_test
    feature_list = [num_bogo_offer_completed, num_info_offer_completed, num_discount_offer_completed]

    for df_feature in feature_list:
        feature_name = df_feature.columns[1]
        df_train = df_train.merge(df_feature, how = 'left', on = 'customer_id')
        df_train[feature_name] = df_train[feature_name].fillna(0)

    for df_feature in feature_list:
        feature_name = df_feature.columns[1]
        df_test = df_test.merge(df_feature, how = 'left', on = 'customer_id')
        df_test[feature_name] = df_test[feature_name].fillna(0)
    return df_train, df_test


# num_transactions_no_oc, num_transactions_oc_direct, num_transactions_oc_indirect, percent_oc_direct_transactions
def transaction_count_transformer(df_train, df_test, df_trans_engage):
    # direct: sum of transaction_amount >= sum of rewards where offer_completed per customer_id
    # indirect: sum of transaction_amount < sum of rewards where offer_completed per customer_id
    # sum transaction amount >= max difficulty -> direct_offer_completed ELSE indirect_offer_completed

    num_transactions_no_oc = df_trans_engage[df_trans_engage['offer_completed'] == 0].groupby('customer_id')[
        'transaction_id'].nunique().reset_index()
    num_transactions_no_oc.rename(columns = {'transaction_id': 'num_transactions_no_oc'}, inplace = True)
    doc = df_trans_engage[df_trans_engage['offer_completed'] == 1].groupby(
        ['customer_id', 'transaction_id', 'transaction_amount'])['difficulty'].max().reset_index()
    num_transactions_oc_direct = doc[doc['transaction_amount'] >= doc['difficulty']].groupby('customer_id')[
        'transaction_id'].nunique().reset_index()
    num_transactions_oc_direct.rename(columns = {'transaction_id': 'num_transactions_oc_direct'}, inplace = True)
    num_transactions_oc_indirect = doc[doc['transaction_amount'] < doc['difficulty']].groupby('customer_id')[
        'transaction_id'].nunique().reset_index()
    num_transactions_oc_indirect.rename(columns = {'transaction_id': 'num_transactions_oc_indirect'}, inplace = True)

    feature_list = [num_transactions_no_oc, num_transactions_oc_direct, num_transactions_oc_indirect]

    for df_feature in feature_list:
        feature_name = df_feature.columns[1]
        df_train = df_train.merge(df_feature, how = 'left', on = 'customer_id')
        df_train[feature_name] = df_train[feature_name].fillna(0)

    for df_feature in feature_list:
        feature_name = df_feature.columns[1]
        df_test = df_test.merge(df_feature, how = 'left', on = 'customer_id')
        df_test[feature_name] = df_test[feature_name].fillna(0)

    # percent_oc_direct_transactions feature
    df_train['percent_oc_direct_transactions'] = round(
        df_train['num_transactions_oc_direct'] / df_train['transaction_cnt'], 4)
    df_test['percent_oc_direct_transactions'] = round(df_test['num_transactions_oc_direct'] / df_test['transaction_cnt'], 4)
    return df_train, df_test


# offer_view_rate, offer_completion_rate, info_view_rate
def offer_ratio_calculations(df_train, df_test):
    # offer_view_rate = df_train.num_offer_viewed / df_train.num_offer_received
    df_train['offer_view_rate'] = round(1.0 * df_train.num_offer_viewed / df_train.num_offer_received, 2)
    df_train['offer_view_rate'].fillna(0, inplace = True)
    df_test['offer_view_rate'] = round(1.0 * df_test.num_offer_viewed / df_test.num_offer_received, 2)
    df_test['offer_view_rate'].fillna(0, inplace = True)

    # offer_completion_rate = df_train.num_offer_completed / (df_train.num_offer_received - df_train.num_info_offer_received)
    df_train['offer_completion_rate'] = round(
        1.0 * df_train.num_offer_completed / (df_train.num_offer_received - df_train.num_info_offer_received), 2)
    df_train['offer_completion_rate'].fillna(0, inplace = True)
    df_test['offer_completion_rate'] = round(
        1.0 * df_test.num_offer_completed / (df_test.num_offer_received - df_test.num_info_offer_received), 2)
    df_test['offer_completion_rate'].fillna(0, inplace = True)

    # info_view_rate
    df_train['info_view_rate'] = 1.0 * df_train['num_info_offer_viewed'] / df_train['num_info_offer_received']
    df_train['info_view_rate'].fillna(0, inplace = True)
    df_test['info_view_rate'] = 1.0 * df_test['num_info_offer_viewed'] / df_test['num_info_offer_received']
    df_test['info_view_rate'].fillna(0, inplace = True)
    return df_train, df_test


# total_reward_amount, avg_reward_per_oc_transaction
def offer_reward_transformer(df_train, df_test, df_trans_engage):
    # total_reward_amount
    total_reward_amount = df_trans_engage.groupby('customer_id')['reward'].sum().reset_index()
    total_reward_amount.rename(columns = {'reward': 'total_reward_amount'}, inplace = True)

    df_train = df_train.merge(total_reward_amount, how = 'left', on = 'customer_id')
    df_train['total_reward_amount'].fillna(0, inplace = True)
    df_test = df_test.merge(total_reward_amount, how = 'left', on = 'customer_id')
    df_test['total_reward_amount'].fillna(0, inplace = True)

    # avg_reward_per_oc_transaction
    df_train['avg_reward_per_oc_transaction'] = round(1.0 * df_train['total_reward_amount'] / (
                df_train['num_transactions_oc_direct'] + df_train['num_transactions_oc_indirect']), 2)
    df_train['avg_reward_per_oc_transaction'].fillna(0, inplace = True)

    df_test['avg_reward_per_oc_transaction'] = round(1.0 * df_test['total_reward_amount'] / (
                df_test['num_transactions_oc_direct'] + df_test['num_transactions_oc_indirect']), 2)
    df_test['avg_reward_per_oc_transaction'].fillna(0, inplace = True)
    return df_train, df_test


# transaction_oc_amount, transaction_aos_oc, transaction_no_oc_amount, transaction_aos_no_oc
def transaction_amount_transformer(df_train, df_test, df_trans_engage):
    # transaction_oc_amount
    transaction_oc_amount = df_trans_engage[df_trans_engage['offer_completed'] == 1].groupby('customer_id')[
        'transaction_amount'].sum().reset_index()
    transaction_oc_amount.rename(columns = {'transaction_amount': 'transaction_oc_amount'}, inplace = True)

    # transaction_no_oc_amount
    transaction_no_oc_amount = df_trans_engage[df_trans_engage['offer_completed'] == 0].groupby('customer_id')[
        'transaction_amount'].sum().reset_index()
    transaction_no_oc_amount.rename(columns = {'transaction_amount': 'transaction_no_oc_amount'}, inplace = True)

    # transaction_aos_oc
    df_train = df_train.merge(transaction_oc_amount, how = 'left', on = 'customer_id')
    df_train['transaction_oc_amount'].fillna(0, inplace = True)
    df_train['transaction_aos_oc'] = 1.0 * df_train['transaction_oc_amount'] / (
                df_train['num_transactions_oc_direct'] + df_train['num_transactions_oc_indirect'])
    df_train['transaction_aos_oc'].fillna(0, inplace = True)
    df_test = df_test.merge(transaction_oc_amount, how = 'left', on = 'customer_id')
    df_test['transaction_oc_amount'].fillna(0, inplace = True)
    df_test['transaction_aos_oc'] = 1.0 * df_test['transaction_oc_amount'] / (
                df_test['num_transactions_oc_direct'] + df_test['num_transactions_oc_indirect'])
    df_test['transaction_aos_oc'].fillna(0, inplace = True)

    # transaction_aos_no_oc
    df_train = df_train.merge(transaction_no_oc_amount, how = 'left', on = 'customer_id')
    df_train['transaction_no_oc_amount'].fillna(0, inplace = True)
    df_train['transaction_aos_no_oc'] = 1.0 * df_train['transaction_no_oc_amount'] / df_train['num_transactions_no_oc']
    df_train['transaction_aos_no_oc'].fillna(0, inplace = True)
    df_test = df_test.merge(transaction_no_oc_amount, how = 'left', on = 'customer_id')
    df_test['transaction_no_oc_amount'].fillna(0, inplace = True)
    df_test['transaction_aos_no_oc'] = 1.0 * df_test['transaction_no_oc_amount'] / df_test['num_transactions_no_oc']
    df_test['transaction_aos_no_oc'].fillna(0, inplace = True)
    return df_train, df_test


# median_offer_duration
def median_offer_duration_calc(df_train, df_test, df_trans_engage):
    # take median value of all offer durations per customer_id
    median_offer_duration = df_trans_engage[df_trans_engage['offer_completed'] == 1].groupby('customer_id')[
        'duration'].median().reset_index()
    median_offer_duration.rename(columns = {'duration': 'median_offer_duration'}, inplace = True)

    df_train = df_train.merge(median_offer_duration, how = 'left', on = 'customer_id')
    df_train['median_offer_duration'].fillna(0, inplace = True)
    df_test = df_test.merge(median_offer_duration, how = 'left', on = 'customer_id')
    df_test['median_offer_duration'].fillna(0, inplace = True)
    return df_train, df_test


# avg_offer_completion_time
def avg_offer_completion_time_calc(df_train, df_test, df_trans_engage):
    # avg_offer_completion_time = sum(offer_completed_time - offer_received_time) / count(offer_completed)
    feature_columns = ['customer_id', 'offer_completed_time', 'offer_received_time', 'offer_completed']
    avg_oc_time_stg = df_trans_engage[df_trans_engage['offer_completed'] == 1][feature_columns]
    avg_oc_time_stg['oct-ort'] = avg_oc_time_stg['offer_completed_time'] - avg_oc_time_stg['offer_received_time']

    sum_diff_time = avg_oc_time_stg.groupby('customer_id')['oct-ort'].sum().reset_index()
    sum_offers_completed = avg_oc_time_stg.groupby('customer_id')['offer_completed'].sum().reset_index()

    avg_oc_time_stg_v2 = sum_diff_time.merge(sum_offers_completed, how = 'left', on = 'customer_id')
    avg_oc_time_stg_v2['avg_offer_completion_time'] = 1.0 * avg_oc_time_stg_v2['oct-ort'] / avg_oc_time_stg_v2[
        'offer_completed']

    df_train = df_train.merge(avg_oc_time_stg_v2[['customer_id', 'avg_offer_completion_time']], how = 'left',
                              on = 'customer_id')
    df_train['avg_offer_completion_time'].fillna(0, inplace = True)

    df_test = df_test.merge(avg_oc_time_stg_v2[['customer_id', 'avg_offer_completion_time']], how = 'left',
                            on = 'customer_id')
    df_test['avg_offer_completion_time'].fillna(0, inplace = True)
    return df_train, df_test


# Function iterates over each customer_id and creates a list of all transaction_time per transaction_id
# and calculates the difference between current and previous transaction, and then averages the spreads.
def avg_hrs_bw_trans(df_train, df_test, df_trans_engage):
    df_transactions_stg = df_trans_engage.groupby(['transaction_id', 'customer_id', 'transaction_time'])[
        'transaction_amount'].max().reset_index()
    list_of_ids = df_transactions_stg.customer_id.unique()

    customer_ids = []
    avg_hrs_bw_transactions = []
    counter = 0
    for id_ in list_of_ids:
        transaction_times = df_trans_engage[df_trans_engage['customer_id'] == id_]['transaction_time']
        transaction_times = sorted(transaction_times)
        if len(transaction_times) > 1:
            diff_list = []
            n = len(transaction_times) - 1
            for idx in range(n):
                t1 = transaction_times[idx]
                t2 = transaction_times[idx + 1]
                diff = t2 - t1
                diff_list.append(diff)
            avg_hrs = np.mean(diff_list)
            customer_ids.append(id_)
            avg_hrs_bw_transactions.append(avg_hrs)
            counter += 1
        else:
            customer_ids.append(id_)
            avg_hrs_bw_transactions.append(0)
            counter += 1
    avg_hrs_trans = pd.DataFrame({'customer_id': customer_ids, 'avg_hrs_bw_transactions': avg_hrs_bw_transactions})
    # append to df_train and df_test
    df_train = df_train.merge(avg_hrs_trans, how = 'left', on = 'customer_id')
    df_train['avg_hrs_bw_transactions'].fillna(0, inplace = True)
    df_test = df_test.merge(avg_hrs_trans, how = 'left', on = 'customer_id')
    df_test['avg_hrs_bw_transactions'].fillna(0, inplace = True)
    return df_train, df_test


# num_oc_ch_social, num_oc_ch_web, num_oc_ch_mobile, num_oc_ch_email
def offer_channel_counter(df_train, df_test, df_trans_engage):
    num_oc_by_channel = df_trans_engage[df_trans_engage['offer_completed'] == 1].groupby(
        'customer_id')[['ch_web', 'ch_social', 'ch_mobile', 'ch_email']].sum().reset_index()
    num_oc_by_channel.rename(columns = {'ch_web': 'num_oc_ch_web',
                                        'ch_social': 'num_oc_ch_social',
                                        'ch_mobile': 'num_oc_ch_mobile',
                                        'ch_email': 'num_oc_ch_email'}, inplace = True)

    df_train = df_train.merge(num_oc_by_channel, how = 'left', on = 'customer_id')
    df_test = df_test.merge(num_oc_by_channel, how = 'left', on = 'customer_id')

    features = ['num_oc_ch_social', 'num_oc_ch_web', 'num_oc_ch_mobile', 'num_oc_ch_email']
    for feature in features:
        df_train[feature].fillna(0, inplace = True)
        df_test[feature].fillna(0, inplace = True)
    return df_train, df_test


# Avg offer received frequency (avg_offer_received_freq)
def avg_offer_received_frequency(df_train, df_test, df_received):
    customer_ids = df_received.customer_id.unique()
    customer_id = []
    avg_or_freq = []
    counter = 1
    for id_ in customer_ids:
        time_series = df_received[df_received['customer_id'] == id_].offer_received_time.to_list()
        n = len(time_series)
        if n > 1:
            t_diff = []
            for idx in range(n-1):
                t1 = time_series[idx]
                t2 = time_series[idx+1]
                diff = t2 - t1
                t_diff.append(diff)
            t_mean = np.mean(t_diff)
            avg_or_freq.append(t_mean)
            customer_id.append(id_)
            counter += 1
        else:
            t_single = time_series[0]
            avg_or_freq.append(t_single)
            customer_id.append(id_)
            counter += 1
    avg_offer_received_freq = pd.DataFrame({'customer_id': customer_id, 'avg_offer_received_freq': avg_or_freq})
    df_train = df_train.merge(avg_offer_received_freq, how = 'left', on = 'customer_id')
    df_train['avg_offer_received_freq'].fillna(0, inplace = True)
    df_test = df_test.merge(avg_offer_received_freq, how = 'left', on = 'customer_id')
    df_test['avg_offer_received_freq'].fillna(0, inplace = True)
    return df_train, df_test


if __name__ == '__main__':
    start_time = datetime.now()

    # read transaction engagement aggregated data
    file_gzip = os.path.join(config.DATA_DIR, config.TRANSACTION_ENGAGEMENT_FILE)
    trans_engage = pd.read_csv(file_gzip, compression = 'gzip')

    # read in the json files
    profile = pd.read_json(os.path.join(config.DATA_DIR, config.PROFILE_FILE), orient = 'records', lines = True)
    portfolio = pd.read_json(os.path.join(config.DATA_DIR, config.PORTFOLIO_FILE), orient = 'records', lines = True)
    transcript = pd.read_json(os.path.join(config.DATA_DIR, config.TRANSCRIPT_FILE), orient = 'records', lines = True)

    # build all base dataframes
    base = BaseTransformDF()
    portfolio = base.portfolio_expanded(portfolio)
    transcript = base.create_transcript_copy(transcript)
    completed = base.create_df_base_completed(transcript)
    received = base.create_df_base_received(transcript)
    viewed = base.create_df_base_viewed(transcript)

    X_train_0, X_test_0 = base_train_test_split(profile, trans_engage)
    print("X_train_0, X_test_0: base_train_test_split completed")
    X_train_1, X_test_1 = missing_outlier_imputer(X_train_0, X_test_0)
    print("X_train_1, X_test_1: missing_outlier_imputer completed")
    X_train_2, X_test_2 = age_quantile_transformer(X_train_1, X_test_1)
    print("X_train_2, X_test_2: age_quantile_transformer completed")
    X_train_3, X_test_3 = income_quantile_transformer(X_train_2, X_test_2)
    print("X_train_3, X_test_3: income_quantile_transformer completed")
    X_train_4, X_test_4 = date_registered_transformer(X_train_3, X_test_3)
    print("X_train_4, X_test_4: date_registered_transformer completed")
    X_train_5, X_test_5 = offer_received_transformer(X_train_4, X_test_4, received, portfolio)
    print("X_train_5, X_test_5: offer_received_transformer completed")
    X_train_6, X_test_6 = offer_viewed_transformer(X_train_5, X_test_5, viewed, portfolio)
    print("X_train_6, X_test_6: offer_viewed_transformer completed")
    X_train_7, X_test_7 = offer_completed_transformer(X_train_6, X_test_6, trans_engage)
    print("X_train_7, X_test_7: offer_completed_transformer completed")
    X_train_8, X_test_8 = offer_completed_by_offer_type(X_train_7, X_test_7, completed, portfolio)
    print("X_train_8, X_test_8: offer_completed_by_offer_type completed")
    X_train_9, X_test_9 = transaction_count_transformer(X_train_8, X_test_8, trans_engage)
    print("X_train_9, X_test_9: transaction_count_transformer completed")
    X_train_10, X_test_10 = offer_ratio_calculations(X_train_9, X_test_9)
    print("X_train_10, X_test_10: offer_ratio_calculations completed")
    X_train_11, X_test_11 = offer_reward_transformer(X_train_10, X_test_10, trans_engage)
    print("X_train_11, X_test_11: offer_reward_transformer completed")
    X_train_12, X_test_12 = transaction_amount_transformer(X_train_11, X_test_11, trans_engage)
    print("X_train_12, X_test_12: transaction_amount_transformer completed")
    X_train_13, X_test_13 = median_offer_duration_calc(X_train_12, X_test_12, trans_engage)
    print("X_train_13, X_test_13: median_offer_duration_calc completed")
    X_train_14, X_test_14 = avg_offer_completion_time_calc(X_train_13, X_test_13, trans_engage)
    print("X_train_14, X_test_14: avg_offer_completion_time_calc completed")
    X_train_15, X_test_15 = avg_hrs_bw_trans(X_train_14, X_test_14, trans_engage)
    print("X_train_15, X_test_15: avg_hrs_bw_trans completed")
    X_train_16, X_test_16 = offer_channel_counter(X_train_15, X_test_15, trans_engage)
    print("X_train_16, X_test_16: offer_channel_counter completed")

    X_train, X_test = avg_offer_received_frequency(X_train_16, X_test_16, received)
    print("X_train, X_test: avg_offer_received_frequency completed.")
    print("----")

    # write df to gzipped csv file
    Path(os.path.join(config.DATA_DIR)).mkdir(parents = True, exist_ok = True)
    train_file_path = os.path.join(config.DATA_DIR, config.TRAIN_FILE)
    test_file_path = os.path.join(config.DATA_DIR, config.TEST_FILE)

    try:
        X_train.to_csv(train_file_path, index = False, compression = 'gzip')
        print("Success: {} created".format(config.TRAIN_FILE))
        print("  File_path_name: {}".format(train_file_path))
        X_test.to_csv(test_file_path, index = False, compression = 'gzip')
        print("Success: {} created".format(config.TEST_FILE))
        print("  File_path_name: {}".format(test_file_path))
    except:
        print("Error: File creations failed.")
    # measure script run time
    print('Script_run_time: {} (hour:minute:second:microsecond)'.format((datetime.now() - start_time)))
