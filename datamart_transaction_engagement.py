import os
import uuid
from pathlib import Path
from datetime import datetime

import pandas as pd
from base_transforms import BaseTransformDF
import config


def create_df_transaction_engagement(df_transactions, df_completed, df_received, df_viewed, df_portfolio) -> pd.DataFrame:
    """create final df_transaction_engagement"""
    # ---------------------------------------- #
    # step 1 ---- create df_engagement_v1 ---- #
    # ---------------------------------------- #
    df_engagement_v1 = pd.merge(df_transactions, df_completed, how = 'left',
                                left_on = ['customer_id', 'transaction_time'],
                                right_on = ['customer_id', 'offer_completed_time'],
                                suffixes = ['', '_drop'])
    df_engagement_v1['offer_completed'] = df_engagement_v1['offer_completed'].fillna(0).astype(int)
    df_engagement_v1['offer_completed_time'] = df_engagement_v1['offer_completed_time'].fillna(-1).astype(int)
    df_engagement_v1['offer_id'] = df_engagement_v1['offer_id'].fillna('no-offer')
    df_engagement_v1.drop_duplicates(inplace = True)

    # create record_id to track number of unique records/rows per transaction_id
    record_ids = []
    n = len(df_engagement_v1)
    for _ in range(n):
        _id = uuid.uuid4().hex
        record_ids.append(_id)

    df_engagement_v1.insert(loc = 0, column = 'record_id', value = record_ids)

    # ----------------------------------------------- #
    # step 2 ---- create df_received_conversions ---- #
    # ----------------------------------------------- #
    df_received_conversions = pd.merge(df_engagement_v1, df_received, how = 'inner', on = ['customer_id', 'offer_id'])
    df_received_conversions['tt-rt'] = df_received_conversions['transaction_time'] - df_received_conversions['offer_received_time']
    df_received_conversions = df_received_conversions[df_received_conversions['tt-rt'] >= 0]
    # take min to remove duplicates; offer_viewed_time column not included
    df_received_conversions = df_received_conversions.groupby(
        ['record_id', 'transaction_id', 'customer_id', 'transaction_time', 'transaction_amount',
         'offer_completed', 'offer_completed_time', 'offer_id', 'offer_received'])['tt-rt'].min().reset_index()

    df_received_conversions['offer_received_time'] = df_received_conversions['transaction_time'] - df_received_conversions['tt-rt']
    df_received_conversions.drop(columns = ['tt-rt'], inplace = True)
    df_received_conversions = df_received_conversions[['record_id', 'offer_received', 'offer_received_time']]

    # ---------------------------------------- #
    # step 3 ---- create df_engagement_v2 ---- #
    # ---------------------------------------- #
    df_engagement_v2 = pd.merge(df_engagement_v1, df_received_conversions, how = 'left', on = 'record_id')
    df_engagement_v2['offer_received'] = df_engagement_v2['offer_received'].fillna(0).astype(int)
    df_engagement_v2['offer_received_time'] = df_engagement_v2['offer_received_time'].fillna(-1).astype(int)

    # --------------------------------------------- #
    # step 4 ---- create df_viewed_conversions ---- #
    # --------------------------------------------- #
    df_viewed_conversions = pd.merge(df_engagement_v2, df_viewed, how = 'inner', on = ['customer_id', 'offer_id'])

    df_viewed_conversions = df_viewed_conversions[
        (df_viewed_conversions['transaction_time'] >= df_viewed_conversions['offer_viewed_time'])
        & (df_viewed_conversions['offer_viewed_time'] >= df_viewed_conversions['offer_received_time'])]

    df_viewed_conversions = df_viewed_conversions[['record_id', 'offer_viewed', 'offer_viewed_time']]

    # ---------------------------------------- #
    # step 5 ---- create df_engagement_v3 ---- #
    # ---------------------------------------- #
    df_engagement_v3 = pd.merge(df_engagement_v2, df_viewed_conversions, how = 'left', on = 'record_id')
    df_engagement_v3['offer_viewed'] = df_engagement_v3['offer_viewed'].fillna(0).astype(int)
    df_engagement_v3['offer_viewed_time'] = df_engagement_v3['offer_viewed_time'].fillna(-1).astype(int)

    # ----------------------------------------------------------------- #
    # step 6 ---- create df_engagement_v4 with portfolio dataframe ---- #
    # ----------------------------------------------------------------- #
    # merge df_engagement_v3 with portfolio to add offer metadata
    df_engagement_v4 = pd.merge(df_engagement_v3, df_portfolio, how='left', on='offer_id')
    df_engagement_v4 = df_engagement_v4.drop(columns=['channels', 'offer_type']).fillna(0)
    return df_engagement_v4


def main():
    """
    Script Output
    -------------
    Success: Transaction engagement dataframe created -- df_transaction_engagement.
    Dataframe_shape: (141915, 23)
    Success: starbucks_transaction_engagement.csv.gz created.
    File_path_name: /Users/dlee/ds/repos/udacity-starbucks-capstone-project/data/starbucks_transaction_engagement.csv.gz
    Script_run_time: 0:00:07.083148 (hour:minute:second:microsecond)
    """
    start_time = datetime.now()
    # read in the json files
    portfolio = pd.read_json(os.path.join(config.DATA_DIR, config.PORTFOLIO_FILE), orient = 'records', lines = True)
    transcript = pd.read_json(os.path.join(config.DATA_DIR, config.TRANSCRIPT_FILE), orient = 'records', lines = True)

    # build all base dataframes - execute functions
    df_base = BaseTransformDF()
    portfolio_expanded = df_base.portfolio_expanded(portfolio)
    transcript_v2 = df_base.create_transcript_copy(transcript)
    transactions = df_base.create_df_transactions(transcript_v2)
    completed = df_base.create_df_base_completed(transcript_v2)
    received = df_base.create_df_base_received(transcript_v2)
    viewed = df_base.create_df_base_viewed(transcript_v2)

    transaction_engagement = create_df_transaction_engagement(transactions, completed, received, viewed, portfolio_expanded)
    transaction_engagement.drop_duplicates(inplace = True)
    print("Success: Transaction engagement dataframe created -- df_transaction_engagement.")
    dim = transaction_engagement.shape
    print("Dataframe_shape: {}".format(dim))

    # create data directory if not exists
    Path(os.path.join(config.DATA_DIR)).mkdir(parents=True, exist_ok=True)
    # write df to gzipped csv file
    file_path = os.path.join(config.DATA_DIR, config.TRANSACTION_ENGAGEMENT_FILE)
    transaction_engagement.to_csv(file_path, index=False, compression='gzip')
    print("Success: {} created".format(config.TRANSACTION_ENGAGEMENT_FILE))
    print("  File_path_name: {}".format(file_path))
    print('Script_run_time: {} (hour:minute:second:microsecond)'.format((datetime.now() - start_time)))


if __name__ == '__main__':
    main()
