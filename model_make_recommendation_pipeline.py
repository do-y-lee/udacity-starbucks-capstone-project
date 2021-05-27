import os
import json
import random
from datetime import datetime
import pandas as pd
import config
import warnings
warnings.filterwarnings('ignore')


def make_offer_recommendations(offer_type_list, top_offer_type_list, cid_list):
    # start time
    start_time = datetime.now()

    # dictionaries to capture random and predicted recommendations
    model_recommendations = {}
    random_recommendations = {}

    # counters to measure script progression
    num_customers = 0
    num_model = 0
    num_random = 0

    for cid in cid_list:
        pred_proba_offers = {}
        for offer_type in offer_type_list:
            preds = pd.read_csv('{}/{}_test_predictions.csv.gz'.format(config.PREDICTIONS_DIR, offer_type), compression ='gzip')
            try:
                pred = preds[(preds.customer_id == cid) & (preds.y_predict == 1)]
                if len(pred) != 0:
                    pred_max = pred.groupby(['customer_id', 'y_predict'])['y_max_pred_proba'].max().reset_index()
                    pred_proba_offers[offer_type] = float(pred_max.y_max_pred_proba)
            except:
                continue

        if len(pred_proba_offers) != 0:
            pred_proba_offers_sorted = sorted(pred_proba_offers.items(), key = lambda x: x[1], reverse = True)
            model_recommendations[cid] = list(pred_proba_offers_sorted[0])
            num_model += 1
        else:
            offer_type_choice = random.choice(top_offer_type_list)
            random_recommendations[cid] = offer_type_choice
            num_random += 1

        num_customers += 1
        if num_customers % 1000 == 0:
            print("{} customer recommendations generated.".format(num_customers))
            print("    {} model recommendations generated.".format(num_model))
            print("    {} random recommendations generated.".format(num_random))
            print(" ... Process time: {}".format(datetime.now() - start_time))

    model_rec_df = pd.DataFrame.from_dict(model_recommendations, orient = 'index', columns = ['offer_type', 'pred_proba'])
    model_rec_df.to_csv(os.path.join(config.RECOMMENDATIONS_DIR, config.MODEL_RECOMMENDATION_FILE),
                        index = False,
                        compression = 'gzip')

    random_rec_df = pd.DataFrame.from_dict(random_recommendations, orient = 'index', columns = ['offer_type'])
    random_rec_df.to_csv(os.path.join(config.RECOMMENDATIONS_DIR, config.RANDOM_RECOMMENDATION_FILE),
                         index = False,
                         compression = 'gzip')


def identify_top_offer_types():
    # top 4 offer types using rank_contribution_score for random assignment to customers without "1" predictions
    offer_funnel = pd.read_csv(os.path.join(config.DATA_DIR, config.OFFER_FUNNEL_FILE), compression = 'gzip')
    offer_funnel = offer_funnel[offer_funnel.offer_type != 'informational']
    offer_funnel['engagement_score'] = offer_funnel['viewed_rate'] * offer_funnel['completion_rate']
    offer_funnel['normalized_difficulty'] = 1.0 * offer_funnel.difficulty / offer_funnel.duration
    offer_funnel['contribution_score'] = offer_funnel.engagement_score * (1 - offer_funnel.normalized_difficulty)
    offer_funnel['rank_contribution_score'] = offer_funnel.contribution_score.rank(method = 'dense', ascending = False)
    offer_funnel['offer_type_v2'] = offer_funnel.apply(
        lambda x: x.offer_type+'-'+str(x.reward)+'-'+str(x.difficulty)+'-'+str(x.duration), axis = 1)
    top_four = offer_funnel[offer_funnel.rank_contribution_score <= 4].offer_type_v2.tolist()
    return top_four


if __name__ == '__main__':
    # load training_feature_sets.json to create list of offer types
    feature_sets = json.load(open(os.path.join(config.TRAINED_MODELS_DIR, config.TRAIN_FEATURE_SETS_FILE)))
    offer_types = feature_sets.keys()
    # identify top four offer types
    top_offer_types = identify_top_offer_types()
    # load test_v3_starbucks.csv.gz to create customer_id list
    test_v3 = pd.read_csv(os.path.join(config.DATA_DIR, config.TEST_V3_FILE), compression = 'gzip')
    cids = test_v3.customer_id.unique()
    # generate recommendations
    make_offer_recommendations(offer_types, top_offer_types, cids)
