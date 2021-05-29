# list of directories
DATA_DIR = 'data'
OUTPUT_DIR = 'output'
DIAGNOSTIC_METRICS_DIR = 'output/diagnostic_metrics'
PREDICTIONS_DIR = 'output/predictions'
RECOMMENDATIONS_DIR = 'output/recommendations'
TRAINED_MODELS_DIR = 'trained_models'

# original raw json files
PORTFOLIO_FILE = 'portfolio.json'
PROFILE_FILE = 'profile.json'
TRANSCRIPT_FILE = 'transcript.json'

# test set accuracy scores file name
ACCURACY_TEST_FILE = 'accuracy_test_scores.json'
# for each offer type, a list of selected features
TRAIN_FEATURE_SETS_FILE = 'training_feature_sets.json'

# aggregated csv files
TRANSACTION_ENGAGEMENT_FILE = 'starbucks_transaction_engagement.csv.gz'
OFFER_FUNNEL_FILE = 'starbucks_offers_funnel_analysis.csv.gz'
TRAIN_FILE = 'train_starbucks.csv.gz'
TEST_FILE = 'test_starbucks.csv.gz'
TRAIN_V3_FILE = 'train_v3_starbucks.csv.gz'
TEST_V3_FILE = 'test_v3_starbucks.csv.gz'

# recommendation logic output
MODEL_RECOMMENDATION_FILE = 'model_recommendations.csv.gz'
RANDOM_RECOMMENDATION_FILE = 'random_recommendations.csv.gz'

# ordinal (categorical) encoding
ORDINAL_COLUMNS = ['age_quantile_label', 'days_reg_quantile_label', 'income_quantile_label']

DROP_GENERIC_COLUMNS = ['gender_NA', 'income_NA', 'age_NA']

DROP_REDUNDANT_COLUMNS = ['gender', 'age_quantile_range', 'date_registered', 'age_quantile_label',
                          'days_reg_quantile_label', 'income_quantile_label', 'days_reg_quantile_range',
                          'income_quantile_range']

DROP_FEAT_SELECT_COLUMNS = ['customer_id', 'reward', 'difficulty', 'duration', 'num_channels',
                            'ch_web', 'ch_email', 'ch_mobile', 'ch_social', 'type_bogo', 'type_informational',
                            'type_discount', 'offer_id', 'offer_type', 'offer_type_v2', 'offer_completed_time']
