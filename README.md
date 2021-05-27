# Starbucks Capstone Project

> Predicting the best offer type for every customer with the highest probability of completion.

---

## Description

- Using the raw JSON files (transcript, profile, and portfolio), the goal is to create a model that can predict if a customer will complete an offer or not.
- The prediction pipeline is composed of eight random forest models for each reward-driven offers.
- The pipeline identifies the positive predictions and identifies the offer with the highest prediction probability as the winner.
- The customers who received all negative predictions are assigned a random offer from top four performing offers.


## Full Pipeline Design

![](output/diagnostic_metrics/offer_model_pred_flow.png)


## Dev Environment

- PyCharm
- Python 3.7.*
- conda environment: `environment.yml`


## Execution Setup

| Script Name | Description |
| ---- | ----------- |
| base_transforms/base_transforms_df.py | Contains class object to create base dataframes. |
| datamart_offer_funnel_view.py | Offer funnel data mart. |
| datamart_transaction_engagement.py | Transaction engagement data mart. |
| preprocessor_feat_engine_layer_1.py | Feature engineering to produce train and test v1. |
| preprocessor_feat_engine_layer_2.py | Feature engineering to produce train and test v3. |
| preprocessor_feat_select_pipeline.py | Feature selection to identify most important features per offer type. |
| model_train_predict_pipeline.py | Training models using train v3 sets and generate predictions using test v3 sets.  |
| model_make_recommendation_pipeline.py | The logic takes the saved test predictions and makes the decision of best offer per customer. |
| output_model_stats.py | Generates model diagnostic metrics and plots for train and test sets. |

