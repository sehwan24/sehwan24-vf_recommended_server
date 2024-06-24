import sys
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, MultiLabelBinarizer
from sklearn.metrics.pairwise import cosine_similarity

import content_based
import collaborate_filtering

import boto3
import io

client = boto3.client('s3', aws_access_key_id = '', aws_secret_access_key = '')
bucket = 'voicefinder-bucket'
obj = client.get_object(Bucket = bucket, Key = 'markets10c.csv')
market_data = pd.read_csv(io.BytesIO(obj['Body'].read()))

market_data['market_id'] = market_data['market_id'].astype(int)


# In[21]:

# 하이브리드 추천 시스템
def get_recommendations(user_profile, alpha=0.5):
    user_id = user_profile['user_id']
    print(user_id)
    cb_recommendation = content_based.get_recommendations(user_profile, 3)
    print("1")
    print(cb_recommendation)
    cf_recommendation = collaborate_filtering.get_recommendations(user_id)
    print("2")
    print(cb_recommendation)
    print(cf_recommendation)
    cb_recommendation.index = cb_recommendation.index.astype(int)
    cf_recommendation.index = cf_recommendation.index.astype(int)
    combined_recommendation = alpha * cb_recommendation + (1 - alpha) * cf_recommendation
    print(combined_recommendation)
    print("3")
    print(combined_recommendation.sort_values(ascending=False).head(10))
    top_recommendations = combined_recommendation.sort_values(ascending=False).head(3)
    return market_data[market_data['market_id'].isin(top_recommendations.index)]['market_id']




