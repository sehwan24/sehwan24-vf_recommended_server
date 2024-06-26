#!/usr/bin/env python
# coding: utf-8

# In[4]:


import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, MultiLabelBinarizer
from sklearn.metrics.pairwise import cosine_similarity
import logging

# 파일에서 함수 불러오기
import md_cb
import md_cf

#logging.basicConfig(level=logging.DEBUG)

client = boto3.client('s3', aws_access_key_id = '', aws_secret_access_key = '')
bucket = 'voicefinder-bucket'
obj3 = client.get_object(Bucket = bucket, Key = 'markets10c.csv')
market_data = pd.read_csv(io.BytesIO(obj3['Body'].read())) 
market_data['market_id'] = market_data['market_id'].astype(int)


# In[6]:


def scale_ratings(ratings):
    min_rating = ratings.min()
    max_rating = ratings.max()
    scaled_ratings = (ratings - min_rating) / (max_rating - min_rating)
    return scaled_ratings


# In[9]:


# 하이브리드 추천 시스템
def get_recommendations(user_profile):
    try:
        user_id = user_profile['user_id']

        cb_recommendation = md_cb.get_recommendations(user_profile)
        cf_recommendation = md_cf.get_recommendations(user_id)

        # 스케일 조정
        cb_recommendation_scaled = scale_ratings(cb_recommendation)
        cf_recommendation_scaled = scale_ratings(cf_recommendation)

        alpha=0.5
        # 결합 추천
        combined_recommendation = alpha * cb_recommendation_scaled + (1 - alpha) * cf_recommendation_scaled

        top_recommendations = combined_recommendation.sort_values(ascending=False).head(3)
        return market_data[market_data['market_id'].isin(top_recommendations.index)]['market_id']

    except Exception as e:
            logging.error(f"Hybrid combination error: {e}")
            return None


# In[10]:





# In[ ]:




