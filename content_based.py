import pandas as pd
import numpy as np
import boto3
import io

client = boto3.client('s3', aws_access_key_id = '', aws_secret_access_key = '')
bucket = 'voicefinder-bucket'
obj = client.get_object(Bucket = bucket, Key = 'markets10c.csv')
market_data = pd.read_csv(io.BytesIO(obj['Body'].read()))
market_data['market_id'] = market_data['market_id'].astype(int)

"""# 2. market profile 구성"""

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# TF-IDF 벡터화
tfidf_vectorizer_menu = TfidfVectorizer()
tfidf_matrix_menu = tfidf_vectorizer_menu.fit_transform(market_data['menu'])

tfidf_vectorizer_ingredient = TfidfVectorizer()
tfidf_matrix_ingredient = tfidf_vectorizer_ingredient.fit_transform(market_data['ingredient'])

def filter_markets(data, cannot_eat_vec, feature_names):
    df_cleaned = data.copy()
    for idx, food in enumerate(feature_names):
        if cannot_eat_vec[0, idx] > 0:  # cannot_eat 벡터에서 해당 음식 항목의 값이 0보다 큰 경우
            df_cleaned = df_cleaned[~df_cleaned['ingredient'].str.contains(food)]
    return df_cleaned

def get_recommendations(user_profiles, n_cnt):
    # 사용자 가져오기
    user_fav_vector = tfidf_vectorizer_menu.transform([user_profiles['fav_food']])

    cannot_eat_vector = tfidf_vectorizer_ingredient.transform([user_profiles['cannot_eat']])
    feature_names = tfidf_vectorizer_ingredient.get_feature_names_out()

    filtered_markets = filter_markets(market_data, cannot_eat_vector, feature_names)

    if filtered_markets.empty:
            return pd.DataFrame()  # 필터링 후 남은 음식점이 없는 경우 빈 데이터프레임 반환

    # 필터링된 음식점 메뉴 벡터화
    filtered_tfidf_matrix_menu = tfidf_vectorizer_menu.transform(filtered_markets['menu'])

    # 코사인 유사도 계산
    cosine_similarities = cosine_similarity(user_fav_vector, filtered_tfidf_matrix_menu).flatten()

    # 예측 평점 벡터 생성
    recommendation_scores = pd.Series(cosine_similarities, index=filtered_markets['market_id'])

    print("content")
    print(recommendation_scores)
    return recommendation_scores

