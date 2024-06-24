import pandas as pd
from sklearn.preprocessing import LabelEncoder, MultiLabelBinarizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import boto3
import io
import chardet
import logging
from datetime import datetime

client = boto3.client('s3', aws_access_key_id = '', aws_secret_access_key = '')
bucket = 'voicefinder-bucket'
obj = client.get_object(Bucket = bucket, Key = 'sample_data.csv')
obj2 = client.get_object(Bucket = bucket, Key = 'users.csv')
obj3 = client.get_object(Bucket = bucket, Key = 'markets10c.csv')
obj4 = client.get_object(Bucket = bucket, Key = 'rating10f.csv')
sample_data = pd.read_csv(io.BytesIO(obj['Body'].read()))
users_data = pd.read_csv(io.BytesIO(obj2['Body'].read()))
markets = pd.read_csv(io.BytesIO(obj3['Body'].read()))
ratings = pd.read_csv(io.BytesIO(obj4['Body'].read()))
print(ratings)
markets['market_id'] = markets['market_id'].astype(int)
users_data['user_id'] = users_data['user_id'].astype(int)


users = users_data[['user_id', 'birth_date', 'gender', 'considerations']]

gender_le = LabelEncoder()
users['gender_encoded'] = gender_le.fit_transform(users['gender'])

def calculate_age(birthdate):
    today = datetime.today()
    birthdate = datetime.strptime(birthdate, "%Y-%m-%d")
    age = today.year - birthdate.year - ((today.month, today.day) < (birthdate.month, birthdate.day))
    return age

users['birth_date'] = users['birth_date'].apply(calculate_age)
# 나이 그룹화
age_bins = [10, 20, 30, 40, 50, 60]
age_labels = ['10대', '20대', '30대', '40대', '50대']
users['age_group'] = pd.cut(users['birth_date'], bins=age_bins, labels=age_labels, right=False)
users['age_group_encoded'] = LabelEncoder().fit_transform(users['age_group'])

# 고려 기준 인코딩 (One-Hot Encoding)
considerations = ['기타', '할인 및 프로모션', '거리', '주차의 편리함', '부대시설', '예약의 용이함', '교통의 편리성', '메뉴의 다양성',
                  '음식의 양', '음식의 맛', '건강에 좋은 요리', '분위기', '서비스 정도', '가격 수준', '음식점의 청결도']

# 고려 기준 인코딩 (MultiLabelBinarizer 사용)
mlb = MultiLabelBinarizer(classes=considerations)
considerations_encoded = mlb.fit_transform(users['considerations'])

# 사용자 프로필 매트릭스 생성
user_profiles = pd.concat([users[['user_id', 'gender_encoded', 'age_group_encoded']], pd.DataFrame(considerations_encoded)], axis=1).set_index('user_id')
user_profiles

#rating_path = "/content/drive/MyDrive/Colab Notebooks/voiceFinder/recsys/notebooks/rating.csv"


ratings = ratings.set_index('user_id')
print(ratings)


# NaN 값을 각 사용자의 평균 값으로 대체
ratings_mean = ratings.apply(lambda row: row.fillna(row.mean()), axis=1)
ratings_mean = ratings_mean.fillna(ratings_mean.mean())

# 사용자 프로필 매트릭스와 사용자-음식점 평점 매트릭스 병합
user_profiles_ratings = pd.concat([user_profiles, ratings_mean], axis=1)
user_profiles_ratings

# 사용자 유사도 계산 (성별, 연령대, 고려 기준 기반)
profile_similarity = cosine_similarity(ratings_mean)
profile_similarity_df = pd.DataFrame(profile_similarity, index=user_profiles_ratings.index, columns=user_profiles_ratings.index)

profile_similarity_df

def get_recommendations(user_id):
    # 유사한 사용자 찾기
    similar_users = profile_similarity_df[user_id].sort_values(ascending=False).index[1:]  # 첫 번째는 자기 자신이므로 제외
    # 유사한 사용자의 평점 가져오기
    similar_users_ratings = ratings_mean.loc[similar_users]

    # 음식점 평점 평균 계산 (유사한 사용자 기반)
    restaurant_recommendations = similar_users_ratings.mean(axis=0).sort_values(ascending=False)
    print(type(restaurant_recommendations))
    sorted_recommendations = restaurant_recommendations.sort_index(key=lambda x: x.astype(int))
    # sort_result = restaurant_recommendations.reindex(sorted_recommendations)
    # 이미 평가한 음식점 제외
    rated_restaurants = ratings.loc[user_id].dropna()
    # 평가하지 않은 음식점 추천
    unrated_restaurants = restaurant_recommendations[~restaurant_recommendations.index.isin(rated_restaurants.index)]

    print("coll")
    print(sorted_recommendations)
    return sorted_recommendations
