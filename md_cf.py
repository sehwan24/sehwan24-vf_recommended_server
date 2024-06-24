import pandas as pd
from sklearn.preprocessing import LabelEncoder, MultiLabelBinarizer
from sklearn.metrics.pairwise import cosine_similarity
from datetime import datetime
import numpy as np
import boto3
import io


client = boto3.client('s3', aws_access_key_id = '', aws_secret_access_key = '')
bucket = 'voicefinder-bucket'
obj3 = client.get_object(Bucket = bucket, Key = 'markets10c.csv')
market_data = pd.read_csv(io.BytesIO(obj3['Body'].read())) 
market_data['market_id'] = market_data['market_id'].astype(int)

obj2 = client.get_object(Bucket = bucket, Key = 'users.csv')
users_data = pd.read_csv(io.BytesIO(obj2['Body'].read()))
users = users_data[['user_id', 'birth_date', 'gender', 'considerations']]

gender_le = LabelEncoder()
#users['gender_encoded'] = gender_le.fit_transform(users['gender'])
users.loc[:, 'gender_encoded'] = gender_le.fit_transform(users['gender'])

def calculate_age(birthdate):
    today = datetime.today()
    birthdate = datetime.strptime(birthdate, "%Y-%m-%d")
    age = today.year - birthdate.year - ((today.month, today.day) < (birthdate.month, birthdate.day))
    return age

#users['birth_date'] = users['birth_date'].apply(calculate_age)
users.loc[:, 'birth_date'] = users['birth_date'].apply(calculate_age)

# 나이 그룹화
age_bins = [10, 20, 30, 40, 50, 60]
age_labels = ['10대', '20대', '30대', '40대', '50대']
#users['age_group'] = pd.cut(users['birth_date'], bins=age_bins, labels=age_labels, right=False)
#users['age_group_encoded'] = LabelEncoder().fit_transform(users['age_group'])

users.loc[:, 'age_group'] = pd.cut(users['birth_date'], bins=age_bins, labels=age_labels, right=False)

# 연령대 인코딩
users.loc[:, 'age_group_encoded'] = LabelEncoder().fit_transform(users['age_group'])


# 고려 기준 인코딩 (One-Hot Encoding)
considerations = ['기타', '할인 및 프로모션', '거리', '주차의 편리함', '부대시설', '예약의 용이함', '교통의 편리성', '메뉴의 다양성',
                  '음식의 양', '음식의 맛', '건강에 좋은 요리', '분위기', '서비스 정도', '가격 수준', '음식점의 청결도']

# 고려 기준 인코딩 (MultiLabelBinarizer 사용)
mlb = MultiLabelBinarizer(classes=considerations)
considerations_encoded = mlb.fit_transform(users['considerations'])

# 사용자 프로필 매트릭스 생성
user_profiles = pd.concat([users[['user_id', 'gender_encoded', 'age_group_encoded']], pd.DataFrame(considerations_encoded)], axis=1).set_index('user_id')

obj4 = client.get_object(Bucket = bucket, Key = 'rating10f.csv')
ratings = pd.read_csv(io.BytesIO(obj4['Body'].read()))
ratings = ratings.set_index('user_id')

# NaN 값을 각 사용자의 평균 값으로 대체
ratings_mean = ratings.apply(lambda row: row.fillna(row.mean()), axis=1)

# 사용자 프로필 매트릭스와 사용자-음식점 평점 매트릭스 병합
user_profiles_ratings = pd.concat([user_profiles, ratings_mean], axis=1)

# 사용자 유사도 계산 (성별, 연령대, 고려 기준 기반)
profile_similarity = cosine_similarity(ratings_mean)
profile_similarity_df = pd.DataFrame(profile_similarity, index=user_profiles_ratings.index, columns=user_profiles_ratings.index)

def get_recommendations(user_id):
    # 유사한 사용자 찾기
    similar_users = profile_similarity_df[user_id].sort_values(ascending=False).index[1:]  # 첫 번째는 자기 자신이므로 제외
    # 유사한 사용자의 평점 가져오기
    similar_users_ratings = ratings_mean.loc[similar_users]

    # 음식점 평점 평균 계산 (유사한 사용자 기반)
    restaurant_recommendations = similar_users_ratings.mean(axis=0).sort_values(ascending=False)

     # 이미 평가한 음식점
    rated_restaurants = ratings.loc[user_id].dropna().index

    # 모든 음식점 목록 가져오기
    all_restaurants = ratings.columns

    # 결과 저장을 위한 시리즈 초기화
    recommendation_series = pd.Series(0, index=all_restaurants)

    # 평가하지 않은 음식점의 평점 추가
    for restaurant in restaurant_recommendations.index:
        if restaurant not in rated_restaurants:
            recommendation_series[restaurant] = restaurant_recommendations[restaurant]

    return recommendation_series
