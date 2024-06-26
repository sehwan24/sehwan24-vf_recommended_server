#!/usr/bin/env python
# coding: utf-8

# In[1]:

get_ipython().system('pip install -U sentence-transformers')
#!pip install -r /content/drive/MyDrive/requirements.txt


# In[2]:
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
import numpy as np




client = boto3.client('s3', aws_access_key_id = '', aws_secret_access_key = '')
bucket = 'voicefinder-bucket'
obj3 = client.get_object(Bucket = bucket, Key = 'markets10c.csv')
market_data = pd.read_csv(io.BytesIO(obj3['Body'].read())) 
market_data['market_id'] = market_data['market_id'].astype(int)


# In[12]:


#로컬 서버에서 최초 실행 시 사용할 코드, 한번 모델 저장된 이후엔 모두 주석처리하고 안써도 된다.
'''CHECKPOINT_NAME = 'jhgan/ko-sroberta-multitask'
model = SentenceTransformer(CHECKPOINT_NAME)
model.save('saved_model/')'''

# 로컬 디스크에 저장된 모델 경로
#MODEL_PATH = '/content/drive/MyDrive/Colab Notebooks/voiceFinder/recsys/ko-sroberta-multitask/'
MODEL_PATH = './saved_model/'
# 모델 로드
model = SentenceTransformer(MODEL_PATH)


# In[13]:


tfidf_vectorizer = TfidfVectorizer()
tfidf_matrix = tfidf_vectorizer.fit_transform(market_data['menu'] + " " + market_data['ingredient'])


# In[14]:


def get_sentence_embedding(sentences):
    return model.encode(sentences)


# In[15]:


# 유사도 기반 필터링 함수
def filter_markets(data, cannot_eat_list, threshold=0.5):
    cannot_eat_vectors = get_sentence_embedding(cannot_eat_list)
    filtered_data = []
    zero_similarity_markets = []

    for idx, row in data.iterrows():
        ingredients = row['ingredient'].split(', ')
        ingredient_vectors = get_sentence_embedding(ingredients)

        exclude = False
        for cannot_eat_vector in cannot_eat_vectors:
            cosine_similarities = cosine_similarity([cannot_eat_vector], ingredient_vectors).flatten()
            if any(similarity >= threshold for similarity in cosine_similarities):
                exclude = True
                break

        if exclude:
            zero_similarity_markets.append(row['market_id'])
        filtered_data.append(row)

    return pd.DataFrame(filtered_data), zero_similarity_markets


# In[16]:


def harmonic_mean(similarities):
    return len(similarities) / sum(1.0 / sim for sim in similarities)


# In[17]:


def get_recommendations(user_profiles):
    # 사용자 선호 음식 및 못 먹는 음식
    fav_food_items = user_profiles['fav_food'].split(", ")
    fav_food_vectors = get_sentence_embedding(fav_food_items)
    cannot_eat_list = user_profiles['cant_eat'].split(", ")

    filtered_markets, zero_similarity_markets = filter_markets(market_data, cannot_eat_list)

    if filtered_markets.empty:
        return pd.DataFrame()  # 필터링 후 남은 음식점이 없는 경우 빈 데이터프레임 반환

    # 결과 저장을 위한 리스트
    results = []

    # 필터링된 음식점 메뉴 벡터화 및 코사인 유사도 계산
    for idx, row in filtered_markets.iterrows():
        if row['market_id'] in zero_similarity_markets:
            results.append((row['market_id'], 0))
            continue

        menu_embeddings = get_sentence_embedding([row['menu']])
        tfidf_vector = tfidf_vectorizer.transform([row['menu'] + " " + row['ingredient']])

        # 각 선호 음식과 메뉴 항목 간의 유사도를 계산하여 합산
        fav_food_similarities = []

        for fav_food_item, fav_vector in zip(fav_food_items, fav_food_vectors):
            # SBERT 유사도
            sbert_similarity = cosine_similarity([fav_vector], menu_embeddings).flatten()[0]
            # TF-IDF 유사도
            user_tfidf_vector = tfidf_vectorizer.transform([fav_food_item])
            tfidf_similarity = cosine_similarity(user_tfidf_vector, tfidf_vector).flatten()[0]
            # 유사도 합산
            item_similarity = 0.8 * sbert_similarity + 0.2 * tfidf_similarity
            fav_food_similarities.append(item_similarity)

        #조화평균 계산
        total_similarity = harmonic_mean(fav_food_similarities)
        results.append((row['market_id'], total_similarity))

    # 결과를 시리즈로 변환
    recommendation_series = pd.Series(dict(results))
    return recommendation_series

