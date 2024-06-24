from flask import Flask, jsonify, request
import sys
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, MultiLabelBinarizer
from sklearn.metrics.pairwise import cosine_similarity
import collaborate_filtering
import content_based
import hybrid_recsys
import boto3
import io

app=Flask(__name__)

@app.route('/')
def hello():
   client = boto3.client('s3', aws_access_key_id = '', aws_secret_access_key = '')
   bucket = 'voicefinder-bucket'
   obj = client.get_object(Bucket = bucket, Key = 'sample_data.csv')
   df = pd.read_csv(io.BytesIO(obj['Body'].read()))
   json_data1 = df.to_json(orient='records')
   return jsonify(json_data1)


@app.route('/recommend', methods=['POST'])
def recommend():
   data = request.get_json()
   user_id = data.get('user_id')
   fav_food = data.get('fav_foods', [])
   cannot_eat = data.get('cant_foods', [])
    
   user_profile = {
       "user_id": int(user_id),
       "cannot_eat": cannot_eat,
       "fav_food": fav_food
   }
    
   try:
       # Here you should call your hybrid recommendation model
       recommended_market_ids = hybrid_recsys.get_recommendations(user_profile)
       # recommended_market_ids = md_hybrid.get_recommendations(user_profile)
       list_ids = recommended_market_ids.tolist()
       json_data2 = {'ids': list_ids}
       print(json_data2)
       return jsonify(json_data2)
   except Exception as e:
       return jsonify({'error': str(e)}), 500


if __name__=='__main__':
   app.run(host='0.0.0.0', port=5000, debug=True)


