import boto3
import os

def download_s3_folder(bucket_name, s3_folder, local_dir=None):
    if local_dir is None:
        local_dir = s3_folder

    # S3 클라이언트 생성
    client = boto3.client('s3', aws_access_key_id = '', aws_secret_access_key = '')

    # S3 폴더 내의 객체 목록을 받아옴
    paginator = client.get_paginator('list_objects_v2')
    pages = paginator.paginate(Bucket=bucket_name, Prefix=s3_folder)

    # 각 객체에 대해
    for page in pages:
        if 'Contents' in page:
            for obj in page['Contents']:
                path, filename = os.path.split(obj['Key'])
                # 로컬 경로에 디렉토리가 존재하는지 확인하고 생성
                os.makedirs(os.path.join(local_dir, path), exist_ok=True)
                # 파일 다운로드
                client.download_file(bucket_name, obj['Key'], os.path.join(local_dir, obj['Key']))

download_s3_folder('voicefinder-bucket', 'saved_model/', '')  
