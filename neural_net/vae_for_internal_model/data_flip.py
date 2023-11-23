# -*- coding: utf-8 -*-
import os
from PIL import Image

def flip_images(folder_path):
    # 폴더 내의 모든 파일을 가져옵니다.
    files = os.listdir(folder_path)
    
    for file in files:
        # 좌우 반전된 이미지는 제외
        if file.endswith('-f.jpg'):
            continue

        # 원본 이미지 파일 경로
        original_file_path = os.path.join(folder_path, file)

        # 반전된 이미지 파일 이름 생성
        flipped_file_name = file.replace('.jpg', '-f.jpg')
        flipped_file_path = os.path.join(folder_path, flipped_file_name)

        # 반전된 이미지가 없으면 생성
        if not os.path.exists(flipped_file_path):
            with Image.open(original_file_path) as img:
                flipped_img = img.transpose(Image.FLIP_LEFT_RIGHT)
                flipped_img.save(flipped_file_path)
                print(f'Flipped and saved: {flipped_file_name}')

# 사용 예시
flip_images('/home2/kdh/vae/new_dataset/2023-08-22-17-26-04')
