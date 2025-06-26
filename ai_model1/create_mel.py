# ai_model1/create_mel.py

# 이 스크립트는 audio_samples/ 폴더 안의 각 소리 카테고리 폴더 내의
# .wav 오디오 파일들을 멜 스펙트로그램(Mel Spectrogram) 이미지로 변환하여
# spectrograms/ 폴더의 동일한 카테고리 하위 폴더에 저장합니다.
# 이미지 데이터셋 구조는 TensorFlow의 image_dataset_from_directory 학습 형식에 맞춰 구성됩니다.

import os
import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np

# 입력 오디오가 들어 있는 루트 폴더
INPUT_DIR = "audio_samples"

# 출력 이미지가 저장될 루트 폴더
OUTPUT_DIR = "spectrograms"

# 오디오 파일을 멜 스펙트로그램 이미지로 변환하고 저장하는 함수
def process_file(file_path, output_path):
    # 오디오 파일 로드 (y: 오디오 데이터, sr: 샘플링 레이트)
    y, sr = librosa.load(file_path, sr=22050)

    # 멜 스펙트로그램 계산 (128개의 멜 필터 사용)
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)

    # 스펙트로그램을 데시벨 단위로 변환
    S_dB = librosa.power_to_db(S, ref=np.max)

    # 그래프 크기 설정
    plt.figure(figsize=(3, 3))

    # 멜 스펙트로그램 시각화
    librosa.display.specshow(S_dB, sr=sr, x_axis='time', y_axis='mel')

    # 축 숨기기 (학습용 이미지에서는 불필요)
    plt.axis('off')

    # 여백 최소화
    plt.tight_layout()

    # 이미지 저장
    plt.savefig(output_path, bbox_inches='tight', pad_inches=0)

    # 메모리 정리
    plt.close()

# 모든 소리 카테고리 폴더 순회
for category in os.listdir(INPUT_DIR):
    input_dir = os.path.join(INPUT_DIR, category)       # 입력 폴더 경로
    output_dir = os.path.join(OUTPUT_DIR, category)     # 출력 폴더 경로

    # 출력 폴더가 없으면 생성
    os.makedirs(output_dir, exist_ok=True)

    # 카테고리 폴더 안의 모든 .wav 파일 순회
    for filename in os.listdir(input_dir):
        if filename.lower().endswith(".wav"):
            input_path = os.path.join(input_dir, filename)  # 오디오 파일 경로
            output_path = os.path.join(
                output_dir,
                filename.replace(".wav", ".png")             # 이미지로 저장될 파일명
            )
            process_file(input_path, output_path)
            print(f" Converted: {input_path} → {output_path}")
