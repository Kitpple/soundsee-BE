# ai_model1/create_mel.py

# 이 스크립트는 audio_samples/ 폴더 안의 .wav 파일들을 Mel Spectrogram 이미지로 변환해서 spectrograms/ 폴더에 저장합니다.

import os
import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np

# 폴더 경로 설정
INPUT_DIR = "audio_samples"
OUTPUT_DIR = "spectrograms"

# 출력 폴더가 없다면 생성
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Mel Spectrogram 생성 함수
def process_file(file_path, output_path):
    y, sr = librosa.load(file_path, sr=22050)
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
    S_dB = librosa.power_to_db(S, ref=np.max)

    plt.figure(figsize=(3, 3))
    librosa.display.specshow(S_dB, sr=sr, x_axis='time', y_axis='mel')
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches='tight', pad_inches=0)
    plt.close()

# 폴더 내 모든 .wav 파일 변환
for fname in os.listdir(INPUT_DIR):
    if fname.lower().endswith(".wav"):
        input_path = os.path.join(INPUT_DIR, fname)
        output_path = os.path.join(OUTPUT_DIR, fname.replace(".wav", ".png"))
        process_file(input_path, output_path)
        print(f" Converted: {fname} → {output_path}")
