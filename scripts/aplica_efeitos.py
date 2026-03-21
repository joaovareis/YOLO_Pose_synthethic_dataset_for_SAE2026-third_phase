import cv2
import albumentations as A
import numpy as np
import os

#Copie o caminho da pasta para o dataset em seu computador para os campos abaixo e só da play

INPUT_DIR = r"D:\dataset_generator\dataset\images" 
OUTPUT_DIR = r"D:\dataset_generator\dataset\images"

def main():

    pipeline = A.Compose([

        A.GaussNoise(std_range=(0.1, 0.15), p=0.8),
        
        A.OneOf([
            A.MotionBlur(blur_limit=7, p=1),
            A.GaussianBlur(blur_limit=(3, 5), p=1),
        ], p=0.5),

        A.ImageCompression(quality_range=(30, 40), p=0.8),

    ])

    imagens_processadas = 0

    for filename in os.listdir(INPUT_DIR):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            
            filepath_in = os.path.join(INPUT_DIR, filename)
            filepath_out = os.path.join(OUTPUT_DIR, f"{filename}")

            image = cv2.imread(filepath_in)
            
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            augmented = pipeline(image=image_rgb)["image"]

            result = cv2.cvtColor(augmented, cv2.COLOR_RGB2BGR)

            cv2.imwrite(filepath_out, result)
            imagens_processadas += 1

    print(f"{imagens_processadas} imagens foram transformadas")

if __name__ == "__main__":
    main()