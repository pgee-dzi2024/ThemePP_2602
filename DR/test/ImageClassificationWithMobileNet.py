import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input, decode_predictions
from tensorflow.keras.preprocessing import image
import numpy as np

# 1. Зареждаме готовия модел MobileNetV2, обучен на ImageNet
print("Зареждане на модела...")
model = MobileNetV2(weights='imagenet')

def classify_image(img_path):
    # 2. Зареждаме снимката и я преоразмеряваме до 224x224 (изискване на модела)
    img = image.load_img(img_path, target_size=(224, 224))
    
    # 3. Превръщаме снимката в масив от числа
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)

    # 4. Правим предсказание
    preds = model.predict(x)
    
    # 5. Декодираме резултатите (вземаме топ 3 предсказания)
    print('Резултати:')
    for imagenet_id, label, score in decode_predictions(preds, top=3)[0]:
        print(f"{label}: {score*100:.2f}%")

# Тук сложи пътя до някоя твоя снимка (напр. 'rose.jpg')
# classify_image('път_към_твоята_снимка.jpg')
